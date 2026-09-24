import torch, time, math
import attn8 as A
import model2 as M2
torch.manual_seed(0); dev='cuda'
B,T,H,KV,hd=1,1024,16,4,128
qkv=(torch.randn(B*T,(H+2*KV)*hd,device=dev)*2).bfloat16()
qn=(1+0.2*torch.randn(hd,device=dev)).bfloat16(); kn=(1+0.2*torch.randn(hd,device=dev)).bfloat16()
cos,sin=M2.rope_cache(M2.Config(),dev)
Q8,K8,VT8,sv=A.prep(qkv,qn,kn,cos,sin,B,T,H,KV)
O,L=A.fwd(Q8,K8,VT8,sv,B,T,H,KV); torch.cuda.synchronize()
# fp32 reference
x=qkv.float().view(B,T,H+2*KV,hd); q,k,v=x.split([H,KV,KV],2)
def nr(t,w):
    t=t*torch.rsqrt(t.pow(2).mean(-1,keepdim=True)+1e-5)*w.float()
    c,s=cos[:T,None,:].float(),sin[:T,None,:].float(); a,b_=t[...,:64],t[...,64:]
    return torch.cat([a*c-b_*s,b_*c+a*s],-1)
q=nr(q,qn); k=nr(k,kn)
qh=q.transpose(1,2); kh=k.repeat_interleave(H//KV,2).transpose(1,2); vh=v.repeat_interleave(H//KV,2).transpose(1,2)
s=(qh@kh.transpose(-1,-2))/math.sqrt(hd); s=s.masked_fill(torch.triu(torch.ones(T,T,device=dev,dtype=torch.bool),1),-1e30)
lse=torch.logsumexp(s,-1); o=(s.softmax(-1)@vh).transpose(1,2)
rel=((O.float()-o).norm()/o.norm()).item()
print(f"O rel err {rel:.4f}   LSE max abs err {(L/1.4426950408889634-lse).abs().max().item():.4f}")
# bf16 cuDNN-level error for comparison
ob=torch.nn.functional.scaled_dot_product_attention(qh.bfloat16(),kh.bfloat16(),vh.bfloat16(),is_causal=True).transpose(1,2)
print(f"bf16 SDPA rel err {((ob.float()-o).norm()/o.norm()).item():.4f}")
# ---- reference from the kernel's own fp8 operands: isolates the in-kernel error (P quantization only)
f8=torch.float8_e4m3fn
qd=Q8.view(f8).float()                       # [B,H,T,hd]
kd=K8.view(f8).float().repeat_interleave(H//KV,1)
perm=torch.tensor([ (p&16)+((2*((p&15)>>2)+(p&3)) if (p&3)<2 else (8+2*((p&15)>>2)+(p&3)-2)) for p in range(32)],device=dev)
vt=VT8.view(f8).float()*sv[...,None]*1.0   # [B,KV,hd,T] permuted, scaled back
Tb=T//32; vt=vt.view(B,KV,hd,Tb,32); vu=torch.empty_like(vt); vu[...,perm]=vt; vd=vu.view(B,KV,hd,T).transpose(-1,-2).repeat_interleave(H//KV,1)
s2=(qd@kd.transpose(-1,-2))/math.sqrt(hd); s2=s2.masked_fill(torch.triu(torch.ones(T,T,device=dev,dtype=torch.bool),1),-1e30)
o2=(s2.softmax(-1)@vd).transpose(1,2)
print(f"kernel vs ref-on-fp8-inputs rel err {((O.float()-o2).norm()/o2.norm()).item():.4f} ; fp8 inputs vs exact {((o2-o).norm()/o.norm()).item():.4f}")
cs=torch.nn.functional.cosine_similarity(O.float().flatten(0,2),o.flatten(0,2),dim=-1)
print(f"per-row cosine vs exact: mean {cs.mean().item():.5f}  min {cs.min().item():.5f}  (first 32 tokens min {cs.view(B,T,H)[:, :32].min().item():.4f})")
