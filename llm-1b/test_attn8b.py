import torch, math
import attn8 as A, model2 as M2
torch.manual_seed(0); dev='cuda'
B,T,H,KV,hd=1,1024,16,4,128
qkv=(torch.randn(B*T,(H+2*KV)*hd,device=dev)*2).bfloat16()
qn=(1+0.2*torch.randn(hd,device=dev)).bfloat16(); kn=(1+0.2*torch.randn(hd,device=dev)).bfloat16()
cos,sin=M2.rope_cache(M2.Config(),dev)
Q8,K8,VT8,sv=A.prep(qkv,qn,kn,cos,sin,B,T,H,KV)
O,L=A.fwd(Q8,K8,VT8,sv,B,T,H,KV)
dO=(torch.randn(B,T,H,hd,device=dev)*0.01).bfloat16()
# fp32 autograd reference on the normed/roped q,k and v
x=qkv.float().view(B,T,H+2*KV,hd); q,k,v=x.split([H,KV,KV],2)
def nr(t,w):
    t=t*torch.rsqrt(t.pow(2).mean(-1,keepdim=True)+1e-5)*w.float()
    c,s=cos[:T,None,:].float(),sin[:T,None,:].float(); a,b_=t[...,:64],t[...,64:]
    return torch.cat([a*c-b_*s,b_*c+a*s],-1)
qr=nr(q,qn).transpose(1,2).contiguous().requires_grad_(); kr=nr(k,kn).transpose(1,2).contiguous().requires_grad_(); vr=v.transpose(1,2).contiguous().requires_grad_()
s=(qr@kr.repeat_interleave(H//KV,1).transpose(-1,-2))/math.sqrt(hd); s=s.masked_fill(torch.triu(torch.ones(T,T,device=dev,dtype=torch.bool),1),-1e30)
o=(s.softmax(-1)@vr.repeat_interleave(H//KV,1))
o.backward(dO.float().transpose(1,2))
def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm()).item()
def cosm(a,b): a,b=a.float().flatten(),b.float().flatten(); return (a@b/(a.norm()*b.norm())).item()
au=torch.tensor([1.0],device=dev); ac=torch.zeros(1,dtype=torch.int32,device=dev)
for it in range(2):   # 1st pass calibrates the delayed dS scale
    dq,dk,dv=A.bwd(dO,O,L,Q8,K8,qkv,au,ac,B,T,H,KV)
    au.copy_(ac.view(torch.float32)*2); ac.zero_()
torch.cuda.synchronize()
print("dS amax", au.item()/2)
print(f"dq rel {rel(dq,qr.grad):.4f} cos {cosm(dq,qr.grad):.5f}")
print(f"dk rel {rel(dk,kr.grad):.4f} cos {cosm(dk,kr.grad):.5f}")
print(f"dv rel {rel(dv,vr.grad):.4f} cos {cosm(dv,vr.grad):.5f}")
# bf16 cuDNN backward error for scale
from torch.nn.attention import sdpa_kernel, SDPBackend
qb,kb_,vb=[t.detach().bfloat16().requires_grad_() for t in (qr,kr,vr)]
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    ob=torch.nn.functional.scaled_dot_product_attention(qb,kb_.repeat_interleave(H//KV,1),vb.repeat_interleave(H//KV,1),is_causal=True)
ob.backward(dO.transpose(1,2))
print(f"[bf16 cuDNN] dq rel {rel(qb.grad,qr.grad):.4f} dk rel {rel(kb_.grad,kr.grad):.4f} dv rel {rel(vb.grad,vr.grad):.4f}")
