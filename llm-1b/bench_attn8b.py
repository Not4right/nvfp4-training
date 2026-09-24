import torch, statistics
import attn8 as A, model2 as M2
dev='cuda'; B,T,H,KV,hd=1,2048,16,4,128
qkv=(torch.randn(B*T,(H+2*KV)*hd,device=dev)*2).bfloat16()
qn=torch.ones(hd,device=dev).bfloat16(); kn=qn.clone(); cos,sin=M2.rope_cache(M2.Config(),dev)
Q8,K8,VT8,sv=A.prep(qkv,qn,kn,cos,sin,B,T,H,KV); O,L=A.fwd(Q8,K8,VT8,sv,B,T,H,KV)
dO=(torch.randn(B,T,H,hd,device=dev)*0.01).bfloat16()
au=torch.tensor([0.5],device=dev); ac=torch.zeros(1,dtype=torch.int32,device=dev)
q=torch.randn(B,T,H,hd,device=dev,dtype=torch.bfloat16).transpose(1,2); k=torch.randn_like(q); v=torch.randn_like(q)
r=torch.ops.aten._scaled_dot_product_cudnn_attention(q,k,v,None,True,0.0,True,False)
do4=dO.transpose(1,2)
def cud(): return torch.ops.aten._scaled_dot_product_cudnn_attention_backward(do4,q,k,v,r[0],r[1],r[6],r[7],None,r[2],r[3],r[4],r[5],0.0,True)
fns={"fp8 bwd (all)":lambda: A.bwd(dO,O,L,Q8,K8,qkv,au,ac,B,T,H,KV),"cudnn bwd":cud,
     "fp8 fwd+prep":lambda: (A.prep(qkv,qn,kn,cos,sin,B,T,H,KV),A.fwd(Q8,K8,VT8,sv,B,T,H,KV))}
def tm(f,n=10):
    e0,e1=torch.cuda.Event(True),torch.cuda.Event(True); e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)/n
for f in fns.values(): tm(f,3)
res={nm:[] for nm in fns}
for _ in range(7):
    for nm,f in fns.items(): res[nm].append(tm(f))
for nm,v_ in res.items(): print(f"{nm:14s} {statistics.median(v_):.3f} ms")
