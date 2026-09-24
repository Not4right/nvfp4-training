import torch, time, statistics
import attn8 as A, model2 as M2
from torch.nn.attention import sdpa_kernel, SDPBackend
dev='cuda'; B,T,H,KV,hd=1,2048,16,4,128
qkv=(torch.randn(B*T,(H+2*KV)*hd,device=dev)*2).bfloat16()
qn=torch.ones(hd,device=dev).bfloat16(); kn=qn.clone(); cos,sin=M2.rope_cache(M2.Config(),dev)
Q8,K8,VT8,sv=A.prep(qkv,qn,kn,cos,sin,B,T,H,KV)
q=torch.randn(B,T,H,hd,device=dev,dtype=torch.bfloat16).transpose(1,2); k=torch.randn_like(q); v=torch.randn_like(q)
def cud():
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION): return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
fns={"fp8 fwd":lambda: A.fwd(Q8,K8,VT8,sv,B,T,H,KV),"prep":lambda: A.prep(qkv,qn,kn,cos,sin,B,T,H,KV),"cudnn fwd":cud}
def tm(f,n=20):
    e0,e1=torch.cuda.Event(True),torch.cuda.Event(True); e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)/n
for f in fns.values(): tm(f,5)
res={k:[] for k in fns}
for _ in range(7):
    for nm,f in fns.items(): res[nm].append(tm(f))
fl=4*B*H*T*T*hd/2
for k,v_ in res.items(): m=statistics.median(v_); print(f"{k:10s} {m:.3f} ms" + (f"  {fl/m/1e9:.1f} TF" if "fwd" in k else ""))
