import torch, statistics, cupy
import attn8 as A, model2 as M2
from nvrtc_util import launch
dev='cuda'; B,T,H,KV,hd=1,2048,16,4,128
qkv=(torch.randn(B*T,(H+2*KV)*hd,device=dev)*2).bfloat16()
qn=torch.ones(hd,device=dev).bfloat16(); kn=qn.clone(); cos,sin=M2.rope_cache(M2.Config(),dev)
Q8,K8,VT8,sv=A.prep(qkv,qn,kn,cos,sin,B,T,H,KV); O,L=A.fwd(Q8,K8,VT8,sv,B,T,H,KV)
dO=(torch.randn(B,T,H,hd,device=dev)*0.01).bfloat16()
au=torch.tensor([0.5],device=dev); ac=torch.zeros(1,dtype=torch.int32,device=dev)
for nm,sm in (("attn_bwd8",A.BWD_SMEM),("attn_fwd8",A.FWD_SMEM)):
    f=A.fn(nm,sm); print(nm,"regs",cupy.cuda.driver.funcGetAttribute(4,f.ptr),"local bytes",cupy.cuda.driver.funcGetAttribute(3,f.ptr),"smem",sm)
from torch.profiler import profile, ProfilerActivity
for _ in range(3): A.bwd(dO,O,L,Q8,K8,qkv,au,ac,B,T,H,KV)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(5): A.bwd(dO,O,L,Q8,K8,qkv,au,ac,B,T,H,KV)
    torch.cuda.synchronize()
ev={}
for e in pr.events():
    if e.device_type.name=="CUDA": ev.setdefault(e.name[:40],[0,0]); ev[e.name[:40]][0]+=e.device_time_total/5; ev[e.name[:40]][1]+=1
for k,v in sorted(ev.items(),key=lambda kv:-kv[1][0]): print(f"{v[0]/1e3:7.3f} ms  {k}")
