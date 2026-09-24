import torch, sys
torch.cuda.set_per_process_memory_fraction(0.9)
import torch._inductor.config as ic; ic.use_static_cuda_launcher=False
import model2 as M2
from torch.profiler import profile, ProfilerActivity
B=int(sys.argv[1]); T=2048
cfg=M2.Config(); m=M2.Model(cfg,B,T); m.P.init(0); m.fp4=sys.argv[2]=="fp4"; m.acc_scale=1/32; m.refresh_weights()
idx=torch.randint(0,cfg.vocab,(B,T),device='cuda'); mask=torch.ones(B,T,device='cuda')
for _ in range(3): m.microbatch(idx,idx,mask)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    m.microbatch(idx,idx,mask); torch.cuda.synchronize()
ev={}
for e in pr.events():
    if e.device_type.name=="CUDA" and not e.name.startswith("## Call"):
        k=e.name[:80]; ev.setdefault(k,[0,0]); ev[k][0]+=e.device_time_total; ev[k][1]+=1
tot=sum(v[0] for v in ev.values())
print(f"GPU kernel total {tot/1e3:.1f} ms for {B*T} tokens")
for k,v in sorted(ev.items(),key=lambda kv:-kv[1][0])[:45]: print(f"{v[0]/1e3:8.2f} ms {v[1]:5d}x  {k}")
