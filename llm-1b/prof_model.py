import torch, sys, os
torch.cuda.set_per_process_memory_fraction(0.85)
import torch._inductor.config as ic; ic.use_static_cuda_launcher=False
from model import GPT, Config, LinCtx
from torch.profiler import profile, ProfilerActivity
B=int(sys.argv[1]) if len(sys.argv)>1 else 4; T=2048
m=GPT(Config()).bfloat16().cuda(); idx=torch.randint(0,49152,(B,T),device='cuda')
for p in m.parameters(): p.register_post_accumulate_grad_hook(lambda p: setattr(p,'grad',None))
def step():
    loss=m(idx,idx); loss.backward()
for _ in range(2): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as pr:
    step(); torch.cuda.synchronize()
ev={}
for e in pr.events():
    if e.device_type.name=="CUDA":
        k=e.name[:70]; ev.setdefault(k,[0,0]); ev[k][0]+=e.device_time_total; ev[k][1]+=1
tot=sum(v[0] for v in ev.values())
print(f"GPU total {tot/1e3:.0f} ms for {B*T} tokens; CPU wall {pr.key_averages().self_cpu_time_total/1e3:.0f} ms")
for k,v in sorted(ev.items(),key=lambda kv:-kv[1][0])[:40]: print(f"{v[0]/1e3:8.1f} ms {v[1]:5d}x  {k}")
