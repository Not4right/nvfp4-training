import torch, time, sys, os
torch.cuda.set_per_process_memory_fraction(0.85)   # never spill into shared memory (froze Windows once)
import torch._inductor.config as ic; ic.use_static_cuda_launcher=False
from model import GPT, Config, LinCtx, n_params
B=int(sys.argv[1]); mode=sys.argv[2]=="fp4"; T=2048
cfg=Config(); m=GPT(cfg).bfloat16().cuda()
for p in m.parameters(): p.register_post_accumulate_grad_hook(lambda p: setattr(p,'grad',None))  # stands in for CPU offload
idx=torch.randint(0,cfg.vocab,(B,T),device='cuda')
LinCtx.fp4=mode; torch.cuda.reset_peak_memory_stats()
for i in range(5):
    if i==2: torch.cuda.synchronize(); t=time.perf_counter()
    LinCtx.seed=i
    loss=m(idx,idx); loss.backward()
torch.cuda.synchronize(); dt=(time.perf_counter()-t)/3
print(f"B={B} fp4={mode} loss {loss.item():.3f} step {dt*1e3:.0f} ms  {B*T/dt:.0f} tok/s  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
