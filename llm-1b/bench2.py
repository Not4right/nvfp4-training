import torch, time, sys
torch.cuda.set_per_process_memory_fraction(0.9)
import torch._inductor.config as ic; ic.use_static_cuda_launcher=False
import model2 as M2
B=int(sys.argv[1]); fp4=sys.argv[2]=="fp4"; graph=len(sys.argv)>3 and sys.argv[3]=="graph"; T=2048
cfg=M2.Config(); m=M2.Model(cfg,B,T); m.P.init(0); m.fp4=fp4; m.acc_scale=1/32
import os; m.attn=os.environ.get("ATTN","fp8")
print(f"static alloc {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)
m.refresh_weights()
idx=torch.randint(0,cfg.vocab,(B,T),device='cuda'); mask=torch.ones(B,T,device='cuda')
torch.cuda.reset_peak_memory_stats()
step=lambda: m.microbatch(idx,idx,mask)
for _ in range(3): step()
torch.cuda.synchronize()
if graph:
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): step()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=step()
    step=lambda: g.replay()
    for _ in range(2): step()
torch.cuda.synchronize(); t=time.perf_counter(); n=5
for _ in range(n): step()
torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
print(f"attn={m.attn} B={B} fp4={fp4} graph={graph}: {dt*1e3:.0f} ms/microbatch  {B*T/dt:.0f} tok/s  loss {m.loss.item():.3f}  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB  reserved {torch.cuda.max_memory_reserved()/2**30:.2f}", flush=True)
