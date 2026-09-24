import torch, time
torch.cuda.set_per_process_memory_fraction(0.85)
import emit as E
from nvrtc_util import launch
dev='cuda'; torch.manual_seed(0)
def bench(f,n=30):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
T=4096; rng=torch.tensor([5,0xABCD],dtype=torch.int32,device=dev)
gu=torch.randn(T,11264,device=dev).bfloat16(); rin=torch.rand(T,device=dev)+.5; gw=torch.tensor([.7],device=dev)
q1=E.RowQ(T,5632); q2=E.RowQ(T,5632); a1=torch.empty(T,5632,device=dev,dtype=torch.bfloat16); a2=torch.empty_like(a1)
E.swiglu_emit(gu,rin,gw,q1,a1)
launch(E.fn("swiglu_emit"),((T+7)//8,),(256,),[gu,rin,gw,5632,a2,q2.d,q2.s,q2.rs,T,1])
print("swiglu new==old:", torch.equal(q1.d,q2.d), torch.equal(q1.s,q2.s), torch.equal(q1.rs,q2.rs), torch.equal(a1,a2))
print(f"swiglu new {bench(lambda: E.swiglu_emit(gu,rin,gw,q1)):.3f} ms  old {bench(lambda: launch(E.fn('swiglu_emit'),((T+7)//8,),(256,),[gu,rin,gw,5632,0,q2.d,q2.s,q2.rs,T,1])):.3f} ms")
for C in (2048,3072,11264):
    x=torch.randn(T,C,device=dev).bfloat16(); qa=E.RowQ(T,C); qb=E.RowQ(T,C)
    E.row_emit(x,qa,rng,sr=False)
    launch(E.fn("row_emit"),((T+7)//8,),(256,),[x,x.stride(0),C,qb.d,qb.s,qb.rs,T,0,rng,0])
    ok=torch.equal(qa.d,qb.d) and torch.equal(qa.s,qb.s) and torch.equal(qa.rs,qb.rs)
    E.row_emit(x,qa,rng,sr=True); err=((E.dequant(qa.d,qa.s,qa.rs)-x.float()).norm()/x.float().norm()).item()
    print(f"row {C}: RN identical {ok}  SR err {err:.4f}  new {bench(lambda: E.row_emit(x,qa,rng,sr=True)):.3f} ms  old {bench(lambda: launch(E.fn('row_emit'),((T+7)//8,),(256,),[x,x.stride(0),C,qb.d,qb.s,qb.rs,T,1,rng,0])):.3f} ms")
