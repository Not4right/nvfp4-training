import torch, time
torch.cuda.set_per_process_memory_fraction(0.85)
import emit as E
torch.manual_seed(0); dev='cuda'
def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm()).item()
def bench(f,n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
T=8192; D=2048; F=5632
rng=torch.tensor([12345, 0b1011001110001101],dtype=torch.int32,device=dev)
x=torch.randn(T,D,device=dev).bfloat16(); y=torch.randn(T,D,device=dev).bfloat16()
w=(1+0.1*torch.randn(D,device=dev)).bfloat16()
ys=torch.rand(T,device=dev)*0.01; ygw=torch.tensor([0.5],device=dev)
q=E.RowQ(T,D); h=torch.empty_like(x); xo=torch.empty_like(x); rstd=torch.empty(T,device=dev)
E.rms_emit(x,w,q,h=h,rstd=rstd,y=y,ys=ys,ygw=ygw,xout=xo)
xr=(x.float()+y.float()*ys[:,None]*0.5).bfloat16().float()
hr=xr*torch.rsqrt(xr.pow(2).mean(-1,keepdim=True)+1e-5)*w.float()
print("rms_emit: xout", rel(xo,xr), " h", rel(h,hr), " quant", rel(E.dequant(q.d,q.s,q.rs),hr),
      f" {bench(lambda: E.rms_emit(x,w,q,h=None,rstd=rstd,y=y,ys=ys,ygw=ygw,xout=xo)):.3f} ms")
# swiglu
gu=torch.randn(T,2*F,device=dev).bfloat16(); rin=torch.rand(T,device=dev)+0.5; gw=torch.tensor([0.7],device=dev)
qa=E.RowQ(T,F); a=torch.empty(T,F,device=dev,dtype=torch.bfloat16)
E.swiglu_emit(gu,rin,gw,qa,a)
s=(rin*0.7)[:,None]; g_,u_=gu.float().chunk(2,-1); ar=torch.nn.functional.silu(g_*s)*(u_*s)
print("swiglu_emit: a", rel(a,ar), " quant", rel(E.dequant(qa.d,qa.s,qa.rs),ar), f" {bench(lambda: E.swiglu_emit(gu,rin,gw,qa,None)):.3f} ms")
# row_emit SR unbiasedness
g=torch.randn(T,D,device=dev).bfloat16(); qg=E.RowQ(T,D)
acc=torch.zeros(T,D,device=dev)
for i in range(32):
    rng[0]=i*7919+1; E.row_emit(g,qg,rng,sr=True); acc+=E.dequant(qg.d,qg.s,qg.rs)
E.row_emit(g,qg,rng,sr=False)
print("row_emit: RN err", rel(E.dequant(qg.d,qg.s,qg.rs),g), " SR mean(32) err", rel(acc/32,g), f" {bench(lambda: E.row_emit(g,qg,rng,sr=True)):.3f} ms")
# col_emit + wgrad GEMM, delayed amax calibrated
gg=(torch.randn(T,F,device=dev)*1e-3).bfloat16(); aa=torch.randn(T,D,device=dev).bfloat16()
cu=torch.ones(2,device=dev)*1e9; cc=torch.zeros(2,dtype=torch.int32,device=dev)
qc1=E.ColQ(T,F); qc2=E.ColQ(T,D)
E.col_emit(gg,qc1,cu[0:1],cc[0:1],rng,sr=True); E.col_emit(aa,qc2,cu[1:2],cc[1:2],rng)
cu.copy_(cc.view(torch.float32)*2.0); print("observed RHT amax", cc.view(torch.float32).tolist())
E.col_emit(gg,qc1,cu[0:1],cc[0:1],rng,sr=True,salt=3); E.col_emit(aa,qc2,cu[1:2],cc[1:2],rng)
raw=E.mm_col(qc1,qc2); dW=raw.float()*(cu[0]/2688)*(cu[1]/2688)
ref=gg.float().t()@aa.float()
print("col_emit wgrad rel err", rel(dW,ref), f"  col_emit {bench(lambda: E.col_emit(gg,qc1,cu[0:1],cc[0:1],rng,sr=True)):.3f} ms for {T}x{F}")
# weights
W=(torch.randn(F,D,device=dev)*0.02).bfloat16(); wq=E.WQ(F,D); wq.refresh(W)
hq=E.RowQ(T,D); E.rms_emit(x,w,hq,h=h)
yraw=E.mm_row_w(hq,wq); yv=yraw.float()*hq.rs[:,None]*wq.gs
print("fwd GEMM vs exact", rel(yv, h.float()@W.float().t()))
dq=E.RowQ(T,F); E.row_emit(gg,dq,rng,sr=True)
dx=E.mm_row_wt(dq,wq).float()*dq.rs[:,None]*wq.gs
print("dgrad GEMM vs exact", rel(dx, gg.float()@W.float()))
print(f"WQ.refresh {bench(lambda: wq.refresh(W)):.3f} ms")
# momentum acc
m=torch.zeros(F,D,device=dev,dtype=torch.bfloat16); al=torch.tensor([1.0],device=dev)
ref=torch.zeros(F,D,device=dev)
for i in range(64):
    gi=torch.randn(F,D,device=dev).bfloat16(); rng[0]=i+100; E.mom_acc(m,gi,al,1/64,rng); ref+=gi.float()/64
print("mom_acc (64 SR adds) rel err", rel(m,ref), f"  {bench(lambda: E.mom_acc(m,gi,al,1/64,rng)):.3f} ms")
