import torch, time
import nvq
torch.manual_seed(0)
dev='cuda'
def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm()).item()
M,N,K=512,384,1024
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)*3
w=(torch.randn(N,K,device=dev)*0.02).bfloat16()
qx=nvq.quant_rows(x); qw,qwt=nvq.quant_w2d(w)
dx=nvq.dequant(qx); dw=nvq.dequant(qw); dwt=nvq.dequant(qwt)
print("quant err x", rel(dx,x), "w", rel(dw,w), "wT==w.T", (dwt-dw.T).abs().max().item())
y=nvq.mm(qx,qw); print("gemm vs dequant ref", rel(y, dx@dw.T), " vs exact", rel(y, x.float()@w.float().T))
# transposed / RHT
signs=0b1011001110001101
g=torch.randn(M,N,device=dev,dtype=torch.bfloat16)
qg=nvq.quant_cols_rht(g,signs,sr=True,seed=5); qxt=nvq.quant_cols_rht(x,signs)
dW=nvq.mm(qg,qxt); ref=g.float().T@x.float()
print("wgrad rel err", rel(dW,ref))
# SR unbiasedness: average many SR quantizations
acc=torch.zeros(M,N,device=dev)
for s in range(64): acc+=nvq.dequant(nvq.quant_rows(g,sr=True,seed=s))
print("SR mean err", rel(acc/64,g), " single RN err", rel(nvq.dequant(nvq.quant_rows(g)),g))
