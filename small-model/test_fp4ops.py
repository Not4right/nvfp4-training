import torch, time
import fp4ops as F
from model_ref import nvfp4_1d, nvfp4_2d

torch.manual_seed(0)
x = torch.randn(4096, 512, device="cuda").bfloat16()
q = F.quant_rows(x)
dq = F.dequant(q)
ref = nvfp4_1d(x.float())
print("quant_rows vs sim  max|diff|", (dq - ref).abs().max().item(), " rel err vs x", ((dq - x.float()).norm() / x.float().norm()).item())

qs = F.quant_rows(x, sr=True, seed=7)
acc = torch.zeros_like(dq)
for s in range(64):
    acc += F.dequant(F.quant_rows(x, sr=True, seed=s))
print("SR mean-of-64 rel err (should be << RN):", ((acc / 64 - x.float()).norm() / x.float().norm()).item())

w = torch.randn(384, 128, device="cuda")
wq, wqt = F.quant_w2d(w)
print("quant_w2d vs sim", (F.dequant(wq) - nvfp4_2d(w)).abs().max().item(), " transpose consistent",
      (F.dequant(wqt) - F.dequant(wq).t()).abs().max().item())

a = torch.randn(2048, 256, device="cuda").bfloat16(); b = torch.randn(384, 256, device="cuda").bfloat16()
qa, qb = F.quant_rows(a), F.quant_rows(b)
c = F.gemm(qa, qb).float()
cref = F.dequant(qa) @ F.dequant(qb).t()
print("gemm vs dequant-matmul rel", ((c - cref).norm() / cref.norm()).item())

# wgrad path: dW = g^T x over M, both quantized along M with RHT
M = 16384
g = torch.randn(M, 128, device="cuda").bfloat16(); xx = torch.randn(M, 512, device="cuda").bfloat16()
qg = F.quant_cols_rht(g, 0xA5C3, sr=True, seed=3); qx = F.quant_cols_rht(xx, 0xA5C3)
dw = F.gemm(qg, qx, splitk=16)
dwref = g.float().t() @ xx.float()
print("wgrad (RHT+SR) rel err vs exact", ((dw - dwref).norm() / dwref.norm()).item())
dwq = F.dequant(qg) @ F.dequant(qx).t()
print("wgrad gemm vs dequant-matmul rel", ((dw - dwq).norm() / dwq.norm()).item())

# speed
A = F.quant_rows(torch.randn(16384, 512, device="cuda").bfloat16()); B = F.quant_rows(torch.randn(512, 512, device="cuda").bfloat16())
for _ in range(3): F.gemm(A, B)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): F.gemm(A, B)
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 50
print(f"gemm 16384x512x512: {dt*1e6:.1f} us, {2*16384*512*512/dt/1e12:.1f} TFLOPS")
Ab = torch.randn(16384, 512, device="cuda").bfloat16(); Bb = torch.randn(512, 512, device="cuda").bfloat16()
for _ in range(3): Ab @ Bb.t()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): Ab @ Bb.t()
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 50
print(f"cuBLAS bf16 same: {dt*1e6:.1f} us, {2*16384*512*512/dt/1e12:.1f} TFLOPS")
