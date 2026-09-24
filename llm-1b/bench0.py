import torch, time
import fp4ops as F
torch.manual_seed(0)
def tm(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n
for (M,N,K) in [(16384,3072,2048),(16384,11264,2048),(16384,2048,5632),(2048,5632,16384)]:
    a=torch.randn(M,K,device='cuda',dtype=torch.bfloat16); b=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)
    t=tm(lambda: a@b.T); print(f"bf16 {M}x{N}x{K}: {2*M*N*K/t/1e12:.1f} TF")
    qa=F.quant_rows(a); qb=F.quant_rows(b)
    t=tm(lambda: F.gemm(qa,qb)); print(f"fp4k {M}x{N}x{K}: {2*M*N*K/t/1e12:.1f} TF")
    t=tm(lambda: F.quant_rows(a)); print(f"  quant_rows A {t*1e3:.2f} ms ({a.numel()*2.56/t/1e9:.0f} GB/s)")
# torch scaled_mm fp4 probe
try:
    M,N,K=8192,8192,8192
    A=torch.randint(0,255,(M,K//2),device='cuda',dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    B=torch.randint(0,255,(N,K//2),device='cuda',dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    sa=torch.ones(M*K//16,device='cuda').to(torch.float8_e4m3fn); sb=torch.ones(N*K//16,device='cuda').to(torch.float8_e4m3fn)
    f=lambda: torch._scaled_mm(A,B.T,sa,sb,out_dtype=torch.bfloat16)
    t=tm(f); print(f"torch._scaled_mm fp4 {2*M*N*K/t/1e12:.1f} TF")
except Exception as e: print("scaled_mm fp4 failed:", repr(e)[:300])
# PCIe
h=torch.empty(512*2**20,dtype=torch.uint8).pin_memory(); d=torch.empty_like(h,device='cuda')
t=tm(lambda: d.copy_(h,non_blocking=True),10); print(f"H2D {h.numel()/t/1e9:.1f} GB/s")
t=tm(lambda: h.copy_(d,non_blocking=True),10); print(f"D2H {h.numel()/t/1e9:.1f} GB/s")
