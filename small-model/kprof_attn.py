import torch, cupy, fusedops as F
from nvrtc_util import Module, launch
torch.manual_seed(0)
dev, M = "cuda", 16384
wq = torch.randn(384, 128, device=dev) / 128 ** .5; wo = torch.randn(128, 128, device=dev) / 128 ** .5
Wq, Wo = F.PackedW(wq, ffN=True, rowperm=F.qkv_perm()), F.PackedW(wo)
x = torch.randn(M, 128, device=dev).bfloat16(); lnw = torch.ones(128, device=dev)
dx1 = (torch.randn(M, 128, device=dev) * 1e-3).bfloat16(); dl = torch.zeros(128, device=dev)
rng = torch.tensor([1, 2], dtype=torch.int32, device=dev)
ab = {k: F.WgradBuf(n, M) for k, n in (("dx", 128), ("o", 128), ("dq", 384), ("h", 128))}
au5, ac5 = torch.full((5,), 0.01, device=dev), torch.zeros(5, dtype=torch.int32, device=dev)
src = "".join(open(n).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu", "mlp.cu", "attn.cu", "attn_bwd.cu"))
mod = Module(src, opts=("-DKPROF",)); f = mod.fn("attn_bwd", F.ATTB_SMEM)
args = [x, dx1, torch.empty_like(x), lnw, dl, Wq.f.d, Wq.f.s, Wq.t.d, Wq.t.s, Wq.amax, Wo.t.d, Wo.t.s, Wo.amax,
        ab["dx"].d, ab["dx"].s, ab["o"].d, ab["o"].s, ab["dq"].d, ab["dq"].s, ab["h"].d, ab["h"].s, au5, ac5, M, rng]
grid = min(2 * F.NSM, M // 64)
for _ in range(3): launch(f, (grid,), (256,), args, smem=F.ATTB_SMEM)
ptr = mod.mod.get_global_var("g_prof")
buf = cupy.ndarray((32,), dtype=cupy.uint64, memptr=cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(int(ptr), 256, None), 0))
buf.fill(0); torch.cuda.synchronize()
N = 20
for _ in range(N): launch(f, (grid,), (256,), args, smem=F.ATTB_SMEM)
torch.cuda.synchronize()
v = buf.get().astype(float)[16:30] / N / grid
names = ["phase0 LN/DX1->smem", "QKV head GEMM", "softmax attn fwd", "emit O", "DX1 quant + dO GEMM", "dP, dS", "dV dK dQ",
         "emit dq dk dv", "dqkv quant->smem", "sync A->B", "dh1 GEMM (6 ks)", "LN bwd + store", "emit h1, dx1", "sync end"]
tot = v.sum()
for i, n in enumerate(names):
    print(f"{n:22s} {v[i]/1e3:7.1f} kcyc/CTA  {100*v[i]/tot:5.1f}%")
