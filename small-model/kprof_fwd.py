import torch, cupy, fusedops as F
from nvrtc_util import Module, launch
torch.manual_seed(0)
M = 16384
w1 = torch.randn(512, 128, device="cuda") / 128 ** .5; w2 = torch.randn(128, 512, device="cuda") / 512 ** .5
W1 = F.PackedW(w1, ffN=True); W2 = F.PackedW(w2, ffK=True)
x1 = torch.randn(M, 128, device="cuda").bfloat16(); lnw = torch.ones(128, device="cuda"); y = torch.empty_like(x1)
src = "".join(open(n).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu", "mlp.cu"))
mod = Module(src, opts=("-DKPROF",)); f = mod.fn("mlp_fwd", F.MLPF_SMEM)
args = [x1, y, lnw, W1.f.d, W1.f.s, W1.amax, W2.f.d, W2.f.s, W2.amax, W1.rn, M]
for _ in range(3): launch(f, (F.NSM,), (256,), args, smem=F.MLPF_SMEM)
ptr = mod.mod.get_global_var("g_prof2")
buf = cupy.ndarray((16,), dtype=cupy.uint64, memptr=cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(int(ptr), 128, None), 0))
buf.fill(0); torch.cuda.synchronize()
N = 20
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
for _ in range(N): launch(f, (F.NSM,), (256,), args, smem=F.MLPF_SMEM)
e1.record(); torch.cuda.synchronize()
v = buf.get().astype(float)[:7] / N / F.NSM
tot = v.sum()
print(f"kernel {e0.elapsed_time(e1)/N*1e3:.1f} us ; per-CTA cycles {tot/1e3:.1f}k -> clock ~ {tot/(e0.elapsed_time(e1)/N*1e3)/1e3:.2f} GHz")
for i, n in enumerate(["wait prev/prefetch", "LN+quant", "u MMAs", "gelu", "quant a", "y MMAs", "epilogue"]):
    print(f"{n:20s} {v[i]/1e3:7.1f} kcyc  {100*v[i]/tot:5.1f}%")
