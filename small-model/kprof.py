"""Phase timing of mlp_bwd via clock64 instrumentation (separate -DKPROF build)."""
import os, torch, cupy
import fusedops as F
from nvrtc_util import Module, launch
src = "".join(open(n).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu", "mlp.cu"))
mod = Module(src, opts=("-DKPROF",))
import test_mlp_bwd_setup as S  # noqa
f = mod.fn("mlp_bwd", F.MLPB_SMEM)
args = [S.x1, S.dy, torch.empty_like(S.x1), S.lnw, S.dlnw, S.W1.f.d, S.W1.f.s, S.W2.t.d, S.W2.t.s, S.W1.t.d, S.W1.t.s, S.W1.amax, S.W2.amax, S.W2.cn,
        S.bufs["dy"].d, S.bufs["dy"].s, S.bufs["a"].d, S.bufs["a"].s, S.bufs["du"].d, S.bufs["du"].s, S.bufs["h"].d, S.bufs["h"].s,
        S.amax_use, S.amax_cur, S.M, S.rng3, 0]
for _ in range(3): launch(f, (F.NSM,), (256,), args, smem=F.MLPB_SMEM)
ptr = mod.mod.get_global_var("g_prof")
buf = cupy.ndarray((32,), dtype=cupy.uint64, memptr=cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(int(ptr), 256, None), 0))
buf.fill(0); torch.cuda.synchronize()
N = 20
for _ in range(N): launch(f, (F.NSM,), (256,), args, smem=F.MLPB_SMEM)
torch.cuda.synchronize()
v = buf.get().astype(float) / N / F.NSM  # cycles per CTA (thread 0 = warp 0's view)
names = ["LN+quant X", "load+quant DY", "u/da MMAs", "gelu'", "emit a,du", "quant du", "dh MMAs", "LN bwd+store", "emit h,dy"]
tot = v[:9].sum()
for i, n in enumerate(names):
    print(f"{n:16s} {v[i]/1e3:8.1f} kcyc/CTA  {100*v[i]/tot:5.1f}%")
print(f"total {tot/1e3:.1f} kcyc/CTA = {tot/1.56e3:.0f} us at 1.56 GHz")
