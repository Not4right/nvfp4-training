import torch, fusedops as F
from nvrtc_util import Module, launch
from testutil import bench
import test_mlp_bwd_setup as S
src = "".join(open(n).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu", "mlp.cu"))
for opts, name in (((), "normal"), (("-DNOSTORE",), "no emission data stores")):
    mod = Module(src, opts=opts); f = mod.fn("mlp_bwd", F.MLPB_SMEM)
    args = [S.x1, S.dy, torch.empty_like(S.x1), S.lnw, S.dlnw, S.W1.f.d, S.W1.f.s, S.W2.t.d, S.W2.t.s, S.W1.t.d, S.W1.t.s, S.W1.amax, S.W2.amax, S.W2.cn,
            S.bufs["dy"].d, S.bufs["dy"].s, S.bufs["a"].d, S.bufs["a"].s, S.bufs["du"].d, S.bufs["du"].s, S.bufs["h"].d, S.bufs["h"].s,
            S.amax_use, S.amax_cur, S.M, S.rng3, 0]
    print(f"{name}: {bench(lambda: launch(f, (F.NSM,), (256,), args, smem=F.MLPB_SMEM)):.1f} us")
