"""A/B compile-flag variants of mlp_bwd and attn_bwd, interleaved in one process (median of rounds)."""
import sys, statistics, torch
import fusedops as F
from nvrtc_util import Module, launch
import test_mlp_bwd_setup as S

VARIANTS = [(), ("-DXU_PACK",), ("-DXU_H2MAX",), ("-DXU_SRINT",), ("-DXU_PACK", "-DXU_H2MAX", "-DXU_SRINT")]
if len(sys.argv) > 1:
    VARIANTS = [tuple(a.split(",")) if a else () for a in sys.argv[1:]]
src = "".join(open(n).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu", "mlp.cu", "attn.cu", "attn_bwd.cu"))

# attention setup
M, dev = S.M, "cuda"
wq = torch.randn(384, 128, device=dev) / 128 ** .5; wo = torch.randn(128, 128, device=dev) / 128 ** .5
Wq, Wo = F.PackedW(wq, ffN=True, rowperm=F.qkv_perm()), F.PackedW(wo)
ab = {k: F.WgradBuf(n, M) for k, n in (("dx", 128), ("o", 128), ("dq", 384), ("h", 128))}
au5, ac5 = torch.full((5,), 0.01, device=dev), torch.zeros(5, dtype=torch.int32, device=dev)
xa = torch.randn(M, 128, device=dev).bfloat16(); dxa = (torch.randn(M, 128, device=dev) * 1e-3).bfloat16()

fns = []
for opts in VARIANTS:
    mod = Module(src, opts=opts)
    fm, fa = mod.fn("mlp_bwd", F.MLPB_SMEM), mod.fn("attn_bwd", F.ATTB_SMEM)
    margs = [S.x1, S.dy, torch.empty_like(S.x1), S.lnw, S.dlnw, S.W1.f.d, S.W1.f.s, S.W2.t.d, S.W2.t.s, S.W1.t.d, S.W1.t.s, S.W1.amax, S.W2.amax, S.W2.cn,
             S.bufs["dy"].d, S.bufs["dy"].s, S.bufs["a"].d, S.bufs["a"].s, S.bufs["du"].d, S.bufs["du"].s, S.bufs["h"].d, S.bufs["h"].s,
             S.amax_use, S.amax_cur, M, S.rng3, 0]
    aargs = [xa, dxa, torch.empty_like(xa), S.lnw, S.dlnw, Wq.f.d, Wq.f.s, Wq.t.d, Wq.t.s, Wq.amax, Wo.t.d, Wo.t.s, Wo.amax,
             ab["dx"].d, ab["dx"].s, ab["o"].d, ab["o"].s, ab["dq"].d, ab["dq"].s, ab["h"].d, ab["h"].s, au5, ac5, M, S.rng3]
    fns.append((opts, lambda fm=fm, margs=margs: launch(fm, (F.NSM,), (256,), margs, smem=F.MLPB_SMEM),
                lambda fa=fa, aargs=aargs: launch(fa, (2 * F.NSM,), (256,), aargs, smem=F.ATTB_SMEM)))


def timeit(f, n=30):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1e3


# warm clocks
for _ in range(200): fns[0][1]()
res = {o: ([], []) for o, _, _ in fns}
for _ in range(7):
    for o, fm, fa in fns:
        res[o][0].append(timeit(fm)); res[o][1].append(timeit(fa))
for o, _, _ in fns:
    print(f"{' '.join(o) or 'baseline':45s} mlp_bwd {statistics.median(res[o][0]):6.1f} us   attn_bwd {statistics.median(res[o][1]):6.1f} us")
