import statistics, torch, fusedops as F
from nvrtc_util import Module, launch
import test_mlp_bwd_setup as S
M = S.M
src = "".join(open(n).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu", "mlp.cu"))
y = torch.empty_like(S.x1)
def timeit(f, n=30):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n * 1e3
cfgs = [0, 300, 600, 1000, 1600, 2500]
runs = []
for ns in cfgs:
    mod = Module(src, opts=(f"-DSTAGGER_F={ns}", f"-DSTAGGER_B={ns}"))
    ff, fb = mod.fn("mlp_fwd", F.MLPF_SMEM), mod.fn("mlp_bwd", F.MLPB_SMEM)
    fa = [S.x1, y, S.lnw, S.W1.f.d, S.W1.f.s, S.W1.amax, S.W2.f.d, S.W2.f.s, S.W2.amax, S.W1.rn, M]
    ba = [S.x1, S.dy, torch.empty_like(S.x1), S.lnw, S.dlnw, S.W1.f.d, S.W1.f.s, S.W2.t.d, S.W2.t.s, S.W1.t.d, S.W1.t.s, S.W1.amax, S.W2.amax, S.W2.cn,
          S.bufs["dy"].d, S.bufs["dy"].s, S.bufs["a"].d, S.bufs["a"].s, S.bufs["du"].d, S.bufs["du"].s, S.bufs["h"].d, S.bufs["h"].s, S.amax_use, S.amax_cur, M, S.rng3, 0]
    runs.append((ns, lambda ff=ff, fa=fa: launch(ff, (F.NSM,), (256,), fa, smem=F.MLPF_SMEM), lambda fb=fb, ba=ba: launch(fb, (F.NSM,), (256,), ba, smem=F.MLPB_SMEM)))
for _ in range(100): runs[0][1]()
res = {ns: ([], []) for ns in cfgs}
for _ in range(7):
    for ns, f1, f2 in runs:
        res[ns][0].append(timeit(f1)); res[ns][1].append(timeit(f2))
for ns in cfgs:
    print(f"stagger {ns:5d} ns: mlp_fwd {statistics.median(res[ns][0]):6.1f} us  mlp_bwd {statistics.median(res[ns][1]):6.1f} us")
