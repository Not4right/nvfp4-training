import statistics, torch, fusedops as F
from nvrtc_util import Module, launch
import test_mlp_bwd_setup as S
M = S.M
src = "".join(open(n).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu"))
dW2 = torch.zeros(128, 512, device="cuda"); dW1 = torch.zeros(512, 128, device="cuda")
res = {}
for opts in ((), ("-DWG_NOATOMIC",)):
    mod = Module(src, opts=opts); f = mod.fn("wg_gemm", F.WG_SMEM)
    for ns in (4, 8, 16):
        wg = F.WGGemm([(S.bufs["dy"], S.amax_use[0:1], S.bufs["a"], S.amax_use[1:2], dW2, ns), (S.bufs["du"], S.amax_use[2:3], S.bufs["h"], S.amax_use[3:4], dW1, ns)], M)
        run = lambda: launch(f, (sum(wg.nblks),), (256,), [wg.desc, wg.nblks[0], M], smem=F.WG_SMEM)
        for _ in range(50): run()
        ts = []
        for _ in range(7):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
            for _ in range(30): run()
            e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) / 30 * 1e3)
        print(f"{' '.join(opts) or 'atomics':14s} nsplit={ns:2d}: {statistics.median(ts):6.1f} us  ({2*2*128*512*M/statistics.median(ts)/1e6:.0f} TFLOPS)")
