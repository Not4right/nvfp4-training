"""Launchers for fused.cu + weight packing."""
import os
import torch
from nvrtc_util import Module, launch

_D = os.path.dirname(os.path.abspath(__file__))
_mod = None
NSM = torch.cuda.get_device_properties(0).multi_processor_count


def mod():
    global _mod
    if _mod is None:
        _mod = Module("".join(open(os.path.join(_D, n)).read() for n in ("fp4k.cu", "fused.cu", "fused_bwd.cu", "wg16.cu", "mlp.cu", "attn.cu", "attn_bwd.cu", "wprep.cu", "glue.cu")))
    return _mod


def fn(name, smem=0):
    return mod().fn(name, smem)


class Frag:
    __slots__ = ("d", "s", "amax")

    def __init__(self, d, s, amax):
        self.d, self.s, self.amax = d, s, amax


class PackedW:
    """Quantized weight W[N,K] (2D 16x16 blocks, ff dims pi-grouped) and its frag-native forms.
    ffN/ffK: whether W's N/K dim is the MLP hidden (ff) dim."""

    def __init__(self, w, ffN=False, ffK=False, rowperm=None):
        self.src, self.rowperm = w, rowperm
        self.w = w if rowperm is None else w.detach()[rowperm].contiguous()
        self.ffN, self.ffK = ffN, ffK
        N, K = w.shape
        dev = w.device
        self.amax = torch.zeros(1, device=dev)
        self.rn = torch.zeros(1, device=dev); self.cn = torch.zeros(1, device=dev)
        self.nb = torch.empty(N, K, dtype=torch.uint8, device=dev)
        self.bs = torch.empty(N // 16, K // 16, dtype=torch.uint8, device=dev)
        # fwd frag: (N rows, K contraction); bwd frag: (K rows, N contraction)
        self.f = Frag(torch.empty(N * K // 16, 2, dtype=torch.int32, device=dev),
                      torch.empty(N * K // 256, dtype=torch.int32, device=dev), self.amax)
        self.t = Frag(torch.empty(N * K // 16, 2, dtype=torch.int32, device=dev),
                      torch.empty(N * K // 256, dtype=torch.int32, device=dev), self.amax)
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        if self.rowperm is not None:
            torch.index_select(self.src.detach(), 0, self.rowperm, out=self.w)
        N, K = self.w.shape
        torch.amax(self.w.detach().abs().reshape(-1), 0, keepdim=True, out=self.amax)
        wd = self.w.detach()
        # Cauchy-Schwarz bound helpers (x1.25 margin for quantization): max row / col L2 norm
        torch.mul(wd.norm(dim=1).amax().reshape(1), 1.25, out=self.rn)
        torch.mul(wd.norm(dim=0).amax().reshape(1), 1.25, out=self.cn)
        nblk = (N // 16) * (K // 16)
        launch(fn("wq_nibbles"), ((nblk * 32 + 255) // 256,), (256,), [self.w, self.nb, self.bs, self.amax, N, K, int(self.ffN), int(self.ffK)])
        for fr, trans in ((self.f, 0), (self.t, 1)):
            Nf, Kf = (K, N) if trans else (N, K)
            nth = (Nf // 8) * (Kf // 64) * 32
            ffN = self.ffK if trans else self.ffN
            ffK = self.ffN if trans else self.ffK
            launch(fn("wq_frag"), ((nth + 255) // 256,), (256,), [self.nb, self.bs, fr.d, fr.s, N, K, trans, int(ffN), int(ffK)])

    def dequant(self):
        """logical dequantized weight (for tests)."""
        E = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=self.w.device)
        N, K = self.w.shape
        vals = E[self.nb.long()]
        gn = _grp(N, self.ffN, self.w.device); gk = _grp(K, self.ffK, self.w.device)
        sc = self.bs.view(torch.float8_e4m3fn).float()[gn][:, gk]
        return vals * sc * (self.amax / 2688.0)


def _pi_fwd(c):
    j, t, e = c // 8, (c % 8) // 2, c % 2
    return torch.where(j < 4, t * 8 + 2 * j + e, 32 + t * 8 + 2 * (j - 4) + e)


def _grp(n, perm, dev):
    c = torch.arange(n, device=dev)
    return (c // 64) * 4 + _pi_fwd(c % 64) // 16 if perm else c // 16


MLPF_SMEM = 67584


def mlp_fwd(x1, lnw, W1: PackedW, W2: PackedW, out=None):
    M = x1.shape[0]
    y = torch.empty_like(x1) if out is None else out
    launch(fn("mlp_fwd", MLPF_SMEM), (min(NSM, M // 128),), (256,),
           [x1, y, lnw, W1.f.d, W1.f.s, W1.amax, W2.f.d, W2.f.s, W2.amax, W1.rn, M], smem=MLPF_SMEM)
    return y


MLPB_SMEM = 101376


class WgradBuf:
    """token-major fp4 wgrad operand [F, M/2] + scales [F, M/16]"""

    def __init__(self, F, M, dev="cuda"):
        self.d = torch.empty(F, M // 2, dtype=torch.uint8, device=dev)
        self.s = torch.empty(F, M // 16, dtype=torch.uint8, device=dev)


def fattr(name, smem=0):
    import cupy
    f = fn(name, smem)
    return dict(regs=cupy.cuda.driver.funcGetAttribute(4, f.ptr), local=cupy.cuda.driver.funcGetAttribute(3, f.ptr))


def mlp_bwd(x1, dy, lnw, dlnw, W1: PackedW, W2: PackedW, bufs, amax_use, amax_cur, rngp, dx1=None, flags=0):
    """bufs: dict dy,a,du,h -> WgradBuf. amax_use/amax_cur: float32[4] / int32[4] (dy,a,du,h)."""
    M = x1.shape[0]
    dx1 = torch.empty_like(x1) if dx1 is None else dx1
    launch(fn("mlp_bwd", MLPB_SMEM), (min(NSM, M // 128),), (256,),
           [x1, dy, dx1, lnw, dlnw, W1.f.d, W1.f.s, W2.t.d, W2.t.s, W1.t.d, W1.t.s, W1.amax, W2.amax, W2.cn,
            bufs["dy"].d, bufs["dy"].s, bufs["a"].d, bufs["a"].s, bufs["du"].d, bufs["du"].s, bufs["h"].d, bufs["h"].s,
            amax_use, amax_cur, M, rngp, flags], smem=MLPB_SMEM)
    return dx1


ATTF_SMEM = 33792 + 128 * 136 * 2


def attn_fwd(x, lnw, Wqkv: PackedW, Wo: PackedW, out=None):
    M = x.shape[0]
    x1 = torch.empty_like(x) if out is None else out
    launch(fn("attn_fwd", ATTF_SMEM), (min(NSM, M // 128),), (256,),
           [x, x1, lnw, Wqkv.f.d, Wqkv.f.s, Wqkv.amax, Wo.f.d, Wo.f.s, Wo.amax, M], smem=ATTF_SMEM)
    return x1


def qkv_perm(dev="cuda"):
    """reordered row index -> natural Wqkv row (q|k|v blocks of 128, heads of 32)."""
    idx = []
    for p in range(2):
        for i in range(2):
            h = 2 * p + i
            idx += [h * 32 + j for j in range(32)] + [128 + h * 32 + j for j in range(32)]
        idx += [256 + (2 * p) * 32 + j for j in range(32)] + [256 + (2 * p + 1) * 32 + j for j in range(32)]
    return torch.tensor(idx, device=dev)


ATTB_SMEM = 86784


def attn_bwd(x, dx1, lnw, dlnw, Wqkv: PackedW, Wo: PackedW, bufs, amax_use, amax_cur, rngp, dx=None):
    """bufs: dx,o,dq,h -> WgradBuf (dq has 384 rows). amax slots: dx1,o,dqkv(wg),h1,dqkv(dgrad)."""
    M = x.shape[0]
    dx = torch.empty_like(x) if dx is None else dx
    launch(fn("attn_bwd", ATTB_SMEM), (min(2 * NSM, M // 64),), (256,),
           [x, dx1, dx, lnw, dlnw, Wqkv.f.d, Wqkv.f.s, Wqkv.t.d, Wqkv.t.s, Wqkv.amax, Wo.t.d, Wo.t.s, Wo.amax,
            bufs["dx"].d, bufs["dx"].s, bufs["o"].d, bufs["o"].s, bufs["dq"].d, bufs["dq"].s, bufs["h"].d, bufs["h"].s,
            amax_use, amax_cur, M, rngp], smem=ATTB_SMEM)
    return dx


class WeightSet:
    """All PackedW of the model re-quantized by 3 batched launches (stats, nibbles, frags)."""

    def __init__(self, pws):
        import numpy as np
        self.pws = pws
        dev = pws[0].w.device
        self.stats = torch.zeros(sum(2 + pw.w.shape[1] for pw in pws), dtype=torch.int32, device=dev)
        recs, off = [], 0
        for pw in pws:
            N, K = pw.w.shape
            src = pw.src if pw.rowperm is not None else pw.w
            if pw.rowperm is not None:
                pw.permi32 = pw.rowperm.to(torch.int32)
            ptrs = [src.data_ptr(), pw.permi32.data_ptr() if pw.rowperm is not None else 0, pw.nb.data_ptr(), pw.bs.data_ptr(),
                    pw.f.d.data_ptr(), pw.f.s.data_ptr(), pw.t.d.data_ptr(), pw.t.s.data_ptr(),
                    self.stats.data_ptr() + 4 * off, pw.amax.data_ptr(), pw.rn.data_ptr(), pw.cn.data_ptr()]
            off += 2 + K
            recs.append(np.array(ptrs, dtype=np.uint64).tobytes() + np.array([N, K, int(pw.ffN), int(pw.ffK)], dtype=np.int32).tobytes())
        self.desc = torch.frombuffer(bytearray(b"".join(recs)), dtype=torch.uint8).to(dev)
        self.nw = len(pws)

    def refresh(self):
        self.stats.zero_()
        launch(fn("wp_stats"), (8, self.nw), (256,), [self.desc])
        launch(fn("wp_nib"), (16, self.nw), (256,), [self.desc])
        launch(fn("wp_frag"), (16, self.nw, 2), (256,), [self.desc])


WG_SMEM = 4 * 8 * 576 * 2


class WGGemm:
    """Grouped (<=2 problems) split-K GEMM over WG16 operands; descriptors live on the device."""

    def __init__(self, probs, M, dev="cuda"):
        import numpy as np
        recs, self.nblks = [], []
        for (A, amaxA, B, amaxB, out, nsplit) in probs:
            FA, FB = A.d.shape[0], B.d.shape[0]
            assert FA % 128 == 0 and FB % 128 == 0
            ptrs = [A.d.data_ptr(), A.s.data_ptr(), B.d.data_ptr(), B.s.data_ptr(), out.data_ptr(), amaxA.data_ptr(), amaxB.data_ptr()]
            recs.append(np.array(ptrs, dtype=np.uint64).tobytes() + np.array([FA, FB, nsplit, 0], dtype=np.int32).tobytes())
            self.nblks.append((FA // 128) * (FB // 128) * nsplit)
        if len(recs) == 1:
            recs.append(recs[0]); self.nblks.append(0)
        self.desc = torch.frombuffer(bytearray(b"".join(recs)), dtype=torch.uint8).to(dev)
        self.M = M

    def __call__(self):
        launch(fn("wg_gemm", WG_SMEM), (sum(self.nblks),), (256,), [self.desc, self.nblks[0], self.M], smem=WG_SMEM)
