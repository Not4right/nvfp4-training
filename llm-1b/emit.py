"""Launchers for emit.cu (NVFP4 operand emission) + the cuBLAS block-scaled FP4 GEMM."""
import os
import torch
from nvrtc_util import Module, launch

_mod = None
F4, F8 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn


def mod():
    global _mod
    if _mod is None:
        _mod = Module(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "emit.cu")).read())
    return _mod


def fn(name):
    return mod().fn(name)


class RowQ:
    """Row operand [R, C]: data (fp4x2 [R, C/2]), swizzled scales, rs[R] per-row fp32 scale."""
    __slots__ = ("d", "s", "rs")

    def __init__(self, R, C, dev="cuda"):
        self.d = torch.empty(R, C // 2, dtype=torch.uint8, device=dev)
        self.s = torch.empty(R * C // 16, dtype=torch.uint8, device=dev)
        self.rs = torch.empty(R, dtype=torch.float32, device=dev)


class ColQ:
    """Column operand of X [M, C]: X^T quantized along M -> data [C, M/2], swizzled scales."""
    __slots__ = ("d", "s")

    def __init__(self, M, C, dev="cuda"):
        self.d = torch.empty(C, M // 2, dtype=torch.uint8, device=dev)
        self.s = torch.empty(C * M // 16, dtype=torch.uint8, device=dev)


def rms_emit(x, w, q: RowQ = None, h=None, rstd=None, y=None, ys=None, ygw=None, xout=None, eps=1e-5):
    T = x.shape[0]
    assert x.shape[1] == 2048 and x.is_contiguous()
    launch(fn("rms_emit"), ((T + 7) // 8,), (256,),
           [x, y if y is not None else 0, ys if ys is not None else 0, ygw if ygw is not None else 0,
            xout if xout is not None else 0, w, h if h is not None else 0,
            q.d if q else 0, q.s if q else 0, q.rs if q else 0, rstd if rstd is not None else 0,
            T, float(eps), int(q is not None)])


def swiglu_emit(gu, rs_in, gw, q: RowQ = None, a=None):
    T, F2 = gu.shape
    if F2 == 2 * 5632:     # single-pass register kernel: 2 warps per row, 4 rows per CTA
        launch(fn("swiglu_emit_5632"), ((T + 3) // 4,), (256,),
               [gu, rs_in, gw, a if a is not None else 0, q.d if q else 0, q.s if q else 0, q.rs if q else 0, T, int(q is not None)])
        return
    launch(fn("swiglu_emit"), ((T + 7) // 8,), (256,),
           [gu, rs_in, gw, F2 // 2, a if a is not None else 0, q.d if q else 0, q.s if q else 0, q.rs if q else 0,
            T, int(q is not None)])


def row_emit(x, q: RowQ, rng, sr=False, salt=0):
    T, C = x.shape
    assert x.stride(1) == 1 and C % 256 == 0
    nw = {11264: 4}.get(C)     # narrow rows: the two-pass kernel is as fast (row stays in L1/L2)
    if nw:                 # single-pass register kernels
        launch(fn(f"row_emit_{C}"), ((T * nw + 7) // 8,), (256,), [x, x.stride(0), q.d, q.s, q.rs, T, int(sr), rng, salt & 0xFFFFFFFF])
        return
    launch(fn("row_emit"), ((T + 7) // 8,), (256,), [x, x.stride(0), C, q.d, q.s, q.rs, T, int(sr), rng, salt & 0xFFFFFFFF])


def col_emit(x, q: ColQ, amax_use, amax_cur, rng, sr=False, salt=0):
    M, C = x.shape
    assert x.is_contiguous() and M % 256 == 0 and C % 64 == 0
    launch(fn("col_emit"), (C // 64, M // 256), (256,), [x, M, C, q.d, q.s, amax_use, amax_cur, int(sr), rng, salt & 0xFFFFFFFF])


def mom_acc(m, g, alpha, scale, rng, salt=0):
    n = m.numel()
    assert n % 8 == 0
    launch(fn("mom_acc"), (min(4096, (n // 8 + 255) // 256),), (256,), [m, g, alpha, float(scale), n, rng, salt & 0xFFFFFFFF])


def amax(x, out):
    out.zero_()
    n = x.numel()
    launch(fn("amax_bf16"), (min(1024, (n // 8 + 255) // 256),), (256,), [x, n, out])
    return out


_WSCR = {}


class WQ:
    """FP4 form of a weight W [N, K]: fwd operand (N rows, K contraction) and W^T (K rows, N contraction).
    The per-tensor amax is refreshed once per optimizer step; the FP4 data is produced just in time into a
    scratch buffer shared by all weights of the same shape (1B model: a full per-step cache costs 1.1 GB of
    VRAM, re-quantizing costs ~1% of a microbatch)."""

    def __init__(self, N, K, dev="cuda"):
        self.N, self.K, self.w = N, K, None
        key = (N, K)
        if key not in _WSCR:
            _WSCR[key] = (torch.empty(N, K // 2, dtype=torch.uint8, device=dev), torch.empty(N * K // 16, dtype=torch.uint8, device=dev),
                          torch.empty(K, N // 2, dtype=torch.uint8, device=dev), torch.empty(K * N // 16, dtype=torch.uint8, device=dev))
        self.d, self.s, self.td, self.ts = _WSCR[key]
        self.amax = torch.zeros(1, dtype=torch.float32, device=dev)
        self.gs = torch.zeros(1, dtype=torch.float32, device=dev)   # amax / 2688

    def refresh(self, w):
        self.w = w
        amax(w, self.amax)
        torch.mul(self.amax, 1.0 / 2688.0, out=self.gs)

    def prep(self):
        nw = (self.N // 16) * (self.K // 16)
        launch(fn("quant_w2d"), ((nw * 32 + 255) // 256,), (256,), [self.w, self.d, self.s, self.td, self.ts, self.amax, self.N, self.K])
        return self


def mm(ad, as_, bd, bs, out_dtype=torch.bfloat16):
    """raw = A @ B^T with block scales only. A [M,K] and B [N,K] fp4 row-major (K contraction)."""
    return torch._scaled_mm(ad.view(F4), bd.view(F4).t(), as_.view(F8), bs.view(F8), out_dtype=out_dtype)


def mm_row_w(q: RowQ, w: WQ):
    """forward:  y_raw = Q(x) @ Q(W)^T   (true y = y_raw * q.rs[:,None] * w.gs).  Quantizes W just in time
    (also leaves W^T in the scratch for a later dgrad of the same weight)."""
    w.prep()
    return mm(q.d, q.s, w.d, w.s)


def mm_row_wt(q: RowQ, w: WQ):
    """dgrad:   dx_raw = Q(dy) @ Q(W)    (true = raw * q.rs[:,None] * w.gs)"""
    return mm(q.d, q.s, w.td, w.ts)


def mm_col(a: ColQ, b: ColQ):
    """wgrad:   dW_raw = Q(H dy)^T Q(H x)  [N, K]  (true = raw * gsA * gsB)"""
    return mm(a.d, a.s, b.d, b.s)


# ---------------- reference decode (tests)
_E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def unswizzle(s, R, C16):
    t = s.view(torch.uint8).reshape(R // 128, C16 // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(R, C16)
    return t


def dequant(d, s, gs_rows=None, gs=1.0):
    d = d.view(torch.uint8)
    R, C = d.shape[0], d.shape[1] * 2
    tab = _E2M1.to(d.device)
    v = torch.stack([tab[(d & 15).long()], tab[(d >> 4).long()]], -1).reshape(R, C)
    sc = unswizzle(s, R, C // 16).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    out = v * sc * gs
    if gs_rows is not None:
        out = out * gs_rows[:, None]
    return out
