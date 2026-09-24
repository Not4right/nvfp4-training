"""NVFP4 quantization + cuBLAS block-scaled FP4 GEMM (torch._scaled_mm) for sm_120.

Recipe (NVIDIA, "Pretraining LLMs with NVFP4", 2025):
  * E2M1 values, one UE4M3 scale per 16 values, one FP32 per-tensor scale
  * weights: 2D 16x16 blocks so W and W^T quantize to the same values
  * gradients: stochastic rounding
  * wgrad: 16-point random Hadamard transform along the token dim on both operands
"""
import os
import torch
from nvrtc_util import Module, launch

_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvq.cu")).read()
_mod = None


def mod():
    global _mod
    if _mod is None:
        _mod = Module(_SRC)
    return _mod


class Q:
    """Packed NVFP4 matrix [R, C]: d [R, C/2] fp4x2, s swizzled e4m3 (flat), gs = per-tensor scale (device, fp32)."""
    __slots__ = ("d", "s", "gs")

    def __init__(self, d, s, gs):
        self.d, self.s, self.gs = d, s, gs


def amax(x, out=None):
    if out is None:
        out = torch.zeros(1, dtype=torch.float32, device=x.device)
    else:
        out.zero_()
    n = x.numel()
    assert n % 8 == 0 and x.is_contiguous()
    launch(mod().fn("amax_bf16"), (min(1024, (n // 8 + 255) // 256),), (256,), [x, n, out])
    return out


def _gs(am, k=1.0):
    return am * (k / 2688.0)


def quant_rows(x, sr=False, seed=0, am=None):
    """x [R, C] bf16 (row-contiguous) -> Q quantized along C."""
    R, C = x.shape
    assert x.stride(1) == 1 and C % 64 == 0 and R % 128 == 0, (R, C)
    if am is None:
        am = amax(x.contiguous())
    d = torch.empty(R, C // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(R * C // 16, dtype=torch.uint8, device=x.device)
    n = R * C // 16
    launch(mod().fn("quant_rows"), ((n + 255) // 256,), (256,), [x, x.stride(0), d, s, am, R, C, int(sr), seed & 0xFFFFFFFF])
    return Q(d.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn), _gs(am))


def quant_cols_rht(x, signs, sr=False, seed=0, am=None):
    """x [M, C] -> Q of x^T [C, M] quantized along M, after a 16-pt RHT along M."""
    M, C = x.shape
    x = x.contiguous()
    assert M % 64 == 0 and C % 128 == 0, (M, C)
    if am is None:
        am = amax(x)
    d = torch.empty(C, M // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(C * M // 16, dtype=torch.uint8, device=x.device)
    launch(mod().fn("quant_cols_rht"), (C // 64, M // 64), (256,), [x, d, s, am, M, C, signs, int(sr), seed & 0xFFFFFFFF])
    return Q(d.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn), _gs(am, 4.0))


def quant_w2d(w, am=None):
    """w [N, K] bf16 -> (Q of w, Q of w^T), identical values (16x16 blocks)."""
    N, K = w.shape
    w = w.contiguous()
    if am is None:
        am = amax(w)
    q = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    s = torch.empty(N * K // 16, dtype=torch.uint8, device=w.device)
    qt = torch.empty(K, N // 2, dtype=torch.uint8, device=w.device)
    st = torch.empty(K * N // 16, dtype=torch.uint8, device=w.device)
    nw = (N // 16) * (K // 16)
    launch(mod().fn("quant_w2d"), ((nw * 32 + 255) // 256,), (256,), [w, q, s, qt, st, am, N, K])
    g = _gs(am)
    f4, f8 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
    return Q(q.view(f4), s.view(f8), g), Q(qt.view(f4), st.view(f8), g)


def mm(a: Q, b: Q, out_dtype=torch.bfloat16, scale=True):
    """a [M,K], b [N,K] (both K-major) -> a @ b^T [M, N].  If scale=False returns (raw, alpha)."""
    raw = torch._scaled_mm(a.d, b.d.t(), a.s, b.s, out_dtype=out_dtype)
    alpha = a.gs * b.gs
    if not scale:
        return raw, alpha
    return raw.mul_(alpha.to(raw.dtype)) if out_dtype == torch.bfloat16 else raw.mul_(alpha)


# ---------- reference decode (tests)
_E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def unswizzle(s, R, C16):
    ncb = C16 // 4
    t = s.view(torch.uint8).reshape(R // 128, ncb, 32, 4, 4)       # rb, cb, r%32, (r%128)/32, c%4
    t = t.permute(0, 3, 2, 1, 4).reshape(R, C16)                    # rb,(r/32),(r%32), cb, c%4
    return t


def dequant(q: Q):
    d = q.d.view(torch.uint8)
    R, C = d.shape[0], d.shape[1] * 2
    lo, hi = (d & 15).long(), (d >> 4).long()
    tab = _E2M1.to(d.device)
    v = torch.stack([tab[lo], tab[hi]], -1).reshape(R, C)
    s = unswizzle(q.s, R, C // 16).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    return v * s * q.gs
