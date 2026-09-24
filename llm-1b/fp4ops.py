"""Python launchers for fp4k.cu kernels."""
import os
import torch
from nvrtc_util import Module, launch

_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp4k.cu")).read()
_mod = None
SMEM_GEMM = 3 * (128 * 64 * 2 + 128 * 8 * 2)


def mod():
    global _mod
    if _mod is None:
        _mod = Module(_SRC)
    return _mod


class Q:
    """Packed NVFP4 tensor: data [R, C/2] uint8, scales [R, C/16] uint8, amax (device scalar), k (extra scale)."""
    __slots__ = ("d", "s", "amax", "k")

    def __init__(self, d, s, amax, k=1.0):
        self.d, self.s, self.amax, self.k = d, s, amax, k


def amax_of(x):
    return x.abs().amax().float().reshape(1)


def quant_rows(x, sr=False, seed=0, amax=None):
    R, C = x.shape
    x = x.contiguous()
    a = amax_of(x) if amax is None else amax
    d = torch.empty(R, C // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(R, C // 16, dtype=torch.uint8, device=x.device)
    nblk = R * C // 16
    launch(mod().fn("quant_rows"), ((nblk + 255) // 256,), (256,), [x, d, s, a, nblk, int(sr), seed & 0xFFFFFFFF])
    return Q(d, s, a)


def quant_cols_rht(x, signs, sr=False, seed=0, amax=None):
    """x [M, C] -> quantized along M of x^T: data [C, M/2]. RHT applied along M (16-blocks)."""
    M, C = x.shape
    x = x.contiguous()
    a = amax_of(x) if amax is None else amax
    d = torch.empty(C, M // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(C, M // 16, dtype=torch.uint8, device=x.device)
    launch(mod().fn("quant_cols_rht"), ((C + 127) // 128, M // 16), (128,), [x, d, s, a, M, C, signs, int(sr), seed & 0xFFFFFFFF])
    return Q(d, s, a, 4.0)


def quant_w2d(w):
    N, K = w.shape
    a = amax_of(w)
    q = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    s = torch.empty(N, K // 16, dtype=torch.uint8, device=w.device)
    qt = torch.empty(K, N // 2, dtype=torch.uint8, device=w.device)
    st = torch.empty(K, N // 16, dtype=torch.uint8, device=w.device)
    nw = (N // 16) * (K // 16)
    launch(mod().fn("quant_w2d"), ((nw * 32 + 255) // 256,), (256,), [w.contiguous(), q, s, qt, st, a, N, K])
    return Q(q, s, a), Q(qt, st, a)


def gemm(a: Q, b: Q, out=None, splitk=1):
    """a: [M,K] fp4, b: [N,K] fp4 -> [M,N]. splitk>1 -> fp32 atomic accumulate into out."""
    M, N, K = a.d.shape[0], b.d.shape[0], a.s.shape[1] * 16
    assert M % 128 == 0 and N % 128 == 0 and K % 128 == 0, (M, N, K)
    if splitk == 1 and out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a.d.device)
        mode = 0
    else:
        if out is None:
            out = torch.zeros(M, N, dtype=torch.float32, device=a.d.device)
        mode = 1
    klen = ((K // splitk + 127) // 128) * 128
    nz = (K + klen - 1) // klen
    launch(mod().fn("gemm_fp4", SMEM_GEMM), (N // 128, M // 128, nz), (256,),
           [a.d, a.s, b.d, b.s, out, a.amax, b.amax, float(a.k), float(b.k), M, N, K, klen, mode,
            a.d.stride(0), a.s.stride(0), b.d.stride(0), b.s.stride(0)], smem=SMEM_GEMM)
    return out


# ---------- reference decode (for tests)
_E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def dequant(q: Q):
    d = q.d
    lo, hi = (d & 15).long(), (d >> 4).long()
    v = torch.stack([_E2M1.to(d.device)[lo], _E2M1.to(d.device)[hi]], -1).reshape(d.shape[0], -1)
    s = q.s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    return v * s * (q.amax * q.k / 2688.0)
