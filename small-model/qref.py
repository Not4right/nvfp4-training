"""Exact torch emulation of the fused kernels' activation quantization (for tests)."""
import torch
from model_ref import fp4_round, e4m3
from fusedops import _pi_fwd


def pi_inv_idx(n, dev):
    """phys position p -> logical column, per 64 chunk, for n columns."""
    c = torch.arange(n, device=dev)
    phys = (c // 64) * 64 + _pi_fwd(c % 64)
    inv = torch.empty_like(c); inv[phys] = c
    return inv


def q_rows(x, chunk=None, perm=False):
    """quantize-dequantize rows of x [M,K]; per-row (or per-row-per-chunk) fp32 scale, 16-blocks
    in phys order (pi-permuted inside 64 chunks when perm)."""
    x = x.float()
    M, K = x.shape
    idx = pi_inv_idx(K, x.device) if perm else torch.arange(K, device=x.device)
    xp = x[:, idx]
    ch = chunk or K
    xc = xp.reshape(M, K // ch, ch)
    gs = xc.abs().amax(-1, keepdim=True) / 2688.0
    xb = (xc / gs.clamp(min=1e-30)).reshape(M, K // ch, ch // 16, 16)
    bs = e4m3(xb.abs().amax(-1, keepdim=True) / 6.0)
    q = fp4_round(xb / bs.clamp(min=1e-30)) * bs
    q = (q.reshape(M, K // ch, ch) * gs).reshape(M, K)
    out = torch.empty_like(q); out[:, idx] = q
    return out


def q_rows_gs(x, gs, perm=False):
    """like q_rows but with a given per-row global scale gs [M,1]"""
    x = x.float()
    M, K = x.shape
    idx = pi_inv_idx(K, x.device) if perm else torch.arange(K, device=x.device)
    xb = (x[:, idx] / gs).reshape(M, K // 16, 16)
    bs = e4m3(xb.abs().amax(-1, keepdim=True) / 6.0)
    q = (fp4_round(xb / bs.clamp(min=1e-30)) * bs).reshape(M, K) * gs
    out = torch.empty_like(q); out[:, idx] = q
    return out
