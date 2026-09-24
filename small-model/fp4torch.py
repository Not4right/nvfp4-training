"""NVFP4 linear layer as torch custom ops (opaque to torch.compile), real FP4 tensor cores.

fprop : y  = Q_rn(x)  @ Q2d(W)^T
dgrad : dx = Q_sr(dy) @ Q2d(W)          (W^T stored pre-quantized, same values)
wgrad : dW = Q_sr(H dy)^T @ Q_rn(H x)   (16-pt random Hadamard along tokens)
"""
import random
import torch
import fp4ops as F

_ctr = [0]


def _seed():
    _ctr[0] += 1
    return (_ctr[0] * 2654435761) & 0xFFFFFFFF


@torch.library.custom_op("nvfp4::fwd", mutates_args=())
def fp4_fwd(x: torch.Tensor, wd: torch.Tensor, ws: torch.Tensor, wamax: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    y = F.gemm(F.quant_rows(x2), F.Q(wd, ws, wamax))
    return y.reshape(*x.shape[:-1], wd.shape[0])


@fp4_fwd.register_fake
def _(x, wd, ws, wamax):
    return x.new_empty(*x.shape[:-1], wd.shape[0])


@torch.library.custom_op("nvfp4::bwd", mutates_args=())
def fp4_bwd(g: torch.Tensor, x: torch.Tensor, wtd: torch.Tensor, wts: torch.Tensor, wamax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g2 = g.reshape(-1, g.shape[-1]).contiguous()
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    dx = F.gemm(F.quant_rows(g2, sr=True, seed=_seed()), F.Q(wtd, wts, wamax))
    signs = random.getrandbits(16)
    dw = F.gemm(F.quant_cols_rht(g2, signs, sr=True, seed=_seed()), F.quant_cols_rht(x2, signs), splitk=16)
    return dx.reshape(x.shape), dw


@fp4_bwd.register_fake
def _(g, x, wtd, wts, wamax):
    return x.new_empty(x.shape), g.new_empty(g.shape[-1], x.shape[-1], dtype=torch.float32)


@torch.library.custom_op("nvfp4::linear", mutates_args=())
def fp4_linear(x: torch.Tensor, w: torch.Tensor, wd: torch.Tensor, ws: torch.Tensor, wtd: torch.Tensor,
               wts: torch.Tensor, wamax: torch.Tensor) -> torch.Tensor:
    return fp4_fwd(x, wd, ws, wamax)


@fp4_linear.register_fake
def _(x, w, wd, ws, wtd, wts, wamax):
    return x.new_empty(*x.shape[:-1], w.shape[0])


def _setup(ctx, inputs, output):
    x, w, wd, ws, wtd, wts, wamax = inputs
    ctx.save_for_backward(x, wtd, wts, wamax)


def _backward(ctx, g):
    x, wtd, wts, wamax = ctx.saved_tensors
    dx, dw = fp4_bwd(g.contiguous(), x, wtd, wts, wamax)
    return dx, dw, None, None, None, None, None


torch.library.register_autograd("nvfp4::linear", _backward, setup_context=_setup)


class FP4Weight:
    """Holds the per-step quantized copies of a master fp32 weight."""

    def __init__(self, w):
        self.w = w
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        q, qt = F.quant_w2d(self.w.data)
        if not hasattr(self, "wd"):
            self.wd, self.ws, self.wtd, self.wts, self.amax = q.d, q.s, qt.d, qt.s, q.amax
        else:
            self.wd.copy_(q.d); self.ws.copy_(q.s); self.wtd.copy_(qt.d); self.wts.copy_(qt.s); self.amax.copy_(q.amax)
