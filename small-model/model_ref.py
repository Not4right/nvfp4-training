"""Reference GPT (PyTorch) + bit-faithful NVFP4 fake-quant simulation of the recipe.

NVFP4 recipe (NVIDIA, "Pretraining LLMs with NVFP4"):
  * FP4 E2M1 values, one E4M3 scale per 16 contiguous values, plus an FP32 per-tensor scale
  * weights: 2D 16x16 blocks (so W and W^T quantize identically), round-to-nearest
  * activations: 1x16 blocks along the contraction dim, round-to-nearest
  * gradients: 1x16 blocks, stochastic rounding
  * wgrad: both operands get a 16-point random Hadamard transform along tokens first
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E4M3_MAX, FP4_MAX = 448.0, 6.0


def fp4_round(y, stochastic=False):
    """y already scaled into [-6, 6]. Returns nearest (or stochastic) E2M1 value."""
    a = y.abs().clamp(max=6.0)
    # step size of the grid at a: 0.5 below 2, 1 in [2,4), 2 in [4,6]
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    lo = torch.floor(a / step) * step
    frac = (a - lo) / step
    if stochastic:
        up = torch.rand_like(a) < frac
    else:  # round half to even on the grid index
        idx_lo = torch.round(lo / 0.5).where(lo < 2, torch.where(lo < 4, lo + 2, lo / 2 + 4))
        up = (frac > 0.5) | ((frac == 0.5) & (idx_lo % 2 == 1))
    q = torch.where(up, lo + step, lo)
    return torch.copysign(q.clamp(max=6.0), y)


def e4m3(x):
    return x.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).float()


def nvfp4_1d(x, stochastic=False, gamax=None):
    """Quantize-dequantize along the last dim in blocks of 16."""
    xf = x.float()
    amax = xf.abs().max() if gamax is None else gamax
    gs = (amax / (FP4_MAX * E4M3_MAX)).clamp(min=1e-30)
    xb = xf.reshape(*xf.shape[:-1], -1, 16) / gs
    bs = e4m3(xb.abs().amax(-1, keepdim=True) / FP4_MAX)
    q = fp4_round(xb / bs.clamp(min=1e-30), stochastic) * bs
    return (q * gs).reshape(xf.shape)


def nvfp4_2d(w):
    """16x16 block quantization for weights (out, in)."""
    wf = w.float()
    o, i = wf.shape
    gs = (wf.abs().max() / (FP4_MAX * E4M3_MAX)).clamp(min=1e-30)
    wb = wf.reshape(o // 16, 16, i // 16, 16).permute(0, 2, 1, 3) / gs
    bs = e4m3(wb.abs().amax((-1, -2), keepdim=True) / FP4_MAX)
    q = fp4_round(wb / bs.clamp(min=1e-30)) * bs
    return (q * gs).permute(0, 2, 1, 3).reshape(o, i)


def hadamard16(device):
    h = torch.tensor([[1.0]])
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / 4).to(device)


class _FP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, signs, rht):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        wq = nvfp4_2d(w)
        xq = nvfp4_1d(x2)
        ctx.save_for_backward(x2, wq, signs)
        ctx.rht, ctx.shp = rht, shp
        return (xq @ wq.t()).to(x.dtype).reshape(*shp[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        x2, wq, signs = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1]).float()
        dx = nvfp4_1d(g2, stochastic=True) @ wq  # dgrad: contract over out-features
        # wgrad: contract over tokens -> quantize along tokens (transpose), RHT first
        xt, gt = x2.float().t(), g2.t()  # (in, M), (out, M)
        if ctx.rht:
            H = hadamard16(g.device) * signs[None, :]
            xt = (xt.reshape(xt.shape[0], -1, 16) @ H.t()).reshape(xt.shape)
            gt = (gt.reshape(gt.shape[0], -1, 16) @ H.t()).reshape(gt.shape)
        dw = nvfp4_1d(gt, stochastic=True) @ nvfp4_1d(xt).t()
        return dx.to(g.dtype).reshape(*ctx.shp), dw, None, None


class Linear(nn.Module):
    def __init__(self, i, o, mode="bf16"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) / math.sqrt(i))
        self.mode = mode
        self.register_buffer("signs", torch.ones(16))

    def forward(self, x):
        if self.mode == "fp4":
            q = self.q
            return torch.ops.nvfp4.linear(x.to(torch.bfloat16), self.weight, q.wd, q.ws, q.wtd, q.wts, q.amax)
        if self.mode == "fp4sim":
            if self.training:
                self.signs = (torch.randint(0, 2, (16,), device=x.device) * 2 - 1).float()
            return _FP4Linear.apply(x, self.weight, self.signs, True)
        return F.linear(x, self.weight.to(x.dtype))


class Block(nn.Module):
    def __init__(self, d, h, ff, mode):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d, bias=False)
        self.qkv = Linear(d, 3 * d, mode)
        self.proj = Linear(d, d, mode)
        self.ln2 = nn.LayerNorm(d, bias=False)
        self.fc1 = Linear(d, ff, mode)
        self.fc2 = Linear(ff, d, mode)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, vocab=16, T=32, d=128, L=5, h=4, ff=512, mode="bf16"):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, h, ff, mode) for _ in range(L))
        self.lnf = nn.LayerNorm(d, bias=False)
        self.head = nn.Linear(d, vocab, bias=False)  # stays BF16 per recipe
        for b in self.blocks:  # GPT-2 style residual-branch scaling
            b.proj.weight.data /= math.sqrt(2 * L)
            b.fc2.weight.data /= math.sqrt(2 * L)

    def forward(self, idx):
        x = self.tok(idx) + self.pos[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def fp4_refresh(model):
    import fp4torch
    for m in model.modules():
        if isinstance(m, Linear) and m.mode == "fp4":
            if not hasattr(m, "q"):
                m.q = fp4torch.FP4Weight(m.weight)
            else:
                m.q.refresh()


def masked_loss(logits, y, w):
    l = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    return (l * w.reshape(-1)).sum() / w.sum()
