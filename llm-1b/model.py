"""1B decoder-only transformer with NVFP4 linear layers (fprop / dgrad / wgrad).

Llama-style: pre-RMSNorm, GQA, RoPE, QK-norm, SwiGLU, tied embeddings.
Block linears run through FP4Linear; attention math, norms, embedding and LM head stay BF16.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend

import nvq


@dataclass
class Config:
    vocab: int = 49152
    d: int = 2048
    layers: int = 20
    heads: int = 16
    kv_heads: int = 4
    ff: int = 5632
    rope_theta: float = 100000.0
    max_seq: int = 4096
    norm_eps: float = 1e-5

    @property
    def hd(self):
        return self.d // self.heads


# ------------------------------------------------------------------ FP4 linear
class LinCtx:
    """Global switches for the linear layers (set by the trainer)."""
    fp4 = True
    seed = 0          # changes every microbatch (stochastic rounding)


_RHT_SIGNS = 0b1010011010110001


class FP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, lid):
        # x [T, K] bf16, w [N, K] bf16
        ctx.save_for_backward(x, w)
        ctx.lid = lid
        qx = nvq.quant_rows(x)
        qw, _ = nvq.quant_w2d(w)
        return nvq.mm(qx, qw)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gy = gy.contiguous()
        seed = (LinCtx.seed * 1000003 + ctx.lid * 7919) & 0xFFFFFFFF
        ag = nvq.amax(gy)
        # dgrad: gx = gy @ w     (quantized along N; W^T uses the same 2D-block values)
        _, qwt = nvq.quant_w2d(w)
        qg = nvq.quant_rows(gy, sr=True, seed=seed, am=ag)
        gx = nvq.mm(qg, qwt)
        # wgrad: gw = gy^T @ x   (both quantized along tokens, after a 16-pt RHT)
        qgt = nvq.quant_cols_rht(gy, _RHT_SIGNS, sr=True, seed=seed ^ 0x5bd1e995, am=ag)
        qxt = nvq.quant_cols_rht(x, _RHT_SIGNS)
        gw = nvq.mm(qgt, qxt)
        return gx, gw, None


class Linear(nn.Module):
    """Bias-free linear whose GEMMs run in NVFP4 when LinCtx.fp4, else BF16."""
    _next_id = 0

    def __init__(self, fin, fout, std):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fout, fin))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)
        self.lid = Linear._next_id
        Linear._next_id += 1

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if LinCtx.fp4:
            y = FP4LinearFn.apply(x2, self.weight, self.lid)
        else:
            y = x2 @ self.weight.t()
        return y.view(*shp[:-1], -1)


# ------------------------------------------------------------------ blocks
@torch.compile(dynamic=False)
def rms_norm(x, w, eps: float):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


@torch.compile(dynamic=False)
def qk_prep(q, k, v, qn, kn, cos, sin, eps: float, r: int):
    """QK-norm + RoPE, heads-first layout, KV heads expanded for cuDNN attention."""
    def nr(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        h = xf.shape[-1] // 2
        c, s = cos[: xf.shape[1], None].float(), sin[: xf.shape[1], None].float()
        x1, x2 = xf[..., :h], xf[..., h:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype).transpose(1, 2)
    q = nr(q, qn)
    k = nr(k, kn).repeat_interleave(r, 1)
    v = v.transpose(1, 2).repeat_interleave(r, 1)
    return q.contiguous(), k.contiguous(), v.contiguous()


@torch.compile(dynamic=False)
def swiglu(gu):
    g, u = gu.chunk(2, dim=-1)
    return (F.silu(g.float()) * u.float()).to(gu.dtype)


def rope_cache(cfg, device):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.hd, 2, device=device).float() / cfg.hd))
    t = torch.arange(cfg.max_seq, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f).bfloat16(), torch.sin(f).bfloat16()


def apply_rope(x, cos, sin):
    # x [B, H, T, hd]; rotate halves (NeoX style)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[: x.shape[2]], sin[: x.shape[2]]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


class Block(nn.Module):
    def __init__(self, cfg: Config, li: int):
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.d, cfg.hd
        std = 0.02
        out_std = 0.02 / math.sqrt(2 * cfg.layers)
        self.n1 = nn.Parameter(torch.ones(d))
        self.n2 = nn.Parameter(torch.ones(d))
        self.qn = nn.Parameter(torch.ones(hd))
        self.kn = nn.Parameter(torch.ones(hd))
        self.qkv = Linear(d, (cfg.heads + 2 * cfg.kv_heads) * hd, std)
        self.o = Linear(cfg.heads * hd, d, out_std)
        self.gu = Linear(d, 2 * cfg.ff, std)
        self.down = Linear(cfg.ff, d, out_std)

    def forward(self, x, cos, sin):
        cfg = self.cfg
        B, T, _ = x.shape
        H, KV, hd = cfg.heads, cfg.kv_heads, cfg.hd
        h = rms_norm(x, self.n1, cfg.norm_eps)
        qkv = self.qkv(h).view(B, T, H + 2 * KV, hd)
        q, k, v = qkv.split([H, KV, KV], dim=2)
        # cuDNN is the only fast SDPA backend in the Windows build and it has no GQA
        # (enable_gqa silently falls back to the fp32 math path), so KV heads are expanded.
        q, k, v = qk_prep(q, k, v, self.qn, self.kn, cos, sin, cfg.norm_eps, H // KV)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, H * hd))
        h = rms_norm(x, self.n2, cfg.norm_eps)
        x = x + self.down(swiglu(self.gu(h)))
        return x


# ------------------------------------------------------------------ LM head + CE (chunked)
class ChunkedCE(torch.autograd.Function):
    """loss = mean CE(h @ W^T, y) over positions with mask, + z_loss * mean(lse^2).
    Logits never exist for the full batch; backward recomputes them chunk by chunk."""

    @staticmethod
    def forward(ctx, h, W, y, m, z, chunk):
        T = h.shape[0]
        n = m.sum().clamp_min(1).float()
        loss = torch.zeros((), device=h.device, dtype=torch.float32)
        for i in range(0, T, chunk):
            lg = (h[i:i + chunk] @ W.t()).float()
            lse = torch.logsumexp(lg, -1)
            tgt = lg.gather(1, y[i:i + chunk, None]).squeeze(1)
            mm = m[i:i + chunk]
            loss += (((lse - tgt) + z * lse * lse) * mm).sum()
        ctx.save_for_backward(h, W, y, m)
        ctx.z, ctx.chunk, ctx.n = z, chunk, n
        return loss / n

    @staticmethod
    def backward(ctx, g):
        h, W, y, m = ctx.saved_tensors
        z, chunk, n = ctx.z, ctx.chunk, ctx.n
        gh = torch.empty_like(h)
        gW = torch.zeros(W.shape, device=W.device, dtype=torch.float32)
        scale = g / n
        for i in range(0, h.shape[0], chunk):
            hc = h[i:i + chunk]
            lg = (hc @ W.t()).float()
            lse = torch.logsumexp(lg, -1, keepdim=True)
            p = torch.exp(lg - lse)
            mm = m[i:i + chunk, None].float() * scale
            gl = p * (mm * (1 + 2 * z * lse))
            gl.scatter_add_(1, y[i:i + chunk, None], -mm)
            gl = gl.bfloat16()
            gh[i:i + chunk] = gl @ W
            gW += (gl.t() @ hc).float()
        return gh, gW.to(W.dtype), None, None, None, None


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        Linear._next_id = 0
        self.emb = nn.Parameter(torch.empty(cfg.vocab, cfg.d))
        nn.init.normal_(self.emb, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.layers)])
        self.nf = nn.Parameter(torch.ones(cfg.d))
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self.grad_ckpt = True
        self.z_loss = 1e-4

    def _rope(self, dev):
        if self.cos is None or self.cos.device != dev:
            self.cos, self.sin = rope_cache(self.cfg, dev)

    def hidden(self, idx):
        self._rope(idx.device)
        x = F.embedding(idx, self.emb)
        for b in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(b, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = b(x, self.cos, self.sin)
        return rms_norm(x, self.nf, self.cfg.norm_eps)

    def forward(self, idx, targets, mask=None, chunk=2048):
        h = self.hidden(idx)
        h = h.reshape(-1, h.shape[-1])
        y = targets.reshape(-1)
        m = (torch.ones_like(y, dtype=torch.bfloat16) if mask is None else mask.reshape(-1).to(torch.bfloat16))
        return ChunkedCE.apply(h, self.emb, y, m, self.z_loss, chunk)

    @torch.no_grad()
    def logits_last(self, idx):
        h = self.hidden(idx)
        return (h[:, -1] @ self.emb.t()).float()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
