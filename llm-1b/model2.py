"""1B NVFP4 transformer with a hand-written forward/backward (no autograd graph except inside attention).

Everything lives in VRAM:
  * params:   one flat BF16 buffer (views per tensor); matrices updated with stochastic rounding
  * momentum: one flat BF16 buffer for the block matrices (Muon); every microbatch's wgrad is added into it
              directly (mom_acc, SR), so there is no gradient buffer
  * emb grad: FP32 [V, D] (tied embedding / LM head, Adam), small params: FP32 grads
  * FP4 weights: per-tensor amax refreshed once per step; W / W^T quantized just in time (shared scratch)
Activation checkpoint = the residual stream at every block input; blocks are recomputed in backward.

NVFP4 recipe (NVIDIA 2025) + Desktop/nvfp4 techniques:
  row operands (fprop activations, dgrad gradients): exact per-row FP32 scale, emitted by the producer
  column operands (wgrad): 16-pt RHT along tokens, delayed per-tensor scale (x2 margin), SR on gradients
  weights: 2D 16x16 blocks; RHT signs + SR seeds from a device RNG refreshed every microbatch
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import emit as E
import attn8 as A8


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
    eps: float = 1e-5
    z_loss: float = 1e-4

    @property
    def hd(self):
        return self.d // self.heads

    @property
    def qkv(self):
        return (self.heads + 2 * self.kv_heads) * self.hd


# column-operand amax slots per layer
SLOTS = ("g2", "a", "dgu", "h2", "g1", "o", "dqkv", "h1")
NS = len(SLOTS)
MARGIN = 2.0


# ------------------------------------------------------------------ fused elementwise (torch.compile)
@torch.compile(dynamic=False)
def qk_prep(qkv, rs, gs, qn, kn, cos, sin, B: int, T: int, H: int, KV: int, hd: int, eps: float):
    """raw qkv [B*T, (H+2KV)*hd] (true = raw*rs*gs) -> q,k,v as [B,H,T,hd] views of [B,T,H,hd] memory."""
    xs = (qkv.float() * (rs[:, None] * gs)).to(torch.bfloat16)
    x = xs.float().view(B, T, H + 2 * KV, hd)
    q, k, v = x.split([H, KV, KV], dim=2)

    def nr(t, w):
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        h = hd // 2
        c, s = cos[:T, None, :].float(), sin[:T, None, :].float()
        t1, t2 = t[..., :h], t[..., h:]
        return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], -1)
    q = nr(q, qn).to(torch.bfloat16)
    k = nr(k, kn).to(torch.bfloat16).repeat_interleave(H // KV, 2)
    v = v.to(torch.bfloat16).repeat_interleave(H // KV, 2)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), xs


def attn_fwd(q, k, v):
    """cuDNN flash attention (causal); returns o [B,H,T,hd] (memory [B,T,H,hd]) and the saved state."""
    r = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, True, False)
    return r[0], r[1:8]


def attn_bwd(do, q, k, v, o, st):
    lse, cq, ck, mq, mk, ps, po = st
    return torch.ops.aten._scaled_dot_product_cudnn_attention_backward(do, q, k, v, o, lse, ps, po, None, cq, ck, mq, mk, 0.0, True)


def qk_prep_ag(qkv_t, qn, kn, cos, sin, B, T, H, KV, hd, eps):
    """same math on the true-scaled qkv, differentiable (used in backward)."""
    x = qkv_t.view(B, T, H + 2 * KV, hd)
    q, k, v = x.split([H, KV, KV], dim=2)

    def nr(t, w):
        tf = t.float()
        tf = tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        h = hd // 2
        c, s = cos[:T, None, :].float(), sin[:T, None, :].float()
        t1, t2 = tf[..., :h], tf[..., h:]
        return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], -1).to(torch.bfloat16)
    q = nr(q, qn)
    k = nr(k, kn).repeat_interleave(H // KV, 2)
    v = v.repeat_interleave(H // KV, 2)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


@torch.compile(dynamic=False)
def qk_bwd(dq4, dk4, dv4, qkv_t, qn, kn, cos, sin, B: int, T: int, H: int, KV: int, hd: int, eps: float):
    """Backward of qk_prep on the true-scaled qkv: d(qkv) [B*T, (H+2KV)*hd] bf16, d(qn), d(kn) fp32.
    dq4/dk4/dv4: grads of the (expanded) [B,H,T,hd] attention inputs."""
    x = qkv_t.float().view(B, T, H + 2 * KV, hd)
    q, k, _ = x.split([H, KV, KV], dim=2)
    r = H // KV
    c, s = cos[:T, None, :].float(), sin[:T, None, :].float()
    h = hd // 2

    def nb(t, w, dout):
        rr = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
        tn = t * rr
        d1, d2 = dout[..., :h], dout[..., h:]
        dy = torch.cat([d1 * c + d2 * s, d2 * c - d1 * s], -1)
        dw = (dy * tn).sum((0, 1, 2))
        dtn = dy * w.float()
        return rr * (dtn - tn * (dtn * tn).mean(-1, keepdim=True)), dw
    dq, dwq = nb(q, qn, dq4.transpose(1, 2).float())
    dkx = dk4.transpose(1, 2).float().view(B, T, KV, r, hd).sum(3)
    dk, dwk = nb(k, kn, dkx)
    dv = dv4.transpose(1, 2).float().view(B, T, KV, r, hd).sum(3)
    d = torch.cat([dq, dk, dv], 2).view(B * T, (H + 2 * KV) * hd).to(torch.bfloat16)
    return d, dwq, dwk


@torch.compile(dynamic=False)
def qk_bwd8(dq, dk, dv, qkv_t, qn, kn, cos, sin, B: int, T: int, H: int, KV: int, hd: int, eps: float):
    """qk_bwd for attn8's grads: dq fp32 [B,H,T,hd], dk/dv bf16 [B,KV,T,hd] (already summed over the group)."""
    x = qkv_t.float().view(B, T, H + 2 * KV, hd)
    q, k, _ = x.split([H, KV, KV], dim=2)
    c, s = cos[:T, None, :].float(), sin[:T, None, :].float()
    h = hd // 2

    def nb(t, w, dout):
        rr = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
        tn = t * rr
        d1, d2 = dout[..., :h], dout[..., h:]
        dy = torch.cat([d1 * c + d2 * s, d2 * c - d1 * s], -1)
        dw = (dy * tn).sum((0, 1, 2))
        dtn = dy * w.float()
        return rr * (dtn - tn * (dtn * tn).mean(-1, keepdim=True)), dw
    dqq, dwq = nb(q, qn, dq.transpose(1, 2))
    dkk, dwk = nb(k, kn, dk.transpose(1, 2).float())
    d = torch.cat([dqq, dkk, dv.transpose(1, 2).float()], 2).view(B * T, (H + 2 * KV) * hd).to(torch.bfloat16)
    return d, dwq, dwk


@torch.compile(dynamic=False)
def scale_rows(raw, rs, gs):
    return (raw.float() * (rs[:, None] * gs)).to(torch.bfloat16)


@torch.compile(dynamic=False)
def swiglu_bwd(da_raw, rs_da, gs_da, gu_raw, rs_gu, gs_gu):
    """d(gate|up) from dA (true = raw*rs*gs) and the raw gate/up GEMM output."""
    da = da_raw.float() * (rs_da[:, None] * gs_da)
    gu = gu_raw.float() * (rs_gu[:, None] * gs_gu)
    g, u = gu.chunk(2, dim=-1)
    sg = torch.sigmoid(g)
    silu = g * sg
    dg = da * u * (sg * (1 + g * (1 - sg)))
    du = da * silu
    return torch.cat([dg, du], -1).to(torch.bfloat16)


@torch.compile(dynamic=False)
def rms_bwd(dh_raw, rs, gs, x, rstd, w, g_res):
    """h = x*rstd*w ; returns g_res + dL/dx (bf16) and dL/dw (fp32 [D])."""
    dh = dh_raw.float() * (rs[:, None] * gs)
    xf = x.float()
    xhat = xf * rstd[:, None]
    dw = (dh * xhat).sum(0)
    dxh = dh * w.float()
    dx = rstd[:, None] * (dxh - xhat * (dxh * xhat).mean(-1, keepdim=True))
    return (g_res.float() + dx).to(torch.bfloat16), dw


@torch.compile(dynamic=False)
def rms_bwd_bf16(dh, x, rstd, w, g_res):
    dh = dh.float()
    xf = x.float()
    xhat = xf * rstd[:, None]
    dw = (dh * xhat).sum(0)
    dxh = dh * w.float()
    dx = rstd[:, None] * (dxh - xhat * (dxh * xhat).mean(-1, keepdim=True))
    return (g_res.float() + dx).to(torch.bfloat16), dw


@torch.compile(dynamic=False)
def ce_chunk(logits, y, m, z: float, inv_n):
    """fused CE (+z-loss) forward and d(logits); logits bf16 [c, V]."""
    lg = logits.float()
    lse = torch.logsumexp(lg, -1, keepdim=True)
    p = torch.exp(lg - lse)
    tgt = lg.gather(1, y[:, None])
    mf = m[:, None] * inv_n
    loss = (((lse - tgt) + z * lse * lse) * mf).sum()
    gl = p * (mf * (1 + 2 * z * lse))
    gl = gl.scatter_add(1, y[:, None], -mf)
    return loss, gl.to(torch.bfloat16)


def rope_cache(cfg, device):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.hd, 2, device=device).float() / cfg.hd))
    f = torch.outer(torch.arange(cfg.max_seq, device=device).float(), inv)
    return torch.cos(f).bfloat16(), torch.sin(f).bfloat16()


# ------------------------------------------------------------------ parameters
class Params:
    """Flat BF16 parameter buffer + views. Order: emb, per layer [wqkv, wo, wgu, wdown, n1, n2, qn, kn], nf."""

    def __init__(self, cfg: Config, dev="cuda"):
        c = cfg
        self.cfg = c
        self.mats = [("wqkv", (c.qkv, c.d)), ("wo", (c.d, c.d)), ("wgu", (2 * c.ff, c.d)), ("wdown", (c.d, c.ff))]
        self.vecs = [("n1", (c.d,)), ("n2", (c.d,)), ("qn", (c.hd,)), ("kn", (c.hd,))]
        n = c.vocab * c.d + c.layers * (sum(a * b for _, (a, b) in self.mats) + sum(a for _, (a,) in self.vecs)) + c.d
        self.flat = torch.zeros(n, dtype=torch.bfloat16, device=dev)
        self.nmat = c.layers * sum(a * b for _, (a, b) in self.mats)
        self.mom = torch.zeros(self.nmat, dtype=torch.bfloat16, device=dev)    # Muon momentum (block matrices)
        o = 0

        def take(shape, buf=None, off=None):
            nonlocal o
            k = math.prod(shape)
            t = self.flat[o:o + k].view(shape)
            o += k
            return t
        self.emb = take((c.vocab, c.d))
        self.L = []
        mo = 0
        for l in range(c.layers):
            d = {}
            for name, shp in self.mats:
                d[name] = take(shp)
                k = math.prod(shp)
                d["m_" + name] = self.mom[mo:mo + k].view(shp)
                mo += k
            for name, shp in self.vecs:
                d[name] = take(shp)
            self.L.append(d)
        self.nf = take((c.d,))
        assert o == n
        self.n = n

    def small(self):
        """(name, tensor) of the non-matrix params (norm gains)"""
        out = [("nf", self.nf)]
        for l, d in enumerate(self.L):
            out += [(f"{k}{l}", d[k]) for k, _ in self.vecs]
        return out

    @torch.no_grad()
    def init(self, seed=0):
        c = self.cfg
        g = torch.Generator(device=self.flat.device).manual_seed(seed)
        self.emb.copy_(torch.randn(self.emb.shape, generator=g, device=self.flat.device) * 0.02)
        out_std = 0.02 / math.sqrt(2 * c.layers)
        for d in self.L:
            for name, shp in self.mats:
                std = out_std if name in ("wo", "wdown") else 0.02
                d[name].copy_(torch.randn(shp, generator=g, device=self.flat.device).clamp_(-3, 3) * std)
            for name, _ in self.vecs:
                d[name].fill_(1.0)
        self.nf.fill_(1.0)


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, cfg: Config, B: int, T: int, dev="cuda"):
        self.cfg, self.B, self.T, self.dev = cfg, B, T, dev
        c = cfg
        self.M = M = B * T
        self.P = Params(cfg, dev)
        self.cos, self.sin = rope_cache(cfg, dev)
        self.fp4 = True
        # fp32 grads for the tied embedding and small params
        self.g_emb = torch.zeros(c.vocab, c.d, dtype=torch.float32, device=dev)
        self.g_small = {k: torch.zeros(v.shape, dtype=torch.float32, device=dev) for k, v in self.P.small()}
        # FP4 weight caches
        self.WQ = [{k: E.WQ(*shp, dev) for k, shp in self.P.mats} for _ in range(c.layers)]
        # delayed-scale state for column operands
        self.amax_use = torch.ones(c.layers, NS, device=dev)
        self.amax_cur = torch.zeros(c.layers, NS, dtype=torch.int32, device=dev)
        self.rng = torch.zeros(2, dtype=torch.int32, device=dev)
        self.acc_scale = 1.0          # 1 / microbatches per step (set by trainer)
        # static activations / checkpoints
        self.X = torch.empty(c.layers + 1, M, c.d, dtype=torch.bfloat16, device=dev)
        self.one = torch.ones(1, device=dev)
        self.ones_T = torch.ones(M, device=dev)
        self.loss = torch.zeros((), device=dev)
        self.saved = [None] * c.layers   # per layer: qkv (true scale), attention out + cuDNN state, x1
        self.attn = "fp8"                # "fp8": attn8.cu FP8 flash attention ; "cudnn": BF16 cuDNN
        self.ds_use = torch.ones(c.layers, device=dev)                     # delayed dS amax (fp8 attention bwd)
        self.ds_cur = torch.zeros(c.layers, dtype=torch.int32, device=dev)

    # ---------------- per-step weight quantization
    @torch.no_grad()
    def refresh_weights(self):
        if not self.fp4:
            return
        for d, wq in zip(self.P.L, self.WQ):
            for k in wq:
                wq[k].refresh(d[k])

    def slot(self, l, name):
        i = SLOTS.index(name)
        return self.amax_use[l, i:i + 1], self.amax_cur[l, i:i + 1]

    def alpha(self, l, a, b):
        ia, ib = SLOTS.index(a), SLOTS.index(b)
        return self.amax_use[l, ia:ia + 1] * self.amax_use[l, ib:ib + 1] * (1.0 / (2688.0 * 2688.0))

    # ---------------- forward of one block from its (already normed+quantized) input
    def _attn(self, l, hq, h1, keep):
        c, P, W = self.cfg, self.P.L[l], self.WQ[l]
        B, T, H, KV, hd = self.B, self.T, c.heads, c.kv_heads, c.hd
        if self.fp4:
            qkv = E.mm_row_w(hq, W["wqkv"])
            rs, gs = hq.rs, W["wqkv"].gs
        else:
            qkv = h1 @ P["wqkv"].t()
            rs, gs = self.ones_T, self.one
        if self.attn == "fp8":
            qkv_t = scale_rows(qkv, rs, gs)
            Q8, K8, VT8, sv = A8.prep(qkv_t, P["qn"], P["kn"], self.cos, self.sin, B, T, H, KV, c.eps)
            O, L2 = A8.fwd(Q8, K8, VT8, sv, B, T, H, KV)          # O [B,T,H,hd] == token-major [M, D]
            return qkv_t, O.view(B, T, H, hd).transpose(1, 2), ("fp8", L2, Q8, K8)
        q, k, v, qkv_t = qk_prep(qkv, rs, gs, P["qn"], P["kn"], self.cos, self.sin, B, T, H, KV, hd, c.eps)
        o4, st = attn_fwd(q, k, v)
        return qkv_t, o4, st

    # ---------------- forward + fused loss (no autograd)
    @torch.no_grad()
    def forward_loss(self, idx, tgt, mask, chunk=512, grad=True):
        """Runs the model, the CE loss and the LM-head backward. Returns (loss, dL/dX[L] bf16).
        grad=False: evaluation only (no gradient side effects), returns (loss, None)."""
        c, M, P, X, dev = self.cfg, self.M, self.P, self.X, self.dev
        X[0].copy_(F.embedding(idx.view(-1), P.emb))
        src, y, ys, ygw = X[0], None, None, None
        for l in range(c.layers):
            Pl, W = P.L[l], self.WQ[l]
            xl = X[l]
            if self.fp4:
                hq = E.RowQ(M, c.d, dev)
                E.rms_emit(src, Pl["n1"], hq, y=y, ys=ys, ygw=ygw, xout=xl if y is not None else None)
                qkv_t, o4, st = self._attn(l, hq, None, False)
                o = o4.transpose(1, 2).reshape(M, c.d)
                oq = E.RowQ(M, c.d, dev)
                E.row_emit(o, oq, self.rng)
                yo = E.mm_row_w(oq, W["wo"])
                x1 = torch.empty(M, c.d, dtype=torch.bfloat16, device=dev)
                h2q = E.RowQ(M, c.d, dev)
                E.rms_emit(xl, Pl["n2"], h2q, y=yo, ys=oq.rs, ygw=W["wo"].gs, xout=x1)
                gu = E.mm_row_w(h2q, W["wgu"])
                aq = E.RowQ(M, c.ff, dev)
                E.swiglu_emit(gu, h2q.rs, W["wgu"].gs, aq)
                y, ys, ygw = E.mm_row_w(aq, W["wdown"]), aq.rs, W["wdown"].gs
            else:
                h1 = torch.empty(M, c.d, dtype=torch.bfloat16, device=dev)
                E.rms_emit(src, Pl["n1"], None, h=h1, y=y, ys=ys, ygw=ygw, xout=xl if y is not None else None)
                qkv_t, o4, st = self._attn(l, None, h1, False)
                o = o4.transpose(1, 2).reshape(M, c.d)
                yo = o @ Pl["wo"].t()
                x1 = torch.empty(M, c.d, dtype=torch.bfloat16, device=dev)
                h2 = torch.empty_like(x1)
                E.rms_emit(xl, Pl["n2"], None, h=h2, y=yo, ys=self.ones_T, ygw=self.one, xout=x1)
                gu = h2 @ Pl["wgu"].t()
                a = torch.empty(M, c.ff, dtype=torch.bfloat16, device=dev)
                E.swiglu_emit(gu, self.ones_T, self.one, None, a)
                y, ys, ygw = a @ Pl["wdown"].t(), self.ones_T, self.one
            self.saved[l] = (qkv_t, o4, st, x1)
            src = x1
        hf = torch.empty(M, c.d, dtype=torch.bfloat16, device=dev)
        rf = torch.empty(M, dtype=torch.float32, device=dev)
        E.rms_emit(src, P.nf, None, h=hf, rstd=rf, y=y, ys=ys, ygw=ygw, xout=X[c.layers])
        # ---- tied LM head + CE: forward and backward in one pass (as nvfp4's lm_tail)
        tg, mk = tgt.view(-1), mask.view(-1).float()
        inv_n = 1.0 / mk.sum().clamp_min(1.0)
        gh = torch.empty(M, c.d, dtype=torch.bfloat16, device=dev)
        loss = torch.zeros((), device=dev)
        for i in range(0, M, chunk):
            hc = hf[i:i + chunk]
            l_, gl = ce_chunk(hc @ P.emb.t(), tg[i:i + chunk], mk[i:i + chunk], 0.0 if not grad else c.z_loss, inv_n)
            loss += l_
            if not grad:
                continue
            torch.mm(gl, P.emb, out=gh[i:i + chunk])
            torch.addmm(self.g_emb, gl.t(), hc, out_dtype=torch.float32, out=self.g_emb)
        if not grad:
            self.saved = [None] * c.layers
            return loss, None
        g, dnf = rms_bwd_bf16(gh, X[c.layers], rf, P.nf, torch.zeros_like(gh))
        self.g_small["nf"] += dnf
        return loss, g

    # ---------------- backward of one block (recompute + manual grads)
    def block_bwd(self, l, g):
        c, M, dev, P = self.cfg, self.M, self.dev, self.P
        Pl, W = P.L[l], self.WQ[l]
        B, T, H, KV, hd = self.B, self.T, c.heads, c.kv_heads, c.hd
        x = self.X[l]
        s = self.acc_scale
        salt = l * 64
        fp4 = self.fp4
        # ----- recompute (attention and its projection are saved from the forward)
        qkv_t, o4, st, x1 = self.saved[l]
        o = o4.transpose(1, 2).reshape(M, c.d)
        with torch.no_grad():
            h1 = torch.empty(M, c.d, dtype=torch.bfloat16, device=dev)
            r1 = torch.empty(M, dtype=torch.float32, device=dev)
            E.rms_emit(x, Pl["n1"], None, h=h1, rstd=r1)
            h2 = torch.empty_like(h1)
            r2 = torch.empty(M, dtype=torch.float32, device=dev)
            a = torch.empty(M, c.ff, dtype=torch.bfloat16, device=dev)
            if fp4:
                h2q = E.RowQ(M, c.d, dev)
                E.rms_emit(x1, Pl["n2"], h2q, h=h2, rstd=r2)
                gu = E.mm_row_w(h2q, W["wgu"])
                E.swiglu_emit(gu, h2q.rs, W["wgu"].gs, None, a)
            else:
                E.rms_emit(x1, Pl["n2"], None, h=h2, rstd=r2)
                gu = h2 @ Pl["wgu"].t()
                E.swiglu_emit(gu, self.ones_T, self.one, None, a)

            # ----- MLP backward
            if fp4:
                g2r = E.RowQ(M, c.d, dev); E.row_emit(g, g2r, self.rng, sr=True, salt=salt + 1)
                g2c = E.ColQ(M, c.d, dev); E.col_emit(g, g2c, *self.slot(l, "g2"), self.rng, sr=True, salt=salt + 2)
                ac = E.ColQ(M, c.ff, dev); E.col_emit(a, ac, *self.slot(l, "a"), self.rng)
                E.mom_acc(Pl["m_wdown"], E.mm_col(g2c, ac), self.alpha(l, "g2", "a"), s, self.rng, salt + 3)
                W["wdown"].prep()      # the only weight not re-quantized by the recompute
                dgu = swiglu_bwd(E.mm_row_wt(g2r, W["wdown"]), g2r.rs, W["wdown"].gs, gu, h2q.rs, W["wgu"].gs)
                dgur = E.RowQ(M, 2 * c.ff, dev); E.row_emit(dgu, dgur, self.rng, sr=True, salt=salt + 4)
                dguc = E.ColQ(M, 2 * c.ff, dev); E.col_emit(dgu, dguc, *self.slot(l, "dgu"), self.rng, sr=True, salt=salt + 5)
                h2c = E.ColQ(M, c.d, dev); E.col_emit(h2, h2c, *self.slot(l, "h2"), self.rng)
                E.mom_acc(Pl["m_wgu"], E.mm_col(dguc, h2c), self.alpha(l, "dgu", "h2"), s, self.rng, salt + 6)
                g1, dn2 = rms_bwd(E.mm_row_wt(dgur, W["wgu"]), dgur.rs, W["wgu"].gs, x1, r2, Pl["n2"], g)
            else:
                E.mom_acc(Pl["m_wdown"], g.t() @ a, self.one, s, self.rng, salt + 3)
                dgu = swiglu_bwd(g @ Pl["wdown"], self.ones_T, self.one, gu, self.ones_T, self.one)
                E.mom_acc(Pl["m_wgu"], dgu.t() @ h2, self.one, s, self.rng, salt + 6)
                g1, dn2 = rms_bwd_bf16(dgu @ Pl["wgu"], x1, r2, Pl["n2"], g)
            self.g_small[f"n2{l}"] += dn2

            # ----- attention backward
            if fp4:
                g1r = E.RowQ(M, c.d, dev); E.row_emit(g1, g1r, self.rng, sr=True, salt=salt + 7)
                g1c = E.ColQ(M, c.d, dev); E.col_emit(g1, g1c, *self.slot(l, "g1"), self.rng, sr=True, salt=salt + 8)
                oc = E.ColQ(M, c.d, dev); E.col_emit(o, oc, *self.slot(l, "o"), self.rng)
                E.mom_acc(Pl["m_wo"], E.mm_col(g1c, oc), self.alpha(l, "g1", "o"), s, self.rng, salt + 9)
                W["wo"].prep()
                do = scale_rows(E.mm_row_wt(g1r, W["wo"]), g1r.rs, W["wo"].gs)
            else:
                E.mom_acc(Pl["m_wo"], g1.t() @ o, self.one, s, self.rng, salt + 9)
                do = g1 @ Pl["wo"]
        with torch.no_grad():
            if isinstance(st, tuple) and len(st) == 4 and st[0] == "fp8":
                _, L2, Q8, K8 = st
                dq, dk, dv = A8.bwd(do.view(B, T, H, hd), o.view(B, T, H, hd), L2, Q8, K8, qkv_t,
                                    self.ds_use[l:l + 1], self.ds_cur[l:l + 1], B, T, H, KV)
                dqkv, dqn, dkn = qk_bwd8(dq, dk, dv, qkv_t, Pl["qn"], Pl["kn"], self.cos, self.sin, B, T, H, KV, hd, c.eps)
            else:
                q, k, v, _ = qk_prep(qkv_t, self.ones_T, self.one, Pl["qn"], Pl["kn"], self.cos, self.sin, B, T, H, KV, hd, c.eps)
                dq4, dk4, dv4 = attn_bwd(do.view(B, T, H, hd).transpose(1, 2), q, k, v, o4, st)
                dqkv, dqn, dkn = qk_bwd(dq4, dk4, dv4, qkv_t, Pl["qn"], Pl["kn"], self.cos, self.sin, B, T, H, KV, hd, c.eps)
            self.g_small[f"qn{l}"] += dqn
            self.g_small[f"kn{l}"] += dkn
            if fp4:
                dqr = E.RowQ(M, c.qkv, dev); E.row_emit(dqkv, dqr, self.rng, sr=True, salt=salt + 10)
                dqc = E.ColQ(M, c.qkv, dev); E.col_emit(dqkv, dqc, *self.slot(l, "dqkv"), self.rng, sr=True, salt=salt + 11)
                h1c = E.ColQ(M, c.d, dev); E.col_emit(h1, h1c, *self.slot(l, "h1"), self.rng)
                E.mom_acc(Pl["m_wqkv"], E.mm_col(dqc, h1c), self.alpha(l, "dqkv", "h1"), s, self.rng, salt + 12)
                W["wqkv"].prep()
                g0, dn1 = rms_bwd(E.mm_row_wt(dqr, W["wqkv"]), dqr.rs, W["wqkv"].gs, x, r1, Pl["n1"], g1)
            else:
                E.mom_acc(Pl["m_wqkv"], dqkv.t() @ h1, self.one, s, self.rng, salt + 12)
                g0, dn1 = rms_bwd_bf16(dqkv @ Pl["wqkv"], x, r1, Pl["n1"], g1)
            self.g_small[f"n1{l}"] += dn1
        self.saved[l] = None
        return g0

    def microbatch(self, idx, tgt, mask):
        """one fwd+bwd; wgrads go into the momentum, embedding/small grads into fp32 buffers. Graph-capturable."""
        with torch.no_grad():
            self.rng.copy_(torch.randint(-2**31, 2**31 - 1, (2,), device=self.dev, dtype=torch.int32))
        loss, g = self.forward_loss(idx, tgt, mask)
        for l in reversed(range(self.cfg.layers)):
            g = self.block_bwd(l, g)
        with torch.no_grad():
            self.g_emb.index_add_(0, idx.view(-1), g.float())
            # delayed scaling: this microbatch's observed amax (x margin) scales the next one
            torch.mul(self.amax_cur.view(torch.float32).clamp(min=1e-30), MARGIN, out=self.amax_use)
            self.amax_cur.zero_()
            if self.attn == "fp8":
                torch.mul(self.ds_cur.view(torch.float32).clamp(min=1e-30), MARGIN, out=self.ds_use)
                self.ds_cur.zero_()
            self.loss.copy_(loss)
        return self.loss

