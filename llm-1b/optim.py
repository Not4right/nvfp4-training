"""Optimizer step, entirely on the GPU.

Block matrices: Muon (Newton-Schulz orthogonalized momentum), update RMS matched to AdamW (Moonlight,
"Muon is Scalable for LLM Training", 2025): W <- W(1 - lr*wd) - lr * 0.2*sqrt(max(A,B)) * NS(M).
The momentum already holds  beta*M_prev + mean_microbatch(grad)  (wgrads are accumulated into it by
mom_acc), so after the update it is multiplied by beta for the next step.  q/k/v and gate/up are
orthogonalized as separate matrices.  BF16 weights are written with stochastic rounding.

Tied embedding + norm gains: AdamW (embedding states BF16, norm gains FP32 master + FP32 states),
global-norm clip on their gradient.
"""
import math
import torch

NS_COEF = (3.4445, -4.7750, 2.0315)


@torch.compile(dynamic=False)
def _ns5(G):
    a, b, c = NS_COEF
    X = G.to(torch.bfloat16)
    X = X / (X.float().norm() + 1e-7).to(torch.bfloat16)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.t()
    for _ in range(5):
        A = X @ X.t()
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    return X.t() if tr else X


@torch.compile(dynamic=False)
def _sr_update(w, O, lr, wd, scale):
    upd = w.float() * (1 - lr * wd) - (lr * scale) * O.float()
    noise = torch.randint_like(upd, 0, 65536, dtype=torch.int32)
    bits = (upd.view(torch.int32) + noise) & -65536
    return bits.view(torch.float32).to(torch.bfloat16)


@torch.compile(dynamic=False)
def _adam_bf16(w, g, m, v, lr, wd, b1: float, b2: float, bc1, bc2, gscale):
    g = g * gscale
    m32 = m.float() * b1 + (1 - b1) * g
    v32 = v.float() * b2 + (1 - b2) * g * g
    upd = w.float() * (1 - lr * wd) - lr * (m32 / bc1) / (torch.sqrt(v32 / bc2) + 1e-8)
    noise = torch.randint_like(upd, 0, 65536, dtype=torch.int32)
    wn = ((upd.view(torch.int32) + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    return wn, m32.to(torch.bfloat16), v32.to(torch.bfloat16)


def splits(name, c):
    """row ranges orthogonalized separately"""
    if name == "wqkv":
        H, KV, hd = c.heads, c.kv_heads, c.hd
        return [(0, H * hd), (H * hd, (H + KV) * hd), ((H + KV) * hd, (H + 2 * KV) * hd)]
    if name == "wgu":
        return [(0, c.ff), (c.ff, 2 * c.ff)]
    return None


class Optim:
    def __init__(self, model, b1=0.9, b2=0.95, beta_muon=0.95, wd=0.1, clip=1.0):
        self.m = model
        self.b1, self.b2, self.beta, self.wd, self.clip = b1, b2, beta_muon, wd, clip
        P, dev = model.P, model.dev
        self.emb_m = torch.zeros_like(P.emb)
        self.emb_v = torch.zeros_like(P.emb)
        self.small = {k: v.detach().float().clone() for k, v in P.small()}          # fp32 masters
        self.small_m = {k: torch.zeros_like(v) for k, v in self.small.items()}
        self.small_v = {k: torch.zeros_like(v) for k, v in self.small.items()}
        self.t = 0
        self.lr_t = torch.zeros((), device=dev)
        self.last_gnorm = 0.0

    @torch.no_grad()
    def step(self, lr, n_micro):
        m, P, c = self.m, self.m.P, self.m.cfg
        self.t += 1
        self.lr_t.fill_(lr)
        lr_t, wd_t = self.lr_t, torch.tensor(self.wd, device=m.dev)
        # ---- Muon on block matrices
        for d in P.L:
            for name, (A, B) in P.mats:
                w, mo = d[name], d["m_" + name]
                sp = splits(name, c) or [(0, A)]
                for r0, r1 in sp:
                    O = _ns5(mo[r0:r1])
                    scale = 0.2 * math.sqrt(max(r1 - r0, B))
                    w[r0:r1].copy_(_sr_update(w[r0:r1], O, lr_t, wd_t, torch.tensor(scale, device=m.dev)))
            # (all matrices of this layer done) -> decay momentum for the next step's accumulation
        P.mom.mul_(self.beta)
        # ---- AdamW on embedding + norm gains (grads are sums over microbatches of per-microbatch means)
        inv = 1.0 / n_micro
        sq = (m.g_emb.float().pow(2).sum() + sum(g.pow(2).sum() for g in m.g_small.values())) * (inv * inv)
        gnorm = sq.sqrt()
        gscale = torch.clamp(self.clip / (gnorm + 1e-6), max=1.0) * inv
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        wn, mn, vn = _adam_bf16(P.emb, m.g_emb, self.emb_m, self.emb_v, lr_t, wd_t, self.b1, self.b2,
                                torch.tensor(bc1, device=m.dev), torch.tensor(bc2, device=m.dev), gscale)
        P.emb.copy_(wn); self.emb_m.copy_(mn); self.emb_v.copy_(vn)
        for (k, p) in P.small():
            g = m.g_small[k] * gscale
            mm, vv, w = self.small_m[k], self.small_v[k], self.small[k]
            mm.mul_(self.b1).add_(g, alpha=1 - self.b1)
            vv.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            w.sub_(lr_t * (mm / bc1) / ((vv / bc2).sqrt() + 1e-8))
            p.copy_(w)
        m.g_emb.zero_()
        for g in m.g_small.values():
            g.zero_()
        m.refresh_weights()
        self.last_gnorm = gnorm
        return gnorm

    def state_dict(self):
        return {"emb_m": self.emb_m, "emb_v": self.emb_v, "small": self.small, "small_m": self.small_m,
                "small_v": self.small_v, "t": self.t}

    def load_state_dict(self, sd):
        self.emb_m.copy_(sd["emb_m"]); self.emb_v.copy_(sd["emb_v"])
        for k in self.small:
            self.small[k].copy_(sd["small"][k]); self.small_m[k].copy_(sd["small_m"][k]); self.small_v[k].copy_(sd["small_v"][k])
        self.t = sd["t"]
