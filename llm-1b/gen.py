"""Sample from a checkpoint (plain PyTorch, fp32, KV cache). Runs on CPU so it works while training
holds the GPU.  Same math as model2 in BF16 mode: pre-RMSNorm, GQA, QK-norm, NeoX RoPE, SwiGLU, tied head.

  python gen.py "The capital of France is" [--ckpt ckpt/latest.pt] [--n 120] [--temp 0.8] [--chat]
"""
import argparse
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def load(path, device="cpu"):
    import model2 as M2
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M2.Config()
    flat = sd["flat"].float()
    c = cfg
    o = 0

    def take(*shape):
        nonlocal o
        k = math.prod(shape)
        t = flat[o:o + k].view(*shape)
        o += k
        return t.to(device)
    W = {"emb": take(c.vocab, c.d), "L": []}
    for _ in range(c.layers):
        d = {"wqkv": take(c.qkv, c.d), "wo": take(c.d, c.d), "wgu": take(2 * c.ff, c.d), "wdown": take(c.d, c.ff),
             "n1": take(c.d), "n2": take(c.d), "qn": take(c.hd), "kn": take(c.hd)}
        W["L"].append(d)
    W["nf"] = take(c.d)
    assert o == flat.numel()
    return cfg, W, sd


def rms(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class Gen:
    def __init__(self, cfg, W):
        self.c, self.W = cfg, W
        c = cfg
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.hd, 2).float() / c.hd))
        f = torch.outer(torch.arange(c.max_seq).float(), inv)
        self.cos, self.sin = torch.cos(f), torch.sin(f)

    def rope(self, x, pos):
        h = x.shape[-1] // 2
        c, s = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    @torch.no_grad()
    def step(self, ids, cache):
        """ids [n] new tokens; cache: list of (k, v) per layer -> logits of the last token"""
        c, W = self.c, self.W
        H, KV, hd = c.heads, c.kv_heads, c.hd
        start = cache[0][0].shape[0] if cache[0] is not None else 0
        pos = torch.arange(start, start + len(ids))
        x = W["emb"][ids]
        for l, d in enumerate(W["L"]):
            h = rms(x, d["n1"])
            qkv = (h @ d["wqkv"].t()).view(len(ids), H + 2 * KV, hd)
            q, k, v = qkv.split([H, KV, KV], 1)
            q = self.rope(rms(q, d["qn"]), pos)
            k = self.rope(rms(k, d["kn"]), pos)
            if cache[l] is not None:
                k = torch.cat([cache[l][0], k]); v = torch.cat([cache[l][1], v])
            cache[l] = (k, v)
            kk = k.repeat_interleave(H // KV, 1).transpose(0, 1)      # [H, S, hd]
            vv = v.repeat_interleave(H // KV, 1).transpose(0, 1)
            qq = q.transpose(0, 1)                                     # [H, n, hd]
            s = qq @ kk.transpose(1, 2) / math.sqrt(hd)
            S = kk.shape[1]
            mask = torch.arange(S)[None, :] > (start + torch.arange(len(ids)))[:, None]
            s = s.masked_fill(mask, float("-inf"))
            o = (s.softmax(-1) @ vv).transpose(0, 1).reshape(len(ids), H * hd)
            x = x + o @ d["wo"].t()
            g, u = (rms(x, d["n2"]) @ d["wgu"].t()).chunk(2, -1)
            x = x + (torch.nn.functional.silu(g) * u) @ d["wdown"].t()
        return rms(x[-1], W["nf"]) @ W["emb"].t()

    def generate(self, ids, n, temp=0.8, top_p=0.9, rep=1.1, stop=None):
        cache = [None] * self.c.layers
        out = list(ids)
        logits = self.step(torch.tensor(ids), cache)
        for _ in range(n):
            lg = logits.clone()
            for t in set(out[-64:]):
                lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
            if temp <= 0:
                nxt = int(lg.argmax())
            else:
                p = torch.softmax(lg / temp, -1)
                sp, si = p.sort(descending=True)
                keep = sp.cumsum(0) - sp < top_p
                sp = sp * keep
                nxt = int(si[torch.multinomial(sp / sp.sum(), 1)])
            out.append(nxt)
            if stop is not None and nxt in stop:
                break
            logits = self.step(torch.tensor([nxt]), cache)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt", "latest.pt"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tok = Tokenizer.from_file(hf_hub_download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json"))
    t0 = time.time()
    cfg, W, sd = load(a.ckpt)
    print(f"[{os.path.basename(a.ckpt)}: phase {sd['phase']} step {sd['step']}, {sd['tokens']/1e9:.3f}B tokens; loaded in {time.time()-t0:.0f}s]\n")
    g = Gen(cfg, W)
    for pr in a.prompts or ["The history of the Roman Empire"]:
        text = f"<|im_start|>user\n{pr}<|im_end|>\n<|im_start|>assistant\n" if a.chat else pr
        ids = tok.encode(text).ids
        t0 = time.time()
        out = g.generate(ids, a.n, a.temp, stop={0, 2})
        dt = time.time() - t0
        print(tok.decode(out, skip_special_tokens=False))
        print(f"   [{len(out)-len(ids)} tokens, {(len(out)-len(ids))/dt:.1f} tok/s]\n" + "-" * 70)


if __name__ == "__main__":
    main()
