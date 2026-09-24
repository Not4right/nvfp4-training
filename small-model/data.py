"""Arithmetic equation dataset, char level.

Each 32-token row packs two 16-token equations:  "1234+567=" + reversed answer + "\n", pad-filled.
Answers are written least-significant digit first (the standard trick that makes
carries learnable for small transformers). Loss is only on answer tokens + EOS,
so the numbers we compare are exactly "how well can it do arithmetic".
Ops: + and - on 0..9999, * on 0..999 x 0..99.
"""
import os
import numpy as np
import torch

VOCAB = "0123456789+-*=\n_"  # '_' = pad
V = len(VOCAB)
PAD, EOS, EQ = VOCAB.index("_"), VOCAB.index("\n"), VOCAB.index("=")
T = 32
SEG = 16


def _gen(n, rng):
    op = rng.integers(0, 3, n)
    a = np.where(op == 2, rng.integers(0, 1000, n), rng.integers(0, 10000, n))
    b = np.where(op == 2, rng.integers(0, 100, n), rng.integers(0, 10000, n))
    c = np.where(op == 0, a + b, np.where(op == 1, a - b, a * b))
    seqs = np.full((n, SEG), PAD, np.uint8)
    mask = np.zeros((n, SEG), np.uint8)
    opch = np.array([10, 11, 12])
    for i in range(n):  # 2M rows takes ~15 s once, then cached
        s = [int(ch) for ch in str(a[i])] + [opch[op[i]]] + [int(ch) for ch in str(b[i])] + [EQ]
        ans = str(abs(c[i]))[::-1]
        r = [int(ch) for ch in ans] + ([11] if c[i] < 0 else []) + [EOS]
        seqs[i, :len(s)] = s
        seqs[i, len(s):len(s) + len(r)] = r
        mask[i, len(s):len(s) + len(r)] = 1
    return seqs, mask


def build(n_rows, seed):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f".data_{n_rows}_{seed}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["x"], z["m"]
    rng = np.random.default_rng(seed)
    s, m = _gen(2 * n_rows, rng)
    x = s.reshape(n_rows, T)
    m = m.reshape(n_rows, T)
    np.savez(path, x=x, m=m)
    return x, m


class Data:
    """Inputs are x[:, :-1]-style shifted on the fly: token t predicts token t+1.
    We keep full T=32 inputs and predict row[1:] + PAD, masking the final position."""

    def __init__(self, n_train=1 << 21, n_eval=8192, device="cuda"):
        x, m = build(n_train, 1234)
        xe, me = build(n_eval, 999)
        self.x = torch.from_numpy(x).to(device)
        self.m = torch.from_numpy(m).to(device)
        self.xe = torch.from_numpy(xe).to(device)
        self.me = torch.from_numpy(me).to(device)

    @staticmethod
    def targets(x, m):
        y = torch.cat([x[:, 1:], torch.full_like(x[:, :1], PAD)], 1).long()
        w = torch.cat([m[:, 1:], torch.zeros_like(m[:, :1])], 1).float()
        return y, w

    def batch(self, step, bs):
        n = self.x.shape[0]
        i = (step * bs) % n
        x, m = self.x[i:i + bs], self.m[i:i + bs]
        y, w = self.targets(x, m)
        return x.long(), y, w

    def eval_set(self):
        y, w = self.targets(self.xe, self.me)
        return self.xe.long(), y, w


def decode(row):
    return "".join(VOCAB[int(t)] for t in row)


if __name__ == "__main__":
    d = Data()
    x, y, w = d.batch(0, 4)
    for r in range(4):
        print(repr(decode(x[r])), (w[r] > 0).int().tolist())
    print("train rows", d.x.shape, "answer tokens/row", d.m.float().sum(1).mean().item())
