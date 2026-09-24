"""Mixture data loader over the uint16 token shards written by prep_data.py.

Each source is one long token stream (its shards concatenated in order).  A sequence of T+1 tokens is
read from a sampled source at that source's cursor; cursors only move forward, so no token is reused
until a source is exhausted (then it wraps and logs it).  The first VAL tokens of every source are
held out for validation.  Sources that have no data on disk yet (still downloading) are skipped and
the mixture is renormalized over the ones available.  State (cursors + RNG) goes into the checkpoint.
"""
import glob
import os
import threading
import queue

import numpy as np

VAL = 1_000_000

STABLE = {"fwedu": .45, "ufw": .20, "dclm": .10, "fmath": .10, "iwm": .04, "cosmo": .11}
ANNEAL = {"fwedu": .25, "ufw": .08, "fmath": .12, "iwm": .05, "cosmo": .08,
          "think": .15, "omi2": .15, "numina": .07, "magpie": .05}


class Source:
    def __init__(self, root, name):
        self.dir = os.path.join(root, name)
        self.name = name
        self.files, self.maps, self.starts, self.total = [], [], [0], 0
        self.rescan()

    def rescan(self):
        fs = sorted(f for f in glob.glob(os.path.join(self.dir, "*.bin")) if not f.endswith(".mask.bin"))
        for f in fs[len(self.files):]:
            mm = np.memmap(f, dtype=np.uint16, mode="r")
            if len(mm) == 0:
                continue
            self.files.append(f)
            self.maps.append(mm)
            self.total += len(mm)
            self.starts.append(self.total)

    def read(self, pos, n):
        out = np.empty(n, dtype=np.uint16)
        k = 0
        i = int(np.searchsorted(self.starts, pos, side="right")) - 1
        while k < n:
            mm, s0 = self.maps[i], self.starts[i]
            off = pos + k - s0
            take = min(n - k, len(mm) - off)
            out[k:k + take] = mm[off:off + take]
            k += take
            i += 1
        return out


class Mixture:
    def __init__(self, root, T, seed=0):
        self.root, self.T = root, T
        self.src = {}
        self.pos = {}
        self.wraps = {}
        self.rng = np.random.default_rng(seed)
        self.weights = STABLE
        self.rescan()

    def rescan(self):
        for d in sorted(os.listdir(self.root)):
            if d.startswith("_") or d.startswith("sft") or not os.path.isdir(os.path.join(self.root, d)):
                continue
            if d not in self.src:
                self.src[d] = Source(self.root, d)
                self.pos.setdefault(d, VAL)
                self.wraps.setdefault(d, 0)
            else:
                self.src[d].rescan()

    def available(self, name):
        s = self.src.get(name)
        return s is not None and s.total >= VAL + 4 * (self.T + 1)

    def next_seq(self):
        names = [n for n in self.weights if self.available(n)]
        while not names:                      # still downloading: wait for the first shard
            import time
            time.sleep(30)
            self.rescan()
            names = [n for n in self.weights if self.available(n)]
        w = np.array([self.weights[n] for n in names], dtype=np.float64)
        n = names[self.rng.choice(len(names), p=w / w.sum())]
        s = self.src[n]
        if self.pos[n] + self.T + 1 > s.total:
            s.rescan()
            if self.pos[n] + self.T + 1 > s.total:     # exhausted: wrap to just after the held-out part
                self.pos[n] = VAL
                self.wraps[n] += 1
        seq = s.read(self.pos[n], self.T + 1)
        self.pos[n] += self.T + 1
        return seq

    def batch(self, B):
        self._nb = getattr(self, "_nb", 0) + 1
        if self._nb % 500 == 0:
            self.rescan()
        a = np.stack([self.next_seq() for _ in range(B)]).astype(np.int64)
        return a[:, :-1], a[:, 1:]

    def val_batches(self, names, B, nb):
        """fixed held-out batches from the first VAL tokens of each source"""
        out = {}
        for n in names:
            if not self.available(n):
                continue
            s, seqs = self.src[n], []
            for i in range(nb * B):
                seqs.append(s.read(i * (self.T + 1), self.T + 1))
            a = np.stack(seqs).astype(np.int64)
            out[n] = [(a[j * B:(j + 1) * B, :-1], a[j * B:(j + 1) * B, 1:]) for j in range(nb)]
        return out

    def state_dict(self):
        return {"pos": dict(self.pos), "wraps": dict(self.wraps), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, sd):
        self.pos.update(sd["pos"]); self.wraps.update(sd["wraps"])
        self.rng.bit_generator.state = sd["rng"]


class SFTData:
    """packed chat data with loss masks (data/sft*): sequences of T tokens, docs never split across rows."""

    def __init__(self, root, T, seed=0):
        self.T = T
        toks, masks = [], []
        for name in ("sft", "sftmath"):
            for f in sorted(glob.glob(os.path.join(root, name, "*.bin"))):
                if f.endswith(".mask.bin"):
                    continue
                toks.append(np.fromfile(f, dtype=np.uint16))
                masks.append(np.fromfile(f[:-4] + ".mask.bin", dtype=np.uint8))
        t = np.concatenate(toks); m = np.concatenate(masks)
        # split into documents at EOS (token 0), pack greedily into rows of T+1
        ends = np.flatnonzero(t == 0) + 1
        starts = np.concatenate([[0], ends[:-1]])
        order = np.random.default_rng(seed).permutation(len(starts))
        rows_t, rows_m, cur_t, cur_m = [], [], [], []
        for i in order:
            a, b = starts[i], ends[i]
            if b - a > T + 1:
                continue
            if sum(len(x) for x in cur_t) + (b - a) > T + 1:
                rows_t.append(np.concatenate(cur_t)); rows_m.append(np.concatenate(cur_m)); cur_t, cur_m = [], []
            cur_t.append(t[a:b]); cur_m.append(m[a:b])
        if cur_t:
            rows_t.append(np.concatenate(cur_t)); rows_m.append(np.concatenate(cur_m))
        R = len(rows_t)
        self.tok = np.zeros((R, T + 1), dtype=np.int64)
        self.mask = np.zeros((R, T + 1), dtype=np.float32)
        for i in range(R):
            n = len(rows_t[i]); self.tok[i, :n] = rows_t[i]; self.mask[i, :n] = rows_m[i]
        self.i = 0
        self.ntok = int(self.mask.sum())

    def batch(self, B):
        idx = [(self.i + j) % len(self.tok) for j in range(B)]
        self.i += B
        t, m = self.tok[idx], self.mask[idx]
        return t[:, :-1], t[:, 1:], m[:, 1:]


class Prefetch:
    """background thread producing batches (numpy) from a callable"""

    def __init__(self, fn, depth=4):
        self.q = queue.Queue(depth)
        self.fn = fn
        self.err = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            while True:
                self.q.put(self.fn())
        except Exception as e:
            self.err = e
            self.q.put(None)

    def get(self):
        x = self.q.get()
        if x is None:
            raise self.err
        return x
