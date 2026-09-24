"""Download + tokenize the training corpus (resumable, round-robin across sources).

Every source is filled in rounds (10% of its target per round) so training can start
after the first round with the full mixture available.  Output per source:
  data/<name>/<k>.bin        uint16 tokens, documents separated by EOS (0)
  data/<name>/<k>.mask.bin   (sft only) uint8, 1 = trained token
  data/<name>/state.json     files done + token count
"""
import json
import os
import random
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
DL = os.path.join(DATA, "_dl")
B = 1_000_000_000
M = 1_000_000

EOS, IM_START, IM_END = 0, 1, 2
TOK_REPO = "HuggingFaceTB/SmolLM2-135M"

# name: repo, file prefix, kind, target tokens, max tokens per doc (chat)
SOURCES = {
    # ---- pretraining web / edu / math (stable phase and anneal)
    "fwedu":   dict(repo="HuggingFaceTB/smollm-corpus", prefix="fineweb-edu-dedup/", kind="text", col="text", target=4.5 * B),
    "ufw":     dict(repo="openbmb/Ultra-FineWeb", prefix="data/ultrafineweb_en/", kind="text", col="content", target=2.0 * B),
    "dclm":    dict(repo="mlfoundations/dclm-baseline-1.0-parquet", prefix="filtered/", kind="text", col="text", target=1.0 * B),
    "fmath":   dict(repo="HuggingFaceTB/finemath", prefix="finemath-4plus/", kind="text", col="text", target=1.0 * B),
    "iwm":     dict(repo="HuggingFaceTB/finemath", prefix="infiwebmath-4plus/", kind="text", col="text", target=0.4 * B),
    "cosmo":   dict(repo="HuggingFaceTB/smollm-corpus", prefix="cosmopedia-v2/", kind="text", col="text", target=1.0 * B),
    # ---- reasoning / instruction data (anneal phase, trained on all tokens)
    "think":   dict(repo="HuggingFaceTB/smoltalk2", prefix="Mid/", kind="chat", target=0.3 * B, maxlen=2048),
    "omi2":    dict(repo="nvidia/OpenMathInstruct-2", prefix="data/train", kind="chat", target=0.25 * B, maxlen=2048),
    "numina":  dict(repo="AI-MO/NuminaMath-CoT", prefix="data/train", kind="chat", target=0.12 * B, maxlen=2048),
    "magpie":  dict(repo="HuggingFaceTB/smoltalk2", prefix="SFT/smoltalk_smollm3_smol_magpie_ultra", kind="chat", target=0.1 * B, maxlen=2048),
    # ---- SFT (assistant-only loss, 4096 context)
    "sft":     dict(repo="HuggingFaceTB/smoltalk2", prefix="SFT/", kind="sft", target=60 * M, maxlen=4096,
                    skip=("LongAlign", "multilingual", "aya_", "xlam", "hermes_function", "smolagents", "table_gpt", "OpenHermes"),
                    # OpenThoughts3 think is huge; cap its share so everyday chat/IF data is represented
                    cap={"OpenThoughts3_1.2M_think": 25 * M, "smol_magpie_ultra": 15 * M}),
    "sftmath": dict(repo="nvidia/OpenMathInstruct-2", prefix="data/train", kind="sft", target=15 * M, maxlen=4096, reverse=True),
}

_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(hf_hub_download(TOK_REPO, "tokenizer.json"))
    return _tok


def list_files(src):
    api = HfApi()
    for i in range(5):
        try:
            fl = [s.rfilename for s in api.dataset_info(src["repo"]).siblings]
            break
        except Exception as e:
            print("list retry", e, flush=True); time.sleep(10 * (i + 1))
    fl = sorted(f for f in fl if f.startswith(src["prefix"]) and f.endswith(".parquet")
                and not any(s in f for s in src.get("skip", ())))
    if src["kind"] == "text":
        random.Random(1234).shuffle(fl)      # spread across dumps / shards
    if src.get("reverse"):
        fl = fl[::-1]                        # disjoint from the anneal copy of the same repo
    return fl


def download(repo, f):
    for i in range(8):
        try:
            return hf_hub_download(repo, f, repo_type="dataset", local_dir=DL)
        except Exception as e:
            print(f"  download retry {i} {f}: {e!r}"[:300], flush=True)
            time.sleep(15 * (i + 1))
    raise RuntimeError("download failed " + f)


# ---------------------------------------------------------------- rendering
def messages_of(row):
    if "messages" in row and row["messages"] is not None:
        m = row["messages"]
        if isinstance(m, str):
            import ast
            m = ast.literal_eval(m)
        return [(x["role"], x["content"]) for x in m]
    if "generated_solution" in row:
        return [("user", row["problem"]), ("assistant", row["generated_solution"])]
    if "solution" in row:
        return [("user", row["problem"]), ("assistant", row["solution"])]
    return None


def chat_segments(msgs):
    """ChatML segments [(text, trained)]; assistant content + <|im_end|> is trained."""
    segs = []
    for role, content in msgs:
        if role not in ("system", "user", "assistant"):
            return None
        content = (content or "").strip()
        if role == "assistant":
            segs.append(("<|im_start|>assistant\n", False))
            segs.append((content + "<|im_end|>", True))
            segs.append(("\n", False))
        else:
            segs.append((f"<|im_start|>{role}\n{content}<|im_end|>\n", False))
    if not segs or not any(t for _, t in segs):
        return None
    return segs


# ---------------------------------------------------------------- per file
def process_file(name, src, path, budget_left):
    """Returns (tokens array, mask array or None, ntokens, per-file source counts)."""
    kind = src["kind"]
    pf = pq.ParquetFile(path)
    out, masks, n = [], [], 0
    fname = os.path.basename(path)
    tk = tok()
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        if kind == "text":
            texts = [t for t in tbl.column(src["col"]).to_pylist() if t]
            for i in range(0, len(texts), 2000):
                enc = tk.encode_batch(texts[i:i + 2000], add_special_tokens=False)
                for e in enc:
                    ids = e.ids
                    if len(ids) < 16:
                        continue
                    out.append(np.asarray(ids + [EOS], dtype=np.uint16))
                    n += len(ids) + 1
                if n >= budget_left:
                    break
        else:
            rows = tbl.to_pylist()
            for i in range(0, len(rows), 1000):
                segs_list = []
                for r in rows[i:i + 1000]:
                    if "inference_mode" in r and r["inference_mode"] != "cot":
                        continue
                    msgs = messages_of(r)
                    segs = chat_segments(msgs) if msgs else None
                    if segs:
                        segs_list.append(segs)
                if not segs_list:
                    continue
                flat = [s for segs in segs_list for s, _ in segs]
                enc = tk.encode_batch(flat, add_special_tokens=False)
                k = 0
                for segs in segs_list:
                    ids, m = [], []
                    for s, trained in segs:
                        e = enc[k].ids; k += 1
                        ids += e; m += [int(trained)] * len(e)
                    if len(ids) + 1 > src["maxlen"]:
                        continue
                    ids.append(EOS); m.append(0)
                    out.append(np.asarray(ids, dtype=np.uint16))
                    if kind == "sft":
                        masks.append(np.asarray(m, dtype=np.uint8))
                    n += len(ids)
                if n >= budget_left:
                    break
        if n >= budget_left:
            break
    toks = np.concatenate(out) if out else np.zeros(0, np.uint16)
    mk = (np.concatenate(masks) if masks else np.zeros(0, np.uint8)) if kind == "sft" else None
    return toks, mk, n


def load_state(name):
    p = os.path.join(DATA, name, "state.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"done": [], "tokens": 0, "per_group": {}}


def save_state(name, st):
    d = os.path.join(DATA, name)
    tmp = os.path.join(d, "state.json.tmp")
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(d, "state.json"))


def group_of(src, f):
    for k in src.get("cap", {}):
        if k in f:
            return k
    return None


def fill(name, upto, pool):
    src = SOURCES[name]
    os.makedirs(os.path.join(DATA, name), exist_ok=True)
    st = load_state(name)
    if st["tokens"] >= upto:
        return False
    files = [f for f in list_files(src) if f not in st["done"]]
    if src["kind"] == "sft":
        # interleave subsets so a partial fill is still a mixture; skip capped groups that are full
        files = [f for f in files if not (group_of(src, f) and st["per_group"].get(group_of(src, f), 0) >= src["cap"][group_of(src, f)])]
    if not files:
        print(f"[{name}] no files left ({st['tokens'] / M:.0f}M tokens)", flush=True)
        return False
    fut = {0: pool.submit(download, src["repo"], files[0])}
    i = 0
    while st["tokens"] < upto and i < len(files):
        if i + 1 < len(files):
            fut[i + 1] = pool.submit(download, src["repo"], files[i + 1])
        f = files[i]
        path = fut.pop(i).result()
        g = group_of(src, f)
        left = upto - st["tokens"]
        if g:
            left = min(left, src["cap"][g] - st["per_group"].get(g, 0))
        if left <= 0:
            i += 1; os.remove(path); continue
        t0 = time.time()
        toks, mk, n = process_file(name, src, path, left)
        k = len(st["done"])
        base = os.path.join(DATA, name, f"{k:05d}")
        toks.tofile(base + ".bin.tmp"); os.replace(base + ".bin.tmp", base + ".bin")
        if mk is not None:
            mk.tofile(base + ".mask.bin.tmp"); os.replace(base + ".mask.bin.tmp", base + ".mask.bin")
        st["done"].append(f)
        st["tokens"] += int(n)
        if g:
            st["per_group"][g] = st["per_group"].get(g, 0) + int(n)
        save_state(name, st)
        os.remove(path)
        print(f"[{name}] {f.split('/')[-1][:60]} +{n / M:.1f}M -> {st['tokens'] / M:.0f}M / {src['target'] / M:.0f}M "
              f"({n / max(time.time() - t0, 1e-3) / 1e6:.2f}M tok/s)", flush=True)
        i += 1
    for j, fu in fut.items():   # drop prefetched but unused downloads
        try:
            os.remove(fu.result())
        except Exception:
            pass
    return True


def main():
    rounds = int(os.environ.get("ROUNDS", "10"))
    only = sys.argv[1:]
    os.makedirs(DL, exist_ok=True)
    pool = ThreadPoolExecutor(2)
    for r in range(1, rounds + 1):
        for name, src in SOURCES.items():
            if only and name not in only:
                continue
            upto = src["target"] if src["kind"] == "sft" else src["target"] * r / rounds
            fill(name, upto, pool)
        tot = sum(load_state(n)["tokens"] for n in SOURCES)
        print(f"==== round {r}/{rounds} done, {tot / B:.2f}B tokens on disk", flush=True)
        open(os.path.join(DATA, "ROUND"), "w").write(str(r))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
