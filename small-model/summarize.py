"""Aggregate final 3-seed runs (logs/final_*.log) into a table."""
import re, statistics, glob

rows = {}
for kind, pat in (("BF16 (torch.compile + CUDA graph)", "logs/final_bf16_s*.log"),
                  ("NVFP4 fused", "logs/final_fp4_s*.log"),
                  ("NVFP4 fused, last 15% BF16", "logs/final_fp4sw_s*.log")):
    res = []
    for f in sorted(glob.glob(pat)):
        txt = open(f).read()
        lines = [l for l in txt.splitlines() if l.startswith("step 16000")]
        if not lines:
            continue
        l = lines[-1]
        t = float(re.search(r"t=(\d+)s", l).group(1))
        m = re.search(r"eval ([\d.]+) acc ([\d.]+)%", l)
        ev, acc = float(m.group(1)), float(m.group(2))
        mb = re.search(r"bf16-inference eval ([\d.]+) acc ([\d.]+)%", l)
        evb, accb = (float(mb.group(1)), float(mb.group(2))) if mb else (ev, acc)
        res.append((ev, acc, evb, accb, t))
    rows[kind] = res

print(f"{'run':38s} {'n':>2s} {'eval loss':>18s} {'exact-match':>14s} {'eval loss (BF16 inf.)':>22s} {'acc (BF16 inf.)':>16s} {'wall s':>8s}")
for k, r in rows.items():
    if not r:
        continue
    f = lambda i: (statistics.mean(x[i] for x in r), statistics.pstdev(x[i] for x in r))
    ev, acc, evb, accb, t = (f(i) for i in range(5))
    print(f"{k:38s} {len(r):2d} {ev[0]:9.4f} ±{ev[1]:.4f} {acc[0]:8.2f}% ±{acc[1]:.1f} {evb[0]:13.4f} ±{evb[1]:.4f} {accb[0]:9.2f}% ±{accb[1]:.1f} {t[0]:8.0f}")
