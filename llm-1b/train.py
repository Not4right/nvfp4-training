"""NVFP4 pretraining (+ BF16 tail, + SFT) of the 1B model, fully in VRAM, crash-safe.

Phases (one process per phase; run.ps1 restarts on exit code 10 = phase change, or after crashes):
  pretrain  first SWITCH (85%) of the token budget in NVFP4, WSD learning rate (warmup, flat)
            last 15% in BF16 (Desktop/nvfp4 --switch_frac 0.15) = the LR cooldown, with the
            math/reasoning/instruction "anneal" data mixture
  sft       chat + reasoning SFT, BF16, 4096 context, loss on assistant tokens only
  done

Files:   ckpt/latest.pt (every --ckpt_min minutes, atomic)   ckpt/backup.pt (copy every --backup_h hours)
         logs/train.log, logs/metrics.jsonl, logs/status.json
Flags:   PAUSE  -> checkpoint and exit (run.ps1 then stops)       FINISH -> start the BF16 cooldown now
"""
import argparse
import ctypes
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "C:/tic")
os.environ.setdefault("TRITON_CACHE_DIR", "C:/tc")

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=float, default=10e9, help="pretraining token budget")
p.add_argument("--batch_tokens", type=int, default=524288)
p.add_argument("--B", type=int, default=2)
p.add_argument("--T", type=int, default=2048)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--warmup", type=int, default=300)
p.add_argument("--switch", type=float, default=0.85, help="fraction of pretraining in NVFP4 (rest BF16 + cooldown)")
p.add_argument("--sft_epochs", type=float, default=1.0)
p.add_argument("--sft_lr", type=float, default=1.5e-4)
p.add_argument("--sft_batch_tokens", type=int, default=262144)
p.add_argument("--ckpt_min", type=float, default=60)
p.add_argument("--backup_h", type=float, default=5)
p.add_argument("--val_every", type=int, default=200)
p.add_argument("--mem_frac", type=float, default=0.92)
p.add_argument("--max_steps_this_run", type=int, default=0, help="testing: exit after N steps")
p.add_argument("--tag", default="")
p.add_argument("--data", default="data")
p.add_argument("--attn", default="fp8", choices=["fp8", "cudnn"])
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(args.mem_frac)   # never spill into shared memory (froze Windows once)
import torch._inductor.config as _ic
_ic.use_static_cuda_launcher = False
import model2 as M2
from optim import Optim
from loader import Mixture, SFTData, Prefetch, STABLE, ANNEAL

CK = os.path.join(ROOT, "ckpt" + args.tag)
LOGD = os.path.join(ROOT, "logs")
os.makedirs(CK, exist_ok=True); os.makedirs(LOGD, exist_ok=True)
LATEST, BACKUP = os.path.join(CK, "latest.pt"), os.path.join(CK, "backup.pt")
PAUSE, FINISH = os.path.join(ROOT, "PAUSE"), os.path.join(ROOT, "FINISH")
logf = open(os.path.join(LOGD, f"train{args.tag}.log"), "a", buffering=1)
metf = open(os.path.join(LOGD, f"metrics{args.tag}.jsonl"), "a", buffering=1)


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")


# keep Windows awake while training (sleep/hibernate silently stops CUDA work)
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass

dev = "cuda"
state = torch.load(LATEST, map_location="cpu", weights_only=False) if os.path.exists(LATEST) else None
phase = state["phase"] if state else "pretrain"
if phase == "done":
    log("training already finished"); sys.exit(0)

cfg = M2.Config()
if phase == "pretrain":
    B, T = args.B, args.T
else:
    B, T = 1, 4096
model = M2.Model(cfg, B, T, dev)
model.attn = args.attn
opt = Optim(model)
M = B * T

if state:
    model.P.flat.copy_(state["flat"]); model.P.mom.copy_(state["mom"])
    opt.load_state_dict(state["opt"])
    model.amax_use.copy_(state["amax_use"])
    if "ds_use" in state:
        model.ds_use.copy_(state["ds_use"])
    step, tokens = state["step"], state["tokens"]
    total_steps, switch_step = state["total_steps"], state["switch_step"]
    run_t0_tokens = tokens
    log(f"resumed from {LATEST}: phase {phase} step {step} tokens {tokens/1e9:.3f}B")
else:
    model.P.init(0)
    for k, v in model.P.small():
        opt.small[k].copy_(v.float())
    step, tokens = 0, 0
    total_steps = int(args.tokens // args.batch_tokens)
    switch_step = int(total_steps * args.switch)
    log(f"new run: {total_steps} steps x {args.batch_tokens} tokens, NVFP4 until step {switch_step}, BF16 after")
del state
torch.cuda.empty_cache()


def save(ph=None):
    sd = {"flat": model.P.flat, "mom": model.P.mom, "opt": opt.state_dict(), "amax_use": model.amax_use, "ds_use": model.ds_use,
          "step": step, "tokens": tokens, "phase": ph or phase, "total_steps": total_steps, "switch_step": switch_step,
          "data": data.state_dict() if phase == "pretrain" else {"i": data.i}, "args": vars(args), "cfg": vars(cfg),
          "time": time.time()}
    tmp = LATEST + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, LATEST)


def backup():
    tmp = BACKUP + ".tmp"
    shutil.copyfile(LATEST, tmp)
    os.replace(tmp, BACKUP)
    log(f"backup written: {BACKUP}")


def status(**kw):
    json.dump(dict(phase=phase, step=step, total_steps=total_steps, tokens=tokens, time=time.time(), **kw),
              open(os.path.join(LOGD, "status.json.tmp"), "w"))
    os.replace(os.path.join(LOGD, "status.json.tmp"), os.path.join(LOGD, "status.json"))


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    if s < switch_step:
        return args.lr
    f = (s - switch_step) / max(1, total_steps - switch_step)      # 1-sqrt cooldown (Hagele et al. 2024)
    return args.lr * max(0.0, 1 - math.sqrt(min(f, 1.0)))


def to_dev(a):
    return torch.from_numpy(a).pin_memory().to(dev, non_blocking=True)


# ======================================================================= pretraining
if phase == "pretrain":
    data = Mixture(os.path.join(ROOT, args.data), T)
    if os.path.exists(LATEST):
        data.load_state_dict(torch.load(LATEST, map_location="cpu", weights_only=False)["data"])
    n_micro = args.batch_tokens // M
    model.acc_scale = 1.0 / n_micro
    mask = torch.ones(B, T, device=dev)

    def set_phase_mode():
        fp4 = step < switch_step
        model.fp4 = fp4
        model.attn = args.attn if fp4 else "cudnn"     # low-precision attention only in the fast phase
        data.weights = STABLE if fp4 else ANNEAL
        model.refresh_weights()
    set_phase_mode()
    pf = Prefetch(lambda: data.batch(B), depth=8)

    vset = None

    def run_val():
        global vset
        if vset is None:
            vset = {k: [(to_dev(x), to_dev(y)) for x, y in v]
                    for k, v in data.val_batches(["fwedu", "fmath", "cosmo", "think", "dclm"], B, 4).items()}
        out = {}
        for k, bs in vset.items():
            ls = [model.forward_loss(x, y, mask, grad=False)[0].item() for x, y in bs]
            out[k] = sum(ls) / len(ls)
        return out

    if step == 0:
        # calibrate the delayed wgrad scales (nvfp4: two passes without applying the update)
        for _ in range(2):
            x, y = pf.get()
            model.microbatch(to_dev(x), to_dev(y), mask)
        model.P.mom.zero_(); model.g_emb.zero_()
        for g in model.g_small.values():
            g.zero_()
        log("calibrated wgrad amax:", [round(v, 4) for v in model.amax_use[0].tolist()])

    t_ck, t_bk = time.time(), time.time()
    t_run0, tok_run0 = time.time(), tokens
    loss_acc = torch.zeros((), device=dev)
    steps_this_run = 0
    while step < total_steps:
        if os.path.exists(FINISH) and step < switch_step:
            os.remove(FINISH)
            switch_step = step
            total_steps = max(step + 1, int(math.ceil(step / args.switch)))
            log(f"FINISH requested: BF16 cooldown now, ending at step {total_steps}")
        if step == switch_step:
            log(f"=== step {step}: switching to BF16 + cooldown + anneal mixture ===")
        set_phase_mode() if step == switch_step else None
        lr = lr_at(step)
        t0 = time.time()
        loss_acc.zero_()
        for i in range(n_micro):
            x, y = pf.get()
            loss_acc += model.microbatch(to_dev(x), to_dev(y), mask)
        gn = opt.step(lr, n_micro)
        loss = (loss_acc / n_micro).item()
        dt = time.time() - t0
        step += 1; steps_this_run += 1
        tokens += n_micro * M
        if not math.isfinite(loss):
            log(f"non-finite loss at step {step}; reloading {LATEST} and continuing with new data")
            sd = torch.load(LATEST, map_location="cpu", weights_only=False)
            model.P.flat.copy_(sd["flat"]); model.P.mom.copy_(sd["mom"]); opt.load_state_dict(sd["opt"])
            model.amax_use.copy_(sd["amax_use"]); step, tokens = sd["step"], sd["tokens"]
            model.refresh_weights(); del sd
            continue
        tps = (tokens - tok_run0) / (time.time() - t_run0)
        left = (total_steps - step) * args.batch_tokens / max(tps, 1)
        rec = dict(step=step, loss=round(loss, 4), lr=lr, gnorm=round(float(gn), 3), tok=tokens, tps=round(tps),
                   fp4=model.fp4, dt=round(dt, 2))
        metf.write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step < 20:
            log(f"step {step}/{total_steps} loss {loss:.4f} lr {lr:.2e} gnorm {float(gn):.3f} "
                f"{'FP4' if model.fp4 else 'BF16'} {tps:.0f} tok/s  {tokens/1e9:.3f}B tok  ETA {left/3600:.1f} h"
                + (f"  wraps {data.wraps}" if any(data.wraps.values()) else ""))
        status(loss=loss, tps=tps, eta_h=left / 3600, fp4=model.fp4)
        if step % args.val_every == 0:
            v = run_val()
            log("val " + " ".join(f"{k} {l:.4f}" for k, l in v.items()))
            metf.write(json.dumps(dict(step=step, val=v)) + "\n")
        now = time.time()
        if now - t_ck > args.ckpt_min * 60:
            save(); t_ck = now
            log(f"checkpoint {LATEST}")
            if now - t_bk > args.backup_h * 3600:
                backup(); t_bk = now
        if os.path.exists(PAUSE):
            save(); log("PAUSE: checkpoint written, exiting"); sys.exit(3)
        if args.max_steps_this_run and steps_this_run >= args.max_steps_this_run:
            save(); log("max_steps_this_run reached"); sys.exit(0)
    # pretraining complete
    v = run_val(); log("final val " + " ".join(f"{k} {l:.4f}" for k, l in v.items()))
    save()
    shutil.copyfile(LATEST, os.path.join(CK, "pretrained.pt"))
    step = 0
    save("sft")                      # the SFT process starts its loader at 0
    log("pretraining done -> SFT phase (restart)")
    sys.exit(10)

# ======================================================================= SFT
if phase == "sft":
    data = SFTData(os.path.join(ROOT, args.data), T)
    if os.path.exists(LATEST):
        d = torch.load(LATEST, map_location="cpu", weights_only=False)["data"]
        data.i = d.get("i", 0)
    model.fp4 = False
    model.attn = "cudnn"
    model.refresh_weights()
    n_micro = args.sft_batch_tokens // M
    model.acc_scale = 1.0 / n_micro
    rows = len(data.tok)
    total = max(1, int(args.sft_epochs * rows // (n_micro * B)))
    if step == 0:
        total_steps = total
        log(f"SFT: {rows} packed rows of {T}, {data.ntok/1e6:.1f}M trained tokens, {total} steps")
    t_ck = time.time()
    while step < total_steps:
        lr = args.sft_lr * min(1.0, (step + 1) / 20) * (1 - step / total_steps)
        t0 = time.time()
        acc = 0.0
        for i in range(n_micro):
            x, y, m = data.batch(B)
            acc += model.microbatch(to_dev(x), to_dev(y), to_dev(m)).item()
        opt.step(lr, n_micro)
        step += 1
        tokens += n_micro * M
        log(f"sft step {step}/{total_steps} loss {acc/n_micro:.4f} lr {lr:.2e} {n_micro*M/(time.time()-t0):.0f} tok/s")
        status(loss=acc / n_micro)
        if time.time() - t_ck > args.ckpt_min * 60:
            save(); t_ck = time.time()
        if os.path.exists(PAUSE):
            save(); log("PAUSE: checkpoint written, exiting"); sys.exit(3)
    save("done")
    shutil.copyfile(LATEST, os.path.join(CK, "final.pt"))
    log("ALL DONE: final model in ckpt/final.pt")
    sys.exit(0)
