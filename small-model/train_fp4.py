"""End-to-end NVFP4 training with the fused kernels (same model / data / init / optimizer as train_ref).

Per step:  embed -> 5 x (attn_fwd, mlp_fwd) -> [LN_f + head + masked CE in torch]
           -> 5 x (mlp_bwd, 2 wgrad GEMMs, attn_bwd, 2 wgrad GEMMs) -> embed bwd -> clip -> AdamW
           -> re-quantize weights.  Optionally captured as one CUDA graph.
"""
import os, sys, time, math, json, argparse
import torch
import torch.nn.functional as TF
from data import Data
from model_ref import GPT, masked_loss
import fusedops as F
import fp4ops

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--bs", type=int, default=512)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--bench", type=int, default=0)
p.add_argument("--graph", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--margin", type=float, default=2.0, help="delayed-amax headroom for wgrad/dgrad operand scales")
p.add_argument("--tag", default="")
p.add_argument("--switch_frac", type=float, default=0.0, help="final fraction of steps trained in BF16 (NVIDIA recipe)")
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.9)
torch.manual_seed(args.seed)
dev = "cuda"
data = Data()
ref = GPT(mode="bf16").to(dev)   # identical init to the baseline run with the same seed
L, D, T, B = len(ref.blocks), 128, 32, args.bs
M = B * T
print("params", sum(q.numel() for q in ref.parameters()))

QP = F.qkv_perm(dev)
amax_use_all = torch.ones(L * 9, device=dev)
amax_cur_all = torch.zeros(L * 9, dtype=torch.int32, device=dev)
layers = []
for blk in ref.blocks:
    layers.append(dict(
        ln1=blk.ln1.weight, ln2=blk.ln2.weight,
        Wq=F.PackedW(blk.qkv.weight, ffN=True, rowperm=QP), Wo=F.PackedW(blk.proj.weight),
        W1=F.PackedW(blk.fc1.weight, ffN=True), W2=F.PackedW(blk.fc2.weight, ffK=True),
        x=torch.empty(M, D, dtype=torch.bfloat16, device=dev), x1=torch.empty(M, D, dtype=torch.bfloat16, device=dev),
    ))
    i9 = 9 * len(layers[:-1])
    layers[-1].update(amax_use_m=amax_use_all[i9:i9 + 4], amax_cur_m=amax_cur_all[i9:i9 + 4],
                      amax_use_a=amax_use_all[i9 + 4:i9 + 9], amax_cur_a=amax_cur_all[i9 + 4:i9 + 9])
# params / grads / Adam state are views of flat buffers -> one kernel each per step
NP = sum(q.numel() for q in ref.parameters())
pflat = torch.zeros(NP, device=dev); gflat = torch.zeros(NP, device=dev)
mflat = torch.zeros(NP, device=dev); vflat = torch.zeros(NP, device=dev)
wdmask = torch.zeros(NP, dtype=torch.uint8, device=dev)
_o = 0
for n_, q in ref.named_parameters():
    k_ = q.numel()
    pflat[_o:_o + k_].copy_(q.detach().reshape(-1))
    q.data = pflat[_o:_o + k_].view_as(q)
    q.grad = gflat[_o:_o + k_].view_as(q)
    if q.dim() == 2 and "pos" not in n_:
        wdmask[_o:_o + k_] = 1
    _o += k_
hp = torch.tensor([args.lr, 0.0], device=dev)       # lr, step
gsq = torch.zeros(1, device=dev); loss_acc = torch.zeros(1, device=dev)
wsum = torch.zeros(1, device=dev)
wset = F.WeightSet([l[k] for l in layers for k in ("Wq", "Wo", "W1", "W2")])
xL = torch.empty(M, D, dtype=torch.bfloat16, device=dev)
dxa = torch.empty(M, D, dtype=torch.bfloat16, device=dev)
dxb = torch.empty(M, D, dtype=torch.bfloat16, device=dev)
mb = {k: F.WgradBuf(n, M) for k, n in (("dy", 128), ("a", 512), ("du", 512), ("h", 128))}
ab = {k: F.WgradBuf(n, M) for k, n in (("dx", 128), ("o", 128), ("dq", 384), ("h", 128))}
rngp = torch.zeros(2, dtype=torch.int32, device=dev)
# grouped wgrad GEMMs per layer (operand buffers are shared across layers; outputs are the grads)
for l, blk in zip(layers, ref.blocks):
    l["wg_m"] = F.WGGemm([(mb["dy"], l["amax_use_m"][0:1], mb["a"], l["amax_use_m"][1:2], blk.fc2.weight.grad, 16),
                          (mb["du"], l["amax_use_m"][2:3], mb["h"], l["amax_use_m"][3:4], blk.fc1.weight.grad, 16)], M)
    l["wg_a"] = F.WGGemm([(ab["dx"], l["amax_use_a"][0:1], ab["o"], l["amax_use_a"][1:2], blk.proj.weight.grad, 16),
                          (ab["dq"], l["amax_use_a"][2:3], ab["h"], l["amax_use_a"][3:4], blk.qkv.weight.grad, 16)], M)
Q = fp4ops.Q

static_x = torch.zeros(B, T, dtype=torch.long, device=dev)
static_y = torch.zeros(B, T, dtype=torch.long, device=dev)
static_w = torch.zeros(B, T, device=dev)


def forward(idx, Mx):
    x0 = layers[0]["x"]
    F.launch(F.fn("emb_fwd"), ((Mx * 16 + 255) // 256,), (256,), [idx, ref.tok.weight, ref.pos, x0, Mx])
    for i, l in enumerate(layers):
        F.attn_fwd(l["x"][:Mx], l["ln1"], l["Wq"], l["Wo"], out=l["x1"][:Mx])
        F.mlp_fwd(l["x1"][:Mx], l["ln2"], l["W1"], l["W2"], out=(layers[i + 1]["x"] if i + 1 < L else xL)[:Mx])
    return xL[:Mx]


def train_step():
    # new randomness for stochastic rounding + Hadamard signs every step (graph-safe: device side)
    rngp.copy_(torch.randint(-2**31, 2**31 - 1, (2,), device=dev, dtype=torch.int32))
    gflat.zero_(); amax_cur_all.zero_(); gsq.zero_(); loss_acc.zero_()
    torch.sum(static_w, dim=(0, 1), keepdim=False, out=wsum[0])
    x = forward(static_x, M)
    F.launch(F.fn("lm_tail_tc"), (2 * F.NSM,), (256,), [x, ref.lnf.weight, ref.head.weight, static_y, static_w, wsum,
                                                 dxb, ref.head.weight.grad, ref.lnf.weight.grad, loss_acc, M])
    dx = dxb
    for i in reversed(range(L)):
        l, blk = layers[i], ref.blocks[i]
        # ---- MLP
        dx1 = F.mlp_bwd(l["x1"], dx, l["ln2"], blk.ln2.weight.grad, l["W1"], l["W2"], mb,
                        l["amax_use_m"], l["amax_cur_m"], rngp, dx1=dxa)
        l["wg_m"]()
        # ---- attention
        dx = F.attn_bwd(l["x"], dx1, l["ln1"], blk.ln1.weight.grad, l["Wq"], l["Wo"], ab,
                        l["amax_use_a"], l["amax_cur_a"], rngp, dx=dxb)
        l["wg_a"]()
    # delayed scaling: this step's observed amax (with headroom) is next step's scale
    torch.mul(amax_cur_all.view(torch.float32).clamp(min=1e-20), args.margin, out=amax_use_all)
    F.launch(F.fn("emb_bwd"), (2 * F.NSM,), (128,), [static_x, dx, ref.tok.weight.grad, ref.pos.grad, M, 16])
    F.launch(F.fn("grad_sq"), (4 * F.NSM,), (256,), [gflat, gsq, NP])
    hp[1:2].add_(1.0)
    F.launch(F.fn("adamw_flat"), (4 * F.NSM,), (256,), [pflat, gflat, mflat, vflat, wdmask, hp, gsq, NP,
                                                        0.9, 0.98, 1e-8, 0.1, 1.0])
    wset.refresh()
    return loss_acc


import torch._inductor.config as _ic
_ic.use_static_cuda_launcher = False
os.environ.setdefault("TRITON_CACHE_DIR", "C:/tc"); os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "C:/ti")


def _bf16_loss(x, y, w):
    with torch.autocast("cuda", torch.bfloat16):
        return masked_loss(ref(x), y, w)


_bf16_loss_c = torch.compile(_bf16_loss)


def train_step_bf16():
    gflat.zero_(); gsq.zero_()
    loss = _bf16_loss_c(static_x, static_y, static_w)
    loss.backward()
    F.launch(F.fn("grad_sq"), (4 * F.NSM,), (256,), [gflat, gsq, NP])
    hp[1:2].add_(1.0)
    F.launch(F.fn("adamw_flat"), (4 * F.NSM,), (256,), [pflat, gflat, mflat, vflat, wdmask, hp, gsq, NP,
                                                        0.9, 0.98, 1e-8, 0.1, 1.0])
    return loss.detach()


def lr_at(s):
    w = 100
    if s < w:
        return args.lr * (s + 1) / w
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, args.steps - w))))


@torch.no_grad()
def evaluate():
    """FP4 inference with the fused forward kernels (true NVFP4 model quality)."""
    xe, ye, we = data.eval_set()
    tot, n, correct = 0.0, 0.0, 0
    for i in range(0, xe.shape[0], B):
        x, y, w = xe[i:i + B], ye[i:i + B], we[i:i + B]
        h = forward(x, M)
        with torch.autocast("cuda", torch.bfloat16):
            lg = ref.head(ref.lnf(h.float())).float().view(B, T, -1)
        l = TF.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view_as(w)
        tot += (l * w).sum().item(); n += w.sum().item()
        ok = (lg.argmax(-1) == y) | (w == 0)
        correct += ok[:, :16].all(1).sum().item() + ok[:, 16:].all(1).sum().item()
    return tot / n, correct / (2 * xe.shape[0])


@torch.no_grad()
def eval_bf16():
    """same weights, plain PyTorch BF16 inference (SDPA causal attention)"""
    xe, ye, we = data.eval_set()
    tot, n, correct = 0.0, 0.0, 0
    for i in range(0, xe.shape[0], 1024):
        x, y, w = xe[i:i + 1024], ye[i:i + 1024], we[i:i + 1024]
        with torch.autocast("cuda", torch.bfloat16):
            lg = ref(x).float()
        l = TF.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view_as(w)
        tot += (l * w).sum().item(); n += w.sum().item()
        ok = (lg.argmax(-1) == y) | (w == 0)
        correct += ok[:, :16].all(1).sum().item() + ok[:, 16:].all(1).sum().item()
    return tot / n, correct / (2 * xe.shape[0])


def load_batch(s):
    x, y, w = data.batch(s, B)
    static_x.copy_(x); static_y.copy_(y); static_w.copy_(w)


# ---- calibrate delayed scales: one fwd/bwd without applying the update
load_batch(0)
saved = pflat.clone()


def reset_opt():
    pflat.copy_(saved); mflat.zero_(); vflat.zero_(); hp[1] = 0.0
    wset.refresh()


for _ in range(2):
    train_step()
    reset_opt()
print("calibrated amax (layer0 mlp/attn):", layers[0]["amax_use_m"].tolist(), layers[0]["amax_use_a"].tolist())

step_fn = train_step
step_fn_bf16 = None
if args.switch_frac > 0:
    ref.train()
    pbak, mbak, vbak, hbak = pflat.clone(), mflat.clone(), vflat.clone(), hp.clone()
    s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(3):
            load_batch(0); train_step_bf16()
    torch.cuda.current_stream().wait_stream(s2)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        static_loss2 = train_step_bf16()
    pflat.copy_(pbak); mflat.copy_(mbak); vflat.copy_(vbak); hp.copy_(hbak)
    def step_fn_bf16():
        g2.replay()
        return static_loss2
if args.graph:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # warm up on the side stream
            load_batch(0); train_step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_loss = train_step()
    reset_opt()
    def step_fn():
        g.replay()
        return static_loss

if os.environ.get("PROFILE"):
    from torch.profiler import profile, ProfilerActivity
    hp[0] = 1e-4
    for s_ in range(5):
        load_batch(s_); train_step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for s_ in range(10):
            load_batch(s_); train_step()
        torch.cuda.synchronize()
    ev = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            k = e.name[:60]
            ev.setdefault(k, [0.0, 0])
            ev[k][0] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            ev[k][1] += 1
    tot = sum(v[0] for v in ev.values())
    print(f"total GPU time per step: {tot/10/1e3:.3f} ms")
    for k, v in sorted(ev.items(), key=lambda kv: -kv[1][0])[:30]:
        print(f"{v[0]/10:9.1f} us/step  {v[1]//10:4d}x  {k}")
    sys.exit()

if args.bench:
    hp[0] = 1e-4
    for s_ in range(30):
        load_batch(s_); step_fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s_ in range(args.bench):
        load_batch(s_); step_fn()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / args.bench
    print(f"FP4 fused graph={args.graph} bs={B}: {dt*1e3:.3f} ms/step, {M/dt/1e6:.2f} M tok/s, peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB (reserved {torch.cuda.max_memory_reserved()/2**20:.0f})")
    sys.exit()

log = {"args": vars(args), "train": [], "eval": []}
t0 = time.perf_counter()
switch_at = int(args.steps * (1 - args.switch_frac)) if args.switch_frac > 0 else args.steps + 1
for s_ in range(args.steps):
    hp[0] = lr_at(s_)
    load_batch(s_)
    if s_ == switch_at:
        print(f"-- switching to BF16 at step {s_}", flush=True)
    loss = step_fn_bf16() if s_ >= switch_at else step_fn()
    if s_ % 50 == 0:
        log["train"].append((s_, loss.item()))
    if (s_ + 1) % args.eval_every == 0 or s_ == args.steps - 1:
        el, acc = evaluate()
        elb, accb = eval_bf16()
        log["eval"].append((s_ + 1, el, acc, elb, accb))
        print(f"step {s_+1} train {loss.item():.4f} eval {el:.4f} acc {acc*100:.2f}% | bf16-inference eval {elb:.4f} acc {accb*100:.2f}% t={time.perf_counter()-t0:.0f}s", flush=True)
# ---- sanity checks
# (1) causality: perturb token j, fused-FP4 outputs at positions < j must be bit-identical
with torch.no_grad():
    xe, ye, we = data.eval_set()
    xb = xe[:B].clone()
    h0 = forward(xb, M).view(B, T, D).clone()
    for j in (5, 12, 20, 31):
        xp = xb.clone(); xp[:, j] = (xp[:, j] + 3) % 16
        h1 = forward(xp, M).view(B, T, D)
        print(f"causality j={j}: max|diff| before j = {(h1[:, :j] - h0[:, :j]).abs().max().item():.3e}, at/after j = {(h1[:, j:] - h0[:, j:]).abs().max().item():.3e}")
# (2) evaluate the FP4-trained master weights with the plain PyTorch BF16 model (SDPA causal attention)
@torch.no_grad()
def eval_torch():
    ref.eval()
    tot, n, correct = 0.0, 0.0, 0
    for i in range(0, xe.shape[0], 1024):
        x, y, w = xe[i:i + 1024], ye[i:i + 1024], we[i:i + 1024]
        with torch.autocast("cuda", torch.bfloat16):
            lg = ref(x).float()
        l = TF.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view_as(w)
        tot += (l * w).sum().item(); n += w.sum().item()
        ok = (lg.argmax(-1) == y) | (w == 0)
        correct += ok[:, :16].all(1).sum().item() + ok[:, 16:].all(1).sum().item()
    return tot / n, correct / (2 * xe.shape[0])
el, acc = eval_torch()
print(f"FP4-trained weights evaluated in plain PyTorch BF16: eval {el:.4f} acc {acc*100:.2f}%")
log["eval_bf16_model"] = (el, acc)
os.makedirs("runs", exist_ok=True)
json.dump(log, open(f"runs/fp4fused{args.tag}.json", "w"))
