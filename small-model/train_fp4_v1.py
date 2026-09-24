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
# all gradients are views of one flat buffer -> a single zero per step
gflat = torch.zeros(sum(q.numel() for q in ref.parameters()), device=dev)
_o = 0
for q in ref.parameters():
    q.grad = gflat[_o:_o + q.numel()].view_as(q); _o += q.numel()
wset = F.WeightSet([l[k] for l in layers for k in ("Wq", "Wo", "W1", "W2")])
xL = torch.empty(M, D, dtype=torch.bfloat16, device=dev)
dxa = torch.empty(M, D, dtype=torch.bfloat16, device=dev)
dxb = torch.empty(M, D, dtype=torch.bfloat16, device=dev)
mb = {k: F.WgradBuf(n, M) for k, n in (("dy", 128), ("a", 512), ("du", 512), ("h", 128))}
ab = {k: F.WgradBuf(n, M) for k, n in (("dx", 128), ("o", 128), ("dq", 384), ("h", 128))}
rngp = torch.zeros(2, dtype=torch.int32, device=dev)
Q = fp4ops.Q

decay = [q for n, q in ref.named_parameters() if q.dim() == 2 and "pos" not in n]
nodecay = [q for n, q in ref.named_parameters() if not (q.dim() == 2 and "pos" not in n)]
lr_t = torch.tensor(args.lr, device=dev)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=lr_t, betas=(0.9, 0.98), fused=True, capturable=True)
static_x = torch.zeros(B, T, dtype=torch.long, device=dev)
static_y = torch.zeros(B, T, dtype=torch.long, device=dev)
static_w = torch.zeros(B, T, device=dev)


def forward(idx, Mx):
    x = (ref.tok.weight[idx] + ref.pos[:T]).reshape(Mx, D).to(torch.bfloat16)
    for l in layers:
        l["x"][:Mx].copy_(x)
        F.attn_fwd(l["x"][:Mx], l["ln1"], l["Wq"], l["Wo"], out=l["x1"][:Mx])
        x = F.mlp_fwd(l["x1"][:Mx], l["ln2"], l["W1"], l["W2"], out=xL[:Mx] if l is layers[-1] else None)
    return x


def train_step():
    # new randomness for stochastic rounding + Hadamard signs every step (graph-safe: device side)
    rngp.copy_(torch.randint(-2**31, 2**31 - 1, (2,), device=dev, dtype=torch.int32))
    gflat.zero_(); amax_cur_all.zero_()
    x = forward(static_x, M)
    xf = x.float().requires_grad_()
    with torch.autocast("cuda", torch.bfloat16):
        logits = ref.head(ref.lnf(xf))
    loss = masked_loss(logits, static_y, static_w)
    loss.backward()
    dx = xf.grad.to(torch.bfloat16)
    for i in reversed(range(L)):
        l, blk = layers[i], ref.blocks[i]
        # ---- MLP
        dx1 = F.mlp_bwd(l["x1"], dx, l["ln2"], blk.ln2.weight.grad, l["W1"], l["W2"], mb,
                        l["amax_use_m"], l["amax_cur_m"], rngp, dx1=dxa)
        au = l["amax_use_m"]
        fp4ops.gemm(Q(mb["dy"].d, mb["dy"].s, au[0:1]), Q(mb["a"].d, mb["a"].s, au[1:2]), out=blk.fc2.weight.grad, splitk=16)
        fp4ops.gemm(Q(mb["du"].d, mb["du"].s, au[2:3]), Q(mb["h"].d, mb["h"].s, au[3:4]), out=blk.fc1.weight.grad, splitk=16)
        # ---- attention
        dx = F.attn_bwd(l["x"], dx1, l["ln1"], blk.ln1.weight.grad, l["Wq"], l["Wo"], ab,
                        l["amax_use_a"], l["amax_cur_a"], rngp, dx=dxb)
        au = l["amax_use_a"]
        fp4ops.gemm(Q(ab["dx"].d, ab["dx"].s, au[0:1]), Q(ab["o"].d, ab["o"].s, au[1:2]), out=blk.proj.weight.grad, splitk=16)
        fp4ops.gemm(Q(ab["dq"].d, ab["dq"].s, au[2:3]), Q(ab["h"].d, ab["h"].s, au[3:4]), out=blk.qkv.weight.grad, splitk=16)
    # delayed scaling: this step's observed amax (with headroom) is next step's scale
    torch.mul(amax_cur_all.view(torch.float32).clamp(min=1e-20), args.margin, out=amax_use_all)
    g0 = dx.float().view(B, T, D)
    ref.tok.weight.grad.index_add_(0, static_x.reshape(-1), g0.reshape(M, D))
    ref.pos.grad.copy_(g0.sum(0))
    torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0, foreach=True)
    opt.step()
    wset.refresh()
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


def load_batch(s):
    x, y, w = data.batch(s, B)
    static_x.copy_(x); static_y.copy_(y); static_w.copy_(w)


# ---- calibrate delayed scales: one fwd/bwd without applying the update
load_batch(0)
saved = [q.detach().clone() for q in ref.parameters()]
st = {k: v for k, v in opt.state_dict().items()}
for _ in range(2):
    train_step()
    with torch.no_grad():
        for q, s0 in zip(ref.parameters(), saved):
            q.copy_(s0)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=lr_t, betas=(0.9, 0.98), fused=True, capturable=True)
wset.refresh()
print("calibrated amax (layer0 mlp/attn):", layers[0]["amax_use_m"].tolist(), layers[0]["amax_use_a"].tolist())

step_fn = train_step
if args.graph:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # warm up (allocations, autograd state) on the side stream
            lr_t.fill_(0.0); load_batch(0); train_step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_loss = train_step()
    with torch.no_grad():  # undo the warm-up's effect on optimizer state (params unchanged: lr was 0)
        for q, s0 in zip(ref.parameters(), saved):
            q.copy_(s0)
        for stt in opt.state.values():
            for v in stt.values():
                if torch.is_tensor(v):
                    v.zero_()
    def step_fn():
        g.replay()
        return static_loss

if os.environ.get("PROFILE"):
    from torch.profiler import profile, ProfilerActivity
    lr_t.fill_(1e-4)
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
    lr_t.fill_(1e-4)
    for s_ in range(30):
        load_batch(s_); step_fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s_ in range(args.bench):
        load_batch(s_); step_fn()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / args.bench
    print(f"FP4 fused graph={args.graph} bs={B}: {dt*1e3:.3f} ms/step, {M/dt/1e6:.2f} M tok/s")
    sys.exit()

log = {"args": vars(args), "train": [], "eval": []}
t0 = time.perf_counter()
for s_ in range(args.steps):
    lr_t.fill_(lr_at(s_))
    load_batch(s_)
    loss = step_fn()
    if s_ % 50 == 0:
        log["train"].append((s_, loss.item()))
    if (s_ + 1) % args.eval_every == 0 or s_ == args.steps - 1:
        el, acc = evaluate()
        log["eval"].append((s_ + 1, el, acc))
        print(f"step {s_+1} train {loss.item():.4f} eval {el:.4f} acc {acc*100:.2f}% t={time.perf_counter()-t0:.0f}s", flush=True)
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
