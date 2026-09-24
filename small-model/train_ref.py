"""PyTorch reference training: BF16 baseline (eager / torch.compile) and NVFP4 fake-quant sim."""
import os, sys, time, math, json, argparse
os.environ.setdefault("TRITON_CACHE_DIR", "C:/tc")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "C:/ti")
import torch
import torch._inductor.config as _ic
_ic.use_static_cuda_launcher = False  # Windows: static launcher overflows on 64-bit stream handles
from data import Data
from model_ref import GPT, masked_loss, fp4_refresh

p = argparse.ArgumentParser()
p.add_argument("--mode", default="bf16")
p.add_argument("--compile", type=int, default=1)
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--bs", type=int, default=512)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--bench", type=int, default=0, help="only time N steps")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--graph", type=int, default=0, help="capture the whole step (compiled fwd/bwd + clip + AdamW) as a CUDA graph")
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.9)
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
data = Data()
model = GPT(mode=args.mode).cuda()
fp4_refresh(model)
print("params", sum(p.numel() for p in model.parameters()))
decay = [p for n, p in model.named_parameters() if p.dim() == 2 and "pos" not in n]
nodecay = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "pos" not in n)]
lr_t = torch.tensor(args.lr, device="cuda")
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=lr_t if args.graph else args.lr, betas=(0.9, 0.98), fused=True, capturable=bool(args.graph))


def lr_at(s):
    w = 100
    if s < w:
        return args.lr * (s + 1) / w
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, args.steps - w))))


def fwd_loss(x, y, w):
    with torch.autocast("cuda", torch.bfloat16):
        return masked_loss(model(x), y, w)


loss_fn = torch.compile(fwd_loss, mode=os.environ.get("CMODE") or None) if args.compile else fwd_loss


def train_step(x, y, w):
    loss = loss_fn(x, y, w)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    fp4_refresh(model)
    return loss


step_fn = train_step
if args.graph:
    sx = torch.zeros(args.bs, 32, dtype=torch.long, device="cuda"); sy = torch.zeros_like(sx); sw = torch.zeros(args.bs, 32, device="cuda")
    def _cap():
        loss = loss_fn(sx, sy, sw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        opt.step()
        opt.zero_grad(set_to_none=False)
        return loss.detach()
    x0, y0, w0 = data.batch(0, args.bs); sx.copy_(x0); sy.copy_(y0); sw.copy_(w0)
    saved = [q.detach().clone() for q in model.parameters()]
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            lr_t.fill_(0.0); _cap()
    torch.cuda.current_stream().wait_stream(st)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_loss = _cap()
    with torch.no_grad():
        for q, s0 in zip(model.parameters(), saved): q.copy_(s0)
        for stt in opt.state.values():
            for v in stt.values():
                if torch.is_tensor(v): v.zero_()
    def step_fn(x, y, w):
        sx.copy_(x); sy.copy_(y); sw.copy_(w); graph.replay(); return g_loss


@torch.no_grad()
def evaluate():
    model.eval()
    xe, ye, we = data.eval_set()
    tot, n, correct = 0.0, 0.0, 0
    for i in range(0, xe.shape[0], 1024):
        x, y, w = xe[i:i + 1024], ye[i:i + 1024], we[i:i + 1024]
        with torch.autocast("cuda", torch.bfloat16):
            lg = model(x).float()
        l = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view_as(w)
        tot += (l * w).sum().item(); n += w.sum().item()
        ok = (lg.argmax(-1) == y) | (w == 0)
        correct += ok[:, :16].all(1).sum().item() + ok[:, 16:].all(1).sum().item()
    model.train()
    return tot / n, correct / (2 * xe.shape[0])


if args.bench:
    for s in range(20):
        x, y, w = data.batch(s, args.bs); step_fn(x, y, w)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(args.bench):
        x, y, w = data.batch(s, args.bs); step_fn(x, y, w)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / args.bench
    print(f"mode={args.mode} compile={args.compile} bs={args.bs}: {dt*1e3:.2f} ms/step, {args.bs*32/dt/1e6:.2f} M tok/s, peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB (reserved {torch.cuda.max_memory_reserved()/2**20:.0f})")
    sys.exit()

log = {"args": vars(args), "train": [], "eval": []}
t0 = time.perf_counter()
for s in range(args.steps):
    if args.graph:
        lr_t.fill_(lr_at(s))
    else:
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
    x, y, w = data.batch(s, args.bs)
    loss = step_fn(x, y, w)
    if s % 50 == 0:
        log["train"].append((s, loss.item()))
    if (s + 1) % args.eval_every == 0 or s == args.steps - 1:
        el, acc = evaluate()
        log["eval"].append((s + 1, el, acc))
        print(f"step {s+1} train {loss.item():.4f} eval {el:.4f} acc {acc*100:.2f}% t={time.perf_counter()-t0:.0f}s", flush=True)
os.makedirs("runs", exist_ok=True)
json.dump(log, open(f"runs/ref_{args.mode}{args.tag}.json", "w"))
