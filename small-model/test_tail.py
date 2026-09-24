import torch, torch.nn.functional as TF, fusedops as F
from testutil import cos, rel, bench
torch.manual_seed(0)
M = 16384
x = torch.randn(M, 128, device="cuda").bfloat16()
g = 1 + 0.1 * torch.randn(128, device="cuda"); W = 0.1 * torch.randn(16, 128, device="cuda")
y = torch.randint(0, 16, (M,), device="cuda"); w = (torch.rand(M, device="cuda") < 0.35).float()
xr = x.float().requires_grad_(); gr = g.clone().requires_grad_(); Wr = W.clone().requires_grad_()
lg = TF.layer_norm(xr, (128,), gr) @ Wr.t()
loss = (TF.cross_entropy(lg, y, reduction="none") * w).sum() / w.sum(); loss.backward()
for name in ("lm_tail", "lm_tail_tc"):
    dx = torch.empty_like(x); dW = torch.zeros_like(W); dg = torch.zeros_like(g); l = torch.zeros(1, device="cuda")
    ws = w.sum().reshape(1)
    grid = F.NSM if name == "lm_tail" else 2 * F.NSM
    run = lambda: F.launch(F.fn(name), (grid,), (256,), [x, g, W, y, w, ws, dx, dW, dg, l, M])
    run(); torch.cuda.synchronize()
    print(f"{name}: loss {l.item():.5f} vs {loss.item():.5f} | dx cos {cos(dx, xr.grad):.5f} | dW cos {cos(dW, Wr.grad):.5f} | dg cos {cos(dg, gr.grad):.5f} | {bench(run):.1f} us")
