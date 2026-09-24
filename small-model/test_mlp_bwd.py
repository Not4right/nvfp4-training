import torch, time
import torch.nn.functional as TF
import fusedops as F
import fp4ops
from testutil import cos, rel, bench

torch.manual_seed(0)
dev, M = "cuda", 16384
rng3 = torch.tensor([3, 0xBEEF], dtype=torch.int32, device=dev)
w1 = (torch.randn(512, 128, device=dev) / 128 ** .5)
w2 = (torch.randn(128, 512, device=dev) / 512 ** .5)
W1 = F.PackedW(w1, ffN=True); W2 = F.PackedW(w2, ffK=True)
print("mlp_bwd attrs", F.fattr("mlp_bwd", F.MLPB_SMEM), " mlp_fwd attrs", F.fattr("mlp_fwd", F.MLPF_SMEM))
x1 = torch.randn(M, 128, device=dev).bfloat16()
dy = (torch.randn(M, 128, device=dev) * 1e-3).bfloat16()
lnw = (1 + 0.1 * torch.randn(128, device=dev))

# reference with dequantized weights, exact activations
xr = x1.float().requires_grad_(); lr = lnw.clone().requires_grad_()
w1r = W1.dequant().clone().requires_grad_(); w2r = W2.dequant().clone().requires_grad_()
h = TF.layer_norm(xr, (128,), lr)
a = TF.gelu(h @ w1r.t(), approximate="tanh")
y = xr + a @ w2r.t()
y.backward(dy.float())
# amax for wgrad operands: calibrate with exact RHT-free bounds (x4 like the torch path)
bufs = {k: F.WgradBuf(n, M) for k, n in (("dy", 128), ("a", 512), ("du", 512), ("h", 128))}
amax_cur = torch.zeros(4, dtype=torch.int32, device=dev)
amax_use = torch.ones(4, device=dev)
dlnw = torch.zeros(128, device=dev)
F.mlp_bwd(x1, dy, lnw, dlnw, W1, W2, bufs, amax_use, amax_cur, torch.tensor([1, 0xBEEF], dtype=torch.int32, device=dev))   # calibration pass
amax_use = amax_cur.view(torch.float32).clone() * 1.0
print("calibrated wgrad-operand amax (dy,a,du,h):", amax_use.tolist())
amax_cur.zero_(); dlnw.zero_()
dx1 = F.mlp_bwd(x1, dy, lnw, dlnw, W1, W2, bufs, amax_use, amax_cur, torch.tensor([2, 0xBEEF], dtype=torch.int32, device=dev))
print(f"dx1:  cos {cos(dx1, xr.grad):.5f} rel {rel(dx1, xr.grad):.4f}")
print(f"dlnw: cos {cos(dlnw, lr.grad):.5f} rel {rel(dlnw, lr.grad):.4f}")
dW2 = torch.zeros(128, 512, device=dev); dW1 = torch.zeros(512, 128, device=dev)
wg = F.WGGemm([(bufs["dy"], amax_use[0:1], bufs["a"], amax_use[1:2], dW2, 16), (bufs["du"], amax_use[2:3], bufs["h"], amax_use[3:4], dW1, 16)], M)
wg()
print(f"dW2:  cos {cos(dW2, w2r.grad):.5f} rel {rel(dW2, w2r.grad):.4f}")
print(f"dW1:  cos {cos(dW1, w1r.grad):.5f} rel {rel(dW1, w1r.grad):.4f}")
us = bench(lambda: F.mlp_bwd(x1, dy, lnw, dlnw, W1, W2, bufs, amax_use, amax_cur, rng3))
print(f"mlp_bwd: {us:.1f} us")
for ns in (4, 8, 16, 32):
    wgb = F.WGGemm([(bufs["dy"], amax_use[0:1], bufs["a"], amax_use[1:2], dW2, ns), (bufs["du"], amax_use[2:3], bufs["h"], amax_use[3:4], dW1, ns)], M)
    print(f"grouped wgrad gemms (dW2+dW1) nsplit={ns}: {bench(wgb):.1f} us")
xb = x1.clone().requires_grad_(); w1b = w1.bfloat16().requires_grad_(); w2b = w2.bfloat16().requires_grad_(); lb = lnw.bfloat16().requires_grad_()
def tb():
    y = xb + TF.gelu(TF.layer_norm(xb, (128,), lb) @ w1b.t(), approximate="tanh") @ w2b.t()
    y.backward(dy)
print(f"torch bf16 eager fwd+bwd: {bench(tb):.1f} us")

for fl, name in ((1, "no emission"), (2, "no SR"), (3, "neither")):
    us = bench(lambda: F.mlp_bwd(x1, dy, lnw, dlnw, W1, W2, bufs, amax_use, amax_cur, rng3, flags=fl))
    print(f"mlp_bwd [{name}]: {us:.1f} us")
