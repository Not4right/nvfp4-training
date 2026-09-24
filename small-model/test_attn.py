import torch
rng3 = None
import torch.nn.functional as TF
import fusedops as F
from testutil import cos, rel, bench

torch.manual_seed(0)
dev, M = "cuda", 16384
rng3 = torch.tensor([3, 0xBEEF], dtype=torch.int32, device=dev)
wq = torch.randn(384, 128, device=dev) / 128 ** .5
wo = torch.randn(128, 128, device=dev) / 128 ** .5
Wq, Wo = F.PackedW(wq, ffN=True, rowperm=F.qkv_perm()), F.PackedW(wo)
perm = F.qkv_perm(); inv = torch.empty_like(perm); inv[perm] = torch.arange(384, device=dev)
dq_nat = lambda: Wq.dequant()[inv]
print("attn_fwd attrs", F.fattr("attn_fwd", F.ATTF_SMEM))
x = torch.randn(M, 128, device=dev).bfloat16()
lnw = 1 + 0.1 * torch.randn(128, device=dev)
x1 = F.attn_fwd(x, lnw, Wq, Wo)
h = TF.layer_norm(x.float(), (128,), lnw)
qkv = (h @ dq_nat().t()).view(-1, 32, 3, 4, 32).permute(2, 0, 3, 1, 4)
o = TF.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True).transpose(1, 2).reshape(M, 128)
ref = o @ Wo.dequant().t()
br = x1.float() - x.float()
print(f"attn_fwd branch: cos {cos(br, ref):.5f} rel {rel(br, ref):.4f}")
F.bench = bench
print(f"attn_fwd: {bench(lambda: F.attn_fwd(x, lnw, Wq, Wo)):.1f} us")
xb, wqb, wob, lb = x.clone(), wq.bfloat16(), wo.bfloat16(), lnw.bfloat16()
def tref():
    qkv = (TF.layer_norm(xb, (128,), lb) @ wqb.t()).view(-1, 32, 3, 4, 32).permute(2, 0, 3, 1, 4)
    o = TF.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True).transpose(1, 2).reshape(M, 128)
    return xb + o @ wob.t()
print(f"torch bf16 eager: {bench(tref):.1f} us")

# ---------------- backward
print("attn_bwd attrs", F.fattr("attn_bwd", F.ATTB_SMEM))
dx1 = (torch.randn(M, 128, device=dev) * 1e-3).bfloat16()
xr = x.float().requires_grad_(); lr = lnw.clone().requires_grad_()
wqr = dq_nat().clone().requires_grad_(); wor = Wo.dequant().clone().requires_grad_()
h = TF.layer_norm(xr, (128,), lr)
qkv = (h @ wqr.t()).view(-1, 32, 3, 4, 32).permute(2, 0, 3, 1, 4)
o = TF.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True).transpose(1, 2).reshape(M, 128)
(xr + o @ wor.t()).backward(dx1.float())
bufs = {k: F.WgradBuf(n, M) for k, n in (("dx", 128), ("o", 128), ("dq", 384), ("h", 128))}
amax_cur = torch.zeros(5, dtype=torch.int32, device=dev); amax_use = torch.ones(5, device=dev)
dlnw = torch.zeros(128, device=dev)
F.attn_bwd(x, dx1, lnw, dlnw, Wq, Wo, bufs, amax_use, amax_cur, torch.tensor([1, 0xBEEF], dtype=torch.int32, device=dev))
amax_use = amax_cur.view(torch.float32).clone(); print("calibrated amax", amax_use.tolist())
amax_cur.zero_(); dlnw.zero_()
dx = F.attn_bwd(x, dx1, lnw, dlnw, Wq, Wo, bufs, amax_use, amax_cur, torch.tensor([2, 0xBEEF], dtype=torch.int32, device=dev))
print(f"dx:   cos {cos(dx.float() - dx1.float(), xr.grad - dx1.float()):.5f}  (branch part)  full cos {cos(dx, xr.grad):.5f}")
print(f"dlnw: cos {cos(dlnw, lr.grad):.5f}")
import fp4ops
dWo = torch.zeros(128, 128, device=dev); dWq = torch.zeros(384, 128, device=dev)
wg = F.WGGemm([(bufs["dx"], amax_use[0:1], bufs["o"], amax_use[1:2], dWo, 16), (bufs["dq"], amax_use[2:3], bufs["h"], amax_use[3:4], dWq, 16)], M)
wg()
print(f"dWo:  cos {cos(dWo, wor.grad):.5f} rel {rel(dWo, wor.grad):.4f}")
print(f"dWqkv: cos {cos(dWq, wqr.grad):.5f} rel {rel(dWq, wqr.grad):.4f}")
print(f"attn_bwd: {bench(lambda: F.attn_bwd(x, dx1, lnw, dlnw, Wq, Wo, bufs, amax_use, amax_cur, rng3)):.1f} us")
for ns in (8, 16, 32, 64):
    wgb = F.WGGemm([(bufs["dx"], amax_use[0:1], bufs["o"], amax_use[1:2], dWo, ns), (bufs["dq"], amax_use[2:3], bufs["h"], amax_use[3:4], dWq, ns)], M)
    print(f"grouped wgrad (dWo+dWqkv) nsplit={ns}: {bench(wgb):.1f} us")
