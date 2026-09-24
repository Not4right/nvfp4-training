"""Run each fused kernel once on realistic shapes (for Nsight Compute)."""
import torch, fusedops as F, fp4ops
torch.manual_seed(0)
dev, M = "cuda", 16384
wq = torch.randn(384, 128, device=dev) / 128 ** .5; wo = torch.randn(128, 128, device=dev) / 128 ** .5
w1 = torch.randn(512, 128, device=dev) / 128 ** .5; w2 = torch.randn(128, 512, device=dev) / 512 ** .5
Wq, Wo = F.PackedW(wq, ffN=True, rowperm=F.qkv_perm()), F.PackedW(wo)
W1, W2 = F.PackedW(w1, ffN=True), F.PackedW(w2, ffK=True)
x = torch.randn(M, 128, device=dev).bfloat16(); lnw = torch.ones(128, device=dev)
dy = (torch.randn(M, 128, device=dev) * 1e-3).bfloat16(); dl = torch.zeros(128, device=dev)
rng = torch.tensor([1, 2], dtype=torch.int32, device=dev)
mb = {k: F.WgradBuf(n, M) for k, n in (("dy", 128), ("a", 512), ("du", 512), ("h", 128))}
ab = {k: F.WgradBuf(n, M) for k, n in (("dx", 128), ("o", 128), ("dq", 384), ("h", 128))}
au4, ac4 = torch.full((4,), 0.01, device=dev), torch.zeros(4, dtype=torch.int32, device=dev)
au5, ac5 = torch.full((5,), 0.01, device=dev), torch.zeros(5, dtype=torch.int32, device=dev)
for _ in range(2):
    x1 = F.attn_fwd(x, lnw, Wq, Wo)
    y = F.mlp_fwd(x1, lnw, W1, W2)
    dx1 = F.mlp_bwd(x1, dy, lnw, dl, W1, W2, mb, au4, ac4, rng)
    dx = F.attn_bwd(x, dx1, lnw, dl, Wq, Wo, ab, au5, ac5, rng)
    Q = fp4ops.Q
    fp4ops.gemm(Q(mb["dy"].d, mb["dy"].s, au4[0:1]), Q(mb["a"].d, mb["a"].s, au4[1:2]), splitk=16)
torch.cuda.synchronize()
