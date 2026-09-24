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

bufs = {k: F.WgradBuf(n, M) for k, n in (("dy", 128), ("a", 512), ("du", 512), ("h", 128))}
amax_cur = torch.zeros(4, dtype=torch.int32, device=dev)
amax_use = torch.full((4,), 0.01, device=dev)
dlnw = torch.zeros(128, device=dev)
