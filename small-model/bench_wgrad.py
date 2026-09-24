import torch, fp4ops
from testutil import bench
M = 16384
Q = fp4ops.Q
def mk(F, pad):
    d = torch.randint(0, 255, (F, M // 2 + pad), dtype=torch.uint8, device="cuda")[:, :M // 2]
    s = torch.full((F, M // 16 + pad), 0x38, dtype=torch.uint8, device="cuda")[:, :M // 16]
    return Q(d, s, torch.ones(1, device="cuda"))
out = torch.zeros(128, 512, device="cuda")
for pad in (0, 16, 144, 528):
  A, B = mk(128, pad), mk(512, pad)
  print("pad", pad)
  for sk in (4, 8, 16, 32):
    us = bench(lambda: fp4ops.gemm(A, B, out=out, splitk=sk))
    print("  ", f"wgrad 128x512xK={M} splitk={sk}: {us:.1f} us  {2*128*512*M/us/1e6:.1f} TFLOPS")
