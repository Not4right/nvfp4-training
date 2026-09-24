"""Peak throughput: FP4 block-scaled mma vs BF16 mma (register-only loops), plus DRAM bandwidth."""
import torch, time
from nvrtc_util import Module, launch

SRC = r"""
extern "C" __global__ void fp4_loop(float* out, int iters) {
  unsigned a0=threadIdx.x*0x11111111u, a1=a0^0x22222222u, a2=a0^0x33333333u, a3=a0^0x44444444u, b0=a0, b1=a1;
  float c[4][4] = {};
  unsigned sa = 0x38383838u, sb = 0x38383838u;
  for (int i = 0; i < iters; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++)
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%11}, %12, {%11,%11};\n"
      : "+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"h"((unsigned short)0),"r"(sb));
  }
  float s = 0; for (int j=0;j<4;j++) s += c[j][0]+c[j][1]+c[j][2]+c[j][3];
  out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}
extern "C" __global__ void bf16_loop(float* out, int iters) {
  unsigned a0=threadIdx.x*0x00010001u, a1=a0+1, a2=a0+2, a3=a0+3, b0=a0, b1=a1;
  float c[4][4] = {};
  for (int i = 0; i < iters; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  float s = 0; for (int j=0;j<4;j++) s += c[j][0]+c[j][1]+c[j][2]+c[j][3];
  out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}
extern "C" __global__ void fp8_loop(float* out, int iters) {
  unsigned a0=threadIdx.x*0x01010101u, a1=a0+1, a2=a0+2, a3=a0+3, b0=a0, b1=a1;
  float c[4][4] = {};
  for (int i = 0; i < iters; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  float s = 0; for (int j=0;j<4;j++) s += c[j][0]+c[j][1]+c[j][2]+c[j][3];
  out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}
"""
m = Module(SRC)
out = torch.zeros(26 * 8 * 256, device="cuda")
for name, k in (("bf16", 16), ("fp8", 32), ("fp4", 64)):
    f = m.fn(name + "_loop")
    iters = 4000
    launch(f, (26 * 8,), (256,), [out, 10]); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); launch(f, (26 * 8,), (256,), [out, iters]); e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)
    flops = 26 * 8 * 8 * iters * 4 * 2 * 16 * 8 * k
    print(f"{name}: {flops / ms / 1e9:.1f} TFLOPS")

x = torch.empty(512 * 2**20 // 2, dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): y.copy_(x)
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20
print(f"DRAM copy bandwidth: {2 * x.numel() * 2 / dt / 1e9:.0f} GB/s")
