"""Throughput of individual instructions on sm_120 (per SM per clock), independent chains x8."""
import torch
from nvrtc_util import Module, launch

BODY = {
    "fadd (baseline)": ("float", "x[j] = x[j] + 1.0001f;"),
    "cvt e2m1x2.f32": ("u32", 'unsigned short r; asm volatile("{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t;}" : "=h"(r) : "f"(__uint_as_float(x[j])), "f"(__uint_as_float(x[j] ^ 1))); x[j] += r;'),
    "cvt e4m3x2.f32": ("u32", 'unsigned short r; asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(__uint_as_float(x[j])), "f"(__uint_as_float(x[j] ^ 1))); x[j] += r;'),
    "cvt f16x2.e4m3x2": ("u32", 'u32 r; asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)x[j])); x[j] += r;'),
    "cvt bf16x2.f32 pack": ("u32", 'u32 r; asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(__uint_as_float(x[j])), "f"(__uint_as_float(x[j] ^ 1))); x[j] += r;'),
    "movmatrix": ("u32", 'u32 r; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(r) : "r"(x[j])); x[j] += r;'),
    "shfl.xor": ("u32", "x[j] += __shfl_xor_sync(0xffffffffu, x[j], 1);"),
    "rcp.approx (MUFU)": ("float", 'float r; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x[j])); x[j] = r + 1.f;'),
    "tanh.approx.f16x2": ("u32", 'u32 r; asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x[j])); x[j] += r;'),
    "st.shared.b16": ("u32", 'asm volatile("st.shared.b16 [%0], %1;" :: "r"((u32)(threadIdx.x * 2 + j * 512)), "h"((unsigned short)x[j])); x[j] += 1;'),
}
out = torch.zeros(1 << 20, device="cuda")
for name, (ty, body) in BODY.items():
    init = "x[j] = threadIdx.x + j;" if ty == "u32" else "x[j] = (float)(threadIdx.x + j) * 1e-3f;"
    src = f'''typedef unsigned u32;
extern "C" __global__ void k(float* o, int iters) {{
  __shared__ unsigned char smem[8192];
  {ty} x[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) {{ {init} }}
  for (int i = 0; i < iters; i++) {{
    #pragma unroll
    for (int j = 0; j < 8; j++) {{ {body} }}
  }}
  float s = 0; for (int j = 0; j < 8; j++) s += (float)x[j];
  o[blockIdx.x * blockDim.x + threadIdx.x] = s + smem[threadIdx.x];
}}'''
    m = Module(src); f = m.fn("k")
    blocks, threads, iters = 26 * 4, 256, 2000
    launch(f, (blocks,), (threads,), [out, 10]); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); launch(f, (blocks,), (threads,), [out, iters]); e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)
    ops = blocks * threads * iters * 8
    # assume ~2.5 GHz boost; report per-SM per-clock too
    per_ns = ops / (ms * 1e6)
    print(f"{name:22s} {per_ns:8.1f} thread-ops/ns  = {per_ns / 26 / 2.5:6.1f} /clk/SM (@2.5GHz)")
