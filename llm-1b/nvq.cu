// NVFP4 quantizers for cuBLAS block-scaled GEMM (torch._scaled_mm), sm_120a.
// Data: E2M1 packed 2/byte, low nibble = lower index, K-major rows.
// Scales: UE4M3 per 16 values along K, stored in the cuBLAS/"to_blocked" swizzle:
//   tile (128 rows x 4 scale-cols) = 512 bytes, tiles row-block-major;
//   inside a tile: (r%32)*16 + ((r%128)/32)*4 + (c%4).
// Per-tensor FP32 global scale gs = amax/2688 is NOT applied by cuBLAS: callers scale the
// GEMM output by gsA*gsB (alpha).
#include <cuda_bf16.h>
typedef unsigned u32;
typedef unsigned long long u64;
#define FP4_E4M3 2688.0f

__device__ __forceinline__ u32 hash32(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__device__ __forceinline__ u32 e4m3_enc(float s) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(s));
  return r & 0xFF;
}
__device__ __forceinline__ float e4m3_dec(u32 b) {
  unsigned short h = (unsigned short)b;
  u32 r; asm("{.reg .b32 t; cvt.rn.f16x2.e4m3x2 t, %1; mov.b32 %0, t;}" : "=r"(r) : "h"(h));
  return __low2float(*reinterpret_cast<__half2*>(&r));
}
__device__ __forceinline__ u32 fp4x2(float a, float b) {
  unsigned short r;
  asm("{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t;}" : "=h"(r) : "f"(b), "f"(a));
  return r;
}
// stochastic rounding onto the E2M1 grid by mantissa dithering (cvt.rs does not exist on sm_120)
__device__ __forceinline__ float sr_fp4(float v, u32 rnd) {
  float a = fminf(fabsf(v), 6.f);
  bool sub = a < 1.f;
  float b = sub ? a + 1.f : a;
  u32 bits = (__float_as_uint(b) + (rnd & 0x3FFFFFu)) & ~0x3FFFFFu;
  float r = __uint_as_float(bits) - (sub ? 1.f : 0.f);
  return copysignf(fminf(r, 6.f), v);
}
__device__ __forceinline__ long long sw_off(long long r, int c, int ncb) {
  return ((r >> 7) * ncb + (c >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (c & 3);
}
__device__ __forceinline__ void quant16(const float* x, float inv_gs, bool sr, u32 seed, u32 idx, uint2& out, u32& sbyte) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[i]));
  u32 sb = e4m3_enc(amax * inv_gs * (1.f / 6.f));
  float sdec = e4m3_dec(sb);
  float k = sdec > 0.f ? inv_gs / sdec : 0.f;
  u32 w[2] = {0, 0};
  u32 h = sr ? hash32(seed ^ hash32(idx)) : 0u;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float a = x[i] * k, b = x[i + 1] * k;
    if (sr) {
      u32 h1 = hash32(h + i);
      a = sr_fp4(a, h1); b = sr_fp4(b, h1 >> 10 | h1 << 22);
    }
    w[i >> 3] |= fp4x2(a, b) << ((i & 7) * 4);
  }
  out = make_uint2(w[0], w[1]);
  sbyte = sb;
}
__device__ __forceinline__ void atomic_amax(float* p, float v) {
  atomicMax(reinterpret_cast<unsigned*>(p), __float_as_uint(fabsf(v)));
}

// ---- amax of a bf16 tensor (atomicMax on float bits; *out must be zeroed)
extern "C" __global__ void amax_bf16(const __nv_bfloat16* __restrict__ X, long long n, float* out) {
  float m = 0.f;
  const uint4* p = reinterpret_cast<const uint4*>(X);
  long long n8 = n / 8;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n8; i += (long long)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(h[j]); m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y))); }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ float sm[32];
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x == 0) atomic_amax(out, m);
  }
}

// ---- row quantizer: X [R, C] bf16 (row stride ldx) -> Q [R, C/2], S swizzled.  One thread per 16-block.
extern "C" __global__ void quant_rows(const __nv_bfloat16* __restrict__ X, long long ldx, uint2* __restrict__ Q,
                                      unsigned char* __restrict__ S, const float* __restrict__ amax,
                                      int R, int C, int sr, u32 seed) {
  long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int cb = C / 16;
  if (b >= (long long)R * cb) return;
  long long r = b / cb; int c = (int)(b - r * cb);
  float am = *amax;
  float inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  const uint4* p = reinterpret_cast<const uint4*>(X + r * ldx + c * 16);
  uint4 v0 = p[0], v1 = p[1];
  float x[16];
  const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f0 = __bfloat1622float2(h0[i]), f1 = __bfloat1622float2(h1[i]);
    x[2 * i] = f0.x; x[2 * i + 1] = f0.y; x[8 + 2 * i] = f1.x; x[8 + 2 * i + 1] = f1.y;
  }
  uint2 o; u32 sb;
  quant16(x, inv_gs, sr, seed, (u32)b, o, sb);
  Q[b] = o;
  S[sw_off(r, c, cb / 4)] = (unsigned char)sb;
}

// ---- transposed quantizer with 16-point random Hadamard along M (tokens):
// X [M, C] bf16 -> Q [C, M/2] (row c = column c of X), S swizzled [C, M/16].
// CTA: 64 columns x 64 rows (4 m-blocks).  256 threads: thread = (column, m-block).
// Loads coalesced along C into smem, writes 32 contiguous bytes per column.
extern "C" __global__ void __launch_bounds__(256) quant_cols_rht(
    const __nv_bfloat16* __restrict__ X, uint2* __restrict__ Q, unsigned char* __restrict__ S,
    const float* __restrict__ amax, int M, int C, u32 signs, int sr, u32 seed) {
  __shared__ __nv_bfloat16 tile[64][64 + 2];
  int c0 = blockIdx.x * 64, m0 = blockIdx.y * 64;
  int tid = threadIdx.x;
  // load 64 rows x 64 cols: each thread loads 16 elements (8 x bf16x2)
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int id = tid + i * 256;          // 0..2047 pairs
    int row = id >> 5, cp = (id & 31) * 2;
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(X + (long long)(m0 + row) * C + c0 + cp);
    tile[row][cp] = v.x; tile[row][cp + 1] = v.y;
  }
  __syncthreads();
  int col = tid & 63, mb = tid >> 6;  // 4 m-blocks of 16
  float am = *amax * 4.f;
  float inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  float x[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    float v = __bfloat162float(tile[mb * 16 + i][col]);
    x[i] = (signs >> i & 1) ? -v : v;
  }
#pragma unroll
  for (int h = 1; h < 16; h <<= 1)
#pragma unroll
    for (int i = 0; i < 16; i++)
      if (!(i & h)) { float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
#pragma unroll
  for (int i = 0; i < 16; i++) x[i] *= 0.25f;
  uint2 o; u32 sb;
  long long c = c0 + col; int gmb = m0 / 16 + mb;
  quant16(x, inv_gs, sr, seed, (u32)(c * (M / 16) + gmb), o, sb);
  Q[c * (M / 16) + gmb] = o;
  S[sw_off(c, gmb, M / 64)] = (unsigned char)sb;
}

// ---- 2D 16x16 weight quantizer: W [N, K] bf16 -> Q [N, K/2] + S swizzled [N, K/16] and the same
// values transposed QT [K, N/2] + ST swizzled [K, N/16].  One warp per 16x16 block.
extern "C" __global__ void quant_w2d(const __nv_bfloat16* __restrict__ W, u32* __restrict__ Q, unsigned char* __restrict__ S,
                                     u32* __restrict__ QT, unsigned char* __restrict__ ST,
                                     const float* __restrict__ amax, int N, int K) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  int nb = N / 16, kb = K / 16;
  if (warp >= nb * kb) return;
  int bn = warp / kb, bk = warp % kb;
  float am = *amax;
  float inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  int r = lane >> 1, c0 = (lane & 1) * 8;
  float x[8], m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { x[i] = __bfloat162float(W[(long long)(bn * 16 + r) * K + bk * 16 + c0 + i]); m = fmaxf(m, fabsf(x[i])); }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  u32 sb = e4m3_enc(m * inv_gs * (1.f / 6.f));
  float sdec = e4m3_dec(sb);
  float k = sdec > 0.f ? inv_gs / sdec : 0.f;
  u32 w = 0;
  unsigned char nib[8];
#pragma unroll
  for (int i = 0; i < 8; i += 2) { u32 byte = fp4x2(x[i] * k, x[i + 1] * k); w |= byte << (i * 4); nib[i] = byte & 15; nib[i + 1] = byte >> 4; }
  Q[((long long)(bn * 16 + r) * K + bk * 16 + c0) / 8] = w;
  if ((lane & 1) == 0) S[sw_off(bn * 16 + r, bk, kb / 4)] = (unsigned char)sb;
  __shared__ unsigned char tile[8][16][16];
  int wi = (threadIdx.x >> 5);
#pragma unroll
  for (int i = 0; i < 8; i++) tile[wi][r][c0 + i] = nib[i];
  __syncwarp();
  u32 wt = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) wt |= (u32)tile[wi][c0 + i][r] << (i * 4);
  QT[((long long)(bk * 16 + r) * N + bn * 16 + c0) / 8] = wt;
  if ((lane & 1) == 0) ST[sw_off(bk * 16 + r, bn, nb / 4)] = (unsigned char)sb;
}
