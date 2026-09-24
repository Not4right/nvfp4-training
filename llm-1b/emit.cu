// NVFP4 operand *emission* kernels for the 1B trainer (after Desktop/nvfp4: operands are produced by
// the kernel that produces the activation, not by separate quantize passes).
//
// Row operands (contraction along features): exact per-row FP32 scale rs[t] = rowamax/2688, E4M3 block
//   scale per 16, E2M1 values.  cuBLAS applies only block scales; consumers multiply by rs[t]*gw.
// Column operands (contraction along tokens, for wgrad): 16-pt random Hadamard along tokens, delayed
//   per-tensor scale amax_use (observed RHT amax of the previous microbatch x margin), recording the
//   current amax with atomicMax on float bits.
// Scales are written in the cuBLAS block-scaled swizzle (128x4 tiles, 512 B).
// Device RNG: rng[0] = SR seed, rng[1] = Hadamard sign bits (refreshed every microbatch, graph-safe).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
typedef unsigned u32;
typedef unsigned long long u64;
typedef __nv_bfloat16 bf16;
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
// SR onto the E2M1 grid by mantissa dithering (nvfp4/wg16.cu sr2): |v|<1 lifted into [1,2)
__device__ __forceinline__ float sr2(float v, u32 r22) {
  float a = fabsf(v), s = a < 1.f ? 1.f : 0.f;
  float b = a + s;
  float q = __uint_as_float((__float_as_uint(b) + r22) & 0xFFC00000u) - s;
  return __uint_as_float(__float_as_uint(fminf(q, 6.f)) | (__float_as_uint(v) & 0x80000000u));
}
__device__ __forceinline__ u32 rng_next(u32& s) { s = s * 1664525u + 1013904223u; return s ^ (s >> 16); }
__device__ __forceinline__ long long sw_off(long long r, int c, int ncb) {
  return ((r >> 7) * ncb + (c >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (c & 3);
}
// 8 values -> 8 nibbles (u32), value k scaling, optional SR
__device__ __forceinline__ u32 pack8(const float* v, float k, bool sr, u32& rs) {
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = v[i] * k;
  if (sr) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) { u32 r = rng_next(rs); x[i] = sr2(x[i], (r << 11) & 0x3FF800u); x[i + 1] = sr2(x[i + 1], (r >> 5) & 0x3FF800u); }
  }
  return fp4x2(x[0], x[1]) | (fp4x2(x[2], x[3]) << 8) | (fp4x2(x[4], x[5]) << 16) | (fp4x2(x[6], x[7]) << 24);
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ void unpack8(uint4 u, float* o) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ uint4 pack8bf(const float* v) {
  uint4 u; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  return u;
}
// quantize this lane's 8 values (half of a 16-block; partner lane^1 holds the other half) and store.
// col = first feature index of the 8 (multiple of 8), row r, C = row length.
__device__ __forceinline__ void emit_row8(const float* v, float inv, bool sr, u32& rs, long long r, int col, int C,
                                          u32* __restrict__ Q, unsigned char* __restrict__ S, int lane) {
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(v[i]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
  u32 sb = e4m3_enc(m * inv * (1.f / 6.f));
  float sd = e4m3_dec(sb), k = sd > 0.f ? inv / sd : 0.f;
  Q[(r * C + col) >> 3] = pack8(v, k, sr, rs);
  if (!(lane & 1)) S[sw_off(r, col >> 4, C >> 6)] = (unsigned char)sb;
}

// ============ residual-add + RMSNorm + row-quant, warp per row, D = 2048 (64 values per lane) ============
// x_new = X (+ Y * ys[t] * ygw)          (written to Xout if Y given)
// h = x_new * rstd * w  -> row operand (Q,S,rs) ; optional H (bf16 h) and rstd out
#define RD 2048
extern "C" __global__ void __launch_bounds__(256) rms_emit(
    const bf16* __restrict__ X, const bf16* __restrict__ Y, const float* __restrict__ ys, const float* __restrict__ ygw,
    bf16* __restrict__ Xout, const bf16* __restrict__ W, bf16* __restrict__ H,
    u32* __restrict__ Q, unsigned char* __restrict__ S, float* __restrict__ rs_out, float* __restrict__ rstd_out,
    int T, float eps, int do_quant) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  long long r = (long long)blockIdx.x * 8 + warp;
  if (r >= T) return;
  float v[RD / 256][8];
  float ysc = Y ? ys[r] * (*ygw) : 0.f;
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < RD / 256; j++) {
    int c = j * 256 + lane * 8;
    unpack8(*reinterpret_cast<const uint4*>(X + r * RD + c), v[j]);
    if (Y) {
      float y[8]; unpack8(*reinterpret_cast<const uint4*>(Y + r * RD + c), y);
#pragma unroll
      for (int i = 0; i < 8; i++) v[j][i] += y[i] * ysc;
      uint4 o = pack8bf(v[j]);
      *reinterpret_cast<uint4*>(Xout + r * RD + c) = o;
      unpack8(o, v[j]);   // continue from the stored (rounded) residual so recompute is exact
    }
#pragma unroll
    for (int i = 0; i < 8; i++) ss += v[j][i] * v[j][i];
  }
  float rstd = rsqrtf(wsum(ss) * (1.f / RD) + eps);
  float am = 0.f;
#pragma unroll
  for (int j = 0; j < RD / 256; j++) {
    int c = j * 256 + lane * 8;
    float w[8]; unpack8(*reinterpret_cast<const uint4*>(W + c), w);
#pragma unroll
    for (int i = 0; i < 8; i++) { v[j][i] = v[j][i] * rstd * w[i]; am = fmaxf(am, fabsf(v[j][i])); }
    if (H) *reinterpret_cast<uint4*>(H + r * RD + c) = pack8bf(v[j]);
  }
  if (lane == 0 && rstd_out) rstd_out[r] = rstd;
  if (!do_quant) return;
  am = wmax(am);
  float inv = am > 0.f ? FP4_E4M3 / am : 0.f;
  if (lane == 0) rs_out[r] = am * (1.f / FP4_E4M3);
  u32 dummy = 0;
#pragma unroll
  for (int j = 0; j < RD / 256; j++) emit_row8(v[j], inv, false, dummy, r, j * 256 + lane * 8, RD, Q, S, lane);
}

// ============ SwiGLU on the raw gate/up GEMM output + row-quant of a, warp per row (two passes) ============
// gu raw [T, 2F] (gate | up), true value = raw * rs_in[t] * (*gw).  a = silu(g) * u.
extern "C" __global__ void __launch_bounds__(256) swiglu_emit(
    const bf16* __restrict__ GU, const float* __restrict__ rs_in, const float* __restrict__ gw, int F,
    bf16* __restrict__ A, u32* __restrict__ Q, unsigned char* __restrict__ S, float* __restrict__ rs_out,
    int T, int do_quant) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  long long r = (long long)blockIdx.x * 8 + warp;
  if (r >= T) return;
  float sc = rs_in[r] * (*gw);
  const bf16* g = GU + r * 2 * F; const bf16* u = g + F;
  float am = 0.f;
  for (int c = lane * 8; c < F; c += 256) {
    float gv[8], uv[8], a[8];
    unpack8(*reinterpret_cast<const uint4*>(g + c), gv); unpack8(*reinterpret_cast<const uint4*>(u + c), uv);
#pragma unroll
    for (int i = 0; i < 8; i++) { float x = gv[i] * sc; a[i] = x / (1.f + __expf(-x)) * (uv[i] * sc); am = fmaxf(am, fabsf(a[i])); }
    if (A) *reinterpret_cast<uint4*>(A + r * F + c) = pack8bf(a);
  }
  if (!do_quant) return;
  am = wmax(am);
  float inv = am > 0.f ? FP4_E4M3 / am : 0.f;
  if (lane == 0) rs_out[r] = am * (1.f / FP4_E4M3);
  u32 dummy = 0;
  for (int c = lane * 8; c < F; c += 256) {
    float gv[8], uv[8], a[8];
    unpack8(*reinterpret_cast<const uint4*>(g + c), gv); unpack8(*reinterpret_cast<const uint4*>(u + c), uv);
#pragma unroll
    for (int i = 0; i < 8; i++) { float x = gv[i] * sc; a[i] = x / (1.f + __expf(-x)) * (uv[i] * sc); }
    emit_row8(a, inv, false, dummy, r, c, F, Q, S, lane);
  }
}

// ============ single-pass register versions: NW warps per row, NG groups of 8 per lane ============
// row length C = NW * 32 * 8 * NG.  Values stay in registers between the amax and the quantization.
template <int NW, int NG>
__device__ __forceinline__ float row_amax_nw(float am, float* red) {
  am = wmax(am);
  if (NW == 1) return am;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, sub = warp % NW;
  if (lane == 0) red[warp] = am;
  __syncthreads();
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < NW; i++) m = fmaxf(m, red[warp - sub + i]);
  return m;
}
template <int NW, int NG>
__device__ __forceinline__ void swiglu_reg(const bf16* __restrict__ GU, const float* __restrict__ rs_in, const float* __restrict__ gw,
                                           bf16* __restrict__ A, u32* __restrict__ Q, unsigned char* __restrict__ S,
                                           float* __restrict__ rs_out, int T, int do_quant) {
  __shared__ float red[8];
  const int F = NW * 256 * NG;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, sub = warp % NW;
  long long r = (long long)blockIdx.x * (8 / NW) + warp / NW;
  bool ok = r < T;
  float sc = ok ? rs_in[r] * (*gw) : 0.f;
  float a[NG][8], am = 0.f;
#pragma unroll
  for (int j = 0; j < NG; j++) {
    int c = (j * NW + sub) * 256 + lane * 8;
    if (ok) {
      float gv[8], uv[8];
      unpack8(*reinterpret_cast<const uint4*>(GU + r * 2 * F + c), gv);
      unpack8(*reinterpret_cast<const uint4*>(GU + r * 2 * F + F + c), uv);
#pragma unroll
      for (int i = 0; i < 8; i++) { float x = gv[i] * sc; a[j][i] = x / (1.f + __expf(-x)) * (uv[i] * sc); am = fmaxf(am, fabsf(a[j][i])); }
      if (A) *reinterpret_cast<uint4*>(A + r * F + c) = pack8bf(a[j]);
    }
  }
  if (!do_quant) return;
  am = row_amax_nw<NW, NG>(am, red);
  if (!ok) return;
  float inv = am > 0.f ? FP4_E4M3 / am : 0.f;
  if (lane == 0 && sub == 0) rs_out[r] = am * (1.f / FP4_E4M3);
  u32 dummy = 0;
#pragma unroll
  for (int j = 0; j < NG; j++) emit_row8(a[j], inv, false, dummy, r, (j * NW + sub) * 256 + lane * 8, F, Q, S, lane);
}
// F = 5632 = 2 warps x 11 groups
extern "C" __global__ void __launch_bounds__(256) swiglu_emit_5632(
    const bf16* __restrict__ GU, const float* __restrict__ rs_in, const float* __restrict__ gw,
    bf16* __restrict__ A, u32* __restrict__ Q, unsigned char* __restrict__ S, float* __restrict__ rs_out, int T, int do_quant) {
  swiglu_reg<2, 11>(GU, rs_in, gw, A, Q, S, rs_out, T, do_quant);
}

template <int NW, int NG>
__device__ __forceinline__ void row_reg(const bf16* __restrict__ X, long long ldx, u32* __restrict__ Q, unsigned char* __restrict__ S,
                                        float* __restrict__ rs_out, int T, int sr, const u32* __restrict__ rng, u32 salt) {
  __shared__ float red[8];
  const int C = NW * 256 * NG;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, sub = warp % NW;
  long long r = (long long)blockIdx.x * (8 / NW) + warp / NW;
  bool ok = r < T;
  float v[NG][8], am = 0.f;
#pragma unroll
  for (int j = 0; j < NG; j++) {
    if (ok) {
      unpack8(*reinterpret_cast<const uint4*>(X + r * ldx + (j * NW + sub) * 256 + lane * 8), v[j]);
#pragma unroll
      for (int i = 0; i < 8; i++) am = fmaxf(am, fabsf(v[j][i]));
    }
  }
  am = row_amax_nw<NW, NG>(am, red);
  if (!ok) return;
  float inv = am > 0.f ? FP4_E4M3 / am : 0.f;
  if (lane == 0 && sub == 0) rs_out[r] = am * (1.f / FP4_E4M3);
  u32 rs = hash32((sr ? rng[0] : 0u) ^ salt ^ hash32((u32)(r * 128 + sub * 32 + lane)));
#pragma unroll
  for (int j = 0; j < NG; j++) emit_row8(v[j], inv, sr != 0, rs, r, (j * NW + sub) * 256 + lane * 8, C, Q, S, lane);
}
#define ROWK(NAME, NW, NG) \
  extern "C" __global__ void __launch_bounds__(256) NAME(const bf16* __restrict__ X, long long ldx, u32* __restrict__ Q, \
      unsigned char* __restrict__ S, float* __restrict__ rs_out, int T, int sr, const u32* __restrict__ rng, u32 salt) { \
    row_reg<NW, NG>(X, ldx, Q, S, rs_out, T, sr, rng, salt); }
ROWK(row_emit_2048, 1, 8)
ROWK(row_emit_3072, 1, 12)
ROWK(row_emit_11264, 4, 11)

// ============ generic row-quant (exact per-row scale), warp per row, two passes ============
// X [T, C] bf16 with row stride ldx.  sr: stochastic rounding (gradients).
extern "C" __global__ void __launch_bounds__(256) row_emit(
    const bf16* __restrict__ X, long long ldx, int C, u32* __restrict__ Q, unsigned char* __restrict__ S,
    float* __restrict__ rs_out, int T, int sr, const u32* __restrict__ rng, u32 salt) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  long long r = (long long)blockIdx.x * 8 + warp;
  if (r >= T) return;
  const bf16* x = X + r * ldx;
  float am = 0.f;
  for (int c = lane * 8; c < C; c += 256) {
    float v[8]; unpack8(*reinterpret_cast<const uint4*>(x + c), v);
#pragma unroll
    for (int i = 0; i < 8; i++) am = fmaxf(am, fabsf(v[i]));
  }
  am = wmax(am);
  float inv = am > 0.f ? FP4_E4M3 / am : 0.f;
  if (lane == 0) rs_out[r] = am * (1.f / FP4_E4M3);
  u32 rs = hash32((sr ? rng[0] : 0u) ^ salt ^ hash32((u32)(r * 32 + lane)));
  for (int c = lane * 8; c < C; c += 256) {
    float v[8]; unpack8(*reinterpret_cast<const uint4*>(x + c), v);
    emit_row8(v, inv, sr != 0, rs, r, c, C, Q, S, lane);
  }
}

// ============ column operand: RHT-16 along tokens + quant (delayed per-tensor scale) ============
// X [M, C] bf16 -> Q [C, M/2], S swizzled [C, M/16].  CTA: 64 columns x 256 tokens; thread = (column,
// 64-token group): 4 Hadamard blocks, writes 32 contiguous bytes + 4 contiguous scale bytes.
// amax_use: RHT-domain amax used for the scale; amax_cur: observed RHT amax (float bits, atomicMax).
extern "C" __global__ void __launch_bounds__(256) col_emit(
    const bf16* __restrict__ X, int M, int C, uint4* __restrict__ Q, u32* __restrict__ S,
    const float* __restrict__ amax_use, unsigned* __restrict__ amax_cur, int sr, const u32* __restrict__ rng, u32 salt) {
  __shared__ bf16 tile[256][64 + 8];
  int c0 = blockIdx.x * 64, m0 = blockIdx.y * 256, tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < 8; i++) {           // 256 rows x 64 cols = 2048 uint4 chunks of 8 bf16
    int id = tid + i * 256, row = id >> 3, ch = (id & 7) * 8;
    *reinterpret_cast<uint4*>(&tile[row][ch]) = *reinterpret_cast<const uint4*>(X + (long long)(m0 + row) * C + c0 + ch);
  }
  __syncthreads();
  int col = tid & 63, grp = tid >> 6;     // 4 groups x 64 tokens
  u32 signs = rng[1];
  float au = *amax_use;
  float inv = au > 0.f ? FP4_E4M3 / au : 0.f;
  u32 rs = hash32((sr ? rng[0] : 0u) ^ salt ^ hash32((u32)((c0 + col) * 4099 + blockIdx.y * 4 + grp)));
  float mx = 0.f;
  u32 words[8], sbytes = 0;
#pragma unroll
  for (int b = 0; b < 4; b++) {
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
      float v = __bfloat162float(tile[grp * 64 + b * 16 + i][col]);
      x[i] = (signs >> i & 1) ? -v : v;
    }
#pragma unroll
    for (int h = 1; h < 16; h <<= 1)
#pragma unroll
      for (int i = 0; i < 16; i++)
        if (!(i & h)) { float a = x[i], bb = x[i + h]; x[i] = a + bb; x[i + h] = a - bb; }
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) { x[i] *= 0.25f; m = fmaxf(m, fabsf(x[i])); }
    mx = fmaxf(mx, m);
    u32 sb = e4m3_enc(m * inv * (1.f / 6.f));
    float sd = e4m3_dec(sb), k = sd > 0.f ? inv / sd : 0.f;
    words[2 * b] = pack8(x, k, sr != 0, rs);
    words[2 * b + 1] = pack8(x + 8, k, sr != 0, rs);
    sbytes |= sb << (8 * b);
  }
  long long c = c0 + col;
  int mb64 = (m0 + grp * 64) / 64;       // index of this 64-token group along M
  uint4* qp = Q + (c * (M / 2)) / 16 + mb64 * 2;
  qp[0] = make_uint4(words[0], words[1], words[2], words[3]);
  qp[1] = make_uint4(words[4], words[5], words[6], words[7]);
  // 4 scale bytes for k-blocks 4*mb64..+3 are contiguous in the swizzle (c%4 = k%4)
  S[sw_off(c, mb64 * 4, M / 64) >> 2] = sbytes;
  mx = wmax(mx);
  if ((tid & 31) == 0) atomicMax(amax_cur, __float_as_uint(mx));
}

// ============ momentum accumulate: M (bf16) += alpha * scale * G (bf16), stochastic rounding ============
extern "C" __global__ void mom_acc(bf16* __restrict__ Mo, const bf16* __restrict__ G, const float* __restrict__ alpha,
                                   float scale, long long n, const u32* __restrict__ rng, u32 salt) {
  float a = (*alpha) * scale;
  long long n8 = n / 8;
  u32 seed = rng[0] ^ salt;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n8; i += (long long)gridDim.x * blockDim.x) {
    float m[8], g[8];
    unpack8(reinterpret_cast<const uint4*>(Mo)[i], m); unpack8(reinterpret_cast<const uint4*>(G)[i], g);
    u32 h = hash32(seed ^ hash32((u32)i));
    uint4 o; u32* ow = reinterpret_cast<u32*>(&o);
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      h = h * 1664525u + 1013904223u;
      float x0 = m[j] + a * g[j], x1 = m[j + 1] + a * g[j + 1];
      u32 b0 = (__float_as_uint(x0) + (h & 0xFFFFu)) >> 16, b1 = (__float_as_uint(x1) + (h >> 16)) >> 16;
      ow[j / 2] = b0 | (b1 << 16);
    }
    reinterpret_cast<uint4*>(Mo)[i] = o;
  }
}

// ============ amax of a bf16 tensor (float bits atomicMax; *out zeroed by caller) ============
extern "C" __global__ void amax_bf16(const bf16* __restrict__ X, long long n, float* out) {
  float m = 0.f;
  const uint4* p = reinterpret_cast<const uint4*>(X);
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n / 8; i += (long long)gridDim.x * blockDim.x) {
    float v[8]; unpack8(p[i], v);
#pragma unroll
    for (int j = 0; j < 8; j++) m = fmaxf(m, fabsf(v[j]));
  }
  m = wmax(m);
  if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<unsigned*>(out), __float_as_uint(m));
}

// ============ 2D 16x16 weight quantizer (W and W^T with identical values), warp per block ============
extern "C" __global__ void quant_w2d(const bf16* __restrict__ W, u32* __restrict__ Q, unsigned char* __restrict__ S,
                                     u32* __restrict__ QT, unsigned char* __restrict__ ST,
                                     const float* __restrict__ amax, int N, int K) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  int nb = N / 16, kb = K / 16;
  if (warp >= nb * kb) return;
  int bn = warp / kb, bk = warp % kb;
  float am = *amax, inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  int r = lane >> 1, c0 = (lane & 1) * 8;
  float x[8], m = 0.f;
  unpack8(*reinterpret_cast<const uint4*>(W + (long long)(bn * 16 + r) * K + bk * 16 + c0), x);
#pragma unroll
  for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(x[i]));
  m = wmax(m);
  u32 sb = e4m3_enc(m * inv_gs * (1.f / 6.f));
  float sd = e4m3_dec(sb), k = sd > 0.f ? inv_gs / sd : 0.f;
  u32 w = 0; unsigned char nib[8];
#pragma unroll
  for (int i = 0; i < 8; i += 2) { u32 byte = fp4x2(x[i] * k, x[i + 1] * k); w |= byte << (i * 4); nib[i] = byte & 15; nib[i + 1] = byte >> 4; }
  Q[((long long)(bn * 16 + r) * K + bk * 16 + c0) / 8] = w;
  if ((lane & 1) == 0) S[sw_off(bn * 16 + r, bk, kb / 4)] = (unsigned char)sb;
  __shared__ unsigned char tile[8][16][16];
  int wi = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < 8; i++) tile[wi][r][c0 + i] = nib[i];
  __syncwarp();
  u32 wt = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) wt |= (u32)tile[wi][c0 + i][r] << (i * 4);
  QT[((long long)(bk * 16 + r) * N + bn * 16 + c0) / 8] = wt;
  if ((lane & 1) == 0) ST[sw_off(bk * 16 + r, bn, nb / 4)] = (unsigned char)sb;
}
