// FP8 (e4m3) flash attention for sm_120a, head dim 128, causal, GQA.
//
// Operands (produced by attn_prep / v_prep):
//   Q8 [B,H,T,128], K8 [B,KV,T,128]   e4m3, unscaled: after QK-norm |q|,|k| <= sqrt(128)*|gain|, inside e4m3 range
//   VT8 [B,KV,128,T]  V transposed, e4m3, per-(b,kv,channel) scale sv = amax/448, and *permuted inside each
//                     32-token block* (Desktop/nvfp4 WG16 idea) so that the fp32 accumulator of S = QK^T,
//                     converted in registers, is directly the A fragment of O += P V with no shuffles:
//                     phys k p (0..31) <- logical token L(p): p = 4t+j (t<4):  j<2 -> 2t+j,  j>=2 -> 8+2t+j-2
//                                                             p = 16+4t+j:     same + 16
//   P is quantized as e4m3(P*448) (P <= 1 after the running-max shift).
// Output O bf16 [B,T,H,128] (token-major, feeds the projection GEMM), LSE2 [B,H,T] = m + log2(l) in
// log2 units of the scaled scores (used by the backward).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
typedef unsigned u32;
typedef unsigned char u8;
typedef __nv_bfloat16 bf16;
#define HD 128
#define LOG2E 1.4426950408889634f
#define INFINITY __int_as_float(0x7f800000)

__device__ __forceinline__ u32 f2e4m3x4(float a, float b, float c, float d) {   // bytes a,b,c,d (a lowest)
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return (u32)lo | ((u32)hi << 16);
}
__device__ __forceinline__ void mma8(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
  u32 sa = (u32)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ float qmax(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float qsum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
// logical token (0..31) held at physical k position p of a 32-block
__device__ __forceinline__ int perm32(int p) {
  int hi = p & 16, q = p & 15, t = q >> 2, j = q & 3;
  return hi + (j < 2 ? 2 * t + j : 8 + 2 * t + (j - 2));
}

// ============================ prep: QK-norm + RoPE -> Q8, K8 ; V amax ============================
// qkv: true-scaled bf16 [B*T, (H+2KV)*128].  One warp per (token, head slot); lane holds dims 4l..4l+3.
extern "C" __global__ void attn_prep(const bf16* __restrict__ qkv, const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                                     const bf16* __restrict__ cosb, const bf16* __restrict__ sinb,
                                     u8* __restrict__ Q8, u8* __restrict__ K8, unsigned* __restrict__ vamax,
                                     int B, int T, int H, int KV, float eps) {
  int lane = threadIdx.x & 31;
  long long wid = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int S = H + 2 * KV;
  if (wid >= (long long)B * T * S) return;
  int slot = (int)(wid % S); long long tok = wid / S;
  int b = (int)(tok / T), t = (int)(tok % T);
  const bf16* src = qkv + tok * S * HD + slot * HD + lane * 4;
  float x[4];
  {
    uint2 u = *reinterpret_cast<const uint2*>(src);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
    float2 a = __bfloat1622float2(h[0]), c = __bfloat1622float2(h[1]);
    x[0] = a.x; x[1] = a.y; x[2] = c.x; x[3] = c.y;
  }
  if (slot >= H + KV) {           // V: record per-(b, kv, channel) amax
    int kv = slot - H - KV;
#pragma unroll
    for (int i = 0; i < 4; i++) atomicMax(vamax + ((b * KV + kv) * HD + lane * 4 + i), __float_as_uint(fabsf(x[i])));
    return;
  }
  const bf16* gw = slot < H ? qn : kn;
  float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  float r = rsqrtf(ss * (1.f / HD) + eps);
  // RoPE (NeoX halves): dim i < 64 pairs with i+64 = lane ^ 16
  float y[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int d = lane * 4 + i;
    float v = x[i] * r * __bfloat162float(gw[d]);
    float pv = __shfl_xor_sync(0xffffffffu, v, 16);
    int f = d & 63;
    float c = __bfloat162float(cosb[t * 64 + f]), s = __bfloat162float(sinb[t * 64 + f]);
    y[i] = d < 64 ? v * c - pv * s : v * c + pv * s;
  }
  u32 w = f2e4m3x4(y[0], y[1], y[2], y[3]);
  u8* dst = slot < H ? Q8 + (((long long)b * H + slot) * T + t) * HD : K8 + (((long long)b * KV + (slot - H)) * T + t) * HD;
  *reinterpret_cast<u32*>(dst + lane * 4) = w;
}

// ============================ V -> VT8 (transposed, permuted, scaled) ============================
// CTA: one (b, kv) and 64 tokens.  sv_out[b,kv,d] = amax/448.
extern "C" __global__ void v_prep(const bf16* __restrict__ qkv, const unsigned* __restrict__ vamax, u8* __restrict__ VT8,
                                  float* __restrict__ sv_out, int B, int T, int H, int KV) {
  __shared__ float tile[64][HD + 1];
  int bk = blockIdx.y, b = bk / KV, kv = bk % KV, t0 = blockIdx.x * 64;
  int S = H + 2 * KV;
  for (int i = threadIdx.x; i < 64 * HD; i += blockDim.x) {
    int tt = i / HD, d = i % HD;
    tile[tt][d] = __bfloat162float(qkv[((long long)b * T + t0 + tt) * S * HD + (H + KV + kv) * HD + d]);
  }
  __syncthreads();
  // thread -> (d, 4 physical positions)
  for (int i = threadIdx.x; i < HD * 16; i += blockDim.x) {
    int d = i / 16, p4 = (i % 16) * 4;                 // 64 positions per row, 4 per thread
    float am = __uint_as_float(vamax[(b * KV + kv) * HD + d]);
    float inv = am > 0.f ? 448.f / am : 0.f;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int p = p4 + j, blk = p >> 5;
      v[j] = tile[blk * 32 + perm32(p & 31)][d] * inv;
    }
    *reinterpret_cast<u32*>(VT8 + (((long long)b * KV + kv) * HD + d) * T + t0 + p4) = f2e4m3x4(v[0], v[1], v[2], v[3]);
    if (blockIdx.x == 0 && p4 == 0) sv_out[(b * KV + kv) * HD + d] = am / 448.f;
  }
}

// ============================ backward prep ============================
// D[b,h,t] = sum_d dO*O ; dO amax per (b, kv group, channel) (shared by the H/KV heads of a group, so the
// backward accumulates dV over those heads with one scale).  Warp per (b, h, 64-token chunk); lane = 4 dims.
extern "C" __global__ void bwd_prep1(const bf16* __restrict__ dO, const bf16* __restrict__ O, float* __restrict__ D,
                                     unsigned* __restrict__ doamax, int B, int T, int H, int KV) {
  int lane = threadIdx.x & 31;
  long long wid = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int nch = T / 64;
  if (wid >= (long long)B * H * nch) return;
  int ch = (int)(wid % nch); int bh = (int)(wid / nch), b = bh / H, h = bh % H;
  float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
  for (int tt = 0; tt < 64; tt++) {
    int t = ch * 64 + tt;
    long long row = ((long long)b * T + t) * H + h;
    uint2 u = *reinterpret_cast<const uint2*>(dO + row * HD + lane * 4);
    uint2 w = *reinterpret_cast<const uint2*>(O + row * HD + lane * 4);
    const __nv_bfloat162* a = reinterpret_cast<const __nv_bfloat162*>(&u);
    const __nv_bfloat162* c = reinterpret_cast<const __nv_bfloat162*>(&w);
    float2 a0 = __bfloat1622float2(a[0]), a1 = __bfloat1622float2(a[1]), c0 = __bfloat1622float2(c[0]), c1 = __bfloat1622float2(c[1]);
    float sdot = a0.x * c0.x + a0.y * c0.y + a1.x * c1.x + a1.y * c1.y;
#pragma unroll
    for (int o = 16; o; o >>= 1) sdot += __shfl_xor_sync(0xffffffffu, sdot, o);
    if (lane == 0) D[((long long)b * H + h) * T + t] = sdot;
    m0 = fmaxf(m0, fabsf(a0.x)); m1 = fmaxf(m1, fabsf(a0.y)); m2 = fmaxf(m2, fabsf(a1.x)); m3 = fmaxf(m3, fabsf(a1.y));
  }
  unsigned* am = doamax + ((long long)b * KV + h / (H / KV)) * HD + lane * 4;
  atomicMax(am + 0, __float_as_uint(m0)); atomicMax(am + 1, __float_as_uint(m1));
  atomicMax(am + 2, __float_as_uint(m2)); atomicMax(am + 3, __float_as_uint(m3));
}
// transposes for the backward: CTA = (row-group n of NR heads, 64 tokens)
//   mode 0: dO bf16 [B,T,H,128] -> dOT8 [B,H,128,T] permuted, scaled by 448/amax(b,h,d); sdo = amax/448
//   mode 1: Q8 [B,H,T,128] -> QT8 [B,H,128,T] permuted
//   mode 2: K8 [B,KV,T,128] -> KT8 [B,KV,128,T] natural order
extern "C" __global__ void bwd_transpose(const void* __restrict__ src, u8* __restrict__ dst, const unsigned* __restrict__ amax,
                                         float* __restrict__ sdo, int B, int T, int NH, int mode, int G) {
  __shared__ float tile[64][HD + 1];
  int bh = blockIdx.y, b = bh / NH, h = bh % NH, t0 = blockIdx.x * 64;
  for (int i = threadIdx.x; i < 64 * HD; i += blockDim.x) {
    int tt = i / HD, d = i % HD;
    float v;
    if (mode == 0) v = __bfloat162float(reinterpret_cast<const bf16*>(src)[(((long long)b * T + t0 + tt) * NH + h) * HD + d]);
    else {
      u8 q = reinterpret_cast<const u8*>(src)[(((long long)b * NH + h) * T + t0 + tt) * HD + d];
      u32 x2; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(x2) : "h"((unsigned short)q));
      v = __half2float(__ushort_as_half((unsigned short)(x2 & 0xFFFF)));
    }
    tile[tt][d] = v;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < HD * 16; i += blockDim.x) {
    int d = i / 16, p4 = (i % 16) * 4;
    float inv = 1.f;
    if (mode == 0) {
      long long gi = ((long long)b * (NH / G) + h / G) * HD + d;       // scale shared by the kv group
      float am = __uint_as_float(amax[gi]);
      inv = am > 0.f ? 448.f / am : 0.f;
      if (blockIdx.x == 0 && p4 == 0 && h % G == 0) sdo[gi] = am / 448.f;
    }
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int p = p4 + j;
      int tt = mode == 2 ? p : ((p >> 5) * 32 + perm32(p & 31));
      v[j] = tile[tt][d] * inv;
    }
    *reinterpret_cast<u32*>(dst + (((long long)b * NH + h) * HD + d) * T + t0 + p4) = f2e4m3x4(v[0], v[1], v[2], v[3]);
  }
}

// ============================ backward ============================
// CTA = (b, kv head, 64 keys); 4 warps x 16 keys.  Loops over the H/KV query heads of this kv head and over
// 32-query blocks (one fp8 k-step), double buffered.  Per block and warp:
//   S^T = K Q^T (fp8)  ->  P^T = exp2(S^T*sl2 - LSE2)          dP^T = V dO^T (bf16)
//   dS^T = P^T (dP^T - D)       dV += P^T dO (fp8, P*448)       dK += dS^T Q (fp8, dS / s_ds)
//   dS -> smem (fp8);  dQ[32 x 128] += dS K (fp8) split over the 4 warps, fp32 vector atomics.
// dS scale: delayed per-layer amax (x2 margin) in dsamax_use; the observed amax goes to dsamax_cur.
#define QB 32
struct __align__(16) BwdSmem {
  u8 k8[64 * HD];            // K rows (keys) fp8, swizzled like the fwd K tile
  u8 kt8[HD * 64];           // K^T fp8 [d][keys], swizzled like the fwd VT tile
  bf16 v[64 * HD];           // V bf16 [keys][d], 16-B chunks swizzled by (row & 7)
  u8 ds[QB * 64];            // dS fp8 [q][keys], chunk ^ ((q >> 1) & 3)
  struct Stage {
    u8 q8[QB * HD];          // Q rows fp8 [q][d] swizzled (r & 7)
    u8 qt8[HD * QB];         // Q^T fp8 [d][q] (permuted), chunk ^ ((d >> 2) & 1)
    bf16 dO[QB * HD];        // dO bf16 [q][d], chunks ^ (q & 7)
    u8 dot8[HD * QB];        // dO^T fp8 [d][q] (permuted, scaled), chunk ^ ((d >> 2) & 1)
    float lse[QB];
    float dd[QB];
  } st[2];
};
#define BWD_SMEM ((int)sizeof(BwdSmem))

__device__ __forceinline__ void mma16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int swb(int row, int col) {   // byte offset in a 256-B bf16 row, 16-B chunks ^ (row & 7)
  return row * 256 + ((((col * 2) >> 4) ^ (row & 7)) << 4) + ((col * 2) & 15);
}

extern "C" __global__ void __launch_bounds__(128) attn_bwd8(
    const u8* __restrict__ Q8, const u8* __restrict__ QT8, const u8* __restrict__ K8, const u8* __restrict__ KT8,
    const bf16* __restrict__ V, long long ldv, const bf16* __restrict__ dO, const u8* __restrict__ dOT8,
    const float* __restrict__ sdo, const float* __restrict__ LSE2, const float* __restrict__ Dd,
    float* __restrict__ dQ, bf16* __restrict__ dK, bf16* __restrict__ dV,
    const float* __restrict__ dsamax_use, unsigned* __restrict__ dsamax_cur, int T, int H, int KV, float scale) {
  extern __shared__ __align__(16) u8 smraw[];
  BwdSmem& sm = *reinterpret_cast<BwdSmem*>(smraw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  int kb = gridDim.x - 1 - blockIdx.x;                   // key blocks near the start see the most queries
  int bk = blockIdx.y, b = bk / KV, kv = bk % KV, G = H / KV;
  int key0 = kb * 64;
  const float sl2 = scale * LOG2E;
  const float s_ds = dsamax_use[0] * (1.f / 448.f), inv_ds = s_ds > 0.f ? 1.f / s_ds : 0.f;
  // ---- persistent tiles: K8, KT8, V
  {
    const u8* ksrc = K8 + (((long long)b * KV + kv) * T + key0) * HD;
    for (int id = tid; id < 512; id += 128) { int r = id >> 3, c = id & 7; cp16(sm.k8 + r * HD + ((c ^ (r & 7)) << 4), ksrc + r * HD + c * 16); }
    const u8* ktsrc = KT8 + ((long long)b * KV + kv) * HD * T + key0;
    for (int id = tid; id < 512; id += 128) { int d = id >> 2, c = id & 3; cp16(sm.kt8 + d * 64 + ((c ^ ((d >> 1) & 3)) << 4), ktsrc + (long long)d * T + c * 16); }
    const bf16* vsrc = V + ((long long)b * T + key0) * ldv + kv * HD;
    for (int id = tid; id < 1024; id += 128) { int r = id >> 4, c = id & 15; cp16((u8*)sm.v + r * 256 + ((c ^ (r & 7)) << 4), vsrc + (long long)r * ldv + c * 8); }
    asm volatile("cp.async.commit_group;\n");
  }
  float dk[16][4], dv[16][4];
#pragma unroll
  for (int i = 0; i < 16; i++) { dk[i][0] = dk[i][1] = dk[i][2] = dk[i][3] = 0.f; dv[i][0] = dv[i][1] = dv[i][2] = dv[i][3] = 0.f; }
  float dsmax = 0.f;
  int nqb = T / QB, qb0 = key0 / QB;
  int niter = G * (nqb - qb0);
  auto load = [&](int s, int it) {
    int hh = kv * G + it / (nqb - qb0), qb = qb0 + it % (nqb - qb0), q0 = qb * QB;
    auto& S = sm.st[s];
    const u8* qs = Q8 + (((long long)b * H + hh) * T + q0) * HD;
    for (int id = tid; id < 256; id += 128) { int r = id >> 3, c = id & 7; cp16(S.q8 + r * HD + ((c ^ (r & 7)) << 4), qs + r * HD + c * 16); }
    const u8* qts = QT8 + ((long long)b * H + hh) * HD * T + q0;
    for (int id = tid; id < 256; id += 128) { int d = id >> 1, c = id & 1; cp16(S.qt8 + d * QB + ((c ^ ((d >> 2) & 1)) << 4), qts + (long long)d * T + c * 16); }
    const bf16* dos = dO + (((long long)b * T + q0) * H + hh) * HD;
    for (int id = tid; id < 512; id += 128) { int r = id >> 4, c = id & 15; cp16((u8*)S.dO + r * 256 + ((c ^ (r & 7)) << 4), dos + (long long)r * H * HD + c * 8); }
    const u8* dts = dOT8 + ((long long)b * H + hh) * HD * T + q0;
    for (int id = tid; id < 256; id += 128) { int d = id >> 1, c = id & 1; cp16(S.dot8 + d * QB + ((c ^ ((d >> 2) & 1)) << 4), dts + (long long)d * T + c * 16); }
    if (tid < 8) cp16(S.lse + tid * 4, LSE2 + ((long long)b * H + hh) * T + q0 + tid * 4);
    else if (tid < 16) cp16(S.dd + (tid - 8) * 4, Dd + ((long long)b * H + hh) * T + q0 + (tid - 8) * 4);
    asm volatile("cp.async.commit_group;\n");
  };
  load(0, 0);
  for (int it = 0; it < niter; it++) {
    int s = it & 1;
    if (it + 1 < niter) { load(s ^ 1, it + 1); asm volatile("cp.async.wait_group 1;\n"); }
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    int hh = kv * G + it / (nqb - qb0), qb = qb0 + it % (nqb - qb0), q0 = qb * QB;
    auto& S = sm.st[s];
    int kr = warp * 16 + g;                                   // local key rows kr, kr+8
    // ---- S^T = K Q^T  [16 keys x 32 q]
    float st[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      st[j][0] = st[j][1] = st[j][2] = st[j][3] = 0.f;
      int qr = j * 8 + g;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        u32 a0 = *reinterpret_cast<const u32*>(sm.k8 + kr * HD + (((2 * ks) ^ (kr & 7)) << 4) + 4 * t);
        u32 a1 = *reinterpret_cast<const u32*>(sm.k8 + (kr + 8) * HD + (((2 * ks) ^ ((kr + 8) & 7)) << 4) + 4 * t);
        u32 a2 = *reinterpret_cast<const u32*>(sm.k8 + kr * HD + (((2 * ks + 1) ^ (kr & 7)) << 4) + 4 * t);
        u32 a3 = *reinterpret_cast<const u32*>(sm.k8 + (kr + 8) * HD + (((2 * ks + 1) ^ ((kr + 8) & 7)) << 4) + 4 * t);
        u32 b0 = *reinterpret_cast<const u32*>(S.q8 + qr * HD + (((2 * ks) ^ (qr & 7)) << 4) + 4 * t);
        u32 b1 = *reinterpret_cast<const u32*>(S.q8 + qr * HD + (((2 * ks + 1) ^ (qr & 7)) << 4) + 4 * t);
        mma8(st[j], a0, a1, a2, a3, b0, b1);
      }
    }
    // ---- dP^T = V dO^T (bf16)  [16 keys x 32 q]
    float dp[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) dp[j][0] = dp[j][1] = dp[j][2] = dp[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      const u8* vb = (const u8*)sm.v;
      u32 a0 = *reinterpret_cast<const u32*>(vb + swb(kr, ks * 16 + 2 * t));
      u32 a1 = *reinterpret_cast<const u32*>(vb + swb(kr + 8, ks * 16 + 2 * t));
      u32 a2 = *reinterpret_cast<const u32*>(vb + swb(kr, ks * 16 + 8 + 2 * t));
      u32 a3 = *reinterpret_cast<const u32*>(vb + swb(kr + 8, ks * 16 + 8 + 2 * t));
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int qr = j * 8 + g;
        u32 b0 = *reinterpret_cast<const u32*>((const u8*)S.dO + swb(qr, ks * 16 + 2 * t));
        u32 b1 = *reinterpret_cast<const u32*>((const u8*)S.dO + swb(qr, ks * 16 + 8 + 2 * t));
        mma16(dp[j], a0, a1, a2, a3, b0, b1);
      }
    }
    // ---- P^T, dS^T  (rows: keys key0+kr, +8 ; cols: queries q0 + j*8 + 2t + e)
    float p[4][4], ds[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        int ql = j * 8 + 2 * t + (e & 1), key = key0 + kr + (e >> 1) * 8, q = q0 + ql;
        float pv = key <= q ? exp2f(st[j][e] * sl2 - S.lse[ql]) : 0.f;
        p[j][e] = pv;
        float d_ = pv * (dp[j][e] - S.dd[ql]);
        ds[j][e] = d_;
        dsmax = fmaxf(dsmax, fabsf(d_));
      }
    // ---- dV += P^T dO ; dK += dS^T Q  (A from registers in the permuted k order)
    u32 pa0 = f2e4m3x4(p[0][0] * 448.f, p[0][1] * 448.f, p[1][0] * 448.f, p[1][1] * 448.f);
    u32 pa1 = f2e4m3x4(p[0][2] * 448.f, p[0][3] * 448.f, p[1][2] * 448.f, p[1][3] * 448.f);
    u32 pa2 = f2e4m3x4(p[2][0] * 448.f, p[2][1] * 448.f, p[3][0] * 448.f, p[3][1] * 448.f);
    u32 pa3 = f2e4m3x4(p[2][2] * 448.f, p[2][3] * 448.f, p[3][2] * 448.f, p[3][3] * 448.f);
    u32 sa0 = f2e4m3x4(ds[0][0] * inv_ds, ds[0][1] * inv_ds, ds[1][0] * inv_ds, ds[1][1] * inv_ds);
    u32 sa1 = f2e4m3x4(ds[0][2] * inv_ds, ds[0][3] * inv_ds, ds[1][2] * inv_ds, ds[1][3] * inv_ds);
    u32 sa2 = f2e4m3x4(ds[2][0] * inv_ds, ds[2][1] * inv_ds, ds[3][0] * inv_ds, ds[3][1] * inv_ds);
    u32 sa3 = f2e4m3x4(ds[2][2] * inv_ds, ds[2][3] * inv_ds, ds[3][2] * inv_ds, ds[3][3] * inv_ds);
#pragma unroll
    for (int nt = 0; nt < 16; nt++) {
      int d = nt * 8 + g, sw = ((d >> 2) & 1);
      u32 b0 = *reinterpret_cast<const u32*>(S.dot8 + d * QB + ((0 ^ sw) << 4) + 4 * t);
      u32 b1 = *reinterpret_cast<const u32*>(S.dot8 + d * QB + ((1 ^ sw) << 4) + 4 * t);
      mma8(dv[nt], pa0, pa1, pa2, pa3, b0, b1);
      u32 c0 = *reinterpret_cast<const u32*>(S.qt8 + d * QB + ((0 ^ sw) << 4) + 4 * t);
      u32 c1 = *reinterpret_cast<const u32*>(S.qt8 + d * QB + ((1 ^ sw) << 4) + 4 * t);
      mma8(dk[nt], sa0, sa1, sa2, sa3, c0, c1);
    }
    // ---- dS (fp8, [q][key]) -> smem for dQ
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        int ql = j * 8 + 2 * t + (e & 1), kl = kr + (e >> 1) * 8;
        unsigned short hv; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hv) : "f"(0.f), "f"(ds[j][e] * inv_ds));
        sm.ds[ql * 64 + ((((kl >> 4)) ^ ((ql >> 1) & 3)) << 4) + (kl & 15)] = (u8)(hv & 0xFF);
      }
    __syncthreads();
    // ---- dQ[32 x 128] += dS K : warp -> q rows (warp&1)*16.., d cols (warp>>1)*64..
    {
      int qr = (warp & 1) * 16 + g, dbase = (warp >> 1) * 64;
      float acc[8][4];
#pragma unroll
      for (int n = 0; n < 8; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        u32 a0 = *reinterpret_cast<const u32*>(sm.ds + qr * 64 + (((2 * ks) ^ ((qr >> 1) & 3)) << 4) + 4 * t);
        u32 a1 = *reinterpret_cast<const u32*>(sm.ds + (qr + 8) * 64 + (((2 * ks) ^ (((qr + 8) >> 1) & 3)) << 4) + 4 * t);
        u32 a2 = *reinterpret_cast<const u32*>(sm.ds + qr * 64 + (((2 * ks + 1) ^ ((qr >> 1) & 3)) << 4) + 4 * t);
        u32 a3 = *reinterpret_cast<const u32*>(sm.ds + (qr + 8) * 64 + (((2 * ks + 1) ^ (((qr + 8) >> 1) & 3)) << 4) + 4 * t);
#pragma unroll
        for (int n = 0; n < 8; n++) {
          int d = dbase + n * 8 + g, sw = (d >> 1) & 3;
          u32 b0 = *reinterpret_cast<const u32*>(sm.kt8 + d * 64 + (((2 * ks) ^ sw) << 4) + 4 * t);
          u32 b1 = *reinterpret_cast<const u32*>(sm.kt8 + d * 64 + (((2 * ks + 1) ^ sw) << 4) + 4 * t);
          mma8(acc[n], a0, a1, a2, a3, b0, b1);
        }
      }
      float f = s_ds * scale;
      float* dq0 = dQ + (((long long)b * H + hh) * T + q0 + qr) * HD + dbase + 2 * t;
      float* dq1 = dq0 + 8 * HD;
#pragma unroll
      for (int n = 0; n < 8; n++) {
        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(dq0 + n * 8), "f"(acc[n][0] * f), "f"(acc[n][1] * f) : "memory");
        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(dq1 + n * 8), "f"(acc[n][2] * f), "f"(acc[n][3] * f) : "memory");
      }
    }
    __syncthreads();
  }
  // ---- epilogue: dK = acc * s_ds * scale ; dV = acc * sdo / 448
  {
    float fk = s_ds * scale;
    long long r0 = ((long long)b * KV + kv) * T + key0 + warp * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int nt = 0; nt < 16; nt++) {
      int d = nt * 8 + 2 * t;
      *reinterpret_cast<__nv_bfloat162*>(dK + r0 * HD + d) = __floats2bfloat162_rn(dk[nt][0] * fk, dk[nt][1] * fk);
      *reinterpret_cast<__nv_bfloat162*>(dK + r1 * HD + d) = __floats2bfloat162_rn(dk[nt][2] * fk, dk[nt][3] * fk);
      const float* sd = sdo + ((long long)b * KV + kv) * HD;
      float v0 = sd[d] * (1.f / 448.f), v1 = sd[d + 1] * (1.f / 448.f);
      *reinterpret_cast<__nv_bfloat162*>(dV + r0 * HD + d) = __floats2bfloat162_rn(dv[nt][0] * v0, dv[nt][1] * v1);
      *reinterpret_cast<__nv_bfloat162*>(dV + r1 * HD + d) = __floats2bfloat162_rn(dv[nt][2] * v0, dv[nt][3] * v1);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) dsmax = fmaxf(dsmax, __shfl_xor_sync(0xffffffffu, dsmax, o));
  if (lane == 0) atomicMax(dsamax_cur, __float_as_uint(dsmax));
}

// ============================ forward ============================
// CTA: 64 queries (4 warps x 16) of one (b, h); key blocks of 64, cp.async double buffered.
// smem K tile: 64 rows x 128 B, 16-B chunk c of row r stored at chunk (c ^ (r & 7)).
// smem VT tile: 128 rows (d) x 64 B, chunk c (0..3) stored at (c ^ ((d >> 1) & 3)).
#define BQ 64
#define BK 64
struct __align__(16) FwdSmem { u8 k[2][BK * HD]; u8 v[2][HD * BK]; };

extern "C" __global__ void __launch_bounds__(128) attn_fwd8(
    const u8* __restrict__ Q8, const u8* __restrict__ K8, const u8* __restrict__ VT8, const float* __restrict__ sv,
    bf16* __restrict__ O, float* __restrict__ LSE2, int T, int H, int KV, float scale) {
  extern __shared__ __align__(16) u8 smraw[];
  FwdSmem& sm = *reinterpret_cast<FwdSmem*>(smraw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  int qb = gridDim.x - 1 - blockIdx.x;            // heaviest (longest causal) blocks first
  int bh = blockIdx.y, b = bh / H, h = bh % H, kv = h / (H / KV);
  const u8* Kb = K8 + ((long long)b * KV + kv) * T * HD;
  const u8* Vb = VT8 + ((long long)b * KV + kv) * HD * T;
  int q0 = qb * BQ + warp * 16;
  // Q fragments (4 k-steps of 32 dims)
  u32 qa[4][4];
  {
    const u8* qr0 = Q8 + (((long long)b * H + h) * T + q0 + g) * HD;
    const u8* qr1 = qr0 + 8 * HD;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      qa[ks][0] = *reinterpret_cast<const u32*>(qr0 + ks * 32 + 4 * t);
      qa[ks][1] = *reinterpret_cast<const u32*>(qr1 + ks * 32 + 4 * t);
      qa[ks][2] = *reinterpret_cast<const u32*>(qr0 + ks * 32 + 16 + 4 * t);
      qa[ks][3] = *reinterpret_cast<const u32*>(qr1 + ks * 32 + 16 + 4 * t);
    }
  }
  auto load = [&](int buf, int kb) {
#pragma unroll
    for (int i = 0; i < 4; i++) {                 // K: 512 chunks
      int id = tid + i * 128, r = id >> 3, c = id & 7;
      cp16(sm.k[buf] + r * HD + ((c ^ (r & 7)) << 4), Kb + (long long)(kb * BK + r) * HD + c * 16);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {                 // VT: 128 rows x 4 chunks
      int id = tid + i * 128, d = id >> 2, c = id & 3;
      cp16(sm.v[buf] + d * BK + ((c ^ ((d >> 1) & 3)) << 4), Vb + (long long)d * T + kb * BK + c * 16);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sl2 = scale * LOG2E;
  int nkb = qb + 1;
  load(0, 0);
  for (int kb = 0; kb < nkb; kb++) {
    int buf = kb & 1;
    if (kb + 1 < nkb) { load(buf ^ 1, kb + 1); asm volatile("cp.async.wait_group 1;\n"); }
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    const u8* ks_ = sm.k[buf];
    const u8* vs_ = sm.v[buf];
    // ---- S = Q K^T  (16 x 64 per warp)
    float s[8][4];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      int r = j * 8 + g;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        u32 b0 = *reinterpret_cast<const u32*>(ks_ + r * HD + (((2 * ks) ^ (r & 7)) << 4) + 4 * t);
        u32 b1 = *reinterpret_cast<const u32*>(ks_ + r * HD + (((2 * ks + 1) ^ (r & 7)) << 4) + 4 * t);
        mma8(s[j], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
      }
    }
    // ---- scale, causal mask, online softmax (rows g and g+8)
    bool diag = kb == qb;
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        float v = s[j][e] * sl2;
        if (diag) {
          int key = j * 8 + 2 * t + (e & 1), q = warp * 16 + g + (e >> 1) * 8;
          if (key > q) v = -INFINITY;
        }
        s[j][e] = v;
      }
      mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1])); mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
    }
    mx0 = qmax(mx0); mx1 = qmax(mx1);
    float c0 = exp2f(m0 - mx0), c1 = exp2f(m1 - mx1);
    m0 = mx0; m1 = mx1;
    float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      s[j][0] = exp2f(s[j][0] - m0); s[j][1] = exp2f(s[j][1] - m0);
      s[j][2] = exp2f(s[j][2] - m1); s[j][3] = exp2f(s[j][3] - m1);
      rs0 += s[j][0] + s[j][1]; rs1 += s[j][2] + s[j][3];
    }
    l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
#pragma unroll
    for (int i = 0; i < 16; i++) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
    // ---- O += P V : P (x448, e4m3) straight from the S accumulators (permuted k order)
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      const int j0 = 4 * kk;
      u32 a0 = f2e4m3x4(s[j0][0] * 448.f, s[j0][1] * 448.f, s[j0 + 1][0] * 448.f, s[j0 + 1][1] * 448.f);
      u32 a1 = f2e4m3x4(s[j0][2] * 448.f, s[j0][3] * 448.f, s[j0 + 1][2] * 448.f, s[j0 + 1][3] * 448.f);
      u32 a2 = f2e4m3x4(s[j0 + 2][0] * 448.f, s[j0 + 2][1] * 448.f, s[j0 + 3][0] * 448.f, s[j0 + 3][1] * 448.f);
      u32 a3 = f2e4m3x4(s[j0 + 2][2] * 448.f, s[j0 + 2][3] * 448.f, s[j0 + 3][2] * 448.f, s[j0 + 3][3] * 448.f);
#pragma unroll
      for (int nt = 0; nt < 16; nt++) {
        int d = nt * 8 + g;
        const u8* vr = vs_ + d * BK;
        int sw = (d >> 1) & 3;
        u32 b0 = *reinterpret_cast<const u32*>(vr + (((2 * kk) ^ sw) << 4) + 4 * t);
        u32 b1 = *reinterpret_cast<const u32*>(vr + (((2 * kk + 1) ^ sw) << 4) + 4 * t);
        mma8(o[nt], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncthreads();
  }
  // ---- epilogue
  l0 = qsum(l0); l1 = qsum(l1);
  float i0 = 1.f / (l0 * 448.f), i1 = 1.f / (l1 * 448.f);
  const float* svb = sv + ((long long)b * KV + kv) * HD;
  long long r0 = ((long long)b * T + q0 + g) * H + h, r1 = r0 + 8LL * H;
#pragma unroll
  for (int nt = 0; nt < 16; nt++) {
    int d = nt * 8 + 2 * t;
    float s0 = svb[d], s1 = svb[d + 1];
    *reinterpret_cast<__nv_bfloat162*>(O + r0 * HD + d) = __floats2bfloat162_rn(o[nt][0] * i0 * s0, o[nt][1] * i0 * s1);
    *reinterpret_cast<__nv_bfloat162*>(O + r1 * HD + d) = __floats2bfloat162_rn(o[nt][2] * i1 * s0, o[nt][3] * i1 * s1);
  }
  if (t == 0) {
    float* lp = LSE2 + ((long long)b * H + h) * T;
    lp[q0 + g] = m0 + log2f(l0);
    lp[q0 + g + 8] = m1 + log2f(l1);
  }
}
