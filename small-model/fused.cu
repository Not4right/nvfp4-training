// Fused NVFP4 transformer kernels.  Appended after fp4k.cu (shares its helpers).
//
// Conventions
//  * token tile = 128 rows per CTA iteration, 8 warps, warp w owns rows 16w..16w+15
//  * "A-layout": thread (g=lane/4, t=lane%4) holds, for each 64-wide k-step, row g and g+8,
//    cols t*8..t*8+7 ("lo" group) and 32+t*8..+7 ("hi" group)  == m16n8k64 A fragment order
//  * "C-layout": mma accumulator order: row g / g+8, cols 8n+2t, 8n+2t+1
//  * ff (hidden MLP) dim uses permutation pi inside each 64 chunk so that a C-layout chunk
//    is directly an A fragment: logical c = 8j+2t+e  <->  phys (j<4 ? t*8 : 32+t*8) + 2(j%4)+e
//  * weights live in "frag-native" form: bdata[(ntile*KS+ks)*32+lane] (uint2), bscale[(ntile*KS+ks)*8+g]
//  * activations quantized per (row, 16-block) with an exact per-row FP32 scale (current scaling)

#define DM 128
#define FFD 512
#define TROWS 128

__device__ __forceinline__ int pi_fwd(int c) {  // logical -> phys within 64 chunk
  int j = c >> 3, t = (c & 7) >> 1, e = c & 1;
  return (j < 4) ? (t * 8 + 2 * j + e) : (32 + t * 8 + 2 * (j - 4) + e);
}
__device__ __forceinline__ int pi_inv(int p) {
  int hi = p >= 32, q = p - 32 * hi, t = q >> 3, s = q & 7;
  return 8 * ((s >> 1) + 4 * hi) + 2 * t + (s & 1);
}
__device__ __forceinline__ int grp_member(int b, int i, int perm) {
  if (!perm) return b * 16 + i;
  int chunk = b >> 2, p = (b & 3) * 16 + i;
  return chunk * 64 + pi_inv(p);
}
__device__ __forceinline__ int grp_of(int c, int perm) {
  if (!perm) return c >> 4;
  return (c >> 6) * 4 + (pi_fwd(c & 63) >> 4);
}

// ---------------- weight prep ----------------
// W [N,K] fp32 -> nibble per element NB [N,K] (uint8, logical layout), 2D block scales BS [N/16,K/16]
extern "C" __global__ void wq_nibbles(const float* __restrict__ W, unsigned char* __restrict__ NB,
                                      unsigned char* __restrict__ BS, const float* __restrict__ amax,
                                      int N, int K, int permN, int permK) {
  // one warp per 16x16 block; lane -> block row lane/2, 8 block cols from (lane&1)*8
  int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  int nb = N / 16, kb = K / 16;
  if (b >= nb * kb) return;
  int bn = b / kb, bk = b % kb;
  float am = *amax, inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  int n = grp_member(bn, lane >> 1, permN);
  float x[8], m = 0.f;
  int kk[8];
#pragma unroll
  for (int j = 0; j < 8; j++) { kk[j] = grp_member(bk, (lane & 1) * 8 + j, permK); x[j] = W[(long long)n * K + kk[j]]; m = fmaxf(m, fabsf(x[j])); }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  u32 sb = e4m3_enc(m * inv_gs * (1.f / 6.f));
  float sdec = e4m3_dec(sb), k = sdec > 0.f ? inv_gs / sdec : 0.f;
  if (lane == 0) BS[b] = (unsigned char)sb;
#pragma unroll
  for (int j = 0; j < 8; j += 2) {
    u32 byte = fp4x2(x[j] * k, x[j + 1] * k);
    NB[(long long)n * K + kk[j]] = byte & 15;
    NB[(long long)n * K + kk[j + 1]] = byte >> 4;
  }
}

// frag-native B operand of shape (Nf rows, Kf contraction) from NB/BS (logical [N,K]).
// trans: Nf=K, Kf=N.  ffN / ffK: that frag dim is an ff (pi-grouped) dim; ffK also means phys perm.
extern "C" __global__ void wq_frag(const unsigned char* __restrict__ NB, const unsigned char* __restrict__ BS,
                                   uint2* __restrict__ bd, u32* __restrict__ bsc, int N, int K,
                                   int trans, int ffN, int ffK) {
  int Nf = trans ? K : N, Kf = trans ? N : K, KS = Kf / 64;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= (Nf / 8) * KS * 32) return;
  int lane = id & 31, ks = (id >> 5) % KS, nt = (id >> 5) / KS;
  int g = lane >> 2, t = lane & 3, n = nt * 8 + g;
  u32 w[2] = {0, 0};
  for (int h = 0; h < 2; h++)
    for (int i = 0; i < 8; i++) {
      int p = h * 32 + t * 8 + i;  // phys k in chunk
      int kl = ks * 64 + (ffK ? pi_inv(p) : p);
      unsigned char nib = trans ? NB[(long long)kl * K + n] : NB[(long long)n * K + kl];
      w[h] |= (u32)nib << (4 * i);
    }
  bd[id] = make_uint2(w[0], w[1]);
  if (t == 0) {
    int gn = grp_of(n, ffN);
    u32 s = 0;
    for (int q = 0; q < 4; q++) {
      int gk = ks * 4 + q;  // phys block == group id for both perm and natural
      int gr = trans ? gk : gn, gc = trans ? gn : gk;
      s |= (u32)BS[gr * (K / 16) + gc] << (8 * q);
    }
    if ((g & 3) == 0) bsc[(nt * KS + ks) * 2 + (g >> 2)] = s;
  }
}

// ---------------- device helpers ----------------
__device__ __forceinline__ void mma4(float* c, u32 a0, u32 a1, u32 a2, u32 a3, uint2 b, u32 sfa, u32 sfb) {
  asm(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%12}, %11, {%12,%12};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y), "r"(sfa), "r"(sfb), "h"((unsigned short)0));
}
__device__ __forceinline__ float qmax(float v) {  // max over the 4 lanes of a quad
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float qsum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
// quantize one 8-value group whose 16-block partner lives in lane^1. returns nibbles, scale byte.
__device__ __forceinline__ u32 rng_next(u32& s) { s = s * 1664525u + 1013904223u; return s ^ (s >> 16); }
__device__ __forceinline__ u32 qgroup8(const float* v, float inv, u32& sb, bool sr, u32& rs) {
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(v[i]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
  sb = e4m3_enc(m * inv * (1.f / 6.f));
  float sdec = e4m3_dec(sb), k = sdec > 0.f ? inv / sdec : 0.f;
  u32 w = 0;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    float a = v[i] * k, b = v[i + 1] * k;
    if (sr) { u32 r = rng_next(rs); a = sr_fp4(a, (r << 11) & 0x3FF800u); b = sr_fp4(b, (r >> 5) & 0x3FF800u); }
    w |= fp4x2(a, b) << (4 * i);
  }
  return w;
}
// Given scale bytes of this thread's two rows (row g: s0lo,s0hi; row g+8: s1lo,s1hi) build the
// SFA register (lane 4g -> row g, lane 4g+1 -> row g+8).
__device__ __forceinline__ u32 build_sfa(u32 s0lo, u32 s0hi, u32 s1lo, u32 s1hi, int t) {
  u32 mlo = (t & 1) ? s1lo : s0lo, mhi = (t & 1) ? s1hi : s0hi;
  u32 mine = mlo | (mhi << 8);
  u32 oth = __shfl_xor_sync(0xffffffffu, mine, 2);
  return mlo | ((oth & 0xFF) << 8) | (mhi << 16) | ((oth >> 8) << 24);
}
struct AFrag { u32 a0, a1, a2, a3, sf; };
// v0: row g values [16] = lo 8 then hi 8; v1: row g+8.  inv0/inv1: per-row inverse global scale.
__device__ __forceinline__ AFrag quant_afrag(const float* v0, const float* v1, float inv0, float inv1, int t,
                                             bool sr, u32& rs) {
  AFrag f; u32 s0l, s0h, s1l, s1h;
  f.a0 = qgroup8(v0, inv0, s0l, sr, rs);
  f.a2 = qgroup8(v0 + 8, inv0, s0h, sr, rs);
  f.a1 = qgroup8(v1, inv1, s1l, sr, rs);
  f.a3 = qgroup8(v1 + 8, inv1, s1h, sr, rs);
  f.sf = build_sfa(s0l, s0h, s1l, s1h, t);
  return f;
}
__device__ __forceinline__ void ld8bf(const __nv_bfloat16* p, float* o) {
  uint4 v = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ float gelu_t(float u) {
  float th; float z = 0.7978845608f * (u + 0.044715f * u * u * u);
  asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(z));
  return 0.5f * u * (1.f + th);
}
__device__ __forceinline__ float gelu_t_grad(float u, float& gl) {
  float th; float z = 0.7978845608f * (u + 0.044715f * u * u * u);
  asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(z));
  gl = 0.5f * u * (1.f + th);
  return 0.5f * (1.f + th) + 0.5f * u * (1.f - th * th) * 0.7978845608f * (1.f + 0.134145f * u * u);
}
__device__ __forceinline__ void smem_copy(void* dst, const void* src, int bytes) {
  const uint4* s = (const uint4*)src; uint4* d = (uint4*)dst;
  for (int i = threadIdx.x; i < bytes / 16; i += blockDim.x) d[i] = s[i];
}

