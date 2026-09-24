// ================= WG16: fragment-native layout for wgrad operands + its GEMM =================
// Operand with F rows (features) and M tokens, blocks of (16 rows x 64 tokens):
//   data  block b = (fb*(M/64) + kb): 512 B; lane L = 4g+t owns 16 B = words {a0, a2, a1, a3}
//         (a0: row g, k t*8..+7 | a2: row g, k 32+t*8.. | a1/a3: row g+8), low nibble = lower k
//   scale block b: 64 B, row r's u32 at 4r, byte q = k-block q (16 tokens)
// Token -> k position inside a 64-block: group q = (tok%64)/16, and inside a group the emitting
// thread t' holds tokens (2t',2t'+1,8+2t',9+2t') -> k = 16q + 4t' + {0,1,2,3}.  Both operands of a
// wgrad GEMM use the same map, so the contraction over tokens is unchanged.

// Stochastic rounding on the E2M1 grid: FSET gives 1.0/0.0, dither the mantissa of |v|(+1 below 1).
__device__ __forceinline__ float sr2(float v, u32 r22) {
  float a = fabsf(v), s;
  asm("set.lt.f32.f32 %0, %1, 0f3F800000;" : "=f"(s) : "f"(a));  // s = a < 1 ? 1.0 : 0.0
  float b = a + s;
  float q = __uint_as_float((__float_as_uint(b) + r22) & 0xFFC00000u) - s;
  return __uint_as_float(__float_as_uint(q) | (__float_as_uint(v) & 0x80000000u));
}
// SR onto the E2M1 grid and return the 4-bit code directly (no cvt on the XU pipe):
// |v| < 1 is lifted by +1 so the grid below 1 (0, .5) becomes the 1-mantissa-bit binade [1,2).
__device__ __forceinline__ u32 sr_code(float v, u32 r22) {
  u32 vb = __float_as_uint(v), ab = vb & 0x7FFFFFFFu;
  bool lt = ab < 0x3F800000u;
  float b = __uint_as_float(ab) + (lt ? 1.f : 0.f);
  u32 bb = (__float_as_uint(b) + r22) >> 22;          // 2*E + mantissa bit (grid-snapped)
  int code = (int)bb - (lt ? 254 : 252);
  code = min(code, 7);
  return (u32)code | ((vb >> 28) & 8u);
}
__device__ __forceinline__ u32 packbf_fast(float lo, float hi) {  // round-half-up to bf16, 2 IADD + PRMT
  return __byte_perm(__float_as_uint(lo) + 0x8000u, __float_as_uint(hi) + 0x8000u, 0x7632);
}
__device__ __forceinline__ u32 e4m3x2(float lo, float hi) {  // two scale bytes: lo in byte0
  unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float2 e4m3x2_dec(u32 b) {
  u32 r; asm("{.reg .b32 t; cvt.rn.f16x2.e4m3x2 t, %1; mov.b32 %0, t;}" : "=r"(r) : "h"((unsigned short)b));
  __half2 h = *reinterpret_cast<__half2*>(&r);
  return __half22float2(h);
}

// emit one 16-token x 16-feature block (C-layout, 2 ntiles) into WG16.  fa: first feature (16-aligned)
__device__ __forceinline__ float emit16(const float* xa, const float* xb, int fa, int tok0, int M,
                                        const RHT& R, float inv, bool sr, u32& rs,
                                        unsigned char* __restrict__ Q, unsigned char* __restrict__ S, int g, int t) {
#ifdef XU_PACK
  u32 a0 = movtrans(packbf_fast(xa[0], xa[1])), a2 = movtrans(packbf_fast(xa[2], xa[3]));
  u32 a1 = movtrans(packbf_fast(xb[0], xb[1])), a3 = movtrans(packbf_fast(xb[2], xb[3]));
#else
  u32 a0 = movtrans(packbf(xa[0], xa[1])), a2 = movtrans(packbf(xa[2], xa[3]));
  u32 a1 = movtrans(packbf(xb[0], xb[1])), a3 = movtrans(packbf(xb[2], xb[3]));
#endif
  float d0[4], d1[4];
  mma_bf16(d0, a0, a1, a2, a3, R.b[0][0], R.b[0][1]);
  mma_bf16(d1, a0, a1, a2, a3, R.b[1][0], R.b[1][1]);
  // row g (feature fa+g): d0[0],d0[1],d1[0],d1[1] ; row g+8 (feature fa+8+g): d0[2],d0[3],d1[2],d1[3]
  float m0 = fmaxf(fmaxf(fabsf(d0[0]), fabsf(d0[1])), fmaxf(fabsf(d1[0]), fabsf(d1[1])));
  float m1 = fmaxf(fmaxf(fabsf(d0[2]), fabsf(d0[3])), fmaxf(fabsf(d1[2]), fabsf(d1[3])));
#ifdef XU_H2MAX
  {  // both rows' quad-max in one half2 (rounded up, identical on every lane of the quad)
    __half2 h = __halves2half2(__float2half_ru(m0), __float2half_ru(m1));
    u32 hu = *reinterpret_cast<u32*>(&h);
    __half2 o = *reinterpret_cast<__half2*>(&hu);
    u32 x1 = __shfl_xor_sync(0xffffffffu, hu, 1); o = __hmax2(o, *reinterpret_cast<__half2*>(&x1));
    u32 ou = *reinterpret_cast<u32*>(&o);
    u32 x2 = __shfl_xor_sync(0xffffffffu, ou, 2); o = __hmax2(o, *reinterpret_cast<__half2*>(&x2));
    float2 mm = __half22float2(o); m0 = mm.x; m1 = mm.y;
  }
#else
  m0 = qmax(m0); m1 = qmax(m1);
#endif
  u32 sb = e4m3x2(m0 * inv * (1.f / 6.f), m1 * inv * (1.f / 6.f));
  float2 dec = e4m3x2_dec(sb);
  float k0 = dec.x > 0.f ? inv / dec.x : 0.f, k1 = dec.y > 0.f ? inv / dec.y : 0.f;
  float v[8] = {d0[0] * k0, d0[1] * k0, d1[0] * k0, d1[1] * k0, d0[2] * k1, d0[3] * k1, d1[2] * k1, d1[3] * k1};
  u32 va, vb;
#ifdef XU_SRINT
  if (sr) {
    u32 c[8];
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      u32 r = rng_next(rs);
      c[i] = sr_code(v[i], (r << 11) & 0x3FF800u); c[i + 1] = sr_code(v[i + 1], (r >> 5) & 0x3FF800u);
    }
    va = c[0] | (c[1] << 4) | (c[2] << 8) | (c[3] << 12); vb = c[4] | (c[5] << 4) | (c[6] << 8) | (c[7] << 12);
  } else
#else
  if (sr) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      u32 r = rng_next(rs);
      v[i] = sr2(v[i], (r << 11) & 0x3FF800u); v[i + 1] = sr2(v[i + 1], (r >> 5) & 0x3FF800u);
    }
  }
#endif
  { va = fp4x2(v[0], v[1]) | (fp4x2(v[2], v[3]) << 8); vb = fp4x2(v[4], v[5]) | (fp4x2(v[6], v[7]) << 8); }
  int kb = tok0 >> 6, q = (tok0 >> 4) & 3;
  long long blk = (long long)(fa >> 4) * (M >> 6) + kb;
  unsigned char* base = Q + blk * 512 + (g * 4 + (q & 1) * 2 + (t >> 1)) * 16 + (q >> 1) * 4 + (t & 1) * 2;
  *reinterpret_cast<unsigned short*>(base) = (unsigned short)va;       // row g   -> word (q>=2)
  *reinterpret_cast<unsigned short*>(base + 8) = (unsigned short)vb;   // row g+8 -> word 2+(q>=2)
  if (t >= 2) S[blk * 64 + (g + (t - 2) * 8) * 4 + q] = (unsigned char)(t == 2 ? (sb & 0xFF) : (sb >> 8));
  return fmaxf(m0, m1);
}

// ---- grouped split-K GEMM over WG16 operands: out[FA,FB] += alpha * A[FA,M] B[FB,M]^T  (fp32 atomics)
struct WGProb {
  const unsigned char* A; const unsigned char* SA; const unsigned char* B; const unsigned char* SB;
  float* out; const float* amaxA; const float* amaxB; int FA, FB, nsplit;
};
#define WG_STAGES 4
#define WG_STAGE_BYTES (8 * 576 * 2)
extern "C" __global__ void __launch_bounds__(256, 1) wg_gemm(const WGProb* __restrict__ probs, int nblk0, int M) {
  extern __shared__ __align__(16) unsigned char sm[];
  int bid = blockIdx.x;
  const WGProb P = probs[bid < nblk0 ? 0 : 1];
  if (bid >= nblk0) bid -= nblk0;
  int tilesB = P.FB / 128, tilesA = P.FA / 128;
  int ntiles = (P.FA / 128) * (P.FB / 128);
  int tile = bid % ntiles, split = bid / ntiles;  // split-major: concurrent CTAs share k-ranges -> L2 reuse
  int ta = tile / tilesB, tb = tile % tilesB;
  int KB = M >> 6, per = (KB + P.nsplit - 1) / P.nsplit, k0 = split * per, k1 = min(KB, k0 + per), nk = k1 - k0;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  int wm = warp >> 2, wn = warp & 3;  // warp tile: A blocks wm*4..+3 (64 rows), B blocks wn*2..+1 (32 rows)
  (void)tilesA;
  auto load = [&](int s, int kk) {
    unsigned char* st = sm + s * WG_STAGE_BYTES;
    int kb = k0 + kk;
    for (int i = tid; i < 16 * 36; i += 256) {  // 16 blocks x (32 data + 4 scale) 16B chunks
      int b = i / 36, c = i % 36, isB = b >= 8, rb = b & 7;
      int fb = (isB ? tb : ta) * 8 + rb;
      long long blk = (long long)fb * KB + kb;
      const unsigned char* src = c < 32 ? (isB ? P.B : P.A) + blk * 512 + c * 16 : (isB ? P.SB : P.SA) + blk * 64 + (c - 32) * 16;
      cp16(st + b * 576 + c * 16, src);
    }
  };
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
#pragma unroll
  for (int s = 0; s < WG_STAGES - 1; s++) { if (s < nk) load(s, s); asm volatile("cp.async.commit_group;\n"); }
  for (int kk = 0; kk < nk; kk++) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(WG_STAGES - 2));
    __syncthreads();
    if (kk + WG_STAGES - 1 < nk) load((kk + WG_STAGES - 1) % WG_STAGES, kk + WG_STAGES - 1);
    asm volatile("cp.async.commit_group;\n");
    const unsigned char* st = sm + (kk % WG_STAGES) * WG_STAGE_BYTES;
    uint4 bf[2]; u32 sfb[2][2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const unsigned char* bb = st + (8 + wn * 2 + j) * 576;
      bf[j] = *reinterpret_cast<const uint4*>(bb + lane * 16);
      sfb[j][0] = *reinterpret_cast<const u32*>(bb + 512 + g * 4);
      sfb[j][1] = *reinterpret_cast<const u32*>(bb + 512 + (g + 8) * 4);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const unsigned char* ab = st + (wm * 4 + i) * 576;
      uint4 a = *reinterpret_cast<const uint4*>(ab + lane * 16);
      u32 sfa = *reinterpret_cast<const u32*>(ab + 512 + (g + 8 * (lane & 1)) * 4);
#pragma unroll
      for (int j = 0; j < 2; j++) {
        mma4(acc[i][2 * j], a.x, a.z, a.y, a.w, make_uint2(bf[j].x, bf[j].y), sfa, sfb[j][0]);
        mma4(acc[i][2 * j + 1], a.x, a.z, a.y, a.w, make_uint2(bf[j].z, bf[j].w), sfa, sfb[j][1]);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n");
  float alpha = (*P.amaxA) * (*P.amaxB) * (1.f / (FP4_E4M3 * FP4_E4M3));
  // pair lanes t, t^1: even t sends its row-g+8 pair and receives the partner's row-g pair -> each lane
  // owns 4 consecutive columns of one row -> one 16-byte vector atomic instead of two 8-byte ones
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
    {
      float* a = acc[i][j];
      bool odd = t & 1;
      float s0 = odd ? a[0] : a[2], s1 = odd ? a[1] : a[3];
      float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
      int row = ta * 128 + wm * 64 + i * 16 + g + (odd ? 8 : 0);
      int col = tb * 128 + wn * 32 + j * 8 + 2 * (t & ~1);
      float4 v = odd ? make_float4(r0, r1, a[2], a[3]) : make_float4(a[0], a[1], r0, r1);
      v.x *= alpha; v.y *= alpha; v.z *= alpha; v.w *= alpha;
#ifdef WG_NOATOMIC
      if (v.x == 1.2345f)
#endif
      atomicAdd(reinterpret_cast<float4*>(P.out + (long long)row * P.FB + col), v);
    }
  }
}


// ---- faster A-fragment quantizer: 4 groups of 8 (rows g/g+8 x lo/hi), scales converted in pairs,
// stochastic rounding via sr2.  Same numerics as quant_afrag (RN path bit-identical).
__device__ __forceinline__ u32 pack8(const float* v, float k, bool sr, u32& rs) {
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = v[i] * k;
#ifdef XU_SRINT
  if (sr) {
    u32 w = 0;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      u32 r = rng_next(rs);
      w |= (sr_code(x[i], (r << 11) & 0x3FF800u) | (sr_code(x[i + 1], (r >> 5) & 0x3FF800u) << 4)) << (4 * i);
    }
    return w;
  }
#else
  if (sr) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) { u32 r = rng_next(rs); x[i] = sr2(x[i], (r << 11) & 0x3FF800u); x[i + 1] = sr2(x[i + 1], (r >> 5) & 0x3FF800u); }
  }
#endif
  return fp4x2(x[0], x[1]) | (fp4x2(x[2], x[3]) << 8) | (fp4x2(x[4], x[5]) << 16) | (fp4x2(x[6], x[7]) << 24);
}
__device__ __forceinline__ AFrag quant_afrag2(const float* v0, const float* v1, float inv0, float inv1, int t, bool sr, u32& rs) {
  float m[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    m[0] = fmaxf(m[0], fabsf(v0[i])); m[1] = fmaxf(m[1], fabsf(v0[8 + i]));
    m[2] = fmaxf(m[2], fabsf(v1[i])); m[3] = fmaxf(m[3], fabsf(v1[8 + i]));
  }
  // partner lane (^1) holds the other half of each 16-block: pack two maxima per shuffle as half2
  // (upper bound only matters to 1 ulp of f16 -> round the max UP so the scale never under-covers)
  // both lanes of a pair must derive the identical scale: round both maxima UP to f16 on both lanes
  __half2 h01 = __halves2half2(__float2half_ru(m[0]), __float2half_ru(m[1]));
  __half2 h23 = __halves2half2(__float2half_ru(m[2]), __float2half_ru(m[3]));
  u32 u01 = *reinterpret_cast<u32*>(&h01), u23 = *reinterpret_cast<u32*>(&h23);
  u32 o01 = __shfl_xor_sync(0xffffffffu, u01, 1), o23 = __shfl_xor_sync(0xffffffffu, u23, 1);
  __half2 q01 = __hmax2(h01, *reinterpret_cast<__half2*>(&o01)), q23 = __hmax2(h23, *reinterpret_cast<__half2*>(&o23));
  float2 f01 = __half22float2(q01), f23 = __half22float2(q23);
  m[0] = f01.x; m[1] = f01.y; m[2] = f23.x; m[3] = f23.y;
  u32 sA = e4m3x2(m[0] * inv0 * (1.f / 6.f), m[1] * inv0 * (1.f / 6.f));   // row g: lo, hi
  u32 sB = e4m3x2(m[2] * inv1 * (1.f / 6.f), m[3] * inv1 * (1.f / 6.f));   // row g+8: lo, hi
  float2 dA = e4m3x2_dec(sA), dB = e4m3x2_dec(sB);
  AFrag f;
  f.a0 = pack8(v0, dA.x > 0.f ? inv0 / dA.x : 0.f, sr, rs);
  f.a2 = pack8(v0 + 8, dA.y > 0.f ? inv0 / dA.y : 0.f, sr, rs);
  f.a1 = pack8(v1, dB.x > 0.f ? inv1 / dB.x : 0.f, sr, rs);
  f.a3 = pack8(v1 + 8, dB.y > 0.f ? inv1 / dB.y : 0.f, sr, rs);
  f.sf = build_sfa(sA & 0xFF, sA >> 8, sB & 0xFF, sB >> 8, t);
  return f;
}

// GELU(tanh) and its derivative for 2 values at once in f16x2 (inputs/outputs f32)
__device__ __forceinline__ void gelu_grad2(float u0, float u1, float& a0, float& a1, float& d0, float& d1) {
  __half2 u = __floats2half2_rn(u0, u1);
  __half2 u2 = __hmul2(u, u);
  __half2 z = __hmul2(__hmul2(u, __float2half2_rn(0.7978845608f)), __hfma2(u2, __float2half2_rn(0.044715f), __float2half2_rn(1.f)));
  u32 zr = *reinterpret_cast<u32*>(&z), thr;
  asm("tanh.approx.f16x2 %0, %1;" : "=r"(thr) : "r"(zr));
  __half2 th = *reinterpret_cast<__half2*>(&thr);
  __half2 one = __float2half2_rn(1.f), half_ = __float2half2_rn(0.5f);
  __half2 opt = __hadd2(one, th);                                   // 1 + th
  __half2 a = __hmul2(__hmul2(half_, u), opt);                      // gelu
  __half2 sech2 = __hfma2(__hneg2(th), th, one);                    // 1 - th^2
  __half2 dz = __hfma2(u2, __float2half2_rn(0.134145f * 0.7978845608f), __float2half2_rn(0.7978845608f));
  __half2 d = __hfma2(__hmul2(__hmul2(half_, u), sech2), dz, __hmul2(half_, opt));
  float2 af = __half22float2(a), df = __half22float2(d);
  a0 = af.x; a1 = af.y; d0 = df.x; d1 = df.y;
}

// GELU(tanh) for 2 values in f16x2; x is pre-scaled by s (u = x*s)
__device__ __forceinline__ void gelu2(float x0, float x1, float s, float& a0, float& a1) {
  __half2 u = __floats2half2_rn(x0 * s, x1 * s);
  __half2 z = __hmul2(__hmul2(u, __float2half2_rn(0.7978845608f)), __hfma2(__hmul2(u, u), __float2half2_rn(0.044715f), __float2half2_rn(1.f)));
  u32 zr = *reinterpret_cast<u32*>(&z), thr;
  asm("tanh.approx.f16x2 %0, %1;" : "=r"(thr) : "r"(zr));
  __half2 a = __hmul2(__hmul2(__float2half2_rn(0.5f), u), __hadd2(__float2half2_rn(1.f), *reinterpret_cast<__half2*>(&thr)));
  float2 af = __half22float2(a); a0 = af.x; a1 = af.y;
}
