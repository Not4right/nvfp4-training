// ================= fused MLP forward / backward =================
// Row scales for chunked (C-fed) operands come from a Cauchy-Schwarz bound computed before the chunk
// loop (|u_f| <= ||h|| * max_f ||W1[f,:]||, |da_f| <= ||dy|| * max_f ||W2[:,f]||), so all chunks of a
// row share one FP32 scale and the tensor-core accumulators run straight through.  E4M3 block scales
// absorb the (small, ~4x) looseness of the bound.

// raw A-layout rows (2 rows x 2 ksteps x 2 halves = 8 x uint4 of bf16) for register prefetch
__device__ __forceinline__ void ld_rows_raw(const __nv_bfloat16* __restrict__ X, int r0, int r1, int t, uint4* raw) {
#pragma unroll
  for (int ks = 0; ks < 2; ks++) {
    raw[ks * 4 + 0] = *reinterpret_cast<const uint4*>(X + (long long)r0 * DM + ks * 64 + t * 8);
    raw[ks * 4 + 1] = *reinterpret_cast<const uint4*>(X + (long long)r0 * DM + ks * 64 + 32 + t * 8);
    raw[ks * 4 + 2] = *reinterpret_cast<const uint4*>(X + (long long)r1 * DM + ks * 64 + t * 8);
    raw[ks * 4 + 3] = *reinterpret_cast<const uint4*>(X + (long long)r1 * DM + ks * 64 + 32 + t * 8);
  }
}
__device__ __forceinline__ void unpack8bf(uint4 v, float* o) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void ln_quant_raw(const uint4* raw, int t, const float* __restrict__ lnw, AFrag* hq,
                                             float* gsr, float* mean, float* rstd, float* hnorm) {
  float v[2][2][16];
#pragma unroll
  for (int ks = 0; ks < 2; ks++) {
    unpack8bf(raw[ks * 4 + 0], v[0][ks]); unpack8bf(raw[ks * 4 + 1], v[0][ks] + 8);
    unpack8bf(raw[ks * 4 + 2], v[1][ks]); unpack8bf(raw[ks * 4 + 3], v[1][ks] + 8);
  }
  float inv[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    float s = 0.f, s2 = 0.f;
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int i = 0; i < 16; i++) { s += v[h][ks][i]; s2 += v[h][ks][i] * v[h][ks][i]; }
    s = qsum(s); s2 = qsum(s2);
    mean[h] = s * (1.f / DM); rstd[h] = rsqrtf(fmaxf(s2 * (1.f / DM) - mean[h] * mean[h], 0.f) + 1e-5f);
    float am = 0.f, n2 = 0.f;
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int i = 0; i < 16; i++) {
        int col = ks * 64 + (i < 8 ? t * 8 + i : 32 + t * 8 + i - 8);
        float hv = (v[h][ks][i] - mean[h]) * rstd[h] * lnw[col];
        v[h][ks][i] = hv; am = fmaxf(am, fabsf(hv)); n2 += hv * hv;
      }
    am = qmax(am); hnorm[h] = sqrtf(qsum(n2));
    inv[h] = am > 0.f ? FP4_E4M3 / am : 0.f; gsr[h] = am * (1.f / FP4_E4M3);
  }
  u32 dummy = 0;
#pragma unroll
  for (int ks = 0; ks < 2; ks++) hq[ks] = quant_afrag(v[0][ks], v[1][ks], inv[0], inv[1], t, false, dummy);
}

// LayerNorm of 2 rows (A-layout) + NVFP4 quantization. Identical in fwd and bwd (recompute is exact).
__device__ __forceinline__ void ln_quant(const __nv_bfloat16* __restrict__ X, int r0, int r1, int t,
                                         const float* __restrict__ lnw, AFrag* hq, float* gsr, float* mean,
                                         float* rstd, float* hnorm) {
  float v[2][2][16];
#pragma unroll
  for (int ks = 0; ks < 2; ks++) {
    ld8bf(X + (long long)r0 * DM + ks * 64 + t * 8, v[0][ks]); ld8bf(X + (long long)r0 * DM + ks * 64 + 32 + t * 8, v[0][ks] + 8);
    ld8bf(X + (long long)r1 * DM + ks * 64 + t * 8, v[1][ks]); ld8bf(X + (long long)r1 * DM + ks * 64 + 32 + t * 8, v[1][ks] + 8);
  }
  float inv[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    float s = 0.f, s2 = 0.f;
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int i = 0; i < 16; i++) { s += v[h][ks][i]; s2 += v[h][ks][i] * v[h][ks][i]; }
    s = qsum(s); s2 = qsum(s2);
    mean[h] = s * (1.f / DM); rstd[h] = rsqrtf(fmaxf(s2 * (1.f / DM) - mean[h] * mean[h], 0.f) + 1e-5f);
    float am = 0.f, n2 = 0.f;
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int i = 0; i < 16; i++) {
        int col = ks * 64 + (i < 8 ? t * 8 + i : 32 + t * 8 + i - 8);
        float hv = (v[h][ks][i] - mean[h]) * rstd[h] * lnw[col];
        v[h][ks][i] = hv; am = fmaxf(am, fabsf(hv)); n2 += hv * hv;
      }
    am = qmax(am); hnorm[h] = sqrtf(qsum(n2));
    inv[h] = am > 0.f ? FP4_E4M3 / am : 0.f; gsr[h] = am * (1.f / FP4_E4M3);
  }
  u32 dummy = 0;
#pragma unroll
  for (int ks = 0; ks < 2; ks++) hq[ks] = quant_afrag(v[0][ks], v[1][ks], inv[0], inv[1], t, false, dummy);
}

// C-layout chunk (8 ntiles x 4) -> A fragment through pi, with a fixed per-row inverse scale
__device__ __forceinline__ AFrag chunk_afrag(float (*x)[4], float inv0, float inv1, int t, bool sr, u32& rs) {
  float a0v[16], a1v[16];
#pragma unroll
  for (int n = 0; n < 8; n++) {
    int s = (n & 3) * 2 + (n >> 2) * 8;
    a0v[s] = x[n][0]; a0v[s + 1] = x[n][1]; a1v[s] = x[n][2]; a1v[s + 1] = x[n][3];
  }
  return quant_afrag2(a0v, a1v, inv0, inv1, t, sr, rs);
}

#ifdef KPROF
__device__ unsigned long long g_prof2[16];
#endif
#define MLPF_SMEM 67584
extern "C" __global__ void __launch_bounds__(256, 1) mlp_fwd(
    const __nv_bfloat16* __restrict__ X1, __nv_bfloat16* __restrict__ Y, const float* __restrict__ lnw,
    const uint2* __restrict__ w1d, const u32* __restrict__ w1s, const float* __restrict__ w1amax,
    const uint2* __restrict__ w2d, const u32* __restrict__ w2s, const float* __restrict__ w2amax,
    const float* __restrict__ w1rn, int M) {
  extern __shared__ __align__(16) unsigned char sm[];
  smem_copy(sm, w1d, 32768); smem_copy(sm + 32768, w1s, 1024);
  smem_copy(sm + 33792, w2d, 32768); smem_copy(sm + 66560, w2s, 1024);
  __syncthreads();
  const uint2* W1 = (const uint2*)sm; const u32* S1 = (const u32*)(sm + 32768);
  const uint2* W2 = (const uint2*)(sm + 33792); const u32* S2 = (const u32*)(sm + 66560);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  float gw1 = *w1amax * (1.f / FP4_E4M3), gw2 = *w2amax * (1.f / FP4_E4M3), rn1 = *w1rn;
  u32 dummy = 0;

#ifdef KPROF
#define FT(i) do { long long _n = clock64(); if (threadIdx.x == 0) atomicAdd(&g_prof2[i], (unsigned long long)(_n - _ft)); _ft = _n; } while (0)
  long long _ft = clock64();
#else
#define FT(i)
#endif
#if defined(STAGGER_F) && STAGGER_F > 0
  if (warp & 1) __nanosleep(STAGGER_F);
#endif
  uint4 raw[8];
  if (blockIdx.x < M / TROWS) ld_rows_raw(X1, blockIdx.x * TROWS + warp * 16 + g, blockIdx.x * TROWS + warp * 16 + g + 8, t, raw);
  for (int tile = blockIdx.x; tile < M / TROWS; tile += gridDim.x) {
    int r0 = tile * TROWS + warp * 16 + g, r1 = r0 + 8;
    AFrag hq[2]; float gsr[2], mean[2], rstd[2], hn[2];
    FT(0);
    ln_quant_raw(raw, t, lnw, hq, gsr, mean, rstd, hn);
    FT(1);
    {  // prefetch the next tile's rows while this tile computes
      int nt = tile + gridDim.x;
      if (nt < M / TROWS) ld_rows_raw(X1, nt * TROWS + warp * 16 + g, nt * TROWS + warp * 16 + g + 8, t, raw);
    }
    float ba0 = fmaxf(hn[0] * rn1, 0.2f), ba1 = fmaxf(hn[1] * rn1, 0.2f);
    float ia0 = FP4_E4M3 / ba0, ia1 = FP4_E4M3 / ba1;
    float su0 = gsr[0] * gw1, su1 = gsr[1] * gw1;
    float y[16][4];
#pragma unroll
    for (int n = 0; n < 16; n++) y[n][0] = y[n][1] = y[n][2] = y[n][3] = 0.f;
    // software pipeline: FC1 MMAs of chunk c+1 are issued before GELU/quant of chunk c, so the tensor
    // pipe and the ALUs overlap (all warps otherwise alternate between the two in lockstep)
    float ua[8][4], ub[8][4];
#define FC1_CHUNK(U, C)                                                                                   \
    {                                                                                                     \
      _Pragma("unroll") for (int n = 0; n < 8; n++) U[n][0] = U[n][1] = U[n][2] = U[n][3] = 0.f;        \
      _Pragma("unroll") for (int ks = 0; ks < 2; ks++)                                                    \
      _Pragma("unroll") for (int n = 0; n < 8; n++) {                                                     \
        int wi = ((C) * 8 + n) * 2 + ks;                                                                  \
        mma4(U[n], hq[ks].a0, hq[ks].a1, hq[ks].a2, hq[ks].a3, W1[wi * 32 + lane], hq[ks].sf, S1[wi * 2 + (g >> 2)]); \
      }                                                                                                   \
    }
#define FC2_CHUNK(U, C)                                                                                   \
    {                                                                                                     \
      _Pragma("unroll") for (int n = 0; n < 8; n++) {                                                     \
        gelu2(U[n][0], U[n][1], su0, U[n][0], U[n][1]);                                                  \
        gelu2(U[n][2], U[n][3], su1, U[n][2], U[n][3]);                                                  \
      }                                                                                                   \
      AFrag aq = chunk_afrag(U, ia0, ia1, t, false, dummy);                                              \
      _Pragma("unroll") for (int n = 0; n < 16; n++) {                                                    \
        int wi = n * 8 + (C);                                                                             \
        mma4(y[n], aq.a0, aq.a1, aq.a2, aq.a3, W2[wi * 32 + lane], aq.sf, S2[wi * 2 + (g >> 2)]);        \
      }                                                                                                   \
    }
    FC1_CHUNK(ua, 0)
#pragma unroll 1
    for (int c = 0; c < FFD / 64; c += 2) {
      FC1_CHUNK(ub, c + 1)
      FC2_CHUNK(ua, c)
      if (c + 2 < FFD / 64) FC1_CHUNK(ua, c + 2)
      FC2_CHUNK(ub, c + 1)
    }
#undef FC1_CHUNK
#undef FC2_CHUNK
    float f0 = ba0 * (1.f / FP4_E4M3) * gw2, f1 = ba1 * (1.f / FP4_E4M3) * gw2;
#pragma unroll
    for (int n = 0; n < 16; n++) {
      int col = n * 8 + 2 * t;
      float2 x0 = __bfloat1622float2(*(const __nv_bfloat162*)(X1 + (long long)r0 * DM + col));
      float2 x1 = __bfloat1622float2(*(const __nv_bfloat162*)(X1 + (long long)r1 * DM + col));
      *(__nv_bfloat162*)(Y + (long long)r0 * DM + col) = __floats2bfloat162_rn(x0.x + y[n][0] * f0, x0.y + y[n][1] * f0);
      *(__nv_bfloat162*)(Y + (long long)r1 * DM + col) = __floats2bfloat162_rn(x1.x + y[n][2] * f1, x1.y + y[n][3] * f1);
    }
    FT(6);
  }
}

// ---------------- MLP backward ----------------
#ifdef KPROF
__device__ unsigned long long g_prof[32];
#define PT(i) do { long long _n = clock64(); if (threadIdx.x == 0) atomicAdd(&g_prof[i], (unsigned long long)(_n - _pt)); _pt = _n; } while (0)
#define PT0 long long _pt = clock64()
#else
#define PT(i)
#define PT0
#endif
#define MLPB_SMEM 101376
extern "C" __global__ void __launch_bounds__(256, 1) mlp_bwd(
    const __nv_bfloat16* __restrict__ X1, const __nv_bfloat16* __restrict__ DY, __nv_bfloat16* __restrict__ DX1,
    const float* __restrict__ lnw, float* __restrict__ dlnw,
    const uint2* __restrict__ w1d, const u32* __restrict__ w1s, const uint2* __restrict__ w2td, const u32* __restrict__ w2ts,
    const uint2* __restrict__ w1td, const u32* __restrict__ w1ts, const float* __restrict__ w1amax, const float* __restrict__ w2amax,
    const float* __restrict__ w2cn,
    unsigned char* __restrict__ dyT, unsigned char* __restrict__ dyTs, unsigned char* __restrict__ aT, unsigned char* __restrict__ aTs,
    unsigned char* __restrict__ duT, unsigned char* __restrict__ duTs, unsigned char* __restrict__ hT, unsigned char* __restrict__ hTs,
    const float* __restrict__ amax_use, unsigned* __restrict__ amax_cur, int M, const u32* __restrict__ rngp, int flags) {
  u32 seed = rngp[0] ^ (u32)(size_t)amax_cur, signs = rngp[1];
  extern __shared__ __align__(16) unsigned char sm[];
  smem_copy(sm, w1d, 32768); smem_copy(sm + 32768, w1s, 1024);
  smem_copy(sm + 33792, w2td, 32768); smem_copy(sm + 66560, w2ts, 1024);
  smem_copy(sm + 67584, w1td, 32768); smem_copy(sm + 100352, w1ts, 1024);
  __syncthreads();
  const uint2* W1 = (const uint2*)sm; const u32* S1 = (const u32*)(sm + 32768);
  const uint2* W2T = (const uint2*)(sm + 33792); const u32* S2T = (const u32*)(sm + 66560);
  const uint2* W1T = (const uint2*)(sm + 67584); const u32* S1T = (const u32*)(sm + 100352);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  float gw1 = *w1amax * (1.f / FP4_E4M3), gw2 = *w2amax * (1.f / FP4_E4M3), cn2 = *w2cn;
  float inv_dy = FP4_E4M3 / amax_use[0], inv_a = FP4_E4M3 / amax_use[1];
  float inv_du = FP4_E4M3 / amax_use[2], inv_h = FP4_E4M3 / amax_use[3];
  bool emit = !(flags & 1), sr = !(flags & 2);
  RHT R = make_rht(signs, g, t);
  u32 rs = hash32(seed ^ (blockIdx.x * 256 + threadIdx.x) * 0x9E3779B9u);
  float mx_dy = 0.f, mx_a = 0.f, mx_du = 0.f, mx_h = 0.f;
  float dg[4] = {0.f, 0.f, 0.f, 0.f};
#if defined(STAGGER_B) && STAGGER_B > 0
  if (warp & 1) __nanosleep(STAGGER_B);
#endif

  for (int tile = blockIdx.x; tile < M / TROWS; tile += gridDim.x) {
    int tok0 = tile * TROWS + warp * 16;
    int r0 = tok0 + g, r1 = r0 + 8;
    PT0;
    AFrag hq[2], dq[2]; float gsr[2], mean[2], rstd[2], hn[2], gsd[2], dn[2];
    ln_quant(X1, r0, r1, t, lnw, hq, gsr, mean, rstd, hn);
    PT(0);
    {  // DY -> dgrad operand (per-row scale, stochastic rounding)
      float v[2][2][16];
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        ld8bf(DY + (long long)r0 * DM + ks * 64 + t * 8, v[0][ks]); ld8bf(DY + (long long)r0 * DM + ks * 64 + 32 + t * 8, v[0][ks] + 8);
        ld8bf(DY + (long long)r1 * DM + ks * 64 + t * 8, v[1][ks]); ld8bf(DY + (long long)r1 * DM + ks * 64 + 32 + t * 8, v[1][ks] + 8);
      }
      float inv[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        float am = 0.f, n2 = 0.f;
#pragma unroll
        for (int ks = 0; ks < 2; ks++)
#pragma unroll
          for (int i = 0; i < 16; i++) { am = fmaxf(am, fabsf(v[h][ks][i])); n2 += v[h][ks][i] * v[h][ks][i]; }
        am = qmax(am); dn[h] = sqrtf(qsum(n2));
        inv[h] = am > 0.f ? FP4_E4M3 / am : 0.f; gsd[h] = am * (1.f / FP4_E4M3);
      }
#pragma unroll
      for (int ks = 0; ks < 2; ks++) dq[ks] = quant_afrag(v[0][ks], v[1][ks], inv[0], inv[1], t, sr, rs);
    }
    PT(1);
    float bd0 = fmaxf(1.13f * dn[0] * cn2, 1e-30f), bd1 = fmaxf(1.13f * dn[1] * cn2, 1e-30f);
    float id0 = FP4_E4M3 / bd0, id1 = FP4_E4M3 / bd1;
    float su0 = gsr[0] * gw1, su1 = gsr[1] * gw1, sd0 = gsd[0] * gw2, sd1 = gsd[1] * gw2;
    float dh[16][4];
#pragma unroll
    for (int n = 0; n < 16; n++) dh[n][0] = dh[n][1] = dh[n][2] = dh[n][3] = 0.f;

#pragma unroll 1
    for (int c = 0; c < FFD / 64; c++) {
      float u[8][4], da[8][4];
#pragma unroll
      for (int n = 0; n < 8; n++) { u[n][0] = u[n][1] = u[n][2] = u[n][3] = 0.f; da[n][0] = da[n][1] = da[n][2] = da[n][3] = 0.f; }
#pragma unroll
      for (int ks = 0; ks < 2; ks++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
          int wi = (c * 8 + n) * 2 + ks, si = wi * 2 + (g >> 2);
          mma4(u[n], hq[ks].a0, hq[ks].a1, hq[ks].a2, hq[ks].a3, W1[wi * 32 + lane], hq[ks].sf, S1[si]);
          mma4(da[n], dq[ks].a0, dq[ks].a1, dq[ks].a2, dq[ks].a3, W2T[wi * 32 + lane], dq[ks].sf, S2T[si]);
        }
      PT(2);
#pragma unroll
      for (int n = 0; n < 8; n++) {
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          float s_u = e < 2 ? su0 : su1, s_d = e < 2 ? sd0 : sd1, g0, g1, p0, p1;
          gelu_grad2(u[n][e] * s_u, u[n][e + 1] * s_u, g0, g1, p0, p1);
          u[n][e] = g0; u[n][e + 1] = g1;                                   // a
          da[n][e] *= s_d * p0; da[n][e + 1] *= s_d * p1;                   // du
        }
      }
      PT(4);
      AFrag aq = chunk_afrag(da, id0, id1, t, sr, rs);
      PT(5);
#pragma unroll
      for (int n = 0; n < 16; n++) {
        int wi = n * 8 + c;
        mma4(dh[n], aq.a0, aq.a1, aq.a2, aq.a3, W1T[wi * 32 + lane], aq.sf, S1T[wi * 2 + (g >> 2)]);
      }

      PT(3);
      if (emit) {
#pragma unroll
        for (int p = 0; p < 4; p++) {
          int fa = c * 64 + 16 * p, fb = fa + 8;
          mx_a = fmaxf(mx_a, emit16(u[2 * p], u[2 * p + 1], fa, tok0, M, R, inv_a, false, rs, aT, aTs, g, t));
          mx_du = fmaxf(mx_du, emit16(da[2 * p], da[2 * p + 1], fa, tok0, M, R, inv_du, sr, rs, duT, duTs, g, t));
        }
      }
      PT(6);
    }
    float f0 = bd0 * (1.f / FP4_E4M3) * gw1, f1 = bd1 * (1.f / FP4_E4M3) * gw1;
    // ---- LayerNorm backward in C-layout; reload x1 and dy there
    float xh[16][4];
    float s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f};
#pragma unroll
    for (int n = 0; n < 16; n++) {
      int col = n * 8 + 2 * t;
      float2 x0 = __bfloat1622float2(*(const __nv_bfloat162*)(X1 + (long long)r0 * DM + col));
      float2 x1 = __bfloat1622float2(*(const __nv_bfloat162*)(X1 + (long long)r1 * DM + col));
      xh[n][0] = (x0.x - mean[0]) * rstd[0]; xh[n][1] = (x0.y - mean[0]) * rstd[0];
      xh[n][2] = (x1.x - mean[1]) * rstd[1]; xh[n][3] = (x1.y - mean[1]) * rstd[1];
      dh[n][0] *= f0; dh[n][1] *= f0; dh[n][2] *= f1; dh[n][3] *= f1;
      float g0 = lnw[col], g1 = lnw[col + 1];
      float p0 = dh[n][0] * xh[n][0] + dh[n][2] * xh[n][2], p1 = dh[n][1] * xh[n][1] + dh[n][3] * xh[n][3];
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { p0 += __shfl_xor_sync(0xffffffffu, p0, o); p1 += __shfl_xor_sync(0xffffffffu, p1, o); }
      if ((n >> 1) == g) { dg[(n & 1) * 2] += p0; dg[(n & 1) * 2 + 1] += p1; }
      dh[n][0] *= g0; dh[n][1] *= g1; dh[n][2] *= g0; dh[n][3] *= g1;  // dxhat
      s1[0] += dh[n][0] + dh[n][1]; s1[1] += dh[n][2] + dh[n][3];
      s2[0] += dh[n][0] * xh[n][0] + dh[n][1] * xh[n][1]; s2[1] += dh[n][2] * xh[n][2] + dh[n][3] * xh[n][3];
    }
#pragma unroll
    for (int h = 0; h < 2; h++) { s1[h] = qsum(s1[h]) * (1.f / DM); s2[h] = qsum(s2[h]) * (1.f / DM); }
    float dyv[16][4];
#pragma unroll
    for (int n = 0; n < 16; n++) {
      int col = n * 8 + 2 * t;
      float2 y0 = __bfloat1622float2(*(const __nv_bfloat162*)(DY + (long long)r0 * DM + col));
      float2 y1 = __bfloat1622float2(*(const __nv_bfloat162*)(DY + (long long)r1 * DM + col));
      dyv[n][0] = y0.x; dyv[n][1] = y0.y; dyv[n][2] = y1.x; dyv[n][3] = y1.y;
      float o0 = y0.x + rstd[0] * (dh[n][0] - s1[0] - xh[n][0] * s2[0]);
      float o1 = y0.y + rstd[0] * (dh[n][1] - s1[0] - xh[n][1] * s2[0]);
      float o2 = y1.x + rstd[1] * (dh[n][2] - s1[1] - xh[n][2] * s2[1]);
      float o3 = y1.y + rstd[1] * (dh[n][3] - s1[1] - xh[n][3] * s2[1]);
      *(__nv_bfloat162*)(DX1 + (long long)r0 * DM + col) = __floats2bfloat162_rn(o0, o1);
      *(__nv_bfloat162*)(DX1 + (long long)r1 * DM + col) = __floats2bfloat162_rn(o2, o3);
      float g0 = lnw[col], g1 = lnw[col + 1];
      xh[n][0] *= g0; xh[n][1] *= g1; xh[n][2] *= g0; xh[n][3] *= g1;  // h2 = xhat * g
    }
    PT(7);
    if (emit) {
#pragma unroll
      for (int p = 0; p < 8; p++) {
        mx_h = fmaxf(mx_h, emit16(xh[2 * p], xh[2 * p + 1], 16 * p, tok0, M, R, inv_h, false, rs, hT, hTs, g, t));
        mx_dy = fmaxf(mx_dy, emit16(dyv[2 * p], dyv[2 * p + 1], 16 * p, tok0, M, R, inv_dy, sr, rs, dyT, dyTs, g, t));
      }
    }
    PT(8);
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int n = 2 * g + (i >> 1), col = n * 8 + 2 * t + (i & 1);
    atomicAdd(dlnw + col, dg[i]);
  }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    mx_dy = fmaxf(mx_dy, __shfl_xor_sync(0xffffffffu, mx_dy, o)); mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, o));
    mx_du = fmaxf(mx_du, __shfl_xor_sync(0xffffffffu, mx_du, o)); mx_h = fmaxf(mx_h, __shfl_xor_sync(0xffffffffu, mx_h, o));
  }
  if (lane == 0) {
    atomicMax(amax_cur + 0, __float_as_uint(mx_dy)); atomicMax(amax_cur + 1, __float_as_uint(mx_a));
    atomicMax(amax_cur + 2, __float_as_uint(mx_du)); atomicMax(amax_cur + 3, __float_as_uint(mx_h));
  }
}
