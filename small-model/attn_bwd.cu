// ================= fused attention block backward =================
// Tile = 64 tokens = 2 sequences, 8 warps.
// Phase A: warp w -> (seq s = w/4, head h = w%4): recompute LN1/QKV/softmax, dO = Q(DX1) Wo (FP4),
//          attention backward on bf16 tensor cores, emit wgrad operands (O, dq, dk, dv), and write
//          dqkv quantized (FP4, per-tensor delayed scale, SR) into smem in A-fragment layout.
// Phase B: warp w -> (m-tile w%4, column half w/4): dh1 = dqkv Wqkv (FP4), LayerNorm1 backward,
//          DX = DX1 + LNbwd(dh1); emit wgrad operands h1, DX1.
// Wqkv rows are reordered so contraction chunk 3p+i (i<2) = [q_{2p+i} | k_{2p+i}], chunk 3p+2 = [v_2p | v_2p+1].
#define AB_ROWS 64
#define AB_WQ 0          // Wqkv fwd frag (N=384 reordered, K=128): 24576 + 768
#define AB_WQS 24576
#define AB_WOT 25344     // Wo^T frag (N=128 in, K=128 out): 8192 + 256
#define AB_WOTS 33536
#define AB_WQT 33792     // Wqkv^T frag (N=128, K=384 reordered+pi): 24576 + 768
#define AB_WQTS 58368
#define AB_DQF 59136     // dqkv A frags: 4 mtiles x 6 chunks x 32 lanes x uint4 = 12288
#define AB_DQS 71424     // sfa: 4 x 6 x 32 x u32 = 3072
#define AB_RED 74496     // LN reduction scratch: 64 rows x 2 halves x 2 floats = 1024
#define AB_XF 75520      // LN1(X) A frags: 4 mtiles x 2 ks x 32 lanes x uint4 = 4096
#define AB_XS 79616      // their sfa: 4 x 2 x 32 x u32 = 1024
#define AB_DF 80640      // Q(DX1) A frags = 4096
#define AB_DS 84736      // sfa = 1024
#define AB_ROWST 85760   // per row: gsr, mean, rstd, gsd (64 x 4 floats = 1024)
#define ATTB_SMEM 86784


__device__ __forceinline__ void ln_stats(const __nv_bfloat16* X, int r, int t, float& mean, float& rstd) {
  float v[4][8], s = 0.f, s2 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    ld8bf(X + (long long)r * DM + j * 32 + t * 8, v[j]);
#pragma unroll
    for (int i = 0; i < 8; i++) { s += v[j][i]; s2 += v[j][i] * v[j][i]; }
  }
  s = qsum(s); s2 = qsum(s2);
  mean = s * (1.f / DM); rstd = rsqrtf(fmaxf(s2 * (1.f / DM) - mean * mean, 0.f) + 1e-5f);
}

// quantize 4 C-layout ntiles (one 32-col half of a 64-chunk) with a fixed inverse scale (fast path)
__device__ __forceinline__ void quant_half2(float (*x)[4], float inv, int t, u32& rs, u32& w0, u32& w1, u32& sf16) {
  float r0[8], r1[8], m0 = 0.f, m1 = 0.f;
#pragma unroll
  for (int n = 0; n < 4; n++) {
    r0[2 * n] = x[n][0]; r0[2 * n + 1] = x[n][1]; r1[2 * n] = x[n][2]; r1[2 * n + 1] = x[n][3];
    m0 = fmaxf(m0, fmaxf(fabsf(x[n][0]), fabsf(x[n][1]))); m1 = fmaxf(m1, fmaxf(fabsf(x[n][2]), fabsf(x[n][3])));
  }
  __half2 h = __halves2half2(__float2half_ru(m0), __float2half_ru(m1));
  u32 hu = *reinterpret_cast<u32*>(&h), ho = __shfl_xor_sync(0xffffffffu, hu, 1);
  float2 mm = __half22float2(__hmax2(h, *reinterpret_cast<__half2*>(&ho)));
  u32 sb = e4m3x2(mm.x * inv * (1.f / 6.f), mm.y * inv * (1.f / 6.f));
  float2 dec = e4m3x2_dec(sb);
  w0 = pack8(r0, dec.x > 0.f ? inv / dec.x : 0.f, true, rs);
  w1 = pack8(r1, dec.y > 0.f ? inv / dec.y : 0.f, true, rs);
  u32 s0 = sb & 0xFF, s1 = sb >> 8;
  u32 mine = (t & 1) ? s1 : s0;
  u32 oth = __shfl_xor_sync(0xffffffffu, mine, 2);
  sf16 = mine | (oth << 8);
}
// quantize 4 C-layout ntiles (one 32-col half of a 64-chunk) with a fixed inverse scale.
// returns nibble words for rows g / g+8 and the 16-bit sfa half for this lane.
__device__ __forceinline__ void quant_half(float (*x)[4], float inv, int t, u32& rs, u32& w0, u32& w1, u32& sf16) {
  float r0[8], r1[8];
#pragma unroll
  for (int n = 0; n < 4; n++) { r0[2 * n] = x[n][0]; r0[2 * n + 1] = x[n][1]; r1[2 * n] = x[n][2]; r1[2 * n + 1] = x[n][3]; }
  u32 s0, s1;
  w0 = qgroup8(r0, inv, s0, true, rs);
  w1 = qgroup8(r1, inv, s1, true, rs);
  u32 mine = (t & 1) ? s1 : s0;
  u32 oth = __shfl_xor_sync(0xffffffffu, mine, 2);
  sf16 = mine | (oth << 8);
}

extern "C" __global__ void __launch_bounds__(256, 1) attn_bwd(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ DX1, __nv_bfloat16* __restrict__ DX,
    const float* __restrict__ lnw, float* __restrict__ dlnw,
    const uint2* __restrict__ wqd, const u32* __restrict__ wqs, const uint2* __restrict__ wqtd, const u32* __restrict__ wqts,
    const float* __restrict__ wqamax, const uint2* __restrict__ wotd, const u32* __restrict__ wots, const float* __restrict__ woamax,
    unsigned char* __restrict__ dxT, unsigned char* __restrict__ dxTs, unsigned char* __restrict__ oT, unsigned char* __restrict__ oTs,
    unsigned char* __restrict__ dqT, unsigned char* __restrict__ dqTs, unsigned char* __restrict__ hT, unsigned char* __restrict__ hTs,
    const float* __restrict__ amax_use, unsigned* __restrict__ amax_cur, int M, const u32* __restrict__ rngp) {
  u32 seed = rngp[0] ^ (u32)(size_t)amax_cur, signs = rngp[1];
  // amax slots: 0 dx1(wgrad) 1 o(wgrad) 2 dqkv(wgrad) 3 h1(wgrad) 4 dqkv(dgrad operand)
  extern __shared__ __align__(16) unsigned char sm[];
  smem_copy(sm + AB_WQ, wqd, 24576); smem_copy(sm + AB_WQS, wqs, 768);
  smem_copy(sm + AB_WOT, wotd, 8192); smem_copy(sm + AB_WOTS, wots, 256);
  smem_copy(sm + AB_WQT, wqtd, 24576); smem_copy(sm + AB_WQTS, wqts, 768);
  __syncthreads();
  const uint2* WQ = (const uint2*)(sm + AB_WQ); const u32* SQ = (const u32*)(sm + AB_WQS);
  const uint2* WOT = (const uint2*)(sm + AB_WOT); const u32* SOT = (const u32*)(sm + AB_WOTS);
  const uint2* WQT = (const uint2*)(sm + AB_WQT); const u32* SQT = (const u32*)(sm + AB_WQTS);
  uint4* DQF = (uint4*)(sm + AB_DQF); u32* DQS = (u32*)(sm + AB_DQS); float* RED = (float*)(sm + AB_RED);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  float gq = *wqamax * (1.f / FP4_E4M3), go = *woamax * (1.f / FP4_E4M3);
  float inv_dx = FP4_E4M3 / amax_use[0], inv_o = FP4_E4M3 / amax_use[1], inv_dqw = FP4_E4M3 / amax_use[2];
  float inv_h = FP4_E4M3 / amax_use[3], inv_dq = FP4_E4M3 / amax_use[4], gdq = amax_use[4] * (1.f / FP4_E4M3);
  RHT R = make_rht(signs, g, t);
  u32 rs = hash32(seed ^ (blockIdx.x * 256 + threadIdx.x) * 0x9E3779B9u);
  float mx_dx = 0.f, mx_o = 0.f, mx_dqw = 0.f, mx_h = 0.f, mx_dq = 0.f;
  float dg[2] = {0.f, 0.f};

#ifdef KPROF
#define QT(i) do { long long _n = clock64(); if (threadIdx.x == 0) atomicAdd(&g_prof[16 + i], (unsigned long long)(_n - _qt)); _qt = _n; } while (0)
  long long _qt = clock64();
#else
#define QT(i)
#endif
  uint4* XF = (uint4*)(sm + AB_XF); u32* XS = (u32*)(sm + AB_XS);
  uint4* DF = (uint4*)(sm + AB_DF); u32* DS = (u32*)(sm + AB_DS); float* ROWST = (float*)(sm + AB_ROWST);
  for (int tile = blockIdx.x; tile < M / AB_ROWS; tile += gridDim.x) {
    int t0 = tile * AB_ROWS;
    {  // ================= phase 0: LN1(X) and Q(DX1) once per 16-row tile -> smem =================
      int mt = warp & 3, lr0 = mt * 16 + g, r0 = t0 + lr0, r1 = r0 + 8;
      if (warp < 4) {
        AFrag hq[2]; float gsr[2], mean[2], rstd[2], hn[2];
        ln_quant(X, r0, r1, t, lnw, hq, gsr, mean, rstd, hn);
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          XF[(mt * 2 + ks) * 32 + lane] = make_uint4(hq[ks].a0, hq[ks].a1, hq[ks].a2, hq[ks].a3);
          XS[(mt * 2 + ks) * 32 + lane] = hq[ks].sf;
        }
        if (t == 0) {
          ROWST[lr0 * 4 + 0] = gsr[0]; ROWST[lr0 * 4 + 1] = mean[0]; ROWST[lr0 * 4 + 2] = rstd[0];
          ROWST[(lr0 + 8) * 4 + 0] = gsr[1]; ROWST[(lr0 + 8) * 4 + 1] = mean[1]; ROWST[(lr0 + 8) * 4 + 2] = rstd[1];
        }
      } else {
        float va[2][16], vb[2][16], am0 = 0.f, am1 = 0.f;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          ld8bf(DX1 + (long long)r0 * DM + ks * 64 + t * 8, va[ks]); ld8bf(DX1 + (long long)r0 * DM + ks * 64 + 32 + t * 8, va[ks] + 8);
          ld8bf(DX1 + (long long)r1 * DM + ks * 64 + t * 8, vb[ks]); ld8bf(DX1 + (long long)r1 * DM + ks * 64 + 32 + t * 8, vb[ks] + 8);
#pragma unroll
          for (int i = 0; i < 16; i++) { am0 = fmaxf(am0, fabsf(va[ks][i])); am1 = fmaxf(am1, fabsf(vb[ks][i])); }
        }
        am0 = qmax(am0); am1 = qmax(am1);
        float i0 = am0 > 0.f ? FP4_E4M3 / am0 : 0.f, i1 = am1 > 0.f ? FP4_E4M3 / am1 : 0.f;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          AFrag f = quant_afrag2(va[ks], vb[ks], i0, i1, t, true, rs);
          DF[(mt * 2 + ks) * 32 + lane] = make_uint4(f.a0, f.a1, f.a2, f.a3);
          DS[(mt * 2 + ks) * 32 + lane] = f.sf;
        }
        if (t == 0) { ROWST[lr0 * 4 + 3] = am0 * (1.f / FP4_E4M3); ROWST[(lr0 + 8) * 4 + 3] = am1 * (1.f / FP4_E4M3); }
      }
    }
    __syncthreads();
    QT(0);
    {  // ================= phase A =================
      int s = warp >> 2, head = warp & 3, base = t0 + s * SEQ;
      float q[2][4][4], k[2][4][4], v[2][4][4], p[2][4][4], dO[2][4][4];
      {
        AFrag hq[2][2]; float gsr[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
#pragma unroll
          for (int ks = 0; ks < 2; ks++) {
            int i = ((s * 2 + mt) * 2 + ks) * 32 + lane;
            uint4 a = XF[i]; hq[mt][ks].a0 = a.x; hq[mt][ks].a1 = a.y; hq[mt][ks].a2 = a.z; hq[mt][ks].a3 = a.w; hq[mt][ks].sf = XS[i];
          }
          int lr = (s * 2 + mt) * 16 + g;
          gsr[mt][0] = ROWST[lr * 4]; gsr[mt][1] = ROWST[(lr + 8) * 4];
        }
#pragma unroll
        for (int which = 0; which < 3; which++) {
          float (*dst)[4][4] = which == 0 ? q : (which == 1 ? k : v);
#pragma unroll
          for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int n = 0; n < 4; n++) {
              dst[mt][n][0] = dst[mt][n][1] = dst[mt][n][2] = dst[mt][n][3] = 0.f;
              int nt = qkv_ntile(which, head, n);
#pragma unroll
              for (int ks = 0; ks < 2; ks++) {
                int wi = nt * 2 + ks;
                mma4(dst[mt][n], hq[mt][ks].a0, hq[mt][ks].a1, hq[mt][ks].a2, hq[mt][ks].a3, WQ[wi * 32 + lane], hq[mt][ks].sf, SQ[wi * 2 + (g >> 2)]);
              }
              float s0 = gsr[mt][0] * gq, s1 = gsr[mt][1] * gq;
              dst[mt][n][0] *= s0; dst[mt][n][1] *= s0; dst[mt][n][2] *= s1; dst[mt][n][3] *= s1;
            }
        }
      }
      QT(1);
      {  // O recompute -> wgrad operand for Wo (features = head*32 + d)
        float o[2][4][4];
        attn_head(q, k, v, o, p, g, t);
        QT(2);
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
          for (int pr = 0; pr < 2; pr++)
            mx_o = fmaxf(mx_o, emit16(o[mt][2 * pr], o[mt][2 * pr + 1], head * HD + pr * 16, base + mt * 16, M, R, inv_o, false, rs, oT, oTs, g, t));
      }
      QT(3);
      {  // dO = Q(DX1) Wo  (fragments quantized once in phase 0)
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          int fi = ((s * 2 + mt) * 2) * 32 + lane, lr = (s * 2 + mt) * 16 + g;
          uint4 d0 = DF[fi], d1 = DF[fi + 32];
          AFrag dxq0 = {d0.x, d0.y, d0.z, d0.w, DS[fi]}, dxq1 = {d1.x, d1.y, d1.z, d1.w, DS[fi + 32]};
          float s0 = ROWST[lr * 4 + 3] * go, s1 = ROWST[(lr + 8) * 4 + 3] * go;
#pragma unroll
          for (int n = 0; n < 4; n++) {
            dO[mt][n][0] = dO[mt][n][1] = dO[mt][n][2] = dO[mt][n][3] = 0.f;
            int nt = head * 4 + n;
            mma4(dO[mt][n], dxq0.a0, dxq0.a1, dxq0.a2, dxq0.a3, WOT[(nt * 2) * 32 + lane], dxq0.sf, SOT[(nt * 2) * 2 + (g >> 2)]);
            mma4(dO[mt][n], dxq1.a0, dxq1.a1, dxq1.a2, dxq1.a3, WOT[(nt * 2 + 1) * 32 + lane], dxq1.sf, SOT[(nt * 2 + 1) * 2 + (g >> 2)]);
            dO[mt][n][0] *= s0; dO[mt][n][1] *= s0; dO[mt][n][2] *= s1; dO[mt][n][3] *= s1;
          }
        }
      }
      QT(4);
      // dP = dO V^T  (same structure as S = Q K^T)
      float dp[2][4][4];
#pragma unroll
      for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int kt = 0; kt < 4; kt++) {
          dp[mt][kt][0] = dp[mt][kt][1] = dp[mt][kt][2] = dp[mt][kt][3] = 0.f;
          int kmt = kt >> 1, kh = (kt & 1) * 2;
#pragma unroll
          for (int ks = 0; ks < 2; ks++) {
            u32 a0 = packbf(dO[mt][2 * ks][0], dO[mt][2 * ks][1]), a1 = packbf(dO[mt][2 * ks][2], dO[mt][2 * ks][3]);
            u32 a2 = packbf(dO[mt][2 * ks + 1][0], dO[mt][2 * ks + 1][1]), a3 = packbf(dO[mt][2 * ks + 1][2], dO[mt][2 * ks + 1][3]);
            u32 b0 = packbf(v[kmt][2 * ks][kh], v[kmt][2 * ks][kh + 1]), b1 = packbf(v[kmt][2 * ks + 1][kh], v[kmt][2 * ks + 1][kh + 1]);
            mma_bf16_acc(dp[mt][kt], a0, a1, a2, a3, b0, b1);
          }
        }
      // dS = P (dP - rowsum(P dP))
#pragma unroll
      for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float d = 0.f;
#pragma unroll
          for (int kt = 0; kt < 4; kt++) d += p[mt][kt][2 * h] * dp[mt][kt][2 * h] + p[mt][kt][2 * h + 1] * dp[mt][kt][2 * h + 1];
          d = qsum(d);
#pragma unroll
          for (int kt = 0; kt < 4; kt++)
#pragma unroll
            for (int e = 0; e < 2; e++) dp[mt][kt][2 * h + e] = p[mt][kt][2 * h + e] * (dp[mt][kt][2 * h + e] - d) * ATT_SCALE;
        }
      QT(5);
      // dV = P^T dO ; dK = dS^T Q  (A = transposed probs, B = transposed dO / Q); dQ = dS K
      float dv[2][4][4], dk[2][4][4], dq[2][4][4];
#pragma unroll
      for (int km = 0; km < 2; km++)
#pragma unroll
        for (int n = 0; n < 4; n++) {
          dv[km][n][0] = dv[km][n][1] = dv[km][n][2] = dv[km][n][3] = 0.f;
          dk[km][n][0] = dk[km][n][1] = dk[km][n][2] = dk[km][n][3] = 0.f;
        }
#pragma unroll
      for (int kq = 0; kq < 2; kq++)      // query 16-chunk (contraction)
#pragma unroll
        for (int km = 0; km < 2; km++) {  // key m-tile
          if (km > kq) continue;          // fully masked block
          u32 pa0 = movtrans(packbf(p[kq][2 * km][0], p[kq][2 * km][1])), pa1 = movtrans(packbf(p[kq][2 * km + 1][0], p[kq][2 * km + 1][1]));
          u32 pa2 = movtrans(packbf(p[kq][2 * km][2], p[kq][2 * km][3])), pa3 = movtrans(packbf(p[kq][2 * km + 1][2], p[kq][2 * km + 1][3]));
          u32 sa0 = movtrans(packbf(dp[kq][2 * km][0], dp[kq][2 * km][1])), sa1 = movtrans(packbf(dp[kq][2 * km + 1][0], dp[kq][2 * km + 1][1]));
          u32 sa2 = movtrans(packbf(dp[kq][2 * km][2], dp[kq][2 * km][3])), sa3 = movtrans(packbf(dp[kq][2 * km + 1][2], dp[kq][2 * km + 1][3]));
#pragma unroll
          for (int n = 0; n < 4; n++) {
            u32 ob0 = movtrans(packbf(dO[kq][n][0], dO[kq][n][1])), ob1 = movtrans(packbf(dO[kq][n][2], dO[kq][n][3]));
            mma_bf16_acc(dv[km][n], pa0, pa1, pa2, pa3, ob0, ob1);
            u32 qb0 = movtrans(packbf(q[kq][n][0], q[kq][n][1])), qb1 = movtrans(packbf(q[kq][n][2], q[kq][n][3]));
            mma_bf16_acc(dk[km][n], sa0, sa1, sa2, sa3, qb0, qb1);
          }
        }
#pragma unroll
      for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int n = 0; n < 4; n++) {
          dq[mt][n][0] = dq[mt][n][1] = dq[mt][n][2] = dq[mt][n][3] = 0.f;
#pragma unroll
          for (int ks = 0; ks < 2; ks++) {  // key 16-chunk
            if (ks > mt) continue;
            u32 a0 = packbf(dp[mt][2 * ks][0], dp[mt][2 * ks][1]), a1 = packbf(dp[mt][2 * ks][2], dp[mt][2 * ks][3]);
            u32 a2 = packbf(dp[mt][2 * ks + 1][0], dp[mt][2 * ks + 1][1]), a3 = packbf(dp[mt][2 * ks + 1][2], dp[mt][2 * ks + 1][3]);
            u32 b0 = movtrans(packbf(k[ks][n][0], k[ks][n][1])), b1 = movtrans(packbf(k[ks][n][2], k[ks][n][3]));
            mma_bf16_acc(dq[mt][n], a0, a1, a2, a3, b0, b1);
          }
        }
      QT(6);
      // ---- emit wgrad operands dq/dk/dv (features in natural 384 order), SR
#pragma unroll
      for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int pr = 0; pr < 2; pr++) {
          int f = head * HD + pr * 16, tk = base + mt * 16;
          mx_dqw = fmaxf(mx_dqw, emit16(dq[mt][2 * pr], dq[mt][2 * pr + 1], f, tk, M, R, inv_dqw, true, rs, dqT, dqTs, g, t));
          mx_dqw = fmaxf(mx_dqw, emit16(dk[mt][2 * pr], dk[mt][2 * pr + 1], 128 + f, tk, M, R, inv_dqw, true, rs, dqT, dqTs, g, t));
          mx_dqw = fmaxf(mx_dqw, emit16(dv[mt][2 * pr], dv[mt][2 * pr + 1], 256 + f, tk, M, R, inv_dqw, true, rs, dqT, dqTs, g, t));
        }
      QT(7);
      // ---- dgrad operand: quantize dqkv into smem A fragments (per-tensor delayed scale, SR)
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        int mtile = s * 2 + mt;
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
          for (int e = 0; e < 4; e++) mx_dq = fmaxf(mx_dq, fmaxf(fabsf(dq[mt][n][e]), fmaxf(fabsf(dk[mt][n][e]), fabsf(dv[mt][n][e]))));
        u32 w0, w1, w2, w3, sflo, sfhi;
        quant_half2(dq[mt], inv_dq, t, rs, w0, w1, sflo);
        quant_half2(dk[mt], inv_dq, t, rs, w2, w3, sfhi);
        int ci = (mtile * 6 + 3 * (head >> 1) + (head & 1)) * 32 + lane;
        DQF[ci] = make_uint4(w0, w1, w2, w3);
        DQS[ci] = sflo | (sfhi << 16);
        quant_half2(dv[mt], inv_dq, t, rs, w0, w1, sflo);
        int cv = (mtile * 6 + 3 * (head >> 1) + 2) * 32 + lane;
        if (head & 1) {
          reinterpret_cast<uint2*>(&DQF[cv])[1] = make_uint2(w0, w1);
          reinterpret_cast<unsigned short*>(&DQS[cv])[1] = (unsigned short)sflo;
        } else {
          reinterpret_cast<uint2*>(&DQF[cv])[0] = make_uint2(w0, w1);
          reinterpret_cast<unsigned short*>(&DQS[cv])[0] = (unsigned short)sflo;
        }
      }
      QT(8);
    }
    __syncthreads();
    QT(9);
    {  // ================= phase B =================
      int mt = warp & 3, nh = warp >> 2;
      int lr0 = mt * 16 + g, lr1 = lr0 + 8, r0 = t0 + lr0, r1 = t0 + lr1;
      float dh[8][4];
#pragma unroll
      for (int n = 0; n < 8; n++) dh[n][0] = dh[n][1] = dh[n][2] = dh[n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 6; ks++) {
        uint4 a = DQF[(mt * 6 + ks) * 32 + lane];
        u32 sf = DQS[(mt * 6 + ks) * 32 + lane];
#pragma unroll
        for (int n = 0; n < 8; n++) {
          int wi = (nh * 8 + n) * 6 + ks;
          mma4(dh[n], a.x, a.y, a.z, a.w, WQT[wi * 32 + lane], sf, SQT[wi * 2 + (g >> 2)]);
        }
      }
      QT(10);
      float fs = gdq * gq;
      float mean0 = ROWST[lr0 * 4 + 1], rstd0 = ROWST[lr0 * 4 + 2], mean1 = ROWST[lr1 * 4 + 1], rstd1 = ROWST[lr1 * 4 + 2];
      float xh[8][4], s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f};
#pragma unroll
      for (int n = 0; n < 8; n++) {
        int col = (nh * 8 + n) * 8 + 2 * t;
        float2 x0 = __bfloat1622float2(*(const __nv_bfloat162*)(X + (long long)r0 * DM + col));
        float2 x1 = __bfloat1622float2(*(const __nv_bfloat162*)(X + (long long)r1 * DM + col));
        xh[n][0] = (x0.x - mean0) * rstd0; xh[n][1] = (x0.y - mean0) * rstd0;
        xh[n][2] = (x1.x - mean1) * rstd1; xh[n][3] = (x1.y - mean1) * rstd1;
#pragma unroll
        for (int e = 0; e < 4; e++) dh[n][e] *= fs;
        float p0 = dh[n][0] * xh[n][0] + dh[n][2] * xh[n][2], p1 = dh[n][1] * xh[n][1] + dh[n][3] * xh[n][3];
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) { p0 += __shfl_xor_sync(0xffffffffu, p0, o); p1 += __shfl_xor_sync(0xffffffffu, p1, o); }
        if (n == g) { dg[0] += p0; dg[1] += p1; }
        float g0 = lnw[col], g1 = lnw[col + 1];
        dh[n][0] *= g0; dh[n][1] *= g1; dh[n][2] *= g0; dh[n][3] *= g1;
        s1[0] += dh[n][0] + dh[n][1]; s1[1] += dh[n][2] + dh[n][3];
        s2[0] += dh[n][0] * xh[n][0] + dh[n][1] * xh[n][1]; s2[1] += dh[n][2] * xh[n][2] + dh[n][3] * xh[n][3];
      }
#pragma unroll
      for (int h = 0; h < 2; h++) { s1[h] = qsum(s1[h]); s2[h] = qsum(s2[h]); }
      if (t == 0) {
        RED[(lr0 * 2 + nh) * 2] = s1[0]; RED[(lr0 * 2 + nh) * 2 + 1] = s2[0];
        RED[(lr1 * 2 + nh) * 2] = s1[1]; RED[(lr1 * 2 + nh) * 2 + 1] = s2[1];
      }
      __syncthreads();
      float S10 = (RED[(lr0 * 2) * 2] + RED[(lr0 * 2 + 1) * 2]) * (1.f / DM), S20 = (RED[(lr0 * 2) * 2 + 1] + RED[(lr0 * 2 + 1) * 2 + 1]) * (1.f / DM);
      float S11 = (RED[(lr1 * 2) * 2] + RED[(lr1 * 2 + 1) * 2]) * (1.f / DM), S21 = (RED[(lr1 * 2) * 2 + 1] + RED[(lr1 * 2 + 1) * 2 + 1]) * (1.f / DM);
      float dxv[8][4];
#pragma unroll
      for (int n = 0; n < 8; n++) {
        int col = (nh * 8 + n) * 8 + 2 * t;
        float2 y0 = __bfloat1622float2(*(const __nv_bfloat162*)(DX1 + (long long)r0 * DM + col));
        float2 y1 = __bfloat1622float2(*(const __nv_bfloat162*)(DX1 + (long long)r1 * DM + col));
        dxv[n][0] = y0.x; dxv[n][1] = y0.y; dxv[n][2] = y1.x; dxv[n][3] = y1.y;
        float g0 = lnw[col], g1 = lnw[col + 1];
        float o0 = y0.x + rstd0 * (dh[n][0] - S10 - xh[n][0] * S20), o1 = y0.y + rstd0 * (dh[n][1] - S10 - xh[n][1] * S20);
        float o2 = y1.x + rstd1 * (dh[n][2] - S11 - xh[n][2] * S21), o3 = y1.y + rstd1 * (dh[n][3] - S11 - xh[n][3] * S21);
        xh[n][0] *= g0; xh[n][1] *= g1; xh[n][2] *= g0; xh[n][3] *= g1;  // h1 = xhat*g (wgrad operand)
        *(__nv_bfloat162*)(DX + (long long)r0 * DM + col) = __floats2bfloat162_rn(o0, o1);
        *(__nv_bfloat162*)(DX + (long long)r1 * DM + col) = __floats2bfloat162_rn(o2, o3);
      }
      QT(11);
#pragma unroll
      for (int pr = 0; pr < 4; pr++) {
        int f = nh * 64 + pr * 16, tk = t0 + mt * 16;
        mx_h = fmaxf(mx_h, emit16(xh[2 * pr], xh[2 * pr + 1], f, tk, M, R, inv_h, false, rs, hT, hTs, g, t));
        mx_dx = fmaxf(mx_dx, emit16(dxv[2 * pr], dxv[2 * pr + 1], f, tk, M, R, inv_dx, true, rs, dxT, dxTs, g, t));
      }
      QT(12);
    }
    __syncthreads();
    QT(13);
  }
  {
    int nh = warp >> 2;
#pragma unroll
    for (int e = 0; e < 2; e++) atomicAdd(dlnw + (nh * 8 + g) * 8 + 2 * t + e, dg[e]);
  }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    mx_dx = fmaxf(mx_dx, __shfl_xor_sync(0xffffffffu, mx_dx, o)); mx_o = fmaxf(mx_o, __shfl_xor_sync(0xffffffffu, mx_o, o));
    mx_dqw = fmaxf(mx_dqw, __shfl_xor_sync(0xffffffffu, mx_dqw, o)); mx_h = fmaxf(mx_h, __shfl_xor_sync(0xffffffffu, mx_h, o));
    mx_dq = fmaxf(mx_dq, __shfl_xor_sync(0xffffffffu, mx_dq, o));
  }
  if (lane == 0) {
    atomicMax(amax_cur + 0, __float_as_uint(mx_dx)); atomicMax(amax_cur + 1, __float_as_uint(mx_o));
    atomicMax(amax_cur + 2, __float_as_uint(mx_dqw)); atomicMax(amax_cur + 3, __float_as_uint(mx_h));
    atomicMax(amax_cur + 4, __float_as_uint(mx_dq));
  }
}
