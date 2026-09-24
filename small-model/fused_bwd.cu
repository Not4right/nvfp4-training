// ================= wgrad operand emission (RHT on tensor cores) =================
// Takes a 16-token x 16-feature block held in C-layout (two 8-col ntiles, rows g/g+8), transposes it
// in registers (movmatrix), applies the 16-point random Hadamard along tokens with a bf16 mma, and
// writes it quantized along tokens: Q [F, M/2] (fp4), S [F, M/16] (ue4m3).
__device__ __forceinline__ u32 packbf(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<u32*>(&h);
}
__device__ __forceinline__ u32 movtrans(u32 x) {
  u32 r; asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(x)); return r;
}
__device__ __forceinline__ void mma_bf16(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}
struct RHT { u32 b[2][2]; };

__device__ __forceinline__ float hs(u32 signs, int k, int n) {
  float v = (__popc(k & n) & 1) ? -0.25f : 0.25f;
  return ((signs >> k) & 1) ? -v : v;
}
__device__ __forceinline__ RHT make_rht(u32 signs, int g, int t) {
  RHT r;
#pragma unroll
  for (int nh = 0; nh < 2; nh++)
#pragma unroll
    for (int q = 0; q < 2; q++) {
      int n = nh * 8 + g, k0 = 2 * t + 8 * q;
      r.b[nh][q] = packbf(hs(signs, k0, n), hs(signs, k0 + 1, n));
    }
  return r;
}
// xa/xb: ntile A / B values {row g c0, row g c1, row g+8 c0, row g+8 c1}; features fa+{0..7}, fb+{0..7}.
// tok0: first token (multiple of 16).  inv: 2688/amax_use.  Returns max |RHT value| seen.
__device__ __forceinline__ float emit_wg(const float* xa, const float* xb, int fa, int fb, int tok0, int M,
                                         const RHT& R, float inv, bool sr, u32& rs,
                                         unsigned char* __restrict__ Q, unsigned char* __restrict__ S, int g, int t, int lane) {
  u32 a0 = movtrans(packbf(xa[0], xa[1])), a2 = movtrans(packbf(xa[2], xa[3]));
  u32 a1 = movtrans(packbf(xb[0], xb[1])), a3 = movtrans(packbf(xb[2], xb[3]));
  float d0[4], d1[4];
  mma_bf16(d0, a0, a1, a2, a3, R.b[0][0], R.b[0][1]);
  mma_bf16(d1, a0, a1, a2, a3, R.b[1][0], R.b[1][1]);
  // row g -> feature fa+g, row g+8 -> fb+g ; tokens (2t,2t+1) in d0, (8+2t,9+2t) in d1
  float m0 = fmaxf(fmaxf(fabsf(d0[0]), fabsf(d0[1])), fmaxf(fabsf(d1[0]), fabsf(d1[1])));
  float m1 = fmaxf(fmaxf(fabsf(d0[2]), fabsf(d0[3])), fmaxf(fabsf(d1[2]), fabsf(d1[3])));
  m0 = qmax(m0); m1 = qmax(m1);
  u32 sb0 = e4m3_enc(m0 * inv * (1.f / 6.f)), sb1 = e4m3_enc(m1 * inv * (1.f / 6.f));
  float k0 = sb0 ? inv / e4m3_dec(sb0) : 0.f, k1 = sb1 ? inv / e4m3_dec(sb1) : 0.f;
  float v[8] = {d0[0] * k0, d0[1] * k0, d1[0] * k0, d1[1] * k0, d0[2] * k1, d0[3] * k1, d1[2] * k1, d1[3] * k1};
  if (sr) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      u32 r = rng_next(rs);
      v[i] = sr_fp4(v[i], (r << 11) & 0x3FF800u); v[i + 1] = sr_fp4(v[i + 1], (r >> 5) & 0x3FF800u);
    }
  }
  // stored token order inside the 16-group: nibble 4t..4t+3 = tokens 2t,2t+1,8+2t,9+2t (same for all
  // wgrad operands, so the token-contraction is unchanged).  Thread t owns bytes 2t,2t+1.
  u32 va = fp4x2(v[0], v[1]) | (fp4x2(v[2], v[3]) << 8), vb = fp4x2(v[4], v[5]) | (fp4x2(v[6], v[7]) << 8);
  *reinterpret_cast<unsigned short*>(Q + (long long)(fa + g) * (M / 2) + tok0 / 2 + 2 * t) = (unsigned short)va;
  *reinterpret_cast<unsigned short*>(Q + (long long)(fb + g) * (M / 2) + tok0 / 2 + 2 * t) = (unsigned short)vb;
  if (t < 2) S[(long long)((t ? fb : fa) + g) * (M / 16) + tok0 / 16] = (unsigned char)(t ? sb1 : sb0);
  return fmaxf(m0, m1);
}

