// NVFP4 kernels for sm_120a (RTX 50xx): quantizers + block-scaled FP4 tensor-core GEMM.
// Format: E2M1 values packed 2/byte (low nibble = lower index), one UE4M3 scale per 16
// values along the contraction dim, one FP32 per-tensor scale gs = amax / (6*448).
#include <cuda_bf16.h>
typedef unsigned long long u64;
typedef unsigned u32;

#define FP4_E4M3 2688.0f   // 6 * 448

__device__ __forceinline__ u32 hash32(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
// e4m3 encode (round-nearest, saturating) of a positive float -> byte, plus its decoded value
__device__ __forceinline__ u32 e4m3_enc(float s) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(s));
  return r & 0xFF;
}
__device__ __forceinline__ float e4m3_dec(u32 b) {
  unsigned short h = (unsigned short)b;
  u32 r; asm("{.reg .b32 t; cvt.rn.f16x2.e4m3x2 t, %1; mov.b32 %0, t;}" : "=r"(r) : "h"(h));
  __half2 hv = *reinterpret_cast<__half2*>(&r);
  return __low2float(hv);
}
// two floats (already in [-6,6] units) -> byte: a in low nibble, b in high
__device__ __forceinline__ u32 fp4x2(float a, float b) {
  unsigned short r;
  asm("{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t;}" : "=h"(r) : "f"(b), "f"(a));
  return r;
}
// stochastic rounding onto the E2M1 grid by mantissa dithering: after this the value is
// exactly representable so the RN convert is exact.  v in fp4 units.
__device__ __forceinline__ float sr_fp4(float v, u32 rnd) {
  float a = fminf(fabsf(v), 6.f);
  bool sub = a < 1.f;
  float b = sub ? a + 1.f : a;
  u32 bits = (__float_as_uint(b) + (rnd & 0x3FFFFFu)) & ~0x3FFFFFu;
  float r = __uint_as_float(bits) - (sub ? 1.f : 0.f);
  return copysignf(fminf(r, 6.f), v);
}

// quantize 16 floats (one block) -> 8 bytes + scale byte.  gs = global scale.
__device__ __forceinline__ void quant16(const float* x, float inv_gs, bool sr, u32 seed, u32 idx,
                                        uint2& out, u32& sbyte) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[i]));
  u32 sb = e4m3_enc(amax * inv_gs * (1.f / 6.f));
  float sdec = e4m3_dec(sb);
  float k = sdec > 0.f ? inv_gs / sdec : 0.f;
  u32 w[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    float a = x[i] * k, b = x[i + 1] * k;
    if (sr) {
      u32 h = hash32(seed ^ (idx + i) * 0x9E3779B9u);
      a = sr_fp4(a, h); b = sr_fp4(b, h >> 10 | h << 22);
    }
    w[i >> 3] |= fp4x2(a, b) << ((i & 7) * 4);
  }
  out = make_uint2(w[0], w[1]);
  sbyte = sb;
}

// ---- row quantizer: X [R, C] bf16 -> Q [R, C/2], S [R, C/16].  One thread per 16-block.
extern "C" __global__ void quant_rows(const __nv_bfloat16* __restrict__ X, u32* __restrict__ Q,
                                      unsigned char* __restrict__ S, const float* __restrict__ amax,
                                      int nblk, int sr, u32 seed) {
  long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk) return;
  float am = *amax;
  float inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  const uint4* p = reinterpret_cast<const uint4*>(X + b * 16);
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
  quant16(x, inv_gs, sr, seed, (u32)(b * 16), o, sb);
  reinterpret_cast<uint2*>(Q)[b] = o;
  S[b] = (unsigned char)sb;
}

// ---- column quantizer with 16-point random Hadamard along rows (tokens):
// X [M, C] bf16 -> Q [C, M/2], S [C, M/16].  Thread (c, mblock).  amax is |X| amax; the
// transformed values are bounded by 4*amax so we use that for the global scale.
extern "C" __global__ void quant_cols_rht(const __nv_bfloat16* __restrict__ X, u32* __restrict__ Q,
                                          unsigned char* __restrict__ S, const float* __restrict__ amax,
                                          int M, int C, u32 signs, int sr, u32 seed) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int mb = blockIdx.y;
  if (c >= C) return;
  float am = *amax * 4.f;
  float inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  float x[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    float v = __bfloat162float(X[(long long)(mb * 16 + i) * C + c]);
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
  quant16(x, inv_gs, sr, seed, (u32)(c * M + mb * 16), o, sb);
  reinterpret_cast<uint2*>(Q + ((long long)c * M) / 8)[mb] = o;
  S[(long long)c * (M / 16) + mb] = (unsigned char)sb;
}

// ---- 2D 16x16 weight quantizer: W [N, K] fp32 -> Q [N, K/2] + S [N, K/16] and the same
// values transposed QT [K, N/2] + ST [K, N/16].  One warp per 16x16 block, lane = row pair.
extern "C" __global__ void quant_w2d(const float* __restrict__ W, u32* __restrict__ Q, unsigned char* __restrict__ S,
                                     u32* __restrict__ QT, unsigned char* __restrict__ ST,
                                     const float* __restrict__ amax, int N, int K) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  int nb = N / 16, kb = K / 16;
  if (warp >= nb * kb) return;
  int bn = warp / kb, bk = warp % kb;
  float am = *amax;
  float inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  // lane handles row r = lane>>1, 8 cols starting (lane&1)*8
  int r = lane >> 1, c0 = (lane & 1) * 8;
  float x[8], m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { x[i] = W[(long long)(bn * 16 + r) * K + bk * 16 + c0 + i]; m = fmaxf(m, fabsf(x[i])); }
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
  if ((lane & 1) == 0) S[(long long)(bn * 16 + r) * kb + bk] = (unsigned char)sb;
  // transpose via shared: block 16x16 nibbles
  __shared__ unsigned char tile[8][16][16];
  int wi = (threadIdx.x >> 5);
#pragma unroll
  for (int i = 0; i < 8; i++) tile[wi][r][c0 + i] = nib[i];
  __syncwarp();
  // lane writes transposed row rr = lane>>1 (a K index), 8 N-values from (lane&1)*8
  u32 wt = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) wt |= (u32)tile[wi][c0 + i][r] << (i * 4);
  QT[((long long)(bk * 16 + r) * N + bn * 16 + c0) / 8] = wt;
  if ((lane & 1) == 0) ST[(long long)(bk * 16 + r) * nb + bn] = (unsigned char)sb;
}

// ---- block-scaled FP4 GEMM:  C[M,N] = alpha * A[M,K] * B[N,K]^T
// A,B packed fp4 K-major; SA [M,K/16], SB [N,K/16] ue4m3.  alpha = amaxA*amaxB*ka*kb/2688^2.
// CTA 128x128, BK=128 (64 bytes/row), 8 warps as 2(M) x 4(N), warp tile 64x32, 3-stage cp.async.
// mode 0: store bf16.  mode 1: atomic-add fp32 (split-K along gridDim.z).
#define BM 128
#define BN 128
#define BKB 64            // bytes of K per stage (=128 fp4)
#define STAGES 3
struct __align__(16) Stage { unsigned char a[BM * BKB]; unsigned char b[BN * BKB]; unsigned char sa[BM * 8]; unsigned char sb[BN * 8]; };

__device__ __forceinline__ void cp16(void* s, const void* g) {
  u32 sa = (u32)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ void cp8(void* s, const void* g) {
  u32 sa = (u32)__cvta_generic_to_shared(s);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(sa), "l"(g));
}
// swizzled offset of 16B chunk ch (0..3) of row r
__device__ __forceinline__ int swz(int r, int ch) { return r * BKB + ((ch ^ ((r >> 1) & 3)) << 4); }

extern "C" __global__ void __launch_bounds__(256) gemm_fp4(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ SA,
    const unsigned char* __restrict__ B, const unsigned char* __restrict__ SB,
    void* __restrict__ C, const float* __restrict__ amaxA, const float* __restrict__ amaxB,
    float ka, float kb, int M, int N, int K, int ksplit_len, int mode, int lda_, int ldsa_, int ldb_, int ldsb_) {
  extern __shared__ __align__(16) unsigned char smem[];
  Stage* st = reinterpret_cast<Stage*>(smem);
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int wm = warp >> 2, wn = warp & 3;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int kbeg = blockIdx.z * ksplit_len, kend = min(K, kbeg + ksplit_len);
  int nk = (kend - kbeg) / 128;
  long long lda = lda_, ldsa = ldsa_, ldb = ldb_, ldsb = ldsb_;

  auto load = [&](int s, int kt) {
    int kbyte = (kbeg + kt * 128) / 2;
    Stage& S = st[s];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int id = tid + i * 256, r = id >> 2, ch = id & 3;
      cp16(S.a + swz(r, ch), A + (long long)(m0 + r) * lda + kbyte + ch * 16);
      cp16(S.b + swz(r, ch), B + (long long)(n0 + r) * ldb + kbyte + ch * 16);
    }
    int ks = (kbeg + kt * 128) / 16;
    if (tid < 128) cp8(S.sa + tid * 8, SA + (long long)(m0 + tid) * ldsa + ks);
    else cp8(S.sb + (tid - 128) * 8, SB + (long long)(n0 + tid - 128) * ldsb + ks);
  };

  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < nk) load(s, s);
    asm volatile("cp.async.commit_group;\n");
  }
  int g = lane >> 2, t = lane & 3;
  for (int kt = 0; kt < nk; kt++) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
    __syncthreads();
    if (kt + STAGES - 1 < nk) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
    asm volatile("cp.async.commit_group;\n");
    Stage& S = st[kt % STAGES];
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {  // two k64 steps per stage
      u32 af[4][4], bf[4][2], sfa[4], sfb[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int r = wm * 64 + i * 16 + g;
        af[i][0] = *(u32*)(S.a + swz(r, ks * 2) + t * 4);
        af[i][1] = *(u32*)(S.a + swz(r + 8, ks * 2) + t * 4);
        af[i][2] = *(u32*)(S.a + swz(r, ks * 2 + 1) + t * 4);
        af[i][3] = *(u32*)(S.a + swz(r + 8, ks * 2 + 1) + t * 4);
        int rs = wm * 64 + i * 16 + g + ((lane & 1) << 3);
        sfa[i] = *(u32*)(S.sa + rs * 8 + ks * 4);
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int r = wn * 32 + j * 8 + g;
        bf[j][0] = *(u32*)(S.b + swz(r, ks * 2) + t * 4);
        bf[j][1] = *(u32*)(S.b + swz(r, ks * 2 + 1) + t * 4);
        sfb[j] = *(u32*)(S.sb + r * 8 + ks * 4);
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%12}, %11, {%12,%12};\n"
            : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
            : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]),
              "r"(sfa[i]), "r"(sfb[j]), "h"((unsigned short)0));
    }
  }
  asm volatile("cp.async.wait_group 0;\n");
  float alpha = (*amaxA) * (*amaxB) * ka * kb * (1.f / (FP4_E4M3 * FP4_E4M3));
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int r = m0 + wm * 64 + i * 16 + g, c = n0 + wn * 32 + j * 8 + t * 2;
      if (mode == 0) {
        __nv_bfloat16* Cb = (__nv_bfloat16*)C;
        *(__nv_bfloat162*)(Cb + (long long)r * N + c) = __floats2bfloat162_rn(acc[i][j][0] * alpha, acc[i][j][1] * alpha);
        *(__nv_bfloat162*)(Cb + (long long)(r + 8) * N + c) = __floats2bfloat162_rn(acc[i][j][2] * alpha, acc[i][j][3] * alpha);
      } else {
        float* Cf = (float*)C;
        atomicAdd(Cf + (long long)r * N + c, acc[i][j][0] * alpha);
        atomicAdd(Cf + (long long)r * N + c + 1, acc[i][j][1] * alpha);
        atomicAdd(Cf + (long long)(r + 8) * N + c, acc[i][j][2] * alpha);
        atomicAdd(Cf + (long long)(r + 8) * N + c + 1, acc[i][j][3] * alpha);
      }
    }
}
