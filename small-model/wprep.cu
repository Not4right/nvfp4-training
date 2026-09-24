// ================= batched weight re-quantization (all layers, 3 launches) =================
struct WDesc {
  const float* W; const int* perm;           // master weight [N,K] fp32; optional source-row permutation
  unsigned char* nb; unsigned char* bs;      // nibble scratch [N,K], 2D block scales [N/16,K/16]
  uint2* fd; u32* fs; uint2* td; u32* ts;    // frag (N rows,K contraction) and transposed frag
  unsigned* stats;                           // [0]=amax bits [1]=max row sumsq bits [2..2+K) = col sumsq (float)
  float* amax; float* rn; float* cn;         // outputs used by the fused kernels
  int N, K, ffN, ffK;
};

__device__ __forceinline__ float wsrc(const WDesc& d, int n, int k) {
  int r = d.perm ? d.perm[n] : n;
  return d.W[(long long)r * d.K + k];
}

// stats: amax, max row norm^2, column norm^2 (block = 8 warps over a slice of rows)
extern "C" __global__ void wp_stats(const WDesc* __restrict__ descs) {
  const WDesc d = descs[blockIdx.y];
  __shared__ float colsq[512];
  for (int i = threadIdx.x; i < d.K; i += blockDim.x) colsq[i] = 0.f;
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int rows_per = (d.N + gridDim.x - 1) / gridDim.x, r0 = blockIdx.x * rows_per, r1 = min(d.N, r0 + rows_per);
  float am = 0.f, rmax = 0.f;
  for (int n = r0 + warp; n < r1; n += 8) {
    float ss = 0.f;
    for (int k = lane; k < d.K; k += 32) {
      float v = wsrc(d, n, k);
      am = fmaxf(am, fabsf(v)); ss += v * v;
      atomicAdd(&colsq[k], v * v);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    rmax = fmaxf(rmax, ss);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  if (lane == 0) { atomicMax(d.stats, __float_as_uint(am)); atomicMax(d.stats + 1, __float_as_uint(rmax)); }
  __syncthreads();
  for (int i = threadIdx.x; i < d.K; i += blockDim.x) atomicAdd((float*)d.stats + 2 + i, colsq[i]);
}

// nibbles + block scales (warp per 16x16 block); block (0,w) also publishes amax / rn / cn
extern "C" __global__ void wp_nib(const WDesc* __restrict__ descs) {
  const WDesc d = descs[blockIdx.y];
  float am = __uint_as_float(d.stats[0]);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *d.amax = am;
    *d.rn = sqrtf(__uint_as_float(d.stats[1])) * 1.25f;
    float c = 0.f;
    for (int k = 0; k < d.K; k++) c = fmaxf(c, ((float*)d.stats)[2 + k]);
    *d.cn = sqrtf(c) * 1.25f;
  }
  float inv_gs = am > 0.f ? FP4_E4M3 / am : 0.f;
  int nb = d.N / 16, kb = d.K / 16, lane = threadIdx.x & 31;
  for (int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; b < nb * kb; b += (gridDim.x * blockDim.x) >> 5) {
    int bn = b / kb, bk = b % kb;
    int n = grp_member(bn, lane >> 1, d.ffN);
    float x[8], m = 0.f; int kk[8];
#pragma unroll
    for (int j = 0; j < 8; j++) { kk[j] = grp_member(bk, (lane & 1) * 8 + j, d.ffK); x[j] = wsrc(d, n, kk[j]); m = fmaxf(m, fabsf(x[j])); }
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    u32 sb = e4m3_enc(m * inv_gs * (1.f / 6.f));
    float sdec = e4m3_dec(sb), k = sdec > 0.f ? inv_gs / sdec : 0.f;
    if (lane == 0) d.bs[b] = (unsigned char)sb;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      u32 byte = fp4x2(x[j] * k, x[j + 1] * k);
      d.nb[(long long)n * d.K + kk[j]] = byte & 15;
      d.nb[(long long)n * d.K + kk[j + 1]] = byte >> 4;
    }
  }
}

// both frag-native forms (thread per (ntile, kstep, lane)); y-dim = weight, blockIdx.z = transposed?
extern "C" __global__ void wp_frag(const WDesc* __restrict__ descs) {
  const WDesc d = descs[blockIdx.y];
  int trans = blockIdx.z;
  int N = d.N, K = d.K;
  int Nf = trans ? K : N, Kf = trans ? N : K, KS = Kf / 64;
  int ffN = trans ? d.ffK : d.ffN, ffK = trans ? d.ffN : d.ffK;
  uint2* bd = trans ? d.td : d.fd; u32* bsc = trans ? d.ts : d.fs;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < (Nf / 8) * KS * 32; id += gridDim.x * blockDim.x) {
    int lane = id & 31, ks = (id >> 5) % KS, nt = (id >> 5) / KS;
    int g = lane >> 2, t = lane & 3, n = nt * 8 + g;
    u32 w[2] = {0, 0};
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int p = h * 32 + t * 8 + i;
        int kl = ks * 64 + (ffK ? pi_inv(p) : p);
        unsigned char nib = trans ? d.nb[(long long)kl * K + n] : d.nb[(long long)n * K + kl];
        w[h] |= (u32)nib << (4 * i);
      }
    bd[id] = make_uint2(w[0], w[1]);
    if (t == 0 && (g & 3) == 0) {
      int gn = grp_of(n, ffN);
      u32 s = 0;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        int gk = ks * 4 + q;
        int gr = trans ? gk : gn, gc = trans ? gn : gk;
        s |= (u32)d.bs[gr * (K / 16) + gc] << (8 * q);
      }
      bsc[(nt * KS + ks) * 2 + (g >> 2)] = s;
    }
  }
}
