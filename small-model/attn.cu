// ================= fused attention block forward =================
// X1 = X + Wo( attn( Wqkv LN1(X) ) ).  Tile = 128 tokens = 4 sequences of 32, 8 warps.
// Phase 1: warp w -> sequence s = w/2, heads {2(w%2), 2(w%2)+1}: LN1 + FP4 QKV (only that head's
//          96 columns), causal softmax attention on bf16 tensor cores, O -> smem (bf16).
// Phase 2: warp w -> rows 16w..16w+15: quantize O rows, FP4 proj GEMM, residual add.
#define SEQ 32
#define NH 4
#define HD 32
#define ATT_SCALE 0.17677669529663687f  // 1/sqrt(32)

// smem layout (bytes)
#define AF_WQKV 0          // 384x128 frag: 24576 data + 768 scales
#define AF_WQKVS 24576
#define AF_WO 25344        // 128x128 frag: 8192 + 256
#define AF_WOS 33536
#define AF_O 33792         // 128 x 128 bf16, row pitch 136 elems (pad vs bank conflicts)
#define O_PITCH 136
#define ATTF_SMEM (33792 + 128 * O_PITCH * 2)

// Wqkv rows are stored reordered: chunk 3p+i (i<2) = [q_{2p+i} | k_{2p+i}], chunk 3p+2 = [v_2p | v_2p+1]
__device__ __forceinline__ int qkv_ntile(int which, int head, int n) {
  int p = head >> 1, i = head & 1;
  return which < 2 ? (3 * p + i) * 8 + which * 4 + n : (3 * p + 2) * 8 + i * 4 + n;
}

__device__ __forceinline__ void mma_bf16_acc(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// QKV for one head of one sequence (32 rows = 2 m-tiles). hq[mt][ks]. Output q,k,v in C-layout,
// already scaled to real units: q[mt][nt][4] (nt over 4 ntiles = 32 dims) etc.
__device__ __forceinline__ void qkv_head(const AFrag (*hq)[2], const float (*gsr)[2], const uint2* W, const u32* S,
                                         float gw, int head, int lane, int g,
                                         float (*q)[4][4], float (*k)[4][4], float (*v)[4][4]) {
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
          mma4(dst[mt][n], hq[mt][ks].a0, hq[mt][ks].a1, hq[mt][ks].a2, hq[mt][ks].a3, W[wi * 32 + lane], hq[mt][ks].sf, S[wi * 2 + (g >> 2)]);
        }
        float s0 = gsr[mt][0] * gw, s1 = gsr[mt][1] * gw;
        dst[mt][n][0] *= s0; dst[mt][n][1] *= s0; dst[mt][n][2] *= s1; dst[mt][n][3] *= s1;
      }
  }
}

// causal softmax attention for one (seq, head): q,k,v C-layout [mt][nt(dim)][4] -> o C-layout, and
// the probabilities p[mt][key ntile][4] (normalized) + row stats if wanted.
__device__ __forceinline__ void attn_head(float (*q)[4][4], float (*k)[4][4], float (*v)[4][4], float (*o)[4][4],
                                          float (*p)[4][4], int g, int t) {
  // S = Q K^T : A = Q (rows=query tok, k=dim), B = K^T (k=dim, n=key tok) -> K C-layout is the B frag
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int kt = 0; kt < 4; kt++) {  // key ntile (8 keys)
      p[mt][kt][0] = p[mt][kt][1] = p[mt][kt][2] = p[mt][kt][3] = 0.f;
      int kmt = kt >> 1, khalf = (kt & 1) * 2;  // key rows kt*8.. live in k[kmt], regs [khalf..khalf+1]
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {  // dim 16-chunks
        u32 a0 = packbf(q[mt][2 * ks][0], q[mt][2 * ks][1]), a1 = packbf(q[mt][2 * ks][2], q[mt][2 * ks][3]);
        u32 a2 = packbf(q[mt][2 * ks + 1][0], q[mt][2 * ks + 1][1]), a3 = packbf(q[mt][2 * ks + 1][2], q[mt][2 * ks + 1][3]);
        u32 b0 = packbf(k[kmt][2 * ks][khalf], k[kmt][2 * ks][khalf + 1]);
        u32 b1 = packbf(k[kmt][2 * ks + 1][khalf], k[kmt][2 * ks + 1][khalf + 1]);
        mma_bf16_acc(p[mt][kt], a0, a1, a2, a3, b0, b1);
      }
    }
  // causal mask + softmax (row = query; its 32 keys are spread over the quad: kt*8 + 2t + e)
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int qi = mt * 16 + g + 8 * h;
      float m = -1e30f;
#pragma unroll
      for (int kt = 0; kt < 4; kt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          int kj = kt * 8 + 2 * t + e;
          float s = kj <= qi ? p[mt][kt][2 * h + e] * ATT_SCALE : -1e30f;
          p[mt][kt][2 * h + e] = s; m = fmaxf(m, s);
        }
      m = qmax(m);
      float sum = 0.f;
#pragma unroll
      for (int kt = 0; kt < 4; kt++)
#pragma unroll
        for (int e = 0; e < 2; e++) { float ex = __expf(p[mt][kt][2 * h + e] - m); p[mt][kt][2 * h + e] = ex; sum += ex; }
      float inv = 1.f / qsum(sum);
#pragma unroll
      for (int kt = 0; kt < 4; kt++) { p[mt][kt][2 * h] *= inv; p[mt][kt][2 * h + 1] *= inv; }
    }
  // O = P V : A = P (rows=query, k=key), B = V (k=key, n=dim): V C-layout is (key g, dim 2t) -> transpose
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int n = 0; n < 4; n++) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < 2; ks++) {  // key 16-chunk == v m-tile ks
#pragma unroll
    for (int n = 0; n < 4; n++) {   // dim ntile
      // B frag: b0 = {V[key 2t][dim g], V[2t+1][g]} keys 0..7 of chunk; b1 keys 8..15
      u32 b0 = movtrans(packbf(v[ks][n][0], v[ks][n][1]));
      u32 b1 = movtrans(packbf(v[ks][n][2], v[ks][n][3]));
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        u32 a0 = packbf(p[mt][2 * ks][0], p[mt][2 * ks][1]), a1 = packbf(p[mt][2 * ks][2], p[mt][2 * ks][3]);
        u32 a2 = packbf(p[mt][2 * ks + 1][0], p[mt][2 * ks + 1][1]), a3 = packbf(p[mt][2 * ks + 1][2], p[mt][2 * ks + 1][3]);
        mma_bf16_acc(o[mt][n], a0, a1, a2, a3, b0, b1);
      }
    }
  }
}

// LN + quant for the 32 rows of sequence starting at row base (2 m-tiles)
__device__ __forceinline__ void ln_quant_seq(const __nv_bfloat16* X, int base, int g, int t, const float* lnw,
                                             AFrag (*hq)[2], float (*gsr)[2], float (*mean)[2], float (*rstd)[2]) {
  float hn[2];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) ln_quant(X, base + mt * 16 + g, base + mt * 16 + g + 8, t, lnw, hq[mt], gsr[mt], mean[mt], rstd[mt], hn);
}

extern "C" __global__ void __launch_bounds__(256, 1) attn_fwd(
    const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ X1, const float* __restrict__ lnw,
    const uint2* __restrict__ wqd, const u32* __restrict__ wqs, const float* __restrict__ wqamax,
    const uint2* __restrict__ wod, const u32* __restrict__ wos, const float* __restrict__ woamax, int M) {
  extern __shared__ __align__(16) unsigned char sm[];
  smem_copy(sm + AF_WQKV, wqd, 24576); smem_copy(sm + AF_WQKVS, wqs, 768);
  smem_copy(sm + AF_WO, wod, 8192); smem_copy(sm + AF_WOS, wos, 256);
  __syncthreads();
  const uint2* WQ = (const uint2*)(sm + AF_WQKV); const u32* SQ = (const u32*)(sm + AF_WQKVS);
  const uint2* WO = (const uint2*)(sm + AF_WO); const u32* SO = (const u32*)(sm + AF_WOS);
  __nv_bfloat16* Os = (__nv_bfloat16*)(sm + AF_O);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  float gq = *wqamax * (1.f / FP4_E4M3), go = *woamax * (1.f / FP4_E4M3);

  for (int tile = blockIdx.x; tile < M / TROWS; tile += gridDim.x) {
    int t0 = tile * TROWS;
    {  // ---- phase 1: attention per (seq, head pair)
      int s = warp >> 1, base = t0 + s * SEQ;
      AFrag hq[2][2]; float gsr[2][2], mean[2][2], rstd[2][2];
      ln_quant_seq(X, base, g, t, lnw, hq, gsr, mean, rstd);
#pragma unroll 1
      for (int hh = 0; hh < 2; hh++) {
        int head = (warp & 1) * 2 + hh;
        float q[2][4][4], k[2][4][4], v[2][4][4], o[2][4][4], p[2][4][4];
        qkv_head(hq, gsr, WQ, SQ, gq, head, lane, g, q, k, v);
        attn_head(q, k, v, o, p, g, t);
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
          for (int n = 0; n < 4; n++) {
            int row = s * SEQ + mt * 16 + g, col = head * HD + n * 8 + 2 * t;
            *(__nv_bfloat162*)(Os + row * O_PITCH + col) = __floats2bfloat162_rn(o[mt][n][0], o[mt][n][1]);
            *(__nv_bfloat162*)(Os + (row + 8) * O_PITCH + col) = __floats2bfloat162_rn(o[mt][n][2], o[mt][n][3]);
          }
      }
    }
    __syncthreads();
    {  // ---- phase 2: proj + residual for rows 16w..16w+15
      int lr0 = warp * 16 + g, lr1 = lr0 + 8;
      float v0[2][16], v1[2][16], am0 = 0.f, am1 = 0.f;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        ld8bf(Os + lr0 * O_PITCH + ks * 64 + t * 8, v0[ks]); ld8bf(Os + lr0 * O_PITCH + ks * 64 + 32 + t * 8, v0[ks] + 8);
        ld8bf(Os + lr1 * O_PITCH + ks * 64 + t * 8, v1[ks]); ld8bf(Os + lr1 * O_PITCH + ks * 64 + 32 + t * 8, v1[ks] + 8);
#pragma unroll
        for (int i = 0; i < 16; i++) { am0 = fmaxf(am0, fabsf(v0[ks][i])); am1 = fmaxf(am1, fabsf(v1[ks][i])); }
      }
      am0 = qmax(am0); am1 = qmax(am1);
      u32 dummy = 0;
      AFrag oq[2];
#pragma unroll
      for (int ks = 0; ks < 2; ks++)
        oq[ks] = quant_afrag(v0[ks], v1[ks], am0 > 0.f ? FP4_E4M3 / am0 : 0.f, am1 > 0.f ? FP4_E4M3 / am1 : 0.f, t, false, dummy);
      float f0 = am0 * (1.f / FP4_E4M3) * go, f1 = am1 * (1.f / FP4_E4M3) * go;
      int r0 = t0 + lr0, r1 = t0 + lr1;
#pragma unroll
      for (int n = 0; n < 16; n++) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          int wi = n * 2 + ks;
          mma4(acc, oq[ks].a0, oq[ks].a1, oq[ks].a2, oq[ks].a3, WO[wi * 32 + lane], oq[ks].sf, SO[wi * 2 + (g >> 2)]);
        }
        int col = n * 8 + 2 * t;
        float2 x0 = __bfloat1622float2(*(const __nv_bfloat162*)(X + (long long)r0 * DM + col));
        float2 x1 = __bfloat1622float2(*(const __nv_bfloat162*)(X + (long long)r1 * DM + col));
        *(__nv_bfloat162*)(X1 + (long long)r0 * DM + col) = __floats2bfloat162_rn(x0.x + acc[0] * f0, x0.y + acc[1] * f0);
        *(__nv_bfloat162*)(X1 + (long long)r1 * DM + col) = __floats2bfloat162_rn(x1.x + acc[2] * f1, x1.y + acc[3] * f1);
      }
    }
    __syncthreads();
  }
}
