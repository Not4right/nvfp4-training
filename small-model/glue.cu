// ================= glue kernels: embedding, fused LM tail, grad-norm, AdamW =================

// x0[tok] = tok_emb[idx[tok]] + pos[tok % 32]  -> bf16.  thread per 8 dims.
extern "C" __global__ void emb_fwd(const long long* __restrict__ idx, const float* __restrict__ tokw,
                                   const float* __restrict__ pos, __nv_bfloat16* __restrict__ X, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * 16) return;
  int tok = i >> 4, d0 = (i & 15) * 8;
  const float* a = tokw + idx[tok] * DM + d0; const float* b = pos + (tok & 31) * DM + d0;
  uint4 o; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int j = 0; j < 4; j++) h[j] = __floats2bfloat162_rn(a[2 * j] + b[2 * j], a[2 * j + 1] + b[2 * j + 1]);
  *reinterpret_cast<uint4*>(X + (long long)tok * DM + d0) = o;
}

// d tok_emb / d pos from dX0.  block: 128 threads (thread = dim), SMEM accumulators, atomics at end.
extern "C" __global__ void emb_bwd(const long long* __restrict__ idx, const __nv_bfloat16* __restrict__ DX,
                                   float* __restrict__ dtok, float* __restrict__ dpos, int M, int V) {
  __shared__ float at[16][DM], ap[32][DM];
  int d = threadIdx.x;
  for (int v = 0; v < V; v++) at[v][d] = 0.f;
  for (int p = 0; p < 32; p++) ap[p][d] = 0.f;
  int per = (M + gridDim.x - 1) / gridDim.x, t0 = blockIdx.x * per, t1 = min(M, t0 + per);
  for (int tok = t0; tok < t1; tok++) {
    float v = __bfloat162float(DX[(long long)tok * DM + d]);
    at[idx[tok]][d] += v; ap[tok & 31][d] += v;
  }
  for (int v = 0; v < V; v++) atomicAdd(dtok + v * DM + d, at[v][d]);
  for (int p = 0; p < 32; p++) atomicAdd(dpos + p * DM + d, ap[p][d]);
}

// Fused LM tail: y = LN(x)*g ; logits = y W^T (16 classes) ; masked CE (normalised by wsum);
// backward to dX (bf16), dW (16x128), dg (128), loss (scalar).  64 tokens per iteration, 256 threads.
#define TAILC 32
extern "C" __global__ void __launch_bounds__(256) lm_tail(
    const __nv_bfloat16* __restrict__ X, const float* __restrict__ g, const float* __restrict__ W,
    const long long* __restrict__ Y, const float* __restrict__ Wt, const float* __restrict__ wsum,
    __nv_bfloat16* __restrict__ DX, float* __restrict__ dW, float* __restrict__ dg, float* __restrict__ loss, int M) {
  __shared__ float xh[TAILC][DM + 1], dys[TAILC][DM + 1], Ws[16][DM], gs[DM], dl[TAILC][17], rstd_s[TAILC];
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 16 * DM; i += 256) Ws[i / DM][i % DM] = W[i];
  for (int i = tid; i < DM; i += 256) gs[i] = g[i];
  float inv_ws = 1.f / *wsum;
  float accW[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, accg = 0.f, accl = 0.f;
  int wj = tid >> 4, wd0 = (tid & 15) * 8;  // dW ownership
  __syncthreads();
  for (int c0 = blockIdx.x * TAILC; c0 < M; c0 += gridDim.x * TAILC) {
    // LN: warp per 8 tokens, lane = 4 dims
    for (int tt = warp; tt < TAILC; tt += 8) {
      int tok = c0 + tt;
      float v[4], s = 0.f, s2 = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) { v[j] = __bfloat162float(X[(long long)tok * DM + lane * 4 + j]); s += v[j]; s2 += v[j] * v[j]; }
#pragma unroll
      for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
      float mean = s * (1.f / DM), rs = rsqrtf(fmaxf(s2 * (1.f / DM) - mean * mean, 0.f) + 1e-5f);
#pragma unroll
      for (int j = 0; j < 4; j++) xh[tt][lane * 4 + j] = (v[j] - mean) * rs;
      if (lane == 0) rstd_s[tt] = rs;
    }
    __syncthreads();
    {  // logits + softmax CE: 8 threads per token, 2 classes each
      int tt = tid >> 3, q = tid & 7, tok = c0 + tt;
      float lg[2] = {0.f, 0.f};
      for (int d = 0; d < DM; d++) {
        float yv = xh[tt][d] * gs[d];
        lg[0] += yv * Ws[q * 2][d]; lg[1] += yv * Ws[q * 2 + 1][d];
      }
      float m = fmaxf(lg[0], lg[1]);
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      float e[2] = {__expf(lg[0] - m), __expf(lg[1] - m)}, se = e[0] + e[1];
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) se += __shfl_xor_sync(0xffffffffu, se, o);
      int y = (int)Y[tok]; float wt = Wt[tok] * inv_ws;
      float ly = 0.f;
#pragma unroll
      for (int j = 0; j < 2; j++) {
        int cls = q * 2 + j;
        if (cls == y) ly = lg[j];
        dl[tt][cls] = (e[j] / se - (cls == y ? 1.f : 0.f)) * wt;
      }
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) ly += __shfl_xor_sync(0xffffffffu, ly, o);
      if (q == 0) accl += (m + __logf(se) - ly) * wt;
    }
    __syncthreads();
    {  // dy = dl W ; LN backward.  8 threads per token, 16 dims each
      int tt = tid >> 3, q = tid & 7, tok = c0 + tt;
      float dyv[16], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        int d = q * 16 + i;
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) a += dl[tt][j] * Ws[j][d];
        dys[tt][d] = a;
        float dxh = a * gs[d];
        dyv[i] = dxh; s1 += dxh; s2 += dxh * xh[tt][d];
      }
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) { s1 += __shfl_xor_sync(0xffffffffu, s1, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
      s1 *= (1.f / DM); s2 *= (1.f / DM);
      float rs = rstd_s[tt];
#pragma unroll
      for (int i = 0; i < 16; i += 2) {
        int d = q * 16 + i;
        float o0 = rs * (dyv[i] - s1 - xh[tt][d] * s2), o1 = rs * (dyv[i + 1] - s1 - xh[tt][d + 1] * s2);
        *(__nv_bfloat162*)(DX + (long long)tok * DM + d) = __floats2bfloat162_rn(o0, o1);
      }
    }
    __syncthreads();
    // weight grads: dW[j][d] += sum_t dl[t][j] * xh[t][d]*g[d] ; dg[d] += sum_t dys[t][d]*xh[t][d]
    for (int tt = 0; tt < TAILC; tt++) {
      float a = dl[tt][wj];
#pragma unroll
      for (int i = 0; i < 8; i++) accW[i] += a * xh[tt][wd0 + i];
    }
    {
      int d = tid & 127, h = tid >> 7;
      for (int tt = h * (TAILC / 2); tt < (h + 1) * (TAILC / 2); tt++) accg += dys[tt][d] * xh[tt][d];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; i++) atomicAdd(dW + wj * DM + wd0 + i, accW[i] * gs[wd0 + i]);
  atomicAdd(dg + (tid & 127), accg);
#pragma unroll
  for (int o = 16; o; o >>= 1) accl += __shfl_xor_sync(0xffffffffu, accl, o);
  if (lane == 0) atomicAdd(loss, accl);
}

// sum of squares of the flat gradient (for clipping)
extern "C" __global__ void grad_sq(const float* __restrict__ G, float* __restrict__ out, int n) {
  float s = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += G[i] * G[i];
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// AdamW (torch semantics, decoupled wd) with global-norm clipping; hp = {lr, step}
extern "C" __global__ void adamw_flat(float* __restrict__ P, const float* __restrict__ G, float* __restrict__ Mo,
                                      float* __restrict__ Vo, const unsigned char* __restrict__ wdmask,
                                      const float* __restrict__ hp, const float* __restrict__ gsq, int n,
                                      float b1, float b2, float eps, float wd, float maxnorm) {
  float lr = hp[0], step = hp[1];
  float norm = sqrtf(*gsq), clip = fminf(1.f, maxnorm / (norm + 1e-6f));
  float bc1 = 1.f - __powf(b1, step), bc2 = 1.f - __powf(b2, step);
  float step_size = lr / bc1, bc2s = rsqrtf(bc2);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float g = G[i] * clip, p = P[i];
    if (wdmask[i]) p *= 1.f - lr * wd;
    float m = b1 * Mo[i] + (1.f - b1) * g, v = b2 * Vo[i] + (1.f - b2) * g * g;
    Mo[i] = m; Vo[i] = v;
    P[i] = p - step_size * m / (sqrtf(v) * bc2s + eps);
  }
}

// ---- tensor-core LM tail: warp per 16 tokens.  logits = Y W^T, dY = dL W, dW += dL^T Y (bf16 mma)
extern "C" __global__ void __launch_bounds__(256) lm_tail_tc(
    const __nv_bfloat16* __restrict__ X, const float* __restrict__ g, const float* __restrict__ W,
    const long long* __restrict__ Y, const float* __restrict__ Wt, const float* __restrict__ wsum,
    __nv_bfloat16* __restrict__ DX, float* __restrict__ dW, float* __restrict__ dg, float* __restrict__ loss, int M) {
  int lane = threadIdx.x & 31, g8 = lane >> 2, t = lane & 3;
  int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nwarps = (gridDim.x * blockDim.x) >> 5;
  float inv_ws = 1.f / *wsum;
  // B frags for logits: b0 = {W[cls g][d 2t], W[cls g][d 2t+1]}, b1 = d+8 ; per k-step (16 dims) and class ntile
  u32 bl[8][2][2];
#pragma unroll
  for (int ks = 0; ks < 8; ks++)
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      const float* w = W + (nt * 8 + g8) * DM + ks * 16 + 2 * t;
      bl[ks][nt][0] = packbf(w[0], w[1]); bl[ks][nt][1] = packbf(w[8], w[9]);
    }
  float dwacc[16][4], dgacc[16][2];  // dW: rows = class (m16), cols = d (16 ntiles); dg partial per col
#pragma unroll
  for (int n = 0; n < 16; n++) { dwacc[n][0] = dwacc[n][1] = dwacc[n][2] = dwacc[n][3] = 0.f; dgacc[n][0] = dgacc[n][1] = 0.f; }
  float gl[16][2];
#pragma unroll
  for (int n = 0; n < 16; n++) { gl[n][0] = g[n * 8 + 2 * t]; gl[n][1] = g[n * 8 + 2 * t + 1]; }
  float lacc = 0.f;
  for (int tile = warp_g; tile < M / 16; tile += nwarps) {
    int r0 = tile * 16 + g8, r1 = r0 + 8;
    // load rows in C-layout (row g / g+8, cols n*8+2t..) and LayerNorm
    float xh[16][4], s[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f};
#pragma unroll
    for (int n = 0; n < 16; n++) {
      float2 a = __bfloat1622float2(*(const __nv_bfloat162*)(X + (long long)r0 * DM + n * 8 + 2 * t));
      float2 b = __bfloat1622float2(*(const __nv_bfloat162*)(X + (long long)r1 * DM + n * 8 + 2 * t));
      xh[n][0] = a.x; xh[n][1] = a.y; xh[n][2] = b.x; xh[n][3] = b.y;
      s[0] += a.x + a.y; s[1] += b.x + b.y; s2[0] += a.x * a.x + a.y * a.y; s2[1] += b.x * b.x + b.y * b.y;
    }
    float rstd[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      float mean = qsum(s[h]) * (1.f / DM), var = qsum(s2[h]) * (1.f / DM) - mean * mean;
      rstd[h] = rsqrtf(fmaxf(var, 0.f) + 1e-5f);
#pragma unroll
      for (int n = 0; n < 16; n++) { xh[n][2 * h] = (xh[n][2 * h] - mean) * rstd[h]; xh[n][2 * h + 1] = (xh[n][2 * h + 1] - mean) * rstd[h]; }
    }
    // logits (16 tok x 16 cls): A = y = xh*g in C-layout == A frag (k-step ks = ntiles 2ks, 2ks+1)
    float lg[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      int n0 = 2 * ks, n1 = n0 + 1;
      u32 a0 = packbf(xh[n0][0] * gl[n0][0], xh[n0][1] * gl[n0][1]), a1 = packbf(xh[n0][2] * gl[n0][0], xh[n0][3] * gl[n0][1]);
      u32 a2 = packbf(xh[n1][0] * gl[n1][0], xh[n1][1] * gl[n1][1]), a3 = packbf(xh[n1][2] * gl[n1][0], xh[n1][3] * gl[n1][1]);
      mma_bf16_acc(lg[0], a0, a1, a2, a3, bl[ks][0][0], bl[ks][0][1]);
      mma_bf16_acc(lg[1], a0, a1, a2, a3, bl[ks][1][0], bl[ks][1][1]);
    }
    // softmax CE per row (row g: lg[nt][0..1], row g+8: lg[nt][2..3]; classes nt*8+2t+e)
    float dlv[2][4];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int tok = h ? r1 : r0, y = (int)Y[tok];
      float wt = Wt[tok] * inv_ws;
      float m = fmaxf(fmaxf(lg[0][2 * h], lg[0][2 * h + 1]), fmaxf(lg[1][2 * h], lg[1][2 * h + 1]));
      m = qmax(m);
      float e[4] = {__expf(lg[0][2 * h] - m), __expf(lg[0][2 * h + 1] - m), __expf(lg[1][2 * h] - m), __expf(lg[1][2 * h + 1] - m)};
      float se = qsum(e[0] + e[1] + e[2] + e[3]), ise = 1.f / se, ly = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int cls = (i >> 1) * 8 + 2 * t + (i & 1);
        float lv = lg[i >> 1][2 * h + (i & 1)];
        if (cls == y) ly = lv;
        dlv[i >> 1][2 * h + (i & 1)] = (e[i] * ise - (cls == y ? 1.f : 0.f)) * wt;
      }
      ly = qsum(ly);
      if (t == 0) lacc += (m + __logf(se) - ly) * wt;
    }
    // dY = dL W : A = dL (16 tok x 16 cls; C-layout == A frag), B = W [k=cls, n=d]: b0={W[2t][d g],W[2t+1][d g]}
    u32 da0 = packbf(dlv[0][0], dlv[0][1]), da1 = packbf(dlv[0][2], dlv[0][3]);
    u32 da2 = packbf(dlv[1][0], dlv[1][1]), da3 = packbf(dlv[1][2], dlv[1][3]);
    float dyv[16][4], s1[2] = {0.f, 0.f}, sx[2] = {0.f, 0.f};
#pragma unroll
    for (int n = 0; n < 16; n++) {
      int d = n * 8 + g8;
      u32 b0 = packbf(W[(2 * t) * DM + d], W[(2 * t + 1) * DM + d]);
      u32 b1 = packbf(W[(2 * t + 8) * DM + d], W[(2 * t + 9) * DM + d]);
      dyv[n][0] = dyv[n][1] = dyv[n][2] = dyv[n][3] = 0.f;
      mma_bf16_acc(dyv[n], da0, da1, da2, da3, b0, b1);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        dgacc[n][e & 1] += dyv[n][e] * xh[n][e];
        float dxh = dyv[n][e] * gl[n][e & 1];
        dyv[n][e] = dxh; s1[e >> 1] += dxh; sx[e >> 1] += dxh * xh[n][e];
      }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) { s1[h] = qsum(s1[h]) * (1.f / DM); sx[h] = qsum(sx[h]) * (1.f / DM); }
#pragma unroll
    for (int n = 0; n < 16; n++) {
      int col = n * 8 + 2 * t;
      *(__nv_bfloat162*)(DX + (long long)r0 * DM + col) = __floats2bfloat162_rn(rstd[0] * (dyv[n][0] - s1[0] - xh[n][0] * sx[0]), rstd[0] * (dyv[n][1] - s1[0] - xh[n][1] * sx[0]));
      *(__nv_bfloat162*)(DX + (long long)r1 * DM + col) = __floats2bfloat162_rn(rstd[1] * (dyv[n][2] - s1[1] - xh[n][2] * sx[1]), rstd[1] * (dyv[n][3] - s1[1] - xh[n][3] * sx[1]));
    }
    // dW += dL^T Y : A = dL^T (cls x tok) via movmatrix, B = Y [k=tok, n=d] via movmatrix of y C-layout
    u32 ta0 = movtrans(da0), ta1 = movtrans(da2), ta2 = movtrans(da1), ta3 = movtrans(da3);
#pragma unroll
    for (int n = 0; n < 16; n++) {
      u32 b0 = movtrans(packbf(xh[n][0] * gl[n][0], xh[n][1] * gl[n][1]));
      u32 b1 = movtrans(packbf(xh[n][2] * gl[n][0], xh[n][3] * gl[n][1]));
      mma_bf16_acc(dwacc[n], ta0, ta1, ta2, ta3, b0, b1);
    }
  }
  // reductions: dW (class rows g / g+8, cols n*8+2t+e), dg (reduce over rows = lanes with same t)
#pragma unroll
  for (int n = 0; n < 16; n++) {
    int col = n * 8 + 2 * t;
    atomicAdd(dW + g8 * DM + col, dwacc[n][0]); atomicAdd(dW + g8 * DM + col + 1, dwacc[n][1]);
    atomicAdd(dW + (g8 + 8) * DM + col, dwacc[n][2]); atomicAdd(dW + (g8 + 8) * DM + col + 1, dwacc[n][3]);
#pragma unroll
    for (int e = 0; e < 2; e++) {
      float v = dgacc[n][e];
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (g8 == 0) atomicAdd(dg + col + e, v);
    }
  }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) lacc += __shfl_xor_sync(0xffffffffu, lacc, o);
  if (lane == 0) atomicAdd(loss, lacc);
}

// d tok_emb / d pos, parallel version: block = 256 threads = 2 token-streams x 128 dims; each block
// reduces 256 tokens into registers per (pos) and smem per (vocab), then one atomic per entry.
// Tokens are laid out [B, 32]: block b handles rows 8b..8b+7 -> each thread sums 4 rows for its
// (position, dim) pairs; pos grads need no atomics inside a block.
extern "C" __global__ void emb_bwd2(const long long* __restrict__ idx, const __nv_bfloat16* __restrict__ DX,
                                    float* __restrict__ dtok, float* __restrict__ dpos, int B) {
  __shared__ float at[16][DM];
  int tid = threadIdx.x, d = tid & 127, half = tid >> 7;
  for (int i = tid; i < 16 * DM; i += 256) (&at[0][0])[i] = 0.f;
  __syncthreads();
  int rows_per = 16;  // rows (sequences) per block
  int r0 = blockIdx.x * rows_per;
  for (int p = half; p < 32; p += 2) {
    float s = 0.f;
    for (int r = r0; r < min(B, r0 + rows_per); r++) {
      int tok = r * 32 + p;
      float v = __bfloat162float(DX[(long long)tok * DM + d]);
      s += v;
      atomicAdd(&at[idx[tok]][d], v);
    }
    atomicAdd(dpos + p * DM + d, s);
  }
  __syncthreads();
  for (int i = tid; i < 16 * DM; i += 256) {
    float v = (&at[0][0])[i];
    if (v != 0.f) atomicAdd(dtok + i, v);
  }
}
