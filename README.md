# NVFP4 training on a consumer GPU

Training transformers in 4-bit floating point (NVFP4) on an RTX 5060 (8 GB, Blackwell sm_120),
written from scratch: custom CUDA kernels, hand-written forward and backward passes, no training
framework.

| folder | what it is |
|---|---|
| [`small-model/`](small-model/) | A ~1M-parameter GPT where every transformer block is one fused persistent kernel. Controlled NVFP4 vs BF16 comparison, 3 seeds each. |
| [`llm-1b/`](llm-1b/) | A 1.0B-parameter LLM trained entirely inside 8 GB of VRAM with NVFP4 GEMMs, an FP8 attention kernel and the Muon optimizer. 10B-token run in progress. |

## Results so far

**Small model (arithmetic task, 3 seeds, same data and step count):**

| run | wall time | exact-match accuracy | eval loss |
|---|---:|---:|---:|
| BF16 (torch.compile + CUDA graph) | 164 s | 99.72% ± 0.1 | 0.0014 |
| NVFP4 fused | **57 s (2.9x faster)** | 98.05% ± 0.5 | 0.0108 |
| NVFP4 fused, last 15% of steps in BF16 | 71 s (2.3x faster) | 99.37% ± 0.1 (BF16 inference) | 0.0040 |

4-bit training is much faster here but costs some accuracy; a short BF16 finish recovers most of it.
The speedup comes from fusion that only fits because the weights are 4-bit (they stay resident in
shared memory), so it is specific to small models and does not carry over to 1B as-is.

**1B model (RTX 5060, 8 GB):**

- 1.0025B parameters (20 layers, d=2048, 16 query / 4 KV heads, SwiGLU ff=5632, vocab 49,152)
- Full training state fits in 8 GB: ~6.6 GB peak at batch 2 x 2048 tokens
- 9,300 to 9,700 tokens/s in the NVFP4 phase (from 6.0k with the first version)
- FP8 attention kernel: forward 48.5 TFLOPS vs 27.9 for cuDNN on the same GPU (1.74x); backward 1.07x
- Loss 3.54 after 0.27B tokens of a planned 10B

## What is not done yet

- A matched BF16 run of the 1B model (it does not fit in 8 GB in BF16 with this optimizer),
  so the quality cost of NVFP4 at 1B scale is not measured yet.
- Downstream evaluations of the 1B model.

## The NVFP4 recipe

Following NVIDIA's "Pretraining Large Language Models with NVFP4" (2025): E2M1 values with one
E4M3 scale per 16 elements plus an FP32 tensor or row scale; 2D 16x16 weight blocks so W and Wᵀ
quantize identically; stochastic rounding on gradients (done in software, since sm_120 has no
`cvt.rs`); 16-point random Hadamard transforms on the weight-gradient operands; an optional final
phase in BF16. The FP4 tensor cores on this GPU measured 266 TFLOPS against 32 for BF16.

## License

MIT, see [LICENSE](LICENSE).
