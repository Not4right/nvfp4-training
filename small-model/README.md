# NVFP4 training on an RTX 5060 (sm_120a), from scratch

A ~1M-parameter GPT (d=128, 5 layers, 4 heads, ff=512) trained on arithmetic
(`1234+567=`, `-`, `*`, reversed answers). Loss is measured only on answer tokens,
so the numbers measure arithmetic skill, not memorized operands.

## What is here

| file | role |
|---|---|
| `data.py` | dataset (2M packed 32-token rows, fixed eval set of 16k equations) |
| `model_ref.py`, `train_ref.py` | PyTorch reference model; BF16 baseline (`--compile 1 --graph 1` = strongest) |
| `train_fp4.py` | the fused NVFP4 trainer (whole step = one CUDA graph) |
| `fp4k.cu` | NVFP4 quantizers and a generic block-scaled FP4 GEMM (inline PTX `mma ... kind::mxf4nvf4`) |
| `fused.cu`, `wprep.cu` | quantizer helpers, 2D-block weight quantization into fragment-native layouts |
| `mlp.cu` | fused LN → FC1 → GELU → FC2 (+residual) forward, and the fused MLP backward |
| `attn.cu`, `attn_bwd.cu` | fused LN → QKV → causal attention → proj forward / backward |
| `fused_bwd.cu`, `wg16.cu` | weight-gradient operand emission (random Hadamard on tensor cores) and the grouped wgrad GEMM |
| `glue.cu` | embedding, fused LM head + cross-entropy (fwd+bwd), grad-norm, AdamW |
| `kprof*.py`, `ab_*.py` | clock64 phase profilers and interleaved A/B benchmarks |

## NVFP4 recipe (NVIDIA "Pretraining LLMs with NVFP4"), as implemented

* E2M1 values, one E4M3 scale per 16 values, plus a higher-level FP32 scale
  (exact per-row for activations and dgrad operands, delayed per-tensor for wgrad operands)
* weights: 2D 16×16 blocks, so W and Wᵀ quantize identically
* gradients: stochastic rounding (software: mantissa dithering, since `cvt.rs` doesn't exist on sm_120)
* wgrad: 16-point random Hadamard transform along tokens on both operands
* all 20 linear layers in FP4 (fprop, dgrad, wgrad); attention math and the tiny LM head stay BF16
* optional: last N% of steps in BF16 (`--switch_frac`)

## Why it is fast

The model is tiny, so an unfused FP4 implementation is memory-bound and gains nothing
(the first FP4 version was *slower* than BF16). Every transformer sub-block is one
persistent kernel whose FP4 weights stay resident in shared memory. That only fits because
the weights are 4-bit. Activations never leave registers between GEMMs: a K-permutation turns
FC1's accumulator layout directly into FC2's input fragment. The FP4 tensor cores on this GPU
run 8.3x faster than BF16 (266 vs 32 TFLOPS measured).

## Run

    python train_ref.py --compile 1 --graph 1 --bench 200      # BF16 baseline step time
    python train_fp4.py --bench 200                            # NVFP4 step time
    python train_fp4.py --steps 16000 --eval_every 1000        # full NVFP4 training
    python summarize.py                                        # table from logs/final_*.log
