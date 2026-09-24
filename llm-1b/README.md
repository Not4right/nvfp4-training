# 1B LLM in 8 GB: NVFP4 + Muon on an RTX 5060

A 1.0B-parameter decoder-only language model whose entire training state fits in the 8 GB of a
consumer GPU.

## Model

| | |
|---|---|
| parameters | 1.0025B |
| layers / width | 20 / 2048 |
| attention | 16 query heads, 4 KV heads (GQA), RoPE (theta 1e5), QK-norm |
| MLP | SwiGLU, 5632 |
| vocab | 49,152 (SmolLM2 tokenizer) |
| context | 2048 in pretraining (4096 max) |

## How it fits and why it is fast

- **Hand-written forward and backward** (`model2.py`): autograd is not used anywhere; attention
  calls cuDNN's fused kernels directly, or the custom FP8 kernel.
- **NVFP4 GEMMs** through cuBLAS FP4 (`torch._scaled_mm`, 237 TFLOPS) with the swizzled 128x4
  scale layout, fed by producer kernels in `emit.cu`: exact per-row FP32 scales for row operands,
  random Hadamard transform plus delayed amax for the weight-gradient column operands.
- **Weights quantized just in time** instead of cached (a per-step FP4 weight cache would cost 1.1 GB).
- **Muon with BF16 momentum and no gradient buffer:** each microbatch's weight gradient is
  stochastically rounded straight into the momentum. BF16 master weights, also updated with
  stochastic rounding.
- **FP8 attention** (`attn8.cu`): forward 48.5 TFLOPS vs 27.9 for cuDNN (1.74x), backward 1.07x,
  about +6.4% end to end. Q and K unscaled (QK-norm bounds them), per-channel scales for V and dO,
  and dP kept in BF16, since it is the error-sensitive term. Used only in the FP4 phase; the BF16
  tail and SFT use cuDNN.

Speed went from 6.0k to 9.2k tokens/s at batch 2 x 2048 and 6.6 GB peak, and to ~9.35k with FP8 attention.

## Training plan (`train.py`)

- 10B tokens pretraining, batch 524,288 tokens, peak LR 6e-4, 300 warmup steps
- first 85% of steps in NVFP4, the rest in BF16 with a 1-sqrt cooldown
- then supervised fine-tuning
- data (`prep_data.py`): FineWeb-Edu, Ultra-FineWeb, DCLM, FineMath, InfiWebMath and Cosmopedia v2
  for pretraining; SmolTalk2, OpenMathInstruct-2, NuminaMath-CoT and Magpie-Ultra for the
  reasoning anneal and SFT
- hourly checkpoints with automatic resume (`run.ps1` supervisor, `lm.ps1` control)

## Files

| file | role |
|---|---|
| `train.py` | training loop, schedule, checkpoints, metrics |
| `model2.py` | the model with hand-written fwd/bwd |
| `emit.cu` / `emit.py` | NVFP4 operand producers (scales, Hadamard, SR) |
| `fp4k.cu`, `fp4ops.py`, `nvq.cu`, `nvq.py` | FP4 quantization kernels and wrappers |
| `attn8.cu` / `attn8.py` | FP8 attention forward and backward |
| `optim.py` | Muon + AdamW with stochastic rounding |
| `loader.py`, `prep_data.py` | tokenized data pipeline |
| `gen.py` | sampling from a checkpoint |
| `test_*.py`, `bench*.py`, `prof*.py` | correctness tests against references, benchmarks, profilers |

## Requirements

NVIDIA Blackwell GPU with FP4 tensor cores (developed on sm_120, RTX 5060), CUDA 12.9+,
PyTorch 2.9. Kernels are compiled at runtime with NVRTC.
