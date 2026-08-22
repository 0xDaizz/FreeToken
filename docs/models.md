# Supported models

FreeToken loads HF safetensors checkpoints directly (plus native GGUF for
Gemma-4). The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.6 dense | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

### DeepSeek-V4 multi-GPU ETP

DeepSeek-V4 supports `--tp-size N` on the offload-family backends as
expert-intermediate tensor parallel (ETP). The attention trunk, router and shared
expert stay replicated on every rank; the native FP4 routed experts split their
intermediate dimension across ranks. Each rank computes a hidden-size partial and
the existing MoE collective sums those partials before the replicated shared expert
is added. Thus TP2 stores two complementary half banks instead of duplicating the
full host expert bank, while the per-GPU cache holds half-sized payloads for the same
logical LRU expert IDs.

- The checkpoint expert intermediate width must divide evenly into TP ranks and
  each local width must be 128-aligned (the official V4-Flash `I=2048` supports TP2).
- Use `offload`, `cpu`, or `hybrid`; resident `fused` DSV4 TP is rejected before
  weight allocation.
- `--moe-cpu-threads` is per rank. TP ranks automatically receive disjoint physical
  core ranges; an explicit count may leave tail cores for the scheduler/collectives.
- Use the native safetensors checkpoint. Legacy FTW DSV4 banks do not carry rank
  partition metadata and are rejected under TP>1 rather than loaded once per rank.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Multimodal checkpoints are served text-only.
