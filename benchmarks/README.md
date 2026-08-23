# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_dsv4_stream.py`** — exact-token, uncached prompt sweep against an already
running OpenAI-compatible server. It reports client-observed TTFT, an end-to-end
prefill approximation (`uncached prompt tokens / TTFT`), and single-stream decode
from the first through last streamed token. Prompts are unique across repetitions
and `ignore_eos` keeps the requested decode window fixed.

```bash
PYTHONPATH=python:. python benchmarks/bench_dsv4_stream.py \
  --model-path /path/to/DeepSeek-V4-Flash-0731 \
  --api-url http://127.0.0.1:8000/v1/completions \
  --out /tmp/dsv4-stream.json
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
