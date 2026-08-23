#!/usr/bin/env python3
"""Measure end-to-end uncached prefill/TTFT and single-stream decode.

The prefill figure is ``uncached_prompt_tokens / TTFT``. It includes request
scheduling and first-token decode, so it is deliberately labelled as an
end-to-end approximation rather than a kernel-only prefill rate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import httpx

from freetoken.benchmark.client import generate_prompt
from freetoken.utils import load_tokenizer


def quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def exact_prompt(tokenizer, target: int, seed: int) -> str:
    random.seed(seed)
    content_target = target
    for _ in range(16):
        prompt = generate_prompt(tokenizer, content_target)
        actual = len(tokenizer.encode(prompt, add_special_tokens=True))
        if actual == target:
            return prompt
        content_target += target - actual
        if content_target <= 0:
            raise ValueError((target, actual, content_target))
    raise ValueError(f"could not generate exact prompt target={target}")


def stream_once(
    client: httpx.Client,
    api_url: str,
    model: str,
    prompt: str,
    target: int,
    output_tokens: int,
    rep: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    last = None
    finish_reason = None
    usage = None
    events = 0
    token_events = 0
    with client.stream("POST", api_url, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            events += 1
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                text = choice.get("text")
                if text is None:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content")
                if text:
                    now = time.perf_counter()
                    token_events += 1
                    first = now if first is None else first
                    last = now
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if first is None or last is None or usage is None:
        raise RuntimeError({"first": first, "last": last, "usage": usage, "events": events})
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    ttft = first - start
    decode_window = max(last - first, 1e-9)
    decode_tps = (completion_tokens - 1) / decode_window if completion_tokens > 1 else 0.0
    return {
        "target_prompt_tokens": target,
        "rep": rep,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "uncached_prompt_tokens": prompt_tokens - cached_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "prefill_tps_uncached_over_ttft": (prompt_tokens - cached_tokens) / ttft,
        "decode_window_s": decode_window,
        "decode_tps": decode_tps,
        "wall_s": end - start,
        "post_token_overhead_s": end - last,
        "events": events,
        "token_events": token_events,
        "finish_reason": finish_reason,
    }


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for target in sorted({row["target_prompt_tokens"] for row in rows}):
        group = [row for row in rows if row["target_prompt_tokens"] == target]
        metric = {}
        for key in ("ttft_s", "prefill_tps_uncached_over_ttft", "decode_tps", "wall_s"):
            vals = [float(row[key]) for row in group]
            metric[key] = {
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "p10": quantile(vals, 0.10),
                "p90": quantile(vals, 0.90),
                "min": min(vals),
                "max": max(vals),
            }
        out.append({
            "target_prompt_tokens": target,
            "repetitions": len(group),
            "actual_prompt_tokens": sorted({row["prompt_tokens"] for row in group}),
            "cached_tokens": [row["cached_tokens"] for row in group],
            "completion_tokens": [row["completion_tokens"] for row in group],
            "finish_reasons": [row["finish_reason"] for row in group],
            "metrics": metric,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", required=True, help="Local checkpoint path used to load the tokenizer")
    ap.add_argument("--model", default="DeepSeek-V4-Flash-0731", help="Served OpenAI model name")
    ap.add_argument("--api-url", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--lengths", default="128,512,2048,8192,16384,24576,32000")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--model-revision", default="unknown")
    ap.add_argument("--freetoken-commit", default="unknown")
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    prompts = {}
    for target in lengths:
        prompts[target] = [exact_prompt(tokenizer, target, 920000 + target * 10 + rep) for rep in range(args.repetitions)]
        actual = [len(tokenizer.encode(p, add_special_tokens=True)) for p in prompts[target]]
        if actual != [target] * args.repetitions:
            raise AssertionError((target, actual))

    rows = []
    started = time.time()
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0)) as client:
        for target in lengths:
            for rep, prompt in enumerate(prompts[target], 1):
                row = stream_once(
                    client,
                    args.api_url,
                    args.model,
                    prompt,
                    target,
                    args.output_tokens,
                    rep,
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                time.sleep(0.5)
    doc = {
        "schema": "hw5959-freetoken-prefill-decode-sweep-v1",
        "model": args.model,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "freetoken_commit": args.freetoken_commit,
        "endpoint": args.api_url,
        "single_stream": True,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "ignore_eos": True,
        "output_tokens_requested": args.output_tokens,
        "started_epoch": started,
        "finished_epoch": time.time(),
        "metric_definitions": {
            "ttft_s": "client POST start to first non-empty streamed token",
            "prefill_tps_uncached_over_ttft": "(usage.prompt_tokens - usage.cached_tokens) / TTFT; end-to-end approximation including scheduling and first-token decode",
            "decode_tps": "(usage.completion_tokens - 1) / (last streamed token time - first streamed token time)",
        },
        "rows": rows,
        "summary": summarize(rows),
    }
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"out": str(out), "summary": doc["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
