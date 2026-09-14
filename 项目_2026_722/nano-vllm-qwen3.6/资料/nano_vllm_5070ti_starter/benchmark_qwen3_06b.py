#!/usr/bin/env python3
"""Single-request nano-vLLM benchmark with TTFT, TPOT and memory metrics."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt", default="请解释大模型推理中的 prefill 和 decode。")
    parser.add_argument("--input-tokens", type=int, default=0, help="Use synthetic token IDs of an exact length")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def main() -> None:
    args = parse_args()
    model_path = os.path.expanduser(args.model)

    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    request_input: str | list[int]
    if args.input_tokens > 0:
        vocab_size = int(getattr(tokenizer, "vocab_size", 10000))
        safe_token_id = min(1000, vocab_size - 1)
        request_input = [safe_token_id] * args.input_tokens
        prompt_tokens = args.input_tokens
    else:
        request_input = prompt
        prompt_tokens = len(tokenizer.encode(prompt))

    init_started = perf_counter()
    llm = LLM(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=args.eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    torch.cuda.synchronize()
    init_seconds = perf_counter() - init_started

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    def run_once() -> dict[str, float | int]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        llm.add_request(request_input, params)
        torch.cuda.synchronize()
        started = perf_counter()

        # In this engine, the prefill step also samples the first output token.
        torch.cuda.synchronize()
        step_started = perf_counter()
        outputs, scheduled = llm.step()
        torch.cuda.synchronize()
        first_step_seconds = perf_counter() - step_started
        if scheduled <= 0:
            raise RuntimeError("Expected the first step to be prefill")

        decode_seconds = 0.0
        decode_steps = 0
        final_outputs = outputs
        while not llm.is_finished():
            torch.cuda.synchronize()
            step_started = perf_counter()
            outputs, scheduled = llm.step()
            torch.cuda.synchronize()
            elapsed = perf_counter() - step_started
            if scheduled >= 0:
                raise RuntimeError("Expected decode step after prefill")
            decode_seconds += elapsed
            decode_steps += -scheduled
            if outputs:
                final_outputs = outputs

        e2e_seconds = perf_counter() - started
        if not final_outputs:
            raise RuntimeError("No finished output returned")
        output_tokens = len(final_outputs[0][1])
        tpot = decode_seconds / max(output_tokens - 1, 1)
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": first_step_seconds * 1000,
            "tpot_ms": tpot * 1000,
            "prefill_tok_s": prompt_tokens / first_step_seconds,
            "decode_tok_s": max(output_tokens - 1, 0) / max(decode_seconds, 1e-12),
            "e2e_s": e2e_seconds,
            "output_tok_s_e2e": output_tokens / e2e_seconds,
            "decode_steps": decode_steps,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }

    for idx in range(args.warmups):
        result = run_once()
        print(f"warmup {idx + 1}/{args.warmups}: {result['output_tok_s_e2e']:.2f} out tok/s")

    results = []
    for idx in range(args.repeats):
        result = run_once()
        results.append(result)
        print(
            f"run {idx + 1}/{args.repeats}: "
            f"TTFT={result['ttft_ms']:.2f} ms, "
            f"TPOT={result['tpot_ms']:.2f} ms, "
            f"decode={result['decode_tok_s']:.2f} tok/s, "
            f"E2E={result['e2e_s']:.3f} s, "
            f"peak_alloc={result['peak_allocated_gib']:.2f} GiB"
        )

    def values(key: str) -> list[float]:
        return [float(item[key]) for item in results]

    summary = {
        "model": model_path,
        "mode": "eager" if args.eager else "cuda_graph",
        "init_seconds": init_seconds,
        "prompt_tokens": prompt_tokens,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "ttft_ms_mean": statistics.mean(values("ttft_ms")),
        "ttft_ms_p50": percentile(values("ttft_ms"), 0.50),
        "ttft_ms_p95": percentile(values("ttft_ms"), 0.95),
        "tpot_ms_mean": statistics.mean(values("tpot_ms")),
        "tpot_ms_p50": percentile(values("tpot_ms"), 0.50),
        "tpot_ms_p95": percentile(values("tpot_ms"), 0.95),
        "decode_tok_s_mean": statistics.mean(values("decode_tok_s")),
        "output_tok_s_e2e_mean": statistics.mean(values("output_tok_s_e2e")),
        "e2e_s_mean": statistics.mean(values("e2e_s")),
        "peak_allocated_gib_max": max(values("peak_allocated_gib")),
        "peak_reserved_gib_max": max(values("peak_reserved_gib")),
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        output_path = os.path.expanduser(args.json_out)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump({"summary": summary, "runs": results}, file, ensure_ascii=False, indent=2)
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
