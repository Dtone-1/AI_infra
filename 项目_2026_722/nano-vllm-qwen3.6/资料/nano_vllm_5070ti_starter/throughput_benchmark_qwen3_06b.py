#!/usr/bin/env python3
"""Configurable aggregate-throughput benchmark for nano-vLLM."""
from __future__ import annotations

import argparse
import json
import os
import random
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--min-input-len", type=int, default=128)
    parser.add_argument("--max-input-len", type=int, default=128)
    parser.add_argument("--min-output-len", type=int, default=128)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_input_len > args.max_input_len:
        raise ValueError("min-input-len must be <= max-input-len")
    if args.min_output_len > args.max_output_len:
        raise ValueError("min-output-len must be <= max-output-len")
    if args.max_input_len + args.max_output_len > args.max_model_len:
        raise ValueError("max input + max output must be <= max-model-len")
    if args.num_seqs > args.max_num_seqs:
        raise ValueError("num-seqs must be <= max-num-seqs")

    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    random.seed(args.seed)
    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    vocab_size = int(getattr(tokenizer, "vocab_size", 10000))
    token_high = max(2, min(vocab_size - 1, 10000))

    llm = LLM(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=args.eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Small warmup that is not included in the result.
    llm.generate([[100] * min(32, args.max_model_len - 8)], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize()

    prompts: list[list[int]] = []
    params: list[SamplingParams] = []
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    for _ in range(args.num_seqs):
        input_len = random.randint(args.min_input_len, args.max_input_len)
        output_len = random.randint(args.min_output_len, args.max_output_len)
        prompts.append([random.randint(0, token_high) for _ in range(input_len)])
        params.append(SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True))
        input_lengths.append(input_len)
        output_lengths.append(output_len)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = perf_counter() - started

    generated = sum(len(item["token_ids"]) for item in outputs)
    requested_output = sum(output_lengths)
    input_tokens = sum(input_lengths)
    result = {
        "mode": "eager" if args.eager else "cuda_graph",
        "num_seqs": args.num_seqs,
        "input_tokens": input_tokens,
        "generated_tokens": generated,
        "requested_output_tokens": requested_output,
        "elapsed_s": elapsed,
        "aggregate_output_tok_s": generated / elapsed,
        "aggregate_total_tok_s": (input_tokens + generated) / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if generated != requested_output:
        print("WARNING: generated token count differs from requested count")
    if args.json_out:
        output_path = os.path.expanduser(args.json_out)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
