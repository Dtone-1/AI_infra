#!/usr/bin/env python3
"""Conservative first-run smoke test for nano-vLLM + Qwen3-0.6B."""
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt", default="请用三句话介绍一下你自己。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--graph", action="store_true", help="Enable CUDA Graph after eager mode works")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = os.path.expanduser(args.model)

    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("Loading model from:", model_path, flush=True)
    llm = LLM(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=not args.graph,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.80,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=True,
    )

    print("\n=== OUTPUT ===")
    print(outputs[0]["text"].replace("<|im_end|>", ""))
    print("\nGenerated token count:", len(outputs[0]["token_ids"]))


if __name__ == "__main__":
    main()
