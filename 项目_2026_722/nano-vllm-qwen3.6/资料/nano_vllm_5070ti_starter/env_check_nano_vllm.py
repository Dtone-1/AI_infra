#!/usr/bin/env python3
"""Environment diagnostics for nano-vLLM on NVIDIA GPUs."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        text = (result.stdout or result.stderr).strip()
        return text if text else f"exit_code={result.returncode}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def pkg_version(name: str) -> str:
    try:
        module: Any = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        return f"IMPORT FAILED: {exc}"


def main() -> int:
    print("=" * 72)
    print("nano-vLLM environment check")
    print("=" * 72)
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable: {sys.executable}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '<not set>')}")
    print(f"PATH has nvcc: {shutil.which('nvcc')}")
    print(f"nvcc: {run_cmd(['nvcc', '--version']) if shutil.which('nvcc') else 'NOT FOUND'}")
    print(f"nvidia-smi: {run_cmd(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']) if shutil.which('nvidia-smi') else 'NOT FOUND'}")
    print()

    print(f"torch: {pkg_version('torch')}")
    try:
        import torch

        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        print(f"torch compiled arch list: {torch.cuda.get_arch_list()}")
        if not torch.cuda.is_available():
            print("[FAIL] PyTorch cannot access CUDA.")
            return 2

        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cap = torch.cuda.get_device_capability(idx)
            print(
                f"GPU {idx}: {props.name}; capability={cap[0]}.{cap[1]}; "
                f"VRAM={props.total_memory / 1024**3:.2f} GiB"
            )
            arch = f"sm_{cap[0]}{cap[1]}"
            if arch not in torch.cuda.get_arch_list():
                print(f"[FAIL] Current PyTorch wheel does not contain {arch} kernels.")
                return 3

        x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        print(f"CUDA matmul: OK, checksum={float(y[0, 0]):.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] PyTorch CUDA test failed: {exc}")
        return 4

    print()
    print(f"triton: {pkg_version('triton')}")
    print(f"transformers: {pkg_version('transformers')}")
    print(f"flash_attn: {pkg_version('flash_attn')}")
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        assert callable(flash_attn_varlen_func)
        assert callable(flash_attn_with_kvcache)
        print("flash_attn required APIs: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Required FlashAttention APIs unavailable: {exc}")
        return 5

    try:
        import nanovllm

        print(f"nanovllm import: OK ({nanovllm.__file__})")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] nanovllm import failed: {exc}")
        return 6

    print()
    print("[PASS] Core environment is ready for the Qwen3-0.6B smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
