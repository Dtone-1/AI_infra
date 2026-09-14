# nano-kvllm · 模块导航

[[00-AI Infra 项目总览|返回项目总览]]

理解 KV Cache 压缩方案，以及它对缓存管理和 Attention 执行的影响。

## 项目_2026_722/nano-kvllm/源码分析

- [[项目_2026_722/nano-kvllm/源码分析/1-kvllm_llm_engine_对比分析|1-kvllm_llm_engine_对比分析]] — nano-vLLM 与 nano-kvLLM：llm_engine.py 源码对比分析
- [[项目_2026_722/nano-kvllm/源码分析/2-kvllm_sequence_对比分析|2-kvllm_sequence_对比分析]] — nano-vLLM 与 nano-kvLLM：sequence.py 源码对比分析
- [[项目_2026_722/nano-kvllm/源码分析/3-kvllm_scheduler_对比分析|3-kvllm_scheduler_对比分析]] — nano-vLLM 与 nano-kvLLM：scheduler.py 源码对比分析
- [[项目_2026_722/nano-kvllm/源码分析/4-kvllm_block_manager_对比分析|4-kvllm_block_manager_对比分析]] — nano-vLLM 与 nano-kvLLM：block_manager.py 源码对比分析
- [[项目_2026_722/nano-kvllm/源码分析/5-kvllm_model_runner_对比分析|5-kvllm_model_runner_对比分析]] — nano-vLLM 与 nano-kvLLM：model_runner.py 源码对比分析
- [[项目_2026_722/nano-kvllm/源码分析/6-kvllm_attention_对比分析|6-kvllm_attention_对比分析]] — nano-vLLM 与 nano-kvLLM：attention.py 源码对比分析
- [[项目_2026_722/nano-kvllm/源码分析/7-kvLLM_compress_utils_分析|7-kvLLM_compress_utils_分析]] — nano-kvLLM 新增源码分析：compress_utils.py
- [[项目_2026_722/nano-kvllm/源码分析/8-kvLLM_CompressMethod_分析|8-kvLLM_CompressMethod_分析]] — nano-kvLLM 新增源码分析：CompressMethod.py

## 项目_2026_722/nano-kvllm/资料

- [[项目_2026_722/nano-kvllm/资料/1-nano-kvLLM核心思想与整体原理分析|1-nano-kvLLM核心思想与整体原理分析]] — nano-kvLLM 核心思想与整体原理分析
- [[项目_2026_722/nano-kvllm/资料/2-nano-kvLLM项目整体改动与KVCache压缩全链路分析|2-nano-kvLLM项目整体改动与KVCache压缩全链路分析]] — nano-kvLLM 项目整体改动与 KV Cache 压缩全链路分析
- [[项目_2026_722/nano-kvllm/资料/3-nano-kvLLM_一次请求从Prefill到多次压缩的完整生命周期|3-nano-kvLLM_一次请求从Prefill到多次压缩的完整生命周期]] — nano-kvLLM：一次请求从 Prefill 到多次 KV Cache 压缩的完整生命周期
- [[项目_2026_722/nano-kvllm/资料/4-nano-kvLLM_压缩瞬间完整调用链与SnapKV_TopK详解|4-nano-kvLLM_压缩瞬间完整调用链与SnapKV_TopK详解]] — nano-kvLLM：压缩发生瞬间的完整函数调用链与 SnapKV Top-K 选择详解
- [[项目_2026_722/nano-kvllm/资料/5-Qwen3到Qwen3.5_KVCache压缩融合改动详解|5-Qwen3到Qwen3.5_KVCache压缩融合改动详解]] — 从 Qwen3 到 Qwen3.5 Hybrid：KV Cache 压缩融合改动详解
- [[项目_2026_722/nano-kvllm/资料/6-nano_kvLLM_SnapKV筛选机制代码与原理详解|6-nano_kvLLM_SnapKV筛选机制代码与原理详解]] — nano-kvLLM 中 SnapKV 筛选机制：从代码、数学到工程落地的完整理解
- [[项目_2026_722/nano-kvllm/资料/nano-kvllm-v0.2.0-中文版|nano-kvllm-v0.2.0-中文版]] — nano-kvllm v0.2.0

