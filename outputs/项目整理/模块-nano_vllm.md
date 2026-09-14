# nano_vllm · 模块导航

[[00-AI Infra 项目总览|返回项目总览]]

先读整体调用关系，再沿请求入口、调度、缓存管理、模型执行、采样逐文件阅读。

## 跨目录阅读路线

- [[outputs/项目整理/专题-01-请求从输入到输出|专题-01-请求从输入到输出]]
- [[outputs/项目整理/专题-02-KV缓存与调度|专题-02-KV缓存与调度]]
- [[outputs/项目整理/专题-03-Qwen模型适配与MTP|专题-03-Qwen模型适配与MTP]]

## 项目_2026_722/nano_vllm/复习资料

- [[项目_2026_722/nano_vllm/复习资料/AIInfra推理岗位_nano-vLLM_vLLM面试问答_50题|AIInfra推理岗位_nano-vLLM_vLLM面试问答_50题]] — AI Infra 推理岗位面试问答：nano-vLLM / vLLM / LLM Serving 50 题
- [[项目_2026_722/nano_vllm/复习资料/nano-vllm_code_learning_route|nano-vllm_code_learning_route]] — nano-vLLM 代码学习路线（精简版）
- [[项目_2026_722/nano_vllm/复习资料/nano-vLLM基础概念问答-2026-06-12|nano-vLLM基础概念问答-2026-06-12]] — 基础概念问答复习笔记
- [[项目_2026_722/nano_vllm/复习资料/nano-vLLM基础概念问答-2026-06-14|nano-vLLM基础概念问答-2026-06-14]] — 基础概念问答复习笔记
- [[项目_2026_722/nano_vllm/复习资料/nano-vllm源码结构与调用关系总览|nano-vllm源码结构与调用关系总览]] — nano-vLLM 源码结构与调用关系总览
- [[项目_2026_722/nano_vllm/复习资料/nanovllm_chunked_prefill规则完整详解|nanovllm_chunked_prefill规则完整详解]] — nano-vLLM Chunked Prefill 规则完整详解
- [[项目_2026_722/nano_vllm/复习资料/nanovllm_推理全过程|nanovllm_推理全过程]] — nano-vLLM 推理全过程小白版梳理
- [[项目_2026_722/nano_vllm/复习资料/主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili_学习笔记|主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili_学习笔记]] — 课程学习笔记：主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili.txt
- [[项目_2026_722/nano_vllm/复习资料/源码代码问题问答-2026-06-13|源码代码问题问答-2026-06-13]] — 基础概念问答复习笔记
- [[项目_2026_722/nano_vllm/复习资料/源码代码问题问答-2026-06-14|源码代码问题问答-2026-06-14]] — 基础概念问答复习笔记
- [[项目_2026_722/nano_vllm/复习资料/源码代码问题问答-2026-06-17|源码代码问题问答-2026-06-17]] — 基础概念问答复习笔记

## 项目_2026_722/nano_vllm/源码解析

- [[项目_2026_722/nano_vllm/源码解析/activation.py_解析|activation.py_解析]] — activation.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/attention.py_解析|attention.py_解析]] — attention.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/bench.py_解析|bench.py_解析]] — bench.py 源码逐行精读解析
- [[项目_2026_722/nano_vllm/源码解析/block_manager.py_解析|block_manager.py_解析]] — block_manager.py 源码解析
- [[项目_2026_722/nano_vllm/源码解析/config.py_解析|config.py_解析]] — config.py 源码逐行精读解析
- [[项目_2026_722/nano_vllm/源码解析/context.py_解析|context.py_解析]]
- [[项目_2026_722/nano_vllm/源码解析/embed_head.py_解析|embed_head.py_解析]] — embed_head.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/example.py_解析|example.py_解析]] — example.py 源码解析
- [[项目_2026_722/nano_vllm/源码解析/layernorm.py_解析|layernorm.py_解析]] — layernorm.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/linear.py_解析|linear.py_解析]] — linear.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/llm_engine.py_解析|llm_engine.py_解析]] — llm_engine.py 源码解析
- [[项目_2026_722/nano_vllm/源码解析/loader.py_解析|loader.py_解析]] — loader.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/model_runner.py_解析|model_runner.py_解析]] — model_runner.py 源码逐行精读与框架原理解析
- [[项目_2026_722/nano_vllm/源码解析/nanovllm_engine源码结构与调用关系详解|nanovllm_engine源码结构与调用关系详解]] — nano-vLLM engine/ 源码结构与调用关系详解
- [[项目_2026_722/nano_vllm/源码解析/qwen3.py_解析|qwen3.py_解析]] — qwen3.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/README_中文|README_中文]] — Nano-vLLM
- [[项目_2026_722/nano_vllm/源码解析/rotary_embedding.py_解析|rotary_embedding.py_解析]] — rotary_embedding.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/sampler.py_解析|sampler.py_解析]] — sampler.py 源码宏观解析
- [[项目_2026_722/nano_vllm/源码解析/scheduler.py_解析|scheduler.py_解析]] — scheduler.py 源码逐行精读解析
- [[项目_2026_722/nano_vllm/源码解析/sequence.py_解析|sequence.py_解析]] — sequence.py 源码逐行精读与框架原理解析

## 项目_2026_722/nano_vllm/课程学习资料

- [[项目_2026_722/nano_vllm/课程学习资料/主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili_学习笔记|主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili_学习笔记]] — 课程学习笔记：主题一_nano-vllm 推理全流程串讲_概览篇__哔哩哔哩_bilibili.txt
- [[项目_2026_722/nano_vllm/课程学习资料/主题二_nano-vllm 之 PagedAttention 与内存管理_哔哩哔哩_bilibili_学习笔记|主题二_nano-vllm 之 PagedAttention 与内存管理_哔哩哔哩_bilibili_学习笔记]] — 课程学习笔记：主题二_nano-vllm 之 PagedAttention 与内存管理_哔哩哔哩_bilibili

