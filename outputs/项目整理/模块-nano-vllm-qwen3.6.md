# nano-vllm-qwen3.6 · 模块导航

[[00-AI Infra 项目总览|返回项目总览]]

围绕模型适配的增量阅读；笔记同时涉及 Qwen3.5 和 Qwen3.6，具体版本以各篇正文为准。

## 项目_2026_722/nano-vllm-qwen3.6/源码解析

- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/nano-vllm-qwen3.6_根目录文件作用与源码改造分析|nano-vllm-qwen3.6_根目录文件作用与源码改造分析]] — nano-vLLM-qwen3.6 根目录文件作用与源码改造分析
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_block_manager.py_对比分析|qwen3.6_block_manager.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_embed_head.py_对比分析|qwen3.6_embed_head.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_image_processing.py_分析|qwen3.6_image_processing.py_分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_layernorm.py_对比分析|qwen3.6_layernorm.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_linear.py_对比分析|qwen3.6_linear.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_llm_engine.py_对比分析|qwen3.6_llm_engine.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_loader.py_对比分析|qwen3.6_loader.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_model_runner.py_对比分析|qwen3.6_model_runner.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_quant.py_分析|qwen3.6_quant.py_分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_qwen3_5.py_分析|qwen3.6_qwen3_5.py_分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_qwen3_mtp.py_分析|qwen3.6_qwen3_mtp.py_分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_rotary_embedding.py_对比分析|qwen3.6_rotary_embedding.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_sampler.py_对比分析|qwen3.6_sampler.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_scheduler.py_对比分析|qwen3.6_scheduler.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_sequence.py_对比分析|qwen3.6_sequence.py_对比分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/qwen3.6_vision_encoder.py_分析|qwen3.6_vision_encoder.py_分析]]
- [[项目_2026_722/nano-vllm-qwen3.6/源码解析/README_Qwen3.5_Qwen3.6_中文版|README_Qwen3.5_Qwen3.6_中文版]] — Nano-vLLM Qwen3.5/Qwen3.6

## 项目_2026_722/nano-vllm-qwen3.6/资料

- [[项目_2026_722/nano-vllm-qwen3.6/资料/1-README_中文版|1-README_中文版]] — Nano-vLLM Qwen3.5/Qwen3.6
- [[项目_2026_722/nano-vllm-qwen3.6/资料/10-（7）Qwen3_nano-vLLM核心概念疑难点串讲|10-（7）Qwen3_nano-vLLM核心概念疑难点串讲]] — Qwen3 / nano-vLLM 核心概念疑难点串讲
- [[项目_2026_722/nano-vllm-qwen3.6/资料/11-Qwen3_nano-vLLM模型结构面试题30问|11-Qwen3_nano-vLLM模型结构面试题30问]] — Qwen3 / nano-vLLM 模型结构面试题 30+ 问
- [[项目_2026_722/nano-vllm-qwen3.6/资料/12-nano-vllm-qwen3.6_整体改动宏观分析|12-nano-vllm-qwen3.6_整体改动宏观分析]] — nano-vLLM-qwen3.6 整体改动宏观分析
- [[项目_2026_722/nano-vllm-qwen3.6/资料/13-qwen3.6_完整模型架构与关键改动分析|13-qwen3.6_完整模型架构与关键改动分析]] — Qwen3.6 完整模型架构与 nano-vLLM-qwen3.6 代码逻辑总览
- [[项目_2026_722/nano-vllm-qwen3.6/资料/14-Qwen3.6_架构疑问详细解答|14-Qwen3.6_架构疑问详细解答]] — Qwen3.5 / Qwen3.6 架构疑问详细解析：结合 nano-vLLM-qwen3.6 代码理解
- [[项目_2026_722/nano-vllm-qwen3.6/资料/15-qwen3.6_Hybrid运行时状态管理_GDN_state详解|15-qwen3.6_Hybrid运行时状态管理_GDN_state详解]] — nano-vLLM-qwen3.6 大方向二详解：Hybrid 运行时状态管理（KV Cache + GDN State）
- [[项目_2026_722/nano-vllm-qwen3.6/资料/16-qwen3.6_FP8_checkpoint加载与真正FP8量化详解|16-qwen3.6_FP8_checkpoint加载与真正FP8量化详解]] — nano-vLLM-qwen3.6 大方向三详解：FP8 checkpoint 加载与真正 FP8 推理原理
- [[项目_2026_722/nano-vllm-qwen3.6/资料/17-qwen3.6_多模态输入支持链路详解|17-qwen3.6_多模态输入支持链路详解]] — nano-vLLM-qwen3.6 大方向四详解：多模态输入支持链路
- [[项目_2026_722/nano-vllm-qwen3.6/资料/18-qwen3.6_MTP投机解码与TP输出采样链路详解|18-qwen3.6_MTP投机解码与TP输出采样链路详解]] — nano-vLLM-qwen3.6 大方向五与六详解：MTP 投机解码原型与 Tensor Parallel 输出采样链路
- [[项目_2026_722/nano-vllm-qwen3.6/资料/19-qwen3.6_运行与性能测试完整方案|19-qwen3.6_运行与性能测试完整方案]] — nano-vLLM-qwen3.6 成功运行与性能测试完整方案
- [[项目_2026_722/nano-vllm-qwen3.6/资料/2-nanovllm_qwen36_学习路线与代码分析|2-nanovllm_qwen36_学习路线与代码分析]] — Nano-vLLM Qwen3.6 项目学习路线与代码级分析
- [[项目_2026_722/nano-vllm-qwen3.6/资料/20-nano-vLLM_首次跑通后的环境原理与性能实验设计|20-nano-vLLM_首次跑通后的环境原理与性能实验设计]] — nano-vLLM 首次跑通后的环境原理与性能实验设计
- [[项目_2026_722/nano-vllm-qwen3.6/资料/21-Qwen3.5-2B_5070Ti环境配置指南|21-Qwen3.5-2B_5070Ti环境配置指南]] — nano-vLLM-qwen3.6 本地环境配置指南
- [[项目_2026_722/nano-vllm-qwen3.6/资料/21-Qwen3.5-2B环境配置指南 |21-Qwen3.5-2B环境配置指南 ]] — nano-vLLM-qwen3.6 本地配置步骤（按当前文件夹结构）
- [[项目_2026_722/nano-vllm-qwen3.6/资料/3-GPU内部自注意力计算流程学习笔记|3-GPU内部自注意力计算流程学习笔记]] — GPU 内部自注意力计算流程课程学习笔记
- [[项目_2026_722/nano-vllm-qwen3.6/资料/4-FlashAttention课程内容深度整理学习笔记|4-FlashAttention课程内容深度整理学习笔记]] — FlashAttention 课程内容深度整理学习笔记
- [[项目_2026_722/nano-vllm-qwen3.6/资料/5-大模型数字格式与量化基础学习笔记|5-大模型数字格式与量化基础学习笔记]] — 大模型数字格式与量化基础学习笔记
- [[项目_2026_722/nano-vllm-qwen3.6/资料/6-Qwen3_Dense_forward与Qwen3.5_3.6_Hybrid_GatedDeltaNet详解|6-Qwen3_Dense_forward与Qwen3.5_3.6_Hybrid_GatedDeltaNet详解]] — Qwen3 Dense Forward 与 Qwen3.5 / Qwen3.6 Hybrid 架构详解
- [[项目_2026_722/nano-vllm-qwen3.6/资料/7-Qwen3_DecoderOnly架构与nano-vLLM模型代码对应总览|7-Qwen3_DecoderOnly架构与nano-vLLM模型代码对应总览]] — Qwen3 Decoder-only 架构与 nano-vLLM 模型代码对应总览
- [[项目_2026_722/nano-vllm-qwen3.6/资料/8-Qwen3_MLP_FFN_gate_up_down线性层详解|8-Qwen3_MLP_FFN_gate_up_down线性层详解]] — Qwen3 MLP / FFN 中 gate_proj、up_proj、down_proj 线性层详解
- [[项目_2026_722/nano-vllm-qwen3.6/资料/9-Qwen3_RMSNorm_LayerNorm与残差连接详解|9-Qwen3_RMSNorm_LayerNorm与残差连接详解]] — Qwen3 中 LayerNorm / RMSNorm 与残差连接详解

## 项目_2026_722/nano-vllm-qwen3.6/资料/nano_vllm_5070ti_starter

- [[项目_2026_722/nano-vllm-qwen3.6/资料/nano_vllm_5070ti_starter/nano-vLLM_5070Ti_从零跑通与实验指标指南|nano-vLLM_5070Ti_从零跑通与实验指标指南]]（源码附带文档） — nano-vLLM-qwen3.6：RTX 5070 Ti Laptop 从零跑通、代码链路与性能实验指南

