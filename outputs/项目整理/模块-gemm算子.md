# gemm算子 · 模块导航

[[00-AI Infra 项目总览|返回项目总览]]

从 CUDA 线程与存储层次进入 GEMM 分块计算，连接源码分析、性能测试与面试表达。

## 跨目录阅读路线

- [[outputs/项目整理/专题-05-GEMM与推理性能|专题-05-GEMM与推理性能]]

## 项目_2026_722/gemm算子/资料

- [[项目_2026_722/gemm算子/资料/1-CUDA第2章_从C++编程到CUDA线程模型与兼容性|1-CUDA第2章_从C++编程到CUDA线程模型与兼容性]] — 一、本章在 CUDA 学习体系中的位置与学习目标
- [[项目_2026_722/gemm算子/资料/10-gemm_launcher_分析|10-gemm_launcher_分析]] — launcher.cu 源码整体分析
- [[项目_2026_722/gemm算子/资料/11-gemm_launcher_精读|11-gemm_launcher_精读]] — gemm_launcher_精读
- [[项目_2026_722/gemm算子/资料/12-gemm_gemm_warp_tile_分析|12-gemm_gemm_warp_tile_分析]] — gemm_warp_tile.cu 源码整体分析
- [[项目_2026_722/gemm算子/资料/13-gemm_gemm_warp_tile_精读|13-gemm_gemm_warp_tile_精读]] — gemm_gemm_warp_tile_精读
- [[项目_2026_722/gemm算子/资料/14-gemm_gemm_warp_tile_config_精读|14-gemm_gemm_warp_tile_config_精读]] — gemm_gemm_warp_tile_config_精读
- [[项目_2026_722/gemm算子/资料/15-gemm_gemm_warp_tile_config_分析|15-gemm_gemm_warp_tile_config_分析]] — gemm_warp_tile_config.h 源码整体分析
- [[项目_2026_722/gemm算子/资料/16-GPU内部架构与CUDA_GEMM学习笔记|16-GPU内部架构与CUDA_GEMM学习笔记]] — GPU 内部架构与 CUDA/GEMM 学习笔记
- [[项目_2026_722/gemm算子/资料/17-GEMM矩阵划分到线程执行的完整路线详解|17-GEMM矩阵划分到线程执行的完整路线详解]] — 当前 GEMM 项目中“完整矩阵 → Thread Block Tile → Warp Tile → Thread Tile → 线程寄存器输出元素”的完整划分与执行路线
- [[项目_2026_722/gemm算子/资料/18-GEMM项目数据完整流动详解|18-GEMM项目数据完整流动详解]] — GEMM 项目数据流动详解：Global Memory → Shared Memory → Registers → Global Memory
- [[项目_2026_722/gemm算子/资料/19-gemm_warp_tile_代码块宏观作用分析|19-gemm_warp_tile_代码块宏观作用分析]] — gemm_warp_tile.cu 代码块宏观作用分析
- [[项目_2026_722/gemm算子/资料/2-CUDA第3章_矩阵加法错误检查计时与线程组织|2-CUDA第3章_矩阵加法错误检查计时与线程组织]] — 一、本章在 CUDA 学习体系中的位置与学习目标
- [[项目_2026_722/gemm算子/资料/20-GEMM项目基础概念问答-2026-07-21|20-GEMM项目基础概念问答-2026-07-21]] — 基础概念问答复习笔记
- [[项目_2026_722/gemm算子/资料/21-面试手撕_最简单CUDA_GEMM算子详解|21-面试手撕_最简单CUDA_GEMM算子详解]] — 面试手撕：最简单的 CUDA GEMM 算子
- [[项目_2026_722/gemm算子/资料/22-CUDA_Naive_GEMM_矩阵行列运算与Kernel_Launch初学者详解|22-CUDA_Naive_GEMM_矩阵行列运算与Kernel_Launch初学者详解]] — CUDA Naive GEMM：从矩阵行列运算到 Kernel Launch 的完整初学者讲解
- [[项目_2026_722/gemm算子/资料/23-GEMM项目_按执行时间顺序理解完整优化流程_4000字版|23-GEMM项目_按执行时间顺序理解完整优化流程_4000字版]] — 当前 GEMM 项目：按真实执行顺序理解完整优化流程
- [[项目_2026_722/gemm算子/资料/3-CUDA第4章_GPU硬件资源内存模型与性能基础|3-CUDA第4章_GPU硬件资源内存模型与性能基础]] — 一、本章在 CUDA 学习体系中的位置与学习目标
- [[项目_2026_722/gemm算子/资料/4-GEMM项目从零学习与源码阅读路线|4-GEMM项目从零学习与源码阅读路线]] — GEMM 项目从零学习与源码阅读路线
- [[项目_2026_722/gemm算子/资料/5-GEMM项目源码阅读前置基础_从零入门|5-GEMM项目源码阅读前置基础_从零入门]] — GEMM 项目源码阅读前置基础：从零建立 C++、CUDA 与矩阵乘法知识
- [[项目_2026_722/gemm算子/资料/6-gemm_CMakeLists.txt_分析|6-gemm_CMakeLists.txt_分析]] — 1. CMakeLists.txt 的整体概览和该文件的意义
- [[项目_2026_722/gemm算子/资料/7-gemm_CMakeLists_精读|7-gemm_CMakeLists_精读]] — gemm_CMakeLists_精读
- [[项目_2026_722/gemm算子/资料/8-gemm_test_perf_gemm_分析|8-gemm_test_perf_gemm_分析]] — test_perf_gemm.cpp 源码整体分析
- [[项目_2026_722/gemm算子/资料/9-gemm_test_perf_gemm_精读|9-gemm_test_perf_gemm_精读]] — gemm_test_perf_gemm_精读
- [[项目_2026_722/gemm算子/资料/CUDA_GEMM项目从零学习与大厂算子路线|CUDA_GEMM项目从零学习与大厂算子路线]] — CUDA FP32 GEMM 项目从零学习与大厂算子岗位能力路线
- [[项目_2026_722/gemm算子/资料/CUDA基础课程章节学习文档生成提示词|CUDA基础课程章节学习文档生成提示词]] — CUDA 基础课程章节学习文档生成提示词
- [[项目_2026_722/gemm算子/资料/gemm源码分析提示词|gemm源码分析提示词]]
- [[项目_2026_722/gemm算子/资料/gemm源码精读提示词|gemm源码精读提示词]]

## 项目_2026_722/gemm算子/面试qa

- [[项目_2026_722/gemm算子/面试qa/CUDA_GEMM_面试问答|CUDA_GEMM_面试问答]] — CUDA FP32 GEMM 项目面试问答

## 项目_2026_722/gemm算子/面试qa/gemm-project

- [[项目_2026_722/gemm算子/面试qa/gemm-project/README|README]]（源码附带文档） — GEMM 内核基线

## 项目_2026_722/gemm算子/面试qa/gemm-project/docs

- [[项目_2026_722/gemm算子/面试qa/gemm-project/docs/stable_baseline|stable_baseline]]（源码附带文档） — Stable Baseline Note
- [[项目_2026_722/gemm算子/面试qa/gemm-project/docs/warp_tiles|warp_tiles]]（源码附带文档） — warp tiles开发文档

## 项目_2026_722/gemm算子/面试qa/gemm-project/microbenchmark

- [[项目_2026_722/gemm算子/面试qa/gemm-project/microbenchmark/README|README]]（源码附带文档） — for Ada Lovelace GPU (RTX 4090, sm_89):

