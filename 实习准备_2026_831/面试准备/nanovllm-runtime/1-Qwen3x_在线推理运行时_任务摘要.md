# Qwen3.x 在线推理运行时：任务摘要

以 `nano-vllm-qwen3.6` 为基线，重构一个单机四卡的 Qwen3.x Dense/MoE 在线推理运行时。

1. **模型适配系统**：以统一 Adapter/ModelSpec 支持 Qwen3 Dense、Qwen3.5 Hybrid Dense、Qwen3.6 Hybrid MoE、Qwen3.8 Hybrid Dense；不支持 Qwen3.8 超大 MoE。
2. **Qwen3.6 Hybrid MoE**：实现 EP=4 专家并行，包括 Top-K 路由、专家分片、Token Dispatch/Combine 与 TP/EP 协同执行。
3. **Hybrid Cache Manager**：统一管理 Paged KV、GDN recurrent/conv state、KV 压缩、抢占和回收；实现共享前缀只读、压缩私有后缀，以及修改共享缓存时的写时复制。
4. **SLO 感知调度**：实现 Decode 优先 Token Budget、多请求 Chunked Prefill、Prefill 保底与请求老化；根据 TTFT、TPOT、队列、显存和抢占压力动态调整预算与准入。
5. **在线服务层**：提供 OpenAI 兼容 API、SSE 流式输出、异步 Engine Loop、取消/超时、过载控制和 Prometheus 指标。
6. **工程交付**：完成 TP4/EP4 正确性、资源泄漏、在线混合负载及 P99 延迟压测；沉淀可复现 Benchmark、测试、Docker、架构文档和模型支持矩阵。

最终定位：**面向 Qwen3.x 混合架构的单机多卡在线推理运行时原型**。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-面试准备|模块-面试准备]]

%% 项目关联导航：结束 %%
