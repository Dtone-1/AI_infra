<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

一个从零开始构建的轻量级 vLLM 实现。

## 主要特性

* 🚀 **快速离线推理** —— 推理速度可与 vLLM 相媲美
* 📖 **代码易于阅读** —— 使用约 1,200 行 Python 代码实现，结构简洁清晰
* ⚡ **完整的优化套件** —— 支持前缀缓存、张量并行、Torch 编译、CUDA Graph 等优化功能

## 安装

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## 模型下载

如需手动下载模型权重，可使用以下命令：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

使用方法请参考 `example.py`。其 API 与 vLLM 的接口基本一致，仅 `LLM.generate` 方法存在少量差异：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 性能测试

性能测试方法请参考 `bench.py`。

**测试配置：**

- 硬件：RTX 4070 Laptop（8 GB）
- 模型：Qwen3-0.6B
- 请求总数：256 条序列
- 输入长度：在 100～1024 个 token 之间随机采样
- 输出长度：在 100～1024 个 token 之间随机采样

**性能结果：**

| 推理引擎 | 输出 Token 数 | 耗时（秒） | 吞吐量（token/s） |
|---|---:|---:|---:|
| vLLM | 133,966 | 98.37 | 1361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1434.13 |

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-nano_vllm|模块-nano_vllm]]

%% 项目关联导航：结束 %%
