# RLTrain-Flow: 分布式 DPO 训练优化系统

## 项目简介

RLTrain-Flow 是一个面向 AI 基础设施的最小可行架构（MVP），专注于：

- **分布式 DPO 训练**：支持多GPU分布式训练
- **异步推理管道**：推理和训练并行执行，最大化GPU利用率
- **前缀感知调度**：通过 vLLM 前缀缓存复用优化推理性能

## 技术亮点

### 1. 分布式 DPO 训练 (S 级优先级)
- DDP (DistributedDataParallel) 支持
- 多GPU训练
- 内存优化

**文件**：`trainer/distributed_trainer.py`, `trainer/dpo_trainer.py`

### 2. 异步推理管道 (S 级优先级)
核心优化思路：
```
生成批次 i+1 (异步)
        ||
  训练批次 i   (异步)
```

**关键点**：
- Producer-Consumer 模式
- 双缓冲优化
- 消除推理和训练间的等待时间

**文件**：`pipeline/async_pipeline.py`

### 3. 前缀感知调度 (S 级优先级)
```python
# 相同前缀 -> 同一 Worker
# 提高 vLLM prefix cache hit rate
same_prefix -> same_worker
```

**性能提升**：
- 前缀缓存复用率↑
- 推理吞吐量↑
- GPU 利用率↑

**文件**：`scheduler/cache_scheduler.py`

## 项目结构

```
RLTrain-Flow/
│
├── main.py                          # 主入口
├── requirements.txt                 # 依赖配置
│
├── trainer/                         # 训练模块
│   ├── distributed_trainer.py       # 分布式训练基类
│   └── dpo_trainer.py               # DPO 训练器
│
├── rollout/                         # 推理模块
│   ├── rollout_worker.py            # vLLM 推理 Worker
│   └── async_rollout.py             # 异步推理管理
│
├── scheduler/                       # 调度模块
│   └── cache_scheduler.py           # 前缀感知调度器
│
├── pipeline/                        # 管道模块
│   └── async_pipeline.py            # 异步推理-训练管道
│
├── benchmark/                       # 基准测试
│   └── benchmark.py                 # 性能测试工具
│
└── utils/                           # 工具模块
    ├── metrics.py                   # 指标追踪
    └── logger.py                    # 日志系统
```

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行项目

```bash
python main.py
```

### 3. 分布式训练 (可选)

```bash
# 单机多GPU
torchrun --nproc_per_node=4 main.py

# 或使用 accelerate
accelerate launch --multi_gpu main.py
```

## 核心模块说明

### DistributedTrainer (`trainer/distributed_trainer.py`)
- 初始化 DDP 进程组
- 加载和管理模型
- 保存/加载检查点

### DistributedDPOTrainer (`trainer/dpo_trainer.py`)
- 基于 DistributedTrainer
- DPO 损失计算
- 训练步骤优化

### RolloutWorker (`rollout/rollout_worker.py`)
- vLLM 高效推理引擎
- 前缀缓存支持
- 可配置采样参数

### AsyncRolloutManager (`rollout/async_rollout.py`)
- 多 Worker 推理
- 异步队列管理
- 前缀感知调度集成

### CacheAwareScheduler (`scheduler/cache_scheduler.py`)
- 提示词前缀哈希
- 相同前缀分组
- 统计信息输出

### AsyncPipeline (`pipeline/async_pipeline.py`)
- 并行执行推理和训练
- 性能日志记录
- 优雅的错误处理

## 性能指标

```
Throughput: steps/sec
Avg Loss: 减小趋势
Batch Size: 可配置
GPU Memory: ~16GB (单卡)
```

## 使用示例

### 基本使用

```python
# 初始化
trainer = DistributedDPOTrainer(model_name="Qwen/Qwen2.5-1.5B")
rollout_manager = AsyncRolloutManager(
    model_name="Qwen/Qwen2.5-1.5B",
    num_workers=2
)
pipeline = AsyncPipeline(trainer=trainer, rollout_manager=rollout_manager)

# 运行
await pipeline.run(num_rollout_iterations=100, max_training_steps=100)
```

### 自定义配置

```python
# 调整前缀长度
scheduler = CacheAwareScheduler(prefix_len=64)

# 自定义采样参数
worker.generate_with_params(
    prompts,
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)
```

## 优化建议

### 短期优化 (MVP)
1. ✅ 异步推理管道实现
2. ✅ 前缀缓存复用
3. ✅ 基础分布式训练

### 中期优化
1. 集成更强大的 DPO 算法
2. 支持多机分布式训练
3. 动态 batch size 调整
4. 梯度累积优化

### 长期优化
1. 融合内核优化 (Triton)
2. FSDP 完全分片
3. RDMA 支持
4. 自适应调度策略

## 不要做的事情 ⚠️

这个项目刻意避免：
- ❌ 重写 vLLM 推理引擎
- ❌ 自定义 NCCL 通讯
- ❌ Triton 内核开发
- ❌ RDMA 网络编程

这些会让项目迅速失控。保持聚焦很重要。

## 项目定位

**正确的定位方式**：

> "我实现了一个 **Distributed RLHF/DPO Training Optimization System**，重点优化 **rollout-training overlap** 与 **prefix cache reuse**，在 AI 基础设施领域体现了工程能力。"

**不要说**：
- ❌ "我实现了一个 RLHF 框架"（太大）
- ❌ "我做了 LLM 训练系统"（太宽泛）

## 技术栈

| 组件 | 用途 |
|------|------|
| PyTorch | 深度学习框架 |
| Transformers | 模型加载 |
| TRL | DPO 训练支持 |
| vLLM | 推理优化 |
| Accelerate | 分布式训练 |
| Ray | (可选) 分布式任务调度 |

## 性能对标

### 理论性能提升

| 优化 | 提升 |
|------|------|
| 异步管道 | ~20-30% |
| 前缀缓存 | ~15-25% |
| 多Worker | ~2-4x |
| 总体 | ~50-100% |

## 贡献和改进

欢迎提交 Issue 和 Pull Request！

核心改进方向：
- 更好的调度策略
- 额外的基准测试
- 文档和示例改进
- 新的优化技术集成

## 许可证

MIT License

## 作者

RLTrain-Flow Team

---

## 快速开始指南

### 最小化测试

```python
import asyncio
from trainer.dpo_trainer import DistributedDPOTrainer
from rollout.async_rollout import AsyncRolloutManager
from pipeline.async_pipeline import AsyncPipeline

async def test():
    trainer = DistributedDPOTrainer("Qwen/Qwen2.5-1.5B")
    rollout_mgr = AsyncRolloutManager("Qwen/Qwen2.5-1.5B", num_workers=1)
    pipeline = AsyncPipeline(trainer, rollout_mgr)
    await pipeline.run(num_rollout_iterations=5, max_training_steps=5)

asyncio.run(test())
```

---

**最后更新**: 2026-05-27
