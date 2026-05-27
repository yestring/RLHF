# 快速参考指南

## 🚀 30秒项目概览

**RLTrain-Flow** 是一个 AI 基础设施项目，包含：
- ✅ 分布式 DPO 训练（多GPU）
- ✅ 异步推理管道（推理和训练并行）
- ✅ 前缀缓存优化（提高GPU利用率）

---

## ⚡ 快速命令

### 1. 项目验证（推荐首先运行）
```bash
python verify_project.py
```
✓ 验证所有文件和模块是否正确创建

### 2. 快速演示（无需模型）
```bash
python quick_start.py
```
✓ 5个演示：调度、指标、结构、优化、异步

### 3. 单元测试
```bash
python test.py
```
✓ 验证核心功能（调度、指标、异步）

### 4. 使用示例
```bash
python examples.py
```
✓ 5个完整使用示例

### 5. 完整项目（需要模型和GPU）
```bash
# 单GPU
python main.py

# 多GPU
torchrun --nproc_per_node=4 main.py
```

---

## 📁 文件导航

| 文件 | 用途 | 优先级 |
|------|------|-------|
| `main.py` | 项目主入口 | ⭐⭐⭐ |
| `quick_start.py` | 快速演示 | ⭐⭐⭐ |
| `README.md` | 项目文档 | ⭐⭐ |
| `TECHNICAL.md` | 技术详解 | ⭐⭐ |
| `INSTALL.md` | 安装指南 | ⭐⭐ |
| `PROJECT_SUMMARY.md` | 项目总结 | ⭐ |
| `config.py` | 配置管理 | ⭐ |
| `examples.py` | 使用示例 | ⭐ |
| `test.py` | 单元测试 | ⭐ |

### 核心模块

| 模块 | 功能 | 关键类 |
|------|------|--------|
| `trainer/` | 分布式训练 | `DistributedTrainer`, `DistributedDPOTrainer` |
| `rollout/` | 异步推理 | `RolloutWorker`, `AsyncRolloutManager` |
| `scheduler/` | 前缀调度 | `CacheAwareScheduler` |
| `pipeline/` | 异步管道 | `AsyncPipeline` |
| `utils/` | 工具函数 | `AverageMeter`, `MetricsTracker`, `logger` |
| `benchmark/` | 性能测试 | `Benchmark` |

---

## 💡 常见任务

### 任务 1: 理解项目架构
```
1. 运行: python quick_start.py
2. 阅读: README.md
3. 查看: TECHNICAL.md (架构图解)
```

### 任务 2: 自定义配置
```python
from config import Config, TrainerConfig

config = Config(
    trainer=TrainerConfig(
        model_name="Qwen/Qwen2.5-1.5B",
        batch_size=16
    )
)
```

### 任务 3: 添加自定义指标
```python
from utils.metrics import MetricsTracker

tracker = MetricsTracker()
tracker.add_meter("my_metric")
tracker.update("my_metric", value)
print(tracker.get_all_avgs())
```

### 任务 4: 运行单个模块
```python
# 测试前缀调度
from scheduler.cache_scheduler import CacheAwareScheduler
scheduler = CacheAwareScheduler()
grouped = scheduler.group_prompts(prompts)

# 测试指标
from utils.metrics import AverageMeter
meter = AverageMeter("loss")
meter.update(0.5)
print(meter.avg)
```

---

## 🔧 配置调整

### 常用参数

```python
# trainer/config.py
TrainerConfig(
    model_name="Qwen/Qwen2.5-1.5B",
    learning_rate=1e-4,           # 学习率
    batch_size=32,                 # 批大小
    max_grad_norm=1.0              # 梯度裁剪
)

# rollout/config.py
RolloutConfig(
    model_name="Qwen/Qwen2.5-1.5B",
    num_workers=2,                 # 推理Worker数
    temperature=0.8,               # 采样温度
    max_tokens=128                 # 最大生成长度
)

# scheduler/config.py
SchedulerConfig(
    prefix_len=32                  # 前缀长度
)
```

---

## 📊 性能指标

### 预期性能提升

| 优化 | 性能提升 |
|------|----------|
| 异步管道 | ~30% 延迟减少 |
| 前缀缓存 | ~25% 计算减少 |
| DDP (4GPU) | ~3-4x 吞吐量 |
| **总体** | **~50-100%** |

### 监控指标

```python
# 创建指标追踪器
tracker = MetricsTracker()

# 重要指标
tracker.add_meter("loss")
tracker.add_meter("throughput")
tracker.add_meter("latency")

# 定期查看
print(tracker.get_all_avgs())
```

---

## 🐛 故障排除

### 问题：模块导入失败
```bash
# 检查文件是否存在
ls -la trainer/

# 检查 Python 路径
python -c "import sys; print(sys.path)"
```

### 问题：GPU 显存不足
```python
# 减少 batch_size
batch_size = 8  # 从 32 减小

# 启用梯度检查点
model.enable_gradient_checkpointing()
```

### 问题：异步任务卡住
```python
# 增加队列大小
queue = asyncio.Queue(maxsize=16)

# 设置超时
batch = await asyncio.wait_for(
    get_batch(),
    timeout=30.0
)
```

---

## 📚 学习路径

### 第1天：理解架构
- ✓ 运行 `python quick_start.py`
- ✓ 阅读 `README.md`
- ✓ 查看 `PROJECT_SUMMARY.md`

### 第2天：深入技术
- ✓ 阅读 `TECHNICAL.md`
- ✓ 运行 `python test.py`
- ✓ 研究核心模块代码

### 第3天：实践应用
- ✓ 修改 `config.py`
- ✓ 运行 `python examples.py`
- ✓ 自己写小 demo

### 第4-5天：优化和扩展
- ✓ 阅读 `INSTALL.md`
- ✓ 设置开发环境
- ✓ 根据需要修改代码

---

## 🎯 项目核心概念

### 异步管道
```
推理 Worker         训练 GPU
  │                  │
  ├─ 生成批次1 ──→ 训练批次1
  │                  │
  └─ 生成批次2 ──→ 训练批次2
  
特点：并行执行，提高GPU利用率
```

### 前缀缓存
```
提示词: "Explain RLHF algorithm..."
        |------ 缓存 ------|---- 新 ---|

相同前缀 → 复用缓存 → 减少计算
```

### 分布式训练
```
GPU0        GPU1        GPU2        GPU3
├─ Model0 ─ Model1 ─ Model2 ─ Model3
│   同步梯度 (AllReduce)
└─ 更新权重
```

---

## 💼 项目定位

**简洁版本**：
> "分布式 DPO 训练优化系统"

**完整版本**：
> "我实现了一个 Distributed DPO Training Optimization System，
> 重点优化 rollout-training overlap 和 prefix cache reuse，
> 在 AI 基础设施领域展现了系统优化能力。"

---

## 🔗 重要资源

### 内部文档
- [README.md](README.md) - 项目概览
- [TECHNICAL.md](TECHNICAL.md) - 技术原理
- [INSTALL.md](INSTALL.md) - 安装配置
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 完整总结

### 代码文件
- [main.py](main.py) - 主入口
- [quick_start.py](quick_start.py) - 快速演示
- [examples.py](examples.py) - 使用示例
- [test.py](test.py) - 单元测试

### 外部资源
- [PyTorch 文档](https://pytorch.org/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [TRL GitHub](https://github.com/huggingface/trl)
- [Transformers](https://github.com/huggingface/transformers)

---

## ✅ 检查清单

在开始前，确保：

- [ ] 已验证项目结构（运行 `verify_project.py`）
- [ ] 已查看项目文档（阅读 `README.md`）
- [ ] 已运行快速演示（执行 `quick_start.py`）
- [ ] 已理解核心概念（查看 `TECHNICAL.md`）
- [ ] 已安装依赖（运行 `pip install -r requirements.txt`）

---

## 🎓 学习目标

完成这个项目后，你将理解：

- ✅ 分布式深度学习训练
- ✅ Python 异步编程
- ✅ 系统性能优化
- ✅ GPU 内存管理
- ✅ 工程代码组织

---

## 📞 获取帮助

1. **查看文档**: 大多数问题都能在 README.md 和 TECHNICAL.md 中找到答案
2. **运行示例**: examples.py 包含完整的使用示例
3. **检查代码**: 每个模块都有详细的注释
4. **运行测试**: test.py 可以验证各个模块的功能

---

## 🎉 最后

**准备好开始了吗?**

```bash
python quick_start.py     # 第一步：快速演示
python verify_project.py  # 验证项目完整性
cat README.md            # 阅读项目文档
```

祝你学习愉快！🚀

---

**最后更新**: 2026-05-27
**项目状态**: ✅ 完成
