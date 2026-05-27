# 安装和配置指南

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (CUDA 11.8+)
  - 推荐: V100, A100, H100, RTX 4090
  - 最低显存: 16GB (单GPU)
  - 多GPU: 建议 4x 16GB 或更高
  
- **CPU**: 8核+
- **内存**: 32GB+
- **存储**: 100GB+ (用于模型和数据)

### 软件要求
- Python 3.10+
- CUDA 11.8 or 12.1+
- cuDNN 8.6+

## 环境设置

### 步骤 1: 创建 Python 虚拟环境

```bash
# 使用 conda (推荐)
conda create -n rltrainflow python=3.11
conda activate rltrainflow

# 或使用 venv
python3.11 -m venv rltrainflow_env
source rltrainflow_env/bin/activate  # Linux/Mac
# 或 rltrainflow_env\Scripts\activate  # Windows
```

### 步骤 2: 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# PyTorch (CUDA 12.1 示例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 如果需要特定版本
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### 步骤 3: 验证安装

```bash
# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查 CUDA
nvidia-smi

# 检查项目导入
python -c "from trainer.dpo_trainer import DistributedDPOTrainer; print('✓ Project setup OK')"
```

## 快速开始

### 选项 A: 运行演示（无需模型）

```bash
# 运行快速演示
python quick_start.py
```

**输出示例**：
```
[2026-05-27 10:30:00] [quick_start] INFO ======================================
[2026-05-27 10:30:00] [quick_start] INFO RLTrain-Flow - Quick Start Guide
[2026-05-27 10:30:00] [quick_start] INFO ======================================
...
✓ All demos completed successfully!
```

### 选项 B: 运行单元测试

```bash
# 运行测试
python test.py

# 或使用 unittest
python -m unittest test.py -v
```

### 选项 C: 运行完整项目（需要模型）

```bash
# 单GPU
python main.py

# 多GPU (DDP)
torchrun --nproc_per_node=4 main.py

# 或使用 accelerate
accelerate launch --multi_gpu main.py
```

## 配置调整

### 修改 `config.py`

```python
from config import Config, TrainerConfig, RolloutConfig

config = Config(
    trainer=TrainerConfig(
        model_name="Qwen/Qwen2.5-1.5B",
        learning_rate=1e-4,
        batch_size=32
    ),
    rollout=RolloutConfig(
        num_workers=4,
        enable_prefix_caching=True,
        temperature=0.8
    )
)
```

### 常见配置参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `num_workers` | 2-4 | 推理 Worker 数量 |
| `batch_size` | 32-64 | 训练批次大小 |
| `learning_rate` | 1e-4 to 5e-5 | 学习率 |
| `max_tokens` | 128-256 | 最大生成长度 |
| `temperature` | 0.7-0.9 | 采样温度 |
| `prefix_len` | 32-64 | 前缀长度 |

## 故障排除

### 问题 1: CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：
```python
# 1. 减少 batch_size
batch_size = 16  # 从 32 减小

# 2. 启用梯度检查点
model.enable_gradient_checkpointing()

# 3. 使用混合精度训练
from torch.cuda.amp import autocast
with autocast():
    loss = model(...)
```

### 问题 2: vLLM 加载失败

```
ImportError: No module named 'vllm'
```

**解决方案**：
```bash
# 重新安装 vLLM
pip install vllm --no-cache-dir

# 或从源码安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 问题 3: 分布式训练报错

```
RuntimeError: NCCL error
```

**解决方案**：
```bash
# 检查 NCCL 版本
pip show nccl

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

# 重新运行
torchrun --nproc_per_node=4 main.py
```

### 问题 4: 异步任务卡住

```
asyncio.TimeoutError
```

**解决方案**：
```python
# 增加队列大小
self.queue = asyncio.Queue(maxsize=16)

# 或增加超时时间
batch = await asyncio.wait_for(
    self.rollout_manager.get_batch(),
    timeout=30.0
)
```

## 性能优化建议

### 1. GPU 利用率优化

```bash
# 监控 GPU 使用
watch -n 0.1 nvidia-smi

# 目标: GPU 使用率 > 90%
```

### 2. 内存优化

```python
# 启用梯度检查点
torch.utils.checkpoint.checkpoint(model, inputs)

# 启用混合精度
from torch.cuda.amp import GradScaler
scaler = GradScaler()
```

### 3. 数据加载优化

```python
# 增加数据加载线程
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 增加这个值
    pin_memory=True
)
```

## Docker 部署

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /workspace

# 安装 Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# 复制项目
COPY . /workspace

# 安装依赖
RUN pip install -r requirements.txt

# 运行
CMD ["python", "main.py"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t rltrainflow:latest .

# 运行容器
docker run --gpus all -v /path/to/data:/workspace/data rltrainflow:latest
```

## 监控和调试

### 启用日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 使用 Tensorboard

```bash
# 运行 Tensorboard
tensorboard --logdir=./logs

# 在浏览器打开
# http://localhost:6006
```

### 性能分析

```bash
# 使用 PyTorch Profiler
python -m cProfile -o stats.prof main.py

# 查看结果
python -m pstats stats.prof
```

## 常见问题 (FAQ)

**Q: 需要多少 GPU?**
A: 最少 1 个，建议 4 个或以上。

**Q: 支持 CPU 训练吗?**
A: 支持，但会非常慢。不建议用于生产。

**Q: 如何在多机上运行?**
A: 需要配置 SLURM 或 K8s，当前 MVP 主要针对单机多GPU。

**Q: 如何保存模型?**
A: 使用 `trainer.save_checkpoint(path)`.

**Q: 支持什么模型?**
A: 任何 Hugging Face 支持的模型都行。

## 获取帮助

- 📖 查看 [README.md](README.md)
- 🔬 查看 [TECHNICAL.md](TECHNICAL.md)
- 💻 查看 [examples.py](examples.py)
- 🧪 运行 [quick_start.py](quick_start.py)

---

**最后更新**: 2026-05-27
