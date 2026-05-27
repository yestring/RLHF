import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM


class DistributedTrainer:
    """
    分布式训练器基类
    
    支持：
    - DDP (多GPU训练)
    - 模型加载和保存
    - 分布式初始化
    """
    
    def __init__(self, model_name):
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # 初始化分布式训练
        if self.world_size > 1:
            dist.init_process_group("nccl")
            torch.cuda.set_device(self.rank)

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        
        # 移至GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 如果是多GPU，使用 DDP 包装
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank]
            )

    def save_checkpoint(self, path):
        """保存模型检查点"""
        if self.rank == 0:  # 仅主进程保存
            self.model.save_pretrained(path)

    def load_checkpoint(self, path):
        """加载模型检查点"""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
