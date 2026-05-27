"""
配置文件模块

用于管理项目的全局配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerConfig:
    """训练器配置"""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    save_checkpoint_steps: int = 5000


@dataclass
class RolloutConfig:
    """推理配置"""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    num_workers: int = 2
    enable_prefix_caching: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 128
    queue_size: int = 8


@dataclass
class SchedulerConfig:
    """调度器配置"""
    prefix_len: int = 32
    enable_dynamic_grouping: bool = True


@dataclass
class PipelineConfig:
    """管道配置"""
    num_rollout_iterations: int = 100
    max_training_steps: int = 100
    log_interval: int = 10


@dataclass
class Config:
    """全局配置"""
    trainer: TrainerConfig = None
    rollout: RolloutConfig = None
    scheduler: SchedulerConfig = None
    pipeline: PipelineConfig = None
    
    def __post_init__(self):
        if self.trainer is None:
            self.trainer = TrainerConfig()
        if self.rollout is None:
            self.rollout = RolloutConfig()
        if self.scheduler is None:
            self.scheduler = SchedulerConfig()
        if self.pipeline is None:
            self.pipeline = PipelineConfig()


# 默认全局配置
DEFAULT_CONFIG = Config()
