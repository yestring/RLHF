"""
高级示例：展示如何使用 RLTrain-Flow
"""

import asyncio
from trainer.dpo_trainer import DistributedDPOTrainer
from rollout.async_rollout import AsyncRolloutManager
from pipeline.async_pipeline import AsyncPipeline
from scheduler.cache_scheduler import CacheAwareScheduler
from utils.logger import build_logger


logger = build_logger("examples")


async def example_basic_pipeline():
    """
    示例 1: 基础异步管道
    
    展示如何：
    - 初始化各个组件
    - 运行完整的推理-训练管道
    - 监控性能指标
    """
    logger.info("="*60)
    logger.info("Example 1: Basic Async Pipeline")
    logger.info("="*60)
    
    # 初始化组件
    trainer = DistributedDPOTrainer(model_name="Qwen/Qwen2.5-1.5B")
    rollout_manager = AsyncRolloutManager(
        model_name="Qwen/Qwen2.5-1.5B",
        num_workers=2
    )
    pipeline = AsyncPipeline(trainer, rollout_manager, log_interval=5)
    
    # 运行管道
    await pipeline.run(
        num_rollout_iterations=20,
        max_training_steps=20
    )
    
    logger.info("Example 1 completed!")


async def example_prefix_scheduling():
    """
    示例 2: 前缀感知调度
    
    展示如何：
    - 使用前缀调度器分组提示词
    - 分析缓存性能
    """
    logger.info("="*60)
    logger.info("Example 2: Prefix-Aware Scheduling")
    logger.info("="*60)
    
    scheduler = CacheAwareScheduler(prefix_len=32)
    
    # 示例提示词
    prompts = [
        "Explain RLHF algorithm",
        "Explain RLHF in detail",
        "Explain DPO algorithm",
        "Explain DPO training process",
        "What is reward modeling?",
        "What is reward shaping?",
    ]
    
    # 分组
    grouped = scheduler.group_prompts(prompts)
    stats = scheduler.get_group_stats(prompts)
    
    logger.info(f"Total prompts: {stats['total_prompts']}")
    logger.info(f"Number of groups: {stats['num_groups']}")
    logger.info(f"Average group size: {stats['avg_group_size']:.2f}")
    
    # 显示分组详情
    for prefix_hash, group_prompts in grouped.items():
        logger.info(f"  Group {prefix_hash}: {len(group_prompts)} prompts")
    
    logger.info("Example 2 completed!")


async def example_custom_config():
    """
    示例 3: 自定义配置
    
    展示如何：
    - 使用配置对象
    - 自定义参数
    """
    logger.info("="*60)
    logger.info("Example 3: Custom Configuration")
    logger.info("="*60)
    
    from config import Config, TrainerConfig, RolloutConfig
    
    # 创建自定义配置
    config = Config(
        trainer=TrainerConfig(
            model_name="Qwen/Qwen2.5-1.5B",
            learning_rate=5e-5,
            batch_size=16
        ),
        rollout=RolloutConfig(
            num_workers=4,
            temperature=0.7,
            max_tokens=256
        )
    )
    
    logger.info(f"Trainer config: {config.trainer}")
    logger.info(f"Rollout config: {config.rollout}")
    
    logger.info("Example 3 completed!")


def example_metrics_tracking():
    """
    示例 4: 指标追踪
    
    展示如何：
    - 使用 AverageMeter 追踪指标
    - 使用 MetricsTracker 管理多个指标
    """
    logger.info("="*60)
    logger.info("Example 4: Metrics Tracking")
    logger.info("="*60)
    
    from utils.metrics import AverageMeter, MetricsTracker
    
    # 单个指标
    loss_meter = AverageMeter("Loss")
    for loss_value in [0.5, 0.4, 0.3, 0.25]:
        loss_meter.update(loss_value)
    logger.info(f"Average loss: {loss_meter.avg:.4f}")
    
    # 多个指标
    tracker = MetricsTracker()
    tracker.add_meter("loss")
    tracker.add_meter("accuracy")
    tracker.add_meter("throughput")
    
    # 模拟训练
    for step in range(10):
        tracker.update("loss", 0.5 - 0.01 * step)
        tracker.update("accuracy", 0.7 + 0.01 * step)
        tracker.update("throughput", 100 + step * 5)
    
    logger.info(f"Final metrics: {tracker.get_all_avgs()}")
    
    logger.info("Example 4 completed!")


def example_benchmark():
    """
    示例 5: 基准测试
    
    展示如何：
    - 运行性能基准测试
    - 记录性能指标
    """
    logger.info("="*60)
    logger.info("Example 5: Benchmarking")
    logger.info("="*60)
    
    from benchmark.benchmark import Benchmark
    import time
    
    benchmark = Benchmark(name="Training Loop")
    benchmark.start()
    
    # 模拟训练
    for i in range(50):
        time.sleep(0.01)  # 模拟计算
    
    benchmark.steps = 50
    benchmark.end()
    benchmark.report()
    
    logger.info("Example 5 completed!")


async def main():
    """运行所有示例"""
    logger.info("\n" + "="*60)
    logger.info("RLTrain-Flow Examples")
    logger.info("="*60 + "\n")
    
    # 示例 1: 基础管道
    # 注意：这个示例需要实际的模型和数据
    # await example_basic_pipeline()
    
    # 示例 2: 前缀调度
    await example_prefix_scheduling()
    
    # 示例 3: 配置
    await example_custom_config()
    
    # 示例 4: 指标追踪
    example_metrics_tracking()
    
    # 示例 5: 基准测试
    example_benchmark()
    
    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())
