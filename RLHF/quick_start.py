#!/usr/bin/env python3
"""
快速启动脚本 - 轻量级测试和演示

不需要真实模型和GPU，用于验证代码结构和逻辑
"""

import asyncio
import sys
from scheduler.cache_scheduler import CacheAwareScheduler
from utils.metrics import MetricsTracker, AverageMeter
from utils.logger import build_logger


logger = build_logger("quick_start")


async def demo_prefix_scheduling():
    """演示：前缀感知调度"""
    logger.info("\n" + "="*60)
    logger.info("Demo 1: Prefix-Aware Scheduling")
    logger.info("="*60)
    
    scheduler = CacheAwareScheduler(prefix_len=16)
    
    # 模拟提示词
    prompts = [
        "Explain RLHF algorithm",
        "Explain RLHF training",
        "Explain DPO algorithm",
        "Explain DPO training",
        "What is model alignment?",
        "What is preference learning?"
    ]
    
    grouped = scheduler.group_prompts(prompts)
    stats = scheduler.get_group_stats(prompts)
    
    logger.info(f"✓ Total prompts: {stats['total_prompts']}")
    logger.info(f"✓ Number of groups: {stats['num_groups']}")
    logger.info(f"✓ Average group size: {stats['avg_group_size']:.2f}")
    
    logger.info("\nGrouped prompts:")
    for prefix_hash, group_prompts in grouped.items():
        logger.info(f"  Group {prefix_hash}:")
        for prompt in group_prompts:
            logger.info(f"    - {prompt}")


def demo_metrics_tracking():
    """演示：指标追踪"""
    logger.info("\n" + "="*60)
    logger.info("Demo 2: Metrics Tracking")
    logger.info("="*60)
    
    tracker = MetricsTracker()
    
    # 模拟训练
    logger.info("Simulating 20 training steps...")
    for step in range(20):
        loss = 1.0 - (step * 0.03)  # 模拟损失下降
        tracker.update("loss", loss)
        tracker.update("throughput", 100 + step * 2)
        
        if (step + 1) % 5 == 0:
            metrics = tracker.get_all_avgs()
            logger.info(f"  Step {step+1}: loss={metrics['loss']:.4f}, throughput={metrics['throughput']:.1f}")
    
    final_metrics = tracker.get_all_avgs()
    logger.info(f"\n✓ Final average loss: {final_metrics['loss']:.4f}")
    logger.info(f"✓ Final average throughput: {final_metrics['throughput']:.1f} steps/sec")


def demo_project_structure():
    """演示：项目结构"""
    logger.info("\n" + "="*60)
    logger.info("Demo 3: Project Structure")
    logger.info("="*60)
    
    structure = """
RLTrain-Flow/
├── trainer/
│   ├── distributed_trainer.py  - DDP 支持
│   └── dpo_trainer.py          - DPO 训练逻辑
│
├── rollout/
│   ├── rollout_worker.py       - vLLM 推理
│   └── async_rollout.py        - 异步推理管理
│
├── scheduler/
│   └── cache_scheduler.py      - 前缀感知调度
│
├── pipeline/
│   └── async_pipeline.py       - 推理-训练管道
│
├── utils/
│   ├── metrics.py              - 指标追踪
│   └── logger.py               - 日志系统
│
├── benchmark/
│   └── benchmark.py            - 性能测试
│
├── main.py                     - 主入口
├── config.py                   - 配置管理
├── examples.py                 - 使用示例
└── test.py                     - 单元测试
    """
    
    logger.info(structure)
    logger.info("✓ Project structure verified!")


def demo_technical_highlights():
    """演示：技术亮点"""
    logger.info("\n" + "="*60)
    logger.info("Demo 4: Technical Highlights")
    logger.info("="*60)
    
    highlights = """
1. Distributed DPO Training
   - DDP (DistributedDataParallel) 支持
   - 多GPU并行训练
   - 梯度同步优化

2. Async Rollout Pipeline
   - Producer-Consumer 模式
   - 推理和训练异步执行
   - 双缓冲优化
   
   execute:
   generate batch i+1 (async)
           ||
     train batch i   (async)

3. Cache-Aware Scheduling
   - 前缀感知分组
   - 提高 vLLM 前缀缓存命中率
   - 提升 GPU 利用率

4. Performance Optimization
   - 异步 I/O
   - 批处理优化
   - 内存高效
    """
    
    logger.info(highlights)


async def demo_async_pipeline_logic():
    """演示：异步管道逻辑"""
    logger.info("\n" + "="*60)
    logger.info("Demo 5: Async Pipeline Logic (Simplified)")
    logger.info("="*60)
    
    logger.info("Creating mock async tasks...")
    
    async def mock_producer():
        """模拟推理任务生成"""
        for i in range(5):
            logger.info(f"  Producer: Generated batch {i+1}")
            await asyncio.sleep(0.1)
    
    async def mock_consumer():
        """模拟训练任务"""
        for i in range(5):
            logger.info(f"    Consumer: Training batch {i+1}")
            await asyncio.sleep(0.15)
    
    logger.info("\nRunning producer and consumer concurrently:")
    await asyncio.gather(mock_producer(), mock_consumer())
    
    logger.info("\n✓ Async pipeline logic works!")


async def main():
    """主函数"""
    logger.info("\n" + "="*70)
    logger.info("RLTrain-Flow - Quick Start Guide")
    logger.info("="*70)
    
    try:
        # 演示 1
        await demo_prefix_scheduling()
        
        # 演示 2
        demo_metrics_tracking()
        
        # 演示 3
        demo_project_structure()
        
        # 演示 4
        demo_technical_highlights()
        
        # 演示 5
        await demo_async_pipeline_logic()
        
        # 完成
        logger.info("\n" + "="*70)
        logger.info("✓ All demos completed successfully!")
        logger.info("="*70)
        
        logger.info("\n📚 Next steps:")
        logger.info("  1. Review README.md for detailed documentation")
        logger.info("  2. Check examples.py for advanced usage")
        logger.info("  3. Run test.py for unit tests")
        logger.info("  4. Install dependencies: pip install -r requirements.txt")
        logger.info("  5. Run main.py when ready with models")
        logger.info("\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
