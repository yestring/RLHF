import asyncio

from rollout.async_rollout import AsyncRolloutManager
from trainer.dpo_trainer import DistributedDPOTrainer
from pipeline.async_pipeline import AsyncPipeline
from utils.logger import build_logger


logger = build_logger("RLTrain-Flow")


async def main():
    """
    RLTrain-Flow 主函数
    
    工作流程：
    1. 初始化 DPO 训练器
    2. 初始化异步推理管理器
    3. 创建异步管道
    4. 并行运行推理和训练
    """
    
    logger.info("="*50)
    logger.info("RLTrain-Flow: Distributed DPO Training System")
    logger.info("="*50)
    
    # 1. 初始化 DPO 训练器
    logger.info("Initializing DPO Trainer...")
    trainer = DistributedDPOTrainer(
        model_name="Qwen/Qwen2.5-1.5B"
    )
    logger.info("DPO Trainer initialized")
    
    # 2. 初始化异步推理管理器
    logger.info("Initializing Async Rollout Manager...")
    rollout_manager = AsyncRolloutManager(
        model_name="Qwen/Qwen2.5-1.5B",
        num_workers=2
    )
    logger.info("Async Rollout Manager initialized")
    
    # 3. 创建异步管道
    logger.info("Creating Async Pipeline...")
    pipeline = AsyncPipeline(
        trainer=trainer,
        rollout_manager=rollout_manager,
        log_interval=5
    )
    logger.info("Async Pipeline created")
    
    # 4. 运行管道
    logger.info("Starting pipeline execution...")
    logger.info("Rollout and training will run in parallel")
    
    try:
        await pipeline.run(
            num_rollout_iterations=20,
            max_training_steps=20
        )
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    
    logger.info("="*50)
    logger.info("RLTrain-Flow execution completed")
    logger.info("="*50)


if __name__ == "__main__":
    asyncio.run(main())
