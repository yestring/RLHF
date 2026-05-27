import asyncio
import time
from utils.logger import build_logger


class AsyncPipeline:
    """
    异步推理-训练管道
    
    核心设计：
    - Async Producer-Consumer Pattern
    - Rollout 和 Training 并行执行
    - 双缓冲优化
    
    执行流程：
    generate batch i+1  (async)
            ||
      train batch i    (async)
    
    性能优化：
    - I/O 和计算异步执行
    - GPU 利用率最大化
    - 减少等待时间
    """
    
    def __init__(self, trainer, rollout_manager, log_interval=10):
        """
        初始化异步管道
        
        Args:
            trainer: DPO 训练器
            rollout_manager: 异步推理管理器
            log_interval: 日志输出间隔
        """
        self.trainer = trainer
        self.rollout_manager = rollout_manager
        self.log_interval = log_interval
        self.logger = build_logger("AsyncPipeline")
        
        self.step_count = 0
        self.total_loss = 0.0
    
    async def train_loop(self, max_steps=100):
        """
        训练循环
        
        Args:
            max_steps: 最大训练步数
        """
        self.logger.info(f"Starting training loop (max_steps={max_steps})")
        
        for step in range(max_steps):
            # 异步获取推理批次
            batch = await self.rollout_manager.get_batch()
            
            # 训练步骤
            start_time = time.time()
            metrics = self.trainer.train_step(batch)
            train_time = time.time() - start_time
            
            # 更新统计
            self.step_count += 1
            loss = metrics.get("loss", 0.0)
            self.total_loss += loss
            avg_loss = self.total_loss / self.step_count
            
            # 定期输出日志
            if step % self.log_interval == 0:
                self.logger.info(
                    f"Step {step}/{max_steps} | "
                    f"loss={loss:.4f} | "
                    f"avg_loss={avg_loss:.4f} | "
                    f"train_time={train_time:.3f}s | "
                    f"batch_size={metrics.get('batch_size', 0)}"
                )
    
    async def run(self, num_rollout_iterations=100, max_training_steps=100):
        """
        运行完整的管道
        
        并行执行：
        1. Producer: 推理任务生成
        2. Consumer: 模型训练
        
        Args:
            num_rollout_iterations: 推理迭代次数
            max_training_steps: 最大训练步数
        """
        self.logger.info("Starting AsyncPipeline")
        self.logger.info(f"Rollout iterations: {num_rollout_iterations}")
        self.logger.info(f"Max training steps: {max_training_steps}")
        
        # 启动生产者任务
        producer_task = asyncio.create_task(
            self.rollout_manager.producer(num_rollout_iterations)
        )
        
        # 启动训练任务
        trainer_task = asyncio.create_task(
            self.train_loop(max_training_steps)
        )
        
        try:
            # 并发运行两个任务
            await asyncio.gather(
                producer_task,
                trainer_task
            )
        except asyncio.CancelledError:
            self.logger.info("Pipeline cancelled")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise
        
        self.logger.info(f"Pipeline completed. Total steps: {self.step_count}")
        self.logger.info(f"Average loss: {self.total_loss / max(self.step_count, 1):.4f}")
