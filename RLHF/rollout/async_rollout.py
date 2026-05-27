import asyncio
from rollout.rollout_worker import RolloutWorker
from scheduler.cache_scheduler import CacheAwareScheduler


class AsyncRolloutManager:
    """
    异步推理管理器
    
    核心设计：
    - Producer-Consumer 模式
    - 多 Worker 并行推理
    - Prefix-aware 调度
    
    优化：
    - 推理和训练异步执行
    - 前缀缓存复用
    - Worker 负载均衡
    """
    
    def __init__(self, model_name, num_workers=2):
        """
        初始化异步推理管理器
        
        Args:
            model_name: 模型名称
            num_workers: 推理 Worker 数量
        """
        self.queue = asyncio.Queue(maxsize=8)
        
        # 创建多个推理 Worker
        self.workers = [
            RolloutWorker(model_name)
            for _ in range(num_workers)
        ]
        
        # 前缀感知调度器
        self.scheduler = CacheAwareScheduler()
        
        self.worker_id = 0

    async def producer(self, num_iterations=10):
        """
        生产者：生成推理任务
        
        核心优化：
        1. 将相同前缀的提示词分组
        2. 发送到同一 Worker 以复用缓存
        3. 异步队列管理
        
        Args:
            num_iterations: 生产迭代次数
        """
        for iteration in range(num_iterations):
            # 这里可以从数据集中加载实际的提示词
            prompts = [
                "Explain RLHF in machine learning",
                "Explain RLHF in machine learning",
                "Explain DPO in detail",
                "Explain DPO in detail",
                "What is reinforcement learning?",
                "What is reinforcement learning?"
            ]
            
            # 使用调度器对提示词进行分组 (前缀感知)
            grouped = self.scheduler.group_prompts(prompts)
            
            for prefix_hash, batch_prompts in grouped.items():
                # 将批次分配到 Worker
                worker = self.workers[self.worker_id]
                
                # 生成响应
                responses = worker.generate(batch_prompts)
                
                # 创建训练批次
                batch = {
                    "prompt": batch_prompts,
                    "chosen": responses,
                    "rejected": responses[::-1]  # 反序作为拒绝响应
                }
                
                # 将批次放入队列
                await self.queue.put(batch)
                
                # 轮转 Worker
                self.worker_id = (self.worker_id + 1) % len(self.workers)

    async def get_batch(self):
        """
        获取下一个训练批次
        
        Returns:
            包含 prompt、chosen、rejected 的批次字典
        """
        return await self.queue.get()
    
    async def start_producer(self, num_iterations=10):
        """启动生产者"""
        return asyncio.create_task(self.producer(num_iterations))
