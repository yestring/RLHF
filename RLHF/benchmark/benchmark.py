import time


class Benchmark:
    """
    基准测试工具
    
    测量项：
    - 吞吐量 (steps/sec)
    - 单步耗时
    - GPU 利用率 (可选)
    """
    
    def __init__(self, name="RLTrain-Flow"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.steps = 0
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def end(self):
        """结束计时"""
        self.end_time = time.time()
    
    def report(self):
        """生成基准报告"""
        if self.start_time is None or self.end_time is None:
            print("Benchmark not completed. Call start() and end() first.")
            return
        
        elapsed_time = self.end_time - self.start_time
        throughput = self.steps / elapsed_time if elapsed_time > 0 else 0
        avg_time_per_step = elapsed_time / self.steps if self.steps > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*50}")
        print(f"Total steps: {self.steps}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Throughput: {throughput:.2f} steps/s")
        print(f"Avg time per step: {avg_time_per_step:.4f}s")
        print(f"{'='*50}\n")


def benchmark_pipeline(pipeline, steps=20):
    """
    运行管道基准测试
    
    Args:
        pipeline: AsyncPipeline 实例
        steps: 测试步数
    """
    benchmark = Benchmark("AsyncPipeline")
    benchmark.start()
    
    # 这里可以集成到实际的管道运行中
    # 示例：在 pipeline 中记录步数
    
    benchmark.end()
    benchmark.steps = steps
    benchmark.report()


def benchmark_throughput(rollout_manager, num_batches=100):
    """
    测试推理吞吐量
    
    Args:
        rollout_manager: AsyncRolloutManager 实例
        num_batches: 测试批次数
    """
    import asyncio
    
    async def run():
        start = time.time()
        
        for _ in range(num_batches):
            batch = await rollout_manager.get_batch()
        
        end = time.time()
        elapsed = end - start
        throughput = num_batches / elapsed
        
        print(f"\nRollout Throughput Benchmark")
        print(f"Batches: {num_batches}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.2f} batches/s\n")
    
    asyncio.run(run())
