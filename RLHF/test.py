"""
测试模块
"""

import unittest
import asyncio
from scheduler.cache_scheduler import CacheAwareScheduler
from utils.metrics import AverageMeter, MetricsTracker


class TestCacheScheduler(unittest.TestCase):
    """测试前缀感知调度器"""
    
    def setUp(self):
        self.scheduler = CacheAwareScheduler(prefix_len=32)
    
    def test_hash_prefix(self):
        """测试前缀哈希"""
        prompt1 = "Explain RLHF algorithm"
        prompt2 = "Explain RLHF in detail"
        
        hash1 = self.scheduler.hash_prefix(prompt1)
        hash2 = self.scheduler.hash_prefix(prompt2)
        
        # 相同前缀应该有相同的哈希
        self.assertEqual(hash1, hash2)
    
    def test_group_prompts(self):
        """测试提示词分组"""
        prompts = [
            "Explain RLHF",
            "Explain RLHF in detail",
            "Explain DPO",
            "Explain DPO algorithm"
        ]
        
        grouped = self.scheduler.group_prompts(prompts)
        
        # 应该有多个组
        self.assertGreater(len(grouped), 0)
        
        # 所有提示词都应该被分组
        total_prompts = sum(len(group) for group in grouped.values())
        self.assertEqual(total_prompts, len(prompts))
    
    def test_group_stats(self):
        """测试分组统计"""
        prompts = ["Explain RLHF"] * 4 + ["Explain DPO"] * 2
        
        stats = self.scheduler.get_group_stats(prompts)
        
        self.assertEqual(stats['total_prompts'], 6)
        self.assertGreater(stats['num_groups'], 0)
        self.assertGreater(stats['avg_group_size'], 0)


class TestMetrics(unittest.TestCase):
    """测试指标模块"""
    
    def test_average_meter(self):
        """测试平均值计量器"""
        meter = AverageMeter("test_metric")
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            meter.update(v)
        
        expected_avg = sum(values) / len(values)
        self.assertAlmostEqual(meter.avg, expected_avg)
    
    def test_metrics_tracker(self):
        """测试指标追踪器"""
        tracker = MetricsTracker()
        tracker.add_meter("loss")
        tracker.add_meter("accuracy")
        
        for i in range(10):
            tracker.update("loss", 0.5)
            tracker.update("accuracy", 0.9)
        
        self.assertAlmostEqual(tracker.get_avg("loss"), 0.5)
        self.assertAlmostEqual(tracker.get_avg("accuracy"), 0.9)
    
    def test_metrics_reset(self):
        """测试指标重置"""
        tracker = MetricsTracker()
        tracker.add_meter("loss")
        
        tracker.update("loss", 0.5)
        self.assertAlmostEqual(tracker.get_avg("loss"), 0.5)
        
        tracker.reset()
        self.assertAlmostEqual(tracker.get_avg("loss"), 0.0)


class TestAsyncFunctions(unittest.TestCase):
    """测试异步函数"""
    
    def test_async_placeholder(self):
        """占位符异步测试"""
        async def dummy():
            return True
        
        result = asyncio.run(dummy())
        self.assertTrue(result)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
