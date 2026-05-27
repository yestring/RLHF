from collections import defaultdict


class CacheAwareScheduler:
    """
    前缀感知调度器
    
    核心优化：
    - 将相同前缀的提示词分组
    - 提高 vLLM prefix cache hit rate
    - 提升 GPU 利用率和推理吞吐量
    
    设计思路：
    - hash(prefix) -> prompts_list
    - 同一 prefix 发送到同一 Worker
    - Worker 复用缓存中的 KV 状态
    """
    
    def __init__(self, prefix_len=32):
        """
        初始化调度器
        
        Args:
            prefix_len: 用于哈希的前缀长度
        """
        self.prefix_len = prefix_len
    
    def hash_prefix(self, prompt, prefix_len=None):
        """
        计算提示词的前缀哈希
        
        Args:
            prompt: 提示词文本
            prefix_len: 前缀长度 (默认使用初始化时的值)
            
        Returns:
            前缀哈希值
        """
        if prefix_len is None:
            prefix_len = self.prefix_len
        
        prefix = prompt[:prefix_len]
        return hash(prefix)
    
    def group_prompts(self, prompts):
        """
        将提示词按前缀分组
        
        Args:
            prompts: 提示词列表
            
        Returns:
            {前缀哈希: [提示词列表]}
        """
        grouped = defaultdict(list)
        
        for prompt in prompts:
            h = self.hash_prefix(prompt)
            grouped[h].append(prompt)
        
        return grouped
    
    def get_group_stats(self, prompts):
        """
        获取分组统计信息
        
        Args:
            prompts: 提示词列表
            
        Returns:
            分组统计字典
        """
        grouped = self.group_prompts(prompts)
        
        stats = {
            "total_prompts": len(prompts),
            "num_groups": len(grouped),
            "group_sizes": [len(group) for group in grouped.values()],
            "avg_group_size": sum(len(group) for group in grouped.values()) / max(len(grouped), 1)
        }
        
        return stats
