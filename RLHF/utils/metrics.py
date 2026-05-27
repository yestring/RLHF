class AverageMeter:
    """
    平均值计量器
    
    用于追踪训练过程中的各类指标：
    - 损失函数
    - 准确率
    - 其他自定义指标
    """
    
    def __init__(self, name=""):
        self.name = name
        self.total = 0
        self.count = 0
    
    def reset(self):
        """重置计量器"""
        self.total = 0
        self.count = 0
    
    def update(self, value, n=1):
        """
        更新计量器
        
        Args:
            value: 要添加的值
            n: 样本数 (默认为1)
        """
        self.total += value * n
        self.count += n
    
    @property
    def avg(self):
        """获取平均值"""
        return self.total / max(self.count, 1)
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """
    指标追踪器
    
    集中管理多个指标
    """
    
    def __init__(self):
        self.metrics = {}
    
    def add_meter(self, name, meter=None):
        """
        添加指标计量器
        
        Args:
            name: 指标名称
            meter: AverageMeter 实例 (默认创建新实例)
        """
        if meter is None:
            meter = AverageMeter(name)
        self.metrics[name] = meter
    
    def update(self, name, value, n=1):
        """
        更新指标
        
        Args:
            name: 指标名称
            value: 指标值
            n: 样本数
        """
        if name not in self.metrics:
            self.add_meter(name)
        self.metrics[name].update(value, n)
    
    def get_avg(self, name):
        """获取指标平均值"""
        if name in self.metrics:
            return self.metrics[name].avg
        return None
    
    def get_all_avgs(self):
        """获取所有指标的平均值"""
        return {name: meter.avg for name, meter in self.metrics.items()}
    
    def reset(self):
        """重置所有计量器"""
        for meter in self.metrics.values():
            meter.reset()
    
    def __str__(self):
        return " | ".join(str(meter) for meter in self.metrics.values())
