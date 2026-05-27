import logging
import sys
from datetime import datetime


def build_logger(name="rltrainflow"):
    """
    构建日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger


def build_file_logger(name="rltrainflow", log_file="logs/training.log"):
    """
    构建带文件输出的日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
