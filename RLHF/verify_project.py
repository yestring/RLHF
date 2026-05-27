#!/usr/bin/env python3
"""
项目验证脚本 - 检查所有文件和模块是否正确创建

运行: python verify_project.py
"""

import os
import sys
import importlib.util


class ProjectVerifier:
    """项目验证器"""
    
    def __init__(self, root_path):
        self.root_path = root_path
        self.results = {
            "passed": [],
            "failed": []
        }
    
    def check_file_exists(self, file_path, description=""):
        """检查文件是否存在"""
        full_path = os.path.join(self.root_path, file_path)
        exists = os.path.exists(full_path)
        
        status = "✓" if exists else "✗"
        message = f"{status} {file_path}"
        if description:
            message += f" ({description})"
        
        if exists:
            self.results["passed"].append(message)
        else:
            self.results["failed"].append(message)
        
        print(message)
        return exists
    
    def check_module_importable(self, module_name, description=""):
        """检查模块是否可导入"""
        try:
            # 添加根路径到 sys.path
            if self.root_path not in sys.path:
                sys.path.insert(0, self.root_path)
            
            importlib.import_module(module_name)
            status = "✓"
            success = True
        except Exception as e:
            status = "✗"
            success = False
            message = f"{status} {module_name} ({str(e)[:50]})"
        
        if success:
            message = f"{status} {module_name}"
            if description:
                message += f" ({description})"
            self.results["passed"].append(message)
        
        print(message)
        return success
    
    def verify_all(self):
        """验证所有文件和模块"""
        print("\n" + "="*70)
        print("RLTrain-Flow 项目验证")
        print("="*70 + "\n")
        
        # 1. 检查核心文件
        print("【1】核心文件检查")
        print("-" * 70)
        self.check_file_exists("main.py", "主入口")
        self.check_file_exists("config.py", "配置管理")
        self.check_file_exists("requirements.txt", "依赖列表")
        self.check_file_exists(".gitignore", "Git 忽略规则")
        
        print("\n【2】文档检查")
        print("-" * 70)
        self.check_file_exists("README.md", "项目文档")
        self.check_file_exists("TECHNICAL.md", "技术文档")
        self.check_file_exists("INSTALL.md", "安装指南")
        self.check_file_exists("PROJECT_SUMMARY.md", "项目总结")
        
        print("\n【3】示例和测试")
        print("-" * 70)
        self.check_file_exists("quick_start.py", "快速演示")
        self.check_file_exists("examples.py", "使用示例")
        self.check_file_exists("test.py", "单元测试")
        
        print("\n【4】Trainer 模块")
        print("-" * 70)
        self.check_file_exists("trainer/__init__.py")
        self.check_file_exists("trainer/distributed_trainer.py", "DDP 支持")
        self.check_file_exists("trainer/dpo_trainer.py", "DPO 训练")
        
        print("\n【5】Rollout 模块")
        print("-" * 70)
        self.check_file_exists("rollout/__init__.py")
        self.check_file_exists("rollout/rollout_worker.py", "vLLM Worker")
        self.check_file_exists("rollout/async_rollout.py", "异步推理")
        
        print("\n【6】Scheduler 模块")
        print("-" * 70)
        self.check_file_exists("scheduler/__init__.py")
        self.check_file_exists("scheduler/cache_scheduler.py", "前缀调度")
        
        print("\n【7】Pipeline 模块")
        print("-" * 70)
        self.check_file_exists("pipeline/__init__.py")
        self.check_file_exists("pipeline/async_pipeline.py", "异步管道")
        
        print("\n【8】Utils 模块")
        print("-" * 70)
        self.check_file_exists("utils/__init__.py")
        self.check_file_exists("utils/metrics.py", "指标追踪")
        self.check_file_exists("utils/logger.py", "日志系统")
        
        print("\n【9】Benchmark 模块")
        print("-" * 70)
        self.check_file_exists("benchmark/__init__.py")
        self.check_file_exists("benchmark/benchmark.py", "性能测试")
        
        print("\n【10】模块导入检查")
        print("-" * 70)
        self.check_module_importable("config", "配置模块")
        self.check_module_importable("trainer.distributed_trainer", "分布式训练基类")
        self.check_module_importable("trainer.dpo_trainer", "DPO 训练器")
        self.check_module_importable("rollout.rollout_worker", "推理 Worker")
        self.check_module_importable("rollout.async_rollout", "异步推理")
        self.check_module_importable("scheduler.cache_scheduler", "缓存调度器")
        self.check_module_importable("pipeline.async_pipeline", "异步管道")
        self.check_module_importable("utils.metrics", "指标模块")
        self.check_module_importable("utils.logger", "日志模块")
        self.check_module_importable("benchmark.benchmark", "基准测试")
        
        # 生成报告
        self.print_report()
    
    def print_report(self):
        """打印验证报告"""
        print("\n" + "="*70)
        print("验证结果总结")
        print("="*70 + "\n")
        
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        total = passed + failed
        
        print(f"✓ 通过: {passed}/{total}")
        print(f"✗ 失败: {failed}/{total}")
        
        if failed > 0:
            print("\n❌ 失败的项目：")
            for item in self.results["failed"]:
                print(f"  {item}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\n✅ 成功率: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n" + "="*70)
            print("🎉 项目验证通过！所有文件和模块都已正确创建。")
            print("="*70)
            print("\n📚 后续步骤：")
            print("  1. 阅读 README.md 了解项目概览")
            print("  2. 运行 python quick_start.py 查看演示")
            print("  3. 查看 TECHNICAL.md 理解技术原理")
            print("  4. 运行 python test.py 执行单元测试")
            print("  5. 按需修改配置和代码")
            print("\n💡 快速命令：")
            print("  python quick_start.py       # 快速演示")
            print("  python test.py              # 运行测试")
            print("  python examples.py          # 查看示例")
            print("  python main.py              # 完整项目 (需要模型)")
            print("\n")
            return True
        else:
            print("\n" + "="*70)
            print("⚠️  项目验证失败！请检查上述失败的项目。")
            print("="*70 + "\n")
            return False


def main():
    """主函数"""
    # 获取项目根路径
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    # 创建验证器并运行
    verifier = ProjectVerifier(root_path)
    success = verifier.verify_all()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
