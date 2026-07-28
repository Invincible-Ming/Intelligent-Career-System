"""
评测系统安装验证脚本。
检查所有必要的文件和模块是否正确安装。
"""

import sys
from pathlib import Path


def check_files():
    """检查评测系统文件完整性。"""
    
    print("🔍 检查评测系统文件...")
    
    required_files = [
        "app/evaluation/__init__.py",
        "app/evaluation/config.py",
        "app/evaluation/dataset.py",
        "app/evaluation/metrics.py",
        "app/evaluation/executor.py",
        "app/evaluation/runner.py",
        "app/evaluation/report.py",
        "app/evaluation/api.py",
        "app/evaluation/cli.py",
        "app/evaluation/health_check.py",
        "app/evaluation/run_evaluation.py",
        "app/evaluation/test_dataset_30.json",
        "app/evaluation/example_config.json",
        "app/evaluation/README.md",
    ]
    
    missing = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
            print(f"  ✗ {file_path}")
        else:
            print(f"  ✓ {file_path}")
    
    if missing:
        print(f"\n❌ 缺少 {len(missing)} 个文件")
        return False
    
    print(f"\n✅ 所有文件完整 ({len(required_files)} 个)")
    return True


def check_imports():
    """检查模块导入。"""
    
    print("\n🔍 检查模块导入...")
    
    modules = [
        ("app.evaluation.config", "EvaluationConfig"),
        ("app.evaluation.dataset", "TestDataset"),
        ("app.evaluation.metrics", "ragas_metrics"),
        ("app.evaluation.executor", "rag_executor"),
        ("app.evaluation.runner", "EvaluationRunner"),
        ("app.evaluation.report", "ReportGenerator"),
    ]
    
    failed = []
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as exc:
            failed.append((module_name, class_name, exc))
            print(f"  ✗ {module_name}.{class_name} - {exc}")
    
    if failed:
        print(f"\n❌ {len(failed)} 个模块导入失败")
        return False
    
    print(f"\n✅ 所有模块导入成功 ({len(modules)} 个)")
    return True


def check_dataset():
    """检查测试数据集。"""
    
    print("\n🔍 检查测试数据集...")
    
    try:
        from app.evaluation.dataset import TestDataset
        import json
        
        dataset_path = Path("app/evaluation/test_dataset_30.json")
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = TestDataset.model_validate(data)
        
        print(f"  ✓ 数据集名称: {dataset.name}")
        print(f"  ✓ 样本数量: {dataset.size}")
        
        if dataset.size < 30:
            print(f"  ⚠ 样本数少于 30（当前 {dataset.size}）")
        
        print("\n✅ 测试数据集格式正确")
        return True
        
    except Exception as exc:
        print(f"\n❌ 测试数据集检查失败: {exc}")
        return False


def check_dependencies():
    """检查必要的依赖包。"""
    
    print("\n🔍 检查依赖包...")
    
    required_packages = [
        "fastapi",
        "pydantic",
        "asyncio",
        "logging",
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package}")
    
    if missing:
        print(f"\n❌ 缺少依赖包: {', '.join(missing)}")
        return False
    
    print(f"\n✅ 所有依赖包已安装")
    return True


def main():
    """运行所有检查。"""
    
    print("=" * 80)
    print("RAG 评测系统安装验证")
    print("=" * 80 + "\n")
    
    all_ok = True
    
    # 1. 检查文件
    if not check_files():
        all_ok = False
    
    # 2. 检查模块导入
    if not check_imports():
        all_ok = False
    
    # 3. 检查数据集
    if not check_dataset():
        all_ok = False
    
    # 4. 检查依赖
    if not check_dependencies():
        all_ok = False
    
    print("\n" + "=" * 80)
    
    if all_ok:
        print("✅ 评测系统安装验证通过！")
        print("\n可以开始使用评测系统：")
        print("  python -m app.evaluation.cli quick")
        print("  python -m app.evaluation.cli full")
    else:
        print("❌ 评测系统安装验证失败，请检查上述问题")
        sys.exit(1)
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
