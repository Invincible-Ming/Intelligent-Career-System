#!/usr/bin/env python
"""
RAG 评测命令行工具。

用法：
    # 快速评测（使用默认配置）
    python -m app.evaluation.cli quick

    # 完整评测（多配置对比）
    python -m app.evaluation.cli full

    # 自定义评测
    python -m app.evaluation.cli custom --config config.json
"""

import argparse
import asyncio
import sys

from app.evaluation.config import (
    EvaluationConfig,
    ExperimentConfig,
    ReflectionConfig,
    SearchConfig,
)
from app.evaluation.runner import EvaluationRunner


def create_quick_experiment() -> ExperimentConfig:
    """创建快速评测实验配置（仅 baseline）。"""

    baseline = EvaluationConfig(
        name="baseline",
        search=SearchConfig(
            top_k=5,
            search_mode="hybrid",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=True,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    return ExperimentConfig(
        experiment_name="quick_test",
        baseline=baseline,
        variants=[],
        test_dataset_path="app/evaluation/test_dataset_30.json",
        max_concurrency=5,
        timeout_seconds=120,
    )


def create_full_experiment() -> ExperimentConfig:
    """创建完整评测实验配置（多配置对比）。"""

    baseline = EvaluationConfig(
        name="baseline",
        search=SearchConfig(
            top_k=5,
            search_mode="hybrid",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=True,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    variants = [
        # Top-K 对比
        EvaluationConfig(
            name="top_k_10",
            search=SearchConfig(
                top_k=10,
                search_mode="hybrid",
                chunk_size=800,
                chunk_overlap=100,
                enable_reranker=True,
            ),
            reflection=ReflectionConfig(
                enable_reflection=False,
            ),
        ),
        # 检索模式对比
        EvaluationConfig(
            name="dense_only",
            search=SearchConfig(
                top_k=5,
                search_mode="dense",
                chunk_size=800,
                chunk_overlap=100,
                enable_reranker=False,
            ),
            reflection=ReflectionConfig(
                enable_reflection=False,
            ),
        ),
        EvaluationConfig(
            name="bm25_only",
            search=SearchConfig(
                top_k=5,
                search_mode="bm25",
                chunk_size=800,
                chunk_overlap=100,
                enable_reranker=False,
            ),
            reflection=ReflectionConfig(
                enable_reflection=False,
            ),
        ),
        # Chunk Size 对比
        EvaluationConfig(
            name="chunk_400",
            search=SearchConfig(
                top_k=5,
                search_mode="hybrid",
                chunk_size=400,
                chunk_overlap=50,
                enable_reranker=True,
            ),
            reflection=ReflectionConfig(
                enable_reflection=False,
            ),
        ),
        EvaluationConfig(
            name="chunk_1200",
            search=SearchConfig(
                top_k=5,
                search_mode="hybrid",
                chunk_size=1200,
                chunk_overlap=150,
                enable_reranker=True,
            ),
            reflection=ReflectionConfig(
                enable_reflection=False,
            ),
        ),
        # Reflection 对比
        EvaluationConfig(
            name="with_reflection",
            search=SearchConfig(
                top_k=5,
                search_mode="hybrid",
                chunk_size=800,
                chunk_overlap=100,
                enable_reranker=True,
            ),
            reflection=ReflectionConfig(
                enable_reflection=True,
                max_iterations=2,
            ),
        ),
    ]

    return ExperimentConfig(
        experiment_name="full_ab_test",
        baseline=baseline,
        variants=variants,
        test_dataset_path="app/evaluation/test_dataset_30.json",
        max_concurrency=5,
        timeout_seconds=120,
    )


async def run_evaluation(
    experiment_config: ExperimentConfig,
) -> None:
    """运行评测。"""

    print(f"\n{'=' * 80}")
    print(f"🚀 开始评测：{experiment_config.experiment_name}")
    print(f"{'=' * 80}\n")

    runner = EvaluationRunner(experiment_config)

    try:
        report_data = await runner.run()

        # 生成 HTML 报告
        report_path = await runner.generate_html_report()

        print(f"\n{'=' * 80}")
        print("✅ 评测完成！")
        print(f"📊 HTML 报告：{report_path}")
        print(f"{'=' * 80}\n")

        # 打印结果摘要
        print("📈 评测结果摘要：\n")

        for config in report_data["configs"]:
            print(f"【{config['name']}】")
            print(f"  ✓ 成功：{config['summary']['success']} / {config['summary']['total']}")
            print(f"  ✗ 失败：{config['summary']['failure']}")
            print(f"  ⏱ 耗时：{config['summary']['duration']:.2f} 秒")
            print(f"  📊 综合得分：{config['metrics'].get('overall_score', 0):.3f}")
            print()

    except Exception as exc:
        print(f"\n❌ 评测失败：{exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """命令行入口。"""

    parser = argparse.ArgumentParser(
        description="RAG 评测命令行工具"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="子命令",
    )

    # quick 命令
    subparsers.add_parser(
        "quick",
        help="快速评测（仅 baseline）",
    )

    # full 命令
    subparsers.add_parser(
        "full",
        help="完整评测（多配置对比）",
    )

    # custom 命令
    custom_parser = subparsers.add_parser(
        "custom",
        help="自定义评测",
    )
    custom_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="实验配置 JSON 文件路径",
    )

    args = parser.parse_args()

    if args.command == "quick":
        experiment = create_quick_experiment()
    elif args.command == "full":
        experiment = create_full_experiment()
    elif args.command == "custom":
        import json
        from pathlib import Path

        config_path = Path(args.config)

        if not config_path.exists():
            print(f"❌ 配置文件不存在：{args.config}")
            sys.exit(1)

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        experiment = ExperimentConfig.model_validate(
            config_data
        )
    else:
        parser.print_help()
        sys.exit(1)

    asyncio.run(run_evaluation(experiment))


if __name__ == "__main__":
    main()
