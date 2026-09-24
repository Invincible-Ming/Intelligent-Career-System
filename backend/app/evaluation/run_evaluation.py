"""
RAG 评测主脚本。

用法：
    python -m app.evaluation.run_evaluation
"""

import asyncio
import logging

from app.evaluation.config import (
    EvaluationConfig,
    ExperimentConfig,
    ReflectionConfig,
    SearchConfig,
)
from app.evaluation.runner import EvaluationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


async def main():
    """运行 RAG 评测实验。"""

    # 定义多个测试配置
    baseline_config = EvaluationConfig(
        name="baseline",
        search=SearchConfig(
            top_k=5,
            search_mode="hybrid",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=True,
            reranker_top_k=5,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    # 变体 1：更大的 Top-K
    variant_top_k_10 = EvaluationConfig(
        name="top_k_10",
        search=SearchConfig(
            top_k=10,
            search_mode="hybrid",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=True,
            reranker_top_k=5,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    # 变体 2：Dense Only
    variant_dense_only = EvaluationConfig(
        name="dense_only",
        search=SearchConfig(
            top_k=5,
            search_mode="dense",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=False,
            reranker_top_k=5,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    # 变体 3：BM25 Only
    variant_bm25_only = EvaluationConfig(
        name="bm25_only",
        search=SearchConfig(
            top_k=5,
            search_mode="bm25",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=False,
            reranker_top_k=5,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    # 变体 4：更小的 Chunk Size
    variant_chunk_400 = EvaluationConfig(
        name="chunk_400",
        search=SearchConfig(
            top_k=5,
            search_mode="hybrid",
            chunk_size=400,
            chunk_overlap=50,
            enable_reranker=True,
            reranker_top_k=5,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    # 变体 5：启用 Reflection
    variant_with_reflection = EvaluationConfig(
        name="with_reflection",
        search=SearchConfig(
            top_k=5,
            search_mode="hybrid",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=True,
            reranker_top_k=5,
        ),
        reflection=ReflectionConfig(
            enable_reflection=True,
            max_iterations=2,
        ),
    )

    # 变体 6：禁用 Reranker
    variant_no_reranker = EvaluationConfig(
        name="no_reranker",
        search=SearchConfig(
            top_k=5,
            search_mode="hybrid",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=False,
            reranker_top_k=5,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    # 创建实验配置
    experiment = ExperimentConfig(
        experiment_name="rag_ab_test_v1",
        baseline=baseline_config,
        variants=[
            variant_top_k_10,
            variant_dense_only,
            variant_bm25_only,
            variant_chunk_400,
            variant_with_reflection,
            variant_no_reranker,
        ],
        test_dataset_path="app/evaluation/test_dataset_30.json",
        max_concurrency=5,
        timeout_seconds=120,
    )

    # 运行评测
    runner = EvaluationRunner(experiment)

    try:
        report_data = await runner.run()

        # 生成 HTML 报告
        report_path = await runner.generate_html_report()

        print("\n" + "=" * 80)
        print("✅ 评测完成！")
        print(f"📊 HTML 报告：{report_path}")
        print("=" * 80 + "\n")

        # 打印简要结果
        print("📈 评测结果摘要：\n")

        for config in report_data["configs"]:
            print(f"配置：{config['name']}")
            print(f"  成功率：{config['summary']['success']}/{config['summary']['total']}")
            print(f"  耗时：{config['summary']['duration']} 秒")
            print(f"  综合得分：{config['metrics'].get('overall_score', 0):.3f}")
            print()

    except Exception as exc:
        print(f"\n❌ 评测失败：{exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
