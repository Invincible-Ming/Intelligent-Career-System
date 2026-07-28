"""
评测运行器，支持并发控制、超时处理、进度追踪。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evaluation.config import EvaluationConfig, ExperimentConfig
from app.evaluation.dataset import TestCase, TestDataset
from app.evaluation.executor import rag_executor
from app.evaluation.metrics import ragas_metrics
from app.evaluation.report import ReportGenerator

logger = logging.getLogger(__name__)


class EvaluationResult(dict):
    """单个测试用例的评测结果。"""

    pass


class ConfigResult:
    """单个配置的评测结果。"""

    def __init__(
        self,
        config: EvaluationConfig,
    ):
        self.config = config
        self.results: list[EvaluationResult] = []
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.success_count: int = 0
        self.failure_count: int = 0

    @property
    def total_count(self) -> int:
        """总测试数。"""
        return len(self.results)

    @property
    def duration(self) -> float:
        """总耗时（秒）。"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def avg_metrics(self) -> dict[str, float]:
        """平均指标。"""
        if not self.results:
            return {}

        metric_names = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "overall_score",
        ]

        avg = {}

        for name in metric_names:
            values = [
                result.get("metrics", {}).get(name, 0)
                for result in self.results
                if result.get("status") == "success"
            ]

            avg[name] = (
                sum(values) / len(values)
                if values
                else 0.0
            )

        return avg


class EvaluationRunner:
    """评测运行器。"""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
    ):
        self.experiment_config = experiment_config
        self.test_dataset: TestDataset | None = None
        self.config_results: dict[str, ConfigResult] = {}
        self.experiment_id = str(uuid.uuid4())[:8]

    async def run(self) -> dict[str, Any]:
        """
        运行完整的 A/B 实验。
        
        返回实验报告数据。
        """

        logger.info(
            "开始评测实验：%s",
            self.experiment_config.experiment_name,
        )

        # 加载测试数据集
        self.test_dataset = self._load_dataset(
            self.experiment_config.test_dataset_path
        )

        logger.info(
            "加载测试数据集：%d 条样本",
            self.test_dataset.size,
        )

        # 评测所有配置
        all_configs = [
            self.experiment_config.baseline,
            *self.experiment_config.variants,
        ]

        for config in all_configs:
            logger.info(
                "开始评测配置：%s",
                config.name,
            )

            config_result = await self._evaluate_config(config)
            self.config_results[config.name] = config_result

            logger.info(
                "配置 %s 评测完成：%d 成功，%d 失败，耗时 %.2f 秒",
                config.name,
                config_result.success_count,
                config_result.failure_count,
                config_result.duration,
            )

        # 生成报告
        report_data = self._build_report_data()

        logger.info(
            "评测实验完成：%s",
            self.experiment_config.experiment_name,
        )

        return report_data

    async def _evaluate_config(
        self,
        config: EvaluationConfig,
    ) -> ConfigResult:
        """评测单个配置。"""

        config_result = ConfigResult(config)
        config_result.start_time = time.time()

        # 并发控制
        semaphore = asyncio.Semaphore(
            self.experiment_config.max_concurrency
        )

        tasks = [
            self._evaluate_test_case(
                test_case=test_case,
                config=config,
                semaphore=semaphore,
            )
            for test_case in self.test_dataset.test_cases
        ]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                config_result.results.append(
                    {
                        "status": "error",
                        "error": str(result),
                    }
                )
                config_result.failure_count += 1
            elif result.get("status") == "success":
                config_result.results.append(result)
                config_result.success_count += 1
            else:
                config_result.results.append(result)
                config_result.failure_count += 1

        config_result.end_time = time.time()

        return config_result

    async def _evaluate_test_case(
        self,
        *,
        test_case: TestCase,
        config: EvaluationConfig,
        semaphore: asyncio.Semaphore,
    ) -> EvaluationResult:
        """评测单个测试用例。"""

        async with semaphore:
            try:
                # 执行 RAG 查询（带超时）
                rag_result = await asyncio.wait_for(
                    rag_executor.execute(
                        query=test_case.query,
                        config=config,
                        document_type=test_case.document_type,
                    ),
                    timeout=self.experiment_config.timeout_seconds,
                )

                # 计算 RAGAS 指标
                metrics = await ragas_metrics.evaluate(
                    query=test_case.query,
                    answer=rag_result["answer"],
                    contexts=rag_result["contexts"],
                    ground_truth=test_case.ground_truth,
                    reference_contexts=test_case.reference_contexts,
                )

                return EvaluationResult(
                    {
                        "status": "success",
                        "query": test_case.query,
                        "ground_truth": test_case.ground_truth,
                        "answer": rag_result["answer"],
                        "contexts": rag_result["contexts"],
                        "retrieved_count": rag_result["retrieved_count"],
                        "reflection_iterations": rag_result[
                            "reflection_iterations"
                        ],
                        "metrics": metrics,
                    }
                )

            except asyncio.TimeoutError:
                return EvaluationResult(
                    {
                        "status": "timeout",
                        "query": test_case.query,
                        "error": f"查询超时（{self.experiment_config.timeout_seconds}秒）",
                    }
                )

            except Exception as exc:
                logger.error(
                    "评测测试用例失败：%s",
                    test_case.query,
                    exc_info=True,
                )

                return EvaluationResult(
                    {
                        "status": "error",
                        "query": test_case.query,
                        "error": str(exc),
                    }
                )

    def _load_dataset(
        self,
        path: str,
    ) -> TestDataset:
        """加载测试数据集。"""

        dataset_path = Path(path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"测试数据集不存在：{path}"
            )

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return TestDataset.model_validate(data)

    def _build_report_data(self) -> dict[str, Any]:
        """构建报告数据。"""

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "name": self.test_dataset.name,
                "size": self.test_dataset.size,
                "description": self.test_dataset.description,
            },
            "configs": [
                {
                    "name": config_result.config.name,
                    "config": config_result.config.model_dump(),
                    "summary": {
                        "total": config_result.total_count,
                        "success": config_result.success_count,
                        "failure": config_result.failure_count,
                        "duration": round(config_result.duration, 2),
                    },
                    "metrics": config_result.avg_metrics,
                    "results": config_result.results,
                }
                for config_result in self.config_results.values()
            ],
        }

    async def generate_html_report(
        self,
        output_path: str | None = None,
    ) -> str:
        """生成 HTML 报告。"""

        if not self.config_results:
            raise RuntimeError("请先运行评测")

        report_data = self._build_report_data()

        if output_path is None:
            output_dir = Path("evaluation_reports")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                output_dir
                / f"report_{self.experiment_config.experiment_name}_{timestamp}.html"
            )

        generator = ReportGenerator()
        html = generator.generate(report_data)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(
            "HTML 报告已生成：%s",
            output_path,
        )

        return output_path
