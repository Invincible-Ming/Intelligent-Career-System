"""Official RAGAS metrics, with bounded provider calls and no synthetic fallback scores."""
from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version

RAGAS_VERSION = "0.3.9"


class RAGASMetrics:
    """Adapter for the pinned RAGAS package; importing the app does not require it."""

    def ensure_available(self) -> None:
        try:
            installed = version("ragas")
        except PackageNotFoundError as exc:
            raise ValueError(
                "评测需要安装 RAGAS：在 backend 目录执行 "
                "python -m pip install -r requirements-evaluation.txt"
            ) from exc
        if installed != RAGAS_VERSION:
            raise ValueError(f"评测适配器要求 ragas=={RAGAS_VERSION}，当前版本为 {installed}")
        # Resolve optional dependencies before starting a paid/background experiment.
        try:
            from app.evaluation.ragas_adapter import evaluate_sample  # noqa: F401
        except ImportError as exc:
            raise ValueError("RAGAS 依赖不完整，请重新安装 requirements-evaluation.txt") from exc

    async def evaluate(
        self, *, query: str, answer: str, contexts: list[str],
        ground_truth: str | None = None, reference_contexts: list[str] | None = None,
    ) -> dict[str, float]:
        self.ensure_available()
        if not ground_truth or not ground_truth.strip():
            raise ValueError("完整 RAGAS 四指标评测需要非空 ground_truth")
        if not contexts or not answer.strip():
            # An empty retrieval/answer is a real zero, not an infrastructure failure.
            return {name: 0.0 for name in (
                "context_precision", "context_recall", "faithfulness",
                "answer_relevancy", "overall_score",
            )}
        from app.evaluation.ragas_adapter import evaluate_sample

        raw = await evaluate_sample(
            query=query, answer=answer, contexts=contexts, reference=ground_truth,
        )
        metrics = {}
        for name, value in raw.items():
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f"RAGAS 指标 {name} 无有效分数，当前样本应记为失败")
            metrics[name] = max(0.0, min(1.0, score))
        # Genuine zeros participate in the mean; do not discard low scores.
        metrics["overall_score"] = sum(metrics.values()) / len(metrics)
        return metrics


ragas_metrics = RAGASMetrics()
