"""
BGE CrossEncoder 重排服务。

优先使用 Apple Silicon 的 MPS。
如果 MPS 不可用或推理失败，自动切换到 CPU。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """BGE CrossEncoder 文本重排服务。"""

    def __init__(self) -> None:
        self.model: CrossEncoder | None = None
        self.device: str | None = None
        self._load_lock = asyncio.Lock()

    def choose_device(self) -> str:
        """选择推理设备，优先使用 MPS。"""

        configured = settings.RERANKER_DEVICE

        if configured == "cpu":
            return "cpu"

        if configured == "mps":
            if torch.backends.mps.is_available():
                return "mps"

            logger.warning(
                "配置要求使用 MPS，但当前环境不可用，改用 CPU"
            )
            return "cpu"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    async def load(self) -> None:
        """延迟加载模型。"""

        if self.model is not None:
            return

        async with self._load_lock:
            if self.model is not None:
                return

            device = self.choose_device()

            try:
                self.model = await asyncio.to_thread(
                    self._load_model,
                    device,
                )
                self.device = device

                logger.info(
                    "BGE Reranker 加载完成，设备：%s",
                    device,
                )
            except Exception:
                if device != "mps":
                    raise

                logger.warning(
                    "MPS 加载 BGE Reranker 失败，改用 CPU",
                    exc_info=True,
                )

                self.model = await asyncio.to_thread(
                    self._load_model,
                    "cpu",
                )
                self.device = "cpu"

                logger.info(
                    "BGE Reranker 已切换到 CPU"
                )

    @staticmethod
    def _load_model(
        device: str,
    ) -> CrossEncoder:
        """同步加载模型。"""

        return CrossEncoder(
            settings.RERANKER_MODEL,
            device=device,
            max_length=settings.RERANKER_MAX_LENGTH,
        )

    async def rerank(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """对 RRF 结果进行 BGE 重排。"""

        if not results:
            return []

        if not settings.RERANKER_ENABLED:
            return results[:top_k]

        await self.load()

        try:
            return await asyncio.to_thread(
                self._rerank_sync,
                query,
                results,
                top_k,
            )
        except Exception:
            if self.device != "mps":
                raise

            logger.warning(
                "MPS 执行 BGE Reranker 失败，改用 CPU",
                exc_info=True,
            )

            await self.switch_to_cpu()

            return await asyncio.to_thread(
                self._rerank_sync,
                query,
                results,
                top_k,
            )

    def _rerank_sync(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """同步执行 CrossEncoder 预测。"""

        if self.model is None:
            raise RuntimeError(
                "BGE Reranker 尚未加载"
            )

        pairs = [
            [
                query,
                str(result.get("content", "")),
            ]
            for result in results
        ]

        scores = self.model.predict(
            pairs,
            batch_size=settings.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []

        for result, score in zip(
            results,
            scores,
            strict=True,
        ):
            item = dict(result)

            # 保留 RRF 分数，最终 score 改为 BGE 重排分。
            item["rrf_score"] = float(
                result.get(
                    "rrf_score",
                    result.get("score", 0.0),
                )
            )
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            item["source"] = "hybrid_reranked"

            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]

    async def switch_to_cpu(self) -> None:
        """释放当前模型并切换到 CPU。"""

        self.model = None
        self.device = None

        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        async with self._load_lock:
            if self.model is None:
                self.model = await asyncio.to_thread(
                    self._load_model,
                    "cpu",
                )
                self.device = "cpu"

    async def close(self) -> None:
        """释放模型。"""

        self.model = None
        self.device = None

        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


reranker_service = RerankerService()