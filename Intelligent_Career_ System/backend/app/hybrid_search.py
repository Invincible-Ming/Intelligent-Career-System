"""
Milvus Dense Search + BM25 + RRF + BGE Reranker。
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

from app.bailian import bailian_service
from app.bm25_service import bm25_service
from app.config import settings
from app.milvus_service import milvus_service
from app.reranker_service import reranker_service


class HybridSearchService:
    """混合检索服务。"""

    async def search(
        self,
        *,
        query: str,
        document_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        执行：

        Milvus Dense
        → BM25
        → RRF
        → BGE CrossEncoder
        """

        query = query.strip()

        if not query:
            raise ValueError("查询内容不能为空")

        # 两路检索各自多召回一些结果。
        candidate_k = min(
            max(top_k * 2, 10),
            20,
        )

        dense_results, bm25_results = await asyncio.gather(
            self._dense_search(
                query=query,
                document_type=document_type,
                top_k=candidate_k,
            ),
            bm25_service.search(
                query=query,
                document_type=document_type,
                top_k=candidate_k,
            ),
        )

        # RRF 后保留少量候选给 BGE。
        rrf_results = rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=settings.RERANKER_CANDIDATE_K,
        )

        if not rrf_results:
            return []

        if not settings.RERANKER_ENABLED:
            return rrf_results[:top_k]

        return await reranker_service.rerank(
            query=query,
            results=rrf_results,
            top_k=min(
                top_k,
                settings.RERANKER_TOP_K,
            ),
        )

    async def _dense_search(
        self,
        *,
        query: str,
        document_type: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """执行 Milvus Dense Search。"""

        query_vector = await bailian_service.embed_query(
            query
        )

        return await milvus_service.search(
            query_vector=query_vector,
            document_type=document_type,
            top_k=top_k,
        )


def rrf_fusion(
    *,
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """按照结果排名执行 Reciprocal Rank Fusion。"""

    fused: dict[str, dict[str, Any]] = {}

    add_ranked_results(
        fused=fused,
        results=dense_results,
        source="dense",
        rrf_k=rrf_k,
    )

    add_ranked_results(
        fused=fused,
        results=bm25_results,
        source="bm25",
        rrf_k=rrf_k,
    )

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    output: list[dict[str, Any]] = []

    for item in ranked[:top_k]:
        sources = item.pop("_sources")

        output.append(
            {
                "document_id": item["document_id"],
                "document_type": item["document_type"],
                "content": item["content"],
                "score": round(
                    item["rrf_score"],
                    8,
                ),
                "rrf_score": round(
                    item["rrf_score"],
                    8,
                ),
                "source": (
                    "hybrid"
                    if len(sources) > 1
                    else next(iter(sources))
                ),
            }
        )

    return output


def add_ranked_results(
    *,
    fused: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    source: str,
    rrf_k: int,
) -> None:
    """添加一路已经排序的召回结果。"""

    for rank, result in enumerate(
        results,
        start=1,
    ):
        key = build_result_key(result)
        score = 1.0 / (rrf_k + rank)

        if key not in fused:
            fused[key] = {
                "document_id": result.get(
                    "document_id",
                    "",
                ),
                "document_type": result.get(
                    "document_type",
                    "",
                ),
                "content": result.get(
                    "content",
                    "",
                ),
                "rrf_score": 0.0,
                "_sources": set(),
            }

        fused[key]["rrf_score"] += score
        fused[key]["_sources"].add(source)


def build_result_key(
    result: dict[str, Any],
) -> str:
    """根据文档 ID 和文本内容去重。"""

    document_id = str(
        result.get("document_id", "")
    )
    content = normalize_content(
        str(result.get("content", ""))
    )

    raw_key = f"{document_id}\n{content}"

    return hashlib.sha1(
        raw_key.encode("utf-8")
    ).hexdigest()


def normalize_content(content: str) -> str:
    """统一空白字符。"""

    return re.sub(
        r"\s+",
        " ",
        content,
    ).strip()


hybrid_search_service = HybridSearchService()