"""
RAG 查询执行器，支持不同的检索策略和反思机制。
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.limits import BudgetExceeded
from app.core.prompts import RAG_ANSWER_SYSTEM_PROMPT, RAG_REFLECTION_SYSTEM_PROMPT, prompt_data
from app.evaluation.config import EvaluationConfig
from app.evaluation.checkpoint import CaseCheckpoint
from app.evaluation.llm import evaluation_llm
from app.evaluation.scope import normalize_owner_id
from app.services.bailian import bailian_service
from app.services.bm25_service import bm25_service
from app.services.hybrid_search import hybrid_search_service
from app.services.milvus_service import milvus_service, validate_document_id
from app.services.reranker_service import reranker_service

logger = logging.getLogger(__name__)


class RAGExecutor:
    """RAG 查询执行器。"""

    async def execute(
            self,
            *,
            query: str,
            config: EvaluationConfig,
            owner_id: str,
            allowed_document_ids: list[str],
            document_type: str | None = None,
            progress: CaseCheckpoint | None = None,
    ) -> dict[str, Any]:
        """
        执行 RAG 查询。
        
        返回格式：
        {
            "answer": "生成的答案",
            "contexts": ["检索到的上下文1", "上下文2"],
            "retrieved_count": 5,
            "reflection_iterations": 0,
        }
        """

        owner_id = normalize_owner_id(owner_id)
        for identifier in allowed_document_ids:
            validate_document_id(identifier)
        # 第一步：检索
        contexts = progress.get("contexts") if progress is not None else None
        if contexts is None:
            logger.info("开始检索与重排")
            contexts = await self._retrieve(
                query=query,
                config=config,
                owner_id=owner_id,
                allowed_document_ids=allowed_document_ids,
                document_type=document_type,
            )
            if progress is not None:
                progress.save("contexts", contexts)
        else:
            logger.info("复用进度中已完成的检索与重排")

        if not contexts:
            return {
                "answer": "抱歉，没有检索到相关信息。",
                "contexts": [],
                "retrieved_count": 0,
                "reflection_iterations": 0,
            }

        # 第二步：生成答案
        logger.info("检索完成：%d 个上下文，开始生成回答", len(contexts))
        answer = await self._generate_answer(
            query=query,
            contexts=contexts,
        )

        logger.info("回答生成完成")
        reflection_iterations = 0

        # 第三步：反思（如果启用）
        if config.reflection.enable_reflection:
            answer, reflection_iterations = await self._reflect(
                query=query,
                answer=answer,
                contexts=contexts,
                config=config,
            )

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_count": len(contexts),
            "reflection_iterations": reflection_iterations,
        }

    async def _retrieve(
            self,
            *,
            query: str,
            config: EvaluationConfig,
            owner_id: str,
            allowed_document_ids: list[str],
            document_type: str | None,
    ) -> list[str]:
        """根据配置执行检索。"""

        search_mode = config.search.search_mode
        top_k = config.search.top_k
        if not allowed_document_ids:
            return []
        # Over-retrieve for ranking, but keep the candidate budget comparable
        # across Dense/BM25/Hybrid experiments.
        candidate_k = min(max(top_k * 2, 10), 20) if config.search.enable_reranker else top_k

        try:
            if search_mode == "dense":
                results = await self._dense_search(
                    query=query,
                    owner_id=owner_id,
                    allowed_document_ids=allowed_document_ids,
                    document_type=document_type,
                    top_k=candidate_k,
                )
            elif search_mode == "bm25":
                results = await bm25_service.search(
                    query=query,
                    owner_id=owner_id,
                    allowed_document_ids=allowed_document_ids,
                    document_type=document_type,
                    top_k=candidate_k,
                )
            elif search_mode == "hybrid":
                results = await hybrid_search_service.search(
                    query=query,
                    owner_id=owner_id,
                    allowed_document_ids=allowed_document_ids,
                    document_type=document_type,
                    top_k=candidate_k,
                    enable_reranker=False,
                    rrf_candidate_k=candidate_k,
                )
            else:
                raise ValueError(f"不支持的评测检索模式：{search_mode}")

            if config.search.enable_reranker:
                results = await reranker_service.rerank(
                    query=query, results=results,
                    top_k=min(top_k, config.search.reranker_top_k), enabled=True,
                )
            else:
                results = results[:top_k]

            return [
                result.get("content", "")
                for result in results
                if result.get("content")
            ]

        except Exception as exc:
            logger.warning(
                "评测检索失败（mode=%s）：%s",
                search_mode,
                type(exc).__name__,
            )
            raise

    async def _dense_search(
            self,
            *,
            query: str,
            owner_id: str,
            allowed_document_ids: list[str],
            document_type: str | None,
            top_k: int,
    ) -> list[dict[str, Any]]:
        """Dense 向量检索。"""

        query_vector = await bailian_service.embed_query(query)

        return await milvus_service.search(
            query_vector=query_vector,
            owner_id=owner_id,
            allowed_document_ids=allowed_document_ids,
            document_type=document_type,
            top_k=top_k,
        )

    async def _generate_answer(
            self,
            *,
            query: str,
            contexts: list[str],
    ) -> str:
        """基于检索上下文生成答案。"""

        # Let the runner record a failed case instead of scoring an error
        # message as a successful model answer.
        return await evaluation_llm.chat(
            messages=[
                {"role": "system", "content": RAG_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_data(
                    question=query,
                    contexts=[{"source": i + 1, "text": ctx} for i, ctx in enumerate(contexts)],
                )},
            ],
            temperature=0.3,
        )

    async def _reflect(
            self,
            *,
            query: str,
            answer: str,
            contexts: list[str],
            config: EvaluationConfig,
    ) -> tuple[str, int]:
        """
        反思机制：让模型审查并改进自己的答案。
        返回 (最终答案, 反思次数)。
        """

        reflection_config = config.reflection
        max_iterations = reflection_config.max_iterations
        current_answer = answer
        completed_iterations = 0

        for _ in range(max_iterations):
            completed_iterations += 1
            try:
                improved_answer = await evaluation_llm.chat(
                    messages=[
                        {"role": "system", "content": RAG_REFLECTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": prompt_data(
                                question=query, current_answer=current_answer,
                                contexts=[{"source": i + 1, "text": ctx} for i, ctx in enumerate(contexts)],
                                review_focus=reflection_config.reflection_prompt,
                            ),
                        }
                    ],
                    temperature=0.2,
                )

                if not improved_answer.strip():
                    break
                # A changed ending can contain the factual fix; never discard it
                # just because the first 100 characters stayed the same.
                if current_answer.strip() == improved_answer.strip():
                    break

                current_answer = improved_answer

            except BudgetExceeded:
                raise
            except Exception:
                break

        return current_answer, completed_iterations

rag_executor = RAGExecutor()
