"""
RAG 查询执行器，支持不同的检索策略和反思机制。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.bailian import bailian_service
from app.bm25_service import bm25_service
from app.evaluation.config import EvaluationConfig
from app.hybrid_search import hybrid_search_service
from app.milvus_service import milvus_service


class RAGExecutor:
    """RAG 查询执行器。"""

    async def execute(
        self,
        *,
        query: str,
        config: EvaluationConfig,
        document_type: str | None = None,
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

        # 第一步：检索
        contexts = await self._retrieve(
            query=query,
            config=config,
            document_type=document_type,
        )

        if not contexts:
            return {
                "answer": "抱歉，没有检索到相关信息。",
                "contexts": [],
                "retrieved_count": 0,
                "reflection_iterations": 0,
            }

        # 第二步：生成答案
        answer = await self._generate_answer(
            query=query,
            contexts=contexts,
        )

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
        document_type: str | None,
    ) -> list[str]:
        """根据配置执行检索。"""

        search_mode = config.search.search_mode
        top_k = config.search.top_k

        try:
            if search_mode == "dense":
                results = await self._dense_search(
                    query=query,
                    document_type=document_type,
                    top_k=top_k,
                )
            elif search_mode == "bm25":
                results = await bm25_service.search(
                    query=query,
                    document_type=document_type,
                    top_k=top_k,
                )
            elif search_mode == "hybrid":
                results = await hybrid_search_service.search(
                    query=query,
                    document_type=document_type,
                    top_k=top_k,
                )
            else:
                results = []

            return [
                result.get("content", "")
                for result in results
                if result.get("content")
            ]

        except Exception:
            return []

    async def _dense_search(
        self,
        *,
        query: str,
        document_type: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Dense 向量检索。"""

        query_vector = await bailian_service.embed_query(query)

        return await milvus_service.search(
            query_vector=query_vector,
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

        context_text = "\n\n".join(
            f"[{i + 1}] {ctx}"
            for i, ctx in enumerate(contexts[:5])
        )

        prompt = f"""请基于以下检索到的上下文回答问题。

上下文：
{context_text}

问题：{query}

请给出准确、简洁的答案。如果上下文中没有相关信息，请说明。"""

        try:
            return await bailian_service.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.3,
            )
        except Exception as exc:
            return f"生成答案时出错：{exc}"

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

        for iteration in range(max_iterations):
            # 让模型反思
            reflection_prompt = f"""
{reflection_config.reflection_prompt}

原问题：{query}

当前答案：{current_answer}

参考上下文：
{self._format_contexts(contexts)}

请给出改进后的答案，如果当前答案已经足够好，可以保持不变。"""

            try:
                improved_answer = await bailian_service.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": reflection_prompt,
                        }
                    ],
                    temperature=0.2,
                )

                # 简单的停止条件：如果改进后的答案与原答案高度相似，停止反思
                if self._is_similar(current_answer, improved_answer):
                    break

                current_answer = improved_answer

            except Exception:
                break

        return current_answer, iteration + 1

    def _format_contexts(
        self,
        contexts: list[str],
    ) -> str:
        """格式化上下文。"""

        return "\n\n".join(
            f"[{i + 1}] {ctx}"
            for i, ctx in enumerate(contexts[:5])
        )

    def _is_similar(
        self,
        text1: str,
        text2: str,
        threshold: float = 0.9,
    ) -> bool:
        """简单的文本相似度判断。"""

        # 简化版：比较长度和前 100 个字符
        if abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1) > 0.2:
            return False

        prefix_len = min(100, len(text1), len(text2))

        return text1[:prefix_len] == text2[:prefix_len]


rag_executor = RAGExecutor()
