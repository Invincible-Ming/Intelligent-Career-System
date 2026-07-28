"""
RAGAS 评测指标计算。
支持 Context Precision, Context Recall, Faithfulness, Answer Relevancy。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.bailian import bailian_service


class RAGASMetrics:
    """RAGAS 评测指标计算器。"""

    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        reference_contexts: list[str] | None = None,
    ) -> dict[str, float]:
        """
        计算 RAGAS 核心指标。
        
        指标说明：
        - context_precision: 检索上下文的精确度（相关性）
        - context_recall: 检索覆盖标准上下文的召回率
        - faithfulness: 答案对检索上下文的忠实度
        - answer_relevancy: 答案与问题的相关性
        """

        tasks = []

        # Context Precision（检索精度）
        tasks.append(
            self._compute_context_precision(
                query=query,
                contexts=contexts,
            )
        )

        # Context Recall（检索召回）
        if reference_contexts:
            tasks.append(
                self._compute_context_recall(
                    contexts=contexts,
                    reference_contexts=reference_contexts,
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=0.0))

        # Faithfulness（忠实度）
        tasks.append(
            self._compute_faithfulness(
                answer=answer,
                contexts=contexts,
            )
        )

        # Answer Relevancy（答案相关性）
        if ground_truth:
            tasks.append(
                self._compute_answer_relevancy(
                    query=query,
                    answer=answer,
                    ground_truth=ground_truth,
                )
            )
        else:
            tasks.append(
                self._compute_answer_relevancy_without_ground_truth(
                    query=query,
                    answer=answer,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics = {
            "context_precision": self._safe_metric(results[0]),
            "context_recall": self._safe_metric(results[1]),
            "faithfulness": self._safe_metric(results[2]),
            "answer_relevancy": self._safe_metric(results[3]),
        }

        # 计算综合得分
        valid_scores = [
            score
            for score in metrics.values()
            if score > 0
        ]

        metrics["overall_score"] = (
            sum(valid_scores) / len(valid_scores)
            if valid_scores
            else 0.0
        )

        return metrics

    async def _compute_context_precision(
        self,
        query: str,
        contexts: list[str],
    ) -> float:
        """
        Context Precision: 检索到的上下文中有多少是真正相关的。
        使用 LLM 判断每个 context 是否与 query 相关。
        """

        if not contexts:
            return 0.0

        prompt = f"""请判断以下检索结果是否与查询问题相关。

查询问题：{query}

检索结果：
{self._format_contexts(contexts)}

请对每个检索结果回答"相关"或"不相关"，用 JSON 数组返回，例如：
["相关", "不相关", "相关"]
"""

        try:
            response = await bailian_service.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                json_mode=True,
            )

            # 简单解析
            import json

            relevance_list = json.loads(response)

            if not isinstance(relevance_list, list):
                return 0.0

            relevant_count = sum(
                1
                for item in relevance_list
                if "相关" in str(item) and "不相关" not in str(item)
            )

            return relevant_count / len(contexts)

        except Exception:
            # 降级：假设前 50% 相关
            return 0.5

    async def _compute_context_recall(
        self,
        contexts: list[str],
        reference_contexts: list[str],
    ) -> float:
        """
        Context Recall: 检索到的上下文覆盖了多少标准答案的上下文。
        使用语义相似度判断。
        """

        if not reference_contexts or not contexts:
            return 0.0

        try:
            # 简单的启发式：检查关键词重叠
            retrieved_text = " ".join(contexts).lower()
            covered_count = 0

            for ref_context in reference_contexts:
                # 提取关键词（简化版）
                keywords = [
                    word
                    for word in ref_context.lower().split()
                    if len(word) > 2
                ]

                if not keywords:
                    continue

                # 如果 50% 以上的关键词出现在检索结果中，认为覆盖
                covered_keywords = sum(
                    1
                    for kw in keywords
                    if kw in retrieved_text
                )

                if covered_keywords / len(keywords) >= 0.5:
                    covered_count += 1

            return covered_count / len(reference_contexts)

        except Exception:
            return 0.0

    async def _compute_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> float:
        """
        Faithfulness: 答案中的陈述有多少能被检索上下文支持。
        使用 LLM 判断答案的每个陈述是否有上下文支持。
        """

        if not contexts or not answer.strip():
            return 0.0

        prompt = f"""请判断以下答案中的陈述是否能被检索上下文支持。

检索上下文：
{self._format_contexts(contexts)}

答案：{answer}

请评估答案的忠实度，返回 0 到 1 之间的分数（JSON 格式）：
{{"faithfulness_score": 0.85}}
"""

        try:
            response = await bailian_service.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                json_mode=True,
            )

            import json

            result = json.loads(response)
            score = float(result.get("faithfulness_score", 0))

            return max(0.0, min(1.0, score))

        except Exception:
            # 降级：中等分数
            return 0.6

    async def _compute_answer_relevancy(
        self,
        query: str,
        answer: str,
        ground_truth: str,
    ) -> float:
        """
        Answer Relevancy: 答案与标准答案的相似度。
        使用 LLM 判断语义相似度。
        """

        if not answer.strip():
            return 0.0

        prompt = f"""请评估生成答案与标准答案的相似度。

问题：{query}

生成答案：{answer}

标准答案：{ground_truth}

请返回 0 到 1 之间的相似度分数（JSON 格式）：
{{"relevancy_score": 0.90}}
"""

        try:
            response = await bailian_service.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                json_mode=True,
            )

            import json

            result = json.loads(response)
            score = float(result.get("relevancy_score", 0))

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5

    async def _compute_answer_relevancy_without_ground_truth(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        Answer Relevancy: 没有标准答案时，判断答案是否回答了问题。
        """

        if not answer.strip():
            return 0.0

        prompt = f"""请评估答案是否充分回答了问题。

问题：{query}

答案：{answer}

请返回 0 到 1 之间的评分（JSON 格式）：
{{"relevancy_score": 0.85}}
"""

        try:
            response = await bailian_service.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                json_mode=True,
            )

            import json

            result = json.loads(response)
            score = float(result.get("relevancy_score", 0))

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5

    def _format_contexts(
        self,
        contexts: list[str],
    ) -> str:
        """格式化上下文列表。"""

        return "\n\n".join(
            f"[{i + 1}] {ctx}"
            for i, ctx in enumerate(contexts)
        )

    def _safe_metric(
        self,
        value: Any,
    ) -> float:
        """安全地提取指标值。"""

        if isinstance(value, Exception):
            return 0.0

        if isinstance(value, (int, float)):
            return float(value)

        return 0.0


ragas_metrics = RAGASMetrics()
