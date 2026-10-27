"""RAGAS 0.3.9 adapters with an independent LLM and existing embedding model."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time

# Set before importing RAGAS; evaluation telemetry is unnecessary for this app.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from langchain_core.outputs import Generation, LLMResult
from ragas import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.metrics._answer_relevance import ResponseRelevanceInput
from ragas.run_config import RunConfig

from app.core.config import settings
from app.evaluation.llm import evaluation_llm, transient_model_failure
from app.services.bailian import ModelServiceError, bailian_service

logger = logging.getLogger(__name__)


class MetricCallLimit(RuntimeError):
    """A failed sample, not a fabricated score or an experiment-wide cancellation."""


class BailianRagasLLM(BaseRagasLLM):
    def __init__(self):
        super().__init__(run_config=RunConfig(max_retries=1))
        self.calls = 0

    def generate_text(self, *args, **kwargs):
        raise RuntimeError("评测仅支持异步模型调用")

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        generations = []
        temperature = self.get_temperature(n) if temperature is None else temperature
        for _ in range(n):
            for attempt in range(settings.EVALUATION_METRIC_MAX_RETRIES + 1):
                if self.calls >= settings.EVALUATION_METRIC_MAX_CALLS:
                    raise MetricCallLimit("当前样本已达到 RAGAS 模型调用上限（含重试）")
                self.calls += 1
                logger.info("评判请求 %d/%d%s", self.calls, settings.EVALUATION_METRIC_MAX_CALLS,
                            "（重试）" if attempt else "")
                try:
                    text = await evaluation_llm.chat(
                        [{"role": "user", "content": prompt.to_string()}],
                        temperature=temperature,
                        max_output_tokens=settings.EVALUATION_METRIC_MAX_OUTPUT_TOKENS,
                        max_retries=0,
                    )
                    break
                except ModelServiceError as exc:
                    if not transient_model_failure(exc) or attempt >= settings.EVALUATION_METRIC_MAX_RETRIES:
                        raise
                    if self.calls >= settings.EVALUATION_METRIC_MAX_CALLS:
                        raise MetricCallLimit("当前样本已达到 RAGAS 模型调用上限（含重试）") from exc
                    delay = 2 ** attempt
                    logger.warning("评判请求发生临时网络或上游故障，%s 秒后重试（本次调用已计入预算）", delay)
                    await asyncio.sleep(delay)
            generations.append(Generation(text=text))
        return LLMResult(generations=[generations])

    # RAGAS must not layer an additional transport retry over the provider adapter.
    async def generate(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        return await self.agenerate_text(prompt, n, temperature, stop, callbacks)

    def is_finished(self, response):
        return bool(response.generations and response.generations[0])


class PreparedEmbeddings(BaseRagasEmbeddings):
    """RAGAS 0.3.9 relevancy calculates cosine similarity synchronously.

    Vectors are prefetched asynchronously; synchronous scoring only reads memory.
    """

    def __init__(self, vectors):
        super().__init__()
        self.vectors = vectors

    def embed_query(self, text):
        return self.vectors[text]

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


class AsyncResponseRelevancy(ResponseRelevancy):
    async def _ascore(self, row, callbacks):
        # Keep the official question-generation prompt and scoring formula.
        responses = await self.question_generation.generate_multiple(
            data=ResponseRelevanceInput(response=row["response"]),
            llm=self.llm, callbacks=callbacks, n=self.strictness,
        )
        texts = list(dict.fromkeys([row["user_input"], *(item.question for item in responses)]))
        nonempty = [text for text in texts if text.strip()]
        vectors = await bailian_service.embed(nonempty)
        prepared = dict(zip(nonempty, vectors, strict=True))
        if any(not text.strip() for text in texts):
            # The official scorer returns NaN for completely empty generated questions.
            # Supply zero vectors for isolated empty questions without provider requests.
            prepared.update({text: [0.0] * settings.EMBEDDING_DIMENSION
                             for text in texts if not text.strip()})
        self.embeddings = PreparedEmbeddings(prepared)
        return self._calculate_score(responses, row)


async def evaluate_sample(*, query: str, answer: str, contexts: list[str], reference: str, progress=None):
    llm = BailianRagasLLM()
    sample = SingleTurnSample(
        user_input=query, response=answer, retrieved_contexts=contexts, reference=reference,
    )
    metrics = {
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AsyncResponseRelevancy(llm=llm, strictness=3),
    }
    # Serial metrics share one attempt counter, including transport retries.
    scores = (progress.get("metrics") or {}) if progress is not None else {}
    if not isinstance(scores, dict) or set(scores) - set(metrics):
        raise ValueError("评测进度包含未知指标")
    if any(not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1
           for value in scores.values()):
        raise ValueError("评测进度包含无效分数")
    for name, metric in metrics.items():
        if name in scores:
            logger.info("复用已完成指标：%s", name)
            continue
        started = time.monotonic()
        logger.info("开始评判指标：%s", name)
        value = float(await metric.single_turn_ascore(sample))
        if not math.isfinite(value):
            raise ValueError(f"RAGAS 指标 {name} 无有效分数")
        scores[name] = max(0.0, min(1.0, value))
        if progress is not None:
            progress.save("metrics", scores)
        logger.info("指标完成：%s，用时 %.1f 秒", name, time.monotonic() - started)
    return scores
