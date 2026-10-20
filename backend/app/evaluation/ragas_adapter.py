"""RAGAS 0.3.9 provider adapters; all paid requests use the existing Bailian service."""
from __future__ import annotations

import os

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
from app.services.bailian import bailian_service


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
            if self.calls >= settings.EVALUATION_METRIC_MAX_CALLS:
                raise MetricCallLimit("当前样本已达到 RAGAS 模型调用上限")
            self.calls += 1
            text = await bailian_service.chat(
                [{"role": "user", "content": prompt.to_string()}],
                temperature=temperature,
                max_output_tokens=settings.EVALUATION_METRIC_MAX_OUTPUT_TOKENS,
                max_retries=0,
            )
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


async def evaluate_sample(*, query: str, answer: str, contexts: list[str], reference: str):
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
    # Serial metrics share one call counter; no orphan requests after a metric fails.
    return {name: float(await metric.single_turn_ascore(sample))
            for name, metric in metrics.items()}
