"""Independent evaluation LLM; embedding calls remain on the indexing model."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from openai import APIConnectionError, AsyncOpenAI

from app.core.config import settings
from app.services.bailian import BailianService, ModelServiceError

logger = logging.getLogger(__name__)


def transient_model_failure(exc: ModelServiceError) -> bool:
    if exc.code in ("insufficient_quota", "AllocationQuota.FreeTierOnly"):
        return False
    if exc.status_code is not None:
        return exc.status_code in (408, 429, 500, 502, 503, 504)
    cause = exc.__cause__
    seen = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, (APIConnectionError, httpx.TransportError)):
            return True
        cause = cause.__cause__
    return False


class EvaluationLLM(BailianService):
    """Reuse request budgets, token accounting and output checks for evaluation."""

    @property
    def provider(self) -> str:
        return settings.EVALUATION_LLM_PROVIDER

    @property
    def model(self) -> str:
        return settings.EVALUATION_LLM_MODEL.strip() or (
            "mimo-v2.6-flash" if self.provider == "mimo" else settings.BAILIAN_CHAT_MODEL
        )

    def validate_configuration(self) -> None:
        if self.provider == "mimo" and not settings.MIMO_API_KEY.strip():
            raise RuntimeError("评测使用小米 MiMo，请先在后端配置 MIMO_API_KEY。")
        if self.provider == "bailian" and not settings.DASHSCOPE_API_KEY.strip():
            raise RuntimeError("评测使用百炼，请先在后端配置 DASHSCOPE_API_KEY。")

    @property
    def chat_client(self) -> AsyncOpenAI:
        if self.provider == "bailian":
            return super().chat_client
        self.validate_configuration()
        if self._chat_client is None:
            self._chat_client = AsyncOpenAI(
                api_key=settings.MIMO_API_KEY.strip(),
                base_url=settings.MIMO_BASE_URL,
                timeout=settings.MODEL_TIMEOUT,
                max_retries=0,
                http_client=httpx.AsyncClient(trust_env=False, follow_redirects=False),
            )
        return self._chat_client

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.validate_configuration()
        kwargs["model"] = kwargs.get("model") or self.model
        retries = kwargs.pop("max_retries", None)
        retries = settings.EVALUATION_LLM_MAX_RETRIES if retries is None else retries
        if retries < 0:
            raise ValueError("max_retries 不能为负数")
        for attempt in range(retries + 1):
            started = time.monotonic()
            logger.info("模型请求开始：%s，单次总耗时上限 %s 秒%s", self.model,
                        settings.EVALUATION_LLM_REQUEST_TIMEOUT, "（重试）" if attempt else "")
            try:
                try:
                    # HTTPX's read timeout is per response chunk. Bound the whole
                    # attempt too; SDK and base-service retries stay disabled.
                    async with asyncio.timeout(settings.EVALUATION_LLM_REQUEST_TIMEOUT):
                        response = await super().chat(messages, max_retries=0, **kwargs)
                except TimeoutError as exc:
                    raise ModelServiceError("evaluation_request_timeout", 408) from exc
                logger.info("模型请求完成，用时 %.1f 秒", time.monotonic() - started)
                return response
            except ModelServiceError as exc:
                if attempt >= retries or not transient_model_failure(exc):
                    raise
                logger.warning("模型请求临时失败，用时 %.1f 秒，准备重试", time.monotonic() - started)
                await asyncio.sleep(2 ** attempt)

    async def _chat_retry(self, retry_limit=None, client=None, **request_params):
        if self.provider == "mimo":
            # MiMo uses different thinking and output-limit parameter names.
            # Keep evaluation thinking disabled to bound latency and token use.
            request_params["max_completion_tokens"] = request_params.pop("max_tokens")
            request_params.pop("reasoning_effort", None)
            extra = dict(request_params.pop("extra_body", None) or {})
            extra.pop("enable_thinking", None)
            extra["thinking"] = {"type": "disabled"}
            request_params["extra_body"] = extra
        return await super()._chat_retry(
            retry_limit=retry_limit, client=client, **request_params,
        )


evaluation_llm = EvaluationLLM()
