"""
阿里云百炼客户端。

聊天模型：
    使用百炼 OpenAI 兼容接口。

Embedding：
    使用 DashScope SDK 调用 qwen3.7-text-embedding。
"""

from __future__ import annotations

import asyncio
import json
import re  # 🌟 新增：用于强大的正则提取
from collections.abc import AsyncGenerator  # 🌟 新增：导入 AsyncGenerator
from http import HTTPStatus
from typing import Any, TypeVar

import dashscope
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.limits import BudgetExceeded, charge_model_call, model_request_timeout, validate_model_input
from app.core.prompts import JSON_REPAIR_INSTRUCTIONS, JSON_SCHEMA_INSTRUCTIONS, OCR_PROMPT, prompt_data
from app.services.credits import ModelCharge
from app.services.embed_cache import embed_cache

SchemaType = TypeVar(
    "SchemaType",
    bound=BaseModel,
)


class ModelServiceError(RuntimeError):
    def __init__(self, code=None, status_code=None, message=None):
        self.code = code
        self.status_code = status_code
        detail = message or (f"code={code}, status={status_code}" if (code or status_code) else None)
        super().__init__(f'模型服务不可用（{detail}）' if detail else '模型服务不可用')


class StructuredOutputError(RuntimeError):
    """Safe diagnostics without retaining the model's response or source documents."""

    def __init__(self, code: str, fields: list[dict[str, Any]] | None = None):
        self.code = code
        self.fields = fields or []
        super().__init__(code)


class OcrServiceError(RuntimeError):
    """A safe document-facing OCR error, without source images or upstream payloads."""


def public_model_error(exc):
    if getattr(exc, 'code', None) in ('AllocationQuota.FreeTierOnly', 'insufficient_quota'):
        return {'message': '百炼模型免费额度已用完，当前开启“用完即停”；请联系管理员处理模型额度。',
                'type': 'model_quota_exceeded'}
    return {'message': '回复生成失败，请稍后重试', 'type': 'server_error'}


def public_ocr_error(exc, provider="bailian"):
    code, status = getattr(exc, "code", None), getattr(exc, "status_code", None)
    if code in ("AllocationQuota.FreeTierOnly", "insufficient_quota"):
        return "文档需要 OCR 识别，但 OCR 模型额度不足；请检查模型额度或更换有可用额度的 OCR 模型。"
    if status == 404 or code == "model_not_found":
        config_name = "MIMO_OCR_MODEL" if provider == "mimo" else "BAILIAN_OCR_MODEL"
        return f"文档需要 OCR 识别，但当前接口不支持配置的模型；请检查后端 {config_name} 配置。"
    if status == 401:
        vendor = "小米 MiMo" if provider == "mimo" else "百炼"
        return f"文档 OCR 认证失败，请检查{vendor} API Key 与接口地址。"
    if status == 403:
        return "当前账号无权调用 OCR 模型，请检查账号及模型调用权限。"
    if status == 429:
        return "OCR 服务请求频率受限，请稍后重试。"
    return "文档 OCR 识别暂时不可用，请检查模型服务或稍后重试。"


class BailianService:
    """百炼聊天和 Embedding，以及可独立切换的文档 OCR。"""

    def __init__(self) -> None:
        self._chat_client: AsyncOpenAI | None = None
        self._ocr_client: AsyncOpenAI | None = None

    @property
    def ocr_client(self) -> AsyncOpenAI:
        """MiMo 凭证只保留在后端，不传给文档沙箱。"""
        if settings.OCR_PROVIDER == "bailian":
            return self.chat_client
        if not settings.MIMO_API_KEY.strip():
            raise OcrServiceError("未配置小米 OCR 密钥，请检查后端 MIMO_API_KEY。")
        if self._ocr_client is None:
            self._ocr_client = AsyncOpenAI(
                api_key=settings.MIMO_API_KEY.strip(),
                base_url=settings.MIMO_BASE_URL,
                timeout=settings.MODEL_TIMEOUT,
                max_retries=0,
                http_client=httpx.AsyncClient(trust_env=False, follow_redirects=False),
            )
        return self._ocr_client

    @property
    def chat_client(self) -> AsyncOpenAI:
        """延迟创建聊天模型客户端。"""

        if not settings.DASHSCOPE_API_KEY:
            raise RuntimeError(
                "未配置 DASHSCOPE_API_KEY"
            )

        if self._chat_client is None:
            self._chat_client = AsyncOpenAI(
                api_key=settings.DASHSCOPE_API_KEY,
                base_url=settings.BAILIAN_BASE_URL,
                timeout=settings.MODEL_TIMEOUT,
                max_retries=0,
            )

        return self._chat_client

    async def chat(
            self,
            messages: list[dict[str, str]],
            *,
            temperature: float = 0.2,
            json_mode: bool = False,
            max_output_tokens: int | None = None,
            model: str | None = None,
            reasoning_effort: str | None = None,
            enable_thinking: bool | None = None,
            max_retries: int | None = None,
    ) -> str:
        """调用百炼聊天模型。"""

        if not messages:
            raise ValueError("messages 不能为空")

        request_params: dict[str, Any] = {
            "model": model or settings.BAILIAN_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_output_tokens or settings.MODEL_MAX_OUTPUT_TOKENS, settings.MODEL_MAX_OUTPUT_TOKENS),
        }

        if json_mode:
            request_params["response_format"] = {
                "type": "json_object",
            }
        if reasoning_effort is not None:
            request_params["reasoning_effort"] = reasoning_effort
        if enable_thinking is not None:
            request_params["extra_body"] = {"enable_thinking": enable_thinking}

        response = await self._chat_retry(
            retry_limit=max_retries,
            **request_params
        )

        if not response.choices:
            raise RuntimeError(
                "百炼聊天模型没有返回结果"
            )

        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise StructuredOutputError("output_truncated")

        content = response.choices[0].message.content

        if not content:
            raise StructuredOutputError("empty_output")

        if len(content.encode("utf-8")) > request_params["max_tokens"] * 16:
            raise BudgetExceeded("模型输出超过预算")
        return content.strip()

    # 🌟 修改点：更新为 stream_chat，并补充了 AsyncGenerator 类型提示
    async def stream_chat(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.2,
            model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式调用百炼聊天模型。
        接收系统提示词和用户提示词，返回异步生成器，逐块产出内容。
        """

        if not system_prompt or not user_prompt:
            raise ValueError("system_prompt 和 user_prompt 不能为空")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async for token in self.stream_messages(messages=messages, temperature=temperature, model=model):
            yield token

    async def stream_messages(self, *, messages, temperature=.2, max_output_tokens=None, model: str | None = None):
        validate_model_input(messages)
        charge_model_call()
        cap = min(max_output_tokens or settings.MODEL_MAX_OUTPUT_TOKENS, settings.MODEL_MAX_OUTPUT_TOKENS)
        charge = await ModelCharge.begin(model or settings.BAILIAN_CHAT_MODEL, messages, cap)
        stream = None
        consumed = 0
        output = ''
        usage = None
        completed = False
        try:
            stream = await self.chat_client.chat.completions.create(
                model=model or settings.BAILIAN_CHAT_MODEL, messages=messages, temperature=temperature,
                stream=True, stream_options={"include_usage": True}, max_tokens=cap,
                timeout=model_request_timeout(),
            )
            async for chunk in stream:
                usage = getattr(chunk, "usage", None) or usage
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    output += token
                    consumed += len(token.encode('utf-8'))
                    if consumed > cap * 16:
                        raise BudgetExceeded('模型输出超过预算')
                    yield token
            completed = True
        finally:
            charge.record(usage=usage, output=output, completed=completed)
            try:
                if stream is not None:
                    await stream.close()
            finally:
                await charge.finish()

    async def ocr_image(
            self,
            base64_image: str,
            *,
            model: str | None = None,
            temperature: float = 0.1,
    ) -> str:
        """
        使用独立配置的视觉模型提取图片文本。
        """
        model = model or (settings.MIMO_OCR_MODEL if settings.OCR_PROVIDER == "mimo"
                          else settings.BAILIAN_OCR_MODEL)
        if not base64_image:
            raise ValueError("图片 Base64 数据不能为空")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OCR_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]

        request_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if settings.OCR_PROVIDER == "mimo":
            request_params["max_completion_tokens"] = settings.MODEL_MAX_OUTPUT_TOKENS
            request_params["extra_body"] = {"thinking": {"type": "disabled"}}

        try:
            response = await self._chat_retry(client=self.ocr_client, **request_params)
        except ModelServiceError as exc:
            raise OcrServiceError(public_ocr_error(exc, settings.OCR_PROVIDER)) from exc

        if not response.choices:
            raise OcrServiceError("OCR 模型没有返回识别结果，请稍后重试。")

        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise OcrServiceError("OCR 识别结果超过单页输出上限，请拆分密集页面或调整 MODEL_MAX_OUTPUT_TOKENS。")

        content = response.choices[0].message.content

        if not content:
            raise OcrServiceError("OCR 模型返回内容为空，请检查图片清晰度。")

        return content.strip()

    async def structured_chat(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            response_model: type[SchemaType],
            reasoning_effort: str | None = None,
            enable_thinking: bool | None = None,
            validation_context: dict[str, Any] | None = None,
            max_output_tokens: int | None = None,
            schema_attempts: int = 2,
            max_retries: int | None = None,
    ) -> SchemaType:
        """
        调用聊天模型并将返回的 JSON 转换为 Pydantic 模型。

        默认允许一次 Schema 修正；成本敏感调用可设 schema_attempts=1。
        """

        if not 1 <= schema_attempts <= 2:
            raise ValueError("schema_attempts 必须为 1 或 2")
        schema = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"{JSON_SCHEMA_INSTRUCTIONS}\n"
                    f"返回内容必须符合以下 JSON Schema：{schema}"
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        for attempt in range(schema_attempts):
            content = await self.chat(
                messages,
                json_mode=True,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
                **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
            )

            try:
                # 🌟 调用强化后的 clean_json
                return response_model.model_validate_json(
                    clean_json(content), context=validation_context,
                )
            except ValidationError as exc:
                if attempt == schema_attempts - 1:
                    fields = [
                        {"field": list(error["loc"]), "type": error["type"]}
                        for error in exc.errors(include_input=False, include_url=False)[:8]
                    ]
                    code = "invalid_json" if any(item["type"] == "json_invalid" for item in fields) else "invalid_schema"
                    raise StructuredOutputError(code, fields) from exc

                # Do not echo complete input values/secrets from validation errors.
                errors = [
                    {"field": list(error["loc"]), "type": error["type"], "message": error["msg"][:240]}
                    for error in exc.errors(include_input=False, include_url=False)[:8]
                ]
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": content,
                        },
                        {
                            "role": "user",
                            "content": (
                                JSON_REPAIR_INSTRUCTIONS + "\n" + prompt_data(validation_errors=errors)
                            ),
                        },
                    ]
                )

        raise RuntimeError(
            "模型结构化输出失败"
        )

    async def embed(
            self,
            texts: list[str],
    ) -> list[list[float]]:
        """
        使用 qwen3.7-text-embedding 生成文本向量。

        DashScope SDK 是同步接口，因此通过 asyncio.to_thread
        调用，避免阻塞 FastAPI 事件循环。

        命中 Embedding 缓存时不调用模型，也不计模型调用次数。
        """

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        if not settings.DASHSCOPE_API_KEY:
            raise RuntimeError(
                "未配置 DASHSCOPE_API_KEY"
            )

        cached = await embed_cache.lookup(cleaned_texts)
        pending = [
            text for text in cleaned_texts
            if text not in cached
        ]

        if not pending:
            return [cached[text] for text in cleaned_texts]

        # 去重后调用：DashScope 按 text_index 顺序返回，
        # 同一文本重复传入会导致索引与输入的对应关系不明确。
        unique_pending = list(dict.fromkeys(pending))

        validate_model_input([{"content": text} for text in unique_pending])
        response = await self._embedding_retry(
            unique_pending
        )

        output = response.output or {}
        embedding_items = output.get(
            "embeddings",
            [],
        )

        if not embedding_items:
            raise RuntimeError(
                "百炼 Embedding 没有返回向量"
            )

        # 根据 text_index 排序，确保向量顺序与输入文本一致。
        embedding_items = sorted(
            embedding_items,
            key=lambda item: item.get(
                "text_index",
                0,
            ),
        )

        vectors = [
            item["embedding"]
            for item in embedding_items
        ]

        if len(vectors) != len(unique_pending):
            raise RuntimeError(
                "Embedding 返回数量与输入数量不一致："
                f"输入 {len(unique_pending)} 条，"
                f"返回 {len(vectors)} 条"
            )

        for vector in vectors:
            actual_dimension = len(vector)

            if (
                    actual_dimension
                    != settings.EMBEDDING_DIMENSION
            ):
                raise RuntimeError(
                    "Embedding 向量维度不一致："
                    f"配置为 {settings.EMBEDDING_DIMENSION}，"
                    f"实际返回 {actual_dimension}"
                )

        await embed_cache.store(unique_pending, vectors)

        fresh = dict(zip(unique_pending, vectors, strict=True))
        for text, vector in fresh.items():
            cached.setdefault(text, vector)

        return [cached[text] for text in cleaned_texts]

    async def embed_query(
            self,
            query: str,
    ) -> list[float]:
        """生成一条检索问题向量。"""

        query = query.strip()

        if not query:
            raise ValueError(
                "查询文本不能为空"
            )

        vectors = await self.embed([query])
        return vectors[0]

    async def _chat_retry(
            self,
            retry_limit: int | None = None,
            client: AsyncOpenAI | None = None,
            **request_params: Any,
    ) -> Any:
        """聊天模型失败后进行简单重试。"""

        if "max_completion_tokens" not in request_params:
            request_params.setdefault("max_tokens", settings.MODEL_MAX_OUTPUT_TOKENS)
        output_cap = request_params.get("max_completion_tokens", request_params.get("max_tokens"))
        validate_model_input(request_params["messages"])
        last_error: Exception | None = None
        retry_limit = settings.MODEL_MAX_RETRIES if retry_limit is None else retry_limit
        if retry_limit < 0:
            raise ValueError("max_retries 不能为负数")

        for attempt in range(
                retry_limit + 1
        ):
            charge_model_call()
            charge = await ModelCharge.begin(request_params["model"], request_params["messages"], output_cap)
            try:
                response = await (
                    (client if client is not None else self.chat_client)
                    .chat
                    .completions
                    .create(**request_params, timeout=model_request_timeout())
                )
                charge.record(usage=getattr(response, "usage", None), output=''.join(
                    choice.message.content or '' for choice in response.choices))
                for choice in response.choices:
                    content = choice.message.content or ''
                    if len(content.encode('utf-8')) > output_cap * 16:
                        raise BudgetExceeded('模型输出超过预算')
                return response
            except Exception as exc:
                last_error = exc

                status = getattr(exc, "status_code", None)
                if (
                        status is not None and 400 <= status < 500 and status != 429) or attempt >= retry_limit:
                    break

            finally:
                await charge.finish()
            await asyncio.sleep(2 ** attempt)

        raise ModelServiceError(getattr(last_error, "code", None),
                                getattr(last_error, "status_code", None)) from last_error

    async def _embedding_retry(
            self,
            texts: list[str],
    ) -> Any:
        """调用 DashScope Embedding，并进行简单重试。"""

        last_error: Exception | None = None

        for attempt in range(
                settings.MODEL_MAX_RETRIES + 1
        ):
            charge_model_call()
            charge = await ModelCharge.begin(settings.BAILIAN_EMBEDDING_MODEL,
                                             [{"content": text} for text in texts], 0)
            try:
                response = await asyncio.to_thread(
                    dashscope.TextEmbedding.call,
                    model=(
                        settings
                        .BAILIAN_EMBEDDING_MODEL
                    ),
                    input=texts,
                    dimension=(
                        settings
                        .EMBEDDING_DIMENSION
                    ),
                    request_timeout=settings.MODEL_TIMEOUT,
                    api_key=(
                        settings
                        .DASHSCOPE_API_KEY
                    ),
                )

                if response.status_code == HTTPStatus.OK:
                    charge.record(usage=getattr(response, "usage", None))
                    return response

                error_code = getattr(
                    response,
                    "code",
                    "UNKNOWN_ERROR",
                )
                error_message = getattr(
                    response,
                    "message",
                    "未知错误",
                )

                last_error = ModelServiceError(error_code, response.status_code, error_message)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    break

            except Exception as exc:
                last_error = exc
            finally:
                await charge.finish()

            if attempt >= settings.MODEL_MAX_RETRIES:
                break

            await asyncio.sleep(2 ** attempt)

        raise ModelServiceError(getattr(last_error, "code", None),
                                getattr(last_error, "status_code", None)) from last_error

    async def close(self) -> None:
        """关闭聊天模型 HTTP 客户端。"""

        if self._chat_client is not None:
            await self._chat_client.close()
            self._chat_client = None
        if self._ocr_client is not None:
            await self._ocr_client.close()
            self._ocr_client = None


# 🌟 修复点：替换为基于正则的稳健清洗逻辑
def clean_json(content: str) -> str:
    """提取模型返回内容中的 JSON 部分，忽略前后的废话。"""
    content = content.strip()

    # 使用正则非贪婪匹配 ```json 和 ``` 之间的内容
    match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 如果没找到代码块，可能模型直接返回了纯 JSON，尝试寻找首尾的 {} 或 []
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return content[start_idx:end_idx + 1]

    start_idx_arr = content.find("[")
    end_idx_arr = content.rfind("]")
    if start_idx_arr != -1 and end_idx_arr != -1 and end_idx_arr > start_idx_arr:
        return content[start_idx_arr:end_idx_arr + 1]

    return content


bailian_service = BailianService()
