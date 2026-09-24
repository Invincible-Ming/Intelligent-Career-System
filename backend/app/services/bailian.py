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
from http import HTTPStatus
from typing import Any, AsyncGenerator, TypeVar  # 🌟 新增：导入 AsyncGenerator

import dashscope
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.limits import charge_model_call, validate_model_input, BudgetExceeded, model_request_timeout

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


def public_model_error(exc):
    if getattr(exc, 'code', None) == 'AllocationQuota.FreeTierOnly':
        return {'message': '百炼模型免费额度已用完，当前开启“用完即停”；请联系管理员处理模型额度。',
                'type': 'model_quota_exceeded'}
    return {'message': '回复生成失败，请稍后重试', 'type': 'server_error'}


class BailianService:
    """百炼聊天和 Embedding 服务。"""

    def __init__(self) -> None:
        self._chat_client: AsyncOpenAI | None = None

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

        response = await self._chat_retry(
            **request_params
        )

        if not response.choices:
            raise RuntimeError(
                "百炼聊天模型没有返回结果"
            )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "百炼聊天模型返回内容为空"
            )

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
        stream = await self.chat_client.chat.completions.create(
            model=model or settings.BAILIAN_CHAT_MODEL, messages=messages, temperature=temperature,
            stream=True, max_tokens=cap,
            timeout=model_request_timeout(),
        )
        consumed = 0
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    consumed += len(token.encode('utf-8'))
                    if consumed > cap * 16:
                        raise BudgetExceeded('模型输出超过预算')
                    yield token
        finally:
            await stream.close()

    async def ocr_image(
            self,
            base64_image: str,
            *,
            model: str | None = None,
            temperature: float = 0.1,
    ) -> str:
        """
        调用百炼多模态/OCR 模型提取图片文本。
        """
        model = model or settings.BAILIAN_OCR_MODEL
        if not base64_image:
            raise ValueError("图片 Base64 数据不能为空")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一个专业的文档解析助手。请精准提取这张图片里的所有文本内容。请严格遵循以下规则：1. 保持原有的段落结构；2. 如果图片中包含表格，请务必使用Markdown格式的表格输出；3. 不要输出任何除了提取文本之外的解释性废话。"
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

        response = await self._chat_retry(**request_params)

        if not response.choices:
            raise RuntimeError("百炼 OCR 模型没有返回结果")

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("百炼 OCR 模型返回内容为空")

        return content.strip()

    async def structured_chat(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            response_model: type[SchemaType],
            reasoning_effort: str | None = None,
    ) -> SchemaType:
        """
        调用聊天模型并将返回的 JSON 转换为 Pydantic 模型。

        如果第一次输出不符合 Schema，会要求模型修正一次。
        """

        schema = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "你必须只返回合法 JSON，"
                    "不要返回 Markdown 代码块或其他解释。\n"
                    f"返回内容必须符合以下 JSON Schema：{schema}"
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        for attempt in range(2):
            content = await self.chat(
                messages,
                json_mode=True,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
            )

            try:
                # 🌟 调用强化后的 clean_json
                return response_model.model_validate_json(
                    clean_json(content)
                )
            except ValidationError as exc:
                if attempt == 1:
                    raise RuntimeError(
                        "模型结构化输出校验失败："
                        f"{exc}"
                    ) from exc

                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": content,
                        },
                        {
                            "role": "user",
                            "content": (
                                "上面的 JSON 不符合 Schema。"
                                "请根据校验错误修正，并且只返回 JSON。\n"
                                f"校验错误：{exc}"
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

        validate_model_input([{"content": text} for text in cleaned_texts])
        response = await self._embedding_retry(
            cleaned_texts
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

        if len(vectors) != len(cleaned_texts):
            raise RuntimeError(
                "Embedding 返回数量与输入数量不一致："
                f"输入 {len(cleaned_texts)} 条，"
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

        return vectors

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
            **request_params: Any,
    ) -> Any:
        """聊天模型失败后进行简单重试。"""

        request_params.setdefault("max_tokens", settings.MODEL_MAX_OUTPUT_TOKENS)
        validate_model_input(request_params["messages"])
        last_error: Exception | None = None

        for attempt in range(
                settings.MODEL_MAX_RETRIES + 1
        ):
            charge_model_call()
            try:
                response = await (
                    self.chat_client
                    .chat
                    .completions
                    .create(**request_params, timeout=model_request_timeout())
                )
                for choice in response.choices:
                    content = choice.message.content or ''
                    if len(content.encode('utf-8')) > request_params['max_tokens'] * 16:
                        raise BudgetExceeded('模型输出超过预算')
                return response
            except Exception as exc:
                last_error = exc

                status = getattr(exc, "status_code", None)
                if (
                        status is not None and 400 <= status < 500 and status != 429) or attempt >= settings.MODEL_MAX_RETRIES:
                    break

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
