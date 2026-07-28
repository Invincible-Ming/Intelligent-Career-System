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
from http import HTTPStatus
from typing import Any, TypeVar

import dashscope
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings

SchemaType = TypeVar(
    "SchemaType",
    bound=BaseModel,
)


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
    ) -> str:
        """调用百炼聊天模型。"""

        if not messages:
            raise ValueError("messages 不能为空")

        request_params: dict[str, Any] = {
            "model": settings.BAILIAN_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
        }

        if json_mode:
            request_params["response_format"] = {
                "type": "json_object",
            }

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

        return content.strip()

    async def chat_stream(
            self,
            messages: list[dict[str, str]],
            *,
            temperature: float = 0.2,
    ):
        """
        流式调用百炼聊天模型。
        返回异步生成器，逐块产出内容。
        """

        if not messages:
            raise ValueError("messages 不能为空")

        request_params = {
            "model": settings.BAILIAN_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "stream": True,  # 启用流式
        }

        stream = await self.chat_client.chat.completions.create(
            **request_params
        )

        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    async def structured_chat(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            response_model: type[SchemaType],
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
            )

            try:
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

        last_error: Exception | None = None

        for attempt in range(
                settings.MODEL_MAX_RETRIES + 1
        ):
            try:
                return await (
                    self.chat_client
                    .chat
                    .completions
                    .create(**request_params)
                )
            except Exception as exc:
                last_error = exc

                if attempt >= settings.MODEL_MAX_RETRIES:
                    break

                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            f"百炼聊天模型调用失败：{last_error}"
        ) from last_error

    async def _embedding_retry(
            self,
            texts: list[str],
    ) -> Any:
        """调用 DashScope Embedding，并进行简单重试。"""

        last_error: Exception | None = None

        for attempt in range(
                settings.MODEL_MAX_RETRIES + 1
        ):
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

                last_error = RuntimeError(
                    f"{error_code} - {error_message}"
                )

            except Exception as exc:
                last_error = exc

            if attempt >= settings.MODEL_MAX_RETRIES:
                break

            await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            "百炼 Embedding 调用失败："
            f"{last_error}"
        ) from last_error

    async def close(self) -> None:
        """关闭聊天模型 HTTP 客户端。"""

        if self._chat_client is not None:
            await self._chat_client.close()
            self._chat_client = None


def clean_json(content: str) -> str:
    """移除模型偶尔返回的 Markdown JSON 代码块。"""

    content = content.strip()

    if not content.startswith("```"):
        return content

    lines = content.splitlines()

    if lines and lines[0].startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


bailian_service = BailianService()
