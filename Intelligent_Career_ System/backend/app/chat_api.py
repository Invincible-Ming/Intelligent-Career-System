"""
SSE 流式对话 API 端点（支持持久化）。
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.bailian import bailian_service
from app.conversation_service import conversation_service
from app.database import get_db
from app.models import Conversation, Message

router = APIRouter(prefix="/chat", tags=["对话"])

DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


class ChatRequest(BaseModel):
    """对话请求。"""

    conversation_id: uuid.UUID | None = Field(
        default=None,
        description="对话 ID（不传则创建新对话）",
    )
    message: str = Field(
        description="用户消息",
        min_length=1,
    )
    user_id: str | None = Field(
        default=None,
        description="用户标识",
    )
    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
        description="温度参数",
    )
    stream: bool = Field(
        default=True,
        description="是否使用流式输出",
    )


class ConversationResponse(BaseModel):
    """对话响应。"""

    conversation_id: uuid.UUID
    message_id: uuid.UUID
    content: str
    role: str = "assistant"


class ConversationListItem(BaseModel):
    """对话列表项。"""

    id: uuid.UUID
    title: str
    message_count: int
    model: str
    updated_at: str


@router.post("/completions")
async def chat_completions(
    request: ChatRequest,
    session: DatabaseSession,
):
    """
    对话补全接口，支持流式和非流式，自动持久化对话历史。
    
    流式返回：Server-Sent Events (SSE)
    非流式返回：JSON
    """

    # 1. 获取或创建对话
    if request.conversation_id:
        conversation = await conversation_service.get_conversation(
            session=session,
            conversation_id=request.conversation_id,
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="对话不存在",
            )
    else:
        # 创建新对话
        conversation = await conversation_service.create_conversation(
            session=session,
            user_id=request.user_id,
            title="新对话",
        )

    # 2. 保存用户消息
    user_message = await conversation_service.add_message(
        session=session,
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )

    # 3. 获取对话历史
    messages = await conversation_service.get_messages(
        session=session,
        conversation_id=conversation.id,
    )

    messages_dict = conversation_service.messages_to_dict(messages)

    # 4. 调用 LLM
    if request.stream:
        # 流式响应
        return StreamingResponse(
            stream_chat_with_persistence(
                session=session,
                conversation_id=conversation.id,
                messages=messages_dict,
                temperature=request.temperature,
            ),
            media_type="text/event-stream",
        )
    else:
        # 非流式响应
        content = await bailian_service.chat(
            messages=messages_dict,
            temperature=request.temperature,
        )

        # 保存助手消息
        assistant_message = await conversation_service.add_message(
            session=session,
            conversation_id=conversation.id,
            role="assistant",
            content=content,
        )

        return ConversationResponse(
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            content=content,
        )


async def stream_chat_with_persistence(
    *,
    session: AsyncSession,
    conversation_id: uuid.UUID,
    messages: list[dict[str, str]],
    temperature: float,
):
    """SSE 流式生成器（支持持久化）。"""

    full_content = ""

    try:
        async for chunk in bailian_service.chat_stream(
            messages=messages,
            temperature=temperature,
        ):
            full_content += chunk

            # SSE 格式
            data = {
                "conversation_id": str(conversation_id),
                "choices": [
                    {
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 保存助手消息
        assistant_message = await conversation_service.add_message(
            session=session,
            conversation_id=conversation_id,
            role="assistant",
            content=full_content,
        )

        # 发送结束标记
        end_data = {
            "conversation_id": str(conversation_id),
            "message_id": str(assistant_message.id),
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        yield f"data: {json.dumps(end_data)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as exc:
        # 发送错误
        error_data = {
            "error": {
                "message": str(exc),
                "type": "server_error",
            }
        }

        yield f"data: {json.dumps(error_data)}\n\n"


@router.get("/conversations")
async def list_conversations(
    session: DatabaseSession,
    user_id: str | None = None,
    limit: int = 50,
) -> list[ConversationListItem]:
    """列出对话列表。"""

    conversations = await conversation_service.list_conversations(
        session=session,
        user_id=user_id,
        limit=limit,
    )

    return [
        ConversationListItem(
            id=conv.id,
            title=conv.title,
            message_count=conv.message_count,
            model=conv.model,
            updated_at=conv.updated_at.isoformat(),
        )
        for conv in conversations
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID,
    session: DatabaseSession,
):
    """获取对话详情（包含所有消息）。"""

    conversation = await conversation_service.get_conversation(
        session=session,
        conversation_id=conversation_id,
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="对话不存在",
        )

    return {
        "id": conversation.id,
        "title": conversation.title,
        "model": conversation.model,
        "message_count": conversation.message_count,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in conversation.messages
        ],
    }


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: uuid.UUID,
    session: DatabaseSession,
    title: str,
):
    """更新对话标题。"""

    conversation = await conversation_service.update_conversation_title(
        session=session,
        conversation_id=conversation_id,
        title=title,
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="对话不存在",
        )

    return {
        "id": conversation.id,
        "title": conversation.title,
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: uuid.UUID,
    session: DatabaseSession,
):
    """删除对话。"""

    success = await conversation_service.delete_conversation(
        session=session,
        conversation_id=conversation_id,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail="对话不存在",
        )

    return {"message": "对话已删除"}
