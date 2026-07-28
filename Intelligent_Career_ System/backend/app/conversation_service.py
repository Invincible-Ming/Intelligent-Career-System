"""
对话历史持久化服务。
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message


class ConversationService:
    """对话历史管理服务。"""

    async def create_conversation(
        self,
        *,
        session: AsyncSession,
        user_id: str | None = None,
        title: str = "新对话",
        model: str = "qwen-plus",
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """创建新对话会话。"""

        conversation = Conversation(
            user_id=user_id,
            title=title,
            model=model,
            metadata=metadata or {},
        )

        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)

        return conversation

    async def add_message(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """向对话添加消息。"""

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata or {},
        )

        session.add(message)

        # 更新对话消息计数和更新时间
        conversation = await session.get(Conversation, conversation_id)
        if conversation:
            conversation.message_count += 1

        await session.commit()
        await session.refresh(message)

        return message

    async def get_conversation(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> Conversation | None:
        """获取对话（包含所有消息）。"""

        result = await session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
        )

        conversation = result.scalar_one_or_none()

        if conversation:
            # 预加载消息
            await session.refresh(conversation, ["messages"])

        return conversation

    async def list_conversations(
        self,
        *,
        session: AsyncSession,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[Conversation]:
        """列出对话列表。"""

        query = select(Conversation).order_by(
            Conversation.updated_at.desc()
        )

        if user_id:
            query = query.where(Conversation.user_id == user_id)

        query = query.limit(limit)

        result = await session.execute(query)
        return list(result.scalars())

    async def get_messages(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[Message]:
        """获取对话的所有消息。"""

        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )

        return list(result.scalars())

    async def update_conversation_title(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        title: str,
    ) -> Conversation | None:
        """更新对话标题。"""

        conversation = await session.get(Conversation, conversation_id)

        if conversation:
            conversation.title = title
            await session.commit()
            await session.refresh(conversation)

        return conversation

    async def delete_conversation(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> bool:
        """删除对话（级联删除所有消息）。"""

        conversation = await session.get(Conversation, conversation_id)

        if conversation:
            await session.delete(conversation)
            await session.commit()
            return True

        return False

    def messages_to_dict(
        self,
        messages: list[Message],
    ) -> list[dict[str, str]]:
        """将消息列表转换为 API 格式。"""

        return [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in messages
        ]


conversation_service = ConversationService()
