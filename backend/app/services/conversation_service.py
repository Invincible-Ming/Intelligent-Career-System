"""Conversation persistence with ownership enforced on every operation."""
from __future__ import annotations
import uuid
from sqlalchemy import func, select, update
from sqlalchemy.orm import raiseload
from app.core.config import settings
from app.core.models import Conversation, Message


class ConversationService:
    async def create_conversation(self, *, session, owner_id: uuid.UUID, title='新对话'):
        conversation = Conversation(owner_id=owner_id, user_id=str(owner_id), title=title,
                                    model=settings.BAILIAN_CHAT_MODEL, meta_data={})
        session.add(conversation)
        await session.commit()
        return conversation

    async def get_conversation(self, *, session, owner_id, conversation_id):
        return (await session.execute(select(Conversation).options(raiseload(Conversation.messages)).where(
            Conversation.id == conversation_id, Conversation.owner_id == owner_id))).scalar_one_or_none()

    async def add_message(self, *, session, owner_id, conversation_id, role, content, tokens=None, metadata=None):
        if role not in ('user', 'assistant'):
            raise ValueError('不支持的消息角色')
        result = await session.execute(update(Conversation).where(
            Conversation.id == conversation_id, Conversation.owner_id == owner_id).values(
            message_count=Conversation.message_count + 1, updated_at=func.now()))
        if result.rowcount != 1:
            raise ValueError('对话不存在或不可访问')
        message = Message(conversation_id=conversation_id, role=role, content=content,
                          tokens=tokens, meta_data=metadata or {})
        session.add(message)
        await session.commit()
        return message

    async def save_context_summary(
            self, *, session, owner_id, conversation_id, summary, until_message_id
    ):
        """Atomically persist the rolling summary for an owned conversation."""
        result = await session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.owner_id == owner_id,
            )
            .values(
                context_summary=summary,
                summary_until_message_id=until_message_id,
                summary_version=Conversation.summary_version + 1,
                updated_at=func.now(),
            )
        )
        if result.rowcount != 1:
            raise ValueError("对话不存在或不可访问")
        await session.commit()

    async def list_conversations(self, *, session, owner_id, limit=50):
        return list((await session.execute(select(Conversation).options(raiseload(Conversation.messages)).where(
            Conversation.owner_id == owner_id).order_by(
            Conversation.pinned.desc(), Conversation.updated_at.desc()).limit(limit))).scalars())

    async def get_messages(self, *, session, owner_id, conversation_id, limit=40, offset=0):
        rows = (
            await session.execute(select(Message).join(Conversation, Conversation.id == Message.conversation_id).where(
                Message.conversation_id == conversation_id, Conversation.owner_id == owner_id).order_by(
                Message.created_at.desc(), Message.id.desc()).limit(limit).offset(offset))).scalars().all()
        return list(reversed(rows))

    async def update_conversation(self, *, session, owner_id, conversation_id, title=None, pinned=None):
        """按需更新标题或置顶状态；未传字段保持不变。"""
        conversation = await self.get_conversation(session=session, owner_id=owner_id, conversation_id=conversation_id)
        if conversation:
            if title is not None:
                conversation.title = title
            if pinned is not None:
                conversation.pinned = pinned
            await session.commit()
        return conversation

    async def delete_conversation(self, *, session, owner_id, conversation_id):
        conversation = await self.get_conversation(session=session, owner_id=owner_id, conversation_id=conversation_id)
        if not conversation:
            return False
        await session.delete(conversation)
        await session.commit()
        return True

    def messages_to_dict(self, messages):
        return [{'role': item.role, 'content': item.content} for item in messages if
                item.role in ('user', 'assistant') and not (item.meta_data or {}).get('incomplete')]


conversation_service = ConversationService()
