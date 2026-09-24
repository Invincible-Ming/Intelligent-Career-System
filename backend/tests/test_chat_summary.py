"""Rolling chat-summary tests; no cloud model or live database required."""

import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import chat_api
from app.core.config import settings


def make_rows(count):
    rows = []
    for index in range(count):
        rows.append(SimpleNamespace(
            id=uuid.uuid4(),
            role='user' if index % 2 == 0 else 'assistant',
            content=f'消息 {index}',
            meta_data={},
        ))
    return rows


class ChatSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_model_runs_and_keeps_recent_messages(self):
        rows = make_rows(settings.CHAT_SUMMARY_TRIGGER_MESSAGES)
        conversation = SimpleNamespace(
            id=uuid.uuid4(),
            context_summary=None,
            summary_until_message_id=None,
        )
        summary_call = AsyncMock(return_value='用户正在搭建求职系统，待完成对话摘要。')
        save_summary = AsyncMock()

        with patch.object(chat_api.bailian_service, 'chat', summary_call), \
                patch.object(chat_api.conversation_service, 'save_context_summary', save_summary):
            summary, context_rows = await chat_api._maybe_compact_context(
                session=object(),
                owner_id=uuid.uuid4(),
                conversation=conversation,
                history_rows=rows,
            )

        summary_call.assert_awaited_once()
        self.assertEqual(summary_call.await_args.kwargs['model'], settings.BAILIAN_SUMMARY_MODEL)
        self.assertEqual(summary, '用户正在搭建求职系统，待完成对话摘要。')
        self.assertEqual(len(context_rows), settings.CHAT_SUMMARY_KEEP_MESSAGES)
        save_summary.assert_awaited_once()
        self.assertEqual(save_summary.await_args.kwargs['until_message_id'], rows[-9].id)

    async def test_existing_summary_only_sends_messages_after_boundary(self):
        rows = make_rows(12)
        boundary = rows[3].id
        conversation = SimpleNamespace(
            id=uuid.uuid4(),
            context_summary='已有摘要',
            summary_until_message_id=boundary,
        )

        with patch.object(chat_api.bailian_service, 'chat', AsyncMock()) as summary_call:
            summary, context_rows = await chat_api._maybe_compact_context(
                session=object(),
                owner_id=uuid.uuid4(),
                conversation=conversation,
                history_rows=rows,
            )

        summary_call.assert_not_awaited()
        self.assertEqual(summary, '已有摘要')
        self.assertEqual([row.id for row in context_rows], [row.id for row in rows[4:]])


if __name__ == '__main__':
    unittest.main()
