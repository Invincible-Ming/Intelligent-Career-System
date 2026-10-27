"""Cancellation before response headers and before the first body iteration."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import ClientDisconnect

from app.api import chat_api


@pytest.mark.asyncio
async def test_disconnect_cancels_preparation_before_any_stream_headers():
    stopped = asyncio.Event()

    async def prepare(*args):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    with patch.object(chat_api, '_prepare_chat_response', prepare):
        response = await chat_api.chat_completions(chat_api.ChatRequest(message='Python?'), None, None, request)
    assert response.status_code == 499
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_failed_header_send_still_closes_an_unstarted_stream():
    cleanup = AsyncMock()
    started = []

    async def body():
        started.append(True)
        yield 'never sent'

    response = chat_api.ClosingChatResponse(body(), on_close=cleanup)
    with pytest.raises(ClientDisconnect):
        await response({'type': 'http', 'asgi': {'spec_version': '2.4'}},
                       AsyncMock(), AsyncMock(side_effect=OSError('client gone')))
    assert not started
    cleanup.assert_awaited_once()
