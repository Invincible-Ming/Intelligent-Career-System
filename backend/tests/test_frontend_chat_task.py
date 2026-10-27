"""Async stream cancellation and retry recovery; only synthetic HTTP."""
import asyncio
import json
import threading

import httpx
from career_ui.chat import ChatTask


def factory(handler):
    return lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


def packet(value):
    return ('data: ' + json.dumps(value) + '\n\n').encode()


class PausedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.waiting, self.closed = threading.Event(), threading.Event()

    async def __aiter__(self):
        yield packet({'conversation_id': 'c', 'user_message_id': 'q', 'choices': [{'delta': {'content': 'Partial'}}]})
        self.waiting.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed.set()


def test_stop_cancels_a_pending_read_and_closes_the_upstream():
    stream, headers = PausedStream(), []

    def handler(request):
        headers.append(request.headers['authorization'])
        return httpx.Response(200, stream=stream)

    task = ChatTask(prompt='Python?', conversation_id='c', owner='alice', token='alice-only',
                    read_timeout=30, client_factory=factory(handler)).start()
    assert stream.waiting.wait(2)
    task.cancel()
    task.thread.join(2)
    assert not task.thread.is_alive()
    assert stream.closed.is_set()
    assert task.snapshot()['status'] == 'stopped'
    assert task.snapshot()['content'] == 'Partial'
    assert headers == ['Bearer alice-only']


def test_retry_recovers_a_persisted_answer_without_another_generation():
    paths = []

    def handler(request):
        paths.append(request.method)
        return httpx.Response(200, json=[{'id': 'q', 'role': 'user', 'content': 'Python?'},
            {'id': 'a', 'role': 'assistant', 'content': 'Existing answer', 'incomplete': False}])

    task = ChatTask(prompt='Python?', conversation_id='c', owner='alice', token='alice-only',
                    read_timeout=5, retry=True, client_factory=factory(handler)).start()
    task.thread.join(2)
    assert task.snapshot()['status'] == 'complete'
    assert task.snapshot()['content'] == 'Existing answer'
    assert paths == ['GET']


def test_retry_reuses_the_confirmed_message_without_sending_user_identity():
    payloads = []

    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json=[{'id': 'q', 'role': 'user', 'content': 'Python?'}])
        payloads.append(json.loads(request.content))
        return httpx.Response(200, content=packet({'choices': [{'delta': {'content': 'new answer'}}]}) + b'data: [DONE]\n\n')

    task = ChatTask(prompt='Python?', conversation_id='c', owner='alice', token='alice-only',
                    read_timeout=5, retry=True, client_factory=factory(handler)).start()
    task.thread.join(2)
    assert task.snapshot()['status'] == 'complete'
    assert payloads == [{'conversation_id': 'c', 'message': 'Python?', 'stream': True, 'retry_message_id': 'q'}]


def test_retry_rejects_a_stale_question_after_the_conversation_changes():
    def handler(request):
        assert request.method == 'GET'
        return httpx.Response(200, json=[{'id': 'other', 'role': 'user', 'content': 'New question'}])

    task = ChatTask(prompt='Python?', conversation_id='c', owner='alice', token='alice-only',
                    read_timeout=5, retry=True, client_factory=factory(handler)).start()
    task.thread.join(2)
    assert task.snapshot()['status'] == 'failed'
    assert '已有新消息' in task.snapshot()['error']


def test_simultaneous_workers_keep_credentials_and_results_separate():
    def handler(request):
        return httpx.Response(200, content=packet({'choices': [{'delta': {'content': request.headers['authorization']}}]}) + b'data: [DONE]\n\n')

    tasks = [ChatTask(prompt='Python?', conversation_id='c', owner=name, token=name,
                      read_timeout=5, client_factory=factory(handler)).start() for name in ('alice', 'bob')]
    for task in tasks:
        task.thread.join(2)
        assert task.snapshot()['content'] == f'Bearer {task.owner}'
