"""Session-owned chat streaming. Cancelling closes the real upstream HTTP stream."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from urllib.parse import urlsplit

import httpx
import streamlit as st

from career_ui.api import operation_read_timeout
from career_ui.config import API_BASE_URL


class ChatTask:
    """A bounded worker with no Streamlit calls or shared user credentials."""

    def __init__(self, *, prompt, conversation_id, owner, token, read_timeout, retry=False, client_factory=None):
        self.prompt, self.owner = prompt, owner
        self._token = token
        self._read_timeout = read_timeout
        self._client_factory = client_factory or httpx.AsyncClient
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._loop = self._task = None
        self._state = {"conversation_id": conversation_id, "content": "", "status": "running",
                       "error": "", "unauthorized": False, "user_message_id": None, "message_id": None}
        self.retry = retry
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True, name="career-chat-stream")

    def start(self):
        self.thread.start()
        return self

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _update(self, **values):
        with self._lock:
            self._state.update(values)

    def cancel(self):
        self._cancelled.set()
        with self._lock:
            loop, task = self._loop, self._task
        if loop and task and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # The worker has already shut down.

    def _run(self):
        loop = asyncio.new_event_loop()
        task = loop.create_task(self._consume())
        with self._lock:
            self._loop, self._task = loop, task
        if self._cancelled.is_set():
            task.cancel()
        try:
            loop.run_until_complete(task)
            self._update(status="complete")
        except asyncio.CancelledError:
            self._update(status="stopped")
        except Exception as exc:
            message = str(exc) if isinstance(exc, ChatFailure) else "连接已中断，请稍后重试。"
            self._update(status="failed", error=message)
        finally:
            with self._lock:
                self._loop = self._task = None
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _check(self, response):
        if response.status_code == 401:
            self._update(unauthorized=True)
        if response.is_error:
            await response.aread()
            try:
                detail = response.json().get("detail", "请求暂时无法完成，请重试。")
            except ValueError:
                detail = "请求暂时无法完成，请重试。"
            raise ChatFailure(detail if isinstance(detail, str) else "请求暂时无法完成，请重试。")

    async def _consume(self):
        local = urlsplit(API_BASE_URL).hostname in ("localhost", "127.0.0.1", "::1")
        async with self._client_factory(headers={"Authorization": f"Bearer {self._token}"},
                                        timeout=httpx.Timeout(self._read_timeout, connect=5), trust_env=not local) as client:
            identifier = self.snapshot()["conversation_id"]
            if not identifier:
                created = await client.post(API_BASE_URL + "/api/chat/conversations", json={"title": self.prompt[:30]})
                await self._check(created)
                identifier = created.json()["id"]
                self._update(conversation_id=identifier)
            payload = {"conversation_id": identifier, "message": self.prompt, "stream": True}
            if self.retry:
                history = await client.get(API_BASE_URL + f"/api/chat/conversations/{identifier}/messages")
                await self._check(history)
                rows = history.json()
                user_index = next((i for i in range(len(rows) - 1, -1, -1) if rows[i]["role"] == "user"), None)
                if user_index is not None:
                    previous = rows[user_index]
                    if previous["content"] != self.prompt:
                        raise ChatFailure("对话已有新消息，请重新载入记录后再试。")
                    self._update(user_message_id=previous["id"])
                    complete = next((row for row in reversed(rows[user_index + 1:])
                                     if row["role"] == "assistant" and not row.get("incomplete")), None)
                    if complete:
                        self._update(content=complete["content"], message_id=complete["id"])
                        return  # Recover a persisted answer instead of paying to generate it again.
                    payload["retry_message_id"] = previous["id"]
            async with client.stream("POST", API_BASE_URL + "/api/chat/completions", json=payload) as response:
                await self._check(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    value = line[6:]
                    if value == "[DONE]":
                        return
                    try:
                        event = json.loads(value)
                    except ValueError:
                        continue
                    if "error" in event:
                        raise ChatFailure(event["error"].get("message", "回复暂时中断，请重试。"))
                    for key in ("conversation_id", "user_message_id", "message_id"):
                        if event.get(key):
                            self._update(**{key: event[key]})
                    choices = event.get("choices") or []
                    delta = choices[0].get("delta", {}).get("content", "") if choices else event.get("content", "")
                    if isinstance(delta, str) and delta:
                        with self._lock:
                            self._state["content"] += delta
                raise ChatFailure("连接在回复完成前中断，可以原地重试。")


class ChatFailure(Exception):
    pass


def cancel_active_chat():
    task = st.session_state.pop("chat_task", None)
    if task:
        task.cancel()
    st.session_state.pop("chat_retry", None)


def begin_chat(prompt, *, retry=False):
    prompt = prompt.strip()
    if not prompt:
        return False
    old = st.session_state.get("chat_task")
    if old:
        return False
    user = st.session_state.auth_user
    task = ChatTask(prompt=prompt, conversation_id=st.session_state.conversation_id,
                    owner=str(user.get("id") or user["username"]), token=st.session_state.access_token,
                    read_timeout=operation_read_timeout("/api/chat/completions"), retry=retry)
    if not retry:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.pop("chat_retry", None)
    st.session_state.chat_task = task.start()
    return True


@st.fragment(run_every=.25)
def render_live_chat():
    task = st.session_state.get("chat_task")
    if not task:
        return
    user = st.session_state.auth_user
    if task.owner != str(user.get("id") or user["username"]):
        cancel_active_chat()
        return
    state = task.snapshot()
    if state["conversation_id"]:
        st.session_state.conversation_id = state["conversation_id"]
    if state["status"] != "running":
        if state["unauthorized"]:
            from career_ui.session import reset_login_state
            reset_login_state()
        if state["user_message_id"]:
            for message in reversed(st.session_state.chat_messages):
                if message["role"] == "user":
                    message["id"] = state["user_message_id"]
                    break
        if state["content"]:
            st.session_state.chat_messages.append({"role": "assistant", "content": state["content"],
                                                   "id": state["message_id"], "incomplete": state["status"] != "complete"})
        if state["status"] in ("failed", "stopped"):
            st.session_state.chat_retry = {"prompt": task.prompt, "error": state["error"], "stopped": state["status"] == "stopped"}
        st.session_state.pop("chat_task", None)
        st.rerun(scope="app")
    with st.chat_message("assistant"):
        if state["content"]:
            st.markdown(state["content"])
        else:
            st.html('<div class="chat-thinking" role="status"><span></span>正在组织回复</div>')
        progress, stop = st.columns([5, 1.5], vertical_alignment="center")
        progress.caption(f"生成中 · {int(time.monotonic() - task.started)} 秒")
        if stop.button("停止生成", key="stop_chat_generation", icon=":material/stop:", type="tertiary"):
            task.cancel()


def render_chat_history():
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message.get("incomplete"):
                with st.expander("上次中断的回复"):
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])
    if st.session_state.get("chat_task"):
        render_live_chat()
    retry = st.session_state.get("chat_retry")
    if retry and not st.session_state.get("chat_task"):
        with st.container(key="chat_retry_notice"):
            description, action = st.columns([5, 1.5], vertical_alignment="center")
            description.caption("已停止生成，已产生的用量仍会结算。" if retry["stopped"] else retry["error"])
            if action.button("重新生成" if retry["stopped"] else "重试", key="retry_chat_generation", type="tertiary"):
                begin_chat(retry["prompt"], retry=True)
                st.rerun()
