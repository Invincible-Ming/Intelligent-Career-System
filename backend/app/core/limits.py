"""Request-wide budgets, including streams and parallel agent calls."""
from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass

from app.core.config import settings


class BudgetExceeded(BaseException):
    """Must not be swallowed by broad Exception-based retries/fallbacks."""


from langchain_core.callbacks import BaseCallbackHandler


class ModelBudgetCallback(BaseCallbackHandler):
    raise_error = True
    run_inline = True

    def on_tool_start(self, serialized, input_str, **kwargs):
        charge_model_call()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        charge_model_call()
        validate_model_input([{'content': message.content} for batch in messages for message in batch])


class RequestBodyTooLarge(Exception):
    pass


@dataclass
class InvocationBudget:
    deadline: float
    max_calls: int
    calls: int = 0
    model_timeout: float | None = None

    def charge(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("本次操作已达到总耗时上限")
        if self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"本次操作已达到模型调用次数上限（{self.max_calls} 次），请稍后重试或拆分任务"
            )
        self.calls += 1


invocation_budget: ContextVar[InvocationBudget | None] = ContextVar("invocation_budget", default=None)
# 当前请求的操作者是否管理员（由认证依赖注入）；管理员不限模型调用次数。
operation_actor_is_admin: ContextVar[bool] = ContextVar("operation_actor_is_admin", default=False)


def set_operation_actor(*, is_admin: bool) -> None:
    operation_actor_is_admin.set(is_admin)


def model_request_timeout():
    budget = invocation_budget.get()
    ceiling = budget.model_timeout if budget and budget.model_timeout is not None else settings.MODEL_TIMEOUT
    if budget is not None:
        return max(.001, min(ceiling, budget.deadline - time.monotonic()))
    return ceiling


def charge_model_call():
    # 管理员操作不限模型调用次数（总耗时上限仍然生效）。
    if operation_actor_is_admin.get():
        return
    budget = invocation_budget.get()
    if budget is not None:
        budget.charge()


def check_operation_deadline():
    budget = invocation_budget.get()
    if budget is not None and time.monotonic() >= budget.deadline:
        raise BudgetExceeded('本次操作已达到总耗时上限')


def message_cost(message):
    # Provider tokenizer is not distributed with this project. UTF-8 bytes
    # plus envelope overhead provide a deliberately conservative text budget.
    return len(message.get("content", "").encode("utf-8")) + 32


def bound_history(messages, system_prompt, budget=None):
    ceiling = budget or settings.CHAT_CONTEXT_BUDGET
    system = {"role": "system", "content": system_prompt}
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("对话必须以当前用户输入结束")
    latest = messages[-1]
    spent = message_cost(system) + message_cost(latest)
    if spent > ceiling:
        raise ValueError("当前输入超过上下文预算，请缩短内容")
    selected = [latest]
    # Only retain whole user/assistant turns; never take instructions from
    # a persisted system/tool role or leave an orphan assistant message.
    index = len(messages) - 2
    while index >= 1:
        assistant, user = messages[index], messages[index - 1]
        if assistant["role"] != "assistant" or user["role"] != "user":
            index -= 1
            continue
        cost = message_cost(user) + message_cost(assistant)
        if spent + cost > ceiling:
            break
        selected[0:0] = [user, assistant]
        spent += cost
        index -= 2
    return [system, *selected]


def validate_model_input(messages):
    text_cost = sum(message_cost(message) for message in messages if isinstance(message.get("content"), str))
    if text_cost > settings.MODEL_INPUT_BUDGET:
        raise BudgetExceeded("模型输入超过预算，请缩短资料")


def operation_timeout(path):
    prefix = settings.API_PREFIX
    if path == f"{prefix}/chat/completions":
        return settings.CHAT_TOTAL_TIMEOUT
    if path == f"{prefix}/documents/upload":
        return settings.UPLOAD_TOTAL_TIMEOUT
    if path == f"{prefix}/match" or path.startswith(f"{prefix}/match/"):
        return settings.MATCH_TOTAL_TIMEOUT
    if path.startswith((f"{prefix}/interview", f"{prefix}/learning-plan", f"{prefix}/evaluation")):
        return settings.ANALYSIS_TOTAL_TIMEOUT
    return settings.API_TOTAL_TIMEOUT


def heavy_operation(path, method):
    return method == "POST" and path.startswith(tuple(settings.API_PREFIX + suffix for suffix in
                                                      ("/chat/completions", "/documents/upload", "/search", "/match",
                                                       "/interview", "/learning-plan", "/evaluation/experiments")))


class SecurityLimitsMiddleware:
    """ASGI-level deadline wraps the entire response, including SSE bodies."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        limit = (settings.max_upload_size_bytes + 65536 if path == settings.API_PREFIX + "/documents/upload"
                 else settings.JSON_BODY_MAX_BYTES)
        started = ended = disconnected = False
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError()
        except ValueError:
            from starlette.responses import JSONResponse
            return await JSONResponse({"detail": "非法请求长度"}, status_code=400)(scope, receive, send)
        if declared > limit:
            from starlette.responses import JSONResponse
            return await JSONResponse({"detail": "请求体超过大小限制"}, status_code=413)(scope, receive, send)
        consumed = 0

        async def bounded_receive():
            nonlocal consumed, disconnected
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise RequestBodyTooLarge()
            elif message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def tracked_send(message):
            nonlocal started, ended
            if message["type"] == "http.response.start":
                started = True
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                ended = True
            await send(message)

        timeout = operation_timeout(path)
        is_match = path == settings.API_PREFIX + "/match" or path.startswith(settings.API_PREFIX + "/match/")
        token = invocation_budget.set(InvocationBudget(
            time.monotonic() + timeout, settings.MAX_MODEL_CALLS_PER_OPERATION,
            model_timeout=settings.MATCH_MODEL_TIMEOUT if is_match else None,
        ))
        try:
            try:
                async with asyncio.timeout(timeout):
                    await self.app(scope, bounded_receive, tracked_send)
            except BaseException as failure:
                def find_limit(error):
                    if isinstance(error, (TimeoutError, BudgetExceeded)):
                        return error
                    if isinstance(error, BaseExceptionGroup):
                        leaves = [find_limit(child) for child in error.exceptions]
                        return next((leaf for leaf in leaves if leaf is not None), None)
                    return None

                exc = find_limit(failure)
                if exc is None:
                    raise
                message = "本次操作超过总耗时限制，请稍后重试" if isinstance(exc, TimeoutError) else str(exc)
                if ended or disconnected:
                    return
                if not started:
                    from starlette.responses import JSONResponse
                    await JSONResponse({"detail": message}, status_code=504 if isinstance(exc, TimeoutError) else 429)(
                        scope, receive, send)
                else:
                    if path == settings.API_PREFIX + "/chat/completions":
                        body = "data: " + json.dumps({"error": {"message": message, "type": "operation_limit"}},
                                                     ensure_ascii=False) + "\n\ndata: [DONE]\n\n"
                    else:
                        body = "event: error\ndata: " + json.dumps({"message": message}, ensure_ascii=False) + "\n\n"
                    await send({"type": "http.response.body", "body": body.encode(), "more_body": False})
        finally:
            invocation_budget.reset(token)
