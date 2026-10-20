"""访问日志与链路 ID 中间件。

放在最外层，保证所有请求（含被限流、被拒绝的）都有 trace_id 可查。
"""
from __future__ import annotations

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import settings
from app.core.logging_setup import (
    REQUEST_ID_HEADER,
    actor_var,
    bind_request_id,
    log_event,
    request_id_var,
    request_start_var,
)

logger = logging.getLogger("app.access")

# 这些路径每次请求都会打，噪音大且无诊断价值。
_QUIET_PATHS = {"/health", "/favicon.ico", settings.API_PREFIX + "/auth/options"}


class AccessLogMiddleware:
    """给每个 HTTP 请求分配 trace_id，并记录耗时与状态码。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        incoming = headers.get(REQUEST_ID_HEADER.lower().encode())
        request_id = bind_request_id(
            incoming.decode(errors="replace") if incoming else None
        )

        # ContextVar 必须在子协程之外重置，否则会污染后续请求。
        start_token = request_start_var.set(time.perf_counter())
        actor_token = actor_var.set("-")

        status_code = 500
        started = time.perf_counter()

        async def tracked_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # 回写 trace_id，前端报错时可直接报这个值用于检索日志。
                raw_headers = list(message.get("headers", []))
                raw_headers.append(
                    (
                        REQUEST_ID_HEADER.lower().encode(),
                        request_id.encode()[:64],
                    )
                )
                message = {**message, "headers": raw_headers}
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                f"{scope.get('method', '?')} {scope.get('path', '?')} 未处理异常: {type(exc).__name__}",
                method=scope.get("method", "-"),
                path=scope.get("path", "-"),
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error=type(exc).__name__,
            )
            raise
        else:
            self._record(scope, status_code, started)
        finally:
            # ContextVar 是 per-task的：ASGI 每次请求都在新 task 里跑，
            # 因此只需重置令牌，作用域自然随请求结束而消失。
            request_start_var.reset(start_token)
            actor_var.reset(actor_token)

    @staticmethod
    def _record(scope: Scope, status_code: int, started: float) -> None:
        if not settings.LOG_ACCESS_ENABLED:
            return

        path = scope.get("path", "-")
        if path in _QUIET_PATHS:
            return

        duration_ms = (time.perf_counter() - started) * 1000
        level = logging.INFO
        if status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        #慢请求单独抬到 WARNING，便于快速定位瓶颈。
        elif duration_ms > 3000:
            level = logging.WARNING

        log_event(
            logger,
            level,
            f"{scope.get('method', '?')} {path} -> {status_code}",
            method=scope.get("method", "-"),
            path=path,
            status=status_code,
            duration_ms=round(duration_ms, 2),
            query=scope.get("query_string", b"").decode(errors="replace")[:200] or None,
            client=client_ip(scope),
        )


def client_ip(scope: Scope) -> str:
    """取客户端 IP。

    不信任 X-Forwarded-For 的任意值——它可被伪造，
    仅取第一个段并截断，用于日志定位足够。
    """

    headers = dict(scope.get("headers", []))
    forwarded = headers.get(b"x-forwarded-for")
    if forwarded:
        first = forwarded.split(b",")[0].strip()
        if first:
            return first.decode(errors="replace")[:45]
    client = scope.get("client")
    if client:
        return str(client[0])[:45]
    return "-"