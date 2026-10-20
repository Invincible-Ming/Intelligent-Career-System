"""结构化日志与请求链路 ID。

解决的问题：原来只有 `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
这种纯文本日志，出现「用户说匹配失败了」时无法定位到具体是哪次请求、
哪个用户、哪个节点出了问题。

做法：
- 每个请求生成 trace_id，通过 X-Request-ID 头透传（上游有则沿用）；
- trace_id 存进 ContextVar，随日志记录自动带出，无需逐层传参；
- 日志行尾追加结构化字段，便于 grep 与后续接入日志采集。
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"

# 当前请求的链路 ID。SSE 长连接与后台任务会继承所在上下文的值。
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# 请求开始时刻，用于计算总耗时。
request_start_var: ContextVar[float] = ContextVar("request_start", default=0.0)

# 当前操作者，仅用于日志；绝不写入敏感内容。
actor_var: ContextVar[str] = ContextVar("actor", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def bind_request_id(value: str | None) -> str:
    """沿用上游传入的 trace_id，或生成一个新的。"""

    candidate = (value or "").strip()
    # 只接受有限的字符集，避免日志注入与超长字段。
    if candidate and len(candidate) <= 64 and all(
        char.isalnum() or char in "-_." for char in candidate
    ):
        rid = candidate
    else:
        rid = new_request_id()

    request_id_var.set(rid)
    return rid


def current_request_id() -> str:
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    """给每条日志补上链路 ID 与操作者。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        record.actor = actor_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """单行 JSON 输出，便于日志系统直接采集。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        actor = getattr(record, "actor", "-")
        if actor and actor != "-":
            payload["actor"] = actor

        # 附加的结构化上下文（node、duration_ms、status 等）。
        for key in getattr(record, "extra_fields", ()):  # pragma: no branch
            payload[key] = getattr(record, key, None)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """带 trace_id 的可读文本，保留给人看。"""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        request_id = getattr(record, "request_id", "-")
        actor = getattr(record, "actor", "-")
        tail = f"trace={request_id}"
        if actor and actor != "-":
            tail += f" actor={actor}"
        return f"{base}  [{tail}]"


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """初始化根日志。

    幂等：重复调用不会叠加 handler。
    """

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            TextFormatter(
                "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root.addHandler(handler)
    root.setLevel(level)

    # 第三方库默认太吵，只留警告以上。
    for noisy in (
        "httpx",
        "httpx2",  # 部分环境装的是这个包名
        "httpcore",
        "urllib3",
        "pymilvus",
        "openai",
        "dashscope",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: object,
) -> None:
    """记录一条带结构化字段的日志。

    字段值会自动附加到 LogRecord，供 JsonFormatter 采集。
    """

    if not logger.isEnabledFor(level):
        return

    extra = {"extra_fields": tuple(fields.keys())}
    for key, value in fields.items():
        extra[key] = value
    logger.log(level, message, extra=extra)


def timed(message: str = "处理完成"):
    """上下文管理器：记录进入与耗时。

    异常时自动记ERROR 并原样抛出，不改变控制流。
    """

    class _Timer:
        def __init__(self):
            self.logger = logging.getLogger("app.timing")
            self.started = 0.0

        def __enter__(self):
            self.started = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            duration_ms = (time.perf_counter() - self.started) * 1000
            if exc_type is None:
                log_event(
                    self.logger,
                    logging.INFO,
                    message,
                    duration_ms=round(duration_ms, 2),
                )
            else:
                log_event(
                    self.logger,
                    logging.ERROR,
                    f"{message} 失败: {exc_type.__name__}",
                    duration_ms=round(duration_ms, 2),
                    error=exc_type.__name__,
                )
            return False

    return _Timer()