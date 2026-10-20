"""日志与链路 ID 测试。

trace_id 是排查线上问题的第一入口，必须保证：
- 每个请求都能拿到，且互不串号；
- 上游传入的合法 ID 被沿用，非法值被拒绝（日志注入防护）；
- 业务日志与访问日志共享同一个 ID；
- 未处理异常必须被记录，不能静默消失。
"""
import logging
import unittest

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.core.access_log import AccessLogMiddleware, client_ip
from app.core.logging_setup import (
    REQUEST_ID_HEADER,
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    bind_request_id,
    configure_logging,
    current_request_id,
    log_event,
)


async def _whoami(request):
    log_event(
        logging.getLogger("app.test"),
        logging.INFO,
        "业务日志",
        node="verify_agent",
    )
    return JSONResponse({"request_id": current_request_id()})


async def _boom(request):
    raise RuntimeError("故意的未处理异常")


async def _slow(request):
    return JSONResponse({"ok": True})


class TraceIdTests(unittest.TestCase):
    def setUp(self):
        configure_logging(level="INFO", json_output=False)
        app = Starlette(routes=[
            Route("/whoami", _whoami),
            Route("/boom", _boom),
            Route("/health", _slow),
        ])
        app.add_middleware(AccessLogMiddleware)
        # raise_server_exceptions=False 才能观察到 500 响应
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_trace_id_returned_in_header_and_body(self):
        response = self.client.get("/whoami")

        header_id = response.headers.get(REQUEST_ID_HEADER)
        self.assertIsNotNone(header_id)
        self.assertEqual(header_id, response.json()["request_id"])

    def test_each_request_gets_unique_id(self):
        first = self.client.get("/whoami").json()["request_id"]
        second = self.client.get("/whoami").json()["request_id"]

        self.assertNotEqual(first, second, "不同请求不得复用 trace_id")

    def test_upstream_id_is_reused(self):
        response = self.client.get("/whoami", headers={REQUEST_ID_HEADER: "upstream-abc123"})

        self.assertEqual(response.json()["request_id"], "upstream-abc123")

    def test_malicious_id_is_replaced(self):
        """含空格/分号的 ID 必须被拒绝，防止日志注入与字段污染。"""

        response = self.client.get("/whoami", headers={REQUEST_ID_HEADER: "bad id; DROP TABLE"})

        self.assertNotEqual(response.json()["request_id"], "bad id; DROP TABLE")

    def test_overlong_id_is_replaced(self):
        response = self.client.get("/whoami", headers={REQUEST_ID_HEADER: "x" * 200})

        self.assertLessEqual(len(response.json()["request_id"]), 64)

    def test_unhandled_exception_is_logged(self):
        with self.assertLogs("app.access", level="ERROR") as captured:
            response = self.client.get("/boom")

        self.assertEqual(response.status_code, 500)
        # 未处理异常必须留下 ERROR 日志。
        # 注意：assertLogs 会替换 handler、绕过 RequestIdFilter，
        # 因此这里不断言 trace=；trace 由下方test_trace_appears_in_formatted_line 覆盖。
        self.assertTrue(any("RuntimeError" in line for line in captured.output))

    def test_trace_appears_in_formatted_line(self):
        """真实 handler 下日志行必须带 trace=xxx，否则线上无法按ID 检索。"""

        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(TextFormatter("%(message)s"))

        logger = logging.getLogger("app.tracecheck")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            bind_request_id("fixed-trace-id")
            logger.info("探针消息")
        finally:
            logger.handlers.clear()

        output = stream.getvalue()
        self.assertIn("probe", output.replace("探针", "probe"))
        self.assertIn("trace=fixed-trace-id", output)

    def test_quiet_paths_skipped(self):
        """/health 这类高频无诊断价值的路径不应刷访问日志。"""

        # 不额外挂 handler：直接用 assertLogs/assertNoLogs 观察 app.access。
        with self.assertNoLogs("app.access", level="INFO"):
            self.client.get("/health")

        # 对照：普通路径必须记录
        with self.assertLogs("app.access", level="INFO") as captured:
            self.client.get("/whoami")

        self.assertTrue(any("/whoami" in line for line in captured.output))


class BindRequestIdTests(unittest.TestCase):
    def test_generates_when_missing(self):
        self.assertTrue(bind_request_id(None))

    def test_accepts_dotted_dashed_form(self):
        self.assertEqual(bind_request_id("abc-123_x.y"), "abc-123_x.y")

    def test_rejects_non_alphanumeric(self):
        for bad in ("a b", "a;b", "a\nb", "a'", "a\\b", 'a"b'):
            self.assertNotEqual(bind_request_id(bad), bad)

    def test_strips_whitespace(self):
        self.assertEqual(bind_request_id("  spaced  "), "spaced")


class JsonFormatterTests(unittest.TestCase):
    def _record(self, message, exc_info=None, **fields):
        # LogRecord 构造器不接 extra；标准库是在 Logger.makeRecord 里
        # 用 setattr 合并的，这里照做。
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=exc_info,
        )
        for key, value in fields.items():
            setattr(record, key, value)
        if fields:
            record.extra_fields = tuple(fields.keys())
        return record

    def test_contains_core_fields(self):
        import json

        payload = json.loads(JsonFormatter().format(self._record("消息")))

        self.assertIn("msg", payload)
        self.assertIn("request_id", payload)
        self.assertIn("level", payload)
        self.assertEqual(payload["msg"], "消息")

    def test_extra_fields_are_collected(self):
        import json

        payload = json.loads(JsonFormatter().format(
            self._record("节点耗时", node="match_agent", duration_ms=12.5)
        ))

        self.assertEqual(payload["node"], "match_agent")
        self.assertEqual(payload["duration_ms"], 12.5)

    def test_log_event_helper_carries_fields(self):
        """log_event 是业务侧唯一入口，必须真的能把字段带出来。"""

        import json

        with self.assertLogs("app.probe", level="INFO") as captured:
            log_event(
                logging.getLogger("app.probe"),
                logging.INFO,
                "带字段",
                node="verify_agent",
            )

        # assertLogs 用自己的 handler，这里直接验formatter 行为
        record = captured.records[0]
        self.assertEqual(record.node, "verify_agent")

        payload = json.loads(JsonFormatter().format(record))
        self.assertEqual(payload["node"], "verify_agent")

    def test_exception_is_serialized(self):
        import json
        import sys

        try:
            raise ValueError("boom")
        except ValueError:
            record = self._record("失败", exc_info=sys.exc_info())

        payload = json.loads(JsonFormatter().format(record))

        self.assertIn("ValueError", payload["exc"])

    def test_multiline_message_stays_parseable(self):
        import json

        payload = json.loads(JsonFormatter().format(self._record("第一行\n第二行")))

        self.assertIn("第一行", payload["msg"])


class ClientIpTests(unittest.TestCase):
    def _scope(self, headers=None, client=None):
        return {"type": "http", "headers": headers or [], "client": client}

    def test_prefers_forwarded_first_segment(self):
        scope = self._scope(headers=[(b"x-forwarded-for", b"1.2.3.4, 5.6.7.8")])

        self.assertEqual(client_ip(scope), "1.2.3.4")

    def test_falls_back_to_peer(self):
        scope = self._scope(client=("10.0.0.1", 5555))

        self.assertEqual(client_ip(scope), "10.0.0.1")

    def test_truncates_long_value(self):
        scope = self._scope(headers=[(b"x-forwarded-for", b"9" * 200)])

        self.assertLessEqual(len(client_ip(scope)), 45)

    def test_missing_client_returns_placeholder(self):
        self.assertEqual(client_ip(self._scope()), "-")


if __name__ == "__main__":
    unittest.main()
