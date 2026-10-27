"""Render real Streamlit forms with fake HTTP; verify no private load before login."""

import json
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

calls = []
options = {
    "registration_enabled": True,
    "chat_max_input_chars": 4000,
    "chat_max_output_tokens": 2048,
    "chat_total_timeout": 90,
    "analysis_total_timeout": 180,
    "max_upload_size_mb": 20,
    "search_max_input_chars": 2000,
    "jd_max_input_chars": 30000,
    "feedback_max_input_chars": 2000,
}


def response(value, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(value).encode()
    return result


def api_request(method, url, **kwargs):
    calls.append((method, url, kwargs.get("headers", {})))
    if url.endswith("/api/credits/me"):
        return response({"balance": 100, "available": 100, "reserved": 0, "checked_in_today": False,
                         "rules": {"initial_points": 100, "daily_checkin_points": 10,
                                   "points_per_1000_tokens": 1}})
    if url.endswith("/api/auth/options"):
        return response(options)
    if url.endswith("/api/auth/login"):
        return response({"access_token": "test-session-token", "user": {"username": "alice", "is_admin": False}})
    if url.endswith("/api/auth/me"):
        return response({"username": "alice", "is_admin": False})
    if url.endswith("/api/auth/logout"):
        return response({"message": "bye"})
    return response([])


def assert_session_key_absent(app, key):
    """断言用户键不在 session_state 中。

    streamlit 1.64 移除了 SessionState.filtered_state；
    SessionState 实现了 __contains__，可直接用 in 判断用户键。
    """

    assert key not in app.session_state, f"session_state不应包含 {key}"


class FrontendAuthTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()  # AppTest creates its own component registry.

    def test_login_guard_and_logout_clear_private_ui(self):
        calls.clear()
        cookies = {}
        token = "test-session-token" + "x" * 32
        with (
            patch("requests.request", side_effect=api_request),
            patch.object(type(st.context), "cookies", new_callable=PropertyMock, return_value=cookies),
            patch("streamlit.components.v2.component", return_value=lambda **kwargs: None),
        ):
            path = Path(__file__).resolve().parents[2] / "frontend" / "app.py"
            app = AppTest.from_file(str(path), default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertEqual([x.label for x in app.tabs], ["登录", "注册"])
            self.assertTrue(all("/api/auth/" in url for _, url, _ in calls))
            cookies["career_session"] = token
            # A fresh browser connection has no session_state, only the cookie.
            app = AppTest.from_file(str(path), default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs), 0)
            self.assertEqual(len(app.radio), 0)
            self.assertEqual(app.session_state["active_page"], "chat")
            app.button(key="nav_knowledge").click().run()
            self.assertTrue(
                any(
                    "/api/documents" in url and headers.get("Authorization") == "Bearer " + token
                    for _, url, headers in calls
                )
            )
            app.session_state["match_report"] = {"private": "alice report"}
            next(button for button in app.button if button.label == "退出登录").click().run()
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["clear_browser_session"])
            assert_session_key_absent(app, "access_token")
            assert_session_key_absent(app, "match_report")
            self.assertFalse(app.chat_message)
            self.assertFalse(any("alice" in str(row.value) for row in app.sidebar.markdown))
            cookies.clear()  # Browser component clears Cookie and reloads.
            app = AppTest.from_file(str(path), default_timeout=15).run()
            self.assertEqual([x.label for x in app.tabs], ["登录", "注册"])

    def test_about_sits_after_logout_and_closes_the_account_menu(self):
        """关于按钮排在退出登录之后；点击时先收起浮层，再弹对话框。"""
        calls.clear()
        token = "test-session-token" + "x" * 32
        with (
            patch("requests.request", side_effect=api_request),
            patch.object(type(st.context), "cookies", new_callable=PropertyMock,
                         return_value={"career_session": token}),
            patch("streamlit.components.v2.component", return_value=lambda **kwargs: None),
        ):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "frontend" / "app.py"),
                                    default_timeout=15).run()
            self.assertFalse(app.exception)
            labels = [button.label for button in app.button]
            self.assertIn("关于简面职说", labels)
            self.assertLess(labels.index("退出登录"), labels.index("关于简面职说"))

            app.button(key="account_about").click().run()
            self.assertFalse(app.exception)
            # 计数器 +1 换掉 popover 的 key，浮层因此收起；待办标记本轮即被消费掉。
            self.assertEqual(app.session_state["account_menu_generation"], 1)
            assert_session_key_absent(app, "about_pending")

    def test_invalid_cookie_cannot_render_private_content(self):
        calls.clear()

        def invalid_me(method, url, **kwargs):
            if url.endswith("/api/auth/me"):
                return response({"detail": "expired"}, 401)
            return api_request(method, url, **kwargs)

        with (
            patch("requests.request", side_effect=invalid_me),
            patch.object(
                type(st.context), "cookies", new_callable=PropertyMock, return_value={"career_session": "x" * 43}
            ),
            patch("streamlit.components.v2.component", return_value=lambda **kwargs: None),
        ):
            path = Path(__file__).resolve().parents[2] / "frontend" / "app.py"
            app = AppTest.from_file(str(path), default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["clear_browser_session"])
            assert_session_key_absent(app, "access_token")
            self.assertTrue(all("/api/auth/" in url for _, url, _ in calls))

    def test_rules_credits_destination_waits_for_login_then_opens_wallet(self):
        cookies = {}
        calls.clear()

        def linked_api(method, url, **kwargs):
            if url.endswith("/api/credits/entries"):
                return response({"items": [], "has_more": False})
            return api_request(method, url, **kwargs)

        with patch("requests.request", side_effect=linked_api), patch.object(
            type(st.context), "cookies", new_callable=PropertyMock, return_value=cookies,
        ), patch("streamlit.components.v2.component", return_value=lambda **kwargs: None):
            path = Path(__file__).resolve().parents[2] / "frontend" / "app.py"
            app = AppTest.from_file(str(path), default_timeout=15)
            app.query_params["page"] = "credits"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.query_params["page"], ["credits"])
            self.assertTrue(all("/api/auth/" in url for _, url, _ in calls))
            cookies["career_session"] = "synthetic-credit-route-token" + "x" * 32
            # Browser login reloads this URL, preserving the intended page.
            app = AppTest.from_file(str(path), default_timeout=15)
            app.query_params["page"] = "credits"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.active_page, "credits")
            self.assertNotIn("page", app.query_params)
            self.assertEqual(app.metric[0].value, "100")

    def test_match_progress_and_structured_report_are_rendered_before_review(self):
        report = {
            "total_score": 78,
            "match_level": "较高匹配",
            "scores": {key: 80 for key in ["skill", "experience", "responsibility", "education", "bonus"]},
            "matched_skills": ["Python"],
            "missing_skills": ["Testing"],
            "strengths": ["Synthetic evidence"],
            "risks": ["Synthetic gap"],
            "suggestions": ["Add synthetic tests"],
            "resume_analysis": {"skills": ["Python"]},
            "job_analysis": {"job_title": "Synthetic developer"},
        }
        document = {
            "id": "synthetic-resume",
            "filename": "synthetic.txt",
            "document_type": "resume",
            "status": "ready",
            "chunk_count": 1,
        }

        def http_request(method, url, **kwargs):
            if url.endswith("/api/documents"):
                return response([document])
            return api_request(method, url, **kwargs)

        events = [
            ("init", {"run_id": "synthetic-run"}),
            ("start", {}),
            (
                "node_update",
                {
                    "node": "resume_agent",
                    "status": "completed",
                    "message": "简历分析完成（复用已有结果）",
                    "cache_hit": True,
                },
            ),
            (
                "node_update",
                {
                    "node": "jd_agent",
                    "status": "completed",
                    "message": "岗位分析完成（复用已有结果）",
                    "cache_hit": True,
                },
            ),
            ("progress", {"node": "match_agent", "message": "正在生成匹配报告…"}),
            ("heartbeat", {"elapsed_seconds": 5}),
            ("node_update", {"node": "match_agent", "status": "completed", "score": 78}),
            ("interrupt", {"thread_id": "synthetic-run", "current_report": report}),
        ]
        stream = response({})
        stream.close = lambda: None
        stream.iter_lines = lambda **kwargs: iter(
            line for kind, data in events for line in [f"event: {kind}", "data: " + json.dumps(data), ""]
        )
        with (
            patch("requests.request", side_effect=http_request),
            patch("requests.post", return_value=stream),
            patch.object(
                type(st.context), "cookies", new_callable=PropertyMock, return_value={"career_session": "x" * 43}
            ),
            patch("streamlit.components.v2.component", return_value=lambda **kwargs: None),
            patch("streamlit.progress", wraps=st.progress) as progress,
        ):
            path = Path(__file__).resolve().parents[2] / "frontend" / "app.py"
            app = AppTest.from_file(str(path), default_timeout=15).run()
            app.button(key="nav_match").click().run()
            next(field for field in app.text_area if field.label == "输入岗位描述 (JD)").set_value("Synthetic JD")
            next(button for button in app.button if button.label == "开始岗位匹配").click().run()
            self.assertFalse(app.exception)
            progress.assert_called()
            self.assertTrue(app.session_state["match_paused"])
            self.assertEqual(app.session_state["preliminary_report"]["total_score"], 78)
            self.assertTrue(any(metric.label == "综合加权匹配分" for metric in app.metric))
            self.assertTrue(any("Add synthetic tests" in row.value for row in app.markdown))
            self.assertFalse(any("思考中" in row.value for row in app.markdown))


class BrowserAuthURLTests(unittest.TestCase):
    def urls(self, frontend, configured=None):
        from career_ui.session import browser_auth_urls
        environment = {} if configured is None else {"BROWSER_API_BASE_URL": configured}
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(type(st.context), "url", new_callable=PropertyMock, return_value=frontend),
            patch("career_ui.session.API_BASE_URL", "http://localhost:8000"),
        ):
            return browser_auth_urls()

    def test_public_tunnel_uses_same_origin_without_localhost(self):
        self.assertEqual(self.urls("https://career-demo.trycloudflare.com/?page=chat"),
                         ("https://career-demo.trycloudflare.com", "https://career-demo.trycloudflare.com/"))

    def test_local_development_keeps_backend_port_and_host(self):
        self.assertEqual(self.urls("http://127.0.0.1:8501/"),
                         ("http://127.0.0.1:8000", "http://127.0.0.1:8501/"))

    def test_explicit_same_origin_supports_local_gateway(self):
        self.assertEqual(self.urls("http://localhost:8080/", "same-origin"),
                         ("http://localhost:8080", "http://localhost:8080/"))

    def test_explicit_browser_api_address_is_preserved(self):
        self.assertEqual(self.urls("https://career.example/", "https://career.example/backend/"),
                         ("https://career.example/backend", "https://career.example/"))
