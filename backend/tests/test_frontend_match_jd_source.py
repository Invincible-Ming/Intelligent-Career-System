"""岗位匹配页「JD 来源」切换的边界场景。

覆盖 no-JD-document 这类容易变成死路的分支：
- 没有岗位描述文档时选「已上传文档」，必须给出可操作提示与逃生出口；
- 逃生按钮要真能把 radio 切回「直接输入」；
- 有JD 文档时不得误报该警告。
"""
import ast
import json
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "frontend" / "app.py"

OPTIONS = {
    "registration_enabled": True,
    "chat_max_input_chars": 4000,
    "chat_max_output_tokens": 2048,
    "chat_total_timeout": 90,
    "analysis_total_timeout": 180,
    "max_upload_size_mb": 20,
    "search_max_input_chars": 2000,
    "jd_max_input_chars": 30000,
    "feedback_max_input_chars": 2000,
    "auth_cookie_name": "career_session",
}

RESUME = {
    "id": "d1",
    "filename": "简历.pdf",
    "document_type": "resume",
    "status": "ready",
    "chunk_count": 3,
}
JD_DOC = {
    "id": "d2",
    "filename": "岗位.txt",
    "document_type": "job_description",
    "status": "ready",
    "chunk_count": 2,
}
JD_DOC_TWO = {
    "id": "d3",
    "filename": "岗位对照.txt",
    "document_type": "job_description",
    "status": "ready",
    "chunk_count": 1,
}


def _response(value, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(value).encode()
    return result


def _http_factory(documents):
    def _http(method="GET", url="/", **kwargs):
        if url.endswith("/api/documents"):
            return _response(documents)
        if url.endswith("/api/runs"):
            return _response([])
        if url.endswith("/api/auth/options"):
            return _response(OPTIONS)
        if url.endswith("/api/auth/me"):
            return _response({"username": "alice", "is_admin": False})
        return _response([])

    return _http


def _open_match_page(documents):
    """渲染并切到岗位匹配页。

    注意：AppTest 的每次 .run() 都要在patch 上下文内执行，
    因此把交互动作做成回调交给调用方，而不是在这里返回 app。
    """

    st.cache_resource.clear()
    patcher_request = patch("requests.request", side_effect=_http_factory(documents))
    patcher_cookie = patch.object(
        type(st.context),
        "cookies",
        new_callable=PropertyMock,
        return_value={"career_session": "x" * 43},
    )
    patcher_component = patch(
        "streamlit.components.v2.component",
        return_value=lambda **kwargs: None,
    )

    for patcher in (patcher_request, patcher_cookie, patcher_component):
        patcher.start()

    app = AppTest.from_file(str(APP_PATH), default_timeout=25).run()
    assert not app.exception, [item.value for item in app.exception]

    navigation = next(button for button in app.button if button.label == "岗位匹配")
    navigation.click().run()
    assert not app.exception, [item.value for item in app.exception]

    return app


class MatchPageTestCase(unittest.TestCase):
    """把 patch 生命周期覆盖到整个用例期间。"""

    documents: list = []

    def setUp(self):
        st.cache_resource.clear()
        self._patchers = [
            patch("requests.request", side_effect=_http_factory(self.documents)),
            patch.object(
                type(st.context),
                "cookies",
                new_callable=PropertyMock,
                return_value={"career_session": "x" * 43},
            ),
            patch(
                "streamlit.components.v2.component",
                return_value=lambda **kwargs: None,
            ),
        ]
        for patcher in self._patchers:
            patcher.start()
        self.addCleanup(self._stop)

        self.app = AppTest.from_file(str(APP_PATH), default_timeout=25).run()
        navigation = next(
            button for button in self.app.button if button.label == "岗位匹配"
        )
        navigation.click().run()

    def _stop(self):
        for patcher in self._patchers:
            patcher.stop()


class NoJobDocumentDeadEndTests(MatchPageTestCase):
    """没有岗位描述文档时，不得把用户留在无路可走的界面。"""

    documents = [RESUME]

    def test_switching_to_uploaded_option_shows_warning(self):
        self.app.radio[0].set_value("已上传文档").run()

        self.assertFalse(
            self.app.exception, [item.value for item in self.app.exception]
        )
        warnings = " ".join(item.value for item in self.app.warning)
        self.assertIn("还没有上传过", warnings)

    def test_escape_button_is_present(self):
        """必须给出可直接点击的出口，而不是只留一句提示。"""

        self.app.radio[0].set_value("已上传文档").run()

        labels = [button.label for button in self.app.button]
        self.assertIn("改用直接输入 JD", labels)

    def test_escape_button_restores_text_input(self):
        self.app.radio[0].set_value("已上传文档").run()

        escape = next(
            button for button in self.app.button if button.label == "改用直接输入 JD"
        )
        escape.click().run()

        self.assertFalse(
            self.app.exception, [item.value for item in self.app.exception]
        )
        self.assertEqual(
            self.app.radio[0].value, "直接输入",
            "点逃生按钮后应切回直接输入",
        )
        labels = [field.label for field in self.app.text_area]
        self.assertIn("输入岗位描述 (JD)", labels, "切回后应恢复JD 输入框")

    def test_match_button_not_trapped_disabled(self):
        """无 JD 时按钮可点但会提示缺输入，而不是变成无反应的 disabled。"""

        self.app.radio[0].set_value("已上传文档").run()

        match = next(
            button for button in self.app.button if button.label == "开始岗位匹配"
        )
        self.assertFalse(
            match.disabled,
            "不应把按钮置灰，应允许点击并在提交时给出提示",
        )


class HasJobDocumentTests(MatchPageTestCase):
    """有 JD 文档时，走正常选择路径，不误报警告。

    这组用的是真实数据形态（1 份简历 + 2 份 JD），
    对应「刚上传完 JD 就切到匹配页」的用户场景。
    """

    documents = [RESUME, JD_DOC, JD_DOC_TWO]

    def test_selectbox_appears_and_no_warning(self):
        self.app.radio[0].set_value("已上传文档").run()

        self.assertFalse(
            self.app.exception, [item.value for item in self.app.exception]
        )
        warnings = " ".join(item.value for item in self.app.warning)
        self.assertNotIn("还没有上传过", warnings)
        labels = [field.label for field in self.app.selectbox]
        self.assertIn("选择岗位描述文档", labels)

    def test_all_job_documents_are_offered(self):
        """两份 JD 都应出现在下拉框里，不能只显示一份。"""

        self.app.radio[0].set_value("已上传文档").run()

        selector = next(
            field for field in self.app.selectbox
            if field.label == "选择岗位描述文档"
        )
        self.assertEqual(
            len(selector.options), 2,
            f"应有两份 JD，实际 {selector.options}",
        )

    def test_switching_back_and_forth_is_stable(self):
        self.app.radio[0].set_value("已上传文档").run()
        self.app.radio[0].set_value("直接输入").run()

        self.assertFalse(
            self.app.exception, [item.value for item in self.app.exception]
        )
        labels = [field.label for field in self.app.text_area]
        self.assertIn("输入岗位描述 (JD)", labels)

    def test_refresh_entry_exists(self):
        """数据可能是旧快照，需要显式刷新入口。"""

        labels = [button.label for button in self.app.button]
        self.assertIn("刷新文档列表", labels)


class _FormVisitor(ast.NodeVisitor):
    """找出所有位于 st.form 体内的裸 st.button 调用。"""

    def __init__(self):
        self.form_depth = 0
        self.buttons_in_form: list[int] = []

    def visit_With(self, node):
        is_match_form = any(
            item.context_expr is not None
            and isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "form"
            for item in node.items
        )
        self.form_depth += 1 if is_match_form else 0
        self.generic_visit(node)
        self.form_depth -= 1 if is_match_form else 0

    def visit_Call(self, node):
        if (
                self.form_depth > 0
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "button"
        ):
            self.buttons_in_form.append(node.lineno)
        self.generic_visit(node)


class JdSourceControlPlacementTests(unittest.TestCase):
    """来源切换控件必须放在 form 外。

    Streamlit 的 form 内控件只有提交时才返回值：
    - form 内放 st.button 会直接抛 StreamlitInvalidLayoutContextError；
    - form 内放 radio 切换来源会导致整表重跑、状态互相覆盖。
    """

    def setUp(self):
        self.source = APP_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_radio_is_declared_before_form(self):
        lines = self.source.split("\n")
        radio_line = next(
            index for index, line in enumerate(lines)
            if 'key="jd_source_choice"' in line
        )
        form_line = next(
            index for index, line in enumerate(lines)
            if 'with st.form("match_form"' in line
        )

        self.assertLess(
            radio_line, form_line,
            "jd_source radio 必须在 st.form 之前声明",
        )

    def test_escape_button_is_outside_form(self):
        """form 内放 st.button 会让 Streamlit 直接抛异常。"""

        visitor = _FormVisitor()
        visitor.visit(self.tree)

        self.assertEqual(
            visitor.buttons_in_form,
            [],
            f"form 内出现了裸 st.button（行 {visitor.buttons_in_form}），"
            "Streamlit 会抛 StreamlitInvalidLayoutContextError",
        )


if __name__ == "__main__":
    unittest.main()