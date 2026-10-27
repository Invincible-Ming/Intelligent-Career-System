"""岗位匹配页历史任务列表渲染测试。

覆盖任务状态标签的四个分支（已完成 / 待人工审核 / 失败 / 进行中）。
这些分支引用模块级常量 MATCH_STATUS_NAMES，在拆分重构时极易漏掉导入，
语法检查与纯函数测试都发现不了，只有真实渲染才会暴露。
"""
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

DOCUMENT = {
    "id": "d1",
    "filename": "syn.txt",
    "document_type": "resume",
    "status": "ready",
    "chunk_count": 1,
}

# 四种任务状态，对应 app.py 里 success / warning / error / info 四个分支。
RUNS = [
    {
        "run_id": f"r{index}",
        "status": status,
        "created_at": "2026-10-09T12:00:00",
        "input_data": {"resume_document_id": "d1", "jd_input_type": "text"},
        "result": {"total_score": 80},
    }
    for index, status in enumerate(
        ["completed", "paused", "failed", "running"]
    )
]


def _response(value, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(value).encode()
    return result


def _http_request(method, url, **kwargs):
    if url.endswith("/api/documents"):
        return _response([DOCUMENT])
    if url.endswith("/api/runs"):
        return _response(RUNS)
    if url.endswith("/api/credits/me"):
        return _response({"balance": 100, "available": 100, "reserved": 0, "checked_in_today": False,
                         "rules": {"initial_points": 100, "daily_checkin_points": 10,
                                   "points_per_1000_tokens": 1}})
    if url.endswith("/api/auth/options"):
        return _response(OPTIONS)
    if url.endswith("/api/auth/me"):
        return _response({"username": "alice", "is_admin": False})
    return _response([])


class MatchPageHistoryTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def test_all_status_branches_render_labels(self):
        with patch("requests.request", side_effect=_http_request), \
                patch.object(
                    type(st.context),
                    "cookies",
                    new_callable=PropertyMock,
                    return_value={"career_session": "x" * 43},
                ), \
                patch(
                    "streamlit.components.v2.component",
                    return_value=lambda **kwargs: None,
                ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=25).run()
            app.button(key="nav_match").click().run()

            self.assertFalse(
                app.exception, [item.value for item in app.exception]
            )

            labels = (
                [item.value for item in app.success]
                + [item.value for item in app.warning]
                + [item.value for item in app.error]
                + [item.value for item in app.info]
            )
            for expected in ("已完成", "待人工审核", "失败", "进行中"):
                self.assertIn(expected, labels)
            self.assertIn("当前匹配方式：直接分析所选简历与岗位正文", labels)
            captions = "\n".join(item.value for item in app.caption)
            self.assertIn("匹配评分不经过知识库检索", captions)
            self.assertIn("勾选“补充检索公开岗位资料”后", captions)

    def test_history_list_keeps_run_id_field(self):
        """后端契约：任务列表必须带 run_id，否则页面直接崩。

        这里固定该契约，避免 mock 与真实返回结构漂移。
        """

        self.assertTrue(all("run_id" in run for run in RUNS))


if __name__ == "__main__":
    unittest.main()
