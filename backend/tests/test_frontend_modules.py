"""前端 career_ui 模块单元测试。

这些是纯函数/纯逻辑，不依赖 Streamlit 运行时与后端，
因此比 AppTest 更细粒度，适合定位回归。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

from career_ui.api import (  # noqa: E402
    DEFAULT_SECURITY_OPTIONS,
    APIError,
    auth_headers,
    operation_read_timeout,
    security_options,
    set_security_options,
)
from career_ui.config import (  # noqa: E402
    DIFFICULTY_NAMES,
    DOCUMENT_TYPE_NAMES,
    SEARCH_MODE_NAMES,
)
from career_ui.endpoints import match_report_of, match_run_label  # noqa: E402
from career_ui.render import document_label, get_search_score_display  # noqa: E402


class SecurityOptionsTests(unittest.TestCase):
    def setUp(self):
        set_security_options(None)
        self.addCleanup(set_security_options, None)

    def test_defaults_are_complete(self):
        """默认值必须覆盖所有被引用的键，否则页面会 KeyError。"""

        for key in (
            "max_upload_size_mb",
            "chat_max_input_chars",
            "search_max_input_chars",
            "jd_max_input_chars",
            "feedback_max_input_chars",
            "registration_enabled",
            "auth_cookie_name",
        ):
            self.assertIn(key, DEFAULT_SECURITY_OPTIONS)

    def test_server_values_override_defaults(self):
        set_security_options({"chat_max_input_chars": 9999})

        self.assertEqual(security_options()["chat_max_input_chars"], 9999)
        # 未提供的键回落默认值，不应被整体替换
        self.assertEqual(
            security_options()["max_upload_size_mb"],
            DEFAULT_SECURITY_OPTIONS["max_upload_size_mb"],
        )

    def test_partial_payload_keeps_defaults(self):
        set_security_options({"max_upload_size_mb": 5})

        self.assertEqual(security_options()["max_upload_size_mb"], 5)
        self.assertEqual(
            security_options()["jd_max_input_chars"],
            DEFAULT_SECURITY_OPTIONS["jd_max_input_chars"],
        )


class TimeoutSelectionTests(unittest.TestCase):
    def setUp(self):
        set_security_options(None)
        self.addCleanup(set_security_options, None)

    def test_chat_uses_chat_timeout(self):
        self.assertEqual(
            operation_read_timeout("/api/chat/completions"),
            DEFAULT_SECURITY_OPTIONS["chat_total_timeout"] + 10,
        )

    def test_upload_uses_upload_timeout(self):
        self.assertEqual(
            operation_read_timeout("/api/documents/upload"),
            DEFAULT_SECURITY_OPTIONS["upload_total_timeout"] + 10,
        )

    def test_match_uses_match_timeout(self):
        self.assertEqual(
            operation_read_timeout("/api/match/stream"),
            DEFAULT_SECURITY_OPTIONS["match_total_timeout"] + 10,
        )

    def test_evaluation_uses_analysis_timeout(self):
        self.assertEqual(
            operation_read_timeout("/api/evaluation/datasets"),
            DEFAULT_SECURITY_OPTIONS["analysis_total_timeout"] + 10,
        )

    def test_unknown_path_uses_default_timeout(self):
        self.assertEqual(
            operation_read_timeout("/api/documents"),
            DEFAULT_SECURITY_OPTIONS["api_total_timeout"] + 10,
        )

    def test_server_override_changes_timeout(self):
        set_security_options({"match_total_timeout": 1})

        self.assertEqual(operation_read_timeout("/api/match/stream"), 11)


class AuthHeaderTests(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()
        self.addCleanup(st.session_state.clear)

    def test_no_token_means_no_header(self):
        with patch.object(st, "session_state", {}):
            self.assertEqual(auth_headers(), {})

    def test_token_is_wrapped_in_bearer(self):
        with patch.object(st, "session_state", {"access_token": "abc"}):
            self.assertEqual(auth_headers(), {"Authorization": "Bearer abc"})


class MatchReportShapeTests(unittest.TestCase):
    """报告有三种落库形态，都必须被正确解包。"""

    def test_flat_report(self):
        self.assertEqual(
            match_report_of({"result": {"total_score": 88}}),
            {"total_score": 88},
        )

    def test_current_report_wrapper(self):
        self.assertEqual(
            match_report_of({"result": {"current_report": {"total_score": 70}}}),
            {"total_score": 70},
        )

    def test_report_wrapper(self):
        self.assertEqual(
            match_report_of({"result": {"report": {"total_score": 60}}}),
            {"total_score": 60},
        )

    def test_missing_or_malformed_returns_empty(self):
        self.assertEqual(match_report_of({}), {})
        self.assertEqual(match_report_of({"result": None}), {})
        self.assertEqual(match_report_of({"result": "not-a-dict"}), {})

    def test_resume_analysis_is_recognised_as_report(self):
        self.assertEqual(
            match_report_of({"result": {"resume_analysis": {"skills": []}}}),
            {"resume_analysis": {"skills": []}},
        )


class MatchRunLabelTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"id": "d1", "filename": "张三简历.pdf"},
            {"id": "d2", "filename": "后端岗位.docx"},
        ]

    def test_includes_names_status_and_score(self):
        label = match_run_label(
            {
                "input_data": {"resume_document_id": "d1", "jd_document_id": "d2"},
                "status": "completed",
                "created_at": "2026-10-09T12:00:00",
                "result": {"total_score": 85},
            },
            self.documents,
        )

        self.assertIn("张三简历.pdf", label)
        self.assertIn("后端岗位.docx", label)
        self.assertIn("已完成", label)
        self.assertIn("85分", label)

    def test_manual_jd_is_labelled(self):
        label = match_run_label(
            {
                "input_data": {
                    "resume_document_id": "d1",
                    "jd_input_type": "text",
                },
                "status": "paused",
            },
            self.documents,
        )

        self.assertIn("手动输入 JD", label)
        self.assertIn("待人工审核", label)

    def test_unknown_document_id_falls_back(self):
        label = match_run_label(
            {"input_data": {"resume_document_id": "gone"}, "status": "running"},
            [],
        )

        self.assertIn("简历", label)


class DocumentLabelTests(unittest.TestCase):
    def test_known_type_is_translated(self):
        label = document_label(
            {"filename": "a.pdf", "document_type": "resume", "status": "ready"}
        )

        self.assertIn("简历", label)
        self.assertNotIn("ready", label)

    def test_unknown_type_falls_back_to_raw(self):
        label = document_label(
            {"filename": "a.pdf", "document_type": "weird", "status": "ready"}
        )

        self.assertIn("weird", label)


class SearchScoreDisplayTests(unittest.TestCase):
    def test_reranked_source(self):
        name, source, score = get_search_score_display(
            {"source": "hybrid_reranked", "rerank_score": 0.9}, "hybrid"
        )

        self.assertEqual(name, "BGE 重排分")
        self.assertEqual(score, 0.9)

    def test_rrf_only_source(self):
        name, _source, score = get_search_score_display(
            {"source": "hybrid", "rrf_score": 0.03}, "hybrid"
        )

        self.assertEqual(name, "RRF 综合分")
        self.assertAlmostEqual(score, 0.03)

    def test_bm25_source(self):
        name, _source, _score = get_search_score_display(
            {"source": "bm25", "score": 1.5}, "bm25"
        )

        self.assertEqual(name, "BM25 分数")

    def test_defaults_to_cosine(self):
        name, source, _score = get_search_score_display({"score": 0.8}, "dense")

        self.assertEqual(name, "余弦相似度")
        self.assertEqual(source, "Milvus")

    def test_missing_score_defaults_to_zero(self):
        _name, _source, score = get_search_score_display({}, "hybrid")

        self.assertEqual(score, 0.0)


class ConfigCompletenessTests(unittest.TestCase):
    def test_display_names_cover_backend_values(self):
        """后端文档类型必须都有中文名，否则界面会显示原始英文。"""

        self.assertEqual(set(DOCUMENT_TYPE_NAMES), {"resume", "job_description", "knowledge"})
        self.assertTrue(all(DOCUMENT_TYPE_NAMES.values()))

    def test_search_modes_cover_endpoints(self):
        self.assertEqual(set(SEARCH_MODE_NAMES), {"hybrid", "dense", "bm25", "ima"})

    def test_difficulty_levels(self):
        self.assertEqual(set(DIFFICULTY_NAMES), {"junior", "intermediate", "senior"})


class APIErrorTests(unittest.TestCase):
    def test_carries_message(self):
        error = APIError("后端不可用")

        self.assertEqual(str(error), "后端不可用")


if __name__ == "__main__":
    unittest.main()
