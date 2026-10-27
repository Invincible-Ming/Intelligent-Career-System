"""Real Streamlit panels with synthetic HTTP; settings do not eagerly submit work."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

SCRIPT = '''
import streamlit as st
from career_ui.api import set_security_options
from career_ui.knowledge import render_knowledge_page
set_security_options({'knowledge_provider':'subscription'})
st.session_state.setdefault('auth_user', {
    'id':'alice', 'username':'alice', 'is_admin':True, 'knowledge_provider':'milvus',
    'subscription':{'plan':'pro', 'plan_name':'专业版'},
})
render_knowledge_page(st.session_state.auth_user, queued_indexing=False)
'''


def response(value, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(value).encode()
    return result


class SyntheticApi:
    def __init__(self):
        self.calls = []
        self.documents = [{"id": "doc-1", "filename": "resume.txt", "document_type": "resume",
                           "created_at": "2026-10-22T01:00:00+00:00", "status": "ready", "chunk_count": 1}]
        self.fail_upload = False

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if method == "GET" and url.endswith("/api/documents"):
            return response(self.documents)
        if method == "GET" and url.endswith("/download"):
            result = response(None)
            result._content = b"synthetic original\x00\xff"
            return result
        if method == "GET" and "/api/documents/" in url:
            document = next(doc for doc in self.documents if url.endswith("/" + doc["id"]))
            return response({**document, "size_bytes": 2048, "content_type": "text/plain", "preview_available": True})
        if method == "PATCH" and "/api/documents/" in url:
            document = next(doc for doc in self.documents if url.endswith("/" + doc["id"]))
            document["filename"] = kwargs["json"]["filename"]
            return response(document)
        if method == "POST" and url.endswith("/retry"):
            document = next(doc for doc in self.documents if "/" + doc["id"] + "/retry" in url)
            document.update(status="ready", error_message=None)
            return response(document)
        if method == "GET" and url.endswith("/api/knowledge/status"):
            return response({"provider": "milvus", "index": {}})
        if method == "GET" and url.endswith("/api/subscriptions/users"):
            return response([])
        if method == "POST" and "/api/search" in url:
            return response([])
        if method == "POST" and url.endswith("/api/documents/upload"):
            if self.fail_upload:
                return response({"detail": "synthetic upload failure"}, 503)
            new = {**self.documents[0], "id": "doc-2", "filename": kwargs["files"]["file"][0],
                   "document_type": kwargs["data"]["document_type"]}
            self.documents.append(new)
            return response({"document": new, "deduplicated": False})
        if method == "DELETE":
            identifier = url.rsplit("/", 1)[-1]
            self.documents = [doc for doc in self.documents if doc["id"] != identifier]
            return response({"message": "ok"})
        return response({})

    def count(self, method, suffix):
        return sum(m == method and url.endswith(suffix) for m, url, _ in self.calls)


def test_search_settings_are_batched_and_do_not_reload_file_list():
    api = SyntheticApi()
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        assert not app.exception
        assert not app.file_uploader  # Upload is a deliberate dialog action now.
        app.text_input(key="knowledge_search_query").set_value("Python").run()
        app.selectbox(key="knowledge_search_strategy").set_value("bm25").run()
        app.selectbox(key="knowledge_search_document_type").set_value("resume").run()
        app.slider(key="knowledge_search_top_k").set_value(3).run()
        assert not app.exception
        assert api.count("GET", "/api/documents") == 1
        assert api.count("POST", "/api/search/bm25") == 0
        assert api.count("POST", "/api/documents/upload") == 0
        next(button for button in app.button if button.label == "执行检索").click().run()
        assert not app.exception
        assert api.count("POST", "/api/search/bm25") == 1
        call = next(kwargs for method, url, kwargs in api.calls if url.endswith("/api/search/bm25"))
        assert call["json"] == {"query": "Python", "document_type": "resume", "top_k": 3}
        assert api.count("GET", "/api/documents") == 1


def test_file_cache_refresh_and_owner_change_do_not_reuse_private_rows():
    api = SyntheticApi()
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        app.run()
        assert api.count("GET", "/api/documents") == 1
        app.button(key="knowledge_refresh_documents").click().run()
        assert api.count("GET", "/api/documents") == 2
        api.documents = []
        app.session_state.auth_user = {"id": "bob", "username": "bob", "is_admin": False,
                                       "knowledge_provider": "ima"}
        app.run()
        assert not app.exception
        assert api.count("GET", "/api/documents") == 3
        assert not any("resume.txt" in item.value for item in app.markdown)


@pytest.mark.parametrize("provider, expected, excluded", [
    ("ima", "基础检索：ima", "增强检索：Milvus"),
    ("milvus", "增强检索：Milvus", "基础检索：ima"),
])
def test_subscription_search_uses_account_strategy_without_backend_selector(provider, expected, excluded):
    api = SyntheticApi()
    script = SCRIPT.replace("queued_indexing=False", "queued_indexing=True")
    script = script.replace("'knowledge_provider':'milvus'", f"'knowledge_provider':'{provider}'")
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(script).run(timeout=15)
        assert not app.exception
        app.button(key="knowledge_open_settings").click().run()
        captions = "\n".join(item.value for item in app.caption)
        assert expected in captions
        assert excluded not in captions
        app.session_state["knowledge_dialog"] = None
        app = AppTest.from_string(script).run(timeout=15)
        assert not any(box.key == "knowledge_search_strategy" for box in app.selectbox)
        app.text_input(key="knowledge_search_query").set_value("Python").run()
        app.selectbox(key="knowledge_search_document_type").set_value("resume").run()
        assert api.count("POST", "/api/search/hybrid") == 0
        next(button for button in app.button if button.label == "执行检索").click().run()
        assert not app.exception
        assert api.count("POST", "/api/search/hybrid") == 1
        assert api.count("POST", "/api/search") == 0
        assert api.count("POST", "/api/search/bm25") == 0
        call = next(kwargs for method, url, kwargs in api.calls if method == "POST")
        assert call["json"] == {"query": "Python", "document_type": "resume", "top_k": 5}


def test_provider_switch_discards_results_from_previous_strategy():
    api = SyntheticApi()
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        app.session_state.search_results = [{"content": "previous result"}]
        app.session_state.auth_user = {"id": "alice", "is_admin": True, "knowledge_provider": "ima"}
        app.run()
        assert not app.exception
        assert "search_results" not in app.session_state


def test_upload_failure_keeps_selected_file_available_to_retry():
    api = SyntheticApi()
    api.fail_upload = True
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        app.button(key="knowledge_open_upload").click().run()
        app.file_uploader[0].set_value(("new_resume.txt", b"synthetic resume", "text/plain"))
        next(button for button in app.button if button.label == "上传并向量化入库").click().run()
        assert not app.exception
        assert api.count("POST", "/api/documents/upload") == 1
        assert app.file_uploader[0].value.name == "new_resume.txt"
        assert any("synthetic upload failure" in item.value for item in app.error)


def test_upload_and_delete_invalidate_list_and_request_only_fragment_reruns():
    api = SyntheticApi()
    requested_scopes = []
    original_rerun = st.rerun

    def rerun(*, scope="app"):
        requested_scopes.append(scope)
        # AppTest starts a new runner per action instead of browser fragment events.
        # Execute its full rerun while recording the production scope requested by the panel.
        original_rerun()

    with patch("requests.request", side_effect=api), patch("streamlit.rerun", side_effect=rerun):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        app.button(key="knowledge_open_upload").click().run()
        app.file_uploader[0].set_value(("new_resume.txt", b"synthetic resume", "text/plain"))
        next(button for button in app.button if button.label == "上传并向量化入库").click().run()
        assert not app.exception
        assert api.count("POST", "/api/documents/upload") == 1
        assert "knowledge_dialog" not in app.session_state
        # AppTest retains a closed dialog tree; a fresh view reflects the same API data.
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        assert not app.file_uploader
        assert any("new_resume.txt" == item.label for item in app.button)
        app.button(key="delete-document-doc-2").click().run()
        assert not app.exception
        assert api.count("DELETE", "/api/documents/doc-2") == 0
        app.button(key="confirm_document_delete").click().run()
        assert api.count("DELETE", "/api/documents/doc-2") == 1
        assert not any("new_resume.txt" == item.label for item in app.button)
        assert requested_scopes == ["app", "app"]  # Close the completed modal and refresh its list.


def test_subscription_account_switch_reuses_list_and_retains_form_edits():
    calls = []
    users = [{"id": identifier, "username": identifier, "subscription": {
        "plan": "free", "plan_name": "免费版", "expires_at": None,
    }} for identifier in ("alice", "bob")]

    def fake(method, url, **kwargs):
        calls.append((method, url))
        return response(users)

    script = '''
import streamlit as st
from career_ui.subscriptions import render_subscription_admin
st.session_state.setdefault('auth_user', {'id':'admin', 'username':'admin'})
render_subscription_admin()
'''
    with patch("requests.request", side_effect=fake):
        app = AppTest.from_string(script).run(timeout=15)
        app.selectbox(key="subscription_target").set_value("bob").run()
        assert not app.exception
        assert len(calls) == 1  # A local account selection must not fetch the same list again.
        assert any("正在管理：bob" in item.value for item in app.caption)
        app.selectbox[1].set_value("pro").run()
        selected_expiry = app.time_input[0].value
        app.run()
        assert app.selectbox[1].value == "pro"
        assert app.time_input[0].value == selected_expiry
        assert all(method == "GET" for method, _ in calls)  # No accidental grants while editing.
        app.button(key="subscription_refresh_users").click().run()
        assert len(calls) == 2
        app.session_state.auth_user = {"id": "another-admin", "username": "another-admin"}
        app.run()
        assert len(calls) == 3  # The administrator list cache never crosses accounts.


def test_document_download_is_deferred_and_retains_only_own_session_token():
    from career_ui.knowledge import document_download_button
    api = SyntheticApi()
    st.session_state["access_token"] = "synthetic-alice-token"
    try:
        with patch("requests.request", side_effect=api), patch("streamlit.download_button") as control:
            document_download_button(api.documents[0], key="download-test")
            assert not api.calls
            content = control.call_args.kwargs["data"]
            assert callable(content)
            assert control.call_args.kwargs["on_click"] == "ignore"
            st.session_state["access_token"] = "synthetic-bob-token"
            assert content() == b"synthetic original\x00\xff"
            assert api.calls[0][2]["headers"] == {"Authorization": "Bearer synthetic-alice-token"}
    finally:
        st.session_state.pop("access_token", None)


def test_rename_and_details_never_download_or_reprocess_original():
    api = SyntheticApi()
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        app.button(key="rename-document-doc-1").click().run()
        assert not app.exception
        app.text_input(key="knowledge_rename_name").set_value("面试简历").run()
        next(button for button in app.button if button.label == "保存名称").click().run()
        assert not app.exception
        assert api.documents[0]["filename"] == "面试简历.txt"
        assert api.count("PATCH", "/api/documents/doc-1") == 1
        assert api.count("POST", "/api/documents/doc-1/retry") == 0
        assert api.count("GET", "/api/documents/doc-1/download") == 0
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        app.button(key="details-document-doc-1").click().run()
        assert not app.exception
        assert any("2.0 KB" == text.value for text in app.text)
        assert api.count("GET", "/api/documents/doc-1") == 1
        assert api.count("GET", "/api/documents/doc-1/download") == 0


def test_retry_is_only_offered_for_failed_files_and_requires_explicit_start():
    api = SyntheticApi()
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        assert not any(button.key == "retry-document-doc-1" for button in app.button)
        api.documents[0].update(status="failed", error_message="synthetic parsing failure")
        app.button(key="knowledge_refresh_documents").click().run()
        app.button(key="retry-document-doc-1").click().run()
        assert not app.exception
        assert api.count("POST", "/api/documents/doc-1/retry") == 0
        assert any("synthetic parsing failure" in error.value for error in app.warning)
        app.button(key="knowledge_confirm_retry").click().run()
        assert not app.exception
        assert api.count("POST", "/api/documents/doc-1/retry") == 1
        assert api.documents[0]["id"] == "doc-1"
        assert api.documents[0]["status"] == "ready"


def test_batch_selection_is_opt_in_filter_scoped_and_delete_confirmed():
    api = SyntheticApi()
    api.documents.append({**api.documents[0], "id": "doc-2", "filename": "job.txt"})
    with patch("requests.request", side_effect=api):
        app = AppTest.from_string(SCRIPT).run(timeout=15)
        assert not app.checkbox
        app.button(key="knowledge_manage_documents").click().run()
        assert app.button(key="knowledge_manage_documents").label == "完成管理"
        assert len(app.checkbox) == 2
        app.button(key="knowledge_select_all").click().run()
        assert all(box.value for box in app.checkbox)
        app.text_input(key="knowledge_file_filter").set_value("resume").run()
        assert not app.exception
        assert len(app.checkbox) == 1
        assert not app.session_state["knowledge_selected_file_doc-2"]
        app.button(key="knowledge_batch_delete").click().run()
        assert not app.exception
        assert api.count("DELETE", "/api/documents/doc-1") == 0
        app.button(key="confirm_document_delete").click().run()
        assert not app.exception
        assert api.count("DELETE", "/api/documents/doc-1") == 1
        assert api.count("DELETE", "/api/documents/doc-2") == 0
        assert [doc["id"] for doc in api.documents] == ["doc-2"]
