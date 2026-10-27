"""Link routing works on local Streamlit, the gateway and public tunnel origins."""
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from career_ui.links import pricing_page_url
from career_ui import session


@pytest.mark.parametrize("frontend,expected", [
    ("http://127.0.0.1:8501/", "http://127.0.0.1:8000"),
    ("http://localhost:8501/", "http://localhost:8000"),
    ("http://127.0.0.1:8080/", "http://127.0.0.1:8080"),
    ("https://career.example/", "https://career.example"),
])
def test_rules_link_uses_browser_origin_and_effective_theme(frontend, expected, monkeypatch):
    monkeypatch.delenv("BROWSER_API_BASE_URL", raising=False)
    monkeypatch.setattr(session, "API_BASE_URL", "http://127.0.0.1:8000")
    with patch.object(type(st.context), "url", new_callable=PropertyMock, return_value=frontend), patch.object(
        type(st.context), "theme", new_callable=PropertyMock, return_value=SimpleNamespace(type="dark"),
    ):
        link = urlsplit(pricing_page_url())
    assert f"{link.scheme}://{link.netloc}" == expected
    assert link.path == "/pricing"
    assert parse_qs(link.query) == {"return_to": [frontend], "theme": ["dark"]}


def test_public_credits_link_is_consumed_once_and_cannot_select_admin_page():
    script = '''
import streamlit as st
from career_ui.links import apply_page_link
st.session_state.setdefault('active_page','chat')
apply_page_link({'chat','credits'})
st.text(st.session_state.active_page)
'''
    app = AppTest.from_string(script)
    app.query_params["page"] = "credits"
    app.run()
    assert not app.exception
    assert app.session_state.active_page == "credits"
    assert "page" not in app.query_params
    app.session_state.active_page = "chat"
    app.run()
    assert app.session_state.active_page == "chat"
    app.query_params["page"] = "evaluation"
    app.run()
    assert app.session_state.active_page == "chat"
