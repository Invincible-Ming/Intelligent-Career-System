"""Real Streamlit widgets with a synthetic authenticated wallet API."""
import json
from unittest.mock import patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

WALLET = {"balance": 100, "available": 100, "reserved": 0, "checked_in_today": False,
          "rules": {"initial_points": 100, "daily_checkin_points": 10, "points_per_1000_tokens": 1}}


def response(value):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps(value).encode()
    return result


def test_points_page_checkin_refreshes_fragment_and_disables_repeat():
    info = dict(WALLET)
    calls = []
    def fake(method, url, **kwargs):
        calls.append((method, url))
        if url.endswith("/check-in"):
            info.update(balance=110, available=110, checked_in_today=True)
            return response({**info, "awarded": True})
        if url.endswith("/entries"):
            return response({"items": [{"id": "gift", "kind": "initial", "status": "settled", "points": 100,
                "created_at": "2026-10-22T09:00:00+00:00", "input_tokens": 0, "output_tokens": 0}], "has_more": False})
        return response(info)
    scopes = []
    original = st.rerun
    def rerun(*, scope="app"):
        scopes.append(scope)
        original()
    with patch("requests.request", side_effect=fake), patch("streamlit.rerun", side_effect=rerun):
        app = AppTest.from_string("from career_ui.credits import render_credits_page\nrender_credits_page()").run()
        assert not app.exception
        assert app.metric[0].value == "100"
        app.button(key="page_checkin").click().run()
        assert not app.exception
        assert app.metric[0].value == "110"
        assert app.button(key="page_checkin").disabled
        assert scopes == ["fragment"]
        assert sum(method == "POST" for method, _ in calls) == 1


def test_user_provider_choice_is_batched_and_available_without_admin():
    calls = []
    def fake(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return response({"knowledge_provider": kwargs["json"]["provider"], "queued": 0})
    script = '''
import streamlit as st
from career_ui.api import set_security_options
from career_ui.retrieval import render_provider_choice
set_security_options({'knowledge_provider':'credits'})
st.session_state.setdefault('auth_user', {'id':'alice', 'is_admin':False, 'knowledge_provider':'ima'})
render_provider_choice(st.session_state.auth_user)
'''
    with patch("requests.request", side_effect=fake):
        app = AppTest.from_string(script).run()
        assert not app.exception
        # Streamlit exposes segmented_control as button_group in AppTest.
        control = app.get("button_group")[0]
        control.set_value("milvus").run()
        assert not calls
        next(button for button in app.button if button.label == "应用检索方式").click().run()
        assert not app.exception
        assert len(calls) == 1 and calls[0][2]["json"] == {"provider": "milvus"}
        assert app.session_state.auth_user["knowledge_provider"] == "milvus"
