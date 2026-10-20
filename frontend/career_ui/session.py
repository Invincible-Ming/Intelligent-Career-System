"""会话与浏览器登录桥接。

Streamlit 侧只拿到 HttpOnly cookie，真正的登录动作发生在浏览器 iframe 里。
这里负责：读取安全限制参数、清理登录态、驱动浏览器组件。
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import streamlit as st

from career_ui.api import (
    APIError,
    request_api,
    set_security_options,
)
from career_ui.config import API_BASE_URL


def browser_auth_urls() -> tuple[str, str]:
    """Browser and Streamlit must use the same hostname to share a host-only cookie."""

    frontend = urlsplit(st.context.url or "http://localhost:8501/")
    backend = urlsplit(os.getenv("BROWSER_API_BASE_URL", API_BASE_URL))
    if backend.hostname in ("localhost", "127.0.0.1") and frontend.hostname in ("localhost", "127.0.0.1"):
        netloc = frontend.hostname + (f":{backend.port}" if backend.port else "")
        backend = backend._replace(netloc=netloc)
    return urlunsplit(backend).rstrip("/"), f"{frontend.scheme}://{frontend.netloc}/"


def reset_login_state() -> None:
    """清空登录态并重跑。

    注意不要在下一轮 rerun 里恢复旧的初始请求 cookie。
    """

    st.session_state.clear()
    st.session_state.clear_browser_session = True
    st.rerun()


@st.cache_resource
def browser_auth_component():
    assets = Path(__file__).resolve().parent.parent / "auth_browser"
    return st.components.v2.component(
        "browser_auth",
        html=(assets / "form.html").read_text(encoding="utf-8"),
        css=(assets / "style.css").read_text(encoding="utf-8"),
        js=(assets / "auth.js").read_text(encoding="utf-8"),
    )


def clear_browser_cookie() -> None:
    """通知浏览器清除 host-only cookie，然后中断本次运行。"""

    base, return_to = browser_auth_urls()
    browser_auth_component()(
        data={
            "mode": "logout",
            "url": base + "/api/auth/browser-logout",
            "return_to": return_to,
        },
        key="cookie_cleanup",
    )
    st.stop()


def fetch_security_options() -> dict:
    """读取后端下发的限制参数。

    后端不可用时回落默认值，避免登录页因接口失败而无法渲染。
    """

    try:
        options = request_api("GET", "/api/auth/options")
    except APIError:
        options = None

    set_security_options(options)
    return options or {}


def auth_cookie_name() -> str:
    from career_ui.api import security_options

    return security_options().get("auth_cookie_name", "career_session")