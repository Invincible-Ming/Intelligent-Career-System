"""Browser-facing public links; never expose Python's internal backend address remotely."""
from urllib.parse import urlencode, urlsplit

import streamlit as st

from career_ui.session import browser_auth_urls


def public_backend_base() -> str:
    base, _ = browser_auth_urls()
    frontend = urlsplit(st.context.url or "http://localhost:8501/")
    if frontend.port == 8080:
        base = f"{frontend.scheme}://{frontend.netloc}"
    return base


def pricing_page_url() -> str:
    _, return_to = browser_auth_urls()
    base = public_backend_base()
    scheme = st.context.theme.type
    return base + "/pricing?" + urlencode({"return_to": return_to,
                                            "theme": scheme if scheme in ("light", "dark") else "auto"})


def apply_page_link(allowed_pages):
    # Consume the destination only after login. The browser login reload keeps
    # the query string, so a signed-out visitor can still reach credits afterward.
    page = st.query_params.get("page")
    if page in allowed_pages:
        st.session_state.active_page = page
    if "page" in st.query_params:
        del st.query_params["page"]
