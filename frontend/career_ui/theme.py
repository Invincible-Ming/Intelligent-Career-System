"""Keep custom surfaces and native Streamlit controls on the same color scheme."""
from pathlib import Path

import streamlit as st


def preserve_drafts() -> None:
    """Detach non-sensitive drafts from widget cleanup when navigating pages."""
    for key in (
        "match_jd_text", "jd_source_choice", "match_resume_document_id", "match_job_document_id",
        "knowledge_search_query", "knowledge_search_strategy", "knowledge_search_document_type",
        "knowledge_search_top_k", "knowledge_file_filter", "knowledge_file_type",
        "credit_filter_kind", "credit_filter_status", "interview_difficulty", "interview_count",
        "learning_weeks", "learning_hours",
    ):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def apply_theme() -> None:
    # Reading the native theme also makes Streamlit notify this app when the
    # user changes appearance. On the first connection it may be unknown; CSS
    # then follows the system until Streamlit reports the effective scheme.
    scheme = st.context.theme.type
    scheme_css = ""
    if scheme in ("light", "dark"):
        light, dark = ("block", "none") if scheme == "light" else ("none", "block")
        scheme_css = (f":root {{ color-scheme: {scheme}; "
                      f"--brand-light-display: {light}; --brand-dark-display: {dark}; }}")
    stylesheet = Path(__file__).resolve().parents[1] / "styles.css"
    # Style-only HTML is placed outside the content layout (no empty spacer).
    st.html(f"<style>{stylesheet.read_text(encoding='utf-8')}\n{scheme_css}</style>")
