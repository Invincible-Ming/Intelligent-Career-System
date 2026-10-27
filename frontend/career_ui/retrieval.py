"""User-facing descriptions of the server-selected retrieval route."""
import streamlit as st

from career_ui.api import APIError, request_api, security_options


def uses_ima(user):
    return user.get("knowledge_provider", security_options().get("knowledge_provider")) == "ima"


def knowledge_strategy(user):
    if uses_ima(user):
        return "基础检索：ima 知识库 + 本地文件关键词检索"
    return "增强检索：Milvus 语义检索 + BM25 → RRF 融合 → BGE 重排序"


def render_match_strategy(user):
    with st.expander("分析方式与检索说明"):
        st.info("当前匹配方式：直接分析所选简历与岗位正文")
        st.caption("匹配评分不经过知识库检索；勾选“补充检索公开岗位资料”后，才会联网搜索作为补充参考。")
        st.caption(f"当前账号的知识库策略｜{knowledge_strategy(user)}（用于知识检索与对话中的知识查询）")


def render_provider_choice(user):
    if security_options().get("knowledge_provider") != "credits":
        return
    label = "检索方式 · ima 与本地文件" if uses_ima(user) else "检索方式 · 混合检索与重排序"
    with st.expander(label):
        options = ["ima", "milvus"]
        with st.form("knowledge_provider_form", border=False):
            selected = st.segmented_control("知识库检索方式", options, default="ima" if uses_ima(user) else "milvus",
                                             format_func=lambda value: "ima + 本地关键词" if value == "ima" else "混合检索 + 重排序")
            st.caption("两种方式均可选择。切换到混合检索后会准备本账号文件索引，Embedding 用量会扣积分；"
                       "ima 与本地文档分别保存，不会自动互相同步。")
            submitted = st.form_submit_button("应用检索方式", disabled=selected is None)
        if submitted and selected is not None:
            try:
                request_api("PUT", "/api/knowledge/provider", json={"provider": selected})
                user["knowledge_provider"] = selected
                st.session_state.pop("search_results", None)
                st.toast("检索方式已更新")
                st.rerun()
            except APIError as exc:
                st.error(str(exc))


@st.fragment(run_every=5.0)
def render_index_progress():
    try:
        info = request_api("GET", "/api/knowledge/status")
        if info.get("provider") != "milvus":
            return
        counts = info.get("index") or {}
        total = counts.get("total", 0)
        if total and counts.get("ready", 0) < total:
            st.progress(counts["ready"] / total, text=f"正在准备文件检索：{counts['ready']} / {total}")
            if counts.get("failed"):
                st.warning(f"{counts['failed']} 份文件准备失败，请确认积分余额及检索服务后重试。")
            if st.button("重新准备", key="retry_index_progress"):
                request_api("POST", "/api/knowledge/index/retry")
                st.rerun(scope="fragment")
        elif total:
            st.caption(f"{total} 份文件已可用于混合检索")
    except APIError as exc:
        st.warning(str(exc))
