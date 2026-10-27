"""
智能求职 Multi-Agent 系统 Streamlit 前端 (支持 Human-in-the-loop 人机协同、会话断点读档与简洁对话布局)。

UI 架构特点：
- 对话布局：顶部功能导航、侧边历史列表和原生底部输入框
- SSE 逐字打字机：基于 Streamlit 原生 st.write_stream 实现
- 任务存档与断点恢复：支持刷新/重启后一键读档恢复 paused 任务
- 人机协同干预：捕获 LangGraph interrupt 中断事件并提供交互卡片唤醒
- Multi-Agent 状态流：Fan-out 并发与 Self-Correction 纠错流式反馈
- 混合检索与知识库：Dense + BM25 + RRF + BGE Reranker
"""

from __future__ import annotations

import os
from datetime import datetime
from html import escape
from typing import Any
from urllib.parse import urlsplit

import streamlit as st
from career_ui.brand import APP_ICON, DESCRIPTOR, INTRO, NAME, TAGLINE, brand_html
from career_ui.credits import render_credit_account, render_credits_page
from career_ui.theme import apply_theme, preserve_drafts
from career_ui.matching import render_match_setup
from career_ui.chat import begin_chat, render_chat_history
from career_ui.links import apply_page_link, pricing_page_url
from streamlit.components.v1 import html as render_html_component

# ----------------------------------------------------------------------
# 基础配置
# ----------------------------------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

SHORT_TIMEOUT = 20
LONG_TIMEOUT = 600

# ----------------------------------------------------------------------
# 页面配置与界面样式
# ----------------------------------------------------------------------

st.set_page_config(
    page_title=f"{NAME} · AI 求职搭子",
    page_icon=str(APP_ICON),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep presentation separate from authentication and business logic.
apply_theme()


# ----------------------------------------------------------------------
# API 客户端封装
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# 可复用模块：API 客户端 / 会话 / 业务接口 / 视图组件
# ----------------------------------------------------------------------

from career_ui.api import (  # noqa: E402
    APIError,
    request_api,
    security_options,
    stream_match_api,
    stream_match_resume_api,
)
from career_ui.config import (  # noqa: E402
    DIFFICULTY_NAMES,
    SEARCH_MODE_NAMES,
)
from career_ui.endpoints import (  # noqa: E402
    MATCH_STATUS_NAMES,
    create_interview_plan,
    create_learning_plan,
    delete_chat_conversation,
    delete_match_run,
    fetch_match_context,
    get_conversation_messages,
    get_conversations,
    get_evaluation_datasets,
    get_evaluation_report_html,
    list_evaluation_experiments,
    match_report_of,
    match_run_label,
    start_evaluation_experiment,
    update_chat_conversation,
)
from career_ui.knowledge import render_knowledge_page  # noqa: E402
from career_ui.retrieval import render_match_strategy  # noqa: E402
from career_ui.render import (  # noqa: E402
    account_identity,
    account_usage,
    document_label,
    empty_state,
    match_progress_updates,
    match_stepper,
    page_header,
    show_document_preview,
    show_interview_plan,
    show_learning_plan,
    show_match_report,
)
from career_ui.session import (  # noqa: E402
    auth_cookie_name,
    browser_auth_component,
    browser_auth_urls,
    clear_browser_cookie,
    fetch_security_options,
    reset_login_state,
)


def _switch_to_direct_jd_input() -> None:
    """把 JD 来源切回「直接输入」。

    必须通过按钮的 on_click 回调修改 session_state：
    在同一次脚本运行里直接赋值会抛
    StreamlitWidgetAlreadyInstantiatedError（widget 已实例化后再改其 key）。
    """

    st.session_state["jd_source_choice"] = "直接输入"
    st.rerun()


def _continue_nonblocking_review() -> None:
    """Finish legacy technical-only pauses without a model call or a claimed human approval."""
    if not st.session_state.get("match_paused"):
        return
    report = st.session_state.get("preliminary_report") or {}
    review = report.get("verification") or {}
    if review.get("status") not in ("timeout", "service_error") or review.get("issues") or review.get("unresolved_issues"):
        return
    thread_id = st.session_state.get("match_thread_id")
    if not thread_id or st.session_state.get("unverified_continue_attempted") == thread_id:
        return
    st.session_state["unverified_continue_attempted"] = thread_id
    try:
        with st.spinner("正在展示匹配报告…"):
            outcome = request_api("POST", "/api/match/resume", json={"thread_id": thread_id}, timeout=LONG_TIMEOUT)
        if outcome.get("status") == "completed":
            st.session_state["match_report"] = match_report_of(outcome)
            st.session_state["match_paused"] = False
            st.session_state.pop("preliminary_report", None)
            st.rerun()
    except APIError as exc:
        st.warning(f"暂未能恢复报告：{exc}")


def _clear_deleted_match_state(deleted_run_ids: list[str]) -> None:
    identifiers = set(deleted_run_ids)
    current_deleted = any(
        str(st.session_state.get(key) or "") in identifiers
        for key in ("match_run_id", "match_thread_id")
    )
    if current_deleted:
        for key in ("match_run_id", "match_thread_id", "match_report", "preliminary_report", "match_paused"):
            st.session_state.pop(key, None)
    for name in ("interview_plan", "learning_plan"):
        source = st.session_state.get(f"{name}_source_run_id")
        plan_run = st.session_state.get(f"{name}_run_id")
        if source in identifiers or plan_run in identifiers or (current_deleted and source is None):
            for key in (name, f"{name}_source_run_id", f"{name}_run_id"):
                st.session_state.pop(key, None)


@st.dialog("删除匹配记录", width="small")
def _confirm_delete_match_run(run_id: str, label: str) -> None:
    st.write("确认删除这次匹配报告及由它生成的面试、学习计划？")
    st.caption(label)
    st.caption("删除后无法恢复，已上传的简历和岗位文档将保留。")
    confirm, cancel = st.columns(2)
    if confirm.button("确认删除", type="primary", use_container_width=True):
        try:
            response = delete_match_run(run_id)
        except APIError as exc:
            st.error(str(exc))
            return
        _clear_deleted_match_state(response.get("deleted_run_ids") or [run_id])
        st.session_state["match_delete_notice"] = "匹配记录已删除"
        st.rerun(scope="app")
    if cancel.button("取消", use_container_width=True):
        st.rerun(scope="app")


def _run_match_flow(
        *,
        selected_resume: dict[str, Any],
        selected_job: dict[str, Any] | None,
        jd_source: str,
        jd_text: str,
        research_approved: bool = False,
        step_area=None,
) -> None:
    """执行岗位匹配流式请求并把结果写入 session_state。"""

    payload: dict[str, Any] = {
        "resume_document_id": selected_resume["id"], "research_approved": research_approved,
    }
    if jd_source == "直接输入":
        payload["jd_text"] = jd_text.strip()
    else:
        payload["jd_document_id"] = selected_job["id"]

    previous_result = {key: st.session_state[key] for key in (
        "match_report", "match_run_id", "match_paused", "match_thread_id", "preliminary_report",
    ) if key in st.session_state}
    received_result = False
    if step_area is not None:
        step_area.html(match_stepper(2))
    # A failed new attempt must not discard the last completed report.
    st.session_state.pop("match_report", None)
    st.session_state.pop("match_run_id", None)
    st.session_state.pop("match_paused", None)
    st.session_state.pop("match_thread_id", None)
    st.session_state.pop("preliminary_report", None)

    # 动态流式状态框展示
    with st.status("正在分析简历和岗位要求…", expanded=True) as status_box:

        update_progress = match_progress_updates(status_box)

        try:
            for event_type, data in stream_match_api(payload):
                update_progress(event_type, data)
                if event_type == "init":
                    run_id = data.get("run_id")
                    if run_id:
                        st.session_state["match_run_id"] = run_id
                        st.session_state["match_thread_id"] = run_id
                    st.caption("任务已创建，正在准备分析。")

                elif event_type == "start":
                    st.write("正在同时分析简历与岗位要求。")

                # 兼容公开调研提示；匹配报告不再逐字输出。
                elif event_type == "token":
                    node = data.get("node")
                    if node == "match_agent":
                        st.markdown(data.get("content", ""))

                elif event_type == "node_update":
                    node = data.get("node")
                    msg = data.get("message", "")

                    if node == "resume_agent":
                        st.write(msg)
                    elif node == "jd_agent":
                        st.write(msg)
                    elif node == "match_agent":
                        score = data.get("score")
                        score_info = f" (初评总分: {score}分)" if score else ""
                        st.write(f"{msg}{score_info}")
                    elif node == "verify_agent":
                        if data.get("status") == "passed":
                            status_box.write(msg)
                        elif data.get("nonblocking"):
                            status_box.info(msg)
                        else:
                            status_box.warning(msg)
                            if data.get("feedback"):
                                status_box.caption(data["feedback"])

                # 捕获人机协同挂起事件
                elif event_type == "interrupt":
                    received_result = True
                    st.session_state["reveal_match_report"] = True
                    st.session_state["match_paused"] = True
                    st.session_state["match_thread_id"] = data.get("thread_id")
                    st.session_state["preliminary_report"] = data.get("current_report")
                    status_box.update(
                        label="自动校验未通过，等待人工审核",
                        state="complete",
                        expanded=True,
                    )
                    st.rerun()

                elif event_type == "complete":
                    report = data.get("report")
                    if report:
                        received_result = True
                        st.session_state["reveal_match_report"] = True
                        st.session_state["match_report"] = report
                        status_box.update(
                            label="匹配报告已生成，自动校验未完成" if data.get("verification_nonblocking") else "岗位多维匹配与事实校验全部完成！",
                            state="complete",
                            expanded=False,
                        )
                        st.rerun(scope="app")

                elif event_type == "error":
                    st.error(f"执行异常：{data.get('message')}")
                    status_box.update(
                        label="匹配中断",
                        state="error",
                        expanded=True,
                    )
                    break

        except APIError as exc:
            st.error(str(exc))
            status_box.update(
                label="请求失败",
                state="error",
                expanded=True,
            )
        finally:
            if not received_result:
                for key in ("match_report", "match_run_id", "match_paused", "match_thread_id", "preliminary_report"):
                    st.session_state.pop(key, None)
                st.session_state.update(previous_result)
                if previous_result.get("match_report"):
                    st.caption("上一次报告已保留，可继续查看或重新尝试。")
            if step_area is not None:
                stage = 3 if st.session_state.get("match_report") or st.session_state.get("match_paused") else 1
                step_area.html(match_stepper(stage))


# ----------------------------------------------------------------------
# 顶部页面标签
# ----------------------------------------------------------------------

# Authentication must run before loading or rendering private tabs.
# 注意：不要把返回值绑定到 security_options这个名字——
# 它已被导入为函数，会遮蔽后续 security_options() 调用。
try:
    fetch_security_options()
except APIError as exc:
    st.error(str(exc))
    st.stop()

login_area = st.empty()
navigation_area = st.empty()
private_area = st.empty()
with st.sidebar.container(key="sidebar_shell"):
    chat_sidebar_area = st.empty()
    account_area = st.empty()

if st.session_state.get('clear_browser_session'):
    private_area.empty()
    account_area.empty()
    navigation_area.empty()
    chat_sidebar_area.empty()
    with login_area.container():
        clear_browser_cookie()

if not st.session_state.get('access_token'):
    cookie_token = st.context.cookies.get(auth_cookie_name())
    if cookie_token and 32 <= len(cookie_token) <= 128:
        # /auth/me below verifies validity before any private content is rendered.
        st.session_state.access_token = cookie_token

with login_area.container():
    if not st.session_state.get('access_token'):
        private_area.empty()
        account_area.empty()
        navigation_area.empty()
        chat_sidebar_area.empty()
        st.markdown("""
        <style>
        header[data-testid="stHeader"],
        [data-testid="stSidebar"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stAppViewContainer"] {
            background: var(--background-color);
        }
        [data-testid="stMainBlockContainer"] {
            min-height: 100dvh;
            padding: 2rem 1.25rem !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { width: 100%; }
        .st-key-auth_card {
            width: 100%; max-width: 460px; margin: 0 auto;
            padding: 2.3rem; border: 1px solid var(--ui-line); border-radius: 18px;
            background: var(--ui-surface); color: var(--text-color);
            box-shadow: var(--ui-shadow-panel);
        }
        .auth-brand { text-align: center; font-size: 2.3rem; color: var(--ui-accent); }
        .auth-title { text-align: center; font-size: 1.05rem; font-weight: 600; margin-top: .4rem; }
        .auth-subtitle { text-align: center; color: var(--ui-muted); font-size: .9rem; margin: .7rem 0 1.5rem; }
        .st-key-auth_card [data-testid="stForm"] { border: 0; padding: 0; }
        .st-key-auth_card [data-baseweb="tab-list"] { gap: 0; }
        .st-key-auth_card [data-baseweb="tab"] { flex: 1; justify-content: center; }
        .st-key-auth_card [role="tablist"] { gap: 0; }
        .st-key-auth_card [role="tab"] { flex: 1; justify-content: center; }
        .st-key-auth_card [data-testid="stFormSubmitButton"] button {
            border-radius: 10px; min-height: 44px;
        }
        @media (max-width: 520px) {
            .st-key-auth_card { padding: 1.5rem; border-radius: 18px; }
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container(key='auth_card'):
            st.markdown(
                brand_html("auth-logo")
                + f'<div class="auth-title">{DESCRIPTOR}</div>'
                + f'<div class="auth-subtitle">{TAGLINE}</div>',
                unsafe_allow_html=True,
            )
            login_tab, register_tab = st.tabs(['登录', '注册'])
            with login_tab:
                browser_base, return_to = browser_auth_urls()
                if urlsplit(browser_base).hostname != urlsplit(return_to).hostname:
                    st.error('登录需要前后端使用同一域名，请配置 BROWSER_API_BASE_URL。')
                else:
                    browser_auth_component()(data={"mode": "login", "url": browser_base + '/api/auth/browser-login',
                                                   "return_to": return_to}, key="browser_login")
            with register_tab:
                if security_options()['registration_enabled']:
                    with st.form('register_form'):
                        new_username = st.text_input('新用户名', max_chars=32)
                        new_password = st.text_input('新密码', type='password', max_chars=128)
                        confirmation = st.text_input('确认密码', type='password', max_chars=128)
                        st.caption('用户名 3–32 位字母、数字或下划线；密码 12–128 位。')
                        register_submitted = st.form_submit_button('注册账号', type='primary', use_container_width=True)
                    if register_submitted:
                        if new_password != confirmation:
                            st.error('两次密码不一致')
                        else:
                            try:
                                request_api('POST', '/api/auth/register',
                                            json={'username': new_username, 'password': new_password})
                                st.success('注册成功，请在登录页登录。')
                            except APIError as exc:
                                st.error(str(exc))
                else:
                    st.info('当前未开放注册，请联系管理员获取账号。')
        st.stop()

login_area.empty()

try:
    st.session_state.auth_user = request_api('GET', '/api/auth/me')
except APIError as exc:
    st.error(str(exc))
    st.stop()


def select_page(page: str):
    st.session_state.active_page = page


def report_next_actions():
    next_interview, next_learning, _ = st.columns([1, 1.25, 3])
    if next_interview.button("准备面试", key="report_to_interview", type="primary", use_container_width=True):
        select_page("interview")
        st.rerun()
    if next_learning.button("制定学习计划", key="report_to_learning", use_container_width=True):
        select_page("learning")
        st.rerun()



def report_menu():
    current_run_id = st.session_state.get("match_run_id")
    if current_run_id:
        with st.popover("", icon=":material/more_horiz:"):
            if st.button("删除本次匹配", key="delete-current-match", type="tertiary"):
                _confirm_delete_match_run(str(current_run_id), "当前匹配报告")


def start_new_chat():
    from career_ui.chat import cancel_active_chat
    cancel_active_chat()
    st.session_state.active_page = "chat"
    st.session_state.conversation_id = None
    st.session_state.chat_messages = []
    st.session_state.rename_target = None


def close_conversation_dialog():
    st.session_state.pop("conversation_delete_target", None)


@st.dialog("删除对话", width="small", on_dismiss=close_conversation_dialog)
def confirm_delete_conversation(identifier: str, title: str):
    st.write(f"确定删除「{title}」？")
    st.caption("这段对话及其历史消息将被删除，此操作无法撤销。")
    confirm, cancel = st.columns(2)
    if confirm.button("确认删除", key="confirm_conversation_delete", type="primary", use_container_width=True):
        try:
            delete_chat_conversation(identifier)
            if st.session_state.conversation_id == identifier:
                start_new_chat()
            close_conversation_dialog()
            st.toast("对话已删除")
            st.rerun()
        except APIError as exc:
            st.error(str(exc))
    if cancel.button("取消", key="cancel_conversation_delete", use_container_width=True):
        close_conversation_dialog()
        st.rerun()


@st.dialog("关于简面职说")
def show_about():
    st.html(brand_html("about-logo"))
    st.markdown(f"**{DESCRIPTOR}**")
    st.write(TAGLINE)
    st.caption(INTRO)
    st.markdown("- 智能对话：梳理思路，解答求职问题\n"
                "- 知识库：整理资料，随时检索\n"
                "- 岗位匹配：看清优势、差距与改进方向\n"
                "- 面试准备与学习计划：让下一步更具体")


def queue_about():
    """让账户浮层重新挂载（收起）后再弹关于对话框，避免两个浮层叠在一起。"""
    st.session_state.about_pending = True
    st.session_state.account_menu_generation += 1


st.session_state.setdefault("active_page", "chat")
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("rename_target", None)
st.session_state.setdefault("account_menu_generation", 0)
st.session_state.setdefault("about_pending", False)
preserve_drafts()

is_admin = st.session_state.auth_user.get("is_admin", False)
queued_indexing = security_options().get("knowledge_provider") in ("subscription", "credits")
ima_mode = st.session_state.auth_user.get("knowledge_provider", security_options().get("knowledge_provider")) == "ima"

page_options = {
    "chat": "智能对话",
    "knowledge": "知识库",
    "match": "岗位匹配",
    "interview": "面试准备",
    "learning": "学习计划",
    "credits": "我的积分",
}
if is_admin and not ima_mode:
    page_options["evaluation"] = "评测"
apply_page_link(page_options)
active_page = st.session_state.active_page
if active_page not in page_options:
    active_page = st.session_state.active_page = "chat"

# 页面级侧边栏：仅智能对话显示（会话列表/账户区都在侧边栏），其余页面整栏隐藏。
if active_page != "chat":
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
        "[data-testid='stExpandSidebarButton'] {display: none;}</style>",
        unsafe_allow_html=True,
    )

with navigation_area.container():
    with st.container(key="app_header"):
        brand, navigation, account_column = st.columns([1.2, 5, .45], gap="small", vertical_alignment="center")
        brand.html(brand_html())
        with navigation:
            with st.container(key="top_navigation"):
                navigation_pages = {page: label for page, label in page_options.items() if page != "credits"}
                for column, (page, label) in zip(st.columns(len(navigation_pages), gap="small"), navigation_pages.items(), strict=False):
                    with column:
                        st.button(
                            label, key=f"nav_{page}", use_container_width=True,
                            type="primary" if active_page == page else "tertiary",
                            on_click=select_page, args=(page,),
                        )
        with account_column:
            account_area = st.empty()

with chat_sidebar_area.container():
    st.markdown('<div class="sidebar-brand">对话</div>', unsafe_allow_html=True)
    st.button("新对话", icon=":material/edit_square:", key="new_chat",
              use_container_width=True, on_click=start_new_chat)
    st.markdown('<div class="sidebar-label">最近对话</div>', unsafe_allow_html=True)
    with st.container(key="conversation_list"):
        history_convs = get_conversations()
        if not history_convs:
            st.caption("你的对话会显示在这里")
        for conv in history_convs:
            conv_id = str(conv["id"])
            title = conv.get("title") or "新对话"
            is_pinned = bool(conv.get("pinned"))
            is_active = active_page == "chat" and st.session_state.conversation_id == conv_id
            with st.container():
                title_col, menu_col = st.columns([1, 0.18], gap="small", vertical_alignment="center")
                with title_col:
                    if st.button(title, key=f"conv_{conv_id}", help=title,
                                 type="primary" if is_active else "tertiary", use_container_width=True):
                        try:
                            from career_ui.chat import cancel_active_chat
                            cancel_active_chat()
                            history_msgs = get_conversation_messages(conv_id)
                            st.session_state.conversation_id = conv_id
                            st.session_state.chat_messages = history_msgs
                            st.session_state.active_page = "chat"
                            st.rerun()
                        except APIError as exc:
                            st.error(str(exc))
                with menu_col:
                    # ChatGPT 风格：每行一个"…"菜单，包含重命名 / 置顶 / 删除。
                    with st.popover("", icon=":material/more_horiz:", key=f"conv_menu_{conv_id}"):
                        if st.session_state.rename_target == conv_id:
                            new_title = st.text_input("新的标题", value=title, max_chars=100,
                                                      key=f"rename_input_{conv_id}")
                            save_col, cancel_col = st.columns(2)
                            if save_col.button("保存", key=f"rename_save_{conv_id}", type="primary",
                                               use_container_width=True):
                                try:
                                    update_chat_conversation(conv_id, title=new_title.strip() or title)
                                    st.session_state.rename_target = None
                                    st.toast("已重命名")
                                    st.rerun()
                                except APIError as exc:
                                    st.error(str(exc))
                            if cancel_col.button("取消", key=f"rename_cancel_{conv_id}", use_container_width=True):
                                st.session_state.rename_target = None
                                st.rerun()
                        elif st.button("重命名", icon=":material/edit:", key=f"rename_open_{conv_id}",
                                       use_container_width=True):
                            st.session_state.rename_target = conv_id
                            st.rerun()
                        if st.button("取消置顶" if is_pinned else "置顶",
                                     icon=":material/keep_off:" if is_pinned else ":material/keep:",
                                     key=f"pin_{conv_id}", use_container_width=True):
                            try:
                                update_chat_conversation(conv_id, pinned=not is_pinned)
                                st.toast("已置顶" if not is_pinned else "已取消置顶")
                                st.rerun()
                            except APIError as exc:
                                st.error(f"操作失败：{exc}")
                        if st.button("删除", icon=":material/delete:", key=f"delete_{conv_id}",
                                     use_container_width=True):
                            st.session_state["conversation_delete_target"] = (conv_id, title)

if st.session_state.get("conversation_delete_target"):
    confirm_delete_conversation(*st.session_state.conversation_delete_target)

with account_area.container():
    with st.container(key="header_account"):
        username = st.session_state.auth_user['username']
        # The account menu is available from every page in the header.
        # Remount on navigation so an open portal cannot cover the destination page,
        # and on "about" so the popover closes before the dialog opens.
        with st.popover(
                f"{username} 的账户",
                key=f"account_popover_{active_page}_{st.session_state.account_menu_generation}",
                use_container_width=True):
            with st.container(key="account_menu"):
                account_identity(st.session_state.auth_user)
                render_credit_account()
                with st.container(key="quota_section"):
                    try:
                        usage = request_api('GET', '/api/auth/usage')
                    except APIError:
                        usage = None
                    account_usage(usage)
                with st.container(key="account_actions"):
                    if st.button('积分明细', icon=":material/receipt_long:", key="open_credit_page",
                                 type="tertiary", use_container_width=True):
                        select_page("credits")
                        st.rerun()
                    st.link_button("积分规则", pricing_page_url(), key="account_credit_rules",
                                   icon=":material/info:", type="tertiary", use_container_width=True)
                    if st.button('退出登录', icon=":material/logout:", key="account_logout",
                                 type="tertiary", use_container_width=True):
                        try:
                            request_api('POST', '/api/auth/logout')
                        except APIError:
                            pass
                        private_area.empty()
                        account_area.empty()
                        navigation_area.empty()
                        chat_sidebar_area.empty()
                        reset_login_state()
                    st.button("关于简面职说", key="account_about", icon=":material/chat_bubble:",
                              type="tertiary", use_container_width=True, on_click=queue_about)

# 浮层先随 key 变化收起，这一轮再弹对话框，两个浮层不会同时出现。
if st.session_state.pop("about_pending", False):
    show_about()

# At root level Streamlit pins this input to the bottom and handles sidebar resizing.
prompt_input = None
if active_page == "chat":
    prompt_input = st.chat_input("和简面职说聊聊…", key="chat_prompt",
                                 max_chars=security_options()["chat_max_input_chars"],
                                 disabled=bool(st.session_state.get("chat_task")))

with private_area.container():
    # ----------------------------------------------------------------------
    # 1. 智能对话页面
    # ----------------------------------------------------------------------

    if active_page == "chat":
        if prompt_input and not st.session_state.get("chat_task"):
            if begin_chat(prompt_input):
                st.rerun()
            st.toast("请输入一个具体问题")
        if not st.session_state.chat_messages and not st.session_state.conversation_id:
            username = escape(st.session_state.auth_user['username'])
            st.html(f'<div class="chat-welcome"><div class="welcome-kicker">{NAME} · 你的求职搭子</div>'
                    f'<h1>你好，{username}</h1><p>{TAGLINE}</p></div>')
        with st.container(key="chat_history"):
            render_chat_history()

    # ----------------------------------------------------------------------
    # 2. 知识库管理页面
    # ----------------------------------------------------------------------

    if active_page == "knowledge":
        render_knowledge_page(st.session_state.auth_user, queued_indexing=queued_indexing)

    # ----------------------------------------------------------------------
    # 3. 岗位匹配页面 (支持 SSE 节点进度流、断点恢复与 Human-in-the-loop)
    # ----------------------------------------------------------------------

    if active_page == "match":
        page_header("求职准备", "看清匹配程度，再准备面试与学习计划。", eyebrow="让每一次投递更有方向")
        step_area = st.empty()
        step_area.html(match_stepper(3 if st.session_state.get("match_report") or st.session_state.get("match_paused") else 1))
        delete_notice = st.session_state.pop("match_delete_notice", None)
        if delete_notice:
            st.success(delete_notice)

        # ==================================================================
        # 任务历史记录与断点恢复控制台
        # ==================================================================
        history_runs, all_documents = fetch_match_context()

        ready_documents = [d for d in all_documents if d.get("status") == "ready"]
        resume_documents = [
            d for d in ready_documents if d.get("document_type") == "resume"
        ]
        job_documents = [
            d for d in ready_documents if d.get("document_type") == "job_description"
        ]

        if not resume_documents:
            empty_state("先准备一份简历", "在知识库上传简历后，即可结合岗位要求分析优势、差距与改进方向。")
            st.button("前往知识库", key="match_open_knowledge", on_click=select_page, args=("knowledge",))
        else:
            has_report = bool(st.session_state.get("match_report") or st.session_state.get("match_paused"))
            with st.expander("重新匹配" if has_report else "选择简历与岗位", expanded=not has_report):
                render_match_setup(resume_documents=resume_documents, job_documents=job_documents,
                                   ready_documents=ready_documents,
                                   on_run=lambda **values: _run_match_flow(**values, step_area=step_area),
                                   on_switch=_switch_to_direct_jd_input, step_area=step_area)

        # ==================================================================
        # 人机协同 (Human-in-the-loop) 交互卡片
        # ==================================================================
        _continue_nonblocking_review()
        if st.session_state.get("match_paused"):
            st.divider()
            st.warning("**报告尚未通过自动校验，请查看下方“报告校验”信息；可填写修改意见重新评估。**")

            pre_report = st.session_state.get("preliminary_report") or {}
            if pre_report:
                score_val = pre_report.get("total_score")
                level_val = pre_report.get("match_level", "待评定")
                if score_val is not None:
                    st.info(f"**当前评估总分：{score_val} 分** ｜ 匹配等级：{level_val}")
                with st.expander("查看待审核的匹配报告", expanded=True):
                    show_match_report(pre_report)

            feedback_col, action_col, cancel_col = st.columns([3, 1, 1])
            with feedback_col:
                human_feedback_text = st.text_input(
                    "人工修正意见（可选）",
                    placeholder="例如：请根据简历项目经历重新核对技能匹配，并说明缺少的证据。留空表示接受当前报告。",
                    max_chars=security_options()["feedback_max_input_chars"],
                )
                st.caption("填写意见后会重新生成并校验；留空提交表示人工接受当前报告，原有校验提示仍会保留。")

            with action_col:
                st.write("")  # 垂直对齐占位
                st.write("")
                confirm_resume = st.button(
                    "提交审核",
                    type="primary",
                    use_container_width=True,
                )

            with cancel_col:
                st.write("")
                st.write("")
                if st.button("稍后处理", use_container_width=True):
                    st.session_state["match_paused"] = False
                    st.session_state.pop("preliminary_report", None)
                    st.session_state.pop("match_thread_id", None)
                    st.rerun()

            if confirm_resume:
                thread_id = st.session_state["match_thread_id"]

                review_stage = "正在根据审核意见重新评估…" if human_feedback_text.strip() else "正在保存人工确认…"
                with st.status(review_stage, expanded=True) as resume_box:

                    update_progress = match_progress_updates(resume_box, resuming=True)

                    try:
                        for event_type, data in stream_match_resume_api(
                                thread_id=thread_id,
                                human_feedback=human_feedback_text.strip() or None,
                        ):
                            update_progress(event_type, data)
                            if event_type == "resume":
                                resume_box.write(f"**[System]** {data.get('message')}")

                            # 公开资料查询提示由前端展示，结构化报告统一排版。
                            elif event_type == "token":
                                node = data.get("node")
                                if node == "match_agent":
                                    resume_box.markdown(data.get("content", ""))

                            elif event_type == "node_update":
                                node = data.get("node")
                                msg = data.get("message", "")
                                status = data.get("status")

                                if node == "match_agent":
                                    resume_box.write(msg)
                                elif node == "verify_agent":
                                    if status == "passed":
                                        resume_box.write(msg)
                                    elif data.get("nonblocking"):
                                        resume_box.info(msg)
                                    else:
                                        feedback = data.get("feedback", "")
                                        resume_box.warning(msg)
                                        if feedback:
                                            resume_box.caption(f"校验说明：{feedback}")

                            elif event_type == "interrupt":
                                st.session_state["match_paused"] = True
                                st.session_state["preliminary_report"] = data.get("current_report")
                                resume_box.update(label="重新校验仍未通过，请再次审核", state="complete", expanded=True)
                                st.rerun()

                            elif event_type == "complete":
                                report = data.get("report")
                                if report:
                                    st.session_state["match_report"] = report
                                    st.session_state["match_paused"] = False
                                    resume_box.update(
                                        label=data.get("message") or "匹配报告已生成",
                                        state="complete",
                                        expanded=False,
                                    )
                                    st.rerun()

                            elif event_type == "error":
                                resume_box.error(f"恢复失败：{data.get('message')}")
                                resume_box.update(
                                    label="校验异常",
                                    state="error",
                                    expanded=True,
                                )
                                break

                    except APIError as exc:
                        st.error(str(exc))
                        resume_box.update(
                            label="请求失败",
                            state="error",
                            expanded=True,
                        )

        # 渲染匹配完成的报告
        match_report = st.session_state.get("match_report")
        if match_report and not st.session_state.get("match_paused"):
            st.divider()
            show_match_report(match_report, actions=report_next_actions, menu=report_menu)
        render_match_strategy(st.session_state.auth_user)
        if history_runs:
            paused_count = sum(1 for r in history_runs if r.get("status") == "paused")
            with st.expander(
                    "岗位匹配历史记录",
                    expanded=bool(paused_count) or bool(st.session_state.get("match_paused")),
            ):
                if paused_count:
                    st.info(f"检测到 **{paused_count}** 个已挂起、等待人工审核的任务，可直接载入继续。")
                for run in history_runs:
                    run_id = str(run["run_id"])
                    status = run.get("status")
                    with st.container():
                        col_info, col_state, col_action, col_delete = st.columns(
                            [5, 1.3, 1.3, .9], vertical_alignment="center",
                        )
                        with col_info:
                            st.markdown(f"**{match_run_label(run, all_documents)}**")
                            st.caption(f"任务 ID：{run_id[:8]}…")
                        with col_state:
                            if status == "completed":
                                st.success(MATCH_STATUS_NAMES[status])
                            elif status == "paused":
                                st.warning(MATCH_STATUS_NAMES[status])
                            elif status == "failed":
                                st.error(MATCH_STATUS_NAMES[status])
                            else:
                                st.info(MATCH_STATUS_NAMES.get(status, str(status)))
                        with col_action:
                            if status == "paused":
                                if st.button("载入审核", key=f"load-run-{run_id}", use_container_width=True):
                                    res_data = run.get("result") or {}
                                    st.session_state["match_run_id"] = run_id
                                    st.session_state["match_thread_id"] = run_id
                                    st.session_state["match_paused"] = True
                                    st.session_state["preliminary_report"] = res_data.get("current_report") or res_data
                                    st.session_state.pop("match_report", None)
                                    st.success("已成功加载任务存档！")
                                    st.rerun()
                            elif status == "completed":
                                if st.button("查看报告", key=f"view-run-{run_id}", use_container_width=True):
                                    report = match_report_of(run)
                                    if not report:
                                        st.warning("该任务没有可展示的报告")
                                    else:
                                        st.session_state["match_report"] = report
                                        st.session_state["reveal_match_report"] = True
                                        st.session_state["match_paused"] = False
                                        st.session_state.pop("preliminary_report", None)
                                        st.session_state["match_run_id"] = run_id
                                        st.session_state["match_thread_id"] = run_id
                                        st.rerun()
                            else:
                                st.caption("—")
                        with col_delete:
                            can_delete = status in ("completed", "failed", "paused")
                            if st.button(
                                "删除", key=f"delete-match-run-{run_id}", use_container_width=True,
                                disabled=not can_delete,
                                help=None if can_delete else "任务正在运行，请等待完成后再删除",
                            ):
                                _confirm_delete_match_run(run_id, match_run_label(run, all_documents))
                        if status == "failed" and run.get("error_message"):
                            st.caption(f"失败原因：{run['error_message']}")
        else:
            st.caption("暂无历史匹配任务，完成第一次匹配后会显示在这里。")


        if st.session_state.pop("reveal_match_report", False):
            # Trusted presentation code only; never interpolate report/user text here.
            st.html("""<script>requestAnimationFrame(() => requestAnimationFrame(() => {
                document.querySelector('[data-testid="stMain"]')?.scrollTo({top: 0,
                    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
            }));</script>""", unsafe_allow_javascript=True)

    # ----------------------------------------------------------------------
    # 4. 面试准备页面
    # ----------------------------------------------------------------------

    if active_page == "interview":
        page_header("准备你的下一场面试", "基于刚才的岗位匹配报告，生成个人准备方案。", eyebrow="把方向变成行动")

        with st.container(key="interview_setup", border=True):
            history_runs, all_documents = fetch_match_context()
            completed_runs = [r for r in history_runs if r.get("status") == "completed"]

            if completed_runs:
                labels = [match_run_label(r, all_documents) for r in completed_runs]
                current_run_id = st.session_state.get("match_run_id")
                default_index = 0
                for i, run in enumerate(completed_runs):
                    if str(run["run_id"]) == current_run_id:
                        default_index = i
                        break
                selected_index = st.selectbox(
                    "选择岗位匹配任务",
                    options=range(len(completed_runs)),
                    index=default_index,
                    format_func=lambda i: labels[i],
                    help="面试题将基于该任务中匹配出的岗位要求与能力缺口生成",
                )
                interview_run_id = str(completed_runs[selected_index]["run_id"])
            else:
                empty_state("从一次岗位匹配开始", "完成匹配后，在这里获得围绕岗位要求与能力差距的面试问题。")
                st.button("前往岗位匹配", key="interview_open_match", on_click=select_page, args=("match",))
                interview_run_id = None

            interview_left, interview_right = st.columns(2)
            with interview_left:
                interview_difficulty = st.selectbox(
                    "面试难度",
                    options=["junior", "intermediate", "senior"],
                    format_func=lambda value: DIFFICULTY_NAMES[value],
                    key="interview_difficulty",
                )
            with interview_right:
                interview_question_count = st.slider(
                    "预测题目数量", min_value=3, max_value=15,
                    value=None if "interview_count" in st.session_state else 8, key="interview_count",
                )

            if st.button(
                    "生成面试题与回答要点",
                    type="primary",
                    disabled=not interview_run_id,
                    use_container_width=False,
            ):
                try:
                    with st.spinner("正在根据岗位要求和你的经历准备面试题…"):
                        interview_response = create_interview_plan(
                            match_run_id=interview_run_id,
                            difficulty=interview_difficulty,
                            question_count=interview_question_count,
                        )

                    interview_result = interview_response.get("result")
                    if not interview_result:
                        raise APIError("面试接口未返回有效结果")

                    st.session_state["interview_plan"] = interview_result
                    st.session_state["interview_plan_source_run_id"] = interview_run_id
                    st.session_state["interview_plan_run_id"] = str(interview_response.get("run_id") or "")
                    st.success("面试题目生成完成！")
                except APIError as exc:
                    st.error(str(exc))

        interview_plan = st.session_state.get("interview_plan")
        if interview_plan:
            st.divider()
            show_interview_plan(interview_plan)

    # ----------------------------------------------------------------------
    # 5. 学习计划页面
    # ----------------------------------------------------------------------

    if active_page == "learning":
        page_header("为目标留一点时间", "基于刚才的岗位匹配报告，生成个人准备方案。", eyebrow="把方向变成行动")

        with st.container(key="learning_setup", border=True):
            history_runs, all_documents = fetch_match_context()
            completed_runs = [r for r in history_runs if r.get("status") == "completed"]

            if completed_runs:
                labels = [match_run_label(r, all_documents) for r in completed_runs]
                current_run_id = st.session_state.get("match_run_id")
                default_index = 0
                for i, run in enumerate(completed_runs):
                    if str(run["run_id"]) == current_run_id:
                        default_index = i
                        break
                selected_index = st.selectbox(
                    "选择岗位匹配任务",
                    options=range(len(completed_runs)),
                    index=default_index,
                    format_func=lambda i: labels[i],
                    help="系统将基于该匹配任务中找出的缺失技能制定学习路径",
                )
                learning_run_id = str(completed_runs[selected_index]["run_id"])
            else:
                empty_state("先确定你的目标岗位", "完成一次岗位匹配后，结合可投入的时间，制定适合自己的学习安排。")
                st.button("前往岗位匹配", key="learning_open_match", on_click=select_page, args=("match",))
                learning_run_id = None

            learning_left, learning_right = st.columns(2)
            with learning_left:
                available_weeks = st.slider("学习周期（周）", min_value=1, max_value=24,
                                           value=None if "learning_weeks" in st.session_state else 4, key="learning_weeks")
            with learning_right:
                hours_per_week = st.slider("每周投入时间（小时）", min_value=1, max_value=40,
                                          value=None if "learning_hours" in st.session_state else 10, key="learning_hours")

            if st.button(
                    "生成学习计划",
                    type="primary",
                    disabled=not learning_run_id,
                    use_container_width=False,
            ):
                try:
                    with st.spinner("正在安排学习重点与每周计划…"):
                        learning_response = create_learning_plan(
                            match_run_id=learning_run_id,
                            available_weeks=available_weeks,
                            hours_per_week=hours_per_week,
                        )

                    learning_result = learning_response.get("result")
                    if not learning_result:
                        raise APIError("学习计划接口未返回有效结果")

                    st.session_state["learning_plan"] = learning_result
                    st.session_state["learning_plan_source_run_id"] = learning_run_id
                    st.session_state["learning_plan_run_id"] = str(learning_response.get("run_id") or "")
                    st.success("学习路线规划完成！")
                except APIError as exc:
                    st.error(str(exc))

        learning_plan = st.session_state.get("learning_plan")
        if learning_plan:
            st.divider()
            show_learning_plan(learning_plan)

    # ----------------------------------------------------------------------
    # 6. 评测页面（仅管理员）
    # ----------------------------------------------------------------------

    if active_page == "credits":
        render_credits_page()

    if active_page == "evaluation":
        page_header("检索评测", "比较不同检索方案，衡量回答质量与证据支持程度。", eyebrow="管理员工作台")
        with st.expander("评测范围与运行说明"):
            st.caption("仅检索当前账号的就绪文档；各配置复用同一批文档和现有切块，切块参数变化需先建立对应索引。")

        st.session_state.setdefault("experiment_names", {})
        st.session_state.setdefault("variant_count", 0)
        st.session_state.setdefault("show_report_html", None)

        try:
            datasets = get_evaluation_datasets()
            experiments = list_evaluation_experiments()
        except APIError as exc:
            datasets = []
            experiments = []
            st.error(str(exc))

        experiment_title, refresh_col = st.columns([5, 1], vertical_alignment="center")
        experiment_title.subheader("实验列表")
        if refresh_col.button("刷新状态", use_container_width=True):
            st.rerun()
        running_any = False
        if not experiments:
            empty_state("还没有评测实验", "选择数据集并创建实验后，在这里查看运行状态和结果报告。")
        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            status = experiment.get("status")
            name = st.session_state["experiment_names"].get(experiment_id, experiment_id[:8])
            with st.container():
                info_col, state_col = st.columns([3, 2], vertical_alignment="center")
                with info_col:
                    st.markdown(f"**{name}**")
                    st.caption(f"实验 ID：{experiment_id[:8]}…")
                with state_col:
                    if status == "running":
                        running_any = True
                        st.info(f"进行中 · {int(experiment.get('progress', 0) * 100)}%")
                    elif status == "completed":
                        st.success("已完成")
                    elif status == "failed":
                        st.error("失败")
                    else:
                        st.warning(str(status))
                if status == "running":
                    st.progress(experiment.get("progress", 0.0))
                if experiment.get("error"):
                    st.caption(f"失败原因：{experiment['error']}")
                if status == "completed" and experiment.get("report_path"):
                    if st.button("查看报告", key=f"view-report-{experiment_id}", use_container_width=True):
                        try:
                            st.session_state["show_report_html"] = get_evaluation_report_html(experiment["report_path"])
                            st.rerun()
                        except APIError as exc:
                            st.error(str(exc))
        if running_any:
            st.caption("有实验正在运行，点击「刷新状态」更新进度。")

        if st.session_state.get("show_report_html"):
            st.divider()
            report_col, close_col = st.columns([5, 1])
            with report_col:
                st.subheader("报告预览")
            with close_col:
                st.write("")
                if st.button("关闭报告", use_container_width=True):
                    st.session_state["show_report_html"] = None
                    st.rerun()
            render_html_component(st.session_state["show_report_html"], height=900, scrolling=True)
            st.download_button(
                "下载报告 HTML",
                data=st.session_state["show_report_html"],
                file_name="evaluation_report.html",
                mime="text/html",
            )


        def evaluation_config_form(prefix: str, default_name: str) -> dict[str, Any]:
            """渲染一个评测配置（基线或变体），返回与后端 ExperimentConfig 对齐的字典。"""
            name_col, mode_col = st.columns(2)
            config_name = name_col.text_input("配置名称", value=default_name, key=f"{prefix}-name")
            search_mode = mode_col.selectbox(
                "检索策略",
                options=["hybrid", "dense", "bm25"],
                key=f"{prefix}-mode",
                format_func=lambda value: SEARCH_MODE_NAMES[value],
            )
            top_col, rerank_col = st.columns(2)
            top_k = top_col.slider("检索返回数 Top-K", 1, 20, 5, key=f"{prefix}-topk")
            enable_reranker = rerank_col.toggle("启用重排序", value=True, key=f"{prefix}-rerank")
            rerank_top_k = st.slider(
                "重排序返回数", 1, 10, 5, key=f"{prefix}-rtopk", disabled=not enable_reranker,
            )
            if enable_reranker:
                st.caption(f"重排后最多返回 {min(top_k, rerank_top_k)} 条上下文。")
            enable_reflection = st.toggle("启用反思修正", value=False, key=f"{prefix}-reflect")
            max_iterations = st.slider(
                "反思轮数", 1, 5, 2, key=f"{prefix}-iters", disabled=not enable_reflection,
            )
            return {
                "name": config_name.strip() or default_name,
                "search": {
                    "top_k": top_k,
                    "search_mode": search_mode,
                    "enable_reranker": enable_reranker,
                    "reranker_top_k": rerank_top_k,
                },
                "reflection": {
                    "enable_reflection": enable_reflection,
                    "max_iterations": max_iterations,
                },
            }


        st.subheader("新建实验")
        if not datasets:
            st.warning("评测目录（backend/app/evaluation/）下暂无可用数据集，请先放置 JSON 数据集")
        else:
            dataset = st.selectbox(
                "测试数据集",
                options=datasets,
                format_func=lambda d: f"{d['name']}（{d['case_count']} 条）",
                help=datasets[0]["description"] if len(datasets) == 1 else None,
            )
            name_col, concurrency_col, timeout_col = st.columns([2, 1, 1])
            experiment_name = name_col.text_input(
                "实验名称",
                value=f"exp_{datetime.now().strftime('%m%d_%H%M')}",
                key="exp-name",
            )
            max_concurrency = concurrency_col.slider("并发数", 1, 20, 5, key="exp-concurrency")
            timeout_seconds = timeout_col.slider("单查询超时（秒）", 10, 600, 120, step=10, key="exp-timeout")

            with st.expander("基线配置", expanded=True):
                baseline_config = evaluation_config_form("baseline", "baseline")

            variant_count = st.slider("对比变体数量（不含基线）", 0, 3, key="variant_count")
            variant_configs = []
            for i in range(variant_count):
                with st.expander(f"对比变体 {i + 1}"):
                    variant_configs.append(evaluation_config_form(f"variant-{i + 1}", f"variant_{i + 1}"))

            if st.button("启动评测实验", type="primary", use_container_width=True):
                safe_name = experiment_name.strip() or f"exp_{datetime.now().strftime('%m%d_%H%M')}"
                config = {
                    "experiment_name": safe_name,
                    "baseline": baseline_config,
                    "variants": variant_configs,
                    "test_dataset_path": dataset["filename"],
                    "max_concurrency": max_concurrency,
                    "timeout_seconds": timeout_seconds,
                }
                try:
                    created = start_evaluation_experiment(config)
                    st.session_state["experiment_names"][created["experiment_id"]] = safe_name
                    st.success(f"实验已启动：{safe_name}")
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))
