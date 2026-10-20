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
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import streamlit as st
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

DOCUMENT_TYPE_NAMES = {
    "resume": "简历",
    "job_description": "岗位描述",
    "knowledge": "知识资料",
}

DIFFICULTY_NAMES = {
    "junior": "初级",
    "intermediate": "中级",
    "senior": "高级",
}

SEARCH_MODE_NAMES = {
    "hybrid": "混合检索（Milvus + BM25 + RRF + BGE）",
    "dense": "语义检索（Milvus）",
    "bm25": "关键词检索（BM25）",
}

# ----------------------------------------------------------------------
# 页面配置与界面样式
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="智能求职 Multi-Agent 系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep presentation separate from authentication and business logic.
st.markdown(
    "<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)


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
    stream_chat_api,
    stream_match_api,
    stream_match_resume_api,
)
from career_ui.config import (  # noqa: E402
    DIFFICULTY_NAMES,
    DOCUMENT_TYPE_NAMES,
    SEARCH_MODE_NAMES,
)
from career_ui.endpoints import (  # noqa: E402
    MATCH_STATUS_NAMES,
    create_interview_plan,
    create_learning_plan,
    delete_chat_conversation,
    delete_document,
    fetch_match_context,
    get_conversation_messages,
    get_conversations,
    get_documents,
    get_evaluation_datasets,
    get_evaluation_report_html,
    list_evaluation_experiments,
    match_report_of,
    match_run_label,
    search_documents,
    start_evaluation_experiment,
    update_chat_conversation,
    upload_document,
)
from career_ui.render import (  # noqa: E402
    document_label,
    match_progress_updates,
    show_interview_plan,
    show_learning_plan,
    show_match_report,
    show_search_results,
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


def _run_match_flow(
        *,
        selected_resume: dict[str, Any],
        selected_job: dict[str, Any] | None,
        jd_source: str,
        jd_text: str,
        research_approved: bool = False,
) -> None:
    """执行岗位匹配流式请求并把结果写入 session_state。"""

    payload: dict[str, Any] = {
        "resume_document_id": selected_resume["id"], "research_approved": research_approved,
    }
    if jd_source == "直接输入":
        payload["jd_text"] = jd_text.strip()
    else:
        payload["jd_document_id"] = selected_job["id"]

    # 清空历史状态
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
                        st.write(f"**[Resume Agent]** {msg}")
                    elif node == "jd_agent":
                        st.write(f"**[JD Agent]** {msg}")
                    elif node == "match_agent":
                        score = data.get("score")
                        score_info = f" (初评总分: {score}分)" if score else ""
                        st.write(f"**[Match Agent]** {msg}{score_info}")
                    elif node == "verify_agent":
                        if data.get("status") == "passed":
                            status_box.write(f"**[Verify Agent]** {msg}")
                        else:
                            status_box.warning(msg)
                            if data.get("feedback"):
                                status_box.caption(data["feedback"])

                # 捕获人机协同挂起事件
                elif event_type == "interrupt":
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
                        st.session_state["match_report"] = report
                        status_box.update(
                            label="岗位多维匹配与事实校验全部完成！",
                            state="complete",
                            expanded=False,
                        )

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
            background: radial-gradient(ellipse at 50% 20%, #ede9fe 0%, #f8fafc 65%);
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
            padding: 2rem; border: 1px solid #e8eaf2; border-radius: 24px;
            background: #ffffff; color: #172033;
            box-shadow: 0 16px 60px rgba(41, 35, 90, 0.08);
        }
        .auth-brand { text-align: center; font-size: 2.3rem; color: #7256da; }
        .auth-title { text-align: center; font-size: 1.65rem; font-weight: 700; margin-top: .4rem; }
        .auth-subtitle { text-align: center; color: #778196; font-size: .95rem; margin: .65rem 0 1.4rem; }
        .st-key-auth_card [data-testid="stForm"] { border: 0; padding: 0; }
        .st-key-auth_card [data-baseweb="tab-list"] { gap: 0; }
        .st-key-auth_card [data-baseweb="tab"] { flex: 1; justify-content: center; }
        .st-key-auth_card [data-testid="stFormSubmitButton"] button {
            border-radius: 10px; min-height: 44px;
        }
        @media (max-width: 520px) {
            .st-key-auth_card { padding: 1.5rem; border-radius: 18px; }
        }
        @media (prefers-color-scheme: dark) {
            [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 50% 20%, #28213f, #10141c 65%); }
            .st-key-auth_card { background: #171c28; color: #eef1f8; border-color: #30364a; }
            .auth-subtitle { color: #a1aabd; }
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container(key='auth_card'):
            st.markdown(
                '<div class="auth-title">智能求职助手</div>'
                '<div class="auth-subtitle">登录，开启你的求职准备</div>',
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


def start_new_chat():
    st.session_state.active_page = "chat"
    st.session_state.conversation_id = None
    st.session_state.chat_messages = []
    st.session_state.rename_target = None


st.session_state.setdefault("active_page", "chat")
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("rename_target", None)

is_admin = st.session_state.auth_user.get("is_admin", False)

page_options = {
    "chat": "智能对话",
    "knowledge": "知识库",
    "match": "岗位匹配",
    "interview": "面试准备",
    "learning": "学习计划",
}
if is_admin:
    page_options["evaluation"] = "评测"
active_page = st.session_state.active_page

# 页面级侧边栏：仅智能对话显示（会话列表/账户区都在侧边栏），其余页面整栏隐藏。
if active_page != "chat":
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] {display: none;}</style>",
        unsafe_allow_html=True,
    )

with navigation_area.container():
    with st.container(key="top_navigation"):
        for column, (page, label) in zip(st.columns(len(page_options), gap="small"), page_options.items()):
            with column:
                st.button(
                    label, key=f"nav_{page}", use_container_width=True,
                    type="primary" if active_page == page else "tertiary",
                    on_click=select_page, args=(page,),
                )

with chat_sidebar_area.container():
    st.markdown('<div class="sidebar-brand">求职助手<span>CAREER AI</span></div>', unsafe_allow_html=True)
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
                            history_msgs = get_conversation_messages(conv_id)
                            st.session_state.conversation_id = conv_id
                            st.session_state.chat_messages = [
                                {"role": msg["role"], "content": msg["content"]} for msg in history_msgs
                            ]
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
                            try:
                                delete_chat_conversation(conv_id)
                                if st.session_state.conversation_id == conv_id:
                                    start_new_chat()
                                st.toast("对话已删除")
                                st.rerun()
                            except APIError as exc:
                                st.error(f"删除失败：{exc}")

with account_area.container():
    with st.container(key="sidebar_account"):
        username = escape(st.session_state.auth_user['username'])
        # ChatGPT 风格：点击底部用户区域弹出账户菜单，退出登录收纳进菜单。
        with st.popover(username, icon=":material/person:", use_container_width=True):
            st.caption(f"当前登录：{username}")
            # 账户配额展示：管理员不限次数，普通用户展示当日/当分钟余量
            with st.container(key="quota_section"):
                try:
                    usage = request_api('GET', '/api/auth/usage')
                except APIError:
                    usage = None
                if usage and usage.get('is_admin'):
                    st.caption('模型调用次数：不限')
                elif usage:
                    daily = usage['generation_daily']
                    minute = usage['generation_minute']
                    st.caption('今日生成次数')
                    ratio = min(daily['used'] / daily['limit'], 1.0) if daily['limit'] else 0.0
                    st.progress(ratio, text=f"{daily['used']} / {daily['limit']}")
                    st.caption(f"剩余 {daily['remaining']} 次 · 每日北京时间 08:00 重置")
                    st.caption(f"本分钟：{minute['used']} / {minute['limit']}")
            st.link_button('订阅方案与价格', f"{API_BASE_URL}/pricing",
                           icon=":material/workspace_premium:", use_container_width=True)
            if st.button('退出登录', icon=":material/logout:", use_container_width=True):
                try:
                    request_api('POST', '/api/auth/logout')
                except APIError:
                    pass
                private_area.empty()
                account_area.empty()
                navigation_area.empty()
                chat_sidebar_area.empty()
                reset_login_state()

# At root level Streamlit pins this input to the bottom and handles sidebar resizing.
prompt_input = None
if active_page == "chat":
    prompt_input = st.chat_input("向求职助手提问…", key="chat_prompt",
                                 max_chars=security_options()["chat_max_input_chars"])

with private_area.container():
    # ----------------------------------------------------------------------
    # 1. 智能对话页面
    # ----------------------------------------------------------------------

    if active_page == "chat":
        # 欢迎语固定放在对话页顶部；快捷主题已移除，减少首屏占用。
        if not st.session_state.chat_messages:
            username = escape(st.session_state.auth_user['username'])
            pricing_url = f"{API_BASE_URL}/pricing"
            st.markdown(f'<div class="chat-welcome"><h1>你好，{username}</h1>'
                        f'<p>今天想为求职做些什么？ <a class="pricing-link" href="{pricing_url}" target="_blank">查看订阅方案 →</a></p></div>',
                        unsafe_allow_html=True)

        chat_scroll_area = st.container(key="chat_history")

        with chat_scroll_area:
            # 消息随页面滚动，输入框保持在底部。
            for message in st.session_state.chat_messages:
                avatar = ":material/auto_awesome:" if message["role"] == "assistant" else ":material/person:"
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"])

        active_prompt = prompt_input

        if active_prompt:
            st.session_state.chat_messages.append({"role": "user", "content": active_prompt})

            with chat_scroll_area:
                with st.chat_message("user", avatar=":material/person:"):
                    st.write(active_prompt)

                with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                    try:
                        stream_generator = stream_chat_api(
                            prompt=active_prompt,
                            conversation_id=st.session_state.conversation_id,
                        )
                        assistant_message = st.write_stream(stream_generator)

                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": assistant_message}
                        )
                    except APIError as exc:
                        error_msg = f"对话流中断：{exc}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

    # ----------------------------------------------------------------------
    # 2. 知识库管理页面
    # ----------------------------------------------------------------------

    if active_page == "knowledge":
        st.header("知识库与文档管理")

        # 产品规则：普通用户可上传简历与岗位描述；知识资料由管理员维护。
        upload_types = ["resume", "job_description", "knowledge"] if is_admin else ["resume", "job_description"]

        if is_admin:
            upload_column, search_column = st.columns(2)
        else:
            upload_column, search_column = st.container(), None

        with upload_column:
            st.subheader("上传新文档")

            # 上传结果提示：必须在 rerun 之后渲染，因此读session_state。
            notice = st.session_state.pop("upload_notice", None)
            if notice:
                if notice["kind"] == "dedup":
                    st.info(f"ℹ️ {notice['text']}")
                else:
                    st.success(notice["text"])

            upload_type = st.selectbox(
                "文档类型",
                options=upload_types,
                format_func=lambda value: DOCUMENT_TYPE_NAMES[value],
                help=None if is_admin else "知识资料由管理员维护，如需上传请联系管理员",
            )

            # file_uploader 无法通过 session_state 直接清值；
            # 上传成功后轮换 key 强制重建组件，即可清空已选文件。
            uploader_key = f"knowledge_uploader_{st.session_state.get('uploader_nonce', 0)}"
            uploaded_file = st.file_uploader(
                "选择文档文件",
                type=["pdf", "docx", "xlsx", "txt"],
                key=uploader_key,
                help="支持 PDF、DOCX、XLSX 和 TXT 格式",
            )

            if st.button(
                    "上传并向量化入库",
                    type="primary",
                    disabled=uploaded_file is None,
                    use_container_width=True,
            ):
                try:
                    with st.spinner("正在解析、切片、向量化并建立索引..."):
                        upload_result = upload_document(uploaded_file, upload_type)

                    # 成功后轮换 key：下次渲染是全新的上传框（失败时保留文件便于重试）。
                    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1

                    document = upload_result.get("document") or {}
                    # st.rerun() 会清掉本轮渲染输出，
                    # 提示语必须存进 session_state 才能显示在下一轮。
                    st.session_state["upload_notice"] = (
                        {
                            "kind": "dedup",
                            "text": (
                                f"这份内容已上传过："
                                f"{document.get('filename', uploaded_file.name)}"
                                f"（{document.get('chunk_count', 0)} 个文本块），"
                                "本次未重复解析与向量化。"
                            ),
                        }
                        if upload_result.get("deduplicated")
                        else {
                            "kind": "new",
                            "text": (
                                f"上传成功：{uploaded_file.name}，"
                                f"生成 {document.get('chunk_count', 0)} 个文本块"
                            ),
                        }
                    )
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

        if is_admin:
            with search_column:
                st.subheader("多路混合检索验证")
                search_query = st.text_area(
                    "输入检索 Query",
                    placeholder="例如：候选人是否具备微服务架构重构实践？",
                    height=120,
                    max_chars=security_options()["search_max_input_chars"],
                )

                search_mode = st.selectbox(
                    "检索策略",
                    options=["hybrid", "dense", "bm25"],
                    format_func=lambda value: SEARCH_MODE_NAMES[value],
                )

                search_document_type = st.selectbox(
                    "过滤范围",
                    options=[None, "resume", "job_description", "knowledge"],
                    format_func=lambda value: "全部文档"
                    if value is None
                    else DOCUMENT_TYPE_NAMES[value],
                )

                search_top_k = st.slider("返回数量 Top-K", min_value=1, max_value=10, value=5)

                if st.button(
                        "执行检索",
                        disabled=not search_query.strip(),
                        use_container_width=True,
                ):
                    try:
                        with st.spinner(f"正在执行 {SEARCH_MODE_NAMES[search_mode]}..."):
                            search_results = search_documents(
                                query=search_query,
                                document_type=search_document_type,
                                top_k=search_top_k,
                                search_mode=search_mode,
                            )
                        st.session_state["search_results"] = search_results
                        st.session_state["search_mode"] = search_mode
                    except APIError as exc:
                        st.error(str(exc))

            current_search_results = st.session_state.get("search_results", [])
            current_search_mode = st.session_state.get("search_mode", "hybrid")

            if current_search_results:
                st.divider()
                st.subheader("检索结果呈现")
                show_search_results(current_search_results, current_search_mode)

        st.divider()
        st.subheader("已入库文档列表")

        try:
            documents = get_documents()
        except APIError as exc:
            documents = []
            st.error(str(exc))

        if not documents:
            st.info("暂未上传文档")
        else:
            for document in documents:
                first, second, third, fourth = st.columns([4, 2, 2, 1])
                filename = document.get("filename", "未命名文档")
                document_id = document.get("id", "")

                first.markdown(f"**{filename}**")
                created_at = (str(document.get("created_at") or ""))[:16].replace("T", " ")
                first.caption(f"入库时间：{created_at}" if created_at else f"ID：{document_id[:8]}…")

                document_type = document.get("document_type", "")
                second.write(DOCUMENT_TYPE_NAMES.get(document_type, document_type))

                status = document.get("status", "unknown")
                chunk_count = document.get("chunk_count", 0)

                if status == "ready":
                    third.success(f"已就绪 · {chunk_count} 块")
                elif status == "failed":
                    third.error("处理失败")
                else:
                    third.warning(status)

                if fourth.button("删除", key=f"delete-document-{document_id}"):
                    try:
                        delete_document(document_id)
                        st.success("文档删除成功")
                        st.rerun()
                    except APIError as exc:
                        st.error(str(exc))

    # ----------------------------------------------------------------------
    # 3. 岗位匹配页面 (支持 SSE 节点进度流、断点恢复与 Human-in-the-loop)
    # ----------------------------------------------------------------------

    if active_page == "match":
        st.header("简历与岗位多维匹配")

        # ==================================================================
        # 任务历史记录与断点恢复控制台
        # ==================================================================
        history_runs, all_documents = fetch_match_context()

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
                        col_info, col_state, col_action = st.columns([5, 1.6, 1.4], vertical_alignment="center")
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
                                        st.session_state["match_paused"] = False
                                        st.session_state.pop("preliminary_report", None)
                                        st.session_state["match_run_id"] = run_id
                                        st.session_state["match_thread_id"] = run_id
                                        st.rerun()
                            else:
                                st.caption("—")
                        if status == "failed" and run.get("error_message"):
                            st.caption(f"失败原因：{run['error_message']}")
        else:
            st.caption("暂无历史匹配任务，完成第一次匹配后会显示在这里。")

        st.divider()

        ready_documents = [d for d in all_documents if d.get("status") == "ready"]
        resume_documents = [
            d for d in ready_documents if d.get("document_type") == "resume"
        ]
        job_documents = [
            d for d in ready_documents if d.get("document_type") == "job_description"
        ]

        if not resume_documents:
            st.warning("请先在“知识库”页面上传一份已就绪的简历文档")
        else:
            # 数据来自本轮开头的一次拉取。若用户刚在「知识库」页上传了新文档，
            # 这里可能仍是旧快照，给一个显式刷新入口比让用户困惑地看到空列表好。
            if st.button(
                "刷新文档列表",
                key="refresh_match_documents",
                help="在知识库页上传了新文档后，点击此处拉取最新列表",
            ):
                st.rerun()

            # 来源切换是轻量交互，放在 form 外：
            # form 内控件只有提交时才返回值，把 radio 放进去会导致
            # 切换来源时整个 form 重跑、状态互相覆盖。
            jd_source = st.radio(
                "岗位描述 (JD) 来源",
                options=["直接输入", "已上传文档"],
                horizontal=True,
                key="jd_source_choice",
            )

            # 没有 JD 文档时，在 form 外给出提示与逃生出口。
            # 放在 form 内的按钮无法用 st.button（Streamlit 会直接报错），
            # 而 form_submit_button 会触发整表重跑、状态互相覆盖。
            if jd_source == "已上传文档" and not job_documents:
                st.warning(
                    "你还没有上传过「岗位描述」类型的文档，"
                    "所以这里无法选择已上传的 JD。"
                )
                st.caption(
                    f"当前可用的已就绪文档共 {len(ready_documents)} 份，"
                    f"其中简历 {len(resume_documents)} 份。"
                    "请到「知识库」页上传岗位描述，或切换回「直接输入」。"
                )
                # on_click 回调在本次脚本运行之前执行，
                # 因此可以直接改session_state[jd_source_choice] 而不触发
                # StreamlitWidgetAlreadyInstantiatedError。
                st.button(
                    "改用直接输入 JD",
                    key="switch_to_jd_text",
                    use_container_width=True,
                    on_click=_switch_to_direct_jd_input,
                )

            # 用 form 包裹：JD 文本较长，裸控件在每次 rerun 时会丢输入。
            with st.form("match_form", border=False):
                selected_resume = st.selectbox(
                    "选择候选人简历",
                    options=resume_documents,
                    format_func=document_label,
                )

                selected_job = None
                jd_text = ""

                if jd_source == "直接输入":
                    jd_text = st.text_area(
                        "输入岗位描述 (JD)",
                        placeholder="粘贴岗位名称、职责要求与技术栈门槛...",
                        height=200,
                        max_chars=security_options()["jd_max_input_chars"],
                    )
                else:
                    # 无 JD 文档的情形已在 form 外处理（提示 + 逃生按钮）。
                    selected_job = st.selectbox(
                        "选择岗位描述文档",
                        options=job_documents,
                        format_func=document_label,
                    )

                research_approved = st.checkbox("补充检索公开岗位资料", value=False)
                can_match = bool(
                    selected_resume
                    and (
                            (jd_source == "直接输入" and jd_text.strip())
                            or (jd_source == "已上传文档" and selected_job)
                    )
                )

                # 不置灰：Streamlit 的 form 提交值只在下一轮脚本运行才可用，
                # 禁用态会依据上一轮的输入判定，导致刚填完 JD 仍点不动。
                match_submitted = st.form_submit_button(
                    "开始岗位匹配",
                    type="primary",
                    use_container_width=True,
                )

            if match_submitted:
                if not can_match:
                    st.warning("请先选择简历，并补全岗位描述（直接输入正文或选择已上传文档）。")
                else:
                    _run_match_flow(
                        selected_resume=selected_resume,
                        selected_job=selected_job,
                        jd_source=jd_source,
                        jd_text=jd_text,
                        research_approved=research_approved,
                    )

        # ==================================================================
        # 人机协同 (Human-in-the-loop) 交互卡片
        # ==================================================================
        if st.session_state.get("match_paused"):
            st.divider()
            st.warning("**自动校验未通过或已中止，请审核报告；可填写修改意见重新评估。**")

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
                if st.button("放弃此任务", use_container_width=True):
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
                                    resume_box.write(f"**[Match Agent 重新打分]** {msg}")
                                elif node == "verify_agent":
                                    if status == "passed":
                                        resume_box.write(f"**[Verify Agent]** {msg}")
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
                                        label="人工确认已完成" if data.get("manual_accepted") else "匹配报告自动校验完成！",
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
            show_match_report(match_report)
            st.caption(f"任务持久化 ID：`{st.session_state.get('match_run_id', '')}`")

    # ----------------------------------------------------------------------
    # 4. 面试准备页面
    # ----------------------------------------------------------------------

    if active_page == "interview":
        st.header("面试真题预测")

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
            st.info("暂无已完成的岗位匹配任务。请先在「岗位匹配」页完成一次匹配，再回来生成面试题。")
            interview_run_id = None

        interview_left, interview_right = st.columns(2)
        with interview_left:
            interview_difficulty = st.selectbox(
                "面试难度",
                options=["junior", "intermediate", "senior"],
                format_func=lambda value: DIFFICULTY_NAMES[value],
            )
        with interview_right:
            interview_question_count = st.slider(
                "预测题目数量", min_value=3, max_value=15, value=8
            )

        if st.button(
                "生成专属面试预测与要点",
                type="primary",
                disabled=not interview_run_id,
                use_container_width=True,
        ):
            try:
                with st.spinner("Interview Agent 正在结合候选人经历与能力缺口生成面试题目..."):
                    interview_response = create_interview_plan(
                        match_run_id=interview_run_id,
                        difficulty=interview_difficulty,
                        question_count=interview_question_count,
                    )

                interview_result = interview_response.get("result")
                if not interview_result:
                    raise APIError("面试接口未返回有效结果")

                st.session_state["interview_plan"] = interview_result
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
        st.header("定制能力提升计划")

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
            st.info("暂无已完成的岗位匹配任务。请先在「岗位匹配」页完成一次匹配，再回来制定学习计划。")
            learning_run_id = None

        learning_left, learning_right = st.columns(2)
        with learning_left:
            available_weeks = st.slider("学习周期（周）", min_value=1, max_value=24, value=4)
        with learning_right:
            hours_per_week = st.slider("每周投入时间（小时）", min_value=1, max_value=40, value=10)

        if st.button(
                "制定针对性提升计划",
                type="primary",
                disabled=not learning_run_id,
                use_container_width=True,
        ):
            try:
                with st.spinner("Learning Agent 正在计算技能优先级与制定周计划..."):
                    learning_response = create_learning_plan(
                        match_run_id=learning_run_id,
                        available_weeks=available_weeks,
                        hours_per_week=hours_per_week,
                    )

                learning_result = learning_response.get("result")
                if not learning_result:
                    raise APIError("学习计划接口未返回有效结果")

                st.session_state["learning_plan"] = learning_result
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

    if active_page == "evaluation":
        st.header("RAG 评测实验")
        st.caption("对检索与回答质量做 A/B 对比评测；仅管理员可见")
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

        st.subheader("实验列表")
        refresh_col, _ = st.columns([1, 5])
        if refresh_col.button("刷新状态", use_container_width=True):
            st.rerun()
        running_any = False
        if not experiments:
            st.info("暂无评测实验。从下方创建实验后，进度会实时显示在这里。")
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

            st.markdown("**基线配置**")
            baseline_config = evaluation_config_form("baseline", "baseline")

            variant_count = st.slider("对比变体数量（不含基线）", 0, 3, key="variant_count")
            variant_configs = []
            for i in range(variant_count):
                st.markdown(f"**变体 {i + 1}**")
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
