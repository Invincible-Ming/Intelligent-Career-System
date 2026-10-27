"""Small, independently rerunning matching controls; IDs remain the identity."""
import streamlit as st
from career_ui.api import security_options
from career_ui.render import document_label, show_document_preview, match_stepper


def _remember_jd_text():
    if "match_jd_text" in st.session_state:
        st.session_state["match_jd_draft"] = st.session_state["match_jd_text"]


@st.fragment
def render_match_setup(*, resume_documents, job_documents, ready_documents, on_run, on_switch, step_area):
    # Reserve the outside placeholder during the initial fragment render too.
    stage = 3 if st.session_state.get("match_report") or st.session_state.get("match_paused") else 1
    step_area.html(match_stepper(stage))
    with st.container(key="match_setup", border=True):
        # 数据来自本轮开头的一次拉取。若用户刚在「知识库」页上传了新文档，
        # 这里可能仍是旧快照，给一个显式刷新入口比让用户困惑地看到空列表好。
        title_column, refresh_column = st.columns([4, 1.3], vertical_alignment="center")
        title_column.caption("选择一份简历，添加目标岗位要求。")
        if refresh_column.button(
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
            on_change=_remember_jd_text,
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
                on_click=on_switch,
            )

        # Select by immutable ID; a filename is only display text, never an identity.
        resume_by_id = {document["id"]: document for document in resume_documents}
        resume_key = "match_resume_document_id"
        if st.session_state.get(resume_key) not in resume_by_id:
            st.session_state.pop(resume_key, None)
        resume_id = st.selectbox(
            "选择候选人简历",
            options=list(resume_by_id),
            format_func=lambda identifier: document_label(resume_by_id[identifier]),
            key=resume_key,
        )
        selected_resume = resume_by_id.get(resume_id)
        st.caption("最近上传优先 · 北京时间")
        if selected_resume and st.button("查看所选简历原文", key="preview-match-resume"):
            show_document_preview(selected_resume)

        selected_job = None
        if jd_source == "已上传文档":
            job_by_id = {document["id"]: document for document in job_documents}
            job_key = "match_job_document_id"
            if st.session_state.get(job_key) not in job_by_id:
                st.session_state.pop(job_key, None)
            job_id = st.selectbox(
                "选择岗位描述文档",
                options=list(job_by_id),
                format_func=lambda identifier: document_label(job_by_id[identifier]),
                key=job_key,
            )
            selected_job = job_by_id.get(job_id)
            if selected_job and st.button("查看所选岗位原文", key="preview-match-job"):
                show_document_preview(selected_job)

        # Keep text submission in the form; selectors stay outside for immediate previews.
        jd_text = ""
        if "match_jd_text" not in st.session_state:
            st.session_state["match_jd_text"] = st.session_state.get("match_jd_draft", "")
        if jd_source == "直接输入":
            jd_text = st.text_area(
                "输入岗位描述 (JD)",
                placeholder="粘贴岗位名称、职责要求与技术栈门槛...",
                height=200,
                max_chars=security_options()["jd_max_input_chars"],
                key="match_jd_text",
                on_change=_remember_jd_text,
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
        match_submitted = st.button(
            "开始岗位匹配",
            type="primary",
            use_container_width=False,
        )

        if match_submitted:
            if not can_match:
                st.warning("请先选择简历，并补全岗位描述（直接输入正文或选择已上传文档）。")
            else:
                on_run(
                    selected_resume=selected_resume,
                    selected_job=selected_job,
                    jd_source=jd_source,
                    jd_text=jd_text,
                    research_approved=research_approved,
                )

