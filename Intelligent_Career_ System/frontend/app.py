"""
智能求职 Multi-Agent 系统 Streamlit 前端。

启动：

    cd /Users/crazyzm/pythonCode/ZhangXueFengSkill
    conda activate career-agent

    export API_BASE_URL=http://localhost:8000

    python -m streamlit run frontend/app.py \
        --server.port 8501
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

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
# 页面配置
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="智能求职 Multi-Agent 系统",
    page_icon="AI",
    layout="wide",
)


# ----------------------------------------------------------------------
# API 客户端
# ----------------------------------------------------------------------


class APIError(Exception):
    """后端 API 请求异常。"""


def request_api(
        method: str,
        path: str,
        *,
        timeout: int = SHORT_TIMEOUT,
        **kwargs: Any,
) -> Any:
    """调用 FastAPI，并统一处理错误。"""

    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=timeout,
            **kwargs,
        )
    except requests.ConnectionError as exc:
        raise APIError(
            f"无法连接后端服务：{API_BASE_URL}"
        ) from exc
    except requests.Timeout as exc:
        raise APIError(
            "请求超时，请检查后端日志后重试"
        ) from exc
    except requests.RequestException as exc:
        raise APIError(
            f"请求失败：{exc}"
        ) from exc

    if not response.ok:
        try:
            error_data = response.json()
            detail = error_data.get(
                "detail",
                error_data,
            )
        except ValueError:
            detail = response.text or response.reason

        raise APIError(
            f"HTTP {response.status_code}：{detail}"
        )

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError as exc:
        raise APIError(
            "后端返回了无法解析的数据"
        ) from exc


def get_health() -> dict[str, Any]:
    """获取后端及基础设施状态。"""

    return request_api(
        "GET",
        "/health",
    )


def get_documents() -> list[dict[str, Any]]:
    """获取文档列表。"""

    return request_api(
        "GET",
        "/api/documents",
    )


def upload_document(
        uploaded_file: Any,
        document_type: str,
) -> dict[str, Any]:
    """上传文档。"""

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
            or "application/octet-stream",
        ),
    }

    return request_api(
        "POST",
        "/api/documents/upload",
        files=files,
        data={
            "document_type": document_type,
        },
        timeout=LONG_TIMEOUT,
    )


def delete_document(
        document_id: str,
) -> None:
    """删除文档。"""

    request_api(
        "DELETE",
        f"/api/documents/{document_id}",
        timeout=LONG_TIMEOUT,
    )


def search_documents(
        *,
        query: str,
        document_type: str | None,
        top_k: int,
        search_mode: str,
) -> list[dict[str, Any]]:
    """调用 Dense、BM25 或 Hybrid 检索接口。"""

    endpoints = {
        "dense": "/api/search",
        "bm25": "/api/search/bm25",
        "hybrid": "/api/search/hybrid",
    }

    endpoint = endpoints.get(
        search_mode,
        "/api/search/hybrid",
    )

    return request_api(
        "POST",
        endpoint,
        json={
            "query": query,
            "document_type": document_type,
            "top_k": top_k,
        },
        timeout=LONG_TIMEOUT,
    )


def create_match(
        payload: dict[str, Any],
) -> dict[str, Any]:
    """调用岗位匹配接口。"""

    return request_api(
        "POST",
        "/api/match",
        json=payload,
        timeout=LONG_TIMEOUT,
    )


def create_interview_plan(
        *,
        match_run_id: str,
        difficulty: str,
        question_count: int,
) -> dict[str, Any]:
    """调用面试问题接口。"""

    return request_api(
        "POST",
        "/api/interview",
        json={
            "match_run_id": match_run_id,
            "difficulty": difficulty,
            "question_count": question_count,
        },
        timeout=LONG_TIMEOUT,
    )


def create_learning_plan(
        *,
        match_run_id: str,
        available_weeks: int,
        hours_per_week: int,
) -> dict[str, Any]:
    """调用学习计划接口。"""

    return request_api(
        "POST",
        "/api/learning-plan",
        json={
            "match_run_id": match_run_id,
            "available_weeks": available_weeks,
            "hours_per_week": hours_per_week,
        },
        timeout=LONG_TIMEOUT,
    )


# ----------------------------------------------------------------------
# 通用显示函数
# ----------------------------------------------------------------------


def document_label(
        document: dict[str, Any],
) -> str:
    """生成文档下拉框显示文本。"""

    document_type = document.get(
        "document_type",
        "unknown",
    )

    type_name = DOCUMENT_TYPE_NAMES.get(
        document_type,
        document_type,
    )

    return (
        f"{document.get('filename', '未命名文档')} "
        f"· {type_name} "
        f"· {document.get('status', 'unknown')}"
    )


def show_string_list(
        title: str,
        values: list[str] | None,
) -> None:
    """显示字符串列表。"""

    st.markdown(f"#### {title}")

    if not values:
        st.caption("暂无")
        return

    for value in values:
        st.markdown(f"- {value}")


def show_match_report(
        report: dict[str, Any],
) -> None:
    """展示岗位匹配报告。"""

    st.subheader("岗位匹配结果")

    score_column, level_column = st.columns(2)

    with score_column:
        st.metric(
            "综合匹配分",
            f"{float(report.get('total_score', 0)):.1f}",
        )

    with level_column:
        st.metric(
            "匹配等级",
            report.get("match_level", "未知"),
        )

    scores = report.get("scores", {})

    st.markdown("#### 分项得分")

    score_columns = st.columns(5)

    score_items = [
        ("技能匹配", "skill"),
        ("经历匹配", "experience"),
        ("职责匹配", "responsibility"),
        ("教育背景", "education"),
        ("加分项", "bonus"),
    ]

    for column, (label, key) in zip(
            score_columns,
            score_items,
            strict=True,
    ):
        column.metric(
            label,
            f"{float(scores.get(key, 0)):.0f}",
        )

    left_column, right_column = st.columns(2)

    with left_column:
        show_string_list(
            "已匹配技能",
            report.get("matched_skills"),
        )

        show_string_list(
            "候选人优势",
            report.get("strengths"),
        )

    with right_column:
        show_string_list(
            "缺失技能",
            report.get("missing_skills"),
        )

        show_string_list(
            "潜在风险",
            report.get("risks"),
        )

    show_string_list(
        "改进建议",
        report.get("suggestions"),
    )

    with st.expander("查看简历结构化分析"):
        st.json(
            report.get(
                "resume_analysis",
                {},
            )
        )

    with st.expander("查看岗位结构化分析"):
        st.json(
            report.get(
                "job_analysis",
                {},
            )
        )


def show_interview_plan(
        plan: dict[str, Any],
) -> None:
    """展示面试计划。"""

    st.subheader(
        f"面试计划：{plan.get('job_title', '目标岗位')}"
    )

    show_string_list(
        "重点准备方向",
        plan.get("focus_areas"),
    )

    questions = plan.get(
        "questions",
        [],
    )

    if not questions:
        st.info("暂未生成面试问题")
        return

    for index, question in enumerate(
            questions,
            start=1,
    ):
        question_text = question.get(
            "question",
            f"面试问题 {index}",
        )

        with st.expander(
                f"{index}. {question_text}",
                expanded=index == 1,
        ):
            st.markdown(
                f"**问题类型：** "
                f"{question.get('question_type', '未分类')}"
            )

            st.markdown(
                f"**考察目的：** "
                f"{question.get('purpose', '')}"
            )

            show_string_list(
                "回答要点",
                question.get("answer_points"),
            )


def show_learning_plan(
        plan: dict[str, Any],
) -> None:
    """展示学习计划。"""

    st.subheader(
        f"目标岗位：{plan.get('target_role', '未指定')}"
    )

    summary = plan.get(
        "summary",
        "",
    )

    if summary:
        st.write(summary)

    items = plan.get(
        "items",
        [],
    )

    for index, item in enumerate(
            items,
            start=1,
    ):
        priority = item.get(
            "priority",
            "medium",
        )

        priority_name = {
            "high": "高",
            "medium": "中",
            "low": "低",
        }.get(priority, priority)

        skill = item.get(
            "skill",
            f"学习任务 {index}",
        )

        with st.expander(
                f"{index}. {skill} "
                f"｜优先级：{priority_name}",
                expanded=index == 1,
        ):
            st.markdown(
                f"**学习原因：** "
                f"{item.get('reason', '')}"
            )

            st.markdown(
                f"**预计时间：** "
                f"{item.get('estimated_days', 0)} 天"
            )

            show_string_list(
                "具体任务",
                item.get("tasks"),
            )

    show_string_list(
        "每周安排",
        plan.get("weekly_plan"),
    )


def get_search_score_display(
        result: dict[str, Any],
        search_mode: str,
) -> tuple[str, str, float]:
    """
    返回：

    - 分数名称；
    - 召回来源；
    - 用于主标题展示的分数。
    """

    source = result.get(
        "source",
        search_mode,
    )

    if source == "hybrid_reranked":
        return (
            "BGE 重排分",
            "Milvus + BM25 + RRF + BGE",
            float(
                result.get(
                    "rerank_score",
                    result.get("score", 0.0),
                )
            ),
        )

    if source == "hybrid":
        return (
            "RRF 分数",
            "Milvus + BM25 + RRF",
            float(
                result.get(
                    "rrf_score",
                    result.get("score", 0.0),
                )
            ),
        )

    if source == "bm25":
        return (
            "BM25 分数",
            "BM25",
            float(result.get("score", 0.0)),
        )

    return (
        "余弦相似度",
        "Milvus",
        float(result.get("score", 0.0)),
    )


def show_search_results(
        results: list[dict[str, Any]],
        search_mode: str,
) -> None:
    """展示检索结果及不同检索阶段的分数。"""

    if not results:
        st.info("没有检索到相关内容")
        return

    for index, result in enumerate(
            results,
            start=1,
    ):
        score_name, source_name, main_score = (
            get_search_score_display(
                result,
                search_mode,
            )
        )

        with st.expander(
                f"结果 {index}｜"
                f"{score_name} {main_score:.8f}",
                expanded=index == 1,
        ):
            st.caption(
                f"召回来源：{source_name}"
            )

            st.write(
                result.get(
                    "content",
                    "",
                )
            )

            rrf_score = result.get(
                "rrf_score"
            )

            rerank_score = result.get(
                "rerank_score"
            )

            if rrf_score is not None:
                st.caption(
                    f"RRF 分数：{float(rrf_score):.8f}"
                )

            if rerank_score is not None:
                st.caption(
                    f"BGE 重排分："
                    f"{float(rerank_score):.8f}"
                )

            document_type = result.get(
                "document_type",
                "",
            )

            document_type_name = (
                DOCUMENT_TYPE_NAMES.get(
                    document_type,
                    document_type,
                )
            )

            st.caption(
                f"文档类型：{document_type_name}"
            )

            st.caption(
                f"文档 ID："
                f"{result.get('document_id', '')}"
            )


# ----------------------------------------------------------------------
# 页面标题
# ----------------------------------------------------------------------

st.title("智能求职 Multi-Agent 系统")

st.caption(
    "基于百炼、LangGraph、Milvus、BM25、RRF、"
    "BGE Reranker、PostgreSQL 和 MinIO"
)

# ----------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("系统状态")

    st.caption(
        f"后端地址：{API_BASE_URL}"
    )

    if st.button(
            "刷新服务状态",
            use_container_width=True,
    ):
        st.rerun()

    try:
        health = get_health()
        services = health.get(
            "services",
            {},
        )

        service_items = [
            ("postgresql", "PostgreSQL"),
            ("milvus", "Milvus"),
            ("minio", "MinIO"),
        ]

        for key, name in service_items:
            if services.get(key):
                st.success(f"{name}：正常")
            else:
                st.error(f"{name}：不可用")

    except APIError as exc:
        st.error(str(exc))

    st.divider()

    st.caption(
        "状态数据库：PostgreSQL\n\n"
        "向量数据库：Milvus\n\n"
        "文件存储：MinIO\n\n"
        "语义检索：Milvus\n\n"
        "关键词检索：BM25\n\n"
        "排序融合：RRF\n\n"
        "语义重排：BGE CrossEncoder"
    )

# ----------------------------------------------------------------------
# 页面标签
# ----------------------------------------------------------------------

chat_tab, knowledge_tab, match_tab, interview_tab, learning_tab = (
    st.tabs(
        [
            "💬 智能对话",
            "知识库",
            "岗位匹配",
            "面试准备",
            "学习计划",
        ]
    )
)
# ----------------------------------------------------------------------
# 智能对话页面 ⭐ 新增
# ----------------------------------------------------------------------

with chat_tab:
    st.header("💬 智能对话助手")
    st.caption("支持多轮对话，自动记忆上下文 | 所有对话自动保存到数据库")

    # 初始化会话状态
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # 侧边栏：对话管理
    with st.sidebar:
        st.divider()
        st.subheader("💬 对话管理")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🆕 新对话", use_container_width=True):
                st.session_state.conversation_id = None
                st.session_state.chat_messages = []
                st.success("已创建新对话")
                st.rerun()

        with col2:
            if st.button("🔄 刷新", use_container_width=True):
                st.rerun()

        # 显示当前对话信息
        if st.session_state.conversation_id:
            st.success("✅ 对话已连接")
            st.caption(f"ID: {st.session_state.conversation_id[:8]}...")
            st.caption(f"消息数: {len(st.session_state.chat_messages)}")
        else:
            st.info("💡 开始新对话")

    # 显示对话历史
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 用户输入
    if prompt := st.chat_input("输入消息...按 Enter 发送"):
        # 添加用户消息到历史
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })

        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)

        # 调用 API 获取回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                with st.spinner("🤔 思考中..."):
                    # 构建请求
                    payload = {
                        "message": prompt,
                        "stream": False,  # Streamlit 原生不支持 SSE
                    }

                    # 如果有对话 ID，传递以继续对话
                    if st.session_state.conversation_id:
                        payload["conversation_id"] = st.session_state.conversation_id

                    # 调用后端 API
                    response = request_api(
                        "POST",
                        "/api/chat/completions",
                        json=payload,
                        timeout=LONG_TIMEOUT,
                    )

                    # 保存 conversation_id（首次对话）
                    if "conversation_id" in response:
                        st.session_state.conversation_id = response["conversation_id"]

                    # 获取助手回复
                    assistant_message = response.get("content", "抱歉，我没有收到有效的回复。")

                    # 显示回复
                    message_placeholder.write(assistant_message)

                    # 添加到历史
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })

            except APIError as exc:
                error_msg = f"❌ 对话失败：{exc}"
                message_placeholder.error(error_msg)

                # 也添加错误到历史（可选）
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # 欢迎信息（首次进入）
    if not st.session_state.chat_messages:
        st.info("👋 你好！我是智能求职助手，有什么可以帮你的吗？")

        with st.expander("💡 使用提示"):
            st.markdown("""
            **功能特点：**
            - ✅ 支持多轮对话，自动记忆上下文
            - ✅ 所有对话自动保存到数据库
            - ✅ 可以随时新建对话开始新话题
            - ✅ 基于阿里云百炼大模型

            **示例问题：**
            - "Python 后端开发需要掌握哪些技能？"
            - "如何准备技术面试？"
            - "简历中的项目经验应该怎么写？"
            - "数据库索引的作用是什么？"

            **小技巧：**
            - 你可以说"它"、"这个"等代词，我会理解上下文
            - 点击侧边栏的「新对话」开始全新话题
            - 对话会自动保存，下次可以继续
            """)

        # 快速开始按钮
        st.subheader("🚀 快速开始")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💼 求职建议", use_container_width=True):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "我想了解求职准备的建议"
                })
                st.rerun()

        with col2:
            if st.button("🎯 技能学习", use_container_width=True):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "作为后端开发，我应该学习哪些技术？"
                })
                st.rerun()

        with col3:
            if st.button("📝 简历优化", use_container_width=True):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "如何优化简历中的项目经验？"
                })
                st.rerun()

# ----------------------------------------------------------------------
# 知识库页面
# ----------------------------------------------------------------------

with knowledge_tab:
    st.header("知识库管理")

    upload_column, search_column = st.columns(2)

    with upload_column:
        st.subheader("上传文档")

        upload_type = st.selectbox(
            "文档类型",
            options=[
                "resume",
                "job_description",
                "knowledge",
            ],
            format_func=lambda value: (
                DOCUMENT_TYPE_NAMES[value]
            ),
        )

        uploaded_file = st.file_uploader(
            "选择文件",
            type=[
                "pdf",
                "docx",
                "xlsx",
                "txt",
            ],
            help="支持 PDF、DOCX、XLSX 和 TXT",
        )

        if st.button(
                "上传并向量化",
                type="primary",
                disabled=uploaded_file is None,
                use_container_width=True,
        ):
            try:
                with st.spinner(
                        "正在上传、解析、向量化并建立索引..."
                ):
                    upload_result = upload_document(
                        uploaded_file,
                        upload_type,
                    )

                st.success(
                    "上传成功，生成 "
                    f"{upload_result.get('chunk_count', 0)} "
                    "个文本块"
                )

                st.rerun()

            except APIError as exc:
                st.error(str(exc))

    with search_column:
        st.subheader("知识库检索")

        search_query = st.text_area(
            "输入检索问题",
            placeholder=(
                "例如：候选人是否掌握 LangGraph？"
            ),
            height=120,
        )

        search_mode = st.selectbox(
            "检索方式",
            options=[
                "hybrid",
                "dense",
                "bm25",
            ],
            format_func=lambda value: (
                SEARCH_MODE_NAMES[value]
            ),
        )

        search_document_type = st.selectbox(
            "检索范围",
            options=[
                None,
                "resume",
                "job_description",
                "knowledge",
            ],
            format_func=lambda value: (
                "全部文档"
                if value is None
                else DOCUMENT_TYPE_NAMES[value]
            ),
        )

        search_top_k = st.slider(
            "返回结果数",
            min_value=1,
            max_value=10,
            value=5,
        )

        if st.button(
                "开始检索",
                disabled=not search_query.strip(),
                use_container_width=True,
        ):
            try:
                with st.spinner(
                        "正在执行 "
                        f"{SEARCH_MODE_NAMES[search_mode]}..."
                ):
                    search_results = search_documents(
                        query=search_query,
                        document_type=(
                            search_document_type
                        ),
                        top_k=search_top_k,
                        search_mode=search_mode,
                    )

                st.session_state[
                    "search_results"
                ] = search_results

                st.session_state[
                    "search_mode"
                ] = search_mode

            except APIError as exc:
                st.error(str(exc))

    current_search_results = st.session_state.get(
        "search_results",
        [],
    )

    current_search_mode = st.session_state.get(
        "search_mode",
        "hybrid",
    )

    if current_search_results:
        st.divider()
        st.subheader("检索结果")
        show_search_results(
            current_search_results,
            current_search_mode,
        )

    st.divider()
    st.subheader("已上传文档")

    try:
        documents = get_documents()
    except APIError as exc:
        documents = []
        st.error(str(exc))

    if not documents:
        st.info("暂未上传文档")
    else:
        for document in documents:
            first, second, third, fourth = st.columns(
                [4, 2, 2, 1]
            )

            filename = document.get(
                "filename",
                "未命名文档",
            )

            document_id = document.get(
                "id",
                "",
            )

            first.markdown(
                f"**{filename}**"
            )

            first.caption(
                f"ID：{document_id}"
            )

            document_type = document.get(
                "document_type",
                "",
            )

            second.write(
                DOCUMENT_TYPE_NAMES.get(
                    document_type,
                    document_type,
                )
            )

            status = document.get(
                "status",
                "unknown",
            )

            chunk_count = document.get(
                "chunk_count",
                0,
            )

            if status == "ready":
                third.success(
                    f"已就绪 · {chunk_count} 块"
                )
            elif status == "failed":
                third.error("处理失败")

                error_message = document.get(
                    "error_message"
                )

                if error_message:
                    third.caption(error_message)
            else:
                third.warning(status)

            if fourth.button(
                    "删除",
                    key=f"delete-document-{document_id}",
            ):
                try:
                    delete_document(document_id)
                    st.success("文档删除成功")
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

# ----------------------------------------------------------------------
# 岗位匹配页面
# ----------------------------------------------------------------------

with match_tab:
    st.header("简历与岗位匹配")

    try:
        all_documents = get_documents()
    except APIError as exc:
        all_documents = []
        st.error(str(exc))

    ready_documents = [
        item
        for item in all_documents
        if item.get("status") == "ready"
    ]

    resume_documents = [
        item
        for item in ready_documents
        if item.get("document_type") == "resume"
    ]

    job_documents = [
        item
        for item in ready_documents
        if item.get("document_type")
           == "job_description"
    ]

    if not resume_documents:
        st.warning(
            "请先在“知识库”页面上传一份已就绪的简历"
        )
    else:
        selected_resume = st.selectbox(
            "选择简历",
            options=resume_documents,
            format_func=document_label,
        )

        jd_source = st.radio(
            "岗位描述来源",
            options=[
                "直接输入",
                "已上传文档",
            ],
            horizontal=True,
        )

        selected_job = None
        jd_text = ""

        if jd_source == "直接输入":
            jd_text = st.text_area(
                "输入岗位描述",
                placeholder=(
                    "粘贴岗位名称、岗位职责、任职要求等内容"
                ),
                height=230,
            )

        else:
            if not job_documents:
                st.warning(
                    "尚未上传岗位描述文档，"
                    "请切换为“直接输入”"
                )
            else:
                selected_job = st.selectbox(
                    "选择岗位描述文档",
                    options=job_documents,
                    format_func=document_label,
                )

        can_match = bool(
            selected_resume
            and (
                    (
                            jd_source == "直接输入"
                            and jd_text.strip()
                    )
                    or (
                            jd_source == "已上传文档"
                            and selected_job
                    )
            )
        )

        if st.button(
                "开始岗位匹配",
                type="primary",
                disabled=not can_match,
                use_container_width=True,
        ):
            payload: dict[str, Any] = {
                "resume_document_id": selected_resume["id"],
            }

            if jd_source == "直接输入":
                payload["jd_text"] = jd_text.strip()
            else:
                payload["jd_document_id"] = selected_job["id"]

            try:
                with st.spinner(
                        "Resume Agent、JD Agent 和 "
                        "Match Agent 正在分析..."
                ):
                    match_response = create_match(payload)

                run_id = match_response.get(
                    "run_id"
                )

                result = match_response.get(
                    "result"
                )

                if not run_id or not result:
                    raise APIError(
                        "岗位匹配接口没有返回有效结果"
                    )

                st.session_state[
                    "match_run_id"
                ] = run_id

                st.session_state[
                    "match_report"
                ] = result

                st.success("岗位匹配完成")

            except APIError as exc:
                st.error(str(exc))

    match_report = st.session_state.get(
        "match_report"
    )

    if match_report:
        st.divider()
        show_match_report(match_report)

        st.caption(
            "匹配任务 ID："
            f"{st.session_state.get('match_run_id', '')}"
        )

# ----------------------------------------------------------------------
# 面试准备页面
# ----------------------------------------------------------------------

with interview_tab:
    st.header("面试准备")

    interview_run_id = st.text_input(
        "岗位匹配任务 ID",
        value=st.session_state.get(
            "match_run_id",
            "",
        ),
        help="岗位匹配完成后会自动填写",
    )

    interview_left, interview_right = st.columns(2)

    with interview_left:
        interview_difficulty = st.selectbox(
            "面试难度",
            options=[
                "junior",
                "intermediate",
                "senior",
            ],
            format_func=lambda value: (
                DIFFICULTY_NAMES[value]
            ),
        )

    with interview_right:
        interview_question_count = st.slider(
            "问题数量",
            min_value=3,
            max_value=15,
            value=8,
        )

    if st.button(
            "生成面试问题",
            type="primary",
            disabled=not interview_run_id.strip(),
            use_container_width=True,
    ):
        try:
            with st.spinner(
                    "Interview Agent 正在生成面试计划..."
            ):
                interview_response = create_interview_plan(
                    match_run_id=interview_run_id.strip(),
                    difficulty=interview_difficulty,
                    question_count=(
                        interview_question_count
                    ),
                )

            interview_result = interview_response.get(
                "result"
            )

            if not interview_result:
                raise APIError(
                    "面试接口没有返回有效结果"
                )

            st.session_state[
                "interview_plan"
            ] = interview_result

            st.success("面试计划生成完成")

        except APIError as exc:
            st.error(str(exc))

    interview_plan = st.session_state.get(
        "interview_plan"
    )

    if interview_plan:
        st.divider()
        show_interview_plan(interview_plan)

# ----------------------------------------------------------------------
# 学习计划页面
# ----------------------------------------------------------------------

with learning_tab:
    st.header("能力提升计划")

    learning_run_id = st.text_input(
        "岗位匹配任务 ID",
        value=st.session_state.get(
            "match_run_id",
            "",
        ),
        help="学习计划将根据岗位匹配结果生成",
    )

    learning_left, learning_right = st.columns(2)

    with learning_left:
        available_weeks = st.slider(
            "学习周期（周）",
            min_value=1,
            max_value=24,
            value=4,
        )

    with learning_right:
        hours_per_week = st.slider(
            "每周投入时间（小时）",
            min_value=1,
            max_value=40,
            value=10,
        )

    if st.button(
            "生成学习计划",
            type="primary",
            disabled=not learning_run_id.strip(),
            use_container_width=True,
    ):
        try:
            with st.spinner(
                    "Learning Agent 正在制定学习计划..."
            ):
                learning_response = create_learning_plan(
                    match_run_id=learning_run_id.strip(),
                    available_weeks=available_weeks,
                    hours_per_week=hours_per_week,
                )

            learning_result = learning_response.get(
                "result"
            )

            if not learning_result:
                raise APIError(
                    "学习计划接口没有返回有效结果"
                )

            st.session_state[
                "learning_plan"
            ] = learning_result

            st.success("学习计划生成完成")

        except APIError as exc:
            st.error(str(exc))

    learning_plan = st.session_state.get(
        "learning_plan"
    )

    if learning_plan:
        st.divider()
        show_learning_plan(learning_plan)
