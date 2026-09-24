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

import json
import os
import time
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

import requests
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


class APIError(Exception):
    """后端 API 请求异常。"""


def browser_auth_urls() -> tuple[str, str]:
    """Browser and Streamlit must use the same hostname to share a host-only cookie."""
    frontend = urlsplit(st.context.url or "http://localhost:8501/")
    backend = urlsplit(os.getenv("BROWSER_API_BASE_URL", API_BASE_URL))
    if backend.hostname in ("localhost", "127.0.0.1") and frontend.hostname in ("localhost", "127.0.0.1"):
        netloc = frontend.hostname + (f":{backend.port}" if backend.port else "")
        backend = backend._replace(netloc=netloc)
    return urlunsplit(backend).rstrip("/"), f"{frontend.scheme}://{frontend.netloc}/"


def reset_login_state():
    # Do not restore the stale initial-request cookie on the following rerun.
    st.session_state.clear()
    st.session_state.clear_browser_session = True
    st.rerun()


@st.cache_resource
def browser_auth_component():
    assets = Path(__file__).parent / "auth_browser"
    return st.components.v2.component(
        "browser_auth",
        html=(assets / "form.html").read_text(encoding="utf-8"),
        css=(assets / "style.css").read_text(encoding="utf-8"),
        js=(assets / "auth.js").read_text(encoding="utf-8"),
    )


def clear_browser_cookie():
    base, return_to = browser_auth_urls()
    browser_auth_component()(data={"mode": "logout", "url": base + "/api/auth/browser-logout",
                                  "return_to": return_to}, key="cookie_cleanup")
    st.stop()


def auth_headers() -> dict[str, str]:
    token = st.session_state.get('access_token')
    return {'Authorization': f'Bearer {token}'} if token else {}


def operation_read_timeout(path):
    limits = globals().get('security_options', {})
    if '/chat/completions' in path:
        return limits.get('chat_total_timeout', 90) + 10
    if '/documents/upload' in path:
        return limits.get('upload_total_timeout', 120) + 10
    if '/match' in path:
        return limits.get('match_total_timeout', 600) + 10
    if any(part in path for part in ('/interview', '/learning-plan', '/evaluation')):
        return limits.get('analysis_total_timeout', 180) + 10
    return limits.get('api_total_timeout', 30) + 10


def request_api(
        method: str,
        path: str,
        *,
        timeout: int = SHORT_TIMEOUT,
        **kwargs: Any,
) -> Any:
    """调用 FastAPI，并统一处理错误。"""

    kwargs["headers"] = {**kwargs.pop("headers", {}), **auth_headers()}
    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=(5, operation_read_timeout(path) if timeout == LONG_TIMEOUT else timeout),
            **kwargs,
        )
    except requests.ConnectionError as exc:
        raise APIError(f"无法连接后端服务：{API_BASE_URL}") from exc
    except requests.Timeout as exc:
        raise APIError("请求超时，请检查后端日志后重试") from exc
    except requests.RequestException as exc:
        raise APIError(f"请求失败：{exc}") from exc

    if not response.ok:
        if response.status_code == 401 and path != "/api/auth/login":
            reset_login_state()
        try:
            error_data = response.json()
            detail = error_data.get("detail", error_data)
        except ValueError:
            detail = response.text or response.reason

        raise APIError(f"HTTP {response.status_code}：{detail}")

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError as exc:
        raise APIError("后端返回了无法解析的数据") from exc


def stream_chat_api(
        prompt: str,
        conversation_id: str | None = None,
) -> Iterator[str]:
    """调用智能对话 SSE 流式接口，逐 Token 产出文字供打字机渲染。"""
    url = f"{API_BASE_URL}/api/chat/completions"
    payload: dict[str, Any] = {
        "message": prompt,
        "stream": True,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            headers=auth_headers(),
            timeout=(5, operation_read_timeout(url)),
        )
    except requests.ConnectionError as exc:
        raise APIError(f"无法连接后端服务：{API_BASE_URL}") from exc
    except requests.Timeout as exc:
        raise APIError("流式对话连接超时") from exc
    except requests.RequestException as exc:
        raise APIError(f"对话请求失败：{exc}") from exc

    if response.status_code == 401:
        response.close()
        reset_login_state()
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text or response.reason
        response.close()
        raise APIError(f"HTTP {response.status_code}：{detail}")

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_str)
                    if "error" in data_json:
                        raise APIError(data_json["error"].get("message", "回复生成已中断"))

                    # 记录返回的 conversation_id
                    if "conversation_id" in data_json and not st.session_state.conversation_id:
                        st.session_state.conversation_id = data_json["conversation_id"]

                    choices = data_json.get("choices", [])
                    if choices and isinstance(choices, list):
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    elif "content" in data_json:
                        yield data_json["content"]
                except json.JSONDecodeError:
                    if data_str:
                        yield data_str
    except requests.RequestException as exc:
        raise APIError("流式连接已中断，请稍后重试") from exc
    finally:
        response.close()


def stream_match_api(
        payload: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """调用岗位匹配 SSE 流式接口，实时产出 (event_type, payload_dict)。"""
    url = f"{API_BASE_URL}/api/match/stream"

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            headers=auth_headers(),
            timeout=(5, operation_read_timeout(url)),
        )
    except requests.ConnectionError as exc:
        raise APIError(f"无法连接后端服务：{API_BASE_URL}") from exc
    except requests.Timeout as exc:
        raise APIError("岗位匹配流式连接超时") from exc
    except requests.RequestException as exc:
        raise APIError(f"请求失败：{exc}") from exc

    if response.status_code == 401:
        response.close()
        reset_login_state()
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text or response.reason
        response.close()
        raise APIError(f"HTTP {response.status_code}：{detail}")

    current_event = "message"
    try:
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                try:
                    data_json = json.loads(data_str)
                    yield current_event, data_json
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as exc:
        raise APIError("流式连接已中断，请稍后重试") from exc
    finally:
        response.close()


def stream_match_resume_api(
        thread_id: str,
        human_feedback: str | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """调用人工干预与唤醒 SSE 流式接口。"""
    url = f"{API_BASE_URL}/api/match/stream/resume"
    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "human_feedback": human_feedback or None,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            headers=auth_headers(),
            timeout=(5, operation_read_timeout(url)),
        )
    except requests.ConnectionError as exc:
        raise APIError(f"无法连接后端服务：{API_BASE_URL}") from exc
    except requests.Timeout as exc:
        raise APIError("唤醒工作流流式连接超时") from exc
    except requests.RequestException as exc:
        raise APIError(f"请求失败：{exc}") from exc

    if response.status_code == 401:
        response.close()
        reset_login_state()
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text or response.reason
        response.close()
        raise APIError(f"HTTP {response.status_code}：{detail}")

    current_event = "message"
    try:
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                try:
                    data_json = json.loads(data_str)
                    yield current_event, data_json
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as exc:
        raise APIError("流式连接已中断，请稍后重试") from exc
    finally:
        response.close()


def get_documents() -> list[dict[str, Any]]:
    return request_api("GET", "/api/documents")


def get_runs(
        *,
        task_type: str | None = "match",
        status: str | None = None,
        limit: int = 20,
) -> list[dict[str, Any]]:
    """查询历史或待恢复任务列表。"""
    params: dict[str, Any] = {"limit": limit}
    if task_type:
        params["task_type"] = task_type
    if status:
        params["status"] = status
    return request_api("GET", "/api/runs", params=params)


# 获取历史会话列表 API
def get_conversations() -> list[dict[str, Any]]:
    """获取用户的历史对话列表"""
    try:
        return request_api("GET", "/api/chat/conversations")
    except APIError:
        return []


# 获取会话历史消息 API
def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    """获取指定对话的历史消息记录"""
    try:
        return request_api("GET", f"/api/chat/conversations/{conversation_id}/messages")
    except APIError:
        return []


# 新增：删除对话 API
def delete_chat_conversation(conversation_id: str) -> None:
    """删除指定的对话记录"""
    request_api("DELETE", f"/api/chat/conversations/{conversation_id}")


# 新增：重命名 / 置顶对话 API
def update_chat_conversation(
        conversation_id: str,
        *,
        title: str | None = None,
        pinned: bool | None = None,
) -> dict[str, Any]:
    """重命名或置顶指定的对话记录"""
    params: dict[str, Any] = {}
    if title is not None:
        params["title"] = title
    if pinned is not None:
        params["pinned"] = "true" if pinned else "false"
    return request_api("PATCH", f"/api/chat/conversations/{conversation_id}", params=params)


# 岗位匹配任务的状态与报告读取工具（匹配 / 面试 / 学习三个页面共用）
MATCH_STATUS_NAMES = {
    "paused": "待人工审核",
    "completed": "已完成",
    "running": "进行中",
    "failed": "失败",
}


def fetch_match_context(*, limit: int = 20) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """拉取岗位匹配任务历史与文档清单。"""
    try:
        runs = get_runs(task_type="match", limit=limit)
    except Exception:
        runs = []
    try:
        documents = get_documents()
    except APIError:
        documents = []
    return runs, documents


def match_report_of(run: dict[str, Any]) -> dict[str, Any]:
    """兼容三种落库形态：报告本体（SSE 完成）/ {"current_report":...}（同步接口）/ {"report":...}。"""
    data = run.get("result") or {}
    if not isinstance(data, dict):
        return {}
    if "total_score" in data or "resume_analysis" in data:
        return data
    return data.get("current_report") or data.get("report") or {}


def match_run_label(run: dict[str, Any], documents: list[dict[str, Any]]) -> str:
    """用时间 + 简历 × JD + 状态 + 分数生成可读标签，替代裸任务 ID。"""
    doc_name = {d["id"]: d.get("filename", "未命名文档") for d in documents}
    inp = run.get("input_data") or {}
    created = (str(run.get("created_at") or ""))[:16].replace("T", " ")
    resume_name = doc_name.get(inp.get("resume_document_id"), "简历")
    if inp.get("jd_input_type") == "text":
        jd_name = "手动输入 JD"
    else:
        jd_name = doc_name.get(inp.get("jd_document_id"), "JD 文档")
    parts = [created, f"{resume_name} × {jd_name}", MATCH_STATUS_NAMES.get(run.get("status"), run.get("status", ""))]
    score = match_report_of(run).get("total_score")
    if score is not None:
        parts.append(f"{score}分")
    return " ｜ ".join(parts)


# 评测模块 API（仅管理员）
def get_evaluation_datasets() -> list[dict[str, Any]]:
    """列出可用的评测数据集"""
    return request_api("GET", "/api/evaluation/datasets")


def start_evaluation_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """启动评测实验（后台异步执行）"""
    return request_api("POST", "/api/evaluation/experiments", json={"experiment_config": config})


def list_evaluation_experiments() -> list[dict[str, Any]]:
    """列出当前会话中的评测实验状态"""
    return request_api("GET", "/api/evaluation/experiments")


def get_evaluation_report_html(path: str) -> str:
    """读取评测实验的 HTML 报告"""
    return request_api("GET", "/api/evaluation/report", params={"path": path}).get("html", "")


def upload_document(uploaded_file: Any, document_type: str) -> dict[str, Any]:
    if uploaded_file.size > security_options["max_upload_size_mb"] * 1024 * 1024:
        raise APIError("上传文件超过大小限制")
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        ),
    }
    return request_api(
        "POST",
        "/api/documents/upload",
        files=files,
        data={"document_type": document_type},
        timeout=LONG_TIMEOUT,
    )


def delete_document(document_id: str) -> None:
    request_api("DELETE", f"/api/documents/{document_id}", timeout=LONG_TIMEOUT)


def search_documents(
        *, query: str, document_type: str | None, top_k: int, search_mode: str
) -> list[dict[str, Any]]:
    endpoints = {
        "dense": "/api/search",
        "bm25": "/api/search/bm25",
        "hybrid": "/api/search/hybrid",
    }
    endpoint = endpoints.get(search_mode, "/api/search/hybrid")
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


def create_interview_plan(
        *, match_run_id: str, difficulty: str, question_count: int
) -> dict[str, Any]:
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
        *, match_run_id: str, available_weeks: int, hours_per_week: int
) -> dict[str, Any]:
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
# 视图展示组件
# ----------------------------------------------------------------------


def document_label(document: dict[str, Any]) -> str:
    document_type = document.get("document_type", "unknown")
    type_name = DOCUMENT_TYPE_NAMES.get(document_type, document_type)
    return (
        f"{document.get('filename', '未命名文档')} "
        f"· {type_name} "
        f"· {document.get('status', 'unknown')}"
    )


def show_string_list(title: str, values: list[str] | None) -> None:
    st.markdown(f"#### {title}")
    if not values:
        st.caption("暂无")
        return
    for value in values:
        st.markdown(f"- {value}")


def show_match_report(report: dict[str, Any]) -> None:
    st.subheader("岗位多维匹配报告")

    score_column, level_column = st.columns(2)
    with score_column:
        st.metric("综合加权匹配分", f"{float(report.get('total_score', 0)):.1f} 分")
    with level_column:
        st.metric("匹配等级", report.get("match_level", "未知"))

    scores = report.get("scores", {})
    st.markdown("#### 维度分项明细")

    score_columns = st.columns(5)
    score_items = [
        ("技能匹配", "skill"),
        ("经历匹配", "experience"),
        ("职责匹配", "responsibility"),
        ("教育背景", "education"),
        ("加分项", "bonus"),
    ]

    for column, (label, key) in zip(score_columns, score_items, strict=True):
        column.metric(label, f"{float(scores.get(key, 0)):.0f} 分")

    left_column, right_column = st.columns(2)
    with left_column:
        show_string_list("已匹配技能", report.get("matched_skills"))
        show_string_list("候选人核心优势", report.get("strengths"))

    with right_column:
        show_string_list("缺失/待补足技能", report.get("missing_skills"))
        show_string_list("潜在风险点", report.get("risks"))

    show_string_list("专家改进建议", report.get("suggestions"))

    with st.expander("查看简历结构化抽取明细 (Resume Agent)"):
        st.json(report.get("resume_analysis", {}))

    with st.expander("查看岗位要求拆解明细 (JD Agent)"):
        st.json(report.get("job_analysis", {}))


def match_progress_updates(status_box, *, resuming=False):
    """Advance only when a real stage finishes; spinner continues between events."""
    stages = {"match_agent", "verify_agent"} if resuming else {"resume_agent", "jd_agent", "match_agent"}
    completed = set()
    stage = "正在恢复评估…" if resuming else "正在分析简历和岗位要求…"
    bar = st.progress(0.0, text=f"阶段进度 0 / {len(stages)} · {stage}")
    waiting = st.empty()

    def update(event_type, data):
        nonlocal stage
        if event_type == "progress":
            stage = data.get("message", stage)
            status_box.update(label=stage, state="running")
        elif event_type == "heartbeat":
            waiting.caption(f"已等待 {data.get('elapsed_seconds', 0)} 秒 · {stage}")
        elif event_type == "node_update":
            node = data.get("node")
            if node in stages and data.get("status") in ("completed", "passed", "warning"):
                completed.add(node)
            if node == "verify_agent" and data.get("status") == "retry":
                completed.discard("match_agent")
                stage = "正在根据校验意见修正报告…"
                status_box.update(label=stage, state="running")
        elif event_type in ("complete", "interrupt"):
            completed.update(stages)
            waiting.empty()
            stage = "等待你审核报告" if event_type == "interrupt" else "报告已完成"
        bar.progress(len(completed) / len(stages), text=f"阶段进度 {len(completed)} / {len(stages)} · {stage}")

    return update


def show_interview_plan(plan: dict[str, Any]) -> None:
    st.subheader(f"面试预测题库：{plan.get('job_title', '目标岗位')}")
    show_string_list("重点准备方向", plan.get("focus_areas"))

    questions = plan.get("questions", [])
    if not questions:
        st.info("暂未生成面试问题")
        return

    for index, question in enumerate(questions, start=1):
        question_text = question.get("question", f"面试问题 {index}")
        with st.expander(f"{index}. {question_text}", expanded=index == 1):
            st.markdown(f"**问题类型：** {question.get('question_type', '未分类')}")
            st.markdown(f"**考察目的：** {question.get('purpose', '')}")
            show_string_list("回答要点", question.get("answer_points"))


def show_learning_plan(plan: dict[str, Any]) -> None:
    st.subheader(f"能力提升路径：{plan.get('target_role', '目标岗位')}")
    summary = plan.get("summary", "")
    if summary:
        st.info(summary)

    items = plan.get("items", [])
    for index, item in enumerate(items, start=1):
        priority = item.get("priority", "medium")
        priority_name = {"high": "高", "medium": "中", "low": "低"}.get(priority, priority)
        skill = item.get("skill", f"学习任务 {index}")

        with st.expander(f"{index}. {skill} ｜ 优先级：{priority_name}", expanded=index == 1):
            st.markdown(f"**提升原因：** {item.get('reason', '')}")
            st.markdown(f"**预计周期：** {item.get('estimated_days', 0)} 天")
            show_string_list("具体行动任务", item.get("tasks"))

    show_string_list("每周执行安排", plan.get("weekly_plan"))


def get_search_score_display(
        result: dict[str, Any], search_mode: str
) -> tuple[str, str, float]:
    source = result.get("source", search_mode)
    if source == "hybrid_reranked":
        return (
            "BGE 重排分",
            "Milvus + BM25 + RRF + BGE",
            float(result.get("rerank_score", result.get("score", 0.0))),
        )
    if source == "hybrid":
        return (
            "RRF 综合分",
            "Milvus + BM25 + RRF",
            float(result.get("rrf_score", result.get("score", 0.0))),
        )
    if source == "bm25":
        return ("BM25 分数", "BM25", float(result.get("score", 0.0)))

    return ("余弦相似度", "Milvus", float(result.get("score", 0.0)))


def show_search_results(results: list[dict[str, Any]], search_mode: str) -> None:
    if not results:
        st.info("没有检索到相关内容")
        return

    for index, result in enumerate(results, start=1):
        score_name, source_name, main_score = get_search_score_display(
            result, search_mode
        )
        with st.expander(f"结果 {index}｜{score_name} {main_score:.8f}", expanded=index == 1):
            st.caption(f"召回来源：{source_name}")
            st.write(result.get("content", ""))

            rrf_score = result.get("rrf_score")
            rerank_score = result.get("rerank_score")
            if rrf_score is not None:
                st.caption(f"RRF 分数：{float(rrf_score):.8f}")
            if rerank_score is not None:
                st.caption(f"BGE 重排分：{float(rerank_score):.8f}")

            document_type = result.get("document_type", "")
            st.caption(f"文档类型：{DOCUMENT_TYPE_NAMES.get(document_type, document_type)}")
            st.caption(f"文档 ID：{result.get('document_id', '')}")


# ----------------------------------------------------------------------
# 顶部页面标签
# ----------------------------------------------------------------------

# Authentication must run before loading or rendering private tabs.
try:
    security_options = request_api('GET', '/api/auth/options')
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
    cookie_token = st.context.cookies.get(security_options.get('auth_cookie_name', 'career_session'))
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
                if security_options['registration_enabled']:
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
                                request_api('POST', '/api/auth/register', json={'username':new_username, 'password':new_password})
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
                                 max_chars=security_options["chat_max_input_chars"])

with private_area.container():
    # ----------------------------------------------------------------------
    # 1. 智能对话页面
    # ----------------------------------------------------------------------

    if active_page == "chat":
        # 欢迎语固定放在对话页顶部；快捷主题已移除，减少首屏占用。
        if not st.session_state.chat_messages:
            username = escape(st.session_state.auth_user['username'])
            st.markdown(f'<div class="chat-welcome"><h1>你好，{username}</h1><p>今天想为求职做些什么？</p></div>', unsafe_allow_html=True)

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
                    st.toast(
                        f"上传成功：{uploaded_file.name}，生成 {upload_result.get('chunk_count', 0)} 个文本块",
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
                    max_chars=security_options["search_max_input_chars"],
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
            selected_resume = st.selectbox(
                "选择候选人简历",
                options=resume_documents,
                format_func=document_label,
            )

            selected_job = None
            jd_text = ""

            jd_source = st.radio(
                "岗位描述 (JD) 来源",
                options=["直接输入", "已上传文档"],
                horizontal=True,
            )

            if jd_source == "直接输入":
                jd_text = st.text_area(
                    "输入岗位描述 (JD)",
                    placeholder="粘贴岗位名称、职责要求与技术栈门槛...",
                    height=200,
                    max_chars=security_options["jd_max_input_chars"],
                )
            else:
                if not job_documents:
                    st.warning("尚未上传岗位描述文档，请切换为“直接输入”")
                else:
                    selected_job = st.selectbox(
                        "选择岗位描述文档",
                        options=job_documents,
                        format_func=document_label,
                    )

            can_match = bool(
                selected_resume
                and (
                        (jd_source == "直接输入" and jd_text.strip())
                        or (jd_source == "已上传文档" and selected_job)
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

                # 清空历史状态
                st.session_state.pop("match_report", None)
                st.session_state.pop("match_run_id", None)
                st.session_state.pop("match_paused", None)
                st.session_state.pop("match_thread_id", None)
                st.session_state.pop("preliminary_report", None)

                # 动态流式状态框展示
                with st.status(
                        "正在分析简历和岗位要求…", expanded=True
                ) as status_box:

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

                            # 捕获人机协同挂起事件
                            elif event_type == "interrupt":
                                st.session_state["match_paused"] = True
                                st.session_state["match_thread_id"] = data.get("thread_id")
                                st.session_state["preliminary_report"] = data.get("current_report")
                                status_box.update(
                                    label="初步匹配打分完成，等待人工审核确认...",
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

        # ==================================================================
        # 人机协同 (Human-in-the-loop) 交互卡片
        # ==================================================================
        if st.session_state.get("match_paused"):
            st.divider()
            st.warning("**【人机协同】初步匹配已完成，进入人工审核确认节点**")

            pre_report = st.session_state.get("preliminary_report") or {}
            if pre_report:
                score_val = pre_report.get("total_score")
                level_val = pre_report.get("match_level", "待评定")
                if score_val is not None:
                    st.info(f"**初始评估总分：{score_val} 分** ｜ 匹配等级：{level_val}")
                with st.expander("查看待审核的匹配报告", expanded=True):
                    show_match_report(pre_report)

            feedback_col, action_col, cancel_col = st.columns([3, 1, 1])
            with feedback_col:
                human_feedback_text = st.text_input(
                    "人工修正意见（可选）",
                    placeholder="例如：候选人具有 5 年架构经验但简历表述含糊，建议按 85 分重新核算经历匹配分...",
                    max_chars=security_options["feedback_max_input_chars"],
                )

            with action_col:
                st.write("")  # 垂直对齐占位
                st.write("")
                confirm_resume = st.button(
                    "确认并继续",
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

                with st.status("正在更新匹配报告并校验事实…", expanded=True) as resume_box:

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

                            elif event_type == "complete":
                                report = data.get("report")
                                if report:
                                    st.session_state["match_report"] = report
                                    st.session_state["match_paused"] = False
                                    resume_box.update(
                                        label="报告已完成，存在需复核的问题" if data.get("is_degraded") else "匹配报告校验完成！",
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
            top_k = top_col.slider("召回候选数 Top-K", 1, 20, 5, key=f"{prefix}-topk")
            enable_reranker = rerank_col.toggle("启用重排序", value=True, key=f"{prefix}-rerank")
            rerank_top_k = st.slider(
                "重排序返回数", 1, 10, 5, key=f"{prefix}-rtopk", disabled=not enable_reranker,
            )
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
