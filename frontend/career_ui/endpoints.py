"""业务接口薄封装。

只做路径与参数拼装，不含业务判断；异常统一由 api.request_api 包装成 APIError。
"""
from __future__ import annotations

from typing import Any

from career_ui.api import (
    LONG_TIMEOUT,
    APIError,
    request_api,
    security_options,
)
from career_ui.documents import (
    document_identity_label,
    document_upload_time,
    prepare_document_list,
)

MATCH_STATUS_NAMES = {
    "paused": "待人工审核",
    "completed": "已完成",
    "running": "进行中",
    "failed": "失败",
}


# ----------------------------------------------------------------------
# 文档
# ----------------------------------------------------------------------


def get_documents() -> list[dict[str, Any]]:
    return prepare_document_list(request_api("GET", "/api/documents"))


def get_knowledge_status() -> dict[str, Any]:
    return request_api("GET", "/api/knowledge/status")


def get_knowledge_items(cursor: str = "") -> dict[str, Any]:
    return request_api("GET", "/api/knowledge/items", params={"cursor": cursor})


def get_document_preview(document_id: str) -> dict[str, Any]:
    return request_api("GET", f"/api/documents/{document_id}/text")


def upload_document(uploaded_file: Any, document_type: str) -> dict[str, Any]:
    """上传并向量化入库。

    返回 {"document": {...}, "deduplicated": bool}。
    deduplicated 为 True 表示内容哈希与既有文档相同、已复用，未重复入库。
    """

    limit_mb = security_options()["max_upload_size_mb"]
    if uploaded_file.size > limit_mb * 1024 * 1024:
        raise APIError(f"上传文件超过大小限制（{limit_mb} MB）")

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


def get_document_details(document_id: str) -> dict[str, Any]:
    return request_api("GET", f"/api/documents/{document_id}")


def download_document(document_id: str, *, token: str) -> bytes:
    return request_api("GET", f"/api/documents/{document_id}/download", raw=True, access_token=token)


def rename_document(document_id: str, filename: str) -> dict[str, Any]:
    return request_api("PATCH", f"/api/documents/{document_id}", json={"filename": filename})


def retry_document(document_id: str) -> dict[str, Any]:
    return request_api("POST", f"/api/documents/{document_id}/retry", timeout=LONG_TIMEOUT)


def search_documents(
    *, query: str, document_type: str | None, top_k: int, search_mode: str
) -> list[dict[str, Any]]:
    endpoints = {
        "ima": "/api/search/hybrid",
        "dense": "/api/search",
        "bm25": "/api/search/bm25",
        "hybrid": "/api/search/hybrid",
    }
    return request_api(
        "POST",
        endpoints.get(search_mode, "/api/search/hybrid"),
        json={
            "query": query,
            "document_type": document_type,
            "top_k": top_k,
        },
        timeout=LONG_TIMEOUT,
    )


# ----------------------------------------------------------------------
# 匹配任务
# ----------------------------------------------------------------------


def get_runs(
    *,
    task_type: str | None = "match",
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if task_type:
        params["task_type"] = task_type
    if status:
        params["status"] = status
    return request_api("GET", "/api/runs", params=params)


def delete_match_run(run_id: str) -> dict[str, Any]:
    return request_api("DELETE", f"/api/runs/{run_id}", timeout=LONG_TIMEOUT)


def fetch_match_context(
    *, limit: int = 20
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """拉取匹配任务历史与文档清单。

    两处失败都降级为空列表：页面其余部分仍应可用。
    """

    try:
        runs = get_runs(task_type="match", limit=limit)
    except APIError:
        runs = []
    try:
        documents = get_documents()
    except APIError:
        documents = []
    return runs, documents


def match_report_of(run: dict[str, Any]) -> dict[str, Any]:
    """兼容三种落库形态：报告本体/ {"current_report":...} / {"report":...}。"""

    data = run.get("result") or {}
    if not isinstance(data, dict):
        return {}
    if "total_score" in data or "resume_analysis" in data:
        return data
    return data.get("current_report") or data.get("report") or {}


def match_run_label(run: dict[str, Any], documents: list[dict[str, Any]]) -> str:
    """用时间 + 简历 × JD + 状态 + 分数生成可读标签，替代裸任务 ID。"""

    doc_name = {d["id"]: document_identity_label(d) for d in documents}
    inp = run.get("input_data") or {}
    created = document_upload_time(run)
    resume_name = doc_name.get(inp.get("resume_document_id"), "简历")
    jd_name = (
        "手动输入 JD"
        if inp.get("jd_input_type") == "text"
        else doc_name.get(inp.get("jd_document_id"), "JD 文档")
    )
    parts = [
        created,
        f"{resume_name} × {jd_name}",
        MATCH_STATUS_NAMES.get(run.get("status"), run.get("status", "")),
    ]
    score = match_report_of(run).get("total_score")
    if score is not None:
        parts.append(f"{score}分")
    return " ｜ ".join(parts)


# ----------------------------------------------------------------------
# 面试 / 学习计划
# ----------------------------------------------------------------------


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
# 会话
# ----------------------------------------------------------------------


def get_conversations() -> list[dict[str, Any]]:
    """获取用户的历史对话列表；失败降级为空。"""

    try:
        return request_api("GET", "/api/chat/conversations")
    except APIError:
        return []


def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    """获取指定对话的历史消息；失败降级为空。"""

    try:
        return request_api(
            "GET", f"/api/chat/conversations/{conversation_id}/messages"
        )
    except APIError:
        return []


def delete_chat_conversation(conversation_id: str) -> None:
    request_api("DELETE", f"/api/chat/conversations/{conversation_id}")


def update_chat_conversation(
    conversation_id: str,
    *,
    title: str | None = None,
    pinned: bool | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if title is not None:
        params["title"] = title
    if pinned is not None:
        params["pinned"] = "true" if pinned else "false"
    return request_api(
        "PATCH",
        f"/api/chat/conversations/{conversation_id}",
        params=params,
    )


# ----------------------------------------------------------------------
# 评测（仅管理员）
# ----------------------------------------------------------------------


def get_evaluation_datasets() -> list[dict[str, Any]]:
    return request_api("GET", "/api/evaluation/datasets")


def start_evaluation_experiment(config: dict[str, Any]) -> dict[str, Any]:
    return request_api(
        "POST", "/api/evaluation/experiments", json={"experiment_config": config}
    )


def list_evaluation_experiments() -> list[dict[str, Any]]:
    return request_api("GET", "/api/evaluation/experiments")


def get_evaluation_report_html(path: str) -> str:
    return request_api(
        "GET", "/api/evaluation/report", params={"path": path}
    ).get("html", "")
