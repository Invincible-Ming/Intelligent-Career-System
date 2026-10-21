"""视图展示组件。

报告与文档展示组件；文档预览仅在用户打开时读取后端保存的解析文本。
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from career_ui.config import DOCUMENT_TYPE_NAMES
from career_ui.documents import (
    document_identity_label,
    document_name_marker,
    document_upload_time,
)


def document_label(document: dict[str, Any]) -> str:
    document_type = document.get("document_type", "unknown")
    type_name = DOCUMENT_TYPE_NAMES.get(document_type, document_type)
    marker = document_name_marker(document)
    return (
        f"{document_identity_label(document)} · {type_name}"
        f" · {document.get('status', 'unknown')}"
        + (f" · {marker}" if marker else "")
    )


@st.dialog("文档预览", width="large")
def show_document_preview(document: dict[str, Any]) -> None:
    from career_ui.endpoints import APIError, get_document_preview

    try:
        preview = get_document_preview(str(document["id"]))
    except APIError as exc:
        st.warning(str(exc))
        return
    filename = escape(str(preview.get("filename") or document.get("filename") or "未命名文档"))
    upload_time = escape(document_upload_time(document))
    st.html(
        '<header class="document-preview-header">'
        f'<h3>{filename}</h3><p>{upload_time}</p>'
        '</header>'
    )
    if preview.get("truncated"):
        st.info("仅显示前部分内容")
    # Render untrusted document text literally; it must never become active HTML.
    content = escape(str(preview.get("content") or "暂无可预览内容"))
    st.html(
        '<article class="document-preview-body" tabindex="0" aria-label="文档正文">'
        f'{content}</article>'
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
    if report.get("assessment_date"):
        st.caption(f"评估日期：{report['assessment_date']}")

    score_column, level_column = st.columns(2)
    with score_column:
        st.metric("综合加权匹配分", f"{float(report.get('total_score', 0)):.1f} 分")
    with level_column:
        st.metric("匹配等级", report.get("match_level", "未知"))

    scores = report.get("scores", {})
    st.markdown("#### 维度分项明细")

    score_items = [
        ("技能匹配", "skill"),
        ("经历匹配", "experience"),
        ("职责匹配", "responsibility"),
        ("教育背景", "education"),
        ("加分项", "bonus"),
    ]
    for column, (label, key) in zip(st.columns(5), score_items, strict=True):
        column.metric(label, f"{float(scores.get(key, 0)):.0f} 分")

    left_column, right_column = st.columns(2)
    with left_column:
        show_string_list("已匹配技能", report.get("matched_skills"))
        show_string_list("候选人核心优势", report.get("strengths"))

    with right_column:
        show_string_list("缺失/待补足技能", report.get("missing_skills"))
        risks = report.get("risks") or []
        business_risks = [
            item for item in risks
            if not str(item).startswith(("需人工复核：", "自动校验提示（人工已确认）："))
        ]
        show_string_list("潜在风险点", business_risks)
        if len(business_risks) != len(risks):
            st.caption("此历史报告含旧版校验记录，已与求职风险分开；可重新评估获取结构化校验结果。")

    show_string_list("专家改进建议", report.get("suggestions"))

    verification = report.get("verification")
    if isinstance(verification, dict):
        st.markdown("#### 报告校验")
        status = verification.get("status")
        summary = verification.get("summary") or "请核对当前报告。"
        if status == "correction_unverified":
            summary = "已生成修正版，但自动校验请求已用完，最新修正版尚未完成复核。"
        if verification.get("nonblocking"):
            st.info("报告已生成，自动校验未完成，结果暂未经完整复核。")
        elif status == "passed":
            st.success(summary)
        elif status == "manual_accepted":
            st.info(summary)
        else:
            st.warning(summary)
        if verification.get("nonblocking") and verification.get("correction_applied"):
            st.caption("已处理下列修正事项。")
        elif verification.get("correction_applied") and status not in ("passed", "manual_accepted"):
            st.caption("已生成修正版，最新版本尚未通过校验；下列问题来自修正前的报告。")
        issues = verification.get("issues") or []
        corrections = verification.get("corrections") or []
        if issues or corrections:
            label = "查看修正前的问题与修正记录" if verification.get("correction_applied") else "查看具体问题与修正记录"
            with st.expander(label):
                for issue in issues[:3]:
                    if not isinstance(issue, dict):
                        continue
                    st.markdown(f"**{issue.get('field', '报告内容')}：{issue.get('problem', '')}**")
                    st.write("报告原句：", issue.get("claim", ""))
                    source = "简历" if issue.get("source") == "resume" else "岗位要求"
                    st.write(f"{source}依据：", issue.get("evidence") or "原文未找到支持该断言的依据")
                    st.write("修正建议：", issue.get("suggested_fix", ""))
                show_string_list("本次修正记录", corrections[:6])

    with st.expander("查看简历结构化抽取明细 (Resume Agent)"):
        st.json(report.get("resume_analysis", {}))

    with st.expander("查看岗位要求拆解明细 (JD Agent)"):
        st.json(report.get("job_analysis", {}))


def match_progress_updates(status_box, *, resuming: bool = False):
    """Advance only when a real stage finishes; spinner continues between events."""

    stages = (
        {"match_agent", "verify_agent"}
        if resuming
        else {"resume_agent", "jd_agent", "match_agent", "verify_agent"}
    )
    completed: set[str] = set()
    stage = "正在恢复评估…" if resuming else "正在分析简历和岗位要求…"
    bar = st.progress(0.0, text=f"阶段进度 0 / {len(stages)} · {stage}")
    waiting = st.empty()

    def update(event_type, data):
        nonlocal stage
        if event_type == "progress":
            stage = data.get("message", stage)
            status_box.update(label=stage, state="running")
        elif event_type == "heartbeat":
            waiting.caption(
                f"已等待 {data.get('elapsed_seconds', 0)} 秒 · {stage}"
            )
        elif event_type == "node_update":
            node = data.get("node")
            if node in stages and data.get("status") in (
                "completed",
                "passed",
                "warning",
            ):
                completed.add(node)
            if node == "verify_agent" and data.get("status") == "retry":
                completed.discard("match_agent")
                stage = "正在根据校验意见修正报告…"
                status_box.update(label=stage, state="running")
        elif event_type in ("complete", "interrupt"):
            completed.update(stages)
            waiting.empty()
            stage = "等待你审核报告" if event_type == "interrupt" else "报告已完成"
        bar.progress(
            len(completed) / len(stages),
            text=f"阶段进度 {len(completed)} / {len(stages)} · {stage}",
        )

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
            st.markdown(
                f"**问题类型：** {question.get('question_type', '未分类')}"
            )
            st.markdown(f"**考察目的：** {question.get('purpose', '')}")
            show_string_list("回答要点", question.get("answer_points"))


def show_learning_plan(plan: dict[str, Any]) -> None:
    st.subheader(f"能力提升路径：{plan.get('target_role', '目标岗位')}")
    summary = plan.get("summary", "")
    if summary:
        st.info(summary)

    for index, item in enumerate(plan.get("items", []), start=1):
        priority = item.get("priority", "medium")
        priority_name = {"high": "高", "medium": "中", "low": "低"}.get(
            priority, priority
        )
        skill = item.get("skill", f"学习任务 {index}")

        with st.expander(
            f"{index}. {skill} ｜ 优先级：{priority_name}", expanded=index == 1
        ):
            st.markdown(f"**提升原因：** {item.get('reason', '')}")
            st.markdown(f"**预计周期：** {item.get('estimated_days', 0)} 天")
            show_string_list("具体行动任务", item.get("tasks"))

    show_string_list("每周执行安排", plan.get("weekly_plan"))


def get_search_score_display(
    result: dict[str, Any], search_mode: str
) -> tuple[str, str, float]:
    """把不同检索模式下的分数映射到统一的展示元组。"""

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


def show_search_results(
    results: list[dict[str, Any]], search_mode: str
) -> None:
    if not results:
        st.info("没有检索到相关内容")
        return

    for index, result in enumerate(results, start=1):
        score_name, source_name, main_score = get_search_score_display(
            result, search_mode
        )
        with st.expander(
            f"结果 {index}｜{score_name} {main_score:.8f}", expanded=index == 1
        ):
            st.caption(f"召回来源：{source_name}")
            st.write(result.get("content", ""))

            rrf_score = result.get("rrf_score")
            rerank_score = result.get("rerank_score")
            if rrf_score is not None:
                st.caption(f"RRF 分数：{float(rrf_score):.8f}")
            if rerank_score is not None:
                st.caption(f"BGE 重排分：{float(rerank_score):.8f}")

            document_type = result.get("document_type", "")
            st.caption(
                f"文档类型：{DOCUMENT_TYPE_NAMES.get(document_type, document_type)}"
            )
