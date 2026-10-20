"""视图展示组件。

纯展示，不含状态变更。所有函数只依赖 streamlit 与传入数据，
因此可以单独测试，也便于在页面之间复用。
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from career_ui.config import DOCUMENT_TYPE_NAMES


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
        show_string_list("潜在风险点", report.get("risks"))

    show_string_list("专家改进建议", report.get("suggestions"))

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
            st.caption(f"文档 ID：{result.get('document_id', '')}")
