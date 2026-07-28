"""
智能求职 Multi-Agent 工作流。

主流程：

Resume Agent → JD Agent → Match Agent

扩展能力：

Match Report → Interview Agent
Match Report → Learning Agent
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.bailian import bailian_service
from app.schemas import (
    InterviewPlan,
    JobAnalysis,
    LearningPlan,
    MatchEvaluation,
    MatchReport,
    ResumeAnalysis,
)


class CareerState(TypedDict, total=False):
    """LangGraph 节点共享状态。"""

    resume_text: str
    jd_text: str

    resume_analysis: dict[str, Any]
    job_analysis: dict[str, Any]
    match_report: dict[str, Any]

    difficulty: str
    question_count: int

    available_weeks: int
    hours_per_week: int

    result: dict[str, Any]


# ----------------------------------------------------------------------
# Resume Agent
# ----------------------------------------------------------------------


async def resume_agent(
        state: CareerState,
) -> dict[str, Any]:
    """从简历中提取技能、经历和优势。"""

    resume_text = state["resume_text"][:30000]

    result = await bailian_service.structured_chat(
        system_prompt=(
            "你是一名专业的简历分析师。"
            "请从简历中客观提取候选人的技能、工作经历、"
            "项目经历、教育背景和优势。"
            "不要编造简历中没有出现的信息。"
        ),
        user_prompt=f"请分析下面的简历：\n\n{resume_text}",
        response_model=ResumeAnalysis,
    )

    return {
        "resume_analysis": result.model_dump(
            mode="json"
        )
    }


# ----------------------------------------------------------------------
# JD Agent
# ----------------------------------------------------------------------


async def jd_agent(
        state: CareerState,
) -> dict[str, Any]:
    """从 JD 中提取岗位要求。"""

    jd_text = state["jd_text"][:30000]

    result = await bailian_service.structured_chat(
        system_prompt=(
            "你是一名岗位分析师。"
            "请提取岗位名称、必备技能、加分技能、岗位职责、"
            "工作经验要求和学历要求。"
            "只根据岗位描述进行分析。"
        ),
        user_prompt=f"请分析下面的岗位描述：\n\n{jd_text}",
        response_model=JobAnalysis,
    )

    return {
        "job_analysis": result.model_dump(
            mode="json"
        )
    }


# ----------------------------------------------------------------------
# Match Agent
# ----------------------------------------------------------------------


async def match_agent(
        state: CareerState,
) -> dict[str, Any]:
    """比较简历和 JD，生成岗位匹配报告。"""

    resume = ResumeAnalysis.model_validate(
        state["resume_analysis"]
    )
    job = JobAnalysis.model_validate(
        state["job_analysis"]
    )

    result = await bailian_service.structured_chat(
        system_prompt=(
            "你是一名严格的岗位匹配分析师。"
            "请比较候选人简历和岗位要求，给出五项匹配分数。"
            "所有分数范围为 0 到 100。"
            "简历中没有证据的技能不得判定为已掌握。"
            "请同时给出已匹配技能、缺失技能、优势、风险和建议。"
            "不要计算总分，总分将由程序计算。"
        ),
        user_prompt=(
            "【简历分析】\n"
            f"{to_json(resume.model_dump())}\n\n"
            "【岗位分析】\n"
            f"{to_json(job.model_dump())}"
        ),
        response_model=MatchEvaluation,
    )

    total_score = calculate_total_score(result)
    report = MatchReport(
        total_score=total_score,
        match_level=get_match_level(total_score),
        scores=result.scores,
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
        strengths=result.strengths,
        risks=result.risks,
        suggestions=result.suggestions,
        resume_analysis=resume,
        job_analysis=job,
    )

    report_data = report.model_dump(mode="json")

    return {
        "match_report": report_data,
        "result": report_data,
    }


# ----------------------------------------------------------------------
# Interview Agent
# ----------------------------------------------------------------------


async def interview_agent(
        state: CareerState,
) -> dict[str, Any]:
    """根据匹配报告生成面试问题。"""

    report = MatchReport.model_validate(
        state["match_report"]
    )
    difficulty = state.get(
        "difficulty",
        "intermediate",
    )
    question_count = state.get(
        "question_count",
        8,
    )

    result = await bailian_service.structured_chat(
        system_prompt=(
            "你是一名技术面试教练。"
            "请根据岗位要求、候选人经历和能力缺口生成面试问题。"
            "问题应覆盖简历深挖、技术能力和项目经验，"
            "每道题需要给出考察目的和答案要点。"
        ),
        user_prompt=(
            f"面试难度：{difficulty}\n"
            f"问题数量：{question_count}\n\n"
            "【岗位匹配报告】\n"
            f"{to_json(report.model_dump())}"
        ),
        response_model=InterviewPlan,
    )

    result.questions = result.questions[:question_count]

    return {
        "result": result.model_dump(mode="json")
    }


# ----------------------------------------------------------------------
# Learning Agent
# ----------------------------------------------------------------------


async def learning_agent(
        state: CareerState,
) -> dict[str, Any]:
    """根据能力缺口生成学习计划。"""

    report = MatchReport.model_validate(
        state["match_report"]
    )
    available_weeks = state.get(
        "available_weeks",
        4,
    )
    hours_per_week = state.get(
        "hours_per_week",
        10,
    )

    result = await bailian_service.structured_chat(
        system_prompt=(
            "你是一名 AI 求职学习规划师。"
            "请根据岗位要求和候选人的能力缺口制定学习计划。"
            "优先解决对岗位匹配影响最大的技能缺口，"
            "任务必须具体并且能够执行。"
        ),
        user_prompt=(
            f"学习周期：{available_weeks} 周\n"
            f"每周时间：{hours_per_week} 小时\n\n"
            "【岗位匹配报告】\n"
            f"{to_json(report.model_dump())}"
        ),
        response_model=LearningPlan,
    )

    return {
        "result": result.model_dump(mode="json")
    }


# ----------------------------------------------------------------------
# 工作流构建
# ----------------------------------------------------------------------


def build_match_workflow():
    """构建简历与岗位匹配工作流。"""

    graph = StateGraph(CareerState)

    graph.add_node("resume_agent", resume_agent)
    graph.add_node("jd_agent", jd_agent)
    graph.add_node("match_agent", match_agent)

    graph.add_edge(START, "resume_agent")
    graph.add_edge("resume_agent", "jd_agent")
    graph.add_edge("jd_agent", "match_agent")
    graph.add_edge("match_agent", END)

    return graph.compile()


def build_single_agent_workflow(
        name: str,
        agent,
):
    """构建单 Agent 扩展工作流。"""

    graph = StateGraph(CareerState)
    graph.add_node(name, agent)
    graph.add_edge(START, name)
    graph.add_edge(name, END)

    return graph.compile()


match_workflow = build_match_workflow()

interview_workflow = build_single_agent_workflow(
    "interview_agent",
    interview_agent,
)

learning_workflow = build_single_agent_workflow(
    "learning_agent",
    learning_agent,
)


# ----------------------------------------------------------------------
# 对外调用函数
# ----------------------------------------------------------------------


async def run_match(
        *,
        resume_text: str,
        jd_text: str,
) -> MatchReport:
    """执行简历和岗位匹配。"""

    if not resume_text.strip():
        raise ValueError("简历文本不能为空")

    if not jd_text.strip():
        raise ValueError("岗位描述不能为空")

    state = await match_workflow.ainvoke(
        {
            "resume_text": resume_text,
            "jd_text": jd_text,
        }
    )

    return MatchReport.model_validate(
        state["result"]
    )


async def run_interview(
        *,
        match_report: dict[str, Any],
        difficulty: str = "intermediate",
        question_count: int = 8,
) -> InterviewPlan:
    """根据岗位匹配报告生成面试问题。"""

    state = await interview_workflow.ainvoke(
        {
            "match_report": match_report,
            "difficulty": difficulty,
            "question_count": question_count,
        }
    )

    return InterviewPlan.model_validate(
        state["result"]
    )


async def run_learning_plan(
        *,
        match_report: dict[str, Any],
        available_weeks: int = 4,
        hours_per_week: int = 10,
) -> LearningPlan:
    """根据岗位匹配报告生成学习计划。"""

    state = await learning_workflow.ainvoke(
        {
            "match_report": match_report,
            "available_weeks": available_weeks,
            "hours_per_week": hours_per_week,
        }
    )

    return LearningPlan.model_validate(
        state["result"]
    )


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------


def calculate_total_score(
        evaluation: MatchEvaluation,
) -> float:
    """使用固定权重计算岗位匹配总分。"""

    scores = evaluation.scores

    total = (
            scores.skill * 0.40
            + scores.experience * 0.25
            + scores.responsibility * 0.20
            + scores.education * 0.10
            + scores.bonus * 0.05
    )

    return round(
        min(100.0, max(0.0, total)),
        2,
    )


def get_match_level(score: float) -> str:
    """根据总分返回中文匹配等级。"""

    if score >= 85:
        return "高度匹配"

    if score >= 70:
        return "较为匹配"

    if score >= 55:
        return "一般匹配"

    if score >= 40:
        return "匹配度较低"

    return "暂不匹配"


def to_json(data: Any) -> str:
    """将 Python 数据转换为中文 JSON。"""

    return json.dumps(
        data,
        ensure_ascii=False,
    )
