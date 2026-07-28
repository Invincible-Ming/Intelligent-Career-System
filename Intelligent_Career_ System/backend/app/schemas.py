"""
API 请求响应和 Agent 结构化输出模型。
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Schema(BaseModel):
    """Pydantic 公共配置。"""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )


# ----------------------------------------------------------------------
# 文档与检索
# ----------------------------------------------------------------------


class DocumentResponse(Schema):
    id: uuid.UUID
    filename: str
    document_type: str
    status: str
    chunk_count: int
    error_message: str | None = None


class SearchRequest(Schema):
    query: str = Field(min_length=1, max_length=2000)
    document_type: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(Schema):
    document_id: str
    document_type: str
    content: str
    score: float
    source: str = "dense"
    rrf_score: float | None = None
    rerank_score: float | None = None


# ----------------------------------------------------------------------
# 简历与岗位分析
# ----------------------------------------------------------------------


class ResumeAnalysis(Schema):
    """Resume Agent 输出。"""

    skills: list[str] = Field(default_factory=list)
    experiences: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)


class JobAnalysis(Schema):
    """JD Agent 输出。"""

    job_title: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    experience_requirement: str | None = None
    education_requirement: str | None = None


# ----------------------------------------------------------------------
# 岗位匹配
# ----------------------------------------------------------------------


class MatchScores(Schema):
    """岗位匹配分项分数，范围为 0～100。"""

    skill: float = Field(ge=0, le=100)
    experience: float = Field(ge=0, le=100)
    responsibility: float = Field(ge=0, le=100)
    education: float = Field(ge=0, le=100)
    bonus: float = Field(ge=0, le=100)


class MatchEvaluation(Schema):
    """
    Match Agent 输出。

    Agent 只生成分项分数和分析内容，
    total_score 由 Python 根据固定权重计算。
    """

    scores: MatchScores
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class MatchReport(Schema):
    """最终岗位匹配报告。"""

    total_score: float = Field(ge=0, le=100)
    match_level: str
    scores: MatchScores

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    resume_analysis: ResumeAnalysis
    job_analysis: JobAnalysis


class MatchRequest(Schema):
    """岗位匹配请求。"""

    resume_document_id: uuid.UUID
    jd_document_id: uuid.UUID | None = None
    jd_text: str | None = Field(
        default=None,
        max_length=30000,
    )

    @model_validator(mode="after")
    def validate_jd_source(self) -> "MatchRequest":
        if self.jd_document_id is None and not self.jd_text:
            raise ValueError(
                "jd_document_id 和 jd_text 至少提供一项"
            )

        return self


# ----------------------------------------------------------------------
# 面试计划
# ----------------------------------------------------------------------


class InterviewQuestion(Schema):
    question: str
    question_type: str
    purpose: str
    answer_points: list[str] = Field(default_factory=list)


class InterviewPlan(Schema):
    job_title: str
    focus_areas: list[str] = Field(default_factory=list)
    questions: list[InterviewQuestion] = Field(
        default_factory=list
    )


class InterviewRequest(Schema):
    """根据岗位匹配任务生成面试计划。"""

    match_run_id: uuid.UUID
    difficulty: Literal[
        "junior",
        "intermediate",
        "senior",
    ] = "intermediate"
    question_count: int = Field(default=8, ge=3, le=15)


# ----------------------------------------------------------------------
# 学习计划
# ----------------------------------------------------------------------


class LearningItem(Schema):
    skill: str
    priority: Literal["high", "medium", "low"]
    reason: str
    estimated_days: int = Field(ge=1, le=365)
    tasks: list[str] = Field(default_factory=list)


class LearningPlan(Schema):
    target_role: str
    summary: str
    items: list[LearningItem] = Field(default_factory=list)
    weekly_plan: list[str] = Field(default_factory=list)


class LearningRequest(Schema):
    """根据岗位匹配任务生成学习计划。"""

    match_run_id: uuid.UUID
    available_weeks: int = Field(default=4, ge=1, le=24)
    hours_per_week: int = Field(default=10, ge=1, le=80)


# ----------------------------------------------------------------------
# Agent 任务
# ----------------------------------------------------------------------


class RunResponse(Schema):
    run_id: uuid.UUID
    task_type: str
    status: str
    result: dict | None = None
    error_message: str | None = None
