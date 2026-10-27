"""
API 请求响应和 Agent 结构化输出模型。
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from app.core.config import settings


class Schema(BaseModel):
    """Pydantic 公共配置。"""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        extra="forbid",
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
    created_at: datetime | None = None


class DocumentDetailResponse(DocumentResponse):
    size_bytes: int | None = None
    content_type: str | None = None
    preview_available: bool = False


class RenameDocumentRequest(Schema):
    filename: str = Field(min_length=1, max_length=200)


class UploadDocumentResponse(Schema):
    """上传结果。

    deduplicated 为 True 表示内容哈希与既有文档相同，已直接复用，
    未重复解析或向量化。
    """

    document: DocumentResponse
    deduplicated: bool = False


class DocumentTextResponse(Schema):
    id: uuid.UUID
    filename: str
    content: str
    truncated: bool = False


class SearchRequest(Schema):
    query: str = Field(min_length=1, max_length=2000)
    document_type: Literal["resume", "job_description", "knowledge"] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(Schema):
    document_id: str
    document_type: str
    content: str
    score: float
    source: str = "dense"
    rrf_score: float | None = None
    rerank_score: float | None = None
    chunk_metadata: dict | None = None
    parent_context: str | None = None


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


class SkillRequirementGroup(Schema):
    """保留岗位技能要求中的 AND/OR 关系及原文依据。"""

    mode: Literal["all", "any"]
    skills: list[str] = Field(min_length=1, max_length=20)
    source_quote: str = Field(min_length=1, max_length=600)


class JobAnalysis(Schema):
    """JD Agent 输出。"""

    job_title: str
    required_skills: list[str] = Field(default_factory=list, description="逐项必须满足的技能，不展开任选技能组")
    skill_requirements: list[SkillRequirementGroup] = Field(
        default_factory=list,
        description="技能要求分组；any 表示至少一项，all 表示全部满足，保留原文 source_quote",
    )
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    experience_requirement: str | None = None
    education_requirement: str | None = None

    @model_validator(mode="after")
    def validate_requirement_groups(self, info: ValidationInfo) -> JobAnalysis:
        # Enforce only on fresh extraction; historical reports remain readable.
        if not (info.context or {}).get("strict_requirement_groups"):
            return self
        required = {" ".join(skill.split()).casefold() for skill in self.required_skills}
        conflicts = {
            skill for group in self.skill_requirements if group.mode == "any" and len(group.skills) > 1
            for skill in group.skills if " ".join(skill.split()).casefold() in required
        }
        if conflicts:
            names = "、".join(sorted(conflicts)[:5])
            raise ValueError(
                f"任选组成员被重复列为逐项必需：{names}。按JD原文修正；仅任选时移除重复必需项，"
                "若原文另有独立强制要求则保留该必需项，并移除已被满足的冗余任选组。"
            )
        return self


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


class VerificationIssue(Schema):
    """最终确认的问题；不承载模型的思考过程。"""

    kind: Literal["unsupported_claim", "proficiency_overstatement", "requirement_logic", "hard_requirement", "temporal_reasoning"]
    field: str = Field(default="", max_length=80, description="可省略；提供claim_id时由程序确定字段。")
    claim_id: str | None = Field(
        default=None,
        pattern=r"^(?:(?:matched_skills|missing_skills|strengths|risks|suggestions):[1-9]\d{0,4}|scores:(?:skill|experience|responsibility|education|bonus))$",
        description="选择本轮报告条目的真实id，原句由程序回填。",
    )
    claim: str = Field(default="", max_length=160)
    source: Literal["resume", "jd"] = Field(default="resume", description="提供evidence_id时由程序确定来源。")
    evidence_id: str | None = Field(
        default=None, pattern=r"^(resume|jd):[1-9]\d{0,4}$",
        description="选取输入原文片段的id，由程序回填原文；不得编造编号。",
    )
    evidence: str = Field(default="", max_length=240, description="逐字引用原文；原文完全缺乏依据时可为空")
    problem: str = Field(min_length=1, max_length=160)
    suggested_fix: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def require_claim_reference(self) -> VerificationIssue:
        if not self.claim_id and not self.claim:
            raise ValueError("需选择报告条目claim_id，或提供可核对的报告原句")
        if not self.claim_id and not self.field:
            raise ValueError("旧版原句引用需要提供对应字段")
        return self


class VerificationFailureDetail(Schema):
    field: str = Field(max_length=160)
    code: str = Field(max_length=80)


class VerificationAttemptFailure(Schema):
    round: int = Field(ge=1, le=4)
    code: str = Field(max_length=80)
    details: list[VerificationFailureDetail] = Field(default_factory=list, max_length=8)


class VerificationSummary(Schema):
    status: Literal["passed", "needs_review", "correction_unverified", "timeout", "service_error", "manual_accepted"]
    summary: str = Field(max_length=240)
    issues: list[VerificationIssue] = Field(default_factory=list, max_length=3)
    corrections: list[str] = Field(default_factory=list, max_length=6)
    rounds: int = Field(ge=0)
    correction_applied: bool = False
    failure_code: Literal[
        "invalid_json", "invalid_schema", "invalid_evidence", "output_truncated", "empty_output",
        "timeout", "model_service_error", "budget_exceeded",
    ] | None = None
    failure_details: list[VerificationFailureDetail] = Field(default_factory=list, max_length=8)
    attempt_failures: list[VerificationAttemptFailure] = Field(default_factory=list, max_length=4)
    nonblocking: bool = False
    unresolved_issues: list[VerificationIssue] = Field(default_factory=list, max_length=3)


class MatchReport(Schema):
    """最终岗位匹配报告。"""

    assessment_date: date | None = None
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
    verification: VerificationSummary | None = None


class MatchRequest(Schema):
    """岗位匹配请求。"""

    resume_document_id: uuid.UUID
    research_approved: bool = Field(default=False, description="是否授权检索公开岗位资料")
    jd_document_id: uuid.UUID | None = None
    jd_text: str | None = Field(
        default=None,
        max_length=settings.MAX_DOCUMENT_TEXT_CHARS,
    )

    @model_validator(mode="after")
    def validate_jd_source(self) -> MatchRequest:
        if self.jd_document_id is None and not self.jd_text:
            raise ValueError("jd_document_id 和 jd_text 至少提供一项")

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
    questions: list[InterviewQuestion] = Field(default_factory=list)


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
# 结果校验与自我纠错 ⭐ 新增
# ----------------------------------------------------------------------


class MatchEvaluationPatch(Schema):
    """Only changed fields; scores, when changed, still use the complete five dimensions."""

    scores: MatchScores | None = None
    matched_skills: list[str] | None = None
    missing_skills: list[str] | None = None
    strengths: list[str] | None = None
    risks: list[str] | None = None
    suggestions: list[str] | None = None

    @model_validator(mode="after")
    def require_change(self) -> MatchEvaluationPatch:
        if not self.model_dump(exclude_none=True):
            raise ValueError("修正字段不能为空；删除全部风险时使用 risks=[]")
        return self


class VerificationResult(Schema):
    """Self-Correction 验证节点输出。"""

    is_faithful: bool = Field(
        description="生成的内容（如匹配报告、分析等）是否完全基于提供的文档事实（如简历、JD等）？是否存在毫无依据的编造（幻觉）？"
    )
    reason: str = Field(
        min_length=1,
        max_length=120,
        description="只给最终结论，最多120字；不输出推理过程、自问自答或已撤回的疑点",
    )
    issues: list[VerificationIssue] = Field(default_factory=list, max_length=3)
    corrected_fields: MatchEvaluationPatch | None = Field(
        default=None, description="仅返回需要替换的字段，列表提供修正后的完整列表；删除全部风险用 risks=[]。",
    )
    corrected_evaluation: MatchEvaluation | None = Field(
        default=None,
        description="兼容旧版完整修正输出；优先使用 corrected_fields，此字段保持 null，二者不得同时填写。",
    )

    @model_validator(mode="after")
    def validate_verdict(self) -> VerificationResult:
        if self.corrected_fields is not None and self.corrected_evaluation is not None:
            raise ValueError("只使用一种修正方式")
        if self.is_faithful and (self.issues or self.corrected_evaluation is not None or self.corrected_fields is not None):
            raise ValueError("校验通过时不得同时返回问题或修正版")
        if not self.is_faithful and not self.issues:
            raise ValueError("校验未通过时必须提供具体问题")
        return self


# ----------------------------------------------------------------------
# Agent 任务
# ----------------------------------------------------------------------


class RunResponse(Schema):
    run_id: uuid.UUID
    task_type: str
    status: str
    result: dict | None = None
    error_message: str | None = None
    input_data: dict | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
