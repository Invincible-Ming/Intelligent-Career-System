"""Narrow product rules applied before and after model review."""

from __future__ import annotations

import re
from typing import Any

from app.core.evidence import build_report_claims, normalized_quote

UNVERIFIED_SUMMARY = "匹配报告已生成，自动校验未完成，结果暂未经完整复核。"

_AVAILABILITY_TOPIC = re.compile(r"连续.{0,8}实习|到岗|可实习|实习安排|每周.{0,8}天")
_CONFIRMATION_ONLY = re.compile(r"需(?:要)?(?:确认|核实|沟通)|待(?:确认|核实|沟通)|未(?:说明|明确|提及|体现)|尚未明确")
_EXPLICIT_CONFLICT = re.compile(r"明确.{0,8}(?:不满足|不符合)|只能|仅能|无法|冲突|不接受|不足")
_PAST_EXPERIENCE = re.compile(r"既往|过往|累计|经验年限|实习经验|工作经验")
_EXPLICIT_AVAILABILITY = re.compile(
    r"(?:可|能|只能|仅能|计划|承诺|保证|最多|至少).{0,10}(?:连续)?(?:实习|到岗).{0,8}"
    r"(?:[0-9一二两三四五六七八九十]+\s*(?:个?月|周|天))"
    r"|每周.{0,8}(?:到岗|实习).{0,6}(?:[0-9一二两三四五六七八九十]+\s*天)"
    r"|(?:入职|到岗)(?:日期|时间)\s*[:：]\s*\d"
)


def apply_match_policies(
    report: dict[str, Any], *, resume_text: str, jd_text: str,
) -> tuple[dict[str, Any], list[str]]:
    """Move uncertain future availability reminders; never decide an actual eligibility conflict."""
    if _EXPLICIT_AVAILABILITY.search(resume_text):
        return report, []

    risks = report.get("risks", [])
    kept = []
    moved = []
    for risk in risks:
        is_unconfirmed_arrangement = (
            _AVAILABILITY_TOPIC.search(risk)
            and _CONFIRMATION_ONLY.search(risk)
            and not _EXPLICIT_CONFLICT.search(risk)
            and not _PAST_EXPERIENCE.search(risk)
        )
        (moved if is_unconfirmed_arrangement else kept).append(risk)
    if not moved:
        return report, []

    result = dict(report)
    result["risks"] = kept
    suggestions = list(report.get("suggestions", []))
    # Keep one ordinary HR reminder only when the JD actually mentions arrangements.
    if _AVAILABILITY_TOPIC.search(jd_text) and not any(_AVAILABILITY_TOPIC.search(item) for item in suggestions):
        suggestions.append("面试时可与 HR 沟通到岗时间和实习安排。")
    result["suggestions"] = suggestions
    return result, ["到岗安排未说明，已按待沟通事项处理。"]


def unresolved_report_issues(issues: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    """Track known offending claims still present; a changed claim remains unverified, not proven correct."""
    claims = build_report_claims(report)
    remaining = []
    for issue in issues:
        field = str(issue.get("field", "")).split("[", 1)[0].split(".", 1)[0]
        quote = normalized_quote(str(issue.get("claim", "")))
        candidates = [claim for claim in claims.values() if claim.field == field]
        if not quote or field not in {"scores", "matched_skills", "missing_skills", "strengths", "risks", "suggestions"}:
            remaining.append(issue)
        elif any(quote in normalized_quote(claim.text) for claim in candidates):
            remaining.append(issue)
    return remaining


def can_continue_unverified(report: dict[str, Any]) -> bool:
    review = report.get("verification") or {}
    if review.get("status") not in {"timeout", "service_error", "correction_unverified"}:
        return False
    return not unresolved_report_issues(review.get("issues") or [], report)
