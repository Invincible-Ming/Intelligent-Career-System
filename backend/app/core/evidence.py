"""Stable references into the exact source text supplied to the verifier."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EvidenceSpan:
    source: Literal["resume", "jd"]
    text: str


@dataclass(frozen=True)
class ReportClaim:
    field: str
    text: str


def build_report_claims(report: dict) -> dict[str, ReportClaim]:
    claims = {}
    for field in ("matched_skills", "missing_skills", "strengths", "risks", "suggestions"):
        for index, text in enumerate(report.get(field, []), start=1):
            claims[f"{field}:{index}"] = ReportClaim(field, text)
    for field, score in report.get("scores", {}).items():
        claims[f"scores:{field}"] = ReportClaim("scores", f"{field}: {score}")
    return claims


def report_claims_payload(claims: dict[str, ReportClaim]) -> list[dict[str, str]]:
    return [{"id": identifier, "field": claim.field, "text": claim.text} for identifier, claim in claims.items()]


def source_keyword_locations(sources: dict[str, EvidenceSpan], keywords: list[str]) -> dict[str, list[str]]:
    """Lexical locations only: an award mentioning Java does not establish proficiency."""
    locations = {}
    for keyword in keywords[:100]:
        if not keyword or len(keyword) > 80 or keyword in locations:
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(keyword) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
        matches = [identifier for identifier, span in sources.items() if span.source == "resume" and pattern.search(span.text)]
        if matches:
            locations[keyword] = matches[:5]
    return locations


def contradicted_absence_locations(problem: str, sources: dict[str, EvidenceSpan]) -> list[str]:
    """Reject only absolute lexical-absence claims, not claims about unproven experience or depth."""
    pattern = r"(?:未(?:明确)?提[及到]|没有提及|未出现)\s*[“\"‘']?([A-Za-z][A-Za-z0-9_+#.-]*)[”\"’']?(?:\s*语言)?(?=\s*[，。；;,.]|\s*$)"
    matches = []
    for keyword in re.findall(pattern, problem):
        for identifiers in source_keyword_locations(sources, [keyword]).values():
            matches.extend(identifiers)
    return list(dict.fromkeys(matches))[:5]


def build_evidence_sources(resume_text: str, jd_text: str) -> dict[str, EvidenceSpan]:
    sources: dict[str, EvidenceSpan] = {}
    for source, original in (("resume", resume_text), ("jd", jd_text)):
        index = 0
        for match in re.finditer(r"[^\r\n]+", original[:30000]):
            line = match[0]
            for start in range(0, len(line), 220):
                excerpt = line[start:start + 220]
                if not excerpt.strip():
                    continue
                index += 1
                sources[f"{source}:{index}"] = EvidenceSpan(source=source, text=excerpt)
    return sources


def evidence_payload(sources: dict[str, EvidenceSpan]) -> dict[str, list[dict[str, str]]]:
    payload: dict[str, list[dict[str, str]]] = {"resume": [], "jd": []}
    for identifier, span in sources.items():
        payload[span.source].append({"id": identifier, "text": span.text})
    return payload


def normalized_quote(text: str) -> str:
    """Only typography/whitespace normalization; never semantic or fuzzy matching."""
    return "".join(unicodedata.normalize("NFKC", text).split())
