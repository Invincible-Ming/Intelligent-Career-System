"""Document identities and presentation order; filenames never identify a person."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

DISPLAY_TIMEZONE = timezone(timedelta(hours=8))


def document_created_at(document: dict[str, Any]) -> datetime:
    value = document.get("created_at")
    try:
        created = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # Backend timestamps are UTC; older responses may omit the offset.
        return (created if created.tzinfo else created.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return datetime.min.replace(tzinfo=timezone.utc)


def document_upload_time(document: dict[str, Any]) -> str:
    created = document_created_at(document)
    if created.year == 1:
        return "上传时间未知"
    return created.astimezone(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def prepare_document_list(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate repeated response rows by ID, preserving distinct same-name files."""
    by_id: dict[str, dict[str, Any]] = {}
    for document in documents:
        identifier = str(document.get("id") or "")
        if not identifier:
            continue
        previous = by_id.get(identifier)
        if previous is None or document_created_at(document) > document_created_at(previous):
            by_id[identifier] = {**document, "id": identifier}
    ordered = sorted(by_id.values(), key=lambda doc: (document_created_at(doc), doc["id"]), reverse=True)
    counts = Counter((doc.get("document_type"), doc.get("filename")) for doc in ordered)
    seen: set[tuple[Any, Any]] = set()
    for document in ordered:
        group = (document.get("document_type"), document.get("filename"))
        document["_same_name_count"] = counts[group]
        document["_same_name_latest"] = group not in seen
        seen.add(group)
    return ordered


def document_name_marker(document: dict[str, Any]) -> str:
    if document.get("_same_name_count", 0) <= 1:
        return ""
    return "同名最新" if document.get("_same_name_latest") else "同名历史"


def document_identity_label(document: dict[str, Any]) -> str:
    return f"{document.get('filename', '未命名文档')} · {document_upload_time(document)}"
