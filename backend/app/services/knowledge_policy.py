"""Resolve retrieval entitlements exclusively from authenticated database records."""
from datetime import UTC, datetime

from app.core.config import settings

PLAN_NAMES = {"free": "免费版", "starter": "入门版", "pro": "专业版", "enterprise": "企业版"}
PAID_PLANS = frozenset(PLAN_NAMES) - {"free"}


def subscription_info(user, *, now=None):
    now = now or datetime.now(UTC)
    subscription = getattr(user, "subscription", None)
    plan = getattr(subscription, "plan", "free")
    expires_at = getattr(subscription, "expires_at", None)
    active = plan in PAID_PLANS and expires_at is not None and expires_at > now
    effective = plan if active else "free"
    return {"plan": effective, "plan_name": PLAN_NAMES[effective], "active": active,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "expired": plan in PAID_PLANS and expires_at is not None and expires_at <= now}


def knowledge_provider_for(user):
    if settings.KNOWLEDGE_PROVIDER == "credits":
        preference = getattr(user, "knowledge_preference", None)
        return "milvus" if getattr(preference, "provider", None) == "milvus" else "ima"
    if settings.KNOWLEDGE_PROVIDER != "subscription":
        return settings.KNOWLEDGE_PROVIDER
    if subscription_info(user)["active"]:
        return "milvus"
    return "ima"
