"""Manual subscription grants; only administrators can change entitlements."""
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.models import SubscriptionAudit, User, UserSubscription
from app.security.auth import CurrentUser, require_admin
from app.services.knowledge_policy import knowledge_provider_for, subscription_info

router = APIRouter(prefix="/subscriptions", tags=["订阅"])
DatabaseSession = Annotated[AsyncSession, Depends(get_db, scope="request")]
AdminUser = Annotated[User, Depends(require_admin)]


class SubscriptionGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: Literal["free", "starter", "pro", "enterprise"]
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_expiry(self):
        if self.plan == "free":
            if self.expires_at is not None:
                raise ValueError("免费版无需设置到期时间")
        elif (self.expires_at is None or self.expires_at.tzinfo is None
              or self.expires_at <= datetime.now(UTC)):
            raise ValueError("订阅必须设置带时区的未来到期时间")
        return self


@router.get("/me")
async def my_subscription(user: CurrentUser):
    return {**subscription_info(user), "knowledge_provider": knowledge_provider_for(user)}


@router.get("/users")
async def subscription_users(user: AdminUser, session: DatabaseSession,
                             query: str = Query(default="", max_length=32)):
    stmt = select(User).order_by(User.username).limit(50)
    if query.strip():
        stmt = stmt.where(User.username.contains(query.strip().lower(), autoescape=True))
    users = (await session.execute(stmt)).scalars().all()
    return [{"id": str(item.id), "username": item.username, "is_admin": item.is_admin,
             "subscription": subscription_info(item)} for item in users]


@router.put("/users/{user_id}")
async def grant_subscription(user_id: uuid.UUID, request: SubscriptionGrant,
                             user: AdminUser, session: DatabaseSession):
    # Serialize changes for the same account across workers; keep a complete grant audit.
    target = (await session.execute(select(User).where(User.id == user_id).with_for_update())).scalar_one_or_none()
    if target is None:
        raise HTTPException(404, "账号不存在")
    previous = target.subscription.plan if target.subscription else "free"
    subscription = target.subscription
    if subscription is None:
        subscription = UserSubscription(user_id=target.id)
        target.subscription = subscription
    subscription.plan = request.plan
    subscription.expires_at = request.expires_at
    subscription.granted_by = user.id
    session.add(SubscriptionAudit(user_id=target.id, actor_id=user.id, previous_plan=previous,
                                  plan=request.plan, expires_at=request.expires_at))
    await session.flush()
    if knowledge_provider_for(target) == "milvus":
        from app.services.knowledge_index import knowledge_index_service

        await knowledge_index_service.enqueue_owner(session, target.id, retry_failed=True)
    await session.commit()
    return {"id": str(target.id), "username": target.username,
            "subscription": subscription_info(target), "knowledge_provider": knowledge_provider_for(target)}
