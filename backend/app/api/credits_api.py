"""Authenticated wallet, daily check-in and owner-scoped receipts."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.models import CreditEntry
from app.security.auth import CurrentUser
from app.services.credits import check_in, wallet_info

router = APIRouter(prefix="/credits", tags=["积分"])
DatabaseSession = Annotated[AsyncSession, Depends(get_db, scope="request")]


@router.get("/me")
async def my_credits(user: CurrentUser, session: DatabaseSession):
    return await wallet_info(session, user.id)


@router.post("/check-in")
async def daily_check_in(user: CurrentUser, session: DatabaseSession):
    return await check_in(session, user.id)


@router.get("/entries")
async def my_entries(user: CurrentUser, session: DatabaseSession,
                     offset: int = Query(default=0, ge=0, le=100000),
                     limit: int = Query(default=20, ge=1, le=100),
                     kind: Literal["initial", "checkin", "usage"] | None = None,
                     status: Literal["pending", "settled", "released"] | None = None):
    conditions = [CreditEntry.user_id == user.id]
    if kind is not None:
        conditions.append(CreditEntry.kind == kind)
    if status is not None:
        conditions.append(CreditEntry.status == status)
    rows = (await session.execute(select(CreditEntry).where(*conditions)
                                  .order_by(CreditEntry.created_at.desc(), CreditEntry.id.desc())
                                  .offset(offset).limit(limit + 1))).scalars().all()
    return {"has_more": len(rows) > limit, "items": [
        {"id": str(row.id), "kind": row.kind, "status": row.status, "points": row.delta_units / 1000,
         "reserved": row.reserved_units / 1000, "model": row.model, "operation": row.operation,
         "input_tokens": row.input_tokens, "output_tokens": row.output_tokens,
         "usage_source": row.usage_source,
         "balance_after": row.balance_after_units / 1000 if row.balance_after_units is not None else None,
         "created_at": row.created_at.isoformat()}
        for row in rows[:limit]
    ]}
