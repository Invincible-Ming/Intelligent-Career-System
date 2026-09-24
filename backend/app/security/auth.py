"""Opaque, revocable database sessions and server-owned identities."""
from __future__ import annotations

import asyncio
import anyio
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.core.limits import heavy_operation, operation_timeout
from app.core.models import AuthSession, OperationLease, RateLimitBucket, User

bearer = HTTPBearer(auto_error=False)


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=3, maxmem=64 * 1024 * 1024)
    return "scrypt$32768$8$3$" + salt.hex() + "$" + digest.hex()


def verify_password(password, encoded):
    try:
        algorithm, n, r, p, salt, digest = encoded.split("$")
        if (algorithm, n, r, p) != ("scrypt", "32768", "8", "3"):
            return False
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                                maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(actual.hex(), digest)
    except (ValueError, TypeError):
        return False


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


async def consume_limits(entries):
    """Atomic across users/workers, all counters roll back on a rejected request."""
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as session:
        for key, seconds, ceiling in entries:
            timestamp = int(now.timestamp())
            start = datetime.fromtimestamp(timestamp - timestamp % seconds, timezone.utc)
            statement = insert(RateLimitBucket).values(key=key, window_start=start, count=1)
            statement = statement.on_conflict_do_update(
                index_elements=[RateLimitBucket.key, RateLimitBucket.window_start],
                set_={"count": RateLimitBucket.count + 1}, where=RateLimitBucket.count < ceiling,
            ).returning(RateLimitBucket.count)
            if (await session.execute(statement)).scalar_one_or_none() is None:
                await session.rollback()
                remaining = seconds - timestamp % seconds
                raise HTTPException(429, "调用次数已达到限制，请稍后重试", headers={"Retry-After": str(remaining)})
        await session.commit()


async def anonymous_auth_limit(request: Request):
    address = request.client.host if request.client else "unknown"
    key = hashlib.sha256(address.encode()).hexdigest()
    await consume_limits([(f"auth-ip:{key}", 60, settings.AUTH_REQUESTS_PER_MINUTE)])


async def get_current_user(request: Request,
                           credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
                           session: Annotated[AsyncSession, Depends(get_db, scope="request")]) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer" or not 32 <= len(credentials.credentials) <= 128:
        raise HTTPException(401, "请先登录", headers={"WWW-Authenticate": "Bearer"})
    user = (await session.execute(select(User).join(AuthSession, AuthSession.user_id == User.id).where(
        AuthSession.token_hash == token_hash(credentials.credentials), AuthSession.expires_at > func.now(),
        User.is_active.is_(True),
    ))).scalar_one_or_none()
    if user is None:
        raise HTTPException(401, "登录已过期，请重新登录", headers={"WWW-Authenticate": "Bearer"})
    request.state.user_id = user.id
    return user


async def release_lease(identifier):
    async with AsyncSessionLocal() as session:
        await session.execute(delete(OperationLease).where(OperationLease.id == identifier))
        await session.commit()


async def operation_user(request: Request, user: Annotated[User, Depends(get_current_user)]):
    await consume_limits([(f"api:{user.id}", 60, settings.API_REQUESTS_PER_MINUTE)])
    lease_id = None
    if heavy_operation(request.url.path, request.method):
        await consume_limits([(f"generation-minute:{user.id}", 60, settings.GENERATION_REQUESTS_PER_MINUTE),
                              (f"generation-day:{user.id}", 86400, settings.GENERATION_REQUESTS_PER_DAY),
                              ("generation-global-day", 86400, settings.GLOBAL_GENERATION_REQUESTS_PER_DAY)])
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                                  {"key": str(user.id)})
            await session.execute(delete(OperationLease).where(OperationLease.owner_id == user.id,
                                                               OperationLease.expires_at <= func.now()))
            active = (await session.execute(select(func.count()).select_from(OperationLease).where(
                OperationLease.owner_id == user.id))).scalar_one()
            if active >= settings.MAX_CONCURRENT_OPERATIONS:
                raise HTTPException(429, "同时执行的操作过多，请等待当前任务结束", headers={"Retry-After": "5"})
            lease = OperationLease(owner_id=user.id, expires_at=datetime.now(timezone.utc) + timedelta(
                seconds=operation_timeout(request.url.path) + 10))
            session.add(lease)
            await session.commit()
            lease_id = lease.id
    try:
        yield user
    finally:
        if lease_id:
            with anyio.CancelScope(shield=True):
                try:
                    async with asyncio.timeout(3):
                        await release_lease(lease_id)
                except (TimeoutError, asyncio.CancelledError):
                    pass  # Expiring leases also recover crashed/disconnected workers.


CurrentUser = Annotated[User, Depends(operation_user, scope="request")]
AuthenticatedUser = Annotated[User, Depends(get_current_user)]


async def require_admin(user: CurrentUser):
    if not user.is_admin:
        raise HTTPException(403, "该功能仅限管理员")
    return user


async def owned_record(session, model, identifier, owner_id):
    record = (await session.execute(
        select(model).where(model.id == identifier, model.owner_id == owner_id))).scalar_one_or_none()
    if record is None:
        raise HTTPException(404, "记录不存在或不可访问")
    return record
