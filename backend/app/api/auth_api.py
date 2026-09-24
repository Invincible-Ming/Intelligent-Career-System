"""Registration/login/logout; registration can never create an administrator."""
import asyncio
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.security.auth import AuthenticatedUser, CurrentUser, anonymous_auth_limit, consume_limits, hash_password, \
    token_hash, verify_password
from app.core.config import settings
from app.core.database import get_db
from app.core.models import AuthSession, User

router = APIRouter(prefix="/auth", tags=["登录"])
DatabaseSession = Annotated[AsyncSession, Depends(get_db, scope="request")]
DUMMY_HASH = hash_password(secrets.token_urlsafe(24))


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(min_length=12, max_length=128)

    @field_validator("username")
    @classmethod
    def lowercase(cls, value):
        return value.lower()


def user_response(user):
    return {"id": str(user.id), "username": user.username, "is_admin": user.is_admin}


@router.get("/options")
async def options():
    return {"registration_enabled": settings.AUTH_REGISTRATION_ENABLED,
            "auth_cookie_name": settings.AUTH_COOKIE_NAME,
            "chat_max_input_chars": settings.CHAT_MAX_INPUT_CHARS, "chat_context_budget": settings.CHAT_CONTEXT_BUDGET,
            "chat_max_output_tokens": settings.CHAT_MAX_OUTPUT_TOKENS,
            "generation_requests_per_minute": settings.GENERATION_REQUESTS_PER_MINUTE,
            "generation_requests_per_day": settings.GENERATION_REQUESTS_PER_DAY,
            "upload_total_timeout": settings.UPLOAD_TOTAL_TIMEOUT, "api_total_timeout": settings.API_TOTAL_TIMEOUT,
            "chat_total_timeout": settings.CHAT_TOTAL_TIMEOUT,
            "analysis_total_timeout": settings.ANALYSIS_TOTAL_TIMEOUT,
            "match_total_timeout": settings.MATCH_TOTAL_TIMEOUT,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB, "jd_max_input_chars": settings.MAX_DOCUMENT_TEXT_CHARS,
            "feedback_max_input_chars": 2000, "search_max_input_chars": 2000}


@router.post("/register", status_code=201, dependencies=[Depends(anonymous_auth_limit)])
async def register(credentials: Credentials, session: DatabaseSession):
    if not settings.AUTH_REGISTRATION_ENABLED:
        raise HTTPException(403, "当前未开放注册")
    user = User(username=credentials.username,
                password_hash=await asyncio.to_thread(hash_password, credentials.password), is_admin=False)
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "该用户名不可用") from None
    return user_response(user)


@router.post("/login", dependencies=[Depends(anonymous_auth_limit)])
async def login(credentials: Credentials, session: DatabaseSession):
    import hashlib
    username_key = hashlib.sha256(credentials.username.encode()).hexdigest()
    await consume_limits([(f"login-name:{username_key}", 60, 5)])
    user = (await session.execute(select(User).where(User.username == credentials.username))).scalar_one_or_none()
    valid = await asyncio.to_thread(verify_password, credentials.password, user.password_hash if user else DUMMY_HASH)
    if not valid or not user or not user.is_active:
        raise HTTPException(401, "用户名或密码不正确")
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=settings.AUTH_SESSION_HOURS)
    session.add(AuthSession(token_hash=token_hash(token), user_id=user.id, expires_at=expires))
    await session.commit()
    return {"access_token": token, "token_type": "bearer", "expires_at": expires.isoformat(),
            "user": user_response(user)}


@router.get("/me")
async def me(user: CurrentUser):
    return user_response(user)


@router.post("/logout")
async def logout(request: Request, user: AuthenticatedUser, session: DatabaseSession):
    token = request.headers["authorization"].split(None, 1)[1]
    await session.execute(
        delete(AuthSession).where(AuthSession.token_hash == token_hash(token), AuthSession.user_id == user.id))
    await session.commit()
    return {"message": "已退出登录"}


def browser_return_url(request: Request, return_to: str) -> str:
    """Only allow an approved frontend sharing this host-only cookie."""
    parsed = urlsplit(return_to)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if (origin not in settings.cors_origin_list or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or parsed.username or parsed.password
            or parsed.hostname != request.url.hostname):
        raise HTTPException(400, "登录返回地址必须是同域的已配置前端地址")
    if settings.AUTH_COOKIE_SECURE and parsed.scheme != "https":
        raise HTTPException(400, "安全 Cookie 需要 HTTPS 前端地址")
    return origin + "/"


def check_browser_origin(request: Request, return_to: str) -> str:
    destination = browser_return_url(request, return_to)
    backend_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") not in (backend_origin, destination.rstrip("/")):
        raise HTTPException(403, "不允许此来源提交登录操作")
    return destination


@router.post("/browser-login", include_in_schema=False)
async def browser_login(request: Request, session: DatabaseSession,
                        return_to: Annotated[str, Form(max_length=2048)],
                        username: Annotated[str, Form(max_length=32)],
                        password: Annotated[str, Form(max_length=128)]):
    check_browser_origin(request, return_to)
    try:
        await anonymous_auth_limit(request)
        credentials = Credentials(username=username, password=password)
        result = await login(credentials, session)
    except ValidationError:
        return JSONResponse({"detail": "请检查用户名格式及密码长度"}, status_code=422,
                            headers={"Cache-Control": "no-store"})
    except HTTPException as exc:
        if exc.status_code not in (401, 429):
            raise
        return JSONResponse({"detail": str(exc.detail)}, status_code=exc.status_code,
                            headers={"Cache-Control": "no-store", **(exc.headers or {})})
    response = JSONResponse({"logged_in": True}, headers={"Cache-Control": "no-store"})
    response.set_cookie(settings.AUTH_COOKIE_NAME, result["access_token"],
                        max_age=settings.AUTH_SESSION_HOURS * 3600,
                        expires=datetime.fromisoformat(result["expires_at"]),
                        httponly=True, secure=settings.AUTH_COOKIE_SECURE or request.url.scheme == "https",
                        samesite="lax", path="/")
    return response


@router.post("/browser-logout", include_in_schema=False)
async def browser_logout(request: Request, session: DatabaseSession,
                         return_to: Annotated[str, Form(max_length=2048)]):
    # Clearing an expired/revoked cookie must also succeed without authentication.
    check_browser_origin(request, return_to)
    token = request.cookies.get(settings.AUTH_COOKIE_NAME, "")
    if 32 <= len(token) <= 128:
        await session.execute(delete(AuthSession).where(AuthSession.token_hash == token_hash(token)))
        await session.commit()
    response = JSONResponse({"cleared": True}, headers={"Cache-Control": "no-store"})
    response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/", httponly=True,
                           secure=settings.AUTH_COOKIE_SECURE or request.url.scheme == "https", samesite="lax")
    return response
