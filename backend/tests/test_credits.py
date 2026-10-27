"""Real PostgreSQL wallet concurrency; synthetic models only, never production resources."""
import asyncio
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_openai import ChatOpenAI
from psycopg import sql
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import api, chat_api, credits_api
from app.api.auth_api import router as auth_router
from app.core.config import settings
from app.core.database import PSYCOPG_DATABASE_URL, Base, get_db
from app.core.limits import SecurityLimitsMiddleware
from app.core.models import AuthSession, CreditEntry, User
from app.security import auth
from app.services import credits, knowledge_index
from app.services.bailian import BailianService, ModelServiceError, bailian_service
from app.services.credits import CreditModelCallback, InsufficientCredits, ModelCharge, billing_scope


@pytest.fixture
async def credit_env(monkeypatch):
    dbname = "career_credits_test_" + uuid.uuid4().hex[:12]
    admin = await psycopg.AsyncConnection.connect(PSYCOPG_DATABASE_URL, autocommit=True)
    await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    engine = create_async_engine(make_url(settings.DATABASE_URL).set(database=dbname))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    for module in (auth, credits, knowledge_index):
        monkeypatch.setattr(module, "AsyncSessionLocal", factory)
    monkeypatch.setattr(chat_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(chat_api, "engine", engine)
    monkeypatch.setattr(chat_api, "get_chat_agent_tools", lambda **kwargs: [])
    monkeypatch.setattr(auth, "consume_limits", AsyncMock())
    monkeypatch.setattr(settings, "KNOWLEDGE_PROVIDER", "credits")
    monkeypatch.setattr(settings, "CREDITS_INITIAL_POINTS", 100)
    monkeypatch.setattr(settings, "CREDITS_DAILY_CHECKIN_POINTS", 10)
    monkeypatch.setattr(settings, "CREDITS_UNITS_PER_TOKEN", 1)
    monkeypatch.setattr(settings, "MODEL_MAX_RETRIES", 0)
    app = FastAPI()
    app.add_middleware(SecurityLimitsMiddleware)
    for router in (auth_router, credits_api.router, api.router, chat_api.router):
        app.include_router(router, prefix="/api")

    async def db():
        async with factory() as session:
            yield session
    app.dependency_overrides[get_db] = db

    async def account(name="alice", is_admin=False):
        token = secrets.token_urlsafe(32)
        async with factory() as session:
            user = User(username=name, password_hash="unused", is_admin=is_admin)
            session.add(user)
            await session.flush()
            session.add(AuthSession(token_hash=auth.token_hash(token), user_id=user.id,
                                    expires_at=datetime.now(UTC) + timedelta(hours=1)))
            await session.commit()
            return SimpleNamespace(id=user.id, headers={"Authorization": f"Bearer {token}"})

    @app.post("/model")
    async def model(user: auth.CurrentUser):
        charge = await ModelCharge.begin("fake", [{"content": "q"}], 200000)
        await charge.finish()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        try:
            yield SimpleNamespace(client=client, factory=factory, account=account)
        finally:
            await engine.dispose()
            await admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(dbname)))
            await admin.close()


async def wallet(env, owner):
    async with env.factory() as session:
        return await credits.wallet_info(session, owner.id)


@pytest.mark.anyio
async def test_initial_gift_is_once_for_old_accounts_and_repeated_logins(credit_env):
    env = credit_env
    owner = await env.account()
    replies = await asyncio.gather(*(env.client.get("/api/credits/me", headers=owner.headers) for _ in range(5)))
    assert all(reply.json()["balance"] == 100 for reply in replies)
    assert (await env.client.get("/api/auth/me", headers=owner.headers)).json()["credits"]["balance"] == 100
    rows = (await env.client.get("/api/credits/entries", headers=owner.headers)).json()["items"]
    assert [row["kind"] for row in rows] == ["initial"]


@pytest.mark.anyio
async def test_receipt_filters_apply_before_pagination_and_preserve_ownership(credit_env):
    env = credit_env
    alice, bob = await env.account('alice'), await env.account('bob')
    await wallet(env, alice)
    await wallet(env, bob)
    await env.client.post('/api/credits/check-in', headers=alice.headers)
    response = await env.client.get('/api/credits/entries', headers=alice.headers,
                                    params={'kind': 'checkin', 'status': 'settled', 'limit': 1})
    assert response.status_code == 200
    assert len(response.json()['items']) == 1
    assert response.json()['items'][0]['kind'] == 'checkin'
    assert not response.json()['has_more']
    foreign = await env.client.get('/api/credits/entries', headers=bob.headers, params={'kind': 'checkin'})
    assert foreign.json()['items'] == []
    invalid = await env.client.get('/api/credits/entries', headers=bob.headers, params={'kind': 'all_users'})
    assert invalid.status_code == 422


@pytest.mark.anyio
async def test_register_gift_and_client_cannot_assign_points(credit_env):
    env = credit_env
    request = {"username": "new_user", "password": "a-test-password-long"}
    response = await env.client.post("/api/auth/register", json=request)
    assert response.status_code == 201
    assert response.json()["credits"]["balance"] == 100
    assert response.json()["knowledge_provider"] == "ima"
    invalid = await env.client.post("/api/auth/register", json={**request, "credits": 99999})
    assert invalid.status_code == 422


@pytest.mark.anyio
async def test_checkin_concurrency_beijing_midnight_and_owner_boundary(credit_env):
    env = credit_env
    alice, bob = await env.account(), await env.account("bob")
    replies = await asyncio.gather(*(env.client.post("/api/credits/check-in", headers=alice.headers) for _ in range(8)))
    assert sum(reply.json()["awarded"] for reply in replies) == 1
    assert (await wallet(env, alice))["balance"] == 110
    assert (await wallet(env, bob))["balance"] == 100
    assert (await env.client.get("/api/credits/me")).status_code == 401
    assert (await env.client.get("/api/credits/entries?user_id=" + str(alice.id), headers=bob.headers)).json()["items"][0]["kind"] == "initial"
    # A fresh account is checked in just before/after Beijing midnight using server-side time.
    carol = await env.account("carol")
    async with env.factory() as session:
        before = datetime(2026, 10, 22, 15, 59, 59, tzinfo=UTC)
        first = await credits.check_in(session, carol.id, now=before)
        same_day = await credits.check_in(session, carol.id, now=before)
        next_day = await credits.check_in(session, carol.id, now=before + timedelta(seconds=1))
    assert first["awarded"] and not same_day["awarded"] and next_day["awarded"]
    assert next_day["balance"] == 120


@pytest.mark.anyio
async def test_reserve_settle_exact_usage_idempotency_and_concurrent_overspend(credit_env):
    env = credit_env
    owner = await env.account()
    with billing_scope(owner.id, "test"):
        first = await ModelCharge.begin("fake", [{"content": "q"}], 60000)
        with pytest.raises(InsufficientCredits):
            await ModelCharge.begin("fake", [{"content": "q"}], 60000)
        info = await wallet(env, owner)
        assert info["reserved"] == 60.033
        first.record(usage={"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200}, output="ok")
        await first.finish()
        await first.finish()
    info = await wallet(env, owner)
    assert info["balance"] == 99.8 and info["reserved"] == 0
    async with env.factory() as session:
        row = await session.get(CreditEntry, first.entry_id)
        assert row.delta_units == -200 and row.usage_source == "provider"


@pytest.mark.anyio
async def test_parallel_model_calls_cannot_reserve_same_balance(credit_env):
    env = credit_env
    owner = await env.account()
    async def attempt():
        try:
            return await ModelCharge.begin("fake", [{"content": "q"}], 60000)
        except InsufficientCredits:
            return None
    with billing_scope(owner.id, "parallel"):
        results = await asyncio.gather(attempt(), attempt())
        assert sum(result is not None for result in results) == 1
        await next(result for result in results if result is not None).finish()
    assert (await wallet(env, owner))["balance"] == 100
    assert (await wallet(env, owner))["reserved"] == 0


@pytest.mark.anyio
async def test_failed_call_refunds_and_partial_stream_is_estimated(credit_env):
    env = credit_env
    owner = await env.account()
    with billing_scope(owner.id, "test"):
        failed = await ModelCharge.begin("fake", [{"content": "q"}], 1000)
        await failed.finish()
        partial = await ModelCharge.begin("fake", [{"content": "q"}], 1000)
        partial.record(output="hello", completed=False)
        await partial.finish()
    assert (await wallet(env, owner))["balance"] == 99.962
    async with env.factory() as session:
        assert (await session.get(CreditEntry, failed.entry_id)).status == "released"
        assert (await session.get(CreditEntry, partial.entry_id)).usage_source == "estimated"


@pytest.mark.anyio
async def test_stale_reservation_recovery_and_charge_cap(credit_env):
    env = credit_env
    owner = await env.account()
    with billing_scope(owner.id, "test"):
        stale = await ModelCharge.begin("fake", [{"content": "q"}], 10)
        async with env.factory() as session:
            row = await session.get(CreditEntry, stale.entry_id)
            row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
        assert (await wallet(env, owner))["reserved"] == 0
        stale.record(usage={"total_tokens": 1000})
        await stale.finish()
        big = await ModelCharge.begin("fake", [{"content": "q"}], 10)
        big.record(usage={"total_tokens": 200000})
        await big.finish()
    assert (await wallet(env, owner))["balance"] == 99.957


@pytest.mark.anyio
async def test_402_before_provider_request_for_admin_too(credit_env):
    env = credit_env
    owner = await env.account(is_admin=True)
    result = await env.client.post("/model", headers=owner.headers)
    assert result.status_code == 402 and "积分不足" in result.json()["detail"]
    assert (await wallet(env, owner))["balance"] == 100


@pytest.mark.anyio
async def test_real_service_chat_and_embedding_meter_provider_usage(credit_env, monkeypatch):
    env = credit_env
    owner = await env.account()
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "synthetic")
    model_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                                     usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 20, "completion_tokens": 30}))
    request = AsyncMock(return_value=model_response)
    service = BailianService()
    service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
    embedding = SimpleNamespace(status_code=200, usage={"total_tokens": 10})
    monkeypatch.setattr("app.services.bailian.dashscope.TextEmbedding.call", lambda **kwargs: embedding)
    with billing_scope(owner.id, "test"):
        assert await service.chat([{"role": "user", "content": "q"}]) == "answer"
        await service._embedding_retry(["query"])
        request.side_effect = RuntimeError("synthetic failed call")
        with pytest.raises(ModelServiceError):
            await service.chat([{"role": "user", "content": "q"}])
    assert (await wallet(env, owner))["balance"] == 99.94


@pytest.mark.anyio
async def test_langchain_tool_path_meter_uses_usage_metadata(credit_env):
    env = credit_env
    owner = await env.account()
    callback = CreditModelCallback()
    run_id = uuid.uuid4()
    with billing_scope(owner.id, "tool_chat"):
        await callback.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id=run_id,
                                            invocation_params={"max_tokens": 100})
        result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="hi", usage_metadata={
            "input_tokens": 10, "output_tokens": 15, "total_tokens": 25,
        }))]])
        await callback.on_llm_end(result, run_id=run_id)
        await callback.on_llm_end(result, run_id=run_id)
    assert (await wallet(env, owner))["balance"] == 99.975


@pytest.mark.anyio
async def test_langchain_real_callback_wiring_and_stream_usage(credit_env):
    env = credit_env
    owner = await env.account()
    def respond(request):
        return httpx.Response(200, json={
            "id": "synthetic", "model": "fake", "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
        })
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(api_key="synthetic", base_url="http://synthetic.invalid/v1", model="fake",
                           http_async_client=client, callbacks=[CreditModelCallback()], max_tokens=100, max_retries=0)
        with billing_scope(owner.id, "tool_chat"):
            await model.ainvoke([HumanMessage(content="hello")])
    assert (await wallet(env, owner))["balance"] == 99.984


@pytest.mark.anyio
async def test_stream_end_usage_and_partial_disconnect_are_settled(credit_env, monkeypatch):
    env = credit_env
    owner = await env.account()
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "synthetic")
    class Stream:
        def __init__(self):
            self.closed = False
        def __aiter__(self):
            return self.chunks()
        async def chunks(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))], usage=None)
            yield SimpleNamespace(choices=[], usage={"prompt_tokens": 15, "completion_tokens": 5})
        async def close(self):
            self.closed = True
    stream = Stream()
    request = AsyncMock(return_value=stream)
    service = BailianService()
    service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
    with billing_scope(owner.id, "chat"):
        assert "".join([part async for part in service.stream_messages(messages=[{"content": "q"}])]) == "hi"
        assert stream.closed
        assert request.call_args.kwargs["stream_options"] == {"include_usage": True}
        stream.closed = False
        interrupted = service.stream_messages(messages=[{"content": "q"}])
        assert await anext(interrupted) == "hi"
        await interrupted.aclose()
        assert stream.closed
    info = await wallet(env, owner)
    assert info["balance"] == 99.945 and info["reserved"] == 0


@pytest.mark.anyio
async def test_sse_request_keeps_owner_and_returns_credit_error_not_500(credit_env, monkeypatch):
    env = credit_env
    owner = await env.account()
    monkeypatch.setattr(settings, "CREDITS_INITIAL_POINTS", 0)
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "synthetic")
    request = AsyncMock()
    monkeypatch.setattr(bailian_service, "_chat_client",
                        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request))))
    result = await env.client.post("/api/chat/completions", headers=owner.headers,
                                    json={"message": "你好", "stream": True})
    assert result.status_code == 200
    assert "insufficient_credits" in result.text and "[DONE]" in result.text
    request.assert_not_awaited()
    assert (await wallet(env, owner))["reserved"] == 0


@pytest.mark.anyio
async def test_user_chooses_retrieval_without_plan_and_cannot_change_other_user(credit_env):
    env = credit_env
    owner, other = await env.account(), await env.account("bob")
    assert (await env.client.get("/api/auth/me", headers=owner.headers)).json()["knowledge_provider"] == "ima"
    changed = await env.client.put("/api/knowledge/provider", headers=owner.headers, json={"provider": "milvus"})
    assert changed.status_code == 200
    assert (await env.client.get("/api/auth/me", headers=owner.headers)).json()["knowledge_provider"] == "milvus"
    assert (await env.client.get("/api/auth/me", headers=other.headers)).json()["knowledge_provider"] == "ima"
    invalid = await env.client.put("/api/knowledge/provider", headers=owner.headers,
                                   json={"provider": "milvus", "user_id": str(other.id)})
    assert invalid.status_code == 422


def test_usage_reasoning_is_not_double_charged_and_invalid_counts_are_rejected():
    assert credits.provider_usage({"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180,
                                   "completion_tokens_details": {"reasoning_tokens": 50}}) == (100, 80)
    assert credits.provider_usage({"input_tokens": -1}) is None
    assert credits.provider_usage({"total_tokens": True}) is None
