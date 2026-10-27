"""Real PostgreSQL entitlements/index queue; all cloud and vector calls are synthetic."""
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
from psycopg import sql
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import api, subscriptions
from app.api.auth_api import router as auth_router
from app.core.config import settings
from app.core.database import PSYCOPG_DATABASE_URL, Base, get_db
from app.core.models import AuthSession, Document, DocumentParse, DocumentSearchIndex, SubscriptionAudit, User
from app.security import auth
from app.services import document_service, knowledge_index
from app.services.document_pipeline import PIPELINE_VERSION, Block
from app.services.hybrid_search import hybrid_search_service
from app.services.ima_service import ima_service
from app.services.knowledge_policy import knowledge_provider_for, subscription_info


@pytest.fixture
async def subscription_env(monkeypatch):
    dbname = "career_subscription_test_" + uuid.uuid4().hex[:12]
    admin = await psycopg.AsyncConnection.connect(PSYCOPG_DATABASE_URL, autocommit=True)
    await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    engine = create_async_engine(make_url(settings.DATABASE_URL).set(database=dbname))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROVIDER", "subscription")
    monkeypatch.setattr(auth, "AsyncSessionLocal", factory)
    monkeypatch.setattr(auth, "consume_limits", AsyncMock())
    monkeypatch.setattr(knowledge_index, "AsyncSessionLocal", factory)
    index = knowledge_index.KnowledgeIndexService()
    monkeypatch.setattr(knowledge_index, "knowledge_index_service", index)
    monkeypatch.setattr(api, "knowledge_index_service", index)
    for name in ("initialize", "delete_document", "insert"):
        monkeypatch.setattr(knowledge_index.milvus_service, name, AsyncMock(return_value=1))
    monkeypatch.setattr(knowledge_index.bailian_service, "embed", AsyncMock(return_value=[[0.1, 0.2]]))
    monkeypatch.setattr(knowledge_index.bm25_service, "replace_owner", AsyncMock())
    app = FastAPI()
    for router in (auth_router, subscriptions.router, api.router):
        app.include_router(router, prefix="/api")

    async def db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = db

    async def account(name, is_admin=False):
        token = secrets.token_urlsafe(32)
        async with factory() as session:
            user = User(username=name, password_hash="unused", is_admin=is_admin, subscription=None)
            session.add(user)
            await session.flush()
            session.add(AuthSession(token_hash=auth.token_hash(token), user_id=user.id,
                                    expires_at=datetime.now(UTC) + timedelta(hours=1)))
            await session.commit()
            return SimpleNamespace(id=user.id, headers={"Authorization": f"Bearer {token}"})

    async def document(owner, content="Python Agent 项目经验"):
        async with factory() as session:
            doc = Document(owner_id=owner.id, filename="resume.txt", document_type="resume",
                           minio_object_key=uuid.uuid4().hex, status="ready", chunk_count=1)
            session.add(doc)
            await session.flush()
            session.add(DocumentParse(document_id=doc.id, pipeline_version=PIPELINE_VERSION, text=content,
                                       chunks=[{"content": content}], blocks=[{"text": content}]))
            await session.commit()
            return doc.id

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        try:
            yield SimpleNamespace(client=client, factory=factory, account=account, document=document, index=index)
        finally:
            await engine.dispose()
            await admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(dbname)))
            await admin.close()


def grant_body(plan="pro"):
    return {"plan": plan, "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat()}


def test_expiration_boundary_and_legacy_modes(monkeypatch):
    now = datetime.now(UTC)
    user = SimpleNamespace(is_admin=True, subscription=SimpleNamespace(plan="pro", expires_at=now))
    assert subscription_info(user, now=now)["plan"] == "free"
    monkeypatch.setattr(settings, "KNOWLEDGE_PROVIDER", "subscription")
    assert knowledge_provider_for(user) == "ima"  # Administration alone does not grant a paid plan.
    user.subscription.expires_at = now + timedelta(days=1)
    assert knowledge_provider_for(user) == "milvus"
    monkeypatch.setattr(settings, "KNOWLEDGE_PROVIDER", "ima")
    assert knowledge_provider_for(user) == "ima"


@pytest.mark.anyio
async def test_only_admin_can_grant_and_expiry_reverts_existing_login(subscription_env):
    env = subscription_env
    admin = await env.account("admin", True)
    user = await env.account("member")
    doc_id = await env.document(user)
    route = f"/api/subscriptions/users/{user.id}"
    assert (await env.client.put(route, headers=user.headers, json=grant_body())).status_code == 403
    assert (await env.client.put(route, json=grant_body())).status_code == 401
    assert (await env.client.put(route, headers=admin.headers,
                                json={**grant_body(), "is_admin": True})).status_code == 422
    response = await env.client.put(route, headers=admin.headers, json=grant_body())
    assert response.status_code == 200, response.text
    me = (await env.client.get("/api/auth/me", headers=user.headers)).json()
    assert me["subscription"]["plan"] == "pro" and me["knowledge_provider"] == "milvus"
    async with env.factory() as session:
        state = await session.get(DocumentSearchIndex, doc_id)
        assert state.status == "pending"
        target = await session.get(User, user.id)
        target.subscription.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()
        assert len((await session.execute(select(SubscriptionAudit))).scalars().all()) == 1
    me = (await env.client.get("/api/auth/me", headers=user.headers)).json()
    assert me["knowledge_provider"] == "ima" and me["subscription"]["expired"]
    assert not await env.index.process_one()
    for endpoint in ("/api/search", "/api/search/bm25"):
        response = await env.client.post(endpoint, headers=user.headers, json={"query": "Python"})
        assert response.status_code == 403
    assert knowledge_index.bailian_service.embed.await_count == 0


@pytest.mark.anyio
async def test_free_queries_keep_owner_scope_and_cannot_select_paid_backend(subscription_env, monkeypatch):
    env = subscription_env
    user, other = await env.account("member"), await env.account("other")
    own_id = await env.document(user)
    await env.document(other, "Python PRIVATE_OTHER_USER")
    remote = AsyncMock(return_value=[{"document_id": "ima:public", "document_type": "knowledge",
                                     "content": "公开 Python 资料", "score": 1, "source": "ima"}])
    hybrid = AsyncMock()
    monkeypatch.setattr(ima_service, "search", remote)
    monkeypatch.setattr(hybrid_search_service, "search", hybrid)
    response = await env.client.post("/api/search/hybrid", headers=user.headers, json={"query": "Python"})
    assert response.status_code == 200, response.text
    results = response.json()
    assert {x["document_id"] for x in results} == {"ima:public", str(own_id)}
    assert "PRIVATE_OTHER_USER" not in response.text
    hybrid.assert_not_awaited()
    response = await env.client.post("/api/search/hybrid", headers=user.headers,
                                     json={"query": "Python", "knowledge_provider": "milvus"})
    assert response.status_code == 422
    response = await env.client.post("/api/search/hybrid", headers=user.headers,
                                     json={"query": "Python", "document_type": "resume"})
    assert response.status_code == 200
    assert remote.await_count == 1  # Private resume queries never go to ima.


@pytest.mark.anyio
async def test_upgrade_indexes_old_files_once_and_filters_paid_results(subscription_env, monkeypatch):
    env = subscription_env
    admin, user, other = await env.account("admin", True), await env.account("member"), await env.account("other")
    own_id, other_id = await env.document(user), await env.document(other)
    await env.client.put(f"/api/subscriptions/users/{user.id}", headers=admin.headers, json=grant_body("starter"))
    pending = await env.client.post("/api/search/hybrid", headers=user.headers, json={"query": "Python"})
    assert pending.status_code == 503 and "准备" in pending.json()["detail"]
    assert await env.index.process_one()
    assert not await env.index.process_one()
    assert knowledge_index.bailian_service.embed.await_count == 1
    calls = knowledge_index.milvus_service.insert.await_args.kwargs
    assert calls["owner_id"] == str(user.id) and calls["document_id"] == str(own_id)
    hybrid = AsyncMock(return_value=[
        {"document_id": str(identifier), "document_type": "resume", "content": "Python", "score": .9,
         "source": "hybrid"} for identifier in (other_id, own_id)
    ])
    monkeypatch.setattr(hybrid_search_service, "search", hybrid)
    remote = AsyncMock()
    monkeypatch.setattr(ima_service, "search", remote)
    response = await env.client.post("/api/search/hybrid", headers=user.headers, json={"query": "Python"})
    assert response.status_code == 200, response.text
    assert [x["document_id"] for x in response.json()] == [str(own_id)]
    assert hybrid.await_args.kwargs["enable_reranker"] is True
    assert hybrid.await_args.kwargs["allowed_document_ids"] == [str(own_id)]
    remote.assert_not_awaited()
    # Another process can reconstruct BM25 without loading or seeing another user's private corpus.
    second_process = knowledge_index.KnowledgeIndexService()
    async with env.factory() as session:
        assert await second_process.prepare_search(session, user.id) == [str(own_id)]
    assert knowledge_index.bm25_service.replace_owner.await_args.args[0] == str(user.id)
    status = (await env.client.get("/api/knowledge/status", headers=user.headers)).json()
    assert status["index"] == {"total": 1, "ready": 1, "pending": 0, "failed": 0}
    await env.client.put(f"/api/subscriptions/users/{user.id}", headers=admin.headers,
                         json={"plan": "free", "expires_at": None})
    assert (await env.client.get("/api/auth/me", headers=user.headers)).json()["knowledge_provider"] == "ima"


@pytest.mark.anyio
async def test_failed_index_is_not_searchable_and_owner_can_retry(subscription_env, monkeypatch):
    env = subscription_env
    admin, user, other = await env.account("admin", True), await env.account("member"), await env.account("other")
    own_id = await env.document(user)
    await env.client.put(f"/api/subscriptions/users/{user.id}", headers=admin.headers, json=grant_body())
    monkeypatch.setattr(knowledge_index.bailian_service, "embed", AsyncMock(side_effect=RuntimeError("PRIVATE_SECRET")))
    assert await env.index.process_one()
    status = await env.client.get("/api/knowledge/status", headers=user.headers)
    assert status.json()["index"]["failed"] == 1 and "PRIVATE_SECRET" not in status.text
    response = await env.client.post("/api/search/hybrid", headers=user.headers, json={"query": "Python"})
    assert response.status_code == 503 and "PRIVATE_SECRET" not in response.text
    assert (await env.client.post("/api/knowledge/index/retry", headers=other.headers)).status_code == 403
    assert (await env.client.post("/api/knowledge/index/retry", headers=user.headers)).json()["queued"] == 1
    async with env.factory() as session:
        assert (await session.get(DocumentSearchIndex, own_id)).status == "pending"


@pytest.mark.anyio
async def test_concurrent_workers_skip_claimed_job_and_cancellation_is_recoverable(subscription_env, monkeypatch):
    env = subscription_env
    admin, user = await env.account("admin", True), await env.account("member")
    own_id = await env.document(user)
    await env.client.put(f"/api/subscriptions/users/{user.id}", headers=admin.headers, json=grant_body())
    entered = asyncio.Event()

    async def waiting_embed(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(knowledge_index.bailian_service, "embed", waiting_embed)
    task = asyncio.create_task(env.index.process_one())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not await knowledge_index.KnowledgeIndexService().process_one()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with env.factory() as session:
        assert (await session.get(DocumentSearchIndex, own_id)).status == "pending"
    monkeypatch.setattr(knowledge_index.bailian_service, "embed", AsyncMock(return_value=[[.1, .2]]))
    assert await env.index.process_one()


@pytest.mark.anyio
async def test_free_upload_then_upgrade_reuses_parse_and_never_uploads_to_ima(subscription_env, monkeypatch):
    env = subscription_env
    admin, user = await env.account("admin", True), await env.account("member")
    parse = AsyncMock(return_value=[Block(text="Python Agent 项目经验")])
    monkeypatch.setattr(document_service, "parse_document", parse)
    monkeypatch.setattr(document_service.minio_service, "upload", AsyncMock(return_value="synthetic-upload"))
    remote = AsyncMock()
    monkeypatch.setattr(ima_service, "_post", remote)
    response = await env.client.post("/api/documents/upload", headers=user.headers,
        data={"document_type": "resume"}, files={"file": ("resume.txt", b"Python Agent", "text/plain")})
    assert response.status_code == 200, response.text
    document_id = uuid.UUID(response.json()["document"]["id"])
    async with env.factory() as session:
        assert await session.get(DocumentSearchIndex, document_id) is None
    assert knowledge_index.bailian_service.embed.await_count == 0
    await env.client.put(f"/api/subscriptions/users/{user.id}", headers=admin.headers, json=grant_body())
    async with env.factory() as session:
        assert (await session.get(DocumentSearchIndex, document_id)).status == "pending"
    assert parse.await_count == 1
    remote.assert_not_awaited()
