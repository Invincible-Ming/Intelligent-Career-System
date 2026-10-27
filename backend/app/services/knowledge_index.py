"""PostgreSQL-backed subscription indexing queue with bounded, restartable workers."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime

import anyio
from sqlalchemy import exists, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.limits import BudgetExceeded, InvocationBudget, invocation_budget, operation_actor_is_admin
from app.core.models import (
    Document,
    DocumentParse,
    DocumentSearchIndex,
    User,
    UserKnowledgePreference,
    UserSubscription,
)
from app.services.bailian import bailian_service
from app.services.bm25_service import bm25_service
from app.services.credits import InsufficientCredits, billing_scope
from app.services.knowledge_policy import PAID_PLANS
from app.services.milvus_service import milvus_service

logger = logging.getLogger(__name__)


class KnowledgeIndexUnavailable(RuntimeError):
    """Actionable index state without model errors or document contents."""


def eligible_account():
    if settings.KNOWLEDGE_PROVIDER == "credits":
        selected = exists(select(UserKnowledgePreference.user_id).where(
            UserKnowledgePreference.user_id == User.id, UserKnowledgePreference.provider == "milvus",
        ))
        return User.is_active.is_(True) & selected
    subscribed = exists(select(UserSubscription.user_id).where(
        UserSubscription.user_id == User.id, UserSubscription.plan.in_(PAID_PLANS),
        UserSubscription.expires_at > func.now(),
    ))
    return User.is_active.is_(True) & subscribed


class KnowledgeIndexService:
    def __init__(self):
        self._bm25_versions = {}
        self._bm25_lock = asyncio.Lock()

    def signature(self):
        # Immutable document IDs identify the persisted chunks. Model/collection changes invalidate indexes.
        config = ["subscription-index-v1", settings.BAILIAN_BASE_URL, settings.BAILIAN_EMBEDDING_MODEL,
                  settings.EMBEDDING_DIMENSION, milvus_service.collection]
        return hashlib.sha256(json.dumps(config).encode()).hexdigest()

    async def enqueue_owner(self, session, owner_id, *, retry_failed=False):
        if settings.KNOWLEDGE_PROVIDER not in ("subscription", "credits"):
            return 0
        signature = self.signature()
        conditions = [DocumentSearchIndex.document_id.is_(None), DocumentSearchIndex.signature != signature]
        if retry_failed:
            conditions.append(DocumentSearchIndex.status == "failed")
        ids = (await session.execute(select(Document.id).outerjoin(
            DocumentSearchIndex, DocumentSearchIndex.document_id == Document.id,
        ).where(Document.owner_id == owner_id, Document.status == "ready", or_(*conditions)))).scalars().all()
        if ids:
            stmt = insert(DocumentSearchIndex).values([
                {"document_id": identifier, "signature": signature, "status": "pending"} for identifier in ids
            ])
            await session.execute(stmt.on_conflict_do_update(
                index_elements=[DocumentSearchIndex.document_id],
                set_={"signature": signature, "status": "pending", "error_message": None,
                      "updated_at": datetime.now(UTC)},
            ))
        return len(ids)

    async def enqueue_active(self):
        async with AsyncSessionLocal() as session:
            owners = (await session.execute(select(User.id).where(eligible_account()))).scalars().all()
            for owner in owners:
                await self.enqueue_owner(session, owner)
            await session.commit()

    async def status(self, session, owner_id, document_type=None):
        stmt = select(DocumentSearchIndex.status, DocumentSearchIndex.signature).select_from(Document).outerjoin(
            DocumentSearchIndex, DocumentSearchIndex.document_id == Document.id,
        ).where(Document.owner_id == owner_id, Document.status == "ready")
        if document_type is not None:
            stmt = stmt.where(Document.document_type == document_type)
        rows = (await session.execute(stmt)).all()
        counts = {"total": len(rows), "ready": 0, "pending": 0, "failed": 0}
        for status, signature in rows:
            key = status if signature == self.signature() and status in ("ready", "failed") else "pending"
            counts[key] += 1
        return counts

    async def prepare_search(self, session, owner_id, document_type=None):
        status = await self.status(session, owner_id, document_type)
        if status["pending"]:
            raise KnowledgeIndexUnavailable("正在准备文件检索，请在知识库页面查看进度，完成后重试。")
        if status["failed"]:
            raise KnowledgeIndexUnavailable("部分文件索引建立失败，请在知识库页面重试，或联系管理员检查模型额度与检索服务。")
        rows = (await session.execute(select(Document, DocumentParse, DocumentSearchIndex).join(
            DocumentParse, DocumentParse.document_id == Document.id,
        ).join(DocumentSearchIndex, DocumentSearchIndex.document_id == Document.id).where(
            Document.owner_id == owner_id, Document.status == "ready", DocumentSearchIndex.status == "ready",
            DocumentSearchIndex.signature == self.signature(),
        ))).all()
        # Every backend process materializes its own BM25 snapshot from authorized PostgreSQL rows.
        version = tuple(sorted((str(doc.id), state.updated_at.isoformat()) for doc, _, state in rows))
        async with self._bm25_lock:
            if self._bm25_versions.get(str(owner_id)) != version:
                await bm25_service.replace_owner(str(owner_id), [
                    {"document_id": str(doc.id), "document_type": doc.document_type,
                     "chunks": [chunk["content"] for chunk in parsed.chunks]} for doc, parsed, _ in rows
                ])
                self._bm25_versions[str(owner_id)] = version
        return [str(doc.id) for doc, _, _ in rows if document_type is None or doc.document_type == document_type]

    async def process_one(self):
        async with AsyncSessionLocal() as session:
            # Locks are held through the bounded job. A crash rolls back to pending; another worker skips it meanwhile.
            row = (await session.execute(select(Document, DocumentParse, DocumentSearchIndex).join(
                DocumentParse, DocumentParse.document_id == Document.id,
            ).join(DocumentSearchIndex, DocumentSearchIndex.document_id == Document.id).join(
                User, User.id == Document.owner_id,
            ).where(eligible_account(), Document.status == "ready", DocumentSearchIndex.status == "pending",
                    DocumentSearchIndex.signature == self.signature()).order_by(Document.created_at).limit(1)
                .with_for_update(of=(DocumentSearchIndex, Document), skip_locked=True))).first()
            if row is None:
                return False
            document, parsed, state = row
            token = invocation_budget.set(InvocationBudget(
                time.monotonic() + settings.KNOWLEDGE_INDEX_TIMEOUT, settings.KNOWLEDGE_INDEX_MAX_CALLS,
            ))
            actor_token = operation_actor_is_admin.set(False)
            try:
                with billing_scope(document.owner_id, "knowledge_index"), anyio.fail_after(settings.KNOWLEDGE_INDEX_TIMEOUT):
                    await milvus_service.initialize()
                    chunks = [chunk["content"] for chunk in parsed.chunks]
                    if not chunks or not all(isinstance(chunk, str) and chunk.strip() for chunk in chunks):
                        raise ValueError("missing chunks")
                    vectors = []
                    for start in range(0, len(chunks), 10):
                        vectors.extend(await bailian_service.embed(chunks[start:start + 10]))
                    # Retried/crashed jobs replace only this document, so duplicate vectors cannot accumulate.
                    await milvus_service.delete_document(str(document.id))
                    await milvus_service.insert(document_id=str(document.id), owner_id=str(document.owner_id),
                                                document_type=document.document_type, chunks=chunks, vectors=vectors)
                    state.status = "ready"
                    state.error_message = None
            except (Exception, BudgetExceeded, InsufficientCredits) as exc:
                state.status = "failed"
                state.error_message = (str(exc) if isinstance(exc, InsufficientCredits) else
                                       "文件索引建立失败，请检查 Embedding 额度和向量服务后重试。")
                logger.warning("文件索引任务失败（%s），document_id=%s", type(exc).__name__, document.id)
            finally:
                invocation_budget.reset(token)
                operation_actor_is_admin.reset(actor_token)
            state.updated_at = datetime.now(UTC)
            await session.commit()
            return True

    async def run(self):
        while True:
            try:
                await self.enqueue_active()
                if await self.process_one():
                    continue
            except Exception as exc:
                logger.warning("文件索引队列暂不可用（%s），稍后重试", type(exc).__name__)
            await asyncio.sleep(5)


knowledge_index_service = KnowledgeIndexService()
