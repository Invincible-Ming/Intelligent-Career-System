"""Knowledge retrieval selected by backend configuration, scoped to a logged-in user."""
from __future__ import annotations

import asyncio
import re
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.models import Document, DocumentParse, User
from app.services.ima_service import ImaUnavailable, ima_service
from app.services.knowledge_policy import knowledge_provider_for


def query_terms(query: str) -> set[str]:
    import jieba

    return {word.casefold() for word in jieba.lcut(query)
            if len(word.strip()) > 1 and re.search(r"\w", word)}


def rank_local_chunks(rows, query: str, top_k: int) -> list[dict]:
    """Bounded keyword matching for local files; never described as vector search."""
    terms = query_terms(query)
    if not terms:
        terms = {query.casefold()}
    matches = []
    for document_id, document_type, filename, chunks, blocks in rows:
        contents = [c.get("content", "") for c in (chunks or [])]
        if not contents:
            contents = [b.get("text", "") for b in (blocks or [])]
        for content in contents[:settings.MAX_CHUNKS_PER_DOCUMENT]:
            folded = content.casefold()
            score = sum(term in folded for term in terms) / len(terms)
            if not score:
                continue
            matches.append({"document_id": str(document_id), "document_type": document_type,
                            "content": content[:settings.IMA_MAX_CONTENT_CHARS],
                            "score": score, "source": "local_keyword",
                            "chunk_metadata": {"title": filename, "score_kind": "keyword_coverage"}})
    return sorted(matches, key=lambda x: x["score"], reverse=True)[:top_k]


class KnowledgeService:
    async def provider(self, session, owner_id):
        if settings.KNOWLEDGE_PROVIDER not in ("subscription", "credits"):
            return settings.KNOWLEDGE_PROVIDER
        user = await session.get(User, owner_id, populate_existing=True)
        if user is None or not user.is_active:
            raise ValueError("账号不存在或已停用")
        return knowledge_provider_for(user)

    async def search(self, *, session, owner_id: uuid.UUID, query: str,
                     document_type: str | None = None, top_k: int = 5, strategy: str = "hybrid") -> list[dict]:
        owner_id = uuid.UUID(str(owner_id))
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("检索问题应为 1～2000 字符")
        top_k = max(1, min(top_k, 20))
        provider = await self.provider(session, owner_id)
        if provider == "ima" and strategy != "hybrid":
            raise ValueError("当前账号使用 ima 检索，请先在知识库页面切换为混合检索")
        if provider == "milvus":
            from app.services.hybrid_search import hybrid_search_service

            allowed_ids = None
            if settings.KNOWLEDGE_PROVIDER in ("subscription", "credits"):
                from app.services.knowledge_index import knowledge_index_service

                allowed_ids = await knowledge_index_service.prepare_search(session, owner_id, document_type)
                if not allowed_ids:
                    return []
            params = {"query": query, "owner_id": str(owner_id), "document_type": document_type,
                      "top_k": top_k, "allowed_document_ids": allowed_ids}
            if strategy == "dense":
                from app.services.bailian import bailian_service
                from app.services.milvus_service import milvus_service

                params.pop("query")
                results = await milvus_service.search(query_vector=await bailian_service.embed_query(query), **params)
            elif strategy == "bm25":
                from app.services.bm25_service import bm25_service

                results = await bm25_service.search(**params)
            elif strategy == "hybrid":
                if settings.KNOWLEDGE_PROVIDER in ("subscription", "credits"):
                    params["enable_reranker"] = True
                results = await hybrid_search_service.search(**params)
            else:
                raise ValueError("不支持的检索策略")
            # Recheck ownership before any content enters an LLM, including chat tools.
            ids = {uuid.UUID(x["document_id"]) for x in results}
            if not ids:
                return []
            ownership = select(Document.id).where(
                Document.id.in_(ids), Document.owner_id == owner_id, Document.status == "ready",
            )
            if document_type is not None:
                ownership = ownership.where(Document.document_type == document_type)
            allowed = {str(x) for x in (await session.execute(ownership)).scalars()}
            return [x for x in results if x["document_id"] in allowed]

        stmt = select(Document.id, Document.document_type, Document.filename,
                      DocumentParse.chunks, DocumentParse.blocks).join(
            DocumentParse, DocumentParse.document_id == Document.id,
        ).where(Document.owner_id == owner_id, Document.status == "ready")
        if document_type is not None:
            stmt = stmt.where(Document.document_type == document_type)
        rows = (await session.execute(stmt.order_by(Document.created_at.desc()).limit(
            settings.MAX_DOCUMENTS_PER_USER))).all()
        local = await asyncio.to_thread(rank_local_chunks, rows, query, top_k)
        remote = []
        if document_type in (None, "knowledge"):
            try:
                remote = await ima_service.search(query=query, owner_id=str(owner_id), top_k=top_k)
            except TimeoutError as exc:
                raise ImaUnavailable("ima 检索超时，请稍后重试") from exc
        # Scores from two providers are not comparable. Interleave in provider rank order.
        results = []
        for index in range(max(len(remote), len(local))):
            if index < len(remote):
                results.append(remote[index])
            if index < len(local):
                results.append(local[index])
        return results[:top_k]


knowledge_service = KnowledgeService()
