"""Enrich retrieved chunks from the PostgreSQL parsing sidecar."""
import hashlib
import uuid
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.core.models import DocumentParse


async def enrich_results(results):
    ids = {uuid.UUID(r["document_id"]) for r in results if r.get("document_id")}
    if not ids:
        return results
    async with AsyncSessionLocal() as session:
        records = (
            await session.execute(select(DocumentParse).where(DocumentParse.document_id.in_(ids)))).scalars().all()
    metadata = {(str(record.document_id), chunk["content_hash"]): chunk
                for record in records for chunk in record.chunks}
    for result in results:
        key = (result["document_id"], hashlib.sha256(result["content"].encode()).hexdigest())
        chunk = metadata.get(key)
        if chunk:
            result["chunk_metadata"] = {k: v for k, v in chunk.items()
                                        if k not in {"content", "parent_text", "content_hash"}}
            parent = chunk["parent_text"]
            # Bounded parent expansion. Never replace the relevant child with a
            # truncated head of an oversized parent.
            result["parent_context"] = parent if len(parent) <= 2400 else result["content"]
    return results
