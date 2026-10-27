"""Rebuild local document vectors in a NEW collection before switching back from ima.

Default: read-only preflight. --execute invokes Embedding and writes a new collection;
existing collections are never renamed, overwritten or removed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal, close_database
from app.core.models import Document, DocumentParse
from app.services.bailian import bailian_service
from app.services.milvus_service import MilvusService


def stored_chunks(document, parsed) -> list[str]:
    if parsed is None or not parsed.chunks:
        raise ValueError(f"文档 {document.id} 缺少分块记录，停止重建，请先补齐解析数据")
    chunks = [item["content"] for item in parsed.chunks]
    if not all(isinstance(text, str) and text.strip() and len(text.encode()) <= 8192 for text in chunks):
        raise ValueError(f"文档 {document.id} 存在无效分块，停止重建")
    return chunks


async def rebuild(*, execute: bool) -> None:
    target = MilvusService()
    target.collection = f"{settings.MILVUS_COLLECTION}_restore_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    try:
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(select(Document, DocumentParse).outerjoin(
                DocumentParse, DocumentParse.document_id == Document.id,
            ).where(Document.status == "ready").order_by(Document.id))).all()
        records = [(doc, stored_chunks(doc, parsed)) for doc, parsed in rows]
        print(f"待重建：{len(records)} 份文档，{sum(len(chunks) for _, chunks in records)} 个分块")
        if not execute:
            print("预检完成，未调用模型或修改向量库。追加 --execute 才会生成新索引。")
            return
        # The timestamped target must be new; do not reuse a potentially partial prior run.
        if await asyncio.to_thread(target.client.has_collection, collection_name=target.collection):
            raise RuntimeError("目标索引已存在，停止以免覆盖")
        await target.initialize()
        for index, (doc, chunks) in enumerate(records, 1):
            vectors = []
            for start in range(0, len(chunks), 10):
                vectors.extend(await bailian_service.embed(chunks[start:start + 10]))
            await target.insert(document_id=str(doc.id), owner_id=str(doc.owner_id),
                                document_type=doc.document_type, chunks=chunks, vectors=vectors)
            print(f"已完成 {index}/{len(records)} 份文档")
        print("新索引构建完成。停止后端，在 .env 中设置以下两项，再启动后端：")
        print("KNOWLEDGE_PROVIDER=milvus")
        print(f"MILVUS_COLLECTION={target.collection}")
    finally:
        await target.close()
        await bailian_service.close()
        await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="调用模型并构建新索引；保留所有原有索引")
    asyncio.run(rebuild(execute=parser.parse_args().execute))
