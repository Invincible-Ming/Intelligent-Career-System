"""Milvus owner_id 迁移：重建 Collection 并按 PostgreSQL 归属回填。

背景：早期 Collection 未保存 owner_id，数据隔离依赖应用层拼
`document_id in [...]` 过滤表达式。该过滤会随文档量线性劣化，
且无法使用 Milvus 标量索引。

本脚本做两件事：
1. 备份旧 Collection（重命名为 <collection>_backup_<时间戳>）；
2. 创建带 owner_id 字段与 INVERTED 标量索引的新 Collection，
   并按 PostgreSQL documents.owner_id 重新写入向量。

原始文本始终以 PostgreSQL 的 documents / document_parse 为准，
因此重新向量化不会丢内容。

用法：
    # 仅预检，不写任何数据
    python scripts/migrate_milvus_owner.py --dry-run

    # 正式执行（会重新调用 Embedding 模型，产生 Token 消耗）
    python scripts/migrate_milvus_owner.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal, close_database, engine, init_database
from app.core.models import Document, DocumentParse
from app.services.bailian import bailian_service
from app.services.milvus_service import (
    milvus_service,
    validate_document_id,
    validate_document_type,
)

BACKUP_SUFFIX = "_backup_"


async def collect_documents() -> tuple[list[dict], list[str]]:
    """读取可迁移文档及其解析文本。

    以 PostgreSQL 为唯一事实来源：只迁移 status=ready 且有解析结果的文档。
    """

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(Document, DocumentParse).join(
                DocumentParse,
                DocumentParse.document_id == Document.id,
                isouter=True,
            ).where(Document.status == "ready").order_by(Document.created_at)
        )).all()

    documents: list[dict] = []
    missing_text: list[str] = []

    for document, parsed in rows:
        if parsed is None or not (parsed.text or "").strip():
            missing_text.append(str(document.id))
            continue

        documents.append({
            "document_id": str(document.id),
            "owner_id": str(document.owner_id),
            "document_type": document.document_type,
            "text": parsed.text,
        })

    return documents, missing_text


async def embed_all(documents: list[dict], *, dry_run: bool) -> None:
    """按文档重新切块并向量化写入新 Collection。"""

    from app.services.document_pipeline import build_chunks, Block

    total_chunks = 0
    started = time.monotonic()

    for index, document in enumerate(documents, start=1):
        blocks = [Block(part.strip()) for part in document["text"].split("\n\n") if part.strip()]
        structured = build_chunks(blocks, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        chunks = [chunk["content"] for chunk in structured]

        if not chunks:
            print(f"  [{index}/{len(documents)}] {document['document_id']} 切块为空，跳过")
            continue

        if len(chunks) > settings.MAX_CHUNKS_PER_DOCUMENT:
            print(
                f"  [{index}/{len(documents)}] {document['document_id']} "
                f"分块 {len(chunks)} 个超过上限，跳过"
            )
            continue

        if dry_run:
            total_chunks += len(chunks)
            print(f"  [{index}/{len(documents)}] {document['document_id']} 预计写入 {len(chunks)} 个 Chunk")
            continue

        vectors = await embed_chunks(chunks)
        await milvus_service.upsert_chunks(
            document_id=document["document_id"],
            owner_id=document["owner_id"],
            document_type=document["document_type"],
            chunks=chunks,
            vectors=vectors,
        )
        total_chunks += len(chunks)
        print(f"  [{index}/{len(documents)}] {document['document_id']} 已写入 {len(chunks)} 个 Chunk")

    elapsed = time.monotonic() - started
    print(f"\n合计 {total_chunks} 个 Chunk，耗时 {elapsed:.1f}s")


async def embed_chunks(chunks: list[str], batch_size: int = 10) -> list[list[float]]:
    """分批调用 Embedding，迁移期间不经过任何预算/限流上下文。"""

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        vectors.extend(await bailian_service.embed(chunks[start: start + batch_size]))
    return vectors


async def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus owner_id 迁移")
    parser.add_argument("--dry-run", action="store_true", help="仅预检，不做任何写入")
    args = parser.parse_args()

    await init_database()

    collection = settings.MILVUS_COLLECTION
    backup_name = f"{collection}{BACKUP_SUFFIX}{datetime.now().strftime('%Y%m%d%H%M%S')}"

    print("=" * 60)
    print("Milvus owner_id 迁移")
    print("=" * 60)
    print(f"目标 Collection : {collection}")
    print(f"备份 Collection : {backup_name}")
    print(f"模式            : {'预检（不写入）' if args.dry_run else '正式执行'}")

    # 不能调用 milvus_service.initialize()：它的守卫会在缺少 owner_id 时
    # 抛错——而这恰恰是本脚本要处理的场景。这里直连 Milvus 客户端。
    await asyncio.to_thread(
        milvus_service.client.load_collection,
        collection_name=collection,
    )

    fields = await asyncio.to_thread(milvus_service._describe_fields)
    needs_migration = "owner_id" not in fields

    if not needs_migration:
        print("\n当前 Collection 已包含 owner_id 字段，无需迁移。")
        await finish(collection=None, documents=0, chunks=0, dry_run=args.dry_run)
        return

    documents, missing_text = await collect_documents()

    print(f"\n待迁移文档 {len(documents)} 个")
    if missing_text:
        print(f"以下文档无解析文本，将保留在旧 Collection（不影响隔离）：{len(missing_text)} 个")

    if args.dry_run:
        print("\n--- 预检模式，不做任何写入 ---")
        await embed_all(documents, dry_run=True)
        print("\n确认无误后去掉 --dry-run 正式执行。")
        return

    if not documents:
        print("没有可迁移的文档；可直接重建 Collection。")
        await rebuild_collection(collection, backup_name)
        await finish(collection, 0, 0, dry_run=False)
        return

    # 1) 备份旧 Collection：重命名而非删除，出问题可改回配置名。
    await asyncio.to_thread(
        milvus_service.client.rename_collection,
        old_name=collection,
        new_name=backup_name,
    )
    print(f"\n旧 Collection 已重命名为 {backup_name}")

    # 2) 新建带 owner_id 的 Collection。
    await asyncio.to_thread(milvus_service._create_collection)
    print("新 Collection 已创建（含 owner_id 标量索引）")

    # 3) 回填向量。
    try:
        await embed_all(documents, dry_run=False)
    except BaseException as exc:
        print(f"\n回填失败：{type(exc).__name__}: {exc}")
        print(f"旧数据仍完整保留在 {backup_name}，可把 MILVUS_COLLECTION 配置改回该名称回滚。")
        raise

    # 4) BM25 索引在下次启动时自动重建，无需在此处理。
    remaining = await milvus_service.list_chunks()
    print(f"\n校验：新 Collection 当前 {len(remaining)} 个 Chunk")

    await finish(collection, len(documents), len(remaining), dry_run=False)


async def rebuild_collection(collection: str, backup_name: str) -> None:
    """无文档可迁移时，仅重建结构。"""

    await asyncio.to_thread(
        milvus_service.client.rename_collection,
        old_name=collection,
        new_name=backup_name,
    )
    await asyncio.to_thread(milvus_service._create_collection)


async def finish(collection: str | None, documents: int, chunks: int, *, dry_run: bool) -> None:
    summary = {
        "collection": collection,
        "documents": documents,
        "chunks": chunks,
        "dry_run": dry_run,
        "completed_at": datetime.now().isoformat(),
    }
    directory = BACKEND / ".local"
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / "milvus_owner_migration.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    await milvus_service.close()
    await close_database()
    print("\n完成。")


if __name__ == "__main__":
    asyncio.run(main())
