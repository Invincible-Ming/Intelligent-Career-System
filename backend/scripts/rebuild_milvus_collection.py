"""重建空的 Milvus Collection（带 owner_id 与标量索引）。

适用于：存量文档不需要保留，只想拿到带 owner_id 的新结构。
旧 Collection 会被重命名保留为 _backup_<时间戳>，可回滚。

与migrate_milvus_owner.py 的区别：不做任何重新向量化，零 Token 成本。

用法：
    python scripts/rebuild_milvus_collection.py --yes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.core.config import settings
from app.services.milvus_service import milvus_service

BACKUP_SUFFIX = "_backup_"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="重建带 owner_id 的空 Collection（不保留文档）"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认执行；不加此参数只打印计划",
    )
    args = parser.parse_args()

    collection = settings.MILVUS_COLLECTION
    backup_name = f"{collection}{BACKUP_SUFFIX}{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 直连客户端，不走 initialize()：守卫会因缺 owner_id 而抛错。
    exists = await asyncio.to_thread(
        milvus_service.client.has_collection,
        collection_name=collection,
    )

    print("=" * 60)
    print("重建 Milvus Collection")
    print("=" * 60)
    print(f"当前 Collection : {collection}（存在={exists}）")
    print(f"备份 Collection : {backup_name}")
    print("将创建：带 owner_id 字段 + owner_id_index (INVERTED) 的空 Collection")
    print("文档数据：不会重新向量化（零 Token 成本），原文档仍在 PostgreSQL")

    if not exists:
        print("\n当前 Collection 不存在，直接创建新结构。")

    if not args.yes:
        print("\n这是预演。确认无误后追加 --yes 执行。")
        await milvus_service.close()
        return

    if exists:
        await asyncio.to_thread(
            milvus_service.client.rename_collection,
            old_name=collection,
            new_name=backup_name,
        )
        print(f"\n旧 Collection 已重命名为 {backup_name}")

    await asyncio.to_thread(milvus_service._create_collection)
    print("新 Collection 已创建（含 owner_id 字段与标量索引）")

    await asyncio.to_thread(
        milvus_service.client.load_collection,
        collection_name=collection,
    )

    # 校验：新结构必须真的带上了字段与索引，否则后续启动仍会被守卫拦住。
    fields = await asyncio.to_thread(milvus_service._describe_fields)
    indexes = await asyncio.to_thread(
        milvus_service.client.list_indexes,
        collection_name=collection,
    )
    if "owner_id" not in fields:
        raise RuntimeError("重建后仍缺 owner_id 字段，请检查 _create_collection")
    if "owner_id_index" not in indexes:
        raise RuntimeError("重建后仍缺 owner_id_index 索引")

    print("\n校验通过：")
    print(f"  字段: {sorted(fields)}")
    print(f"  索引: {sorted(indexes)}")

    summary = {
        "collection": collection,
        "backup": backup_name if exists else None,
        "recreated_at": datetime.now().isoformat(),
    }
    directory = BACKEND / ".local"
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / "milvus_rebuild.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    await milvus_service.close()
    print("\n完成。重启后端即可正常启动。")


if __name__ == "__main__":
    asyncio.run(main())
