"""清理Milvus 备份 Collection。

只删由 rebuild_milvus_collection.py / migrate_milvus_owner.py 生成的备份
（名字形如 <collection>_backup_<时间戳>），绝不碰当前业务 Collection。

删除前会打印每个备份的 chunk 数量与owner_id 字段情况，
必须显式 --yes 才真正删除。

用法：
    python scripts/cleanup_milvus_backups.py          # 预演
    python scripts/cleanup_milvus_backups.py --yes    # 确认删除
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.core.config import settings
from app.services.milvus_service import milvus_service

BACKUP_PATTERN = re.compile(r"^(?P<base>.+)_backup_\d{8}\d{6}$")


def is_backup(name: str) -> bool:
    """只认自家生成的备份，命名不匹配的一律不碰。"""

    return BACKUP_PATTERN.match(name) is not None


async def inspect(name: str) -> dict:
    """读取备份collection 的规模信息。"""

    # 不能复用 milvus_service._describe_fields()：它硬编码了配置里的
    # 当前 collection 名，这里要查的是任意备份。
    info = await asyncio.to_thread(
        milvus_service.client.describe_collection,
        collection_name=name,
    )
    fields = {str(field.get("name")) for field in info.get("fields", [])}

    # 统计总量（limit=1 只用于确认可读，真实数量用统计接口）。
    try:
        stats = await asyncio.to_thread(
            milvus_service.client.get_collection_stats,
            collection_name=name,
        )
        row_count = int(stats.get("row_count", 0))
    except Exception:
        row_count = -1

    return {
        "name": name,
        "has_owner_id": "owner_id" in fields,
        "row_count": row_count,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="清理 Milvus 备份 Collection")
    parser.add_argument("--yes", action="store_true", help="确认删除；缺省只打印计划")
    args = parser.parse_args()

    client = milvus_service.client
    names = await asyncio.to_thread(client.list_collections)
    current = settings.MILVUS_COLLECTION

    backups = sorted(name for name in names if is_backup(name))
    others = sorted(name for name in names if not is_backup(name))

    print("=" * 60)
    print("Milvus备份清理")
    print("=" * 60)
    print(f"当前业务 Collection: {current}")
    print(f"发现备份 {len(backups)} 个")
    print()

    for name in backups:
        info = await inspect(name)
        print(f"  {name}")
        print(f"    带 owner_id 字段: {info['has_owner_id']}")
        print(f"    Chunk 数量      : {info['row_count']}")

    if others:
        print()
        print("以下不是备份collection，将保持不动:")
        for name in others:
            marker = " (当前业务)" if name == current else ""
            print(f"  {name}{marker}")

    if not backups:
        print("\n没有可清理的备份。")
        await milvus_service.close()
        return

    if not args.yes:
        print("\n这是预演。确认删除以上备份请追加 --yes。")
        await milvus_service.close()
        return

    print()
    for name in backups:
        if name == current:  # 二次保险：绝不删当前业务 collection
            print(f"  跳过 {name}（与当前业务同名）")
            continue
        await asyncio.to_thread(client.drop_collection, collection_name=name)
        print(f"  已删除 {name}")

    remaining = await asyncio.to_thread(client.list_collections)
    print(f"\n清理完成。当前 collections: {sorted(remaining)}")

    await milvus_service.close()


if __name__ == "__main__":
    asyncio.run(main())