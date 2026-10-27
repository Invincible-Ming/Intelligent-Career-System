"""Read-only smoke check using the backend's configured ima library and credentials."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.core.config import settings
from app.services.ima_service import ImaUnavailable, ima_service


async def check(query: str) -> int:
    try:
        if settings.KNOWLEDGE_PROVIDER not in ("ima", "subscription"):
            print("当前知识库模式不是 ima，请检查 backend/.env")
            return 1
        info = await ima_service.library_info()
        page = await ima_service.list_items()
        print(f"已连接 ima 知识库：{info['name']}")
        print(f"首批条目数量：{len(page['items'])}")
        if query:
            # Synthetic identifier is for parse-task audit tags; this trusted CLI never reads local private documents.
            results = await ima_service.search(query=query, owner_id=str(uuid.uuid4()), top_k=3)
            print(f"可用于回答的正文片段：{len(results)}")
            if not results:
                print("请确认该库已添加相关资料且原文可访问；条目名称不作为正文。")
        return 0
    except ImaUnavailable as exc:
        print(f"ima 检查失败：{exc}")
        return 1
    except TimeoutError:
        print("ima 检查超时：请检查网络连接后重试。")
        return 1
    finally:
        await ima_service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="", help="可选，只读检索关键词")
    raise SystemExit(asyncio.run(check(parser.parse_args().query)))
