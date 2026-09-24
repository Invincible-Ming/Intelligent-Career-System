"""Untrusted file parsing through the local, Docker-owning Runner socket."""
from __future__ import annotations

import base64
import asyncio
import json
import uuid

import httpx

from app.core.config import settings
from app.services.document_pipeline import Block
from app.core.limits import check_operation_deadline


async def check_runner() -> bool:
    if not settings.SANDBOX_DOCUMENTS_ENABLED:
        return True
    try:
        transport = httpx.AsyncHTTPTransport(uds=settings.SANDBOX_RUNNER_SOCKET)
        async with httpx.AsyncClient(transport=transport, base_url="http://sandbox", timeout=4) as client:
            response = await client.get("/health")
            return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def parse_document(*, owner_id: uuid.UUID, run_id: uuid.UUID, data: bytes,
                         extension: str, ocr, max_chars: int) -> list[Block]:
    if not settings.SANDBOX_DOCUMENTS_ENABLED:
        raise RuntimeError("文档沙箱未启用")
    transport = httpx.AsyncHTTPTransport(uds=settings.SANDBOX_RUNNER_SOCKET)
    session_id = None
    completed = False
    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox", timeout=None) as client:
        try:
            async with client.stream("POST", "/parse", json={
                "owner_id": str(owner_id), "run_id": str(run_id), "extension": extension,
                "data": base64.b64encode(data).decode("ascii"), "max_chars": max_chars,
            }) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    check_operation_deadline()
                    if not line:
                        continue
                    message = json.loads(line)
                    if message.get("type") == "start":
                        session_id = message["session_id"]
                    elif message.get("type") == "ocr":
                        if not session_id:
                            raise RuntimeError("OCR 会话无效")
                        text = await ocr(message["image"])
                        if len(text) > max_chars:
                            raise ValueError("OCR 结果超出文本限制")
                        reply = await client.post(f"/ocr/{session_id}", json={"text": text})
                        reply.raise_for_status()
                    elif message.get("type") == "result":
                        rows = message.get("blocks")
                        if not isinstance(rows, list) or len(rows) > 10000:
                            raise ValueError("文档结构超出限制")
                        blocks = [Block(**row) for row in rows]
                        if sum(len(block.text) for block in blocks) > max_chars:
                            raise ValueError("文档文本超出限制")
                        completed = True
                        return blocks
                    else:
                        raise RuntimeError(
                            "文档沙箱处理失败：" + json.dumps(message, ensure_ascii=False)[:300]
                        )
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            if exc.response.status_code == 413:
                raise ValueError("文档超过沙箱解析大小上限，请拆分或压缩文件") from exc
            raise RuntimeError(f"文档沙箱处理失败（HTTP {exc.response.status_code}：{body}）") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise RuntimeError("文档沙箱不可用，请启动 Docker 与 Sandbox Runner") from exc
        finally:
            if session_id and not completed:
                async def stop_remote():
                    try:
                        fresh = httpx.AsyncHTTPTransport(uds=settings.SANDBOX_RUNNER_SOCKET)
                        async with httpx.AsyncClient(transport=fresh, base_url="http://sandbox", timeout=3) as cancel_client:
                            await cancel_client.post(f"/cancel/{session_id}")
                    except (httpx.HTTPError, OSError):
                        pass
                stop = asyncio.create_task(stop_remote())
                try:
                    await asyncio.wait_for(asyncio.shield(stop), 4)
                except (TimeoutError, asyncio.CancelledError):
                    pass
    raise RuntimeError("文档沙箱未返回结果")
