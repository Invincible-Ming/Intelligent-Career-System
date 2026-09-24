"""Trusted local Docker controller. Serve only on a permission-restricted Unix socket."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.mcp_container_config import DOCKER_ENV, docker_connection
from deploy.sandbox.audit import (mcp_cancel, mcp_result, mcp_start,
                                  ocr_request, parse_cancel, parse_error,
                                  parse_start, parse_result)

IMAGE = "career-document-sandbox:1"
# 文档大小上限（MB），默认与后端 .env 的 MAX_UPLOAD_SIZE_MB 对齐。
# 可用环境变量 SANDBOX_MAX_DOCUMENT_MB 覆盖（在启动 Runner 前设置）。
MAX_BYTES = int(os.environ.get("SANDBOX_MAX_DOCUMENT_MB", "50")) * 1024 * 1024
MAX_MCP_SESSION_SECONDS = 120
pending: dict[str, asyncio.Future] = {}
active: dict[str, tuple[str, asyncio.subprocess.Process]] = {}
session_meta: dict[str, dict[str, str]] = {}
cleanup_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app):
    # Only this runner's disposable containers are removed after an unclean restart.
    proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-aq", "--filter", "label=career.sandbox=document",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=DOCKER_ENV,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        for container_id in stdout.decode().splitlines():
            await cleanup(container_id)
    proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-aq", "--filter", "label=career.mcp.runner=true",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=DOCKER_ENV,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        for container_id in stdout.decode().splitlines():
            await cleanup(container_id)
    yield


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.get("/health")
async def health():
    proc = await asyncio.create_subprocess_exec(
        "docker", "image", "inspect", IMAGE,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, env=DOCKER_ENV,
    )
    try:
        await asyncio.wait_for(proc.wait(), 3)
    except TimeoutError:
        proc.kill()
        await proc.wait()
    if proc.returncode:
        raise HTTPException(503, "Sandbox image unavailable")
    return {"status": "ok"}


class ParseRequest(BaseModel):
    owner_id: uuid.UUID
    run_id: uuid.UUID
    extension: str
    data: str
    max_chars: int = Field(gt=0, le=200000)


class OCRReply(BaseModel):
    text: str = Field(max_length=200000)


async def cleanup(container_name):
    proc = await asyncio.create_subprocess_exec(
        "docker", "rm", "-f", container_name,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, env=DOCKER_ENV,
    )
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except TimeoutError:
        proc.kill()
        await proc.wait()


@app.post("/parse")
async def parse(request: ParseRequest):
    if request.extension not in (".pdf", ".docx", ".xlsx", ".txt"):
        raise HTTPException(422, "Unsupported extension")
    if len(request.data) > (MAX_BYTES * 4 // 3 + 8):
        raise HTTPException(413, "Document too large")
    try:
        raw = base64.b64decode(request.data, validate=True)
    except ValueError:
        raise HTTPException(422, "Invalid document") from None
    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(413, "Document too large")
    owner_hash = hashlib.sha256(request.owner_id.bytes).hexdigest()[:24]
    name = f"career-sbx-{uuid.uuid4().hex}"
    session_id = uuid.uuid4().hex
    start_time = time.monotonic()
    parse_start(session_id, str(request.run_id), owner_hash, request.extension)
    session_meta[session_id] = {"run_id": str(request.run_id),
                                "owner_hash": owner_hash,
                                "extension": request.extension}

    async def events():
        process = None
        try:
            args = [
                "docker", "run", "--rm", "-i", "--pull=never", "--name", name,
                "--label", "career.sandbox=document", "--label", f"career.owner={owner_hash}",
                "--label", f"career.run={request.run_id}", "--network=none", "--read-only",
                "--user=10001:10001", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                "--memory=256m", "--cpus=1", "--pids-limit=64", "--tmpfs=/tmp:rw,noexec,nosuid,size=32m",
                "--log-driver=none", IMAGE,
            ]
            process = await asyncio.create_subprocess_exec(
                *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, limit=32 * 1024 * 1024, env=DOCKER_ENV,
            )
            active[session_id] = (name, process)
            process.stdin.write(json.dumps({
                "extension": request.extension, "data": request.data,
                "max_chars": request.max_chars,
            }, separators=(",", ":")).encode() + b"\n")
            await process.stdin.drain()
            yield json.dumps({"type": "start", "session_id": session_id}) + "\n"
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), 110)
                if not line or len(line) > 32 * 1024 * 1024:
                    raise RuntimeError("Sandbox exited without result")
                message = json.loads(line)
                if message.get("type") == "ocr":
                    future = asyncio.get_running_loop().create_future()
                    pending[session_id] = future
                    img = message.get("image", "")
                    ocr_request(session_id, hashlib.sha256(img.encode()).hexdigest()[:16])
                    yield json.dumps({"type": "ocr", "image": message["image"]}) + "\n"
                    reply = await asyncio.wait_for(future, 90)
                    pending.pop(session_id, None)
                    process.stdin.write(json.dumps({"type": "ocr_result", "text": reply},
                                                   ensure_ascii=False).encode() + b"\n")
                    await process.stdin.drain()
                elif message.get("type") == "result":
                    yield json.dumps(message, ensure_ascii=False) + "\n"
                    duration = int((time.monotonic() - start_time) * 1000)
                    parse_result(session_id, str(request.run_id), owner_hash,
                                 request.extension, 0, "success", duration)
                    break
                else:
                    raise RuntimeError("Sandbox rejected document")
        except asyncio.CancelledError:
            duration = int((time.monotonic() - start_time) * 1000)
            parse_cancel(session_id, str(request.run_id), owner_hash,
                         "client_disconnected")
            raise
        except Exception as exc:
            duration = int((time.monotonic() - start_time) * 1000)
            parse_error(session_id, str(request.run_id), owner_hash,
                        request.extension, type(exc).__name__, duration)
            yield json.dumps({"type": "error", "message": "文档沙箱处理失败"}) + "\n"
        finally:
            pending.pop(session_id, None)
            session_meta.pop(session_id, None)
            if process is not None:
                async def finish():
                    try:
                        await cleanup(name)
                        if process.returncode is None:
                            process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), 3)
                        except TimeoutError:
                            process.kill()
                            await process.wait()
                    finally:
                        active.pop(session_id, None)

                task = asyncio.create_task(finish())
                cleanup_tasks.add(task)
                task.add_done_callback(cleanup_tasks.discard)
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass  # Detached cleanup continues after HTTP disconnect.

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/cancel/{session_id}")
async def cancel(session_id: str):
    running = active.get(session_id)
    if running is None:
        return {"ok": True}
    future = pending.pop(session_id, None)
    if future is not None and not future.done():
        future.cancel()
    meta = session_meta.pop(session_id, {})
    parse_cancel(session_id, meta.get("run_id", ""), meta.get("owner_hash", ""),
                 "api_cancel")
    await cleanup(running[0])
    return {"ok": True}


@app.post("/ocr/{session_id}")
async def ocr_result(session_id: str, reply: OCRReply):
    future = pending.get(session_id)
    if future is None or future.done():
        raise HTTPException(404, "OCR request expired")
    future.set_result(reply.text)
    return {"ok": True}


@app.websocket("/mcp")
async def mcp(websocket: WebSocket):
    await websocket.accept()
    process = None
    tasks: list[asyncio.Task] = []
    mode = "unknown"
    name = f"career-mcp-run-{uuid.uuid4().hex}"
    session_id = uuid.uuid4().hex
    start_time = time.monotonic()
    try:
        handshake = json.loads(await asyncio.wait_for(websocket.receive_text(), 5))
        mode = handshake.get("mode")
        owner_id, run_id = handshake.get("owner_id"), handshake.get("run_id")
        if set(handshake) != {"mode", "owner_id", "run_id"}:
            raise ValueError("Invalid MCP handshake")
        owner_hash = hashlib.sha256(uuid.UUID(str(owner_id)).bytes).hexdigest()[:24] if owner_id else None
        if mode not in ("search", "filesystem", "postgres"):
            raise ValueError("Invalid MCP scope")
        if (owner_id or run_id) and mode != "search":
            raise ValueError("Only search may be bound to a run")
        mcp_start(session_id, mode, owner_hash, run_id)
        connection = docker_connection(mode, owner_id=owner_id, run_id=run_id)
        args = list(connection["args"])
        args[-5:-5] = ["--name", name, "--label=career.mcp.runner=true", "--log-driver=none"]
        process = await asyncio.create_subprocess_exec(
            connection["command"], *args, env=connection["env"],
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=16 * 1024 * 1024,
        )

        async def input_to_container():
            while True:
                frame = await websocket.receive_bytes()
                if len(frame) > 16 * 1024 * 1024:
                    raise ValueError("MCP message too large")
                process.stdin.write(frame)
                await process.stdin.drain()

        async def output_to_bridge():
            while line := await process.stdout.readline():
                await websocket.send_bytes(line)

        tasks = [asyncio.create_task(input_to_container()), asyncio.create_task(output_to_bridge())]
        finished, waiting = await asyncio.wait(tasks, timeout=MAX_MCP_SESSION_SECONDS,
                                               return_when=asyncio.FIRST_COMPLETED)
        if not finished:
            raise TimeoutError("MCP session time limit reached")
        for task in waiting:
            task.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)
        outcomes = await asyncio.gather(*finished, return_exceptions=True)
        client_closed = False
        for outcome in outcomes:
            if isinstance(outcome, WebSocketDisconnect):
                client_closed = True
            elif isinstance(outcome, BaseException):
                raise outcome
        duration = int((time.monotonic() - start_time) * 1000)
        if client_closed:
            mcp_result(session_id, mode, 0, "client_closed", duration)
        else:
            exit_code = await process.wait()
            if exit_code:
                raise RuntimeError("MCP container exited unsuccessfully")
            mcp_result(session_id, mode, 0, "success", duration)
    except asyncio.CancelledError:
        duration = int((time.monotonic() - start_time) * 1000)
        mcp_cancel(session_id, mode, "websocket_closed")
        raise
    except Exception as exc:
        duration = int((time.monotonic() - start_time) * 1000)
        mcp_cancel(session_id, mode, type(exc).__name__)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
            except asyncio.CancelledError:
                pass
        if process is not None:
            async def finish():
                await cleanup(name)
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except TimeoutError:
                    process.kill()
                    await process.wait()

            task = asyncio.create_task(finish())
            cleanup_tasks.add(task)
            task.add_done_callback(cleanup_tasks.discard)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
