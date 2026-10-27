"""Bounded mobile jobs: poll Redis, execute the existing authorized API unchanged.

Credentials stay in worker memory. Results/runs are still owned by PostgreSQL users.
An interrupted worker is never automatically replayed (which could charge twice).
"""
from __future__ import annotations

import asyncio
import codecs
import contextvars
import hashlib
import json
import logging
import time
from uuid import UUID

import httpx
from fastapi import HTTPException
from redis.exceptions import RedisError

from app.core.config import settings
from app.services.redis_pool import get_redis

TTL = 86400
ACTIVE_JOBS = "mobile-job:v1:active"
LEASE_SECONDS = 45
HEARTBEAT_SECONDS = 3
# 面试与学习任务可能持续生成较久，这里给一个足够长但仍有限的上限：
# 无上限（timeout=None）时，一次卡死的内部请求会永久占住一个任务名额。
JOB_REQUEST_TIMEOUT_SECONDS = 3600.0
logger = logging.getLogger(__name__)
# Reserve identity, progress record and a shared execution slot in ONE Redis
# operation. Different workers cannot overbook or observe a half-created job.
ADMIT_JOB = """
local previous = redis.call('GET', KEYS[1])
if previous then
    if previous == ARGV[1] then return 0 else return -1 end
end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[3])
if redis.call('ZCARD', KEYS[3]) >= tonumber(ARGV[4]) then return 2 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
redis.call('HSET', KEYS[2], 'job_id', ARGV[5], 'kind', ARGV[6],
    'status', 'queued', 'message', ARGV[7], 'created_at', ARGV[3], 'heartbeat', ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[2])
redis.call('ZADD', KEYS[3], ARGV[8], KEYS[2])
return 1
"""
PATHS = {"match": "/match/stream", "resume": "/match/stream/resume",
         "interview": "/interview", "learning": "/learning-plan"}
TERMINAL = {"completed", "paused", "failed", "cancelled"}
_tasks: dict[str, asyncio.Task] = {}


def job_key(owner: UUID, identifier: UUID) -> str:
    return f"mobile-job:v1:{owner}:{identifier}"


def unpack(raw):
    return {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in raw.items()}


def public_job(record):
    return {key: record.get(key, "") for key in (
        "job_id", "kind", "status", "run_id", "message", "error", "created_at",
    )} | {"result": json.loads(record.get("result") or "null")}


class EventDecoder:
    """UTF-8 and SSE frames may both be split at arbitrary byte boundaries."""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""

    def feed(self, data: bytes):
        self.buffer += self.decoder.decode(data)
        self.buffer = self.buffer.replace("\r\n", "\n")
        events = []
        while "\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split("\n\n", 1)
            event, parts = "message", []
            for line in frame.splitlines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                if line.startswith("data:"):
                    parts.append(line[5:].lstrip())
            if parts:
                try:
                    value = json.loads("\n".join(parts))
                except ValueError:
                    continue
                if isinstance(value, dict):
                    events.append((event, value))
        if len(self.buffer) > 2_000_000:
            raise ValueError("Stream frame too large")
        return events


async def load_job(owner: UUID, identifier: UUID):
    try:
        redis = get_redis()
        record = unpack(await redis.hgetall(job_key(owner, identifier)))
        if not record:
            raise HTTPException(404, "任务不存在或已过期，请查看任务历史")
        if record["status"] not in TERMINAL and time.time() - float(record["heartbeat"]) > 30:
            # Heartbeats stop on a worker restart. Do not silently resubmit a paid operation.
            await redis.hset(job_key(owner, identifier), mapping={"status": "failed", "error": "执行连接已中断，请先查看任务历史，确认结果后再决定是否重新提交"})
            record.update(status="failed", error="执行连接已中断，请先查看任务历史")
        return public_job(record)
    except RedisError:
        raise HTTPException(503, "任务进度服务暂不可用，请稍后重试") from None


async def start_job(*, app, owner: UUID, identifier: UUID, kind: str, payload: dict, token: str):
    key = job_key(owner, identifier)
    signature = hashlib.sha256(json.dumps([kind, payload], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    redis = get_redis()
    try:
        now = time.time()
        admitted = await redis.eval(
            ADMIT_JOB, 3, key + ":claim", key, ACTIVE_JOBS,
            signature, TTL, now, settings.MOBILE_JOB_MAX_WORKERS,
            str(identifier), kind, "任务已提交", now + LEASE_SECONDS,
        )
        if admitted == -1:
            raise HTTPException(409, "同一请求编号不能用于不同操作")
        if admitted == 0:
            return await load_job(owner, identifier)
        if admitted == 2:
            raise HTTPException(429, "当前任务较多，请稍后提交", headers={"Retry-After": "3"})
        task = asyncio.create_task(execute_job(app, key, kind, payload, token), context=contextvars.Context())
        _tasks[key] = task
        def finished(done):
            _tasks.pop(key, None)
            if not done.cancelled() and done.exception() is not None:
                logger.error("小程序任务清理失败（%s）", type(done.exception()).__name__)
        task.add_done_callback(finished)
        return await load_job(owner, identifier)
    except RedisError:
        # A connection may drop after Redis accepted the atomic reservation.
        # Keep the client request ID so retries never duplicate a paid operation.
        raise HTTPException(503, "任务服务暂不可用，请使用原请求编号重试或查看任务历史") from None


async def execute_job(app, key, kind, payload, token):
    redis = get_redis()
    decoder = EventDecoder()
    main_task = asyncio.current_task()
    status = 0

    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                if await redis.hget(key, "cancel") == b"1":
                    main_task.cancel()
                    return
                now = time.time()
                # XX prevents a delayed worker from resurrecting an expired slot.
                if not await redis.zadd(ACTIVE_JOBS, {key: now + LEASE_SECONDS}, xx=True):
                    # ZADD without CH counts additions, not updated scores. Check
                    # membership separately to distinguish a renewal from expiry.
                    if await redis.zscore(ACTIVE_JOBS, key) is None:
                        main_task.cancel()
                        return
                await redis.hset(key, "heartbeat", str(now))
        except RedisError:
            # Never keep generating after losing the shared capacity lease.
            main_task.cancel()

    async def observed(scope, receive, send):
        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            if message["type"] == "http.response.body" and status < 400:
                for event, value in decoder.feed(message.get("body", b"")):
                    update = {}
                    if value.get("run_id") or value.get("thread_id"):
                        update["run_id"] = str(value.get("run_id") or value["thread_id"])
                    if value.get("message"):
                        update["message"] = str(value["message"])[:600]
                    if event in ("complete", "interrupt"):
                        update.update(status="completed" if event == "complete" else "paused",
                                      result=json.dumps(value, ensure_ascii=False))
                    elif event == "error":
                        update.update(status="failed", error="任务未完成，请查看进度或稍后重试")
                    if update:
                        await redis.hset(key, mapping=update)
            await send(message)
        await app(scope, receive, capture)

    watcher = None
    try:
        await redis.hset(key, mapping={"status": "running", "heartbeat": str(time.time())})
        watcher = asyncio.create_task(heartbeat())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=observed), base_url="http://127.0.0.1:8000",
                                      timeout=JOB_REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
            response = await client.post(settings.API_PREFIX + PATHS[kind], json=payload,
                                         headers={"Authorization": "Bearer " + token})
        if response.is_error:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            await redis.hset(key, mapping={"status": "failed", "error": detail if isinstance(detail, str) else "操作未完成，请稍后重试"})
        elif kind in ("interview", "learning"):
            result = response.json()
            await redis.hset(key, mapping={"status": result["status"], "run_id": result["run_id"],
                                          "result": json.dumps(result, ensure_ascii=False), "message": "生成完成"})
        elif (await redis.hget(key, "status")) not in (b"completed", b"paused", b"failed"):
            await redis.hset(key, mapping={"status": "failed", "error": "执行结束但未收到完整结果，请查看任务历史"})
    except asyncio.CancelledError:
        await redis.hset(key, mapping={"status": "cancelled", "error": "任务已停止，已产生的模型用量仍会结算"})
    except Exception:
        await redis.hset(key, mapping={"status": "failed", "error": "任务执行中断，请查看任务历史"})
    finally:
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        try:
            await redis.zrem(ACTIVE_JOBS, key)
        except RedisError:
            pass  # Crashed/disconnected workers' slots expire after 45 seconds.


async def cancel_job(owner: UUID, identifier: UUID):
    state = await load_job(owner, identifier)
    if state["status"] not in TERMINAL:
        await get_redis().hset(job_key(owner, identifier), "cancel", "1")
    return state


async def close_mobile_jobs():
    tasks = list(_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
