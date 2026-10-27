"""Mobile jobs use no model keys, DB, Redis server or real provider calls."""
import asyncio
import json
import uuid

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.services import mobile_jobs as jobs


class MemoryRedis:
    def __init__(self):
        self.values, self.hashes, self.sorted = {}, {}, {}

    async def eval(self, script, count, claim, key, slots, signature, ttl, now, capacity, identifier, kind, message, deadline):
        # Model one indivisible Redis script; deliberately no await inside it.
        previous = self.values.get(claim)
        if previous is not None:
            return 0 if previous == signature.encode() else -1
        active = self.sorted.setdefault(slots, {})
        for name in list(active):
            if active[name] <= float(now):
                del active[name]
        if len(active) >= int(capacity):
            return 2
        self.values[claim] = signature.encode()
        self.hashes[key] = {str(k).encode(): str(v).encode() for k, v in {
            "job_id": identifier, "kind": kind, "status": "queued", "message": message,
            "created_at": now, "heartbeat": now,
        }.items()}
        active[key] = float(deadline)
        return 1

    async def zadd(self, key, mapping, xx=False):
        active = self.sorted.setdefault(key, {})
        count = sum(name not in active for name in mapping)
        active.update({name: value for name, value in mapping.items() if not xx or name in active})
        return count

    async def zscore(self, key, name):
        return self.sorted.get(key, {}).get(name)

    async def zrem(self, key, name):
        return self.sorted.get(key, {}).pop(name, None) is not None

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = str(value).encode()
        return True

    async def get(self, key):
        return self.values.get(key)

    async def hset(self, key, field=None, value=None, mapping=None):
        self.hashes.setdefault(key, {}).update({str(k).encode(): str(v).encode() for k, v in (mapping or {field: value}).items()})

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field.encode())

    async def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    async def expire(self, key, ttl):
        return True

    async def delete(self, key):
        self.values.pop(key, None)


@pytest.fixture
async def store(monkeypatch):
    redis = MemoryRedis()
    monkeypatch.setattr(jobs, "get_redis", lambda: redis)
    yield redis
    await jobs.close_mobile_jobs()
    jobs._tasks.clear()


def test_sse_decoder_handles_arbitrary_byte_boundaries():
    value = {"message": "正在检查🙂", "run_id": str(uuid.uuid4())}
    data = ("event: node_update\r\ndata: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n").encode()
    for split in range(len(data) + 1):
        decoder = jobs.EventDecoder()
        assert decoder.feed(data[:split]) + decoder.feed(data[split:]) == [("node_update", value)]


@pytest.mark.anyio
async def test_job_executes_once_with_real_api_auth_header_and_private_owner_namespace(store):
    app = FastAPI()
    calls = []
    run_id = str(uuid.uuid4())

    @app.post("/api/match/stream")
    async def work(request: Request):
        assert request.headers["authorization"] == "Bearer synthetic-mobile-token"
        calls.append(await request.json())

        async def events():
            yield f'event: init\ndata: {{"run_id":"{run_id}"}}\n\n'
            yield 'event: interrupt\ndata: {"status":"paused","current_report":{"total_score":72}}\n\n'
        return StreamingResponse(events(), media_type="text/event-stream")

    owner, identifier = uuid.uuid4(), uuid.uuid4()
    kwargs = dict(app=app, owner=owner, identifier=identifier, kind="match", payload={"jd_text": "synthetic"}, token="synthetic-mobile-token")
    await jobs.start_job(**kwargs)
    await asyncio.gather(*list(jobs._tasks.values()))
    state = await jobs.start_job(**kwargs)
    assert len(calls) == 1
    assert state["status"] == "paused" and state["run_id"] == run_id
    assert state["result"]["current_report"]["total_score"] == 72
    assert "synthetic-mobile-token" not in str(store.hashes) + str(store.values)
    with pytest.raises(HTTPException) as denied:
        await jobs.load_job(uuid.uuid4(), identifier)
    assert denied.value.status_code == 404
    with pytest.raises(HTTPException) as mismatch:
        await jobs.start_job(**(kwargs | {"payload": {"jd_text": "different"}}))
    assert mismatch.value.status_code == 409


@pytest.mark.anyio
async def test_upstream_permission_failure_does_not_run_model_or_claim_success(store):
    app = FastAPI()

    @app.post("/api/interview")
    async def denied():
        raise HTTPException(404, "岗位匹配任务不存在")

    owner, identifier = uuid.uuid4(), uuid.uuid4()
    await jobs.start_job(app=app, owner=owner, identifier=identifier, kind="interview", payload={}, token="fake")
    await asyncio.gather(*list(jobs._tasks.values()))
    state = await jobs.load_job(owner, identifier)
    assert state["status"] == "failed" and state["error"] == "岗位匹配任务不存在"


@pytest.mark.anyio
async def test_restart_is_detected_without_replaying_a_paid_job(store):
    owner, identifier = uuid.uuid4(), uuid.uuid4()
    await store.hset(jobs.job_key(owner, identifier), mapping={"job_id": str(identifier), "kind": "match", "status": "running", "heartbeat": "0"})
    state = await jobs.load_job(owner, identifier)
    assert state["status"] == "failed"
    assert "中断" in state["error"]
    assert not jobs._tasks


@pytest.mark.anyio
async def test_cancel_requires_owner_and_signals_worker_without_replaying(store):
    owner, identifier = uuid.uuid4(), uuid.uuid4()
    import time
    await store.hset(jobs.job_key(owner, identifier), mapping={"job_id": str(identifier), "status": "running", "heartbeat": str(time.time())})
    with pytest.raises(HTTPException):
        await jobs.cancel_job(uuid.uuid4(), identifier)
    assert await store.hget(jobs.job_key(owner, identifier), "cancel") is None
    await jobs.cancel_job(owner, identifier)
    assert await store.hget(jobs.job_key(owner, identifier), "cancel") == b"1"


@pytest.mark.anyio
async def test_concurrent_admission_is_bounded_and_duplicate_is_reusable_when_full(store, monkeypatch):
    monkeypatch.setattr(jobs.settings, "MOBILE_JOB_MAX_WORKERS", 16)
    release = asyncio.Event()
    app = FastAPI()

    @app.post("/api/interview")
    async def work():
        await release.wait()
        return {"status": "completed", "run_id": str(uuid.uuid4()), "result": {}}

    owner = uuid.uuid4()
    identifiers = [uuid.uuid4() for _ in range(40)]
    async def submit(identifier):
        return await jobs.start_job(app=app, owner=owner, identifier=identifier,
                                    kind="interview", payload={}, token="synthetic")
    results = await asyncio.gather(*(submit(i) for i in identifiers), return_exceptions=True)
    accepted = [item for item in results if isinstance(item, dict)]
    assert len(accepted) == len(jobs._tasks) == 16
    assert all(item.status_code == 429 for item in results if isinstance(item, HTTPException))
    # No capacity needed for replaying an already admitted ID.
    assert (await submit(uuid.UUID(accepted[0]["job_id"])))["job_id"] == accepted[0]["job_id"]
    release.set()
    await asyncio.gather(*list(jobs._tasks.values()))
    assert not store.sorted[jobs.ACTIVE_JOBS]
    rejected = next(identifier for identifier, result in zip(identifiers, results, strict=False) if isinstance(result, HTTPException))
    assert (await submit(rejected))["job_id"] == str(rejected)


@pytest.mark.anyio
async def test_expired_worker_slot_recovers_but_does_not_replay_old_job(store, monkeypatch):
    monkeypatch.setattr(jobs.settings, "MOBILE_JOB_MAX_WORKERS", 1)
    old = jobs.job_key(uuid.uuid4(), uuid.uuid4())
    store.sorted[jobs.ACTIVE_JOBS] = {old: 0}
    app = FastAPI()

    @app.post("/api/interview")
    async def work():
        return {"status": "completed", "run_id": str(uuid.uuid4()), "result": {}}

    result = await jobs.start_job(app=app, owner=uuid.uuid4(), identifier=uuid.uuid4(), kind="interview", payload={}, token="synthetic")
    assert result["status"] == "queued"
    assert old not in store.sorted[jobs.ACTIVE_JOBS]


@pytest.mark.anyio
async def test_cancelling_an_active_job_releases_capacity(store, monkeypatch):
    monkeypatch.setattr(jobs, "HEARTBEAT_SECONDS", 0.01)
    app = FastAPI()

    @app.post("/api/interview")
    async def work():
        await asyncio.Event().wait()

    owner, identifier = uuid.uuid4(), uuid.uuid4()
    await jobs.start_job(app=app, owner=owner, identifier=identifier, kind="interview", payload={}, token="synthetic")
    await jobs.cancel_job(owner, identifier)
    await asyncio.wait_for(asyncio.gather(*list(jobs._tasks.values())), 1)
    assert (await jobs.load_job(owner, identifier))["status"] == "cancelled"
    assert not store.sorted[jobs.ACTIVE_JOBS]


@pytest.mark.anyio
async def test_real_redis_script_is_atomic_under_concurrent_admission(monkeypatch):
    """Uses only the isolated test Redis from conftest, never the app Redis."""
    from redis.asyncio import Redis
    from redis.exceptions import RedisError

    redis = Redis.from_url(jobs.settings.REDIS_URL, socket_connect_timeout=0.2)
    try:
        await redis.ping()
    except RedisError:
        await redis.aclose()
        pytest.skip("临时测试 Redis 未启动")
    monkeypatch.setattr(jobs, "get_redis", lambda: redis)
    slots = "mobile-job:test:" + uuid.uuid4().hex
    monkeypatch.setattr(jobs, "ACTIVE_JOBS", slots)
    monkeypatch.setattr(jobs.settings, "MOBILE_JOB_MAX_WORKERS", 3)
    app, release = FastAPI(), asyncio.Event()
    calls = []

    @app.post("/api/interview")
    async def work():
        calls.append(True)
        await release.wait()
        return {"status": "completed", "run_id": str(uuid.uuid4()), "result": {}}

    owner = uuid.uuid4()
    identifiers = [uuid.uuid4() for _ in range(20)]
    async def submit(identifier):
        return await jobs.start_job(app=app, owner=owner, identifier=identifier, kind="interview", payload={}, token="synthetic")
    try:
        results = await asyncio.gather(*(submit(i) for i in identifiers), return_exceptions=True)
        accepted = [x for x in results if isinstance(x, dict)]
        assert len(accepted) == 3
        assert all(isinstance(x, dict) or isinstance(x, HTTPException) and x.status_code == 429 for x in results)
        assert await redis.zcard(slots) == 3
        assert (await submit(uuid.UUID(accepted[0]["job_id"])))["job_id"] == accepted[0]["job_id"]
        release.set()
        await asyncio.gather(*list(jobs._tasks.values()))
        assert len(calls) == 3
        assert await redis.zcard(slots) == 0
    finally:
        await jobs.close_mobile_jobs()
        keys = [jobs.job_key(owner, identifier) for identifier in identifiers]
        await redis.delete(slots, *keys, *(key + ":claim" for key in keys))
        await redis.aclose()
