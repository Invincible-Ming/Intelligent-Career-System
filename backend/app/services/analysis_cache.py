"""User-scoped Redis extraction cache; failures never prevent matching."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import time

import anyio
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.limits import check_operation_deadline, invocation_budget

logger = logging.getLogger(__name__)
RELEASE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class AnalysisCache:
    def __init__(self):
        self._client = None
        self._skip_until = 0.0

    @property
    def client(self):
        if self._client is None:
            self._client = Redis.from_url(
                settings.REDIS_URL, decode_responses=True, max_connections=16,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
                retry=Retry(NoBackoff(), 0),
            )
        return self._client

    def key(self, *, owner_id, kind, text, system_prompt, prompt_version, response_model):
        fingerprint = json.dumps({
            "kind": kind, "text": text, "model": settings.BAILIAN_CHAT_MODEL,
            "generation_options": settings.match_model_options,
            "prompt": system_prompt, "version": prompt_version,
            "schema": response_model.model_json_schema(),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        owner = hashlib.sha256(str(owner_id).encode()).hexdigest()
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()
        return f"career:analysis:v1:{owner}:{kind}:{digest}"

    def unavailable(self, exc):
        self._skip_until = time.monotonic() + 30
        # Never log Redis connection URLs or resume/JD contents.
        logger.warning("Redis 分析缓存暂不可用（%s），继续模型分析", type(exc).__name__)

    async def initialize(self):
        if settings.ANALYSIS_CACHE_ENABLED:
            try:
                await self.client.ping()
                logger.info("Redis 分析缓存已连接")
            except RedisError as exc:
                self.unavailable(exc)

    async def get_or_compute(self, *, owner_id, kind, text, system_prompt,
                             prompt_version, response_model, compute):
        if not owner_id or not settings.ANALYSIS_CACHE_ENABLED or time.monotonic() < self._skip_until:
            return await compute(), False
        key = self.key(owner_id=owner_id, kind=kind, text=text, system_prompt=system_prompt,
                       prompt_version=prompt_version, response_model=response_model)
        lock_key, lock_token = key + ":lock", secrets.token_hex(16)
        budget = invocation_budget.get()
        remaining = budget.deadline - time.monotonic() if budget else settings.MATCH_TOTAL_TIMEOUT
        lock_ttl = max(1, math.ceil(remaining)) + 10
        locked = False

        async def read_validated():
            raw = await self.client.get(key)
            if raw is not None:
                try:
                    return response_model.model_validate_json(raw)
                except (ValidationError, ValueError):
                    await self.client.delete(key)
            return None

        try:
            try:
                while True:
                    check_operation_deadline()
                    cached = await read_validated()
                    if cached is not None:
                        return cached, True
                    locked = bool(await self.client.set(lock_key, lock_token, nx=True, ex=lock_ttl))
                    if locked:
                        # Another producer may have published just before we got the lock.
                        cached = await read_validated()
                        if cached is not None:
                            return cached, True
                        break
                    await asyncio.sleep(.2)
            except RedisError as exc:
                self.unavailable(exc)
                return await compute(), False

            result = await compute()
            # Validate before caching; an upstream error never becomes a cache hit.
            result = response_model.model_validate(result.model_dump(mode="json"))
            try:
                await self.client.set(key, result.model_dump_json(), ex=settings.ANALYSIS_CACHE_TTL)
            except RedisError as exc:
                self.unavailable(exc)  # Return the result without repeating a paid call.
            return result, False
        finally:
            if locked:
                with anyio.CancelScope(shield=True):
                    try:
                        async with asyncio.timeout(1):
                            await self.client.eval(RELEASE_LOCK, 1, lock_key, lock_token)
                    except (RedisError, TimeoutError):
                        pass  # Redis lock expires if cancellation/crash prevents release.

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


analysis_cache = AnalysisCache()
