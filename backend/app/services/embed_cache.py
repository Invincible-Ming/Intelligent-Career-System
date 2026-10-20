"""Embedding 向量缓存。

同一段文本（检索 query、重复上传的同一文档分块）在有效期内
只调用一次 DashScope Embedding。缓存不可用时静默降级为直接调用，
绝不影响检索与入库的正确性。
"""
from __future__ import annotations

import hashlib
import json
import logging
import struct
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.services.redis_pool import get_redis

logger = logging.getLogger(__name__)

KEY_PREFIX = "career:embed:v1"


class EmbedCache:
    """按文本指纹缓存 Embedding 结果。"""

    def __init__(self) -> None:
        self._client: Redis | None = None
        self._skip_until = 0.0
        # 进程内二级缓存：同一批次的重复分块不必走网络往返。
        self._local: dict[str, list[float]] = {}
        self._local_order: list[str] = []
        self._local_limit = 512

    @property
    def enabled(self) -> bool:
        return bool(settings.EMBED_CACHE_ENABLED)

    @property
    def client(self) -> Redis:
        if self._client is None:
            self._client = get_redis()
        return self._client

    def unavailable(self, exc: Exception) -> None:
        """Redis 不可用时熔断 30 秒，避免每次请求都重连。"""

        self._skip_until = time.monotonic() + 30
        logger.warning(
            "Redis Embedding 缓存暂不可用（%s），直接调用 Embedding 模型",
            type(exc).__name__,
        )

    def key(self, text: str) -> str:
        fingerprint = json.dumps(
            {
                "model": settings.BAILIAN_EMBEDDING_MODEL,
                "dimension": settings.EMBEDDING_DIMENSION,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return KEY_PREFIX + ":" + hashlib.sha256(fingerprint.encode()).hexdigest()

    def _local_get(self, key: str) -> list[float] | None:
        return self._local.get(key)

    def _local_put(self, key: str, vector: list[float]) -> None:
        if len(self._local_order) >= self._local_limit:
            evicted = self._local_order.pop(0)
            self._local.pop(evicted, None)
        self._local[key] = vector
        self._local_order.append(key)

    async def initialize(self) -> None:
        """启动时探活 Redis；不可用只告警，不阻断启动。"""

        if not self.enabled:
            return

        try:
            await self.client.ping()
            logger.info("Redis Embedding 缓存已连接")
        except RedisError as exc:
            self.unavailable(exc)

    async def lookup(
            self,
            texts: list[str],
    ) -> dict[str, list[float]]:
        """批量查缓存，返回命中的文本→向量映射。"""

        if not self.enabled or not texts:
            return {}

        keys = {text: self.key(text) for text in texts}
        found: dict[str, list[float]] = {}

        for text, key in keys.items():
            local = self._local_get(key)
            if local is not None:
                found[text] = local

        pending = [text for text in keys if text not in found]
        if not pending:
            return found

        if time.monotonic() < self._skip_until:
            return found

        try:
            values = await self.client.mget([keys[text] for text in pending])
        except RedisError as exc:
            self.unavailable(exc)
            return found

        for text, raw in zip(pending, values, strict=True):
            if raw is None:
                continue
            vector = decode_vector(raw)
            if vector is not None:
                found[text] = vector
                self._local_put(keys[text], vector)

        return found

    async def store(
            self,
            texts: list[str],
            vectors: list[list[float]],
    ) -> None:
        """批量写缓存。失败只记录日志，不影响主流程。"""

        if not self.enabled or not texts:
            return

        keys = {text: self.key(text) for text in texts}
        for text, vector in zip(texts, vectors, strict=True):
            self._local_put(keys[text], vector)

        if time.monotonic() < self._skip_until:
            return

        payload = {
            keys[text]: encode_vector(vector)
            for text, vector in zip(texts, vectors, strict=True)
        }
        try:
            # 逐键设置 TTL：Redis 不支持按前缀批量过期。
            async with self.client.pipeline(transaction=False) as pipe:
                pipe.mset(payload)
                for key in payload:
                    pipe.expire(key, settings.EMBED_CACHE_TTL)
                await pipe.execute()
        except RedisError as exc:
            self.unavailable(exc)

    async def clear(self) -> None:
        """清空进程内缓存（Redis 侧按 TTL 自然过期）。"""

        self._local.clear()
        self._local_order.clear()

    async def close(self) -> None:
        # 共享池不由单个服务关闭，只丢弃自己的客户端引用。
        self._client = None


def encode_vector(vector: list[float]) -> bytes:
    """紧凑编码向量，避免 JSON 存储带来的体积与解析开销。"""

    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(raw: bytes | str) -> list[float] | None:
    """解析缓存中的向量；格式不兼容时视为未命中。"""

    if isinstance(raw, str):
        raw = raw.encode()

    if len(raw) % 4 != 0 or not raw:
        return None

    try:
        count = len(raw) // 4
        return list(struct.unpack(f"<{count}f", raw))
    except struct.error:
        return None


embed_cache = EmbedCache()
