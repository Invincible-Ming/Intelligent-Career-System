"""评测实验状态存储。

原先状态只存在进程内 dict，重启即丢、多 worker 下各自为政。
这里改为 Redis：所有 worker 共享同一份状态，重启后仍可查询。

设计取舍：

- 状态按 owner_id 分前缀，列表查询只扫描当前管理员自己的命名空间，
  不存在跨用户泄漏；与系统其他部分的 owner_id 隔离维度一致。
- Redis 不可用时降级为进程内 dict，保证评测仍能跑完（只是不持久化）。
- 状态不设 TTL：报告文件已落盘，状态需要长期可查；由clear_expired 定期清理。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from redis.exceptions import RedisError

from app.core.config import settings
from app.services.redis_pool import get_redis

logger = logging.getLogger(__name__)

KEY_PREFIX = "career:evaluation:v1"
# 终态实验保留时长：足够排查问题，又不会让 Redis 无限增长。
TERMINAL_TTL = 7 * 24 * 3600


class ExperimentStore:
    """评测实验状态的 Redis 存储，含进程内降级。"""

    def __init__(self) -> None:
        self._client: Redis | None = None
        self._skip_until = 0.0
        self._fallback: dict[str, dict[str, Any]] = {}
        # 启动时无法得知Redis 里有哪些键，降级期间补记已知的键以便回写。
        self._fallback_keys: set[str] = set()

    @property
    def enabled(self) -> bool:
        return bool(settings.EVALUATION_STORE_REDIS_ENABLED)

    @property
    def client(self) -> Redis:
        if self._client is None:
            self._client = get_redis()
        return self._client

    def unavailable(self, exc: Exception) -> None:
        """Redis 不可用时熔断，并同步现有降级状态，供恢复后回写。"""

        self._skip_until = time.monotonic() + 30
        logger.warning(
            "Redis 评测状态存储暂不可用（%s），暂用进程内状态",
            type(exc).__name__,
        )
        for key, status in self._fallback.items():
            self._fallback_keys.add(key)

    @staticmethod
    def key(owner_id: str, experiment_id: str) -> str:
        return f"{KEY_PREFIX}:{owner_id}:{experiment_id}"

    async def initialize(self) -> None:
        if not self.enabled:
            return

        try:
            await self.client.ping()
            logger.info("Redis 评测状态存储已连接")
        except RedisError as exc:
            self.unavailable(exc)

    @property
    def _degraded(self) -> bool:
        return not self.enabled or time.monotonic() < self._skip_until

    async def save(
            self,
            *,
            owner_id: str,
            experiment_id: str,
            status: dict[str, Any],
    ) -> None:
        """写入或更新实验状态。

        存的是拷贝：调用方后续修改原 dict 不应影响已落库的状态。
        """

        key = self.key(owner_id, experiment_id)
        snapshot = dict(status)
        payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        self._fallback[key] = snapshot

        if self._degraded:
            self._fallback_keys.add(key)
            return

        try:
            terminal = snapshot.get("status") in {"completed", "failed"}
            await self.client.set(
                key,
                payload,
                ex=TERMINAL_TTL if terminal else None,
            )
            self._fallback_keys.discard(key)
        except RedisError as exc:
            self.unavailable(exc)

    async def get(
            self,
            *,
            owner_id: str,
            experiment_id: str,
    ) -> dict[str, Any] | None:
        """读取单个实验状态；不属于该owner 时返回 None。

        返回的是拷贝，调用方可以安全地读-改-写后再 save。
        """

        key = self.key(owner_id, experiment_id)

        if not self._degraded:
            try:
                raw = await self.client.get(key)
                if raw is not None:
                    return json.loads(raw)
                return None
            except (RedisError, ValueError) as exc:
                if isinstance(exc, ValueError):
                    # 数据损坏视同未命中，并顺手清掉坏键。
                    try:
                        await self.client.delete(key)
                    except RedisError:
                        pass
                else:
                    self.unavailable(exc)

        stored = self._fallback.get(key)
        return dict(stored) if stored is not None else None

    async def list_for_owner(
            self,
            *,
            owner_id: str,
    ) -> list[dict[str, Any]]:
        """列出该管理员名下全部实验，按创建时间倒序。"""

        prefix = f"{KEY_PREFIX}:{owner_id}:"
        statuses: list[dict[str, Any]] = []

        if not self._degraded:
            try:
                keys = [key async for key in self.client.scan_iter(match=f"{prefix}*")]
                if keys:
                    values = await self.client.mget(keys)
                    for raw in values:
                        if raw is None:
                            continue
                        try:
                            statuses.append(json.loads(raw))
                        except ValueError:
                            continue
            except RedisError as exc:
                self.unavailable(exc)

        if not statuses:
            # 降级路径，或 Redis 里没扫到（可能都在降级期间写入的）。
            statuses = [
                dict(status)
                for key, status in self._fallback.items()
                if key.startswith(prefix)
            ]

        return sorted(
            statuses,
            key=lambda item: item.get("started_at", 0),
            reverse=True,
        )

    async def clear_expired(self) -> int:
        """清理终态实验的进程内副本。

        Redis 侧终态键自带 TTL，无需手动处理；这里只回收本地副本。
        """

        cleared = 0

        for key in list(self._fallback):
            if self._fallback[key].get("status") in {"completed", "failed"}:
                self._fallback.pop(key, None)
                self._fallback_keys.discard(key)
                cleared += 1

        return cleared

    async def close(self) -> None:
        # 共享池不由单个服务关闭，只丢弃自己的客户端引用。
        self._client = None
        self._fallback.clear()
        self._fallback_keys.clear()


experiment_store = ExperimentStore()