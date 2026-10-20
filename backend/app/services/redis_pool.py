"""共享 Redis 连接池。

三个缓存服务（analysis_cache / embed_cache / experiment_store）共用
进程内同一个池，统一限制连接数并复用空闲连接。

`max_connections` 仍需足够大——每条命令执行期间都独占一个连接，
池大小应≥ 预期最大并发命令数。
"""
from __future__ import annotations

import asyncio
import logging

from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
# 池所属的事件循环。IsolatedAsyncioTestCase 会为每个用例新建 loop，
# 跨 loop 复用连接池会抛 "Event loop is closed"。
# 生产环境全程单loop，这里主要为测试与多loop 场景兜底。
_pool_loop: asyncio.AbstractEventLoop | None = None


def get_pool() -> ConnectionPool:
    """返回进程内共享的连接池（首次调用时创建）。"""

    global _pool, _pool_loop

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _pool is not None and _pool_loop is not current_loop:
        # 事件循环换了（测试场景），旧池不可复用，丢弃重建。
        _pool = None

    if _pool is None:
        _pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=False,
            max_connections=settings.REDIS_POOL_MAX_CONNECTIONS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_keepalive=True,
            # 不做自动重试：缓存是可降级组件，重试只会放大延迟。
            retry=Retry(NoBackoff(), 0),
        )
        _pool_loop = current_loop
        logger.info(
            "Redis 共享连接池已创建（max_connections=%d）",
            settings.REDIS_POOL_MAX_CONNECTIONS,
        )

    return _pool


def get_redis() -> Redis:
    """基于共享池创建客户端。

    decode_responses 是**连接级**选项，共享池时无法按客户端切换，
    因此池统一 decode_responses=False（返回 bytes）。
    需要 str 的调用方在读取后自行 decode——调用点少且显式，
    比维护两套池或塞解码回调更不容易出错。
    """

    return Redis(connection_pool=get_pool(), decode_responses=False)


async def warm_up_redis_pool(pool_size: int | None = None) -> int:
    """在启动阶段串行检查独立连接，完成后统一归还共享池。

    每个临时客户端独占一条连接，避免每次 PING 后立即归还、反复复用
    同一连接。返回通过 PING 检查的独立连接数量，不是命令成功次数。
    None 表示池容量，0 跳过预热；显式目标不会超过池容量。
    """

    if pool_size is not None and (not isinstance(pool_size, int) or pool_size < 0):
        raise ValueError("Redis 预热连接数必须是非负整数")
    if pool_size == 0:
        return 0

    pool = get_pool()
    target = min(pool_size if pool_size is not None else pool.max_connections, pool.max_connections)
    held_clients: list[Redis] = []

    async def probe() -> bool:
        client = Redis(
            connection_pool=pool,
            decode_responses=False,
            single_connection_client=True,
        )
        # 先记录再执行：PING 失败或预热被取消时，也必须释放已借出的连接。
        held_clients.append(client)
        try:
            return bool(await client.ping())
        except RedisError:
            return False

    try:
        # 串行建连，同时持有已借出的连接，下一次 probe 才会取得另一条。
        results = [await probe() for _ in range(target)]
    finally:
        for client in held_clients:
            try:
                # 仅归还此客户端的独占连接，不关闭三个服务共用的池。
                await client.aclose(close_connection_pool=False)
            except RedisError as exc:
                logger.warning("Redis 预热连接归还失败（%s）", type(exc).__name__)
    established = sum(results)

    if established == target:
        logger.info("Redis 连接池预热完成：%d/%d 条独立连接通过 PING 检查", established, target)
    elif established:
        logger.warning("Redis 连接池部分预热：%d/%d 条独立连接通过 PING 检查", established, target)
    else:
        logger.warning("Redis 连接池预热失败：缓存服务将按现有策略尝试连接或降级")

    return established


async def close_pool() -> None:
    """关闭连接池（应用关闭时调用）。"""

    global _pool, _pool_loop

    if _pool is not None:
        await _pool.disconnect()
        _pool = None
        _pool_loop = None
