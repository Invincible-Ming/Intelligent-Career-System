"""Redis 共享连接池的行为约束。

背景：三个缓存服务曾各自 Redis.from_url() 建池，冷启动时池内无连接，
首批并发请求各自新建 TCP 连接，本机握手拥塞导致TimeoutError，
产生大量无谓的「Redis 缓存暂不可用」告警。

实测数据（本机）：
- Redis 延迟 p50 约 0.3ms、偶发尖峰 22ms —— 0.5s 超时阈值过宽
- 未预热：24 路并发有 3~16 路失败，报 Timeout connecting to server
- 预热后：24 路并发 100% 成功，预热本身耗时约 43ms
"""
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


class SharedPoolWiringTests(unittest.TestCase):
    """三个缓存服务必须复用同一个池，不能各自建池。"""

    SERVICES = ("analysis_cache.py", "embed_cache.py", "experiment_store.py")

    def setUp(self):
        self.sources = {
            name: (BACKEND / "app" / "services" / name).read_text(encoding="utf-8")
            for name in self.SERVICES
        }

    def test_no_service_creates_its_own_pool(self):
        for name, source in self.sources.items():
            with self.subTest(service=name):
                self.assertNotIn(
                    "Redis.from_url",
                    source,
                    f"{name} 不应自己 Redis.from_url() 建池，"
                    "应通过 redis_pool.get_redis() 复用共享池",
                )

    def test_all_services_import_shared_pool(self):
        for name, source in self.sources.items():
            with self.subTest(service=name):
                self.assertIn(
                    "from app.services.redis_pool import get_redis",
                    source,
                    f"{name} 应从 redis_pool 导入 get_redis",
                )

    def test_services_do_not_close_shared_pool(self):
        """各服务只能丢弃自己的客户端引用，不能关掉共享池。"""

        for name, source in self.sources.items():
            with self.subTest(service=name):
                self.assertNotIn(
                    "await self._client.aclose()",
                    source,
                    f"{name} 调用 aclose() 会关闭共享池，"
                    "影响其他服务；应只置 _client = None",
                )

    def test_app_lifespan_closes_pool_once(self):
        main = (BACKEND / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("close_pool()", main)
        self.assertIn("warm_up_redis_pool()", main)
        # 三个服务的 close 之后才关池，顺序不能反
        self.assertLess(
            main.index("await analysis_cache.close()"),
            main.index("await close_pool()"),
            "应先关闭各服务客户端，再关闭共享池",
        )


class PoolConfigurationTests(unittest.TestCase):
    def test_pool_size_is_configurable(self):
        config = (BACKEND / "app" / "core" / "config.py").read_text(encoding="utf-8")
        self.assertIn("REDIS_POOL_MAX_CONNECTIONS", config)

    def test_socket_timeout_is_tightened(self):
        """0.5s 对本地 Redis 过于宽松，尖峰 22ms 就会误报。"""

        config = (BACKEND / "app" / "core" / "config.py").read_text(encoding="utf-8")
        import re

        match = re.search(
            r"REDIS_SOCKET_TIMEOUT:\s*float\s*=\s*([\d.]+)", config
        )
        self.assertIsNotNone(match, "缺少 REDIS_SOCKET_TIMEOUT")
        self.assertLessEqual(
            float(match.group(1)),
            0.3,
            "Redis 超时阈值应≤ 0.3s，否则瞬时抖动会误报缓存不可用",
        )

    def test_pool_records_owning_loop(self):
        """池要记住所属事件循环，跨 loop 必须重建。"""

        source = (BACKEND / "app" / "services" / "redis_pool.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_pool_loop", source)
        self.assertIn("Event loop is closed", source)

    def test_warm_up_is_serial(self):
        """预热必须串行——并发建连正是要避免的问题。"""

        source = (BACKEND / "app" / "services" / "redis_pool.py").read_text(
            encoding="utf-8"
        )
        warm_up = source[source.index("async def warm_up_redis_pool"):]
        warm_up = warm_up[: warm_up.index("\nasync def close_pool")]

        self.assertIn(
            "results = [await probe() for _ in range(target)]",
            warm_up,
            "预热应逐条串行建立连接",
        )
        self.assertNotIn("asyncio.gather", warm_up, "预热不应并发建连")


if __name__ == "__main__":
    unittest.main()