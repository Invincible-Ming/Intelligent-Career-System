"""Embedding 缓存行为测试。

缓存是否生效直接决定模型调用次数（= Token 成本），因此重点验证：
- 相同文本重复embed 不再调用模型；
- 同批次内的重复文本也只算一次；
- 新旧混合批次只补算未命中的部分；
- 返回向量的顺序与数量必须与输入严格一致。

这些用例需要 redis / dashscope / openai / langchain_core 等完整依赖，
缺任一项时由 conftest 在收集阶段跳过本文件。
"""
import unittest
import uuid
from unittest.mock import patch

from app.core.config import settings
from app.services.bailian import bailian_service
from app.services.embed_cache import EmbedCache, decode_vector, encode_vector

DIM = 16


class FakeEmbeddingResponse:
    """模拟 DashScope 返回结构。"""

    def __init__(self, texts):
        self.output = {
            "embeddings": [
                {
                    "text_index": index,
                    "embedding": [float(index + 1)] * DIM,
                }
                for index, _ in enumerate(texts)
            ],
        }


class VectorCodecTests(unittest.TestCase):
    """二进制编解码必须无损，且坏数据视为未命中。"""

    def test_roundtrip_preserves_values(self):
        vector = [0.5, -0.25, 3.75, 0.0]

        decoded = decode_vector(encode_vector(vector))

        self.assertIsNotNone(decoded)
        self.assertEqual(len(decoded), len(vector))
        for original, restored in zip(vector, decoded, strict=True):
            self.assertAlmostEqual(original, restored, places=5)

    def test_invalid_payload_returns_none(self):
        self.assertIsNone(decode_vector(b""))
        self.assertIsNone(decode_vector(b"abc"))  # 长度不是 4 的倍数

    def test_key_is_stable_and_model_sensitive(self):
        cache = EmbedCache()
        first = cache.key("abc")

        # 同一文本同一配置 -> key 稳定
        self.assertEqual(first, cache.key("abc"))
        self.assertNotEqual(first, cache.key("abd"))

        # 换模型 -> key 必须变，否则会命中旧模型的向量
        original_model = settings.BAILIAN_EMBEDDING_MODEL
        try:
            settings.BAILIAN_EMBEDDING_MODEL = "another-model"
            self.assertNotEqual(first, cache.key("abc"))
        finally:
            settings.BAILIAN_EMBEDDING_MODEL = original_model

    def test_key_changes_with_dimension(self):
        cache = EmbedCache()
        first = cache.key("abc")

        original = settings.EMBEDDING_DIMENSION
        try:
            settings.EMBEDDING_DIMENSION = original + 1
            self.assertNotEqual(first, cache.key("abc"))
        finally:
            settings.EMBEDDING_DIMENSION = original


class EmbedCacheIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """端到端验证 embed() 的缓存命中行为。"""

    async def asyncSetUp(self):
        self.call_count = 0
        self.seen_texts: list[list[str]] = []

        original_key = settings.EMBEDDING_DIMENSION
        settings.EMBEDDING_DIMENSION = DIM
        self.addCleanup(setattr, settings, "EMBEDDING_DIMENSION", original_key)

        original_api_key = settings.DASHSCOPE_API_KEY
        settings.DASHSCOPE_API_KEY = "test-key"
        self.addCleanup(setattr, settings, "DASHSCOPE_API_KEY", original_api_key)

        async def fake_retry(texts):
            self.call_count += 1
            self.seen_texts.append(list(texts))
            return FakeEmbeddingResponse(texts)

        self._patcher = patch.object(
            bailian_service, "_embedding_retry", side_effect=fake_retry
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

        # 每个用例独立的缓存实例，且清空本地副本。
        # 文本带用例唯一前缀：即使开发机上跑着真实 Redis、里面还留着
        # 上一轮跑过的向量，key 也不会碰撞，因此用例不依赖外部状态。
        self.cache = EmbedCache()
        self.addAsyncCleanup(self.cache.clear)
        self._cache_patcher = patch(
            "app.services.bailian.embed_cache", self.cache
        )
        self._cache_patcher.start()
        self.addCleanup(self._cache_patcher.stop)

        # 每次运行都换前缀，杜绝跨进程残留。
        self.prefix = f"{self._testMethodName}-{uuid.uuid4().hex[:8]}-"

    def text(self, label: str) -> str:
        """返回本用例独占的文本。"""

        return f"{self.prefix}{label}"

    async def test_first_call_misses_then_second_hits(self):
        texts = [self.text("q1"), self.text("q2")]

        first = await bailian_service.embed(texts)
        self.assertEqual(self.call_count, 1)
        self.assertEqual(len(first), 2)

        before = self.call_count
        second = await bailian_service.embed(texts)

        self.assertEqual(self.call_count - before, 0, "相同文本不应再次调用模型")
        self.assertEqual(len(second), 2)
        for left, right in zip(first, second, strict=True):
            self.assertEqual(left, right)

    async def test_partial_hit_only_computes_missing(self):
        known = self.text("known")
        await bailian_service.embed([known])
        before = self.call_count

        await bailian_service.embed([known, self.text("fresh")])

        self.assertEqual(self.call_count - before, 1)
        self.assertEqual(self.seen_texts[-1], [self.text("fresh")])

    async def test_duplicate_text_within_one_batch_computed_once(self):
        text = self.text("dup")

        result = await bailian_service.embed([text, text, text])

        self.assertEqual(self.call_count, 1)
        self.assertEqual(self.seen_texts[-1], [text], "去重后才调用模型")
        self.assertEqual(len(result), 3, "返回数量必须与输入一致")

    async def test_result_order_matches_input_order(self):
        first = self.text("d1")
        second = self.text("d2")
        third = self.text("d3")

        await bailian_service.embed([first, second, third])
        # 逆序再取一次，验证是按 key 匹配而非按位置
        result = await bailian_service.embed([third, first, second])

        # 三个文本内容不同 -> 向量首元素分别是 1/2/3
        self.assertAlmostEqual(result[0][0], 3.0)
        self.assertAlmostEqual(result[1][0], 1.0)
        self.assertAlmostEqual(result[2][0], 2.0)

    async def test_empty_input_returns_empty(self):
        self.assertEqual(await bailian_service.embed([]), [])
        self.assertEqual(await bailian_service.embed(["", "   "]), [])
        self.assertEqual(self.call_count, 0)

    async def test_disabled_cache_always_calls_model(self):
        # 开关来自配置，patch 配置项比改只读 property 更贴近真实用法。
        with patch.object(settings, "EMBED_CACHE_ENABLED", False):
            text = self.text("same")
            await bailian_service.embed([text])
            await bailian_service.embed([text])

        self.assertEqual(self.call_count, 2)


class EmbedCacheFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Redis 异常时必须降级为直接调用，而不是抛错中断检索。"""

    async def test_redis_error_does_not_break_lookup(self):
        from redis.exceptions import RedisError

        cache = EmbedCache()

        # 只让底层 mget 抛错，验证 lookup 自己能兜住
        async def boom(*args, **kwargs):
            raise RedisError("redis down")

        with patch.object(settings, "EMBED_CACHE_ENABLED", True), \
                patch.object(cache, "initialize", return_value=None), \
                patch.object(cache.client, "mget", side_effect=boom):
            self.assertEqual(await cache.lookup(["x"]), {})
            # 熔断后应切到降级路径，再次调用依然不抛
            self.assertEqual(await cache.lookup(["x"]), {})

    async def test_store_failure_is_swallowed(self):
        """写缓存失败不能把已经拿到的向量丢掉。"""

        from redis.exceptions import RedisError

        cache = EmbedCache()

        class FailingPipe:
            """模拟 pipeline 在 execute 阶段失败（redis-py 为异步上下文管理器）。"""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def mset(self, payload):
                return None

            def expire(self, key, ttl):
                return None

            async def execute(self):
                raise RedisError("pipeline failed")

        with patch.object(settings, "EMBED_CACHE_ENABLED", True), \
                patch.object(cache, "initialize", return_value=None), \
                patch.object(cache.client, "pipeline", return_value=FailingPipe()):
            # 不抛异常即通过；向量已由调用方持有
            await cache.store(["a"], [[0.1] * 8])

        # 本地副本仍应保留，Redis 恢复后可继续命中
        self.assertEqual(len(cache._local), 1)

    async def test_clear_drops_local_copy(self):
        cache = EmbedCache()
        cache._local["k1"] = [0.1]

        await cache.clear()

        self.assertEqual(cache._local, {})


if __name__ == "__main__":
    unittest.main()