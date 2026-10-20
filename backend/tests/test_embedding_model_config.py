"""Embedding 模型与维度的兼容性约束。

背景：切换 Embedding 模型时，最容易踩的坑是维度与 Milvus Collection 不一致——
向量库写入会静默失败或报错，且报错信息里很少直接说"维度不匹配"。
"""
import unittest

from pydantic import ValidationError

from app.core.config import Settings


def _settings(**overrides) -> Settings:
    """构造一份 Settings 实例；测试只关心校验逻辑，不触碰外部服务。"""

    base = {
        "DASHSCOPE_API_KEY": "test-key",
        "SECRET_KEY": "test-secret-key-for-validation-only",
        "BAILIAN_API_KEY": "test-key",
        "SANDBOX_RUNNER_SOCKET": "/tmp/test-sandbox.sock",
    }
    base.update(overrides)
    return Settings(**base)


class EmbeddingModelDimensionTests(unittest.TestCase):
    def test_flash_model_accepts_default_dimension(self):
        settings = _settings(
            BAILIAN_EMBEDDING_MODEL="qwen3.7-text-embedding-flash",
            EMBEDDING_DIMENSION=1024,
        )

        self.assertEqual(settings.EMBEDDING_DIMENSION, 1024)

    def test_flash_model_accepts_small_dimensions(self):
        for dimension in (256, 512, 768):
            with self.subTest(dimension=dimension):
                settings = _settings(
                    BAILIAN_EMBEDDING_MODEL="qwen3.7-text-embedding-flash",
                    EMBEDDING_DIMENSION=dimension,
                )
                self.assertEqual(settings.EMBEDDING_DIMENSION, dimension)

    def test_flash_model_rejects_dimension_above_1024(self):
        """flash 版上限 1024；超了必须在启动时报错，而不是调用时才炸。"""

        for dimension in (1536, 2048, 2560):
            with self.subTest(dimension=dimension), self.assertRaises(ValidationError):
                _settings(
                    BAILIAN_EMBEDDING_MODEL="qwen3.7-text-embedding-flash",
                    EMBEDDING_DIMENSION=dimension,
                )

    def test_standard_model_allows_up_to_2560(self):
        for dimension in (1024, 1536, 2560):
            with self.subTest(dimension=dimension):
                settings = _settings(
                    BAILIAN_EMBEDDING_MODEL="qwen3.7-text-embedding",
                    EMBEDDING_DIMENSION=dimension,
                )
                self.assertEqual(settings.EMBEDDING_DIMENSION, dimension)

    def test_dimension_below_minimum_is_rejected(self):
        for dimension in (64, 128, 255):
            with self.subTest(dimension=dimension), self.assertRaises(ValidationError):
                _settings(EMBEDDING_DIMENSION=dimension)


class EmbedCacheKeyTests(unittest.TestCase):
    """缓存 key 必须区分模型，否则换模型后会命中旧模型的向量。"""

    def test_key_changes_with_model(self):
        from app.services.embed_cache import EmbedCache

        cache = EmbedCache()
        original = type(cache).enabled
        try:
            from app.core import config as config_module

            cache.key("同一段文本")  # 先取一次基线

            base_model = config_module.settings.BAILIAN_EMBEDDING_MODEL
            try:
                config_module.settings.BAILIAN_EMBEDDING_MODEL = "model-a"
                first = cache.key("同一段文本")
                config_module.settings.BAILIAN_EMBEDDING_MODEL = "model-b"
                second = cache.key("同一段文本")
            finally:
                config_module.settings.BAILIAN_EMBEDDING_MODEL = base_model

            self.assertNotEqual(
                first,
                second,
                "不同模型的缓存 key 必须不同，否则会返回另一模型的向量",
            )
        finally:
            type(cache).enabled = original

    def test_key_is_stable_for_same_config(self):
        from app.services.embed_cache import EmbedCache

        cache = EmbedCache()

        self.assertEqual(cache.key("abc"), cache.key("abc"))
        self.assertNotEqual(cache.key("abc"), cache.key("abd"))


class MilvusDimensionTests(unittest.TestCase):
    """Collection 的向量维度必须等于配置维度。"""

    def test_schema_uses_configured_dimension(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "services"
            / "milvus_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("dim=settings.EMBEDDING_DIMENSION", source)

    def test_embed_validates_dimension_mismatch(self):
        """bailian.embed 必须校验返回维度——这是最直观的错配信号。"""

        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "services"
            / "bailian.py"
        ).read_text(encoding="utf-8")

        self.assertIn("Embedding 向量维度不一致", source)
        self.assertIn("settings.EMBEDDING_DIMENSION", source)


if __name__ == "__main__":
    unittest.main()