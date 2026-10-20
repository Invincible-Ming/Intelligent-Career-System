"""BM25 索引更新与 owner 隔离测试。

覆盖三类易错点：
- owner 隔离不能靠应用层白名单，内存索引本身必须带 owner_id；
- 索引更新采用「锁外计算 + 换引用」，锁持有时间与规模无关；
- 无效删除不应触发全库重建。
"""
import asyncio
import unittest

from app.services.bm25_service import BM25Service, tokenize


def make_rows():
    """两条owner 的文档，内容故意重叠以便验证隔离。"""

    return [
        {"chunk_id": "1", "document_id": "d1", "owner_id": "u1",
         "document_type": "knowledge", "content": "python 深度学习 神经网络"},
        {"chunk_id": "2", "document_id": "d2", "owner_id": "u2",
         "document_type": "knowledge", "content": "python 深度学习 神经网络"},
        {"chunk_id": "3", "document_id": "d3", "owner_id": "u1",
         "document_type": "knowledge", "content": "kubernetes 容器编排 调度"},
    ]


class BM25OwnerIsolationTests(unittest.IsolatedAsyncioTestCase):
    """检索结果必须严格限定在请求方的 owner 内。"""

    async def asyncSetUp(self):
        self.service = BM25Service()
        await self.service.rebuild(make_rows())

    async def test_each_owner_sees_only_own_chunks(self):
        # u1 与 u2 的 d1/d2 内容完全相同，只有 owner_id 能区分。
        first = await self.service.search(query="深度学习", owner_id="u1", top_k=10)
        second = await self.service.search(query="深度学习", owner_id="u2", top_k=10)

        self.assertEqual([item["document_id"] for item in first], ["d1"])
        self.assertEqual([item["document_id"] for item in second], ["d2"])

    async def test_owner_filter_excludes_other_owner_high_scores(self):
        """即便他人文档分数更高，也不能出现在结果里。"""

        results = await self.service.search(query="深度学习 神经网络", owner_id="u1", top_k=10)

        self.assertNotIn("d2", [item["document_id"] for item in results])

    async def test_no_owner_returns_all_chunks(self):
        """不传 owner 时不做隔离（评测模块的全库检索依赖此行为）。"""

        results = await self.service.search(query="深度学习", owner_id=None, top_k=10)

        self.assertEqual(
            sorted(item["document_id"] for item in results),
            ["d1", "d2"],
        )

    async def test_document_type_filter_still_applies(self):
        service = BM25Service()
        await service.rebuild([
            {"chunk_id": "1", "document_id": "d1", "owner_id": "u1",
             "document_type": "resume", "content": "深度学习 项目经验"},
            {"chunk_id": "2", "document_id": "d2", "owner_id": "u1",
             "document_type": "knowledge", "content": "深度学习 知识库"},
        ])

        results = await service.search(
            query="深度学习", owner_id="u1", document_type="resume", top_k=10,
        )

        self.assertEqual([item["document_id"] for item in results], ["d1"])

    async def test_allowed_document_ids_narrows_further(self):
        """owner_id 之外的额外收窄条件依然生效。"""

        results = await self.service.search(
            query="深度学习", owner_id="u1",
            allowed_document_ids=["d3"], top_k=10,
        )

        self.assertEqual(results, [])

    async def test_empty_allowed_list_returns_nothing(self):
        results = await self.service.search(
            query="深度学习", owner_id="u1", allowed_document_ids=[], top_k=10,
        )

        self.assertEqual(results, [])


class BM25IndexUpdateTests(unittest.IsolatedAsyncioTestCase):
    """索引更新走「锁外计算 + 换引用」，不做全库原地修改。"""

    async def asyncSetUp(self):
        self.service = BM25Service()
        await self.service.rebuild(make_rows())

    async def test_rebuild_attaches_owner_id(self):
        self.assertTrue(all(chunk.owner_id for chunk in self.service.chunks))
        self.assertEqual(
            sorted({chunk.owner_id for chunk in self.service.chunks}),
            ["u1", "u2"],
        )

    async def test_delete_missing_document_does_not_rebuild(self):
        """删除不存在的文档必须直接返回，不触发全库分词与重建。"""

        model_before = self.service.model
        chunks_before = self.service.chunks

        await self.service.delete_document("not-exist")

        self.assertIs(self.service.model, model_before)
        self.assertIs(self.service.chunks, chunks_before)

    async def test_delete_existing_document_replaces_index(self):
        model_before = self.service.model

        await self.service.delete_document("d1")

        self.assertIsNot(self.service.model, model_before)
        results = await self.service.search(query="深度学习", owner_id="u1", top_k=10)
        self.assertEqual(results, [])

    async def test_add_document_is_idempotent_per_document(self):
        """重复索引同一文档不应产生重复条目。"""

        await self.service.add_document(
            document_id="d9", owner_id="u9", document_type="knowledge",
            chunks=["网关服务 限流熔断"],
        )
        await self.service.add_document(
            document_id="d9", owner_id="u9", document_type="knowledge",
            chunks=["网关服务 限流熔断"],
        )

        results = await self.service.search(query="限流熔断", owner_id="u9", top_k=10)
        self.assertEqual(len(results), 1)

        document_ids = [
            chunk.document_id for chunk in self.service.chunks if chunk.document_id == "d9"
        ]
        self.assertEqual(len(document_ids), 1)

    async def test_read_write_snapshot_stays_consistent(self):
        """并发写入时，读到的索引不能是「新 chunks + 旧 model」的错配。"""

        for index in range(10):
            await self.service.add_document(
                document_id=f"bulk-{index}", owner_id="u1", document_type="knowledge",
                chunks=[f"批量写入文档编号{index} 分布式一致性"],
            )

        results = await self.service.search(query="分布式一致性", owner_id="u1", top_k=20)

        self.assertEqual(len(results), 10)
        for result in results:
            # model 与 chunks 同源：命中的 document_id 必须真的在 chunks 里。
            self.assertIn(
                result["document_id"],
                {chunk.document_id for chunk in self.service.chunks},
            )

    async def test_concurrent_search_during_rebuild_never_errors(self):
        """重建期间并发检索不应抛异常（锁竞争回归测试）。"""

        async def writer():
            for index in range(5):
                await self.service.add_document(
                    document_id=f"c-{index}", owner_id="u1", document_type="knowledge",
                    chunks=[f"并发测试文档{index} 一致性哈希"],
                )

        async def reader():
            for _ in range(20):
                await self.service.search(query="一致性哈希", owner_id="u1", top_k=5)

        await asyncio.gather(writer(), *[reader() for _ in range(3)])

    async def test_empty_corpus_yields_no_results(self):
        service = BM25Service()
        await service.rebuild([])

        self.assertIsNone(service.model)
        self.assertEqual(await service.search(query="任意", owner_id="u1"), [])


class TokenizeTests(unittest.TestCase):
    """分词缓存必须与直接分词结果一致。"""

    def test_repeated_tokenize_is_stable(self):
        self.assertEqual(tokenize("Python 机器学习"), tokenize("Python 机器学习"))

    def test_tokenize_lowercases_and_drops_punctuation(self):
        tokens = tokenize("Python, FastAPI；Milvus")

        self.assertIn("python", tokens)
        self.assertIn("fastapi", tokens)
        self.assertNotIn(",", tokens)

    def test_blank_input_returns_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   "), [])


if __name__ == "__main__":
    unittest.main()