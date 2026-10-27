"""
轻量 BM25 关键词检索服务。

BM25 索引保存在应用内存中：

- 应用启动时从 Milvus 读取 Chunk 并重建索引；
- 上传文档时更新语料并重建索引；
- 删除文档时移除对应 Chunk 并重建索引。

写入操作串行构建并发布新索引，检索只短暂持有快照锁。
锁与索引均在当前进程内；多进程部署需要额外的索引同步机制。

每个 Chunk 携带 owner_id，检索隔离在内存中按 owner 过滤，
不再依赖应用层传入 document_id 白名单。

适合个人项目和中小型知识库。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from dataclasses import dataclass
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# BM25 专用线程池。
#
# 与默认线程池分开，避免分词、建模和 Embedding、Reranker 任务
# 竞争同一线程池；线程池用于移出事件循环，不保证纯 Python CPU 并行。
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="bm25",
)


async def _run_in_pool(func, *args):
    """在 BM25 专用池里执行 CPU 任务。"""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, func, *args)


@dataclass(slots=True)
class BM25Chunk:
    """BM25 索引中的文本块。"""

    chunk_id: str
    document_id: str
    owner_id: str
    document_type: str
    content: str


# 分词结果缓存：同一文档重建索引时 jieba 结果可直接复用。
_TOKEN_CACHE: dict[str, list[str]] = {}
_TOKEN_CACHE_LIMIT = 20000


class BM25Service:
    """BM25 中文关键词检索服务。"""

    def __init__(self) -> None:
        self.chunks: list[BM25Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.token_sets: list[set[str]] = []
        self.model: BM25Okapi | None = None
        # 写入锁覆盖读取基线、构建和发布，防止并发写入丢失更新。
        self._write_lock = asyncio.Lock()
        # 快照锁只保护读者取引用和发布；构建期间读者继续使用旧索引。
        # 同时持锁时固定先取 _write_lock，再取 lock；检索不取写入锁。
        self.lock = asyncio.Lock()

    async def rebuild(
            self,
            rows: list[dict[str, Any]],
    ) -> None:
        """使用完整语料替换索引，供启动或独立维护进程使用。

        rows 是全量替换快照，不与现有语料合并；在线更新使用 add/delete。
        """

        async with self._write_lock:
            chunks = [
                BM25Chunk(
                    chunk_id=str(row.get("chunk_id", "")),
                    document_id=str(row.get("document_id", "")),
                    owner_id=str(row.get("owner_id", "")),
                    document_type=str(row.get("document_type", "")),
                    content=str(row.get("content", "")),
                )
                for row in rows
                if row.get("content")
            ]
            await self._swap_in(chunks)

    async def add_document(
            self,
            *,
            document_id: str,
            owner_id: str,
            document_type: str,
            chunks: list[str],
    ) -> None:
        """将一个文档的文本块加入 BM25 索引。"""

        async with self._write_lock:
            async with self.lock:
                snapshot = [
                    item
                    for item in self.chunks
                    if item.document_id != document_id
                ]

            for index, content in enumerate(chunks):
                content = content.strip()

                if not content:
                    continue

                snapshot.append(
                    BM25Chunk(
                        chunk_id=f"{document_id}-{index}",
                        document_id=document_id,
                        owner_id=owner_id,
                        document_type=document_type,
                        content=content,
                    )
                )

            await self._swap_in(snapshot)

    async def replace_owner(self, owner_id: str, documents: list[dict]) -> None:
        """Refresh one user's process-local snapshot without disturbing other users."""
        async with self._write_lock:
            async with self.lock:
                snapshot = [item for item in self.chunks if item.owner_id != owner_id]
            for document in documents:
                for index, content in enumerate(document["chunks"]):
                    if content.strip():
                        snapshot.append(BM25Chunk(chunk_id=f"{document['document_id']}-{index}",
                                                  document_id=document["document_id"], owner_id=owner_id,
                                                  document_type=document["document_type"], content=content.strip()))
            await self._swap_in(snapshot)

    async def delete_document(
            self,
            document_id: str,
    ) -> None:
        """从 BM25 索引中删除一个文档。"""

        async with self._write_lock:
            async with self.lock:
                snapshot = [
                    item
                    for item in self.chunks
                    if item.document_id != document_id
                ]
                had_document = len(snapshot) != len(self.chunks)

            if not had_document:
                return

            await self._swap_in(snapshot)

    async def _swap_in(
            self,
            chunks: list[BM25Chunk],
    ) -> None:
        """
        在线程池中构建新模型，构建完成后整体替换引用。

        调用方必须在读取更新基线前取得 _write_lock，并持有到发布结束。
        分词与建模不占用快照锁，因此检索可继续使用完整旧索引。
        所有新对象构建完成后才发布；失败或取消不会发布半成品。
        """

        tokenized = await _run_in_pool(
            _tokenize_all,
            [item.content for item in chunks],
        )

        model = await _run_in_pool(
            _build_model,
            tokenized,
        )

        token_sets = [
            set(tokens)
            for tokens in tokenized
        ]

        # 在同一快照锁内发布全部引用，防止 chunks 与模型版本错配。
        async with self.lock:
            self.chunks = chunks
            self.tokenized_corpus = tokenized
            self.token_sets = token_sets
            self.model = model

        logger.info(
            "BM25 索引已更新，共%s 个 Chunk",
            len(chunks),
        )

    async def search(
            self,
            *,
            query: str,
            owner_id: str | None = None,
            document_type: str | None = None,
            top_k: int = 5,
            allowed_document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """执行 BM25 关键词检索。

        owner_id 是主隔离维度；allowed_document_ids 仅作为额外收窄条件。
        """

        allowed = (
            set(allowed_document_ids)
            if allowed_document_ids is not None
            else None
        )
        if allowed is not None and not allowed:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # 快照引用，避免与重建竞争；重建完成即整体换新，无需在此持锁。
        async with self.lock:
            model = self.model
            chunks = self.chunks
            token_sets = self.token_sets

        if model is None or not chunks:
            return []

        # get_scores 在 5000 chunk 下也只有零点几毫秒，同步执行比提交线程池
        # 更划算；这里已经拿了快照引用，不持锁，不阻塞写入。
        scores = model.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        candidates: list[
            tuple[float, BM25Chunk]
        ] = []

        for index, chunk in enumerate(chunks):
            if owner_id and chunk.owner_id != owner_id:
                continue
            if allowed is not None and chunk.document_id not in allowed:
                continue
            if (
                    document_type
                    and chunk.document_type
                    != document_type
            ):
                continue

            # 至少存在一个关键词交集，避免返回完全无关内容。
            if not (
                    query_token_set
                    & token_sets[index]
            ):
                continue

            candidates.append(
                (
                    float(scores[index]),
                    chunk,
                )
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            {
                "document_id": chunk.document_id,
                "document_type": (
                    chunk.document_type
                ),
                "content": chunk.content,
                "score": round(score, 6),
                "source": "bm25",
            }
            for score, chunk in candidates[:top_k]
        ]

    @property
    def chunk_count(self) -> int:
        """返回当前索引中的 Chunk 数量。"""

        return len(self.chunks)


def _build_model(
        tokenized: list[list[str]],
) -> BM25Okapi | None:
    """按分词结果构造 BM25 模型（同步，供线程池调用）。"""

    if not tokenized:
        return None

    return BM25Okapi(tokenized)


def _tokenize_all(
        contents: list[str],
) -> list[list[str]]:
    """批量分词并复用缓存。"""

    return [
        tokenize(content)
        for content in contents
    ]


def tokenize(text: str) -> list[str]:
    """
    对中文和英文技术文本进行简单分词。

    jieba 能处理中文，英文技术词统一转成小写。
    结果按文本缓存，避免重建索引时重复分词。
    """

    text = text.lower().strip()

    if not text:
        return []

    cached = _TOKEN_CACHE.get(text)
    if cached is not None:
        return cached

    tokens = [
        token.strip()
        for token in jieba.lcut(
            text,
            cut_all=False,
        )
        if token.strip()
           and re.search(
            r"[a-z0-9\u4e00-\u9fff]",
            token,
        )
    ]

    if len(_TOKEN_CACHE) >= _TOKEN_CACHE_LIMIT:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[text] = tokens

    return tokens


bm25_service = BM25Service()
