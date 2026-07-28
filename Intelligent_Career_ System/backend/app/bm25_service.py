"""
轻量 BM25 关键词检索服务。

BM25 索引保存在应用内存中：

- 应用启动时从 Milvus 读取 Chunk 并重建索引；
- 上传文档时更新索引；
- 删除文档时移除对应 Chunk。

适合个人项目和中小型知识库。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import jieba
from rank_bm25 import BM25Okapi


@dataclass(slots=True)
class BM25Chunk:
    """BM25 索引中的文本块。"""

    chunk_id: str
    document_id: str
    document_type: str
    content: str


class BM25Service:
    """BM25 中文关键词检索服务。"""

    def __init__(self) -> None:
        self.chunks: list[BM25Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.token_sets: list[set[str]] = []
        self.model: BM25Okapi | None = None
        self.lock = asyncio.Lock()

    async def rebuild(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """使用全部 Milvus Chunk 重建 BM25 索引。"""

        async with self.lock:
            self.chunks = [
                BM25Chunk(
                    chunk_id=str(
                        row.get("chunk_id", "")
                    ),
                    document_id=str(
                        row.get("document_id", "")
                    ),
                    document_type=str(
                        row.get("document_type", "")
                    ),
                    content=str(
                        row.get("content", "")
                    ),
                )
                for row in rows
                if row.get("content")
            ]

            self._build_model()

    async def add_document(
        self,
        *,
        document_id: str,
        document_type: str,
        chunks: list[str],
    ) -> None:
        """将一个文档的文本块加入 BM25 索引。"""

        async with self.lock:
            # 避免重复上传或重复索引。
            self.chunks = [
                item
                for item in self.chunks
                if item.document_id != document_id
            ]

            for index, content in enumerate(chunks):
                content = content.strip()

                if not content:
                    continue

                self.chunks.append(
                    BM25Chunk(
                        chunk_id=(
                            f"{document_id}-{index}"
                        ),
                        document_id=document_id,
                        document_type=document_type,
                        content=content,
                    )
                )

            self._build_model()

    async def delete_document(
        self,
        document_id: str,
    ) -> None:
        """从 BM25 索引中删除一个文档。"""

        async with self.lock:
            self.chunks = [
                item
                for item in self.chunks
                if item.document_id != document_id
            ]

            self._build_model()

    async def search(
        self,
        *,
        query: str,
        document_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """执行 BM25 关键词检索。"""

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        async with self.lock:
            if self.model is None or not self.chunks:
                return []

            scores = self.model.get_scores(query_tokens)
            query_token_set = set(query_tokens)

            candidates: list[
                tuple[float, BM25Chunk]
            ] = []

            for index, chunk in enumerate(self.chunks):
                if (
                    document_type
                    and chunk.document_type
                    != document_type
                ):
                    continue

                # 至少存在一个关键词交集，避免返回完全无关内容。
                if not (
                    query_token_set
                    & self.token_sets[index]
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

    def _build_model(self) -> None:
        """根据当前 Chunk 列表重新构建 BM25 模型。"""

        self.tokenized_corpus = [
            tokenize(item.content)
            for item in self.chunks
        ]

        self.token_sets = [
            set(tokens)
            for tokens in self.tokenized_corpus
        ]

        if not self.tokenized_corpus:
            self.model = None
            return

        self.model = BM25Okapi(
            self.tokenized_corpus
        )

    @property
    def chunk_count(self) -> int:
        """返回当前索引中的 Chunk 数量。"""

        return len(self.chunks)


def tokenize(text: str) -> list[str]:
    """
    对中文和英文技术文本进行简单分词。

    jieba 能处理中文，英文技术词统一转成小写。
    """

    text = text.lower().strip()

    if not text:
        return []

    tokens = jieba.lcut(
        text,
        cut_all=False,
    )

    return [
        token.strip()
        for token in tokens
        if token.strip()
        and re.search(
            r"[a-z0-9\u4e00-\u9fff]",
            token,
        )
    ]


bm25_service = BM25Service()