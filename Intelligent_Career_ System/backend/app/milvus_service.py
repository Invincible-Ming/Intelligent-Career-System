"""
Milvus 向量存储服务。

Collection 保存：

- chunk_id：文本块 ID
- document_id：文档 ID
- document_type：文档类型
- content：文本块原文
- vector：文本向量
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from pymilvus import DataType, MilvusClient

from app.config import settings

DOCUMENT_TYPES = {
    "resume",
    "job_description",
    "knowledge",
}


class MilvusService:
    """Milvus 文档向量存储服务。"""

    def __init__(self) -> None:
        client_options: dict[str, Any] = {
            "uri": settings.MILVUS_URI,
        }

        if settings.MILVUS_TOKEN:
            client_options["token"] = settings.MILVUS_TOKEN

        self.client = MilvusClient(**client_options)
        self.collection = settings.MILVUS_COLLECTION

    async def initialize(self) -> None:
        """创建并加载 Milvus Collection。"""

        exists = await asyncio.to_thread(
            self.client.has_collection,
            collection_name=self.collection,
        )

        if not exists:
            await asyncio.to_thread(
                self._create_collection,
            )

        await asyncio.to_thread(
            self.client.load_collection,
            collection_name=self.collection,
        )

    def _create_collection(self) -> None:
        """创建 Collection Schema 和 HNSW 索引。"""

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="document_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="document_type",
            datatype=DataType.VARCHAR,
            max_length=50,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=8192,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=settings.EMBEDDING_DIMENSION,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M": 16,
                "efConstruction": 200,
            },
        )

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    async def insert(
            self,
            *,
            document_id: str,
            document_type: str,
            chunks: list[str],
            vectors: list[list[float]],
    ) -> int:
        """批量写入文本块及对应向量。"""

        validate_document_id(document_id)
        validate_document_type(document_type)

        if len(chunks) != len(vectors):
            raise ValueError("文本块数量与向量数量不一致")

        if not chunks:
            return 0

        rows: list[dict[str, Any]] = []

        for content, vector in zip(
                chunks,
                vectors,
                strict=True,
        ):
            validate_vector(vector)

            rows.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "document_type": document_type,
                    "content": content[:8192],
                    "vector": vector,
                }
            )

        result = await asyncio.to_thread(
            self.client.insert,
            collection_name=self.collection,
            data=rows,
        )

        return int(
            result.get("insert_count", len(rows))
        )

    async def search(
            self,
            *,
            query_vector: list[float],
            document_type: str | None = None,
            top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """搜索与查询向量最相近的文本块。"""

        validate_vector(query_vector)

        filter_expression = ""

        if document_type:
            validate_document_type(document_type)
            filter_expression = (
                f'document_type == "{document_type}"'
            )

        result = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection,
            data=[query_vector],
            anns_field="vector",
            filter=filter_expression,
            limit=top_k or settings.RETRIEVAL_TOP_K,
            output_fields=[
                "document_id",
                "document_type",
                "content",
            ],
            search_params={
                "metric_type": "COSINE",
                "params": {
                    "ef": 64,
                },
            },
        )

        if not result:
            return []

        search_results: list[dict[str, Any]] = []

        for hit in result[0]:
            entity = hit.get("entity", {})

            search_results.append(
                {
                    "document_id": entity.get(
                        "document_id",
                        "",
                    ),
                    "document_type": entity.get(
                        "document_type",
                        "",
                    ),
                    "content": entity.get(
                        "content",
                        "",
                    ),
                    "score": float(
                        hit.get("distance", 0)
                    ),
                    "source": "dense",
                }
            )

        return search_results

    async def delete_document(
            self,
            document_id: str,
    ) -> int:
        """删除指定文档的全部向量。"""

        validate_document_id(document_id)

        result = await asyncio.to_thread(
            self.client.delete,
            collection_name=self.collection,
            filter=f'document_id == "{document_id}"',
        )

        return int(result.get("delete_count", 0))

    async def list_chunks(
        self,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """
        读取 Milvus 中的全部文本块。

        用于应用启动时重建 BM25 内存索引。
        当前个人项目最多读取 10000 个 Chunk。
        """

        rows = await asyncio.to_thread(
            self.client.query,
            collection_name=self.collection,
            filter='chunk_id != ""',
            output_fields=[
                "chunk_id",
                "document_id",
                "document_type",
                "content",
            ],
            limit=limit,
        )

        return [
            {
                "chunk_id": str(
                    row.get("chunk_id", "")
                ),
                "document_id": str(
                    row.get("document_id", "")
                ),
                "document_type": str(
                    row.get("document_type", "")
                ),
                "content": str(
                    row.get("content", "")
                ),
            }
            for row in rows
            if row.get("content")
        ]

    async def health_check(self) -> bool:
        """检查 Milvus 是否可以访问。"""

        try:
            collections = await asyncio.to_thread(
                self.client.list_collections,
            )
            return self.collection in collections
        except Exception:
            return False

    async def close(self) -> None:
        """关闭 Milvus 客户端。"""

        await asyncio.to_thread(self.client.close)


def validate_document_id(document_id: str) -> None:
    """校验文档 UUID，防止拼接非法过滤表达式。"""

    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise ValueError(
            f"无效的 document_id：{document_id}"
        ) from exc


def validate_document_type(
        document_type: str,
) -> None:
    """校验文档业务类型。"""

    if document_type not in DOCUMENT_TYPES:
        raise ValueError(
            "document_type 必须是 "
            "resume、job_description 或 knowledge"
        )


def validate_vector(
        vector: list[float],
) -> None:
    """校验向量维度。"""

    if len(vector) != settings.EMBEDDING_DIMENSION:
        raise ValueError(
            "向量维度不正确："
            f"期望 {settings.EMBEDDING_DIMENSION}，"
            f"实际 {len(vector)}"
        )


milvus_service = MilvusService()
