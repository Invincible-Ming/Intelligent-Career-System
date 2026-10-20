"""
Milvus 向量存储服务。

Collection 保存：

- chunk_id：文本块 ID
- document_id：文档 ID
- owner_id：文档所有者 UUID（数据隔离维度）
- document_type：文档类型
- content：文本块原文
- vector：文本向量

owner_id 建INVERTED 标量索引，检索隔离在 Milvus 内完成，
不再由应用层拼 `document_id in [...]` 过滤表达式。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from pymilvus import DataType, MilvusClient

from app.core.config import settings

logger = logging.getLogger(__name__)

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

        self._client_options = client_options
        self._client = None
        self.collection = settings.MILVUS_COLLECTION

    @property
    def client(self):
        if self._client is None:
            self._client = MilvusClient(**self._client_options)
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

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
        else:
            await self._ensure_owner_index()

        await asyncio.to_thread(
            self.client.load_collection,
            collection_name=self.collection,
        )

    def _describe_fields(self) -> set[str]:
        """返回当前 Collection 已定义的字段名。"""

        info = self.client.describe_collection(
            collection_name=self.collection
        )
        return {
            str(field.get("name"))
            for field in info.get("fields", [])
        }

    async def _ensure_owner_index(self) -> None:
        """
        校验既有 Collection 是否具备 owner_id 字段与标量索引。

        schema 设了 enable_dynamic_field=False，无法给存量表补字段；
        字段缺失时直接拒绝启动，避免在无隔离的索引上继续提供检索。
        """

        fields = await asyncio.to_thread(self._describe_fields)

        if "owner_id" not in fields:
            raise RuntimeError(
                f"Milvus collection {self.collection} 缺少 owner_id 字段，"
                "无法保证多用户数据隔离。请先执行 "
                "python scripts/migrate_milvus_owner.py 完成迁移。"
            )

        indexes = await asyncio.to_thread(
            self.client.list_indexes,
            collection_name=self.collection,
        )

        if "owner_id_index" not in indexes:
            await asyncio.to_thread(
                self.client.create_index,
                collection_name=self.collection,
                field_name="owner_id",
                index_name="owner_id_index",
                index_type="INVERTED",
            )
            logger.info("已补建 owner_id 标量索引")

    def _create_collection(self) -> None:
        """创建 Collection Schema、向量 HNSW 索引与 owner_id 标量索引。"""

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
            field_name="owner_id",
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
        index_params.add_index(
            field_name="owner_id",
            index_name="owner_id_index",
            index_type="INVERTED",
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
            owner_id: str,
            document_type: str,
            chunks: list[str],
            vectors: list[list[float]],
    ) -> int:
        """批量写入文本块及对应向量。"""

        validate_document_id(document_id)
        validate_owner_id(owner_id)
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
            if len(content.encode("utf-8")) > 8192:
                raise ValueError("文本块超过 Milvus content 的 8192 字节限制，请降低 CHUNK_SIZE")

            rows.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "owner_id": owner_id,
                    "document_type": document_type,
                    "content": content,
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

    def _build_filter(
            self,
            *,
            document_type: str | None,
            owner_id: str | None,
            allowed_document_ids: list[str] | None,
    ) -> str:
        """拼装检索过滤表达式。

        owner_id 是主隔离维度，必须在 Milvus 内以标量索引过滤；
        allowed_document_ids 仅作为额外的收窄条件，不再作为隔离手段。
        """

        clauses: list[str] = []

        if owner_id:
            validate_owner_id(owner_id)
            clauses.append(
                f'owner_id == "{owner_id}"'
            )

        if document_type:
            validate_document_type(document_type)
            clauses.append(
                f'document_type == "{document_type}"'
            )

        if allowed_document_ids is not None:
            if not allowed_document_ids:
                return "__impossible__"
            for identifier in allowed_document_ids:
                validate_document_id(identifier)
            clauses.append(
                "document_id in " + json.dumps(allowed_document_ids)
            )

        if not clauses:
            return ""

        return " and ".join(clauses)

    async def search(
            self,
            *,
            query_vector: list[float],
            owner_id: str | None = None,
            document_type: str | None = None,
            top_k: int | None = None,
            allowed_document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """搜索与查询向量最相近的文本块。"""

        validate_vector(query_vector)

        filter_expression = self._build_filter(
            document_type=document_type,
            owner_id=owner_id,
            allowed_document_ids=allowed_document_ids,
        )

        if filter_expression == "__impossible__":
            return []

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
            *,
            owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        读取 Milvus 中的全部文本块。

        用于应用启动时重建 BM25 内存索引，以及迁移脚本回填 owner_id。
        当前个人项目最多读取 10000 个 Chunk。
        """

        filter_expression = 'chunk_id != ""'
        if owner_id is not None:
            validate_owner_id(owner_id)
            filter_expression += f' and owner_id == "{owner_id}"'
        rows = await asyncio.to_thread(
            self.client.query,
            collection_name=self.collection,
            filter=filter_expression,
            output_fields=[
                "chunk_id",
                "document_id",
                "owner_id",
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
                "owner_id": str(
                    row.get("owner_id", "")
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

    async def upsert_chunks(
            self,
            *,
            document_id: str,
            owner_id: str,
            document_type: str,
            chunks: list[str],
            vectors: list[list[float]],
    ) -> int:
        """迁移专用：按 document_id 整体覆盖写入。

        与 insert 的区别是会先删掉该文档的旧向量，避免迁移重跑产生重复。
        """

        await self.delete_document(document_id)
        return await self.insert(
            document_id=document_id,
            owner_id=owner_id,
            document_type=document_type,
            chunks=chunks,
            vectors=vectors,
        )

    async def health_check(self) -> bool:
        """检查 Milvus 是否可以访问。"""

        try:
            collections = await asyncio.to_thread(
                self.client.list_collections,
            )
            available = self.collection in collections

            if not available:
                logger.error(
                    "Milvus 连接正常但缺少 collection：%s",
                    self.collection,
                )

            return available
        except Exception as exc:
            # 静默返回 False 会让故障定位失去线索，必须留异常类型。
            logger.error("Milvus 健康检查失败：%s", type(exc).__name__)
            return False

    async def close(self) -> None:
        """关闭 Milvus 客户端。"""

        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None


def validate_document_id(document_id: str) -> None:
    """校验文档 UUID，防止拼接非法过滤表达式。"""

    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise ValueError(
            f"无效的 document_id：{document_id}"
        ) from exc


def validate_owner_id(owner_id: str) -> None:
    """校验 owner UUID，防止拼接非法过滤表达式。"""

    try:
        uuid.UUID(owner_id)
    except ValueError as exc:
        raise ValueError(
            f"无效的 owner_id：{owner_id}"
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
