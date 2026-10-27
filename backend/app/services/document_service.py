"""
文档上传、解析与切块；仅 Milvus 模式执行向量入库。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.models import Document, DocumentParse, DocumentSearchIndex
from app.services.bailian import OcrServiceError, bailian_service
from app.services.bm25_service import bm25_service
from app.services.document_pipeline import PIPELINE_VERSION, Block, build_chunks, serialize_blocks
from app.services.milvus_service import (
    DOCUMENT_TYPES,
    milvus_service,
)
from app.services.minio_service import minio_service
from app.services.sandbox_client import SandboxUnavailable, parse_document

logger = logging.getLogger(__name__)


def record_upload_error(document_id: uuid.UUID, trace: str) -> None:
    """Write diagnostic files in a worker thread, not the request event loop."""
    log_path = Path(".local/upload_errors.log")
    log_path.parent.mkdir(mode=0o700, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== {datetime.now().isoformat()} doc={document_id} ===\n")
        log_file.write(trace)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".txt",
    ".md",
}


class DocumentConflict(RuntimeError):
    """A document is busy or its current state does not permit this action."""


class DocumentNotFound(ValueError):
    pass


class DocumentService:
    """文档上传和知识库入库服务。"""

    async def upload_document(
            self,
            *,
            session: AsyncSession,
            owner_id: uuid.UUID,
            file: UploadFile,
            document_type: str,
    ) -> tuple[Document, bool]:
        """上传并解析文件，按知识库配置决定是否生成向量。

        返回 (文档, 是否命中已有文档)。
        第二项为 True 表示内容哈希相同、已直接复用既有记录（未重复入库），
        调用方可据此提示用户而无需靠比对 id 猜测。
        """

        if document_type not in DOCUMENT_TYPES:
            raise ValueError(
                "document_type 必须是 "
                "resume、job_description 或 knowledge"
            )

        filename = Path(file.filename or "document.txt").name
        if len(filename) > 200 or any(ord(char) < 32 for char in filename):
            raise ValueError("文件名过长或包含非法字符")
        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(
                "仅支持 PDF、DOCX、XLSX、TXT 和 Markdown（.md）文件"
            )

        data = await file.read(settings.max_upload_size_bytes + 1)

        if not data:
            raise ValueError("上传文件不能为空")

        if len(data) > settings.max_upload_size_bytes:
            raise ValueError(
                f"文件不能超过 {settings.MAX_UPLOAD_SIZE_MB} MB"
            )

        # ==========================================
        # 整文件去重；变更文件完整重解析，相同分块通过 Embedding 缓存复用。
        # ==========================================
        file_hash = hashlib.sha256(data).hexdigest()

        # 查询数据库中是否已有该 Hash 且状态为已就绪的文档
        existing_doc_result = await session.execute(
            select(Document).join(DocumentParse, DocumentParse.document_id == Document.id).where(
                Document.file_hash == file_hash,
                # 相同字节的 TXT/Markdown 具有不同的章节解析方式。
                func.lower(Document.filename).endswith(extension),
                Document.owner_id == owner_id,
                Document.document_type == document_type,
                Document.status == "ready",
                DocumentParse.pipeline_version == PIPELINE_VERSION,
            )
        )
        existing_doc = existing_doc_result.scalars().first()

        if existing_doc:
            if settings.KNOWLEDGE_PROVIDER in ("subscription", "credits"):
                await self._queue_index(session, owner_id)
                await session.commit()
            # 内容哈希相同：直接复用既有记录，0 Token 消耗、0 重复入库。
            return existing_doc, True
        # ==========================================

        count = (await session.execute(
            select(func.count()).select_from(Document).where(Document.owner_id == owner_id))).scalar_one()
        if count >= settings.MAX_DOCUMENTS_PER_USER:
            raise ValueError("文档数量已达到限制，请删除不再使用的文档")

        document_id = uuid.uuid4()
        content_type = (
                file.content_type
                or mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
        )

        object_key = await minio_service.upload(
            document_id=str(document_id),
            filename=filename,
            data=data,
            content_type=content_type,
        )

        document = Document(
            owner_id=owner_id,
            id=document_id,
            filename=filename,
            document_type=document_type,
            minio_object_key=object_key,
            file_hash=file_hash,  # ✨ 存入计算好的 Hash 值
            status="processing",
        )

        session.add(document)
        await session.commit()

        await self._process_document(session=session, document=document, data=data)
        return document, False

    async def _locked_document(self, session, owner_id, document_id):
        # A retry holds this row through parsing. Reject concurrent edits/deletes
        # immediately instead of queueing another parse after the first finishes.
        try:
            document = (await session.execute(select(Document).where(
                Document.id == document_id, Document.owner_id == owner_id,
            ).with_for_update(nowait=True))).scalar_one_or_none()
        except DBAPIError as exc:
            await session.rollback()
            if getattr(exc.orig, "sqlstate", None) == "55P03":
                raise DocumentConflict("文件正在处理，请稍后重试") from None
            raise
        if document is None:
            raise DocumentNotFound("文档不存在或不可访问")
        if document.status == "processing":
            raise DocumentConflict("文件正在处理，请稍后重试")
        return document

    async def rename_document(self, *, session, owner_id, document_id, filename):
        document = await self._locked_document(session, owner_id, document_id)
        filename = filename.strip()
        if (not filename or len(filename) > 200 or filename in {".", ".."}
                or any(ord(char) < 32 or ord(char) == 127 for char in filename)
                or any(char in filename for char in '/\\<>:"|?*')):
            raise ValueError("文件名不能为空、超过 200 字或包含路径及特殊字符")
        if Path(filename).suffix.lower() != Path(document.filename).suffix.lower():
            raise ValueError("重命名不能修改文件扩展名")
        document.filename = filename
        await session.commit()
        return document

    async def retry_document(self, *, session, owner_id, document_id):
        document = await self._locked_document(session, owner_id, document_id)
        if document.status != "failed":
            raise DocumentConflict("仅支持重试处理失败的文件")
        data = await minio_service.download(document.minio_object_key)
        if not data or len(data) > settings.max_upload_size_bytes:
            raise ValueError("原文件为空或超过大小限制，请重新上传")
        if settings.KNOWLEDGE_PROVIDER == "milvus":
            await milvus_service.delete_document(str(document_id))
            await bm25_service.delete_document(str(document_id))
        await session.execute(delete(DocumentSearchIndex).where(DocumentSearchIndex.document_id == document_id))
        document.status = "processing"
        document.error_message = None
        return await self._process_document(session=session, document=document, data=data)

    async def _process_document(self, *, session, document, data):
        owner_id, document_id = document.owner_id, document.id
        document_type = document.document_type
        extension = Path(document.filename).suffix.lower()
        try:
            blocks = await parse_document(
                owner_id=owner_id, run_id=document_id, data=data, extension=extension,
                ocr=bailian_service.ocr_image, max_chars=settings.MAX_PARSED_DOCUMENT_CHARS)
            text = "\n\n".join(block.text for block in blocks)

            if not text.strip():
                raise ValueError(
                    "文档中没有提取到有效文本，且兜底策略未检测到内容。"
                )

            if len(text) > settings.MAX_PARSED_DOCUMENT_CHARS:
                raise ValueError("文档解析文本超限，请拆分文件")

            # 4. 智能路由分块
            structured_chunks = build_chunks(blocks, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            if len(structured_chunks) > settings.MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError("文档分块数量超限，请拆分文件")
            chunks = [chunk["content"] for chunk in structured_chunks]

            if not chunks:
                raise ValueError("文档切块结果为空")

            inserted_count = len(chunks)
            if settings.KNOWLEDGE_PROVIDER == "milvus":
                vectors = await embed_chunks(chunks)
                inserted_count = await milvus_service.insert(
                    document_id=str(document_id), owner_id=str(owner_id),
                    document_type=document_type, chunks=chunks, vectors=vectors,
                )
                await bm25_service.add_document(
                    document_id=str(document_id), owner_id=str(owner_id),
                    document_type=document_type, chunks=chunks,
                )

            document.status = "ready"
            parse_values = dict(pipeline_version=PIPELINE_VERSION, text=text,
                                blocks=serialize_blocks(blocks), chunks=structured_chunks)
            await session.execute(pg_insert(DocumentParse).values(
                document_id=document_id, **parse_values,
            ).on_conflict_do_update(index_elements=["document_id"], set_=parse_values))
            document.chunk_count = inserted_count
            document.error_message = None

            if settings.KNOWLEDGE_PROVIDER in ("subscription", "credits"):
                await session.flush()
                await self._queue_index(session, owner_id)

            await session.commit()
            await session.refresh(document)

            return document

        except BaseException as exc:
            await session.rollback()

            failed_document = await session.get(
                Document,
                document_id,
            )

            if failed_document is not None:
                failed_document.status = "failed"
                # 保留可操作的原因：ValueError 是文档本身的问题，
                # SandboxUnavailable 是「沙箱没启动」——两者都需要让用户看到。
                if isinstance(exc, (ValueError, SandboxUnavailable, OcrServiceError)):
                    failed_document.error_message = str(exc)
                else:
                    failed_document.error_message = "文档处理失败或中断，请重新上传"
                await session.commit()

            # 清理可能已经写入的部分向量。
            if settings.KNOWLEDGE_PROVIDER == "milvus":
                try:
                    await milvus_service.delete_document(str(document_id))
                    await bm25_service.delete_document(str(document_id))
                except Exception as cleanup_error:
                    logger.warning("文档索引清理失败（%s）", type(cleanup_error).__name__)

            if not isinstance(exc, Exception):
                raise

            # 诊断日志：真实异常落盘，避免 500 时代码路径不可见。
            try:
                await asyncio.to_thread(record_upload_error, document_id, traceback.format_exc())
            except Exception as log_error:
                logger.debug("上传诊断日志写入失败（%s）", type(log_error).__name__)

            # 业务类错误（大小超限、内容为空等）保留具体原因，供接口返回 400。
            if isinstance(exc, (ValueError, SandboxUnavailable, OcrServiceError)):
                raise

            raise RuntimeError(
                "文档处理失败，请稍后重试"
            ) from exc

    async def get_document_text(
            self,
            *,
            session: AsyncSession,
            owner_id: uuid.UUID,
            document_id: uuid.UUID,
    ) -> str:
        """从 MinIO 下载文档并重新解析文本。"""

        document = await session.get(
            Document,
            document_id,
        )

        if document is None or document.owner_id != owner_id:
            raise ValueError("文档不存在或不可访问")

        if document.status != "ready":
            raise ValueError(
                f"文档当前不可用，状态：{document.status}"
            )

        parsed = await session.get(DocumentParse, document_id)
        if parsed is not None:
            return parsed.text

        data = await minio_service.download(
            document.minio_object_key
        )

        extension = Path(
            document.filename
        ).suffix.lower()

        blocks = await parse_document(
            owner_id=owner_id, run_id=document_id, data=data, extension=extension,
            ocr=bailian_service.ocr_image, max_chars=settings.MAX_PARSED_DOCUMENT_CHARS)
        text = "\n\n".join(block.text for block in blocks)

        if not text.strip():
            raise ValueError("文档中没有有效文本")

        await session.execute(pg_insert(DocumentParse).values(
            document_id=document_id, pipeline_version=PIPELINE_VERSION,
            text=text, blocks=serialize_blocks(blocks), chunks=[],
        ).on_conflict_do_nothing(index_elements=["document_id"]))
        await session.commit()
        return text

    async def _queue_index(self, session, owner_id):
        from app.services.knowledge_index import knowledge_index_service
        from app.services.knowledge_service import knowledge_service

        if await knowledge_service.provider(session, owner_id) == "milvus":
            await knowledge_index_service.enqueue_owner(session, owner_id)

    async def list_documents(
            self,
            *,
            session: AsyncSession,
            owner_id: uuid.UUID,
    ) -> list[Document]:
        """查询全部文档。"""

        result = await session.execute(
            select(Document).where(Document.owner_id == owner_id).order_by(
                Document.created_at.desc()
            )
        )

        return list(result.scalars())

    async def delete_document(
            self,
            *,
            session: AsyncSession,
            owner_id: uuid.UUID,
            document_id: uuid.UUID,
    ) -> None:
        """删除本地文档及原文件；Milvus 模式同时删除向量。"""

        document = await self._locked_document(session, owner_id, document_id)

        if settings.KNOWLEDGE_PROVIDER == "milvus":
            await milvus_service.delete_document(str(document_id))
            await bm25_service.delete_document(str(document_id))

        if settings.KNOWLEDGE_PROVIDER in ("subscription", "credits"):
            # Serialize with an in-flight index worker before removing its vectors.
            await session.execute(select(Document).where(Document.id == document_id).with_for_update())
            state = await session.get(DocumentSearchIndex, document_id)
            if state is not None:
                try:
                    await milvus_service.delete_document(str(document_id))
                except Exception as exc:
                    # Retrieval rechecks both document ownership and live index state, excluding orphan vectors.
                    logger.warning("已删除文件的向量暂未清理（%s）", type(exc).__name__)
            await bm25_service.delete_document(str(document_id))

        await minio_service.delete(
            document.minio_object_key
        )

        await session.delete(document)
        await session.commit()


async def embed_chunks(
        chunks: list[str],
        batch_size: int = 10,
) -> list[list[float]]:
    """分批调用百炼 Embedding，避免单次输入过多。"""

    vectors: list[list[float]] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        batch_vectors = await bailian_service.embed(batch)
        vectors.extend(batch_vectors)

    return vectors


# ==========================================
# 文档解析层：重构 PDF 与 DOCX 解析机制
# ==========================================

def split_text(text: str, document_type: str = "knowledge") -> list[str]:
    """Compatibility entry point using the structure-aware chunker."""
    text = text.replace("\r\n", "\n").strip()
    blocks = [Block(p.strip()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [c["content"] for c in build_chunks(
        blocks, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    )]


document_service = DocumentService()
