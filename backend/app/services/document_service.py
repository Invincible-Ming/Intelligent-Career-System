"""
文档上传、解析、切块和向量入库服务。
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.bm25_service import bm25_service
from app.services.bailian import bailian_service
from app.core.config import settings
from app.services.milvus_service import (
    DOCUMENT_TYPES,
    milvus_service,
)
from app.services.minio_service import minio_service
from app.core.models import Document, DocumentParse
from app.services.document_pipeline import PIPELINE_VERSION, Block, build_chunks, serialize_blocks
from app.services.sandbox_client import parse_document

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".txt",
}


class DocumentService:
    """文档上传和知识库入库服务。"""

    async def upload_document(
            self,
            *,
            session: AsyncSession,
            owner_id: uuid.UUID,
            file: UploadFile,
            document_type: str,
    ) -> Document:
        """上传文件并完成解析和向量入库。"""

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
                "仅支持 PDF、DOCX、XLSX 和 TXT 文件"
            )

        data = await file.read(settings.max_upload_size_bytes + 1)

        if not data:
            raise ValueError("上传文件不能为空")

        if len(data) > settings.max_upload_size_bytes:
            raise ValueError(
                f"文件不能超过 {settings.MAX_UPLOAD_SIZE_MB} MB"
            )

        # ==========================================
        # ✨ 新增：Hash 校验与增量更新（秒传）逻辑
        # ==========================================
        file_hash = hashlib.sha256(data).hexdigest()

        # 查询数据库中是否已有该 Hash 且状态为已就绪的文档
        existing_doc_result = await session.execute(
            select(Document).where(
                Document.file_hash == file_hash,
                Document.owner_id == owner_id,
                Document.document_type == document_type,
                Document.status == "ready"
            )
        )
        existing_doc = existing_doc_result.scalars().first()

        if existing_doc:
            return existing_doc  # 直接返回已存在的文档记录，0 Token消耗，0 延迟！
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

            vectors = await embed_chunks(chunks)

            inserted_count = await milvus_service.insert(
                document_id=str(document_id),
                document_type=document_type,
                chunks=chunks,
                vectors=vectors,
            )

            await bm25_service.add_document(
                document_id=str(document_id),
                document_type=document_type,
                chunks=chunks,
            )

            document.status = "ready"
            session.add(DocumentParse(
                document_id=document_id, pipeline_version=PIPELINE_VERSION,
                text=text, blocks=serialize_blocks(blocks), chunks=structured_chunks,
            ))
            document.chunk_count = inserted_count
            document.error_message = None

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
                failed_document.error_message = (
                    str(exc) if isinstance(exc, ValueError) else "文档处理失败或中断，请重新上传"
                )
                await session.commit()

            # 清理可能已经写入的部分向量。
            try:
                await milvus_service.delete_document(
                    str(document_id)
                )
                await bm25_service.delete_document(
                    str(document_id)
                )
            except Exception:
                pass

            if not isinstance(exc, Exception):
                raise

            # 诊断日志：真实异常落盘，避免 500 时代码路径不可见。
            try:
                log_path = Path(".local/upload_errors.log")
                log_path.parent.mkdir(mode=0o700, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n=== {datetime.now().isoformat()} doc={document_id} ===\n")
                    log_file.write(traceback.format_exc())
            except Exception:
                pass

            # 业务类错误（大小超限、内容为空等）保留具体原因，供接口返回 400。
            if isinstance(exc, ValueError):
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
        """删除 PostgreSQL、Milvus 和 MinIO 中的文档。"""

        document = await session.get(
            Document,
            document_id,
        )

        if document is None or document.owner_id != owner_id:
            raise ValueError("文档不存在或不可访问")

        await milvus_service.delete_document(
            str(document_id)
        )

        await bm25_service.delete_document(
            str(document_id)
        )

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
