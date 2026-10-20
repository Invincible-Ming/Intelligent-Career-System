"""Bind an experiment to its owner's ready documents and existing chunk layout."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.core.models import Document, DocumentParse
from app.evaluation.config import ExperimentConfig
from app.services.document_pipeline import PIPELINE_VERSION, Block, build_chunks


def normalize_owner_id(owner_id: str) -> str:
    try:
        return str(uuid.UUID(str(owner_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("评测必须提供有效的用户 UUID，不能执行无用户范围的检索") from exc


@dataclass(frozen=True)
class EvaluationScope:
    owner_id: str
    document_ids: tuple[str, ...]


def _validate_chunk_layout(rows: list[Any], config: ExperimentConfig) -> None:
    layouts = {
        (item.search.chunk_size, item.search.chunk_overlap)
        for item in [config.baseline, *config.variants]
    }
    for document_id, pipeline_version, raw_blocks, stored_chunks in rows:
        if pipeline_version != PIPELINE_VERSION or not raw_blocks or not stored_chunks:
            raise ValueError(
                f"文档 {document_id} 缺少当前版本的分块记录，请重新上传后评测"
            )
        try:
            blocks = [Block(**block) for block in raw_blocks]
            existing = [chunk["content"] for chunk in stored_chunks]
            for size, overlap in layouts:
                rebuilt = [chunk["content"] for chunk in build_chunks(blocks, size, overlap)]
                if rebuilt != existing:
                    raise ValueError(
                        f"chunk_size={size}、chunk_overlap={overlap} 与文档 {document_id} "
                        "的现有分块不一致。评测使用既有索引；切块对照需要先建立对应的独立索引，"
                        "不能仅修改查询参数。"
                    )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"文档 {document_id} 的分块记录无效，请重新上传后评测") from exc


async def prepare_evaluation_scope(
    *, owner_id: str, config: ExperimentConfig,
) -> EvaluationScope:
    owner_id = normalize_owner_id(owner_id)
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(
                Document.id, DocumentParse.pipeline_version,
                DocumentParse.blocks, DocumentParse.chunks,
            )
            .outerjoin(DocumentParse, DocumentParse.document_id == Document.id)
            .where(Document.owner_id == uuid.UUID(owner_id), Document.status == "ready")
            .order_by(Document.id)
        )).all()
    if not rows:
        raise ValueError("当前用户没有已就绪的文档，请先上传文档再启动评测")
    await asyncio.to_thread(_validate_chunk_layout, rows, config)
    return EvaluationScope(owner_id, tuple(str(row[0]) for row in rows))
