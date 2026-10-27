"""Evaluate one owned, already-indexed document without including legacy files.

This is the same offline maintenance mode as app.evaluation.cli: provider usage
is billable at the provider, but no application credit wallet is debited.
"""
from __future__ import annotations

# SDK configuration must happen before importing application services.
# ruff: noqa: E402
import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
# RAGAS 0.3.9 imports VertexAI classes removed by community 0.4.x. Keep the
# compatible optional package local to this maintenance process, not the app.
compat = BACKEND / ".local" / "ragas_compat"
if compat.is_dir():
    sys.path.insert(0, str(compat))
os.chdir(BACKEND)
for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "DASHSCOPE_BASE_URL"):
    os.environ.pop(key, None)
os.environ["NO_PROXY"] = "*"

from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal, close_database
from app.core.models import Document, DocumentParse, DocumentSearchIndex
from app.core.prompts import PROMPT_VERSIONS
from app.evaluation.checkpoint import EvaluationCheckpoint, fingerprint
from app.evaluation.config import ExperimentConfig
from app.evaluation.metrics import RAGAS_VERSION, ragas_metrics
from app.evaluation.llm import evaluation_llm
from app.evaluation.runner import EvaluationRunner
from app.evaluation.scope import EvaluationScope, _validate_chunk_layout
from app.services.bailian import bailian_service
from app.services.bm25_service import bm25_service
from app.services.knowledge_index import knowledge_index_service
from app.services.milvus_service import milvus_service
from app.services.redis_pool import close_pool


async def run(args):
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("app.evaluation").setLevel(logging.INFO)
    logging.getLogger("jieba").setLevel(logging.WARNING)
    try:
        evaluation_llm.validate_configuration()
        ragas_metrics.ensure_available()
        config = ExperimentConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if args.timeout_seconds is not None:
            config.timeout_seconds = args.timeout_seconds
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(Document, DocumentParse, DocumentSearchIndex).join(
                DocumentParse, DocumentParse.document_id == Document.id,
            ).join(DocumentSearchIndex, DocumentSearchIndex.document_id == Document.id).where(
                Document.id == args.document_id, Document.owner_id == args.owner_id, Document.status == "ready",
            ))).one_or_none()
            if row is None:
                raise ValueError("未找到该用户已解析并建立索引的文档")
            document, parsed, index = row
            if index.status != "ready" or index.signature != knowledge_index_service.signature():
                raise ValueError("该文件索引未就绪或模型配置已变化，请先在知识库重新准备索引")
            _validate_chunk_layout([(document.id, parsed.pipeline_version, parsed.blocks, parsed.chunks)], config)
            expected = Counter(chunk["content"] for chunk in parsed.chunks)
            filename = document.filename

        source = Path(config.test_dataset_path).resolve()
        dataset = json.loads(source.read_text(encoding="utf-8"))
        cases = dataset["test_cases"]
        if args.cases > len(cases):
            raise ValueError(f"数据集只有 {len(cases)} 条")
        selected = cases[:args.cases]
        for case in selected:
            names = {item.get("filename") for item in case.get("metadata", {}).get("sources", [])}
            if names and names != {filename}:
                raise ValueError("题目来源与指定文档不一致，请选择匹配的数据集")

        await milvus_service.initialize()
        rows = [item for item in await milvus_service.list_chunks(owner_id=str(args.owner_id))
                if item["document_id"] == str(args.document_id)]
        if Counter(item["content"] for item in rows) != expected:
            raise ValueError("向量库与当前解析分块不一致，请先重新准备该文件索引")
        await bm25_service.rebuild(rows)
        manifest = {
            "owner_id": str(args.owner_id), "document_id": str(args.document_id),
            "corpus": fingerprint(sorted(rows, key=lambda row: row["chunk_id"])),
            "source_dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "configs": [item.model_dump() for item in [config.baseline, *config.variants]],
            "provider": evaluation_llm.provider, "model": evaluation_llm.model,
            "endpoint": fingerprint(settings.MIMO_BASE_URL if evaluation_llm.provider == "mimo"
                                    else settings.BAILIAN_BASE_URL),
            "embedding": [settings.BAILIAN_EMBEDDING_MODEL, settings.EMBEDDING_DIMENSION],
            "reranker": [settings.RERANKER_MODEL, settings.RERANKER_MAX_LENGTH],
            "prompts": PROMPT_VERSIONS, "ragas": RAGAS_VERSION,
            "output_caps": [settings.MODEL_MAX_OUTPUT_TOKENS, settings.EVALUATION_METRIC_MAX_OUTPUT_TOKENS],
            "implementation": {name: hashlib.sha256((BACKEND / name).read_bytes()).hexdigest()
                               for name in ("app/evaluation/executor.py", "app/evaluation/ragas_adapter.py",
                                            "app/core/prompts.py", "app/services/hybrid_search.py",
                                            "app/services/bm25_service.py", "app/services/reranker_service.py",
                                            "app/services/token_windows.py")},
        }
        if args.resume:
            EvaluationCheckpoint.read_resume(args.resume, manifest)
        if args.check_only:
            print(f"预检通过：{filename}，{len(rows)} 个分块，{len(selected)} 条题目；未调用模型。")
            return

        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output = BACKEND / "evaluation_reports" / f"document_{args.cases}_{stamp}_{uuid.uuid4().hex[:6]}"
        output.mkdir(parents=True)
        sample = output / "dataset.json"
        sample.write_text(json.dumps({**dataset, "test_cases": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
        config.test_dataset_path = str(sample)
        config.max_concurrency = args.concurrency
        scope = EvaluationScope(str(args.owner_id), (str(args.document_id),))
        checkpoint = EvaluationCheckpoint(output / "progress.json", manifest, resume=args.resume)
        runner = EvaluationRunner(config, owner_id=str(args.owner_id), scope=scope, checkpoint=checkpoint)
        (output / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
        print(f"评测文件：{filename}；分块：{len(rows)}；题目：{args.cases}；并发：{args.concurrency}", flush=True)
        print(f"回答与评判模型：{evaluation_llm.provider} / {evaluation_llm.model}", flush=True)
        print(f"每题总耗时上限：{config.timeout_seconds} 秒；每次模型请求上限：{settings.EVALUATION_LLM_REQUEST_TIMEOUT} 秒", flush=True)
        print(f"断点进度：{output / 'progress.json'}", flush=True)
        if args.resume:
            print("已校验并加载旧进度，将复用已完成的检索、回答和指标。", flush=True)
        print("离线维护评测：消耗真实模型 API 额度，不扣应用签到积分。", flush=True)
        report = await runner.run()
        report["source_dataset_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        report["resumed_from"] = str(args.resume) if args.resume else None
        (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await runner.generate_html_report(str(output / "report.html"))
        print(f"报告目录：{output}")
        for item in report["configs"]:
            print(json.dumps({"name": item["name"], "summary": item["summary"], "metrics": item["metrics"]}, ensure_ascii=False))
        if any(item["summary"]["failure"] for item in report["configs"]):
            raise SystemExit("部分题目失败，详细原因见 results.json；这些结果不能视为完整成功评测。")
    finally:
        await evaluation_llm.close()
        await bailian_service.close()
        await milvus_service.close()
        await close_pool()
        await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("app/evaluation/hybrid_80_lc12_config.json"))
    parser.add_argument("--owner-id", type=uuid.UUID, required=True)
    parser.add_argument("--document-id", type=uuid.UUID, required=True)
    parser.add_argument("--cases", type=int, choices=range(1, 81), default=3, metavar="1..80")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--timeout-seconds", type=int, choices=range(10, 601), default=None, metavar="10..600")
    parser.add_argument("--resume", type=Path, help="复用之前同配置评测产生的 progress.json")
    parser.add_argument("--check-only", action="store_true", help="只核对文档、题目和索引，不调用模型")
    asyncio.run(run(parser.parse_args()))
