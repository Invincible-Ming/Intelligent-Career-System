"""
评测 API 端点。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.limits import BudgetExceeded
from app.core.models import User
from app.evaluation.config import ExperimentConfig
from app.evaluation.metrics import ragas_metrics
from app.evaluation.runner import EvaluationRunner
from app.evaluation.scope import EvaluationScope, prepare_evaluation_scope
from app.security.auth import require_admin
from app.services.experiment_store import experiment_store
from app.services.knowledge_index import KnowledgeIndexUnavailable, knowledge_index_service
from app.services.knowledge_policy import knowledge_provider_for

AdminUser = Annotated[User, Depends(require_admin)]
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["评测"])


class StartExperimentRequest(BaseModel):
    """启动评测实验请求。"""

    experiment_config: ExperimentConfig


class ExperimentStatus(BaseModel):
    """实验状态。"""

    experiment_id: str
    status: str = Field(description="running, completed, failed")
    progress: float = Field(default=0.0, ge=0, le=1)
    report_path: str | None = None
    error: str | None = None


@router.post(
    "/experiments",
    response_model=ExperimentStatus,
)
async def start_experiment(
        user: AdminUser,
        request: StartExperimentRequest,
        background_tasks: BackgroundTasks,
) -> ExperimentStatus:
    """启动评测实验（异步后台运行）。"""

    if knowledge_provider_for(user) != "milvus":
        raise HTTPException(409, "Dense/BM25/Hybrid 消融评测需要 Milvus 模式；ima 不暴露这些检索策略")

    if settings.KNOWLEDGE_PROVIDER in ("subscription", "credits"):
        from app.core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            try:
                await knowledge_index_service.prepare_search(session, user.id)
            except KnowledgeIndexUnavailable as exc:
                raise HTTPException(503, str(exc)) from None

    dataset_root = Path(__file__).resolve().parent
    path = Path(request.experiment_config.test_dataset_path)
    candidate = (path if path.is_absolute() else dataset_root / path).resolve()
    if candidate.parent != dataset_root or candidate.suffix != '.json' or not candidate.is_file():
        raise HTTPException(400, '评测只允许使用评测目录中的已审核 JSON 数据集')
    if candidate.stat().st_size > 2 * 1024 * 1024:
        raise HTTPException(400, '评测数据集大小超限')
    if len(request.experiment_config.variants) > 3:
        raise HTTPException(400, '最多允许 3 个评测变体')
    request.experiment_config.test_dataset_path = str(candidate)
    try:
        ragas_metrics.ensure_available()
        scope = await prepare_evaluation_scope(
            owner_id=str(user.id), config=request.experiment_config,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    experiment_id = str(uuid.uuid4())

    experiment_status = {
        "owner_id": str(user.id),
        "experiment_id": experiment_id,
        "status": "running",
        "progress": 0.0,
        "report_path": None,
        "error": None,
        "started_at": time.time(),
    }

    await experiment_store.save(
        owner_id=str(user.id),
        experiment_id=experiment_id,
        status=experiment_status,
    )

    # 在后台执行评测
    background_tasks.add_task(
        run_experiment_background,
        owner_id=str(user.id),
        experiment_id=experiment_id,
        experiment_config=request.experiment_config,
        scope=scope,
    )

    return ExperimentStatus(**experiment_status)


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentStatus,
)
async def get_experiment_status(
        user: AdminUser,
        experiment_id: str,
) -> ExperimentStatus:
    """查询评测实验状态。"""

    # 状态按 owner_id 命名空间存取，越权访问等同于「不存在」。
    status = await experiment_store.get(
        owner_id=str(user.id),
        experiment_id=experiment_id,
    )

    if status is None:
        raise HTTPException(
            status_code=404,
            detail="实验不存在",
        )

    return ExperimentStatus(
        **{k: v for k, v in status.items() if k != "started_at"}
    )


@router.get(
    "/experiments",
    response_model=list[ExperimentStatus],
)
async def list_experiments(user: AdminUser) -> list[ExperimentStatus]:
    """列出当前管理员名下的所有评测实验。"""

    statuses = await experiment_store.list_for_owner(owner_id=str(user.id))

    return [
        ExperimentStatus(**{k: v for k, v in status.items() if k != "started_at"})
        for status in statuses
    ]


@router.get(
    "/datasets",
    response_model=list[dict[str, Any]],
)
async def list_datasets(user: AdminUser) -> list[dict[str, Any]]:
    """列出评测目录中可用的测试数据集。"""

    dataset_root = Path(__file__).resolve().parent
    datasets: list[dict[str, Any]] = []
    for candidate in sorted(dataset_root.glob("*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("跳过不可读的评测数据集 %s（%s）", candidate.name, type(exc).__name__)
            continue
        cases = data.get("test_cases") if isinstance(data, dict) else None
        if not isinstance(cases, list) or not cases:
            continue
        datasets.append({
            "filename": candidate.name,
            "name": data.get("name", candidate.stem),
            "description": data.get("description", ""),
            "case_count": len(cases),
        })
    return datasets


@router.get(
    "/report",
    response_model=dict[str, str],
)
async def get_report(user: AdminUser, path: str) -> dict[str, str]:
    """读取当前管理员名下的 HTML 报告。"""

    allowed_root = Path("evaluation_reports").resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if candidate.parent != allowed_root or candidate.suffix != ".html" or not candidate.is_file():
        raise HTTPException(400, "仅允许访问评测报告目录中的 HTML 报告")
    statuses = await experiment_store.list_for_owner(owner_id=str(user.id))
    owned_paths = {
        Path(status["report_path"]).resolve()
        for status in statuses
        if status.get("status") == "completed" and status.get("report_path")
    }
    if candidate not in owned_paths:
        raise HTTPException(404, "报告不存在或已过期")
    return {"html": candidate.read_text(encoding="utf-8")}


async def run_experiment_background(
        *,
        owner_id: str,
        scope: EvaluationScope,
        experiment_id: str,
        experiment_config: ExperimentConfig,
) -> None:
    """后台运行评测实验。"""

    async def update(**fields: Any) -> None:
        """读-改-写当前状态。

        评测进度只有本协程写入，串行更新不会丢字段；
        写入失败已在 store 内部降级，不阻断评测。
        """

        current = await experiment_store.get(
            owner_id=owner_id,
            experiment_id=experiment_id,
        ) or {"owner_id": owner_id, "experiment_id": experiment_id}
        current.update(fields)
        await experiment_store.save(
            owner_id=owner_id,
            experiment_id=experiment_id,
            status=current,
        )

    from app.services.credits import InsufficientCredits, billing_scope

    billing = billing_scope(owner_id, "evaluation")
    billing.__enter__()
    try:
        runner = EvaluationRunner(experiment_config, owner_id=owner_id, scope=scope)

        await update(progress=0.1)

        # 运行评测
        async with asyncio.timeout(settings.ANALYSIS_TOTAL_TIMEOUT):
            await runner.run()

        await update(progress=0.8)

        # 生成报告
        report_path = await runner.generate_html_report()

        await update(status="completed", progress=1.0, report_path=report_path)

    except BaseException as exc:
        await update(
            status="failed",
            error=str(exc) if isinstance(exc, InsufficientCredits) else "评测失败或已达到调用与耗时上限",
        )
        if not isinstance(exc, (Exception, BudgetExceeded, InsufficientCredits)):
            raise
    finally:
        billing.__exit__(None, None, None)
