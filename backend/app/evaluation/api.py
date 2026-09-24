"""
评测 API 端点。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Annotated
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field

from app.security.auth import require_admin
from app.core.models import User
from app.core.config import settings
from app.core.limits import BudgetExceeded

AdminUser = Annotated[User, Depends(require_admin)]

from app.evaluation.config import ExperimentConfig
from app.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/evaluation", tags=["评测"])

# 存储运行中的评测任务
running_experiments: dict[str, dict[str, Any]] = {}


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
    experiment_id = str(uuid.uuid4())

    experiment_status = {
        "owner_id": str(user.id),
        "experiment_id": experiment_id,
        "status": "running",
        "progress": 0.0,
        "report_path": None,
        "error": None,
    }

    running_experiments[experiment_id] = experiment_status

    # 在后台执行评测
    background_tasks.add_task(
        run_experiment_background,
        experiment_id=experiment_id,
        experiment_config=request.experiment_config,
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

    if experiment_id not in running_experiments or running_experiments[experiment_id]["owner_id"] != str(user.id):
        raise HTTPException(
            status_code=404,
            detail="实验不存在",
        )

    return ExperimentStatus(
        **running_experiments[experiment_id]
    )


@router.get(
    "/experiments",
    response_model=list[ExperimentStatus],
)
async def list_experiments(user: AdminUser) -> list[ExperimentStatus]:
    """列出所有评测实验。"""

    return [
        ExperimentStatus(**status)
        for status in running_experiments.values() if status["owner_id"] == str(user.id)
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
        except Exception:
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
    """读取指定评测实验的 HTML 报告（仅限评测报告目录）。"""

    allowed_root = Path("evaluation_reports").resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if candidate.parent != allowed_root or candidate.suffix != ".html" or not candidate.is_file():
        raise HTTPException(400, "仅允许访问评测报告目录中的 HTML 报告")
    return {"html": candidate.read_text(encoding="utf-8")}


async def run_experiment_background(
        *,
        experiment_id: str,
        experiment_config: ExperimentConfig,
) -> None:
    """后台运行评测实验。"""

    try:
        runner = EvaluationRunner(experiment_config)

        # 更新进度
        running_experiments[experiment_id]["progress"] = 0.1

        # 运行评测
        async with asyncio.timeout(settings.ANALYSIS_TOTAL_TIMEOUT):
            await runner.run()

        running_experiments[experiment_id]["progress"] = 0.8

        # 生成报告
        report_path = await runner.generate_html_report()

        running_experiments[experiment_id]["status"] = "completed"
        running_experiments[experiment_id]["progress"] = 1.0
        running_experiments[experiment_id]["report_path"] = report_path

    except BaseException as exc:
        running_experiments[experiment_id]["status"] = "failed"
        running_experiments[experiment_id]["error"] = "评测失败或已达到调用与耗时上限"
        if not isinstance(exc, (Exception, BudgetExceeded)):
            raise
