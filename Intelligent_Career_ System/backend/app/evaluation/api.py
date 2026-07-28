"""
评测 API 端点。
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

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
    request: StartExperimentRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentStatus:
    """启动评测实验（异步后台运行）。"""

    experiment_id = str(uuid.uuid4())[:8]

    experiment_status = {
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
    experiment_id: str,
) -> ExperimentStatus:
    """查询评测实验状态。"""

    if experiment_id not in running_experiments:
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
async def list_experiments() -> list[ExperimentStatus]:
    """列出所有评测实验。"""

    return [
        ExperimentStatus(**status)
        for status in running_experiments.values()
    ]


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
        await runner.run()

        running_experiments[experiment_id]["progress"] = 0.8

        # 生成报告
        report_path = await runner.generate_html_report()

        running_experiments[experiment_id]["status"] = "completed"
        running_experiments[experiment_id]["progress"] = 1.0
        running_experiments[experiment_id]["report_path"] = report_path

    except Exception as exc:
        running_experiments[experiment_id]["status"] = "failed"
        running_experiments[experiment_id]["error"] = str(exc)
