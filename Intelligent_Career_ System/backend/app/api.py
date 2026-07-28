"""
智能求职系统 API。
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.hybrid_search import hybrid_search_service
from app.bm25_service import bm25_service
from app.bailian import bailian_service
from app.database import get_db
from app.document_service import document_service
from app.milvus_service import milvus_service
from app.models import AgentRun, AnalysisResult
from app.schemas import (
    DocumentResponse,
    InterviewRequest,
    LearningRequest,
    MatchRequest,
    RunResponse,
    SearchRequest,
    SearchResult,
)
from app.workflow import (
    run_interview,
    run_learning_plan,
    run_match,
)

router = APIRouter()
DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_db),
]


# ----------------------------------------------------------------------
# 文档
# ----------------------------------------------------------------------


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    tags=["文档"],
)
async def upload_document(
        session: DatabaseSession,
        file: Annotated[UploadFile, File(...)],
        document_type: Annotated[
            str,
            Form(),
        ] = "knowledge",
) -> DocumentResponse:
    """上传并向量化简历、JD 或知识文档。"""

    try:
        document = await document_service.upload_document(
            session=session,
            file=file,
            document_type=document_type,
        )

        return DocumentResponse.model_validate(document)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    tags=["文档"],
)
async def list_documents(
        session: DatabaseSession,
) -> list[DocumentResponse]:
    """查询已上传文档。"""

    documents = await document_service.list_documents(
        session=session
    )

    return [
        DocumentResponse.model_validate(document)
        for document in documents
    ]


@router.delete(
    "/documents/{document_id}",
    tags=["文档"],
)
async def delete_document(
        document_id: uuid.UUID,
        session: DatabaseSession,
) -> dict[str, str]:
    """删除文档及其 MinIO 文件和 Milvus 向量。"""

    try:
        await document_service.delete_document(
            session=session,
            document_id=document_id,
        )

        return {"message": "文档删除成功"}

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


# ----------------------------------------------------------------------
# 知识库检索
# ----------------------------------------------------------------------


@router.post(
    "/search",
    response_model=list[SearchResult],
    tags=["知识库"],
)
async def search_knowledge(
        request: SearchRequest,
) -> list[SearchResult]:
    """使用 text-embedding-v3 和 Milvus 检索知识库。"""

    try:
        query_vector = await bailian_service.embed_query(
            request.query
        )

        results = await milvus_service.search(
            query_vector=query_vector,
            document_type=request.document_type,
            top_k=request.top_k,
        )

        return [
            SearchResult.model_validate(item)
            for item in results
        ]

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.post(
    "/search/bm25",
    response_model=list[SearchResult],
    tags=["知识库"],
)
async def search_knowledge_with_bm25(
        request: SearchRequest,
) -> list[SearchResult]:
    """使用 BM25 进行关键词检索。"""

    try:
        results = await bm25_service.search(
            query=request.query,
            document_type=request.document_type,
            top_k=request.top_k,
        )

        return [
            SearchResult.model_validate(item)
            for item in results
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"BM25 检索失败：{exc}",
        ) from exc


@router.post(
    "/search/hybrid",
    response_model=list[SearchResult],
    tags=["知识库"],
)
async def search_knowledge_hybrid(
    request: SearchRequest,
) -> list[SearchResult]:
    """执行 Dense、BM25、RRF 和 BGE 重排。"""

    try:
        results = await hybrid_search_service.search(
            query=request.query,
            document_type=request.document_type,
            top_k=request.top_k,
        )

        return [
            SearchResult.model_validate(item)
            for item in results
        ]

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"混合检索失败：{exc}",
        ) from exc


# ----------------------------------------------------------------------
# 岗位匹配
# ----------------------------------------------------------------------


@router.post(
    "/match",
    response_model=RunResponse,
    tags=["求职分析"],
)
async def match_resume(
        request: MatchRequest,
        session: DatabaseSession,
) -> RunResponse:
    """分析简历与岗位描述并生成匹配报告。"""

    resume_text = await get_document_text(
        session=session,
        document_id=request.resume_document_id,
    )

    if request.jd_text:
        jd_text = request.jd_text
    else:
        jd_text = await get_document_text(
            session=session,
            document_id=request.jd_document_id,
        )

    run = await create_run(
        session=session,
        task_type="match",
        input_data={
            "resume_document_id": str(
                request.resume_document_id
            ),
            "jd_document_id": (
                str(request.jd_document_id)
                if request.jd_document_id
                else None
            ),
            "jd_input_type": (
                "text"
                if request.jd_text
                else "document"
            ),
        },
    )

    try:
        report = await run_match(
            resume_text=resume_text,
            jd_text=jd_text,
        )
        result = report.model_dump(mode="json")

        analysis = AnalysisResult(
            run_id=run.id,
            resume_analysis=result["resume_analysis"],
            job_analysis=result["job_analysis"],
            match_report=result,
        )

        run.status = "completed"
        run.result_data = result
        session.add(analysis)
        await session.commit()

        return to_run_response(run)

    except Exception as exc:
        await mark_run_failed(
            session=session,
            run_id=run.id,
            error=exc,
        )

        raise HTTPException(
            status_code=500,
            detail=f"岗位匹配失败：{exc}",
        ) from exc


# ----------------------------------------------------------------------
# 面试计划
# ----------------------------------------------------------------------


@router.post(
    "/interview",
    response_model=RunResponse,
    tags=["求职分析"],
)
async def create_interview_plan(
        request: InterviewRequest,
        session: DatabaseSession,
) -> RunResponse:
    """根据岗位匹配报告生成面试问题。"""

    match_report = await get_match_report(
        session=session,
        run_id=request.match_run_id,
    )

    run = await create_run(
        session=session,
        task_type="interview",
        input_data=request.model_dump(mode="json"),
    )

    try:
        plan = await run_interview(
            match_report=match_report,
            difficulty=request.difficulty,
            question_count=request.question_count,
        )
        result = plan.model_dump(mode="json")

        run.status = "completed"
        run.result_data = result

        session.add(
            AnalysisResult(
                run_id=run.id,
                interview_plan=result,
            )
        )
        await session.commit()

        return to_run_response(run)

    except Exception as exc:
        await mark_run_failed(
            session=session,
            run_id=run.id,
            error=exc,
        )

        raise HTTPException(
            status_code=500,
            detail=f"面试计划生成失败：{exc}",
        ) from exc


# ----------------------------------------------------------------------
# 学习计划
# ----------------------------------------------------------------------


@router.post(
    "/learning-plan",
    response_model=RunResponse,
    tags=["求职分析"],
)
async def create_learning_plan(
        request: LearningRequest,
        session: DatabaseSession,
) -> RunResponse:
    """根据能力缺口生成学习计划。"""

    match_report = await get_match_report(
        session=session,
        run_id=request.match_run_id,
    )

    run = await create_run(
        session=session,
        task_type="learning_plan",
        input_data=request.model_dump(mode="json"),
    )

    try:
        plan = await run_learning_plan(
            match_report=match_report,
            available_weeks=request.available_weeks,
            hours_per_week=request.hours_per_week,
        )
        result = plan.model_dump(mode="json")

        run.status = "completed"
        run.result_data = result

        session.add(
            AnalysisResult(
                run_id=run.id,
                learning_plan=result,
            )
        )
        await session.commit()

        return to_run_response(run)

    except Exception as exc:
        await mark_run_failed(
            session=session,
            run_id=run.id,
            error=exc,
        )

        raise HTTPException(
            status_code=500,
            detail=f"学习计划生成失败：{exc}",
        ) from exc


# ----------------------------------------------------------------------
# 任务查询
# ----------------------------------------------------------------------


@router.get(
    "/runs/{run_id}",
    response_model=RunResponse,
    tags=["任务"],
)
async def get_run(
        run_id: uuid.UUID,
        session: DatabaseSession,
) -> RunResponse:
    """查询 Agent 任务状态和结果。"""

    run = await session.get(AgentRun, run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="任务不存在",
        )

    return to_run_response(run)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------


async def get_document_text(
        *,
        session: AsyncSession,
        document_id: uuid.UUID | None,
) -> str:
    """获取并解析文档文本。"""

    if document_id is None:
        raise HTTPException(
            status_code=400,
            detail="缺少文档 ID",
        )

    try:
        return await document_service.get_document_text(
            session=session,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


async def create_run(
        *,
        session: AsyncSession,
        task_type: str,
        input_data: dict[str, Any],
) -> AgentRun:
    """创建 Agent 任务记录。"""

    run = AgentRun(
        task_type=task_type,
        status="running",
        input_data=input_data,
    )

    session.add(run)
    await session.commit()
    await session.refresh(run)

    return run


async def get_match_report(
        *,
        session: AsyncSession,
        run_id: uuid.UUID,
) -> dict[str, Any]:
    """读取已经完成的岗位匹配报告。"""

    run = await session.get(AgentRun, run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="岗位匹配任务不存在",
        )

    if run.task_type != "match":
        raise HTTPException(
            status_code=400,
            detail="该任务不是岗位匹配任务",
        )

    if run.status != "completed" or not run.result_data:
        raise HTTPException(
            status_code=400,
            detail="岗位匹配任务尚未完成",
        )

    return run.result_data


async def mark_run_failed(
        *,
        session: AsyncSession,
        run_id: uuid.UUID,
        error: Exception,
) -> None:
    """记录 Agent 任务失败状态。"""

    await session.rollback()

    run = await session.get(AgentRun, run_id)

    if run is not None:
        run.status = "failed"
        run.error_message = str(error)[:1000]
        await session.commit()


def to_run_response(
        run: AgentRun,
) -> RunResponse:
    """将数据库任务转换为 API 响应。"""

    return RunResponse(
        run_id=run.id,
        task_type=run.task_type,
        status=run.status,
        result=run.result_data,
        error_message=run.error_message,
    )
