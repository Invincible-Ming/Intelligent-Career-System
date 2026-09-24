"""
智能求职系统 API (支持持久化状态、人机协同中断、流式恢复与历史任务回溯)。
"""

from __future__ import annotations

import asyncio
import anyio
import json
import uuid
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    Query,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.bailian import bailian_service
from app.services.bm25_service import bm25_service
from app.core.database import get_db, AsyncSessionLocal
from app.services.document_service import document_service
from app.services.hybrid_search import hybrid_search_service
from app.services.milvus_service import milvus_service
from app.core.models import AgentRun, AnalysisResult, Document
from app.core.schemas import (
    DocumentResponse,
    InterviewRequest,
    LearningRequest,
    MatchRequest,
    RunResponse,
    SearchRequest,
    SearchResult,
)
from app.agents.workflow import (
    resume_match,
    resume_match_stream,
    run_interview,
    run_learning_plan,
    run_match,
    run_match_stream,
)

from app.security.auth import CurrentUser, owned_record
from app.core.config import settings

router = APIRouter()
DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_db, scope="request"),
]


# ----------------------------------------------------------------------
# 接口请求模型
# ----------------------------------------------------------------------


class MatchResumeRequest(BaseModel):
    """恢复/人工干预岗位匹配任务的请求体。"""

    model_config = ConfigDict(extra="forbid")
    thread_id: uuid.UUID = Field(..., description="工作流线程 ID (通常对应任务的 run_id)")
    run_id: uuid.UUID | None = Field(None, description="任务记录 ID (可选，不填默认取 thread_id)")
    human_feedback: str | None = Field(
        None,
        max_length=2000,
        description="人工审核纠错意见 (若为空则表示确认无误直接继续)",
    )


# ----------------------------------------------------------------------
# 文档管理
# ----------------------------------------------------------------------


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    tags=["文档"],
)
async def upload_document(
        user: CurrentUser,
        session: DatabaseSession,
        file: Annotated[UploadFile, File(...)],
        document_type: Annotated[
            str,
            Form(max_length=50),
        ] = "knowledge",
) -> DocumentResponse:
    """上传并向量化简历、JD 或知识文档。"""

    try:
        document = await document_service.upload_document(
            owner_id=user.id,
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
            detail="文档处理失败，请稍后重试",
        ) from exc


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    tags=["文档"],
)
async def list_documents(
        user: CurrentUser,
        session: DatabaseSession,
) -> list[DocumentResponse]:
    """查询已上传文档。"""

    documents = await document_service.list_documents(session=session, owner_id=user.id)

    return [
        DocumentResponse.model_validate(document)
        for document in documents
    ]


@router.delete(
    "/documents/{document_id}",
    tags=["文档"],
)
async def delete_document(
        user: CurrentUser,
        document_id: uuid.UUID,
        session: DatabaseSession,
) -> dict[str, str]:
    """删除文档及其 MinIO 文件和 Milvus 向量。"""

    try:
        await document_service.delete_document(
            owner_id=user.id,
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
        session: DatabaseSession,
        user: CurrentUser,
        request: SearchRequest,
) -> list[SearchResult]:
    """使用 text-embedding-v3 和 Milvus 检索知识库。"""

    allowed_ids = list(map(str, (await session.execute(select(Document.id).where(
        Document.owner_id == user.id, Document.status == "ready"
    ))).scalars().all()))
    if not allowed_ids:
        return []
    try:
        query_vector = await bailian_service.embed_query(request.query)

        results = await milvus_service.search(
            query_vector=query_vector,
            document_type=request.document_type,
            top_k=request.top_k,
            allowed_document_ids=allowed_ids,
        )

        return [
            SearchResult.model_validate(item)
            for item in results if str(item["document_id"]) in allowed_ids
        ]

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="操作失败，请稍后重试",
        ) from exc


@router.post(
    "/search/bm25",
    response_model=list[SearchResult],
    tags=["知识库"],
)
async def search_knowledge_with_bm25(
        session: DatabaseSession,
        user: CurrentUser,
        request: SearchRequest,
) -> list[SearchResult]:
    """使用 BM25 进行关键词检索。"""

    allowed_ids = list(map(str, (await session.execute(select(Document.id).where(
        Document.owner_id == user.id, Document.status == "ready"
    ))).scalars().all()))
    if not allowed_ids:
        return []
    try:
        results = await bm25_service.search(
            query=request.query,
            document_type=request.document_type,
            top_k=request.top_k,
            allowed_document_ids=allowed_ids,
        )

        return [
            SearchResult.model_validate(item)
            for item in results if str(item["document_id"]) in allowed_ids
        ]

    except Exception as exc:
        if not isinstance(exc, Exception):
            raise
        raise HTTPException(
            status_code=500,
            detail="操作失败，请稍后重试",
        ) from exc


@router.post(
    "/search/hybrid",
    response_model=list[SearchResult],
    tags=["知识库"],
)
async def search_knowledge_hybrid(
        session: DatabaseSession,
        user: CurrentUser,
        request: SearchRequest,
) -> list[SearchResult]:
    """执行 Dense、BM25、RRF 和 BGE 重排。"""

    allowed_ids = list(map(str, (await session.execute(select(Document.id).where(
        Document.owner_id == user.id, Document.status == "ready"
    ))).scalars().all()))
    if not allowed_ids:
        return []
    try:
        results = await hybrid_search_service.search(
            query=request.query,
            document_type=request.document_type,
            top_k=request.top_k,
            allowed_document_ids=allowed_ids,
        )

        return [
            SearchResult.model_validate(item)
            for item in results if str(item["document_id"]) in allowed_ids
        ]

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        if not isinstance(exc, Exception):
            raise
        raise HTTPException(
            status_code=500,
            detail="操作失败，请稍后重试",
        ) from exc


# ----------------------------------------------------------------------
# 岗位匹配与人机协同干预
# ----------------------------------------------------------------------


@router.post(
    "/match",
    response_model=RunResponse,
    tags=["求职分析"],
)
async def match_resume(
        user: CurrentUser,
        request: MatchRequest,
        session: DatabaseSession,
) -> RunResponse:
    """分析简历与岗位描述（同步执行到中断点 verify_agent 前暂停）。"""

    resume_text = await get_document_text(
        owner_id=user.id,
        session=session,
        document_id=request.resume_document_id,
    )

    if request.jd_text:
        jd_text = request.jd_text
    else:
        jd_text = await get_document_text(
            owner_id=user.id,
            session=session,
            document_id=request.jd_document_id,
        )

    run = await create_run(
        owner_id=user.id,
        session=session,
        task_type="match",
        input_data={
            "resume_document_id": str(request.resume_document_id),
            "jd_document_id": (
                str(request.jd_document_id)
                if request.jd_document_id
                else None
            ),
            "jd_input_type": "text" if request.jd_text else "document",
        },
    )

    created_run_id = run.id
    try:
        match_result = await run_match(
            resume_text=resume_text,
            jd_text=jd_text,
            owner_id=str(user.id),
            thread_id=str(run.id),
        )

        run.status = match_result.get("status", "paused")
        run.result_data = match_result
        await session.commit()

        return to_run_response(run)

    except BaseException as exc:
        await mark_run_failed(
            session=session,
            run_id=created_run_id,
            error=exc,
        )

        if not isinstance(exc, Exception):
            raise
        raise HTTPException(
            status_code=500,
            detail="操作失败，请稍后重试",
        ) from exc


@router.post(
    "/match/resume",
    response_model=RunResponse,
    tags=["求职分析"],
    summary="人工确认或修正后唤醒匹配工作流",
)
async def resume_match_endpoint(
        user: CurrentUser,
        request: MatchResumeRequest,
        session: DatabaseSession,
) -> RunResponse:
    """接收人工干预反馈，唤醒被挂起的工作流完成最终的事实校验与报告产出。"""

    target_run_id = request.run_id or request.thread_id
    run = await owned_record(session, AgentRun, target_run_id, user.id)

    if str(request.thread_id) != str(run.id):
        raise HTTPException(400, "thread_id 必须与任务 ID 一致")
    changed = (await session.execute(update(AgentRun).where(
        AgentRun.id == run.id, AgentRun.owner_id == user.id,
        AgentRun.task_type == 'match', AgentRun.status == 'paused',
    ).values(status='running').returning(AgentRun.id))).scalar_one_or_none()
    if changed is None:
        raise HTTPException(409, '只有暂停的匹配任务可以恢复')
    await session.commit()

    created_run_id = run.id
    try:
        report = await resume_match(
            thread_id=str(run.id),
            owner_id=str(user.id),
            human_feedback=request.human_feedback,
        )
        result = report.model_dump(mode="json")

        analysis = AnalysisResult(
            run_id=run.id,
            resume_analysis=result.get("resume_analysis", {}),
            job_analysis=result.get("job_analysis", {}),
            match_report=result,
        )

        run.status = "completed"
        run.result_data = result
        session.add(analysis)
        await session.commit()

        return to_run_response(run)

    except BaseException as exc:
        await mark_run_failed(
            session=session,
            run_id=created_run_id,
            error=exc,
        )
        if not isinstance(exc, Exception):
            raise
        raise HTTPException(
            status_code=500,
            detail="操作失败，请稍后重试",
        ) from exc


@router.post(
    "/match/stream",
    tags=["求职分析"],
    summary="SSE 实时流式岗位匹配接口 (支持中断)",
)
async def match_resume_stream(
        user: CurrentUser,
        request: MatchRequest,
        session: DatabaseSession,
):
    """
    分析简历与岗位描述，通过 SSE 实时流式推送 Agent 节点进展。
    运行至中断点时推送 interrupt 事件并挂起。
    """

    resume_text = await get_document_text(
        owner_id=user.id,
        session=session,
        document_id=request.resume_document_id,
    )

    if request.jd_text:
        jd_text = request.jd_text
    else:
        jd_text = await get_document_text(
            owner_id=user.id,
            session=session,
            document_id=request.jd_document_id,
        )

    run = await create_run(
        owner_id=user.id,
        session=session,
        task_type="match",
        input_data={
            "resume_document_id": str(request.resume_document_id),
            "jd_document_id": (
                str(request.jd_document_id)
                if request.jd_document_id
                else None
            ),
            "jd_input_type": "text" if request.jd_text else "document",
        },
    )

    stream_run_id = run.id

    async def sse_event_generator():
        init_payload = {
            "run_id": str(run.id),
            "thread_id": str(run.id),
            "task_type": "match",
            "status": "running",
            "message": "任务已创建，启动 Multi-Agent 并发流水线...",
        }
        yield f"event: init\ndata: {json.dumps(init_payload, ensure_ascii=False)}\n\n"

        try:
            async for sse_chunk in run_match_stream(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    owner_id=str(user.id),
                    thread_id=str(run.id),
            ):
                if sse_chunk.startswith("event: interrupt"):
                    run.status = "paused"
                    # 挂起时刻同时落库初步报告，历史记录载入时才能回看初评结果。
                    try:
                        for line in sse_chunk.split("\n"):
                            if line.startswith("data:"):
                                payload = json.loads(line[5:].strip())
                                run.result_data = {
                                    "status": "paused",
                                    "current_report": payload.get("current_report"),
                                }
                                break
                    except Exception:
                        pass
                    await session.commit()

                elif sse_chunk.startswith("event: complete"):
                    try:
                        for line in sse_chunk.split("\n"):
                            if line.startswith("data:"):
                                payload = json.loads(line[5:].strip())
                                final_report_result = payload.get("report")

                                if final_report_result:
                                    analysis = AnalysisResult(
                                        run_id=run.id,
                                        resume_analysis=final_report_result.get("resume_analysis", {}),
                                        job_analysis=final_report_result.get("job_analysis", {}),
                                        match_report=final_report_result,
                                    )
                                    run.status = "completed"
                                    run.result_data = final_report_result
                                    session.add(analysis)
                                    await session.commit()
                                break
                    except Exception:
                        raise RuntimeError("报告保存失败")

                yield sse_chunk

        except Exception as exc:
            await mark_run_failed(
                session=session,
                run_id=run.id,
                error=exc,
            )
            error_payload = {"message": "岗位匹配失败，请稍后重试"}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        finally:
            with anyio.CancelScope(shield=True):
                async with asyncio.timeout(3):
                    await fail_unfinished_run(stream_run_id, user.id)

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/match/stream/resume",
    tags=["求职分析"],
    summary="SSE 流式人工唤醒与干预接口",
)
async def resume_match_stream_endpoint(
        user: CurrentUser,
        request: MatchResumeRequest,
        session: DatabaseSession,
):
    """流式唤醒已暂停的工作流，实时查看事实校验员的审核进展与最终报告。"""

    target_run_id = request.run_id or request.thread_id
    run = await owned_record(session, AgentRun, target_run_id, user.id)

    if str(request.thread_id) != str(run.id):
        raise HTTPException(400, "thread_id 必须与任务 ID 一致")
    changed = (await session.execute(update(AgentRun).where(
        AgentRun.id == run.id, AgentRun.owner_id == user.id,
        AgentRun.task_type == 'match', AgentRun.status == 'paused',
    ).values(status='running').returning(AgentRun.id))).scalar_one_or_none()
    if changed is None:
        raise HTTPException(409, '只有暂停的匹配任务可以恢复')
    await session.commit()

    stream_run_id = run.id

    async def sse_event_generator():
        # 🌟 修复关键：定义一个本地 flag 以避免在 session.commit() 后直接读取 run.status 引发 MissingGreenlet 异常
        is_successfully_completed = False

        try:
            async for sse_chunk in resume_match_stream(
                    thread_id=str(run.id),
                    owner_id=str(user.id),
                    human_feedback=request.human_feedback,
            ):
                if sse_chunk.startswith("event: complete"):
                    try:
                        for line in sse_chunk.split("\n"):
                            if line.startswith("data:"):
                                payload = json.loads(line[5:].strip())
                                final_report_result = payload.get("report")

                                if final_report_result:
                                    analysis = AnalysisResult(
                                        run_id=run.id,
                                        resume_analysis=final_report_result.get("resume_analysis", {}),
                                        job_analysis=final_report_result.get("job_analysis", {}),
                                        match_report=final_report_result,
                                    )
                                    run.status = "completed"
                                    run.result_data = final_report_result
                                    session.add(analysis)
                                    await session.commit()
                                    is_successfully_completed = True
                                break
                    except Exception:
                        raise RuntimeError("报告保存失败")

                yield sse_chunk

            # 🌟 使用本地 flag 作为判断，彻底避免隐式 Lazy-Loading
            if not is_successfully_completed:
                await mark_run_failed(
                    session=session,
                    run_id=target_run_id,  # 直接使用目标 ID 避免再次访问 run 属性
                    error=Exception("恢复执行后未产出最终报告"),
                )

        except Exception as exc:
            await mark_run_failed(
                session=session,
                run_id=target_run_id,
                error=exc,
            )
            error_payload = {"message": "工作流恢复失败，请稍后重试"}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        finally:
            with anyio.CancelScope(shield=True):
                async with asyncio.timeout(3):
                    await fail_unfinished_run(stream_run_id, user.id)

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ----------------------------------------------------------------------
# 面试计划
# ----------------------------------------------------------------------


@router.post(
    "/interview",
    response_model=RunResponse,
    tags=["求职分析"],
)
async def create_interview_plan(
        user: CurrentUser,
        request: InterviewRequest,
        session: DatabaseSession,
) -> RunResponse:
    """根据岗位匹配报告生成面试问题。"""

    match_report = await get_match_report(
        owner_id=user.id,
        session=session,
        run_id=request.match_run_id,
    )

    run = await create_run(
        owner_id=user.id,
        session=session,
        task_type="interview",
        input_data=request.model_dump(mode="json"),
    )

    created_run_id = run.id
    try:
        plan = await run_interview(
            match_report=match_report,
            difficulty=request.difficulty,
            question_count=request.question_count,
            thread_id=str(run.id),
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

    except BaseException as exc:
        await mark_run_failed(
            session=session,
            run_id=created_run_id,
            error=exc,
        )

        if not isinstance(exc, Exception):
            raise
        raise HTTPException(
            status_code=500,
            detail="操作失败，请稍后重试",
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
        user: CurrentUser,
        request: LearningRequest,
        session: DatabaseSession,
) -> RunResponse:
    """根据能力缺口生成学习计划。"""

    match_report = await get_match_report(
        owner_id=user.id,
        session=session,
        run_id=request.match_run_id,
    )

    run = await create_run(
        owner_id=user.id,
        session=session,
        task_type="learning_plan",
        input_data=request.model_dump(mode="json"),
    )

    created_run_id = run.id
    try:
        plan = await run_learning_plan(
            match_report=match_report,
            available_weeks=request.available_weeks,
            hours_per_week=request.hours_per_week,
            thread_id=str(run.id),
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

    except BaseException as exc:
        await mark_run_failed(
            session=session,
            run_id=created_run_id,
            error=exc,
        )

        if not isinstance(exc, Exception):
            raise
        raise HTTPException(
            status_code=500,
            detail="操作失败，请稍后重试",
        ) from exc


# ----------------------------------------------------------------------
# 任务查询与历史列表
# ----------------------------------------------------------------------


@router.get(
    "/runs",
    response_model=list[RunResponse],
    tags=["任务"],
    summary="查询历史任务列表",
)
async def list_runs(
        user: CurrentUser,
        session: DatabaseSession,
        task_type: str | None = "match",
        status: str | None = None,
        limit: int = Query(20, ge=1, le=100),
) -> list[RunResponse]:
    """
    查询历史任务列表。
    可按 task_type（默认 match）和 status（如 paused, completed）筛选，按时间倒序排列。
    """

    query = select(AgentRun).where(AgentRun.owner_id == user.id).order_by(desc(AgentRun.created_at)).limit(limit)

    if task_type:
        query = query.where(AgentRun.task_type == task_type)
    if status:
        query = query.where(AgentRun.status == status)

    result = await session.execute(query)
    runs = result.scalars().all()

    return [to_run_response(run) for run in runs]


@router.get(
    "/runs/{run_id}",
    response_model=RunResponse,
    tags=["任务"],
)
async def get_run(
        user: CurrentUser,
        run_id: uuid.UUID,
        session: DatabaseSession,
) -> RunResponse:
    """查询指定 Agent 任务的当前状态和结果。"""

    run = await owned_record(session, AgentRun, run_id, user.id)

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
        owner_id: uuid.UUID,
        session: AsyncSession,
        document_id: uuid.UUID | None,
) -> str:
    """获取并解析文档文本。"""

    if document_id is None:
        raise HTTPException(
            status_code=400,
            detail="缺少文档 ID",
        )

    document = await owned_record(session, Document, document_id, owner_id)
    try:
        content = await document_service.get_document_text(
            owner_id=owner_id,
            session=session,
            document_id=document_id,
        )
        if len(content) > settings.MAX_DOCUMENT_TEXT_CHARS:
            raise HTTPException(400, "分析资料过长，请拆分文档")
        return content
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


async def create_run(
        *,
        owner_id: uuid.UUID,
        session: AsyncSession,
        task_type: str,
        input_data: dict[str, Any],
) -> AgentRun:
    """创建 Agent 任务记录。"""

    run = AgentRun(
        owner_id=owner_id,
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
        owner_id: uuid.UUID,
        session: AsyncSession,
        run_id: uuid.UUID,
) -> dict[str, Any]:
    """读取已经完成的岗位匹配报告。"""

    run = await owned_record(session, AgentRun, run_id, owner_id)

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
        run.error_message = "操作中断或执行失败，请稍后重试"
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
        input_data=run.input_data,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


async def fail_unfinished_run(run_id, owner_id):
    async with AsyncSessionLocal() as session:
        await session.execute(update(AgentRun).where(AgentRun.id == run_id, AgentRun.owner_id == owner_id,
                                                     AgentRun.status == 'running').values(status='failed',
                                                                                          error_message='执行已中断，请重新提交'))
        await session.commit()
