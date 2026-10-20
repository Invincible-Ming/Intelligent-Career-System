"""
智能求职系统 API (支持持久化状态、人机协同中断、流式恢复与历史任务回溯)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Annotated, Any

import anyio
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.workflow import (
    resume_match,
    resume_match_stream,
    run_interview,
    run_learning_plan,
    run_match,
    run_match_stream,
)
from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.core.logging_setup import log_event
from app.core.models import AgentRun, AnalysisResult, Document
from app.core.schemas import (
    DocumentResponse,
    InterviewRequest,
    LearningRequest,
    MatchRequest,
    RunResponse,
    SearchRequest,
    SearchResult,
    UploadDocumentResponse,
)
from app.security.auth import CurrentUser, owned_record
from app.services.bailian import bailian_service
from app.services.bm25_service import bm25_service
from app.services.document_service import document_service
from app.services.hybrid_search import hybrid_search_service
from app.services.milvus_service import milvus_service
from app.services.sandbox_client import SandboxUnavailable

router = APIRouter()
DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_db, scope="request"),
]
logger = logging.getLogger(__name__)


async def has_ready_document(
        *,
        session: AsyncSession,
        owner_id: uuid.UUID,
) -> bool:
    """判断用户是否存在已就绪文档。

    仅用于避免无意义的向量检索与模型调用；真正的数据隔离由
    Milvus / BM25 的 owner_id 过滤和返回前的归属复核完成，不拉取全量 document_id。
    """

    document_id = (await session.execute(
        select(Document.id).where(
            Document.owner_id == owner_id,
            Document.status == "ready",
        ).limit(1)
    )).scalar_one_or_none()

    return document_id is not None


async def owned_search_results(session, owner_id, results, document_type=None):
    """仅复核已召回的少量文档，防止过期/错误索引返回越权内容。

    召回前仍使用 owner_id 索引过滤；此处不加载用户的全量文档 ID。
    """
    candidates = set()
    for item in results:
        try:
            identifier = uuid.UUID(str(item.get("document_id", "")))
        except (ValueError, TypeError, AttributeError):
            continue
        candidates.add(identifier)
    if not candidates:
        return []
    query = select(Document.id).where(
        Document.id.in_(candidates), Document.owner_id == owner_id,
        Document.status == "ready",
    )
    if document_type is not None:
        query = query.where(Document.document_type == document_type)
    allowed = {str(identifier) for identifier in (await session.execute(query)).scalars()}
    return [SearchResult.model_validate(item) for item in results
            if str(item.get("document_id")) in allowed]


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
        description="修改意见；为空表示人工接受当前报告，不代表 AI 已校验通过",
    )


async def persist_match_outcome(session: AsyncSession, run: AgentRun, outcome: dict) -> None:
    """Persist both automatic completion and repeated human-review pauses consistently."""
    status = outcome["status"]
    report = outcome.get("current_report")
    if not report:
        raise RuntimeError("工作流没有返回匹配报告")
    if status not in ("paused", "completed"):
        raise RuntimeError("工作流返回了无效的终态")
    if status == "completed" and not (outcome.get("is_valid") or outcome.get("manual_accepted")):
        raise RuntimeError("报告既未通过自动校验，也未经人工确认")
    run.status = status
    run.result_data = outcome if status == "paused" else report
    run.input_data = {
        **(run.input_data or {}),
        "review_status": "needs_review" if status == "paused" else (
            "manual_accepted" if outcome.get("manual_accepted") else "auto_passed"
        ),
    }
    if status == "completed":
        session.add(AnalysisResult(
            run_id=run.id, resume_analysis=report.get("resume_analysis", {}),
            job_analysis=report.get("job_analysis", {}), match_report=report,
        ))
    await session.commit()
    # onupdate 的数据库时间戳会在 flush 后过期，显式刷新以免响应序列化触发异步懒加载。
    await session.refresh(run)


def match_sse_outcome(chunk: str) -> dict | None:
    if not chunk.startswith(("event: complete\n", "event: interrupt\n")):
        return None
    payload = json.loads(next(line[5:].strip() for line in chunk.splitlines() if line.startswith("data:")))
    return {
        **payload,
        "status": "paused" if chunk.startswith("event: interrupt") else "completed",
        "current_report": payload.get("current_report") or payload.get("report"),
    }


# ----------------------------------------------------------------------
# 文档管理
# ----------------------------------------------------------------------


@router.post(
    "/documents/upload",
    response_model=UploadDocumentResponse,
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
) -> UploadDocumentResponse:
    """上传并向量化简历、JD 或知识文档。

    按内容哈希（SHA-256）判重：同一份内容重复上传会直接复用既有记录，
    不重复解析或向量化，响应中 deduplicated 为 True。
    """

    try:
        document, deduplicated = await document_service.upload_document(
            owner_id=user.id,
            session=session,
            file=file,
            document_type=document_type,
        )

        if deduplicated:
            log_event(
                logger,
                logging.INFO,
                "上传命中已有文档（内容哈希相同），已跳过重复入库",
                document_id=str(document.id),
                document_type=document_type,
            )

        return UploadDocumentResponse(
            document=DocumentResponse.model_validate(document),
            deduplicated=deduplicated,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except SandboxUnavailable as exc:
        # 前置条件没满足（沙箱未启动），属于服务不可用而非文档本身有问题。
        # 503 + 明确指引，比笼统的 500 更容易让人立刻行动。
        log_event(
            logger,
            logging.ERROR,
            f"文档上传被拒：{exc}",
            error=type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # 诊断日志在此：前端只显示通用文案，没有日志就完全无法定位。
        log_event(
            logger,
            logging.ERROR,
            f"文档上传处理失败：{type(exc).__name__}: {exc}",
            error=type(exc).__name__,
            document_type=document_type,
        )
        logger.debug("文档上传异常堆栈", exc_info=True)
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

    owner_id = str(user.id)

    if not await has_ready_document(session=session, owner_id=user.id):
        return []
    try:
        query_vector = await bailian_service.embed_query(request.query)

        results = await milvus_service.search(
            query_vector=query_vector,
            owner_id=owner_id,
            document_type=request.document_type,
            top_k=request.top_k,
        )

        return await owned_search_results(session, user.id, results, request.document_type)

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

    owner_id = str(user.id)

    if not await has_ready_document(session=session, owner_id=user.id):
        return []
    try:
        results = await bm25_service.search(
            query=request.query,
            owner_id=owner_id,
            document_type=request.document_type,
            top_k=request.top_k,
        )

        return await owned_search_results(session, user.id, results, request.document_type)

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

    owner_id = str(user.id)

    if not await has_ready_document(session=session, owner_id=user.id):
        return []
    try:
        results = await hybrid_search_service.search(
            query=request.query,
            owner_id=owner_id,
            document_type=request.document_type,
            top_k=request.top_k,
        )

        return await owned_search_results(session, user.id, results, request.document_type)

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
    """自动分析并校验；仅未通过校验时暂停人工审核。"""

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
            "research_approved": request.research_approved,
        },
    )

    created_run_id = run.id
    try:
        match_result = await run_match(
            resume_text=resume_text,
            jd_text=jd_text,
            owner_id=str(user.id),
            thread_id=str(run.id),
            research_approved=request.research_approved,
        )

        await persist_match_outcome(session, run, match_result)

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
        outcome = await resume_match(
            thread_id=str(run.id),
            owner_id=str(user.id),
            human_feedback=request.human_feedback,
        )
        await persist_match_outcome(session, run, outcome)

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
            "research_approved": request.research_approved,
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
                    research_approved=request.research_approved,
            ):
                outcome = match_sse_outcome(sse_chunk)
                if outcome is not None:
                    await persist_match_outcome(session, run, outcome)

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
        is_settled = False
        try:
            async for sse_chunk in resume_match_stream(
                    thread_id=str(run.id), owner_id=str(user.id),
                    human_feedback=request.human_feedback,
            ):
                outcome = match_sse_outcome(sse_chunk)
                if outcome is not None:
                    await persist_match_outcome(session, run, outcome)
                    is_settled = True
                yield sse_chunk

            if not is_settled:
                await mark_run_failed(
                    session=session, run_id=target_run_id,
                    error=Exception("恢复执行后未返回报告或人工审核状态"),
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

    await owned_record(session, Document, document_id, owner_id)
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
