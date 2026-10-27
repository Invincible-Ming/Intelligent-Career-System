"""
智能求职 Multi-Agent 系统 - FastAPI 应用入口。

技术栈：
- FastAPI: Web 框架 (支持 SSE 流式推流与异步高并发)
- LangGraph: Multi-Agent 编排 (Fan-out 并发 + Self-Correction 纠错 + 状态检查点持久化)
- 阿里云百炼: 大模型服务与 Embedding
- Milvus & BM25: 混合检索 (Dense + Sparse + RRF + BGE Reranker)
- PostgreSQL: 关系型持久化存储 (AgentRun / AnalysisResult / LangGraph Checkpoints)
- MinIO: 对象存储 (原始简历/JD/知识库文档)
- MCP (Model Context Protocol): 容器内受限搜索、共享目录只读访问和聚合统计视图

"""

# SDK 环境清理必须先于应用依赖导入。
# ruff: noqa: E402
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# ----------------------------------------------------------------------
# 运行时 SDK 隔离：配置值一律来自 backend/.env（见 config.py），
# 但 openai / dashscope / httpx 等 SDK 在调用时仍会读取进程环境变量。
# 1) 终端残留的代理变量会让请求发往失效端口，报"模型服务不可用"；
# 2) 终端残留的 DASHSCOPE_BASE_URL 会把 Embedding 重定向到错误端点。
# 本服务只访问国内模型服务与本机/内网中间件，一律直连。
# ----------------------------------------------------------------------
for _sdk_env in ("http_proxy", "https_proxy", "all_proxy",
                 "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "DASHSCOPE_BASE_URL"):
    os.environ.pop(_sdk_env, None)
os.environ["NO_PROXY"] = "*"

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 导入工作流初始化函数（用于创建 LangGraph Postgres Checkpoint 表）
from app.agents.workflow import init_workflow

# 导入路由 - 按功能模块分组
from app.api.api import router as business_router
from app.api.auth_api import router as auth_router
from app.api.chat_api import router as chat_router
from app.api.credits_api import router as credits_router
from app.api.public_pages import router as public_pages_router
from app.core.access_log import AccessLogMiddleware
from app.core.config import settings
from app.core.database import check_database, close_database, init_database
from app.core.limits import RequestBodyTooLarge, SecurityLimitsMiddleware
from app.core.logging_setup import configure_logging, log_event
from app.evaluation.api import router as evaluation_router
from app.evaluation.llm import evaluation_llm
from app.services.analysis_cache import analysis_cache

# 导入底层依赖服务
from app.services.bailian import bailian_service
from app.services.bm25_service import bm25_service
from app.services.embed_cache import embed_cache
from app.services.experiment_store import experiment_store
from app.services.ima_service import ImaUnavailable, ima_service
from app.services.knowledge_index import knowledge_index_service
from app.services.mcp_tools import mcp_service
from app.services.milvus_service import milvus_service
from app.services.minio_service import minio_service
from app.services.redis_pool import close_pool, warm_up_redis_pool
from app.services.mobile_jobs import close_mobile_jobs
from app.api.mobile_api import router as mobile_router
from app.services.sandbox_client import check_runner

# ----------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------

# 每个请求日志都带 trace_id；DEBUG 下同时切到 JSON 便于定位。
configure_logging(
    level=settings.LOG_LEVEL,
    json_output=settings.LOG_JSON or settings.DEBUG,
)

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = 1800


async def periodic_cleanup() -> None:
    """定期清理已终态实验的进程内状态副本。

    Redis 不可用期间写入的状态会留在进程内 dict，
    恢复后由store.save 回写，这里只负责不让它无限增长。
    """

    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            await experiment_store.clear_expired()
        except asyncio.CancelledError:
            logger.info("评测状态清理任务已停止")
            raise
        except Exception as exc:
            # 清理是尽力而为，失败不影响业务。
            logger.warning("评测状态清理跳过：%s", type(exc).__name__)


# ----------------------------------------------------------------------
# 应用生命周期管理 (Lifespan)
# ----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    应用启动和关闭时的生命周期管理。

    启动时初始化：
    - PostgreSQL 业务表与连接池
    - LangGraph PostgresSaver 状态持久化检查点表
    - MinIO 对象存储 Bucket
    - Milvus 向量数据库集合与索引
    - BM25 稀疏倒排索引构建
    - 受限 MCP 容器工具 (Search, FileSystem, Postgres)

    关闭时清理：
    - MCP 客户端配置清理
    - 百炼 HTTP 客户端连接
    - Milvus 客户端连接
    - PostgreSQL 业务与 Checkpointer 连接池
    """

    app.state.startup_errors = []
    # 先预热共享池：冷启动时池内无连接，并发建连会触发
    # TimeoutError 并产生大量无谓的「缓存不可用」告警。
    await warm_up_redis_pool()
    await analysis_cache.initialize()
    if settings.KNOWLEDGE_PROVIDER in ("milvus", "subscription", "credits"):
        await embed_cache.initialize()
    await experiment_store.initialize()
    if not await check_runner():
        message = (
            "文档沙箱 Runner 不可用，上传将失败。"
            f"请先启动：python deploy/sandbox/start.py"
            f"（预期 socket：{settings.SANDBOX_RUNNER_SOCKET}）"
        )
        app.state.startup_errors.append(message)
        # 启动日志里必须显眼：否则用户只会在上传时看到 500，
        # 根本不知道要先起 Runner。
        log_event(
            logger,
            logging.ERROR,
            message,
            socket=settings.SANDBOX_RUNNER_SOCKET,
        )

    # 定期回收终态实验的进程内降级副本；Redis 侧终态键自带 TTL。
    app.state.cleanup_task = asyncio.create_task(periodic_cleanup())

    logger.info("=" * 60)
    logger.info("智能求职 Multi-Agent 系统启动中...")
    logger.info("=" * 60)

    # 1. 初始化 PostgreSQL 基础业务表
    try:
        await init_database()
        logger.info("✓ PostgreSQL 业务数据表初始化完成")
    except Exception as exc:
        error_msg = f"PostgreSQL 初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 2. 初始化 LangGraph 状态持久化表 (Checkpoints)
    try:
        await init_workflow()
        logger.info("✓ LangGraph PostgreSQL 状态持久化检查点初始化完成")
    except Exception as exc:
        error_msg = f"LangGraph 检查点表初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 3. 初始化 MinIO
    try:
        await minio_service.initialize()
        logger.info("✓ MinIO 初始化完成")
    except Exception as exc:
        error_msg = f"MinIO 初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 4/5. 仅初始化所选知识库；ima 模式完全不连接本地向量服务。
    if settings.KNOWLEDGE_PROVIDER in ("ima", "subscription", "credits"):
        try:
            library = await ima_service.library_info()
            logger.info("✓ ima 知识库连接完成：%s", library["name"])
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ImaUnavailable) else "请检查配置及权限"
            app.state.startup_errors.append(f"ima 知识库连接失败：{reason}")
            logger.error("ima 初始化失败（%s）：%s", type(exc).__name__, reason)
    if settings.KNOWLEDGE_PROVIDER in ("milvus", "subscription", "credits"):
        try:
            await milvus_service.initialize()
            logger.info("✓ Milvus 初始化完成")
        except Exception as exc:
            error_msg = f"Milvus 初始化失败: {exc}"
            app.state.startup_errors.append(error_msg)
            logger.error(f"✗ {error_msg}")
        if settings.KNOWLEDGE_PROVIDER == "milvus":
            try:
                chunks = await milvus_service.list_chunks()
                await bm25_service.rebuild(chunks)
                logger.info(f"✓ BM25 索引初始化完成，共加载 {bm25_service.chunk_count} 个 Chunk")
            except Exception as exc:
                error_msg = f"BM25 索引初始化失败: {exc}"
                app.state.startup_errors.append(error_msg)
                logger.error(f"✗ {error_msg}")

    app.state.knowledge_index_task = (asyncio.create_task(knowledge_index_service.run())
                                      if settings.KNOWLEDGE_PROVIDER in ("subscription", "credits") else None)

    # 6. 🌟 初始化 MCP 外部工具能力子进程
    try:
        await mcp_service.initialize()
        app.state.startup_errors.extend(mcp_service.startup_errors)
        logger.info("MCP 工具初始化结束，共加载 %s 个受限工具", len(mcp_service.tools))
    except Exception as exc:
        error_msg = f"MCP 工具初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 启动就绪提示
    if app.state.startup_errors:
        logger.warning("⚠ 应用已启动，但部分底层存储或扩展服务不可用")
        logger.warning(f"异常服务数: {len(app.state.startup_errors)}")
    else:
        logger.info("=" * 60)
        logger.info("✓ 智能求职系统全链路服务启动就绪！")
        logger.info(f"✓ API 交互文档: http://{settings.APP_HOST}:{settings.APP_PORT}/docs")
        logger.info(f"✓ SSE 流式匹配: http://{settings.APP_HOST}:{settings.APP_PORT}{settings.API_PREFIX}/match/stream")
        logger.info(f"✓ 健康检查端点: http://{settings.APP_HOST}:{settings.APP_PORT}/health")
        logger.info("=" * 60)

    yield

    # 服务平滑关闭
    logger.info("智能求职系统正在平滑关闭...")
    await close_mobile_jobs()

    cleanup_task = getattr(app.state, "cleanup_task", None)
    if cleanup_task is not None and not cleanup_task.done():
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("清理任务退出异常：%s", type(exc).__name__)

    if app.state.knowledge_index_task is not None:
        app.state.knowledge_index_task.cancel()
        try:
            await app.state.knowledge_index_task
        except asyncio.CancelledError:
            pass

    await mcp_service.close()
    await evaluation_llm.close()
    await bailian_service.close()
    await embed_cache.close()
    await experiment_store.close()
    await analysis_cache.close()
    # 共享 Redis 池由这里统一关闭——各服务只丢弃自己的客户端引用。
    await close_pool()
    await ima_service.close()
    if settings.KNOWLEDGE_PROVIDER in ("milvus", "subscription", "credits"):
        await milvus_service.close()
    await close_database()

    logger.info("✓ 智能求职系统所有连接已释放并安全关闭")


# ----------------------------------------------------------------------
# FastAPI 应用实例
# ----------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "基于 LangGraph、阿里云百炼、PostgreSQL 和 MinIO，支持 ima 或 Milvus 知识库，"
        "构建的企业级智能求职 Multi-Agent & RAG 系统。\n\n"
        "核心能力：\n"
        "- 📄 **文档处理**：沙箱解析、OCR兜底，Milvus 模式生成向量\n"
        "- 🔍 **知识检索**：ima 内容搜索，或 Milvus + BM25 + RRF + BGE 本地混合检索\n"
        "- 🎯 **架构并行匹配**：Fan-out 并发抽取 + Self-Correction 自我纠错闭环\n"
        "- ⏸️ **人机协同中断恢复**：支持 PostgreSQL 状态检查点与人工干预/打回\n"
        "- ⚡ **SSE 实时流**：Server-Sent Events 毫秒级推送 Agent 思考与节点流转\n"
        "- 🎤 **面试预测**：基于岗位报告与能力缺口深挖生成真题\n"
        "- 📚 **学习规划**：定制化按周学习路径规划\n"
        "- 📊 **实验评测**：Golden Dataset 检索消融评测套件\n"
        "- 🔎 **受限 MCP 工具**：容器内公开搜索、专用目录只读访问和白名单统计视图\n"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ----------------------------------------------------------------------
# 中间件配置
# ----------------------------------------------------------------------

# 顺序敏感：AccessLog 必须最先添加（最外层），
# 这样被限流、被校验拒绝的请求也会带上 trace_id，便于排查。
app.add_middleware(AccessLogMiddleware)
app.add_middleware(SecurityLimitsMiddleware)

# CORS 跨域配置（保障前端 SSE EventSource / Fetch Stream 请求正常通行）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestBodyTooLarge)
async def oversized_request(request, exc):
    return JSONResponse({"detail": "请求体超过大小限制"}, status_code=413)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Do not echo passwords, prompts, bearer tokens, or uploaded data.
    errors = [{"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=422)


app.include_router(auth_router, prefix=settings.API_PREFIX)
app.include_router(credits_router, prefix=settings.API_PREFIX)
app.include_router(public_pages_router)
app.include_router(mobile_router, prefix=settings.API_PREFIX)

# ----------------------------------------------------------------------
# 路由注册 - 按模块分组
# ----------------------------------------------------------------------

# 业务 API（文档管理、向量/混合检索、同步与 SSE 流式匹配、中断恢复、面试生成、学习计划）
app.include_router(
    business_router,
    prefix=settings.API_PREFIX,
    tags=["业务 API"],
)

# 对话 API（智能对话助手与多轮检索问答）
app.include_router(
    chat_router,
    prefix=settings.API_PREFIX,
    tags=["对话 API"],
)

# 评测 API（RAG 评测与消融实验系统）
app.include_router(
    evaluation_router,
    prefix=settings.API_PREFIX,
    tags=["评测 API"],
)


# ----------------------------------------------------------------------
# 全局端点
# ----------------------------------------------------------------------

@app.get(
    "/",
    tags=["系统"],
    summary="系统首页",
    description="返回系统元数据、文档入口与主要业务端点路径",
)
async def root() -> dict:
    """系统首页，返回系统可用模块信息。"""

    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "description": "智能求职 Multi-Agent & RAG 系统",
        "links": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "modules": {
            "documents": f"{settings.API_PREFIX}/documents",
            "hybrid_search": f"{settings.API_PREFIX}/search/hybrid",
            "match_sync": f"{settings.API_PREFIX}/match",
            "match_resume": f"{settings.API_PREFIX}/match/resume",
            "match_stream_sse": f"{settings.API_PREFIX}/match/stream",
            "match_stream_resume_sse": f"{settings.API_PREFIX}/match/stream/resume",
            "interview_plan": f"{settings.API_PREFIX}/interview",
            "learning_plan": f"{settings.API_PREFIX}/learning-plan",
            "chat": f"{settings.API_PREFIX}/chat/completions",
            "evaluation": f"{settings.API_PREFIX}/evaluation/experiments",
        },
    }


# ----------------------------------------------------------------------
# 应用入口
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("以开发调试模式启动应用...")

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )


@app.get('/health', tags=['系统'])
async def health():
    ok = await check_database()
    return JSONResponse({'status': 'ok' if ok else 'unavailable'}, status_code=200 if ok else 503)
