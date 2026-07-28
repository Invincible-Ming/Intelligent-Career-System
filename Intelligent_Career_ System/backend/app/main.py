"""
智能求职 Multi-Agent 系统 - FastAPI 应用入口。

技术栈：
- FastAPI: Web 框架
- LangGraph: Multi-Agent 编排
- 阿里云百炼: 大模型服务
- Milvus: 向量数据库
- PostgreSQL: 关系数据库
- MinIO: 对象存储

启动命令：
    python -m uvicorn main:app --reload
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import check_database, close_database, init_database

# 导入服务
from app.bailian import bailian_service
from app.bm25_service import bm25_service
from app.milvus_service import milvus_service
from app.minio_service import minio_service

# 导入路由 - 按功能模块分组
from app.api import router as business_router
from app.chat_api import router as chat_router
from app.evaluation.api import router as evaluation_router

# ----------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 应用生命周期管理
# ----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    应用启动和关闭时的生命周期管理。
    
    启动时初始化：
    - PostgreSQL 数据库
    - MinIO 对象存储
    - Milvus 向量数据库
    - BM25 检索索引
    
    关闭时清理：
    - 百炼客户端连接
    - Milvus 连接
    - 数据库连接池
    """

    # 用于记录启动错误（不阻塞启动）
    app.state.startup_errors = []

    logger.info("=" * 60)
    logger.info("智能求职系统启动中...")
    logger.info("=" * 60)

    # 1. 初始化 PostgreSQL
    try:
        await init_database()
        logger.info("✓ PostgreSQL 初始化完成")
    except Exception as exc:
        error_msg = f"PostgreSQL 初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 2. 初始化 MinIO
    try:
        await minio_service.initialize()
        logger.info("✓ MinIO 初始化完成")
    except Exception as exc:
        error_msg = f"MinIO 初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 3. 初始化 Milvus
    try:
        await milvus_service.initialize()
        logger.info("✓ Milvus 初始化完成")
    except Exception as exc:
        error_msg = f"Milvus 初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 4. 初始化 BM25 索引
    try:
        chunks = await milvus_service.list_chunks()
        await bm25_service.rebuild(chunks)
        logger.info(f"✓ BM25 索引初始化完成，共 {bm25_service.chunk_count} 个 Chunk")
    except Exception as exc:
        error_msg = f"BM25 索引初始化失败: {exc}"
        app.state.startup_errors.append(error_msg)
        logger.error(f"✗ {error_msg}")

    # 启动完成提示
    if app.state.startup_errors:
        logger.warning("⚠ 应用已启动，但部分基础服务不可用")
        logger.warning(f"失败的服务数: {len(app.state.startup_errors)}")
    else:
        logger.info("=" * 60)
        logger.info("✓ 智能求职系统启动完成！")
        logger.info(f"✓ API 文档: http://{settings.APP_HOST}:{settings.APP_PORT}/docs")
        logger.info(f"✓ 健康检查: http://{settings.APP_HOST}:{settings.APP_PORT}/health")
        logger.info("=" * 60)

    yield

    # 关闭时清理资源
    logger.info("智能求职系统关闭中...")
    
    await bailian_service.close()
    await milvus_service.close()
    await close_database()
    
    logger.info("✓ 智能求职系统已关闭")


# ----------------------------------------------------------------------
# FastAPI 应用实例
# ----------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "基于 LangGraph、阿里云百炼、Milvus、PostgreSQL 和 MinIO "
        "构建的智能求职 Multi-Agent 系统。\n\n"
        "核心功能：\n"
        "- 📄 文档管理（简历/JD/知识库）\n"
        "- 🔍 知识库检索（Dense/BM25/Hybrid）\n"
        "- 🎯 岗位匹配分析\n"
        "- 🎤 面试问题生成\n"
        "- 📚 学习计划制定\n"
        "- 💬 智能对话助手\n"
        "- 📊 RAG 评测系统\n"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ----------------------------------------------------------------------
# 中间件配置
# ----------------------------------------------------------------------

# CORS 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 路由注册 - 按模块分组
# ----------------------------------------------------------------------

# 业务 API（文档、匹配、面试、学习）
app.include_router(
    business_router,
    prefix=settings.API_PREFIX,
    tags=["业务 API"],
)

# 对话 API（智能对话助手）
app.include_router(
    chat_router,
    prefix=settings.API_PREFIX,
    tags=["对话 API"],
)

# 评测 API（RAG 评测系统）
app.include_router(
    evaluation_router,
    prefix=settings.API_PREFIX,
    tags=["评测 API"],
)


# ----------------------------------------------------------------------
# 静态文件服务
# ----------------------------------------------------------------------

try:
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir)),
            name="static",
        )
        logger.info(f"✓ 静态文件服务已启用: /static")
except Exception as exc:
    logger.warning(f"静态文件服务启用失败: {exc}")


# ----------------------------------------------------------------------
# 全局端点
# ----------------------------------------------------------------------


@app.get(
    "/",
    tags=["系统"],
    summary="系统首页",
    description="返回系统基本信息和文档链接",
)
async def root() -> dict:
    """系统首页，返回基本信息。"""
    
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "description": "智能求职 Multi-Agent 系统",
        "links": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "openapi": "/openapi.json",
        },
        "modules": {
            "business": f"{settings.API_PREFIX}/documents",
            "chat": f"{settings.API_PREFIX}/chat/completions",
            "evaluation": f"{settings.API_PREFIX}/evaluation/experiments",
        },
    }


@app.get(
    "/health",
    tags=["系统"],
    summary="健康检查",
    description="检查系统各服务的运行状态",
)
async def health() -> dict:
    """
    健康检查端点，检查所有依赖服务的状态。
    
    返回格式：
    {
        "status": "healthy" | "degraded",
        "services": {
            "postgresql": true/false,
            "milvus": true/false,
            "minio": true/false
        }
    }
    """
    
    # 并发检查所有服务
    postgres_ok, milvus_ok, minio_ok = await asyncio.gather(
        check_database(),
        milvus_service.health_check(),
        check_minio(),
        return_exceptions=True,
    )
    
    # 处理异常
    services = {
        "postgresql": postgres_ok if isinstance(postgres_ok, bool) else False,
        "milvus": milvus_ok if isinstance(milvus_ok, bool) else False,
        "minio": minio_ok if isinstance(minio_ok, bool) else False,
    }
    
    # 判断整体状态
    all_healthy = all(services.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services,
        "timestamp": asyncio.get_event_loop().time(),
    }


async def check_minio() -> bool:
    """检查 MinIO Bucket 是否可以访问。"""
    
    try:
        return await asyncio.to_thread(
            minio_service.client.bucket_exists,
            settings.MINIO_BUCKET,
        )
    except Exception:
        return False


# ----------------------------------------------------------------------
# 应用入口
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("以开发模式启动应用...")
    
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
