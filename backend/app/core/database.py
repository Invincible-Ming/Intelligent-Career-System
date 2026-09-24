"""
PostgreSQL 异步连接配置与状态检查点池。
"""

from collections.abc import AsyncGenerator
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""

    pass


# =====================================================================
# 1. SQLAlchemy ORM 引擎配置
# =====================================================================
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # 不在 SQL 调试日志中记录认证散列或用户资料。
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# =====================================================================
# 2. LangGraph 专用检查点连接池 (Checkpointer Pool)
# =====================================================================
# psycopg 驱动要求协议头为 postgresql://，且在 Mac 下避免 localhost 解析为 ::1 (IPv6)
_raw_url = str(settings.DATABASE_URL)
PSYCOPG_DATABASE_URL = (
    _raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    .replace("@localhost:", "@127.0.0.1:")
    .replace("@localhost/", "@127.0.0.1/")
)

checkpointer_pool = AsyncConnectionPool(
    conninfo=PSYCOPG_DATABASE_URL,
    max_size=20,
    open=False,  # 🌟 关键：延迟打开，避免在 import 模块阶段提前连接导致 pool-1 报错
    kwargs={
        "autocommit": True,  # LangGraph Saver 状态持久化的硬性要求
        "prepare_threshold": 0,  # 避免连接复用时的预编译冲突
    },
)


# =====================================================================
# 3. 依赖注入与数据库生命周期管理
# =====================================================================
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """为 FastAPI 请求提供数据库会话。"""

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """初始化数据库表并开启 LangGraph 连接池。"""

    # 1. 启动检查点异步连接池
    if checkpointer_pool.closed:
        await checkpointer_pool.open()

    # 2. 导入模型并同步建表
    from app import models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        from sqlalchemy import text
        for table in ("documents", "agent_runs", "conversations"):
            await connection.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES users(id)"))
            await connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_owner_id ON {table}(owner_id)"))
            has_legacy = (await connection.execute(text(f"SELECT EXISTS(SELECT 1 FROM {table} WHERE owner_id IS NULL)"))).scalar()
            if not has_legacy:
                await connection.execute(text(f"ALTER TABLE {table} ALTER COLUMN owner_id SET NOT NULL"))
        await connection.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS context_summary TEXT"))
        await connection.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE"))
        await connection.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary_until_message_id UUID"))
        await connection.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary_version INTEGER NOT NULL DEFAULT 0"))
        await connection.execute(text("DELETE FROM auth_sessions WHERE expires_at < now()"))
        await connection.execute(text("DELETE FROM operation_leases WHERE expires_at < now()"))
        await connection.execute(text("DELETE FROM rate_limit_buckets WHERE window_start < now() - interval '2 days'"))


async def check_database() -> bool:
    """检查 PostgreSQL 是否可以正常访问。"""

    from sqlalchemy import text

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def close_database() -> None:
    """关闭数据库连接池（同时释放 SQLAlchemy 引擎与 LangGraph 连接池）。"""

    await engine.dispose()
    if not checkpointer_pool.closed:
        await checkpointer_pool.close()
