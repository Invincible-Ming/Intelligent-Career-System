"""pytest 全局前置。

测试分为两类：

- 纯单元测试（切块、限流计算、前端鉴权断言），只依赖 fastapi/pydantic；
- 集成测试使用临时 PostgreSQL、Redis；模型/向量库/对象存储调用由用例模拟。

这里的目标是：环境缺依赖时，让需要它的测试被skip 并说明原因，
而不是让整个 tests/ 目录在收集阶段就崩溃。
"""

import importlib.util
import os
from urllib.parse import urlsplit

import pytest

from app.core.config import Settings, settings

# 在测试模块导入数据库引擎之前重置配置，避免读取开发者 .env 的真实资源/凭据。
settings.__dict__.update(Settings(_env_file=None).__dict__)
settings.DATABASE_URL = os.environ.get(
    "CAREER_TEST_DATABASE_URL",
    "postgresql+asyncpg://career_ci:career_ci_test@127.0.0.1:15432/career_ci_test",
)
settings.REDIS_URL = os.environ.get("CAREER_TEST_REDIS_URL", "redis://127.0.0.1:16379/15")
for endpoint in (settings.DATABASE_URL, settings.REDIS_URL):
    parsed = urlsplit(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("测试只允许连接本机临时服务")
if "test" not in urlsplit(settings.DATABASE_URL).path:
    raise RuntimeError("测试数据库名称必须包含 test")
settings.DASHSCOPE_API_KEY = ""
settings.BAILIAN_BASE_URL = "http://127.0.0.1:1/v1"
settings.BAIDU_MAP_MCP_URL = ""


def _missing(*modules: str) -> list[str]:
    absent = [name for name in modules if importlib.util.find_spec(name) is None]
    return absent


# 需要完整后端依赖链的模块。
_BACKEND_STACK = (
    "sqlalchemy",
    "redis",
    "langgraph",
    "langchain_core",
    "psycopg",
)

# 需要文档解析栈。
_DOCUMENT_STACK = (
    "fitz",  # PyMuPDF
    "docx",
    "openpyxl",
)

# 需要向量库客户端。
_VECTOR_STACK = ("pymilvus",)

# 沙箱与 MCP 相关。
_SANDBOX_STACK = ("docker",)

# BM25 内存索引相关。
_BM25_STACK = ("jieba", "rank_bm25")

# 前端 career_ui 模块（需要 streamlit；测试里只用纯函数，不启运行时）。
_FRONTEND_STACK = ("streamlit",)

# Embedding 缓存相关：bailian 会连带导入 dashscope / openai / limits(langchain_core)。
_EMBED_STACK = ("redis", "dashscope", "openai", "langchain_core")

# 测试模块在 import阶段就会连带导入 app 包，因此缺依赖时必须在
# 收集阶段就忽略该文件——fixture 的 skip 发生在收集之后，来不及。
_MODULE_STACKS: dict[str, tuple[str, ...]] = {
    "test_analysis_cache.py": _BACKEND_STACK + _VECTOR_STACK,
    "test_auth_limits.py": _BACKEND_STACK,
    "test_chat_summary.py": _BACKEND_STACK,
    "test_match_confirmation.py": _BACKEND_STACK,
    "test_match_speed.py": _BACKEND_STACK,
    "test_workflow_approval.py": _BACKEND_STACK,
    "test_sandbox.py": _SANDBOX_STACK,
    "test_mcp_security.py": _BACKEND_STACK,
    "test_document_pipeline.py": _DOCUMENT_STACK,
    "test_bm25_owner_isolation.py": _BM25_STACK,
    "test_embed_cache.py": _EMBED_STACK,
    "test_redis_pool.py": _EMBED_STACK,
    "test_static_imports.py": (),  # 纯 AST 静态检查，无需任何依赖
    "test_embedding_model_config.py": _EMBED_STACK,
    "test_frontend_modules.py": _FRONTEND_STACK,
    "test_frontend_auth.py": _FRONTEND_STACK,
    "test_frontend_match_page.py": _FRONTEND_STACK,
    "test_frontend_match_jd_source.py": _FRONTEND_STACK,
}

collect_ignore = [name for name, stack in _MODULE_STACKS.items() if _missing(*stack)]


@pytest.fixture(scope="session")
def require_backend_stack():
    """缺少后端核心依赖时跳过整个模块。"""

    absent = _missing(*_BACKEND_STACK)
    if absent:
        pytest.skip(f"缺少后端依赖：{', '.join(absent)}")


@pytest.fixture(scope="session")
def require_document_stack():
    absent = _missing(*_DOCUMENT_STACK)
    if absent:
        pytest.skip(f"缺少文档解析依赖：{', '.join(absent)}")


@pytest.fixture(scope="session")
def require_vector_stack():
    absent = _missing(*_VECTOR_STACK)
    if absent:
        pytest.skip(f"缺少向量库依赖：{', '.join(absent)}")


@pytest.fixture(scope="session")
def require_sandbox_stack():
    absent = _missing(*_SANDBOX_STACK)
    if absent:
        pytest.skip(f"缺少沙箱依赖：{', '.join(absent)}")


@pytest.fixture
def anyio_backend():
    """让 pytest-anyio 风格的异步测试固定跑在 asyncio 上。"""

    return "asyncio"
