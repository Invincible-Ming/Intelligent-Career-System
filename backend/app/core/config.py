"""
应用配置。

从 backend/.env 读取 PostgreSQL、百炼、Milvus 和 MinIO 配置。
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目统一配置。

    配置唯一来源是 backend/.env（加上代码默认值）。
    刻意排除了操作系统环境变量来源：启动终端 export 的旧值（如失效的
    DASHSCOPE_API_KEY、代理地址）不应覆盖 .env，否则行为随启动环境漂移。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
    ):
        # 去掉 env_settings：不读取操作系统环境变量，只认 .env 文件。
        return (init_settings, dotenv_settings, file_secret_settings)

    # FastAPI
    APP_NAME: str = "智能求职 Multi-Agent 系统"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    # 身份与资源预算，由后端强制执行。
    AUTH_REGISTRATION_ENABLED: bool = True
    AUTH_SESSION_HOURS: int = 12
    AUTH_COOKIE_NAME: str = "career_session"
    AUTH_COOKIE_SECURE: bool = False  # HTTPS 请求自动启用；反向代理部署时显式设为 True。
    API_REQUESTS_PER_MINUTE: int = 120
    AUTH_REQUESTS_PER_MINUTE: int = 30
    GENERATION_REQUESTS_PER_MINUTE: int = 10
    GENERATION_REQUESTS_PER_DAY: int = 200
    GLOBAL_GENERATION_REQUESTS_PER_DAY: int = 1000
    MAX_CONCURRENT_OPERATIONS: int = 2
    CHAT_MAX_INPUT_CHARS: int = 4000
    CHAT_CONTEXT_BUDGET: int = 18000
    CHAT_HISTORY_MESSAGES: int = 40
    CHAT_SUMMARY_TRIGGER_MESSAGES: int = 24
    CHAT_SUMMARY_TRIGGER_BYTES: int = 10000
    CHAT_SUMMARY_KEEP_MESSAGES: int = 8
    CHAT_SUMMARY_MAX_OUTPUT_TOKENS: int = 800
    CHAT_MAX_OUTPUT_TOKENS: int = 2048
    MODEL_MAX_OUTPUT_TOKENS: int = 4096
    MODEL_INPUT_BUDGET: int = 200000
    MAX_MODEL_CALLS_PER_OPERATION: int = 24
    CHAT_TOTAL_TIMEOUT: float = 90.0
    ANALYSIS_TOTAL_TIMEOUT: float = 180.0
    MATCH_TOTAL_TIMEOUT: float = 600.0
    MATCH_MODEL_TIMEOUT: float = 120.0
    MATCH_REPORT_TIMEOUT: float = 75.0
    MATCH_VERIFY_TIMEOUT: float = 60.0
    MATCH_REASONING_EFFORT: Literal["none", "low", "medium", "xhigh"] = "none"
    ANALYSIS_CACHE_ENABLED: bool = True
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    ANALYSIS_CACHE_TTL: int = 86400
    REDIS_SOCKET_TIMEOUT: float = 0.5
    UPLOAD_TOTAL_TIMEOUT: float = 120.0
    API_TOTAL_TIMEOUT: float = 30.0
    JSON_BODY_MAX_BYTES: int = 160000
    MAX_DOCUMENTS_PER_USER: int = 200
    MAX_CHUNKS_PER_DOCUMENT: int = 256
    MAX_DOCUMENT_TEXT_CHARS: int = 30000
    MAX_PARSED_DOCUMENT_CHARS: int = 200000
    SANDBOX_DOCUMENTS_ENABLED: bool = True
    SANDBOX_RUNNER_SOCKET: str = str(Path(__file__).resolve().parents[3] / "deploy/.local/sandbox.sock")

    # PostgreSQL
    DATABASE_URL: str = (
        "postgresql+asyncpg://career_user:career_password"
        "@localhost:5432/career_db"
    )

    # 阿里云百炼
    DASHSCOPE_API_KEY: str = ""

    # 聊天模型继续使用百炼 OpenAI 兼容接口。
    BAILIAN_BASE_URL: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    BAILIAN_CHAT_MODEL: str = "qwen3.8-flash"
    BAILIAN_SUMMARY_MODEL: str = "deepseek-v4-flash-0731"

    # 文档解析 OCR 使用的视觉模型；qwen-vl-ocr 系列为文档文字提取专精。
    BAILIAN_OCR_MODEL: str = "qwen-vl-ocr-latest"

    # Embedding 使用 DashScope SDK。
    BAILIAN_EMBEDDING_MODEL: str = "qwen3.7-text-embedding"
    EMBEDDING_DIMENSION: int = 1024

    MODEL_TIMEOUT: float = 60.0
    MODEL_MAX_RETRIES: int = 2

    # Milvus
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_TOKEN: str = ""
    MILVUS_COLLECTION: str = "career_documents"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "career_minio"
    MINIO_SECRET_KEY: str = "career_minio_password"
    MINIO_SECURE: bool = False
    MINIO_BUCKET: str = "career-documents"

    # 文档处理
    MAX_UPLOAD_SIZE_MB: int = 20
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # 检索
    RETRIEVAL_TOP_K: int = 5

    # MCP：仅允许经过加固的本地镜像；绝不回退到本机 npx。
    MCP_ENABLED: bool = True
    MCP_IMAGE: str = "career-mcp:1"
    MCP_SEARCH_SOCKET_VOLUME: str = "career-mcp-search-socket"
    # 百度地图 MCP（远程 Streamable HTTP，AK 在地址中携带）；留空则不加载
    BAIDU_MAP_MCP_URL: str = ""
    MCP_STARTUP_TIMEOUT: float = 30.0
    MCP_RESEARCH_TIMEOUT: float = 45.0
    MATCH_SEARCH_TIMEOUT: float = 20.0
    MCP_RESEARCH_MAX_STEPS: int = 6

    # BGE CrossEncoder 重排
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = (
        "BAAI/bge-reranker-v2-m3"
    )
    RERANKER_DEVICE: str = "auto"
    RERANKER_CANDIDATE_K: int = 8
    RERANKER_TOP_K: int = 5
    RERANKER_BATCH_SIZE: int = 1
    RERANKER_MAX_LENGTH: int = 256

    # Streamlit 跨域地址，多个地址使用英文逗号分隔。
    CORS_ORIGINS: str = (
        "http://localhost:8501,"
        "http://127.0.0.1:8501"
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """校验配置之间的约束。"""

        if not Path(self.SANDBOX_RUNNER_SOCKET).is_absolute():
            raise ValueError("SANDBOX_RUNNER_SOCKET 必须是绝对路径")

        if self.MCP_STARTUP_TIMEOUT <= 0 or self.MCP_RESEARCH_TIMEOUT <= 0 or self.MCP_RESEARCH_MAX_STEPS < 1:
            raise ValueError("MCP 超时和步骤限制必须为正数")

        limits = (self.AUTH_SESSION_HOURS, self.API_REQUESTS_PER_MINUTE, self.AUTH_REQUESTS_PER_MINUTE,
                  self.GENERATION_REQUESTS_PER_MINUTE, self.GENERATION_REQUESTS_PER_DAY,
                  self.GLOBAL_GENERATION_REQUESTS_PER_DAY, self.MAX_CONCURRENT_OPERATIONS,
                  self.CHAT_MAX_INPUT_CHARS, self.CHAT_CONTEXT_BUDGET, self.CHAT_HISTORY_MESSAGES,
                  self.CHAT_SUMMARY_TRIGGER_MESSAGES, self.CHAT_SUMMARY_TRIGGER_BYTES,
                  self.CHAT_SUMMARY_KEEP_MESSAGES, self.CHAT_SUMMARY_MAX_OUTPUT_TOKENS,
                  self.CHAT_MAX_OUTPUT_TOKENS, self.MODEL_MAX_OUTPUT_TOKENS, self.MODEL_INPUT_BUDGET,
                  self.MAX_MODEL_CALLS_PER_OPERATION, self.CHAT_TOTAL_TIMEOUT, self.ANALYSIS_TOTAL_TIMEOUT,
                  self.MATCH_TOTAL_TIMEOUT, self.MATCH_MODEL_TIMEOUT,
                  self.MATCH_REPORT_TIMEOUT, self.MATCH_VERIFY_TIMEOUT, self.MATCH_SEARCH_TIMEOUT,
                  self.ANALYSIS_CACHE_TTL, self.REDIS_SOCKET_TIMEOUT,
                  self.UPLOAD_TOTAL_TIMEOUT, self.API_TOTAL_TIMEOUT, self.JSON_BODY_MAX_BYTES,
                  self.MAX_DOCUMENTS_PER_USER, self.MAX_CHUNKS_PER_DOCUMENT, self.MAX_DOCUMENT_TEXT_CHARS,
                  self.MAX_PARSED_DOCUMENT_CHARS)
        if any(value <= 0 for value in limits):
            raise ValueError("认证与资源限制必须为正数")
        if self.CHAT_CONTEXT_BUDGET < self.CHAT_MAX_INPUT_CHARS * 4 + 1024:
            raise ValueError("聊天上下文预算必须容纳最大输入及系统提示")
        if self.CHAT_SUMMARY_KEEP_MESSAGES < 2 or self.CHAT_SUMMARY_KEEP_MESSAGES >= self.CHAT_SUMMARY_TRIGGER_MESSAGES:
            raise ValueError("摘要保留消息数必须至少为 2 且小于摘要触发消息数")

        if self.CHUNK_SIZE < 64 or not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP 必须小于 CHUNK_SIZE"
            )

        if not 256 <= self.EMBEDDING_DIMENSION <= 2560:
            raise ValueError(
                "qwen3.7-text-embedding 的向量维度"
                "必须在 256～2560 之间"
            )

        if self.MODEL_MAX_RETRIES < 0:
            raise ValueError(
                "MODEL_MAX_RETRIES 不能小于 0"
            )
        if self.RERANKER_MAX_LENGTH < 32:
            raise ValueError("RERANKER_MAX_LENGTH 不能小于 32")

        if self.RERANKER_DEVICE not in {
            "auto",
            "mps",
            "cpu",
        }:
            raise ValueError(
                "RERANKER_DEVICE 必须是 auto、mps 或 cpu"
            )

        if self.RERANKER_CANDIDATE_K < self.RERANKER_TOP_K:
            raise ValueError(
                "RERANKER_CANDIDATE_K 不能小于 "
                "RERANKER_TOP_K"
            )

        return self

    @property
    def cors_origin_list(self) -> list[str]:
        """将 CORS 字符串转换为地址列表。"""

        return [
            item.strip()
            for item in self.CORS_ORIGINS.split(",")
            if item.strip()
        ]

    @property
    def max_upload_size_bytes(self) -> int:
        """最大上传文件字节数。"""

        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def match_model_options(self) -> dict:
        # Only pass this option to models with documented provider support.
        if self.BAILIAN_CHAT_MODEL.startswith("qwen3.8-flash"):
            return {"reasoning_effort": self.MATCH_REASONING_EFFORT}
        return {}


@lru_cache
def get_settings() -> Settings:
    """创建并缓存配置对象。"""

    return Settings()


settings = get_settings()
