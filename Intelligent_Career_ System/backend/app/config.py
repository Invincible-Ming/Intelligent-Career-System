"""
应用配置。

从 backend/.env 读取 PostgreSQL、百炼、Milvus 和 MinIO 配置。
"""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目统一配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # FastAPI
    APP_NAME: str = "智能求职 Multi-Agent 系统"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    API_PREFIX: str = "/api"

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
    BAILIAN_CHAT_MODEL: str = "qwen-plus"

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

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
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


@lru_cache
def get_settings() -> Settings:
    """创建并缓存配置对象。"""

    return Settings()


settings = get_settings()
