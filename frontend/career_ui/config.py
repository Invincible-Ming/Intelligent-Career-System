"""前端配置与常量。

集中放置，避免各模块各自读环境变量导致配置漂移。
"""
from __future__ import annotations

import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

SHORT_TIMEOUT = 20
LONG_TIMEOUT = 600

DOCUMENT_TYPE_NAMES = {
    "resume": "简历",
    "job_description": "岗位描述",
    "knowledge": "知识资料",
}

DIFFICULTY_NAMES = {
    "junior": "初级",
    "intermediate": "中级",
    "senior": "高级",
}

SEARCH_MODE_NAMES = {
    "hybrid": "混合检索（Milvus + BM25 + RRF + BGE）",
    "dense": "语义检索（Milvus）",
    "bm25": "关键词检索（BM25）",
}