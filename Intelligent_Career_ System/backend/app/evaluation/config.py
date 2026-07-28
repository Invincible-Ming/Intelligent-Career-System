"""
评测配置模型。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """检索配置。"""

    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    chunk_size: int = Field(default=800, ge=200, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)
    enable_reranker: bool = True
    reranker_top_k: int = Field(default=5, ge=1, le=10)


class ReflectionConfig(BaseModel):
    """反思配置。"""

    enable_reflection: bool = False
    max_iterations: int = Field(default=2, ge=1, le=5)
    reflection_prompt: str = (
        "请审查上一次的回答，如果存在事实错误、逻辑矛盾或不够准确的地方，"
        "请基于检索到的知识进行修正。"
    )


class EvaluationConfig(BaseModel):
    """单次评测配置。"""

    name: str = Field(description="配置名称，如 'baseline' 或 'optimized'")
    search: SearchConfig = Field(default_factory=SearchConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)

    class Config:
        frozen = True


class ExperimentConfig(BaseModel):
    """A/B 实验配置。"""

    experiment_name: str = Field(description="实验名称")
    baseline: EvaluationConfig
    variants: list[EvaluationConfig] = Field(
        default_factory=list,
        description="变体配置列表",
    )
    test_dataset_path: str = Field(
        description="测试数据集路径（JSON）",
    )
    max_concurrency: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最大并发数",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="单个查询超时时间（秒）",
    )
