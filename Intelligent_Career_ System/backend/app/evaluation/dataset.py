"""
测试数据集模型。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """单个测试用例。"""

    query: str = Field(description="查询问题")
    ground_truth: str = Field(description="标准答案")
    reference_contexts: list[str] = Field(
        default_factory=list,
        description="参考上下文（用于 Context Recall 评测）",
    )
    document_type: str | None = Field(
        default=None,
        description="文档类型过滤",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="额外元数据",
    )


class TestDataset(BaseModel):
    """测试数据集。"""

    name: str = Field(description="数据集名称")
    description: str = Field(default="", description="数据集描述")
    test_cases: list[TestCase] = Field(
        description="测试用例列表",
    )

    @property
    def size(self) -> int:
        """数据集大小。"""
        return len(self.test_cases)
