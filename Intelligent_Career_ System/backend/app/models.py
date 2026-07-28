"""
项目数据库模型。

核心表：
- documents: 上传文档
- agent_runs: Agent 任务
- analysis_results: 分析结果

对话表：
- conversations: 对话会话
- messages: 对话消息

评测表：
- evaluation_records: 评测记录
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ----------------------------------------------------------------------
# 文档和任务模型（原有）
# ----------------------------------------------------------------------


class Document(Base):
    """用户上传的简历、JD 或知识文档。"""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    document_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    minio_object_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        unique=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class AgentRun(Base):
    """Agent 任务执行记录。"""

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    input_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )

    result_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    analysis_result: Mapped["AnalysisResult | None"] = relationship(
        back_populates="run",
        uselist=False,
    )


class AnalysisResult(Base):
    """Agent 分析结果（简历、JD、匹配报告等）。"""

    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "agent_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    resume_analysis: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    job_analysis: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    match_report: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    interview_plan: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    learning_plan: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    run: Mapped[AgentRun] = relationship(
        back_populates="analysis_result",
    )


# ----------------------------------------------------------------------
# 对话历史模型 ⭐ 新增
# ----------------------------------------------------------------------


class Conversation(Base):
    """对话会话。"""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="用户标识（可选）",
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        default="新对话",
        comment="对话标题（自动生成或用户设置）",
    )

    model: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="qwen-plus",
        comment="使用的模型",
    )

    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="消息数量",
    )

    meta_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="额外元数据",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # 关联消息
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """对话消息。"""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="user, assistant, system",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="消息内容",
    )

    tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="token 数量",
    )

    meta_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="额外元数据（如模型参数、延迟等）",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # 关联会话
    conversation: Mapped[Conversation] = relationship(
        back_populates="messages",
    )


# ----------------------------------------------------------------------
# 评测记录模型 ⭐ 新增
# ----------------------------------------------------------------------


class EvaluationRecord(Base):
    """评测记录持久化。"""

    __tablename__ = "evaluation_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    experiment_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
        comment="实验名称",
    )

    config_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="配置名称",
    )

    dataset_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="数据集名称",
    )

    test_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="测试总数",
    )

    success_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="成功数量",
    )

    failure_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="失败数量",
    )

    duration: Mapped[float] = mapped_column(
        nullable=False,
        comment="总耗时（秒）",
    )

    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="平均指标",
    )

    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="评测配置",
    )

    results: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        comment="详细结果",
    )

    report_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="HTML 报告路径",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
