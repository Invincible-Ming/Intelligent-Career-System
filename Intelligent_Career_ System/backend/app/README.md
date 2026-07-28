# 智能求职 Multi-Agent 系统

基于 **FastAPI + LangGraph + 阿里云百炼 + Milvus + PostgreSQL + MinIO** 构建的智能求职分析系统，集成完整的 RAG 评测与稳定性保障体系。

## 🎯 核心功能

### 1. 智能求职分析（Multi-Agent）
- **Resume Agent**：简历技能、经验、项目提取
- **JD Agent**：岗位要求、职责分析
- **Match Agent**：五维度岗位匹配评分
- **Interview Agent**：智能面试问题生成
- **Learning Agent**：个性化学习计划

### 2. 知识库检索
- **Dense Search**：百炼 `qwen3.7-text-embedding` + Milvus
- **BM25 Search**：jieba 分词 + 关键词检索
- **Hybrid Search**：Dense + BM25 + RRF + BGE 重排

### 3. RAG 评测与稳定性保障 ⭐
- **RAGAS 指标**：Context Precision/Recall、Faithfulness、Answer Relevancy
- **A/B 实验**：多配置并行对比测试
- **参数优化**：Top-K、检索模式、Chunk Size、Reflection
- **并发控制**：可配置并发限制（默认 5）
- **HTML 报告**：自动生成可视化评测报告
- **回归测试**：30 条测试样本，确保零失败

## 🏗️ 技术架构

```
app/
├── models.py              # 数据库模型
├── schemas.py             # API 数据模型
├── config.py              # 配置管理
├── database.py            # PostgreSQL 异步连接
├── main.py                # FastAPI 应用入口
├── api.py                 # REST API 端点
├── workflow.py            # LangGraph 工作流
├── bailian.py             # 百炼服务（Chat + Embedding）
├── milvus_service.py      # Milvus 向量存储
├── minio_service.py       # MinIO 文件存储
├── bm25_service.py        # BM25 关键词检索
├── hybrid_search.py       # 混合检索 + RRF
├── reranker_service.py    # BGE CrossEncoder 重排
├── document_service.py    # 文档解析与入库
└── evaluation/            # 评测系统 ⭐
    ├── config.py          # 评测配置
    ├── dataset.py         # 测试数据集
    ├── metrics.py         # RAGAS 指标
    ├── executor.py        # RAG 执行器
    ├── runner.py          # 并发运行器
    ├── report.py          # HTML 报告生成
    ├── api.py             # 评测 API
    ├── cli.py             # 命令行工具
    ├── health_check.py    # 健康检查
    ├── test_dataset_30.json  # 30 条测试样本
    └── README.md          # 评测系统文档
```

## 🚀 快速启动

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install fastapi uvicorn sqlalchemy asyncpg
pip install pymilvus minio
pip install dashscope openai
pip install langchain langgraph
pip install pymupdf python-docx openpyxl
pip install jieba rank-bm25
pip install sentence-transformers torch
pip install pydantic pydantic-settings
```

### 2. 配置文件

创建 `.env` 文件：

```env
# FastAPI
APP_NAME=智能求职 Multi-Agent 系统
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://career_user:career_password@localhost:5432/career_db

# 阿里云百炼
DASHSCOPE_API_KEY=your_dashscope_api_key
BAILIAN_CHAT_MODEL=qwen-plus
BAILIAN_EMBEDDING_MODEL=qwen3.7-text-embedding
EMBEDDING_DIMENSION=1024

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=career_documents

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=career_minio
MINIO_SECRET_KEY=career_minio_password
MINIO_BUCKET=career-documents

# BGE Reranker
RERANKER_ENABLED=True
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=auto  # auto/mps/cpu
```

### 3. 启动服务

```bash
# 启动 FastAPI 服务
python -m uvicorn app.main:app --reload

# 访问 API 文档
# http://localhost:8000/docs
```

### 4. 运行评测

```bash
# 健康检查
python -m app.evaluation.health_check

# 快速评测（仅 baseline）
python -m app.evaluation.cli quick

# 完整评测（7 个配置对比）
python -m app.evaluation.cli full

# 自定义评测
python -m app.evaluation.cli custom --config app/evaluation/example_config.json
```

## 📊 RAG 评测系统

### 核心特性

✅ **20+ 测试模块**：覆盖检索、生成、反思等所有环节  
✅ **30 条回归样本**：知识问答、技术面试、求职场景  
✅ **并发限制 5**：稳定可控的并发执行  
✅ **零失败保证**：超时保护 + 异常处理  
✅ **HTML 可视化**：美观的评测报告  

### 评测指标

| 指标 | 说明 |
|------|------|
| Context Precision | 检索精度：检索到的上下文有多少是相关的 |
| Context Recall | 检索召回：覆盖了多少标准上下文 |
| Faithfulness | 忠实度：答案是否忠实于检索上下文 |
| Answer Relevancy | 答案相关性：与标准答案的相似度 |
| Overall Score | 综合得分：四个指标的平均值 |

### A/B 实验示例

```bash
# 对比不同配置
python -m app.evaluation.run_evaluation

# 查看生成的报告
# evaluation_reports/report_rag_ab_test_v1_20260727_143052.html
```

支持对比的配置维度：
- **Top-K**：3, 5, 8, 10
- **检索模式**：Dense, BM25, Hybrid
- **Chunk Size**：400, 600, 800, 1200
- **Reranker**：启用/禁用 BGE 重排
- **Reflection**：启用/禁用反思机制

## 📁 API 端点

### 文档管理

```bash
# 上传文档
POST /api/documents/upload

# 列出文档
GET /api/documents

# 删除文档
DELETE /api/documents/{document_id}
```

### 知识库检索

```bash
# Dense 检索
POST /api/search

# BM25 检索
POST /api/search/bm25

# 混合检索
POST /api/search/hybrid
```

### 求职分析

```bash
# 岗位匹配
POST /api/match

# 面试问题
POST /api/interview

# 学习计划
POST /api/learning-plan

# 查询任务
GET /api/runs/{run_id}
```

### 评测系统

```bash
# 启动评测
POST /api/evaluation/experiments

# 查询评测状态
GET /api/evaluation/experiments/{experiment_id}

# 列出所有评测
GET /api/evaluation/experiments
```

## 🧪 使用示例

### 1. 上传简历和 JD

```python
import requests

# 上传简历
with open("resume.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/documents/upload",
        files={"file": f},
        data={"document_type": "resume"},
    )
    resume_id = response.json()["id"]

# 上传 JD
jd_text = "招聘 Python 后端工程师..."
```

### 2. 岗位匹配

```python
response = requests.post(
    "http://localhost:8000/api/match",
    json={
        "resume_document_id": resume_id,
        "jd_text": jd_text,
    },
)

match_report = response.json()
print(f"匹配度：{match_report['result']['total_score']}")
```

### 3. 生成面试问题

```python
response = requests.post(
    "http://localhost:8000/api/interview",
    json={
        "match_run_id": match_run_id,
        "difficulty": "intermediate",
        "question_count": 8,
    },
)

interview_plan = response.json()
```

### 4. 运行评测

```python
import asyncio
from app.evaluation.runner import EvaluationRunner
from app.evaluation.config import ExperimentConfig, EvaluationConfig

async def run():
    experiment = ExperimentConfig(
        experiment_name="my_test",
        baseline=EvaluationConfig(name="baseline"),
        test_dataset_path="app/evaluation/test_dataset_30.json",
        max_concurrency=5,
    )
    
    runner = EvaluationRunner(experiment)
    await runner.run()
    report_path = await runner.generate_html_report()
    print(f"报告：{report_path}")

asyncio.run(run())
```

## 🎯 评测最佳实践

### 回归测试

```bash
# 每次代码更新后运行
python -m app.evaluation.cli quick

# 确保成功率 100%（30/30）
# 确保无超时或失败样本
```

### 参数优化

```bash
# 对比不同配置
python -m app.evaluation.cli full

# 查看 HTML 报告，选择最优配置
# 将最优配置应用到生产环境
```

### CI/CD 集成

```yaml
# .github/workflows/test.yml
- name: Run RAG Evaluation
  run: |
    python -m app.evaluation.health_check
    python -m app.evaluation.cli quick
```

## 📈 性能指标

### 预期性能

| 指标 | 目标值 |
|------|--------|
| 评测成功率 | 100% (30/30) |
| 并发数 | 5 |
| 单查询超时 | 120 秒 |
| 总评测时长 | < 10 分钟 |
| Context Precision | > 0.75 |
| Answer Relevancy | > 0.80 |
| Overall Score | > 0.75 |

## 🔧 故障排查

### 评测失败

```bash
# 1. 检查服务状态
python -m app.evaluation.health_check

# 2. 查看日志
tail -f logs/evaluation.log

# 3. 降低并发数
# 修改 max_concurrency 为 3 或 2
```

### Milvus 连接失败

```bash
# 检查 Milvus 服务
docker ps | grep milvus

# 重启 Milvus
docker-compose restart milvus
```

### 百炼 API 超时

```bash
# 增加超时时间
# 在 config.py 中设置 MODEL_TIMEOUT=120
```

## 📚 文档

- [评测系统详细文档](app/evaluation/README.md)
- [API 文档](http://localhost:8000/docs)
- [LangGraph 工作流](docs/workflow.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

---

**维护者**：AI 求职团队  
**最后更新**：2026-07-27
