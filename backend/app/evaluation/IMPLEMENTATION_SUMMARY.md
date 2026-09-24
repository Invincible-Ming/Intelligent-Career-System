# RAG 评测与稳定性保障系统 - 实施总结

## ✅ 已完成功能

### 1. 核心评测框架（20+ 模块）

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置管理 | `config.py` | 评测配置、搜索配置、反思配置、实验配置 |
| 数据集 | `dataset.py` | 测试用例、测试数据集模型 |
| RAGAS 指标 | `metrics.py` | Context Precision/Recall、Faithfulness、Answer Relevancy |
| RAG 执行器 | `executor.py` | Dense/BM25/Hybrid 检索、答案生成、反思机制 |
| 并发运行器 | `runner.py` | 并发控制、超时保护、进度追踪 |
| HTML 报告 | `report.py` | 可视化报告生成、图表、Tab 切换 |
| REST API | `api.py` | 后台任务、状态查询、异步评测 |
| 命令行工具 | `cli.py` | quick/full/custom 三种模式 |
| 健康检查 | `health_check.py` | 服务检测、数据集验证、流程测试 |
| 安装验证 | `verify_installation.py` | 文件完整性、模块导入、依赖检查 |

### 2. 测试数据与配置

✅ **30 条回归样本** (`test_dataset_30.json`)
- 覆盖 Python、数据库、架构、DevOps 等技术领域
- 包含标准答案和参考上下文
- 适用于知识问答、技术面试场景

✅ **示例配置** (`example_config.json`)
- Baseline + 2 个变体配置
- 支持 Top-K、检索模式、Reflection 对比

✅ **快速启动脚本**
- `start_evaluation.sh` (Linux/Mac)
- `start_evaluation.bat` (Windows)

### 3. A/B 实验支持

支持对比的参数维度：

| 参数类型 | 可选值 |
|---------|--------|
| **Top-K** | 3, 5, 8, 10, 15, 20 |
| **检索模式** | dense, bm25, hybrid |
| **Chunk Size** | 200-2000 (建议: 400, 600, 800, 1200) |
| **Chunk Overlap** | 0-500 |
| **Reranker** | enable/disable |
| **Reflection** | enable/disable, max_iterations: 1-5 |

### 4. RAGAS 评测指标

| 指标 | 计算方法 | 范围 |
|------|----------|------|
| **Context Precision** | LLM 判断检索上下文相关性 | 0-1 |
| **Context Recall** | 关键词重叠 + 覆盖率 | 0-1 |
| **Faithfulness** | LLM 判断答案忠实度 | 0-1 |
| **Answer Relevancy** | LLM 判断答案相似度 | 0-1 |
| **Overall Score** | 四项指标平均值 | 0-1 |

### 5. 稳定性保障

✅ **并发控制**: `asyncio.Semaphore(max_concurrency=5)`  
✅ **超时保护**: `asyncio.wait_for(timeout=120)`  
✅ **异常处理**: 捕获并记录，不中断整体流程  
✅ **失败统计**: 成功/失败/超时分类统计  
✅ **进度追踪**: 实时更新评测进度  

### 6. HTML 可视化报告

生成的报告包含：

- **总体摘要**: 配置数、样本数、成功率
- **指标对比**: 横向柱状图对比各配置
- **详细结果**: Tab 分组展示每个配置的测试结果
- **交互功能**: Tab 切换、动态加载

## 📁 文件结构

```
app/
├── requirements.txt           # 依赖清单
├── README.md                  # 项目总文档
└── evaluation/                # 评测系统
    ├── __init__.py            # 模块入口
    ├── config.py              # 配置模型 (1.8 KB)
    ├── dataset.py             # 数据集模型 (1.0 KB)
    ├── metrics.py             # RAGAS 指标 (9.4 KB)
    ├── executor.py            # RAG 执行器 (6.8 KB)
    ├── runner.py              # 并发运行器 (10.3 KB)
    ├── report.py              # HTML 报告生成 (16.6 KB)
    ├── api.py                 # FastAPI 端点 (3.2 KB)
    ├── cli.py                 # 命令行工具 (7.1 KB)
    ├── health_check.py        # 健康检查 (5.0 KB)
    ├── run_evaluation.py      # 主评测脚本 (4.8 KB)
    ├── verify_installation.py # 安装验证 (3.8 KB)
    ├── test_dataset_30.json   # 30 条测试样本 (12.2 KB)
    ├── example_config.json    # 示例配置 (1.3 KB)
    ├── start_evaluation.sh    # Linux 启动脚本
    ├── start_evaluation.bat   # Windows 启动脚本
    └── README.md              # 评测系统文档 (11.0 KB)

总计: 16 个文件, ~83 KB
```

## 🚀 使用方法

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证安装（需要先确保依赖安装完成）
python -m app.evaluation.verify_installation

# 3. 健康检查
python -m app.evaluation.health_check

# 4. 运行评测
python -m app.evaluation.cli quick     # 快速模式
python -m app.evaluation.cli full      # 完整模式
```

### 使用启动脚本（推荐）

```bash
# Linux/Mac
bash app/evaluation/start_evaluation.sh

# Windows
app\evaluation\start_evaluation.bat
```

### Python 脚本方式

```python
import asyncio
from app.evaluation.runner import EvaluationRunner
from app.evaluation.config import ExperimentConfig, EvaluationConfig

async def run():
    experiment = ExperimentConfig(
        experiment_name="my_test",
        baseline=EvaluationConfig(name="baseline"),
        variants=[],
        test_dataset_path="app/evaluation/test_dataset_30.json",
        max_concurrency=5,
        timeout_seconds=120,
    )
    
    runner = EvaluationRunner(experiment)
    await runner.run()
    report_path = await runner.generate_html_report()
    print(f"报告: {report_path}")

asyncio.run(run())
```

### REST API 方式

```bash
# 1. 启动服务
python -m uvicorn app.main:app --reload

# 2. 调用评测 API
curl -X POST http://localhost:8000/api/evaluation/experiments \
  -H "Content-Type: application/json" \
  -d @app/evaluation/example_config.json

# 3. 查询状态
curl http://localhost:8000/api/evaluation/experiments/{experiment_id}
```

## 📊 预期结果

### 成功标准

运行 30 条回归样本后：

✅ **成功率**: 100% (30/30)  
✅ **并发数**: 5  
✅ **失败样本**: 0 个  
✅ **超时样本**: 0 个  
✅ **平均耗时**: 30-120 秒（取决于配置）  

### 指标目标

| 指标 | 目标值 |
|------|--------|
| Context Precision | > 0.75 |
| Context Recall | > 0.60 |
| Faithfulness | > 0.70 |
| Answer Relevancy | > 0.80 |
| Overall Score | > 0.70 |

## 🎯 应用场景

### 1. 回归测试

每次代码更新后运行快速评测，确保系统稳定性：

```bash
python -m app.evaluation.cli quick
```

### 2. 参数优化

对比不同配置，选择最优参数：

```bash
python -m app.evaluation.cli full
```

### 3. CI/CD 集成

在 GitHub Actions 中自动运行评测：

```yaml
- name: Run RAG Evaluation
  run: |
    python -m app.evaluation.health_check
    python -m app.evaluation.cli quick
```

### 4. A/B 实验

创建自定义配置文件，对比新策略：

```bash
python -m app.evaluation.cli custom --config my_config.json
```

## 🔧 配置示例

### 配置 1: Baseline (混合检索)

```json
{
  "name": "baseline",
  "search": {
    "top_k": 5,
    "search_mode": "hybrid",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "enable_reranker": true
  },
  "reflection": {
    "enable_reflection": false
  }
}
```

### 配置 2: 增大 Top-K

```json
{
  "name": "top_k_10",
  "search": {
    "top_k": 10,
    "search_mode": "hybrid",
    ...
  }
}
```

### 配置 3: 启用 Reflection

```json
{
  "name": "with_reflection",
  "search": {...},
  "reflection": {
    "enable_reflection": true,
    "max_iterations": 2
  }
}
```

## 📈 报告示例

HTML 报告路径：`evaluation_reports/report_{experiment_name}_{timestamp}.html`

报告内容：
- 📊 总体摘要卡片（配置数、样本数、成功率）
- 📈 指标对比图表（横向柱状图）
- 📝 详细结果（Tab 分组展示）
- 🎨 响应式设计（支持桌面/移动端）

## 🐛 故障排查

### 问题 1: ModuleNotFoundError

**原因**: 缺少依赖包  
**解决**: `pip install -r requirements.txt`

### 问题 2: 评测超时

**原因**: 网络慢或并发过高  
**解决**: 增加 `timeout_seconds` 或降低 `max_concurrency`

### 问题 3: Milvus 连接失败

**原因**: Milvus 服务未启动  
**解决**: `docker-compose up -d milvus`

### 问题 4: 百炼 API 调用失败

**原因**: API Key 未配置或余额不足  
**解决**: 检查 `.env` 中的 `DASHSCOPE_API_KEY`

## 📝 后续优化建议

### 短期优化

1. **缓存机制**: 缓存 Embedding 结果，避免重复计算
2. **批量评测**: 支持多数据集并行评测
3. **增量评测**: 只评测变更的测试用例
4. **邮件通知**: 评测完成后发送邮件报告

### 长期规划

1. **在线评测平台**: Web UI 管理评测任务
2. **实时监控**: Grafana 仪表盘展示评测趋势
3. **自动调优**: 基于评测结果自动推荐最优配置
4. **多语言支持**: 支持英文数据集和报告

## 🎓 最佳实践

1. **定期回归**: 每周运行完整评测
2. **版本对比**: 每次大版本发布前对比新旧版本
3. **参数记录**: 记录每次实验的配置和结果
4. **阈值告警**: 设置指标阈值，低于阈值自动告警
5. **文档更新**: 及时更新测试数据集和配置

## 📚 参考资料

- [RAGAS 官方文档](https://docs.ragas.io/)
- [RAG 评测论文](https://arxiv.org/abs/2309.15217)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [LangGraph 文档](https://python.langchain.com/docs/langgraph)

---

**实施完成时间**: 2026-07-27  
**系统版本**: v1.0  
**维护者**: AI 求职团队
