# RAG 评测与稳定性保障系统

## 📋 功能概述

完整的 RAG 系统评测框架，支持：

✅ **RAGAS 指标评估**：Context Precision、Context Recall、Faithfulness、Answer Relevancy  
✅ **A/B 实验对比**：支持多配置并行评测  
✅ **参数对比测试**：Top-K、检索模式（Dense/BM25/Hybrid）、Chunk Size、Reflection  
✅ **并发控制**：可配置并发限制（默认 5）  
✅ **超时保护**：单个查询超时控制  
✅ **HTML 可视化报告**：自动生成美观的评测报告  
✅ **30 条回归样本**：预置测试数据集  

## 🏗️ 架构设计

```
app/evaluation/
├── __init__.py          # 模块入口
├── config.py            # 评测配置模型
├── dataset.py           # 测试数据集模型
├── metrics.py           # RAGAS 指标计算
├── executor.py          # RAG 查询执行器
├── runner.py            # 评测运行器（并发控制）
├── report.py            # HTML 报告生成器
├── api.py               # FastAPI 端点
├── cli.py               # 命令行工具
├── run_evaluation.py    # 主评测脚本
└── test_dataset_30.json # 30 条测试样本
```

## 🚀 快速开始

### 1. 命令行快速评测

```bash
# 快速评测（仅 baseline 配置）
python -m app.evaluation.cli quick

# 完整评测（7 个配置对比）
python -m app.evaluation.cli full
```

### 2. Python 脚本评测

```python
import asyncio
from app.evaluation.config import (
    EvaluationConfig,
    ExperimentConfig,
    SearchConfig,
    ReflectionConfig,
)
from app.evaluation.runner import EvaluationRunner

async def run():
    # 定义配置
    baseline = EvaluationConfig(
        name="baseline",
        search=SearchConfig(
            top_k=5,
            search_mode="hybrid",
            chunk_size=800,
            chunk_overlap=100,
            enable_reranker=True,
        ),
        reflection=ReflectionConfig(
            enable_reflection=False,
        ),
    )

    # 创建实验
    experiment = ExperimentConfig(
        experiment_name="my_experiment",
        baseline=baseline,
        variants=[],
        test_dataset_path="app/evaluation/test_dataset_30.json",
        max_concurrency=5,
        timeout_seconds=120,
    )

    # 运行评测
    runner = EvaluationRunner(experiment)
    report_data = await runner.run()

    # 生成 HTML 报告
    report_path = await runner.generate_html_report()
    print(f"报告已生成：{report_path}")

asyncio.run(run())
```

### 3. API 方式评测

```bash
# 启动服务
python -m uvicorn app.main:app --reload

# 调用评测 API
curl -X POST "http://localhost:8000/api/evaluation/experiments" \
  -H "Content-Type: application/json" \
  -d @experiment_config.json

# 查询评测状态
curl "http://localhost:8000/api/evaluation/experiments/{experiment_id}"
```

## 📊 评测指标说明

### RAGAS 核心指标

| 指标 | 说明 | 范围 |
|------|------|------|
| **Context Precision** | 检索到的上下文中有多少是真正相关的 | 0-1 |
| **Context Recall** | 检索到的上下文覆盖了多少标准答案的上下文 | 0-1 |
| **Faithfulness** | 答案中的陈述有多少能被检索上下文支持 | 0-1 |
| **Answer Relevancy** | 答案与标准答案的相似度 | 0-1 |
| **Overall Score** | 四个指标的平均值 | 0-1 |

## 🧪 测试配置示例

### 配置 1：Baseline（混合检索 + 重排）

```python
EvaluationConfig(
    name="baseline",
    search=SearchConfig(
        top_k=5,
        search_mode="hybrid",  # Dense + BM25 + RRF
        chunk_size=800,
        chunk_overlap=100,
        enable_reranker=True,  # BGE 重排
    ),
    reflection=ReflectionConfig(
        enable_reflection=False,
    ),
)
```

### 配置 2：增大 Top-K

```python
EvaluationConfig(
    name="top_k_10",
    search=SearchConfig(
        top_k=10,  # 从 5 增加到 10
        search_mode="hybrid",
        chunk_size=800,
        chunk_overlap=100,
        enable_reranker=True,
    ),
)
```

### 配置 3：Dense Only

```python
EvaluationConfig(
    name="dense_only",
    search=SearchConfig(
        top_k=5,
        search_mode="dense",  # 仅向量检索
        enable_reranker=False,
    ),
)
```

### 配置 4：启用 Reflection

```python
EvaluationConfig(
    name="with_reflection",
    search=SearchConfig(
        top_k=5,
        search_mode="hybrid",
        chunk_size=800,
        chunk_overlap=100,
        enable_reranker=True,
    ),
    reflection=ReflectionConfig(
        enable_reflection=True,
        max_iterations=2,  # 最多反思 2 次
    ),
)
```

## 📝 测试数据集格式

```json
{
  "name": "测试数据集名称",
  "description": "数据集描述",
  "test_cases": [
    {
      "query": "查询问题",
      "ground_truth": "标准答案",
      "reference_contexts": [
        "参考上下文 1",
        "参考上下文 2"
      ],
      "document_type": "knowledge",
      "metadata": {}
    }
  ]
}
```

## 🎯 A/B 实验最佳实践

### 1. 对比不同 Top-K

```python
variants = [
    EvaluationConfig(name="top_k_3", search=SearchConfig(top_k=3)),
    EvaluationConfig(name="top_k_5", search=SearchConfig(top_k=5)),
    EvaluationConfig(name="top_k_10", search=SearchConfig(top_k=10)),
]
```

### 2. 对比检索模式

```python
variants = [
    EvaluationConfig(name="dense", search=SearchConfig(search_mode="dense")),
    EvaluationConfig(name="bm25", search=SearchConfig(search_mode="bm25")),
    EvaluationConfig(name="hybrid", search=SearchConfig(search_mode="hybrid")),
]
```

### 3. 对比 Chunk Size

```python
variants = [
    EvaluationConfig(name="chunk_400", search=SearchConfig(chunk_size=400)),
    EvaluationConfig(name="chunk_800", search=SearchConfig(chunk_size=800)),
    EvaluationConfig(name="chunk_1200", search=SearchConfig(chunk_size=1200)),
]
```

### 4. 对比 Reflection

```python
variants = [
    EvaluationConfig(
        name="no_reflection",
        reflection=ReflectionConfig(enable_reflection=False),
    ),
    EvaluationConfig(
        name="with_reflection",
        reflection=ReflectionConfig(enable_reflection=True, max_iterations=2),
    ),
]
```

## 📈 HTML 报告示例

生成的 HTML 报告包含：

1. **总体摘要**：测试配置数、样本数、成功率
2. **指标对比图表**：各配置的 RAGAS 指标可视化对比
3. **详细结果**：每个测试用例的查询、答案、指标
4. **Tab 切换**：按配置分组展示结果

报告路径：`evaluation_reports/report_{experiment_name}_{timestamp}.html`

## ⚙️ 配置参数说明

### SearchConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_k` | int | 5 | 检索返回的文档数量 |
| `search_mode` | str | "hybrid" | 检索模式：dense/bm25/hybrid |
| `chunk_size` | int | 800 | 文档切块大小 |
| `chunk_overlap` | int | 100 | 切块重叠大小 |
| `enable_reranker` | bool | True | 是否启用 BGE 重排 |
| `reranker_top_k` | int | 5 | 重排后返回的数量 |

### ReflectionConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_reflection` | bool | False | 是否启用反思机制 |
| `max_iterations` | int | 2 | 最大反思次数 |
| `reflection_prompt` | str | ... | 反思提示词 |

### ExperimentConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `experiment_name` | str | - | 实验名称 |
| `baseline` | EvaluationConfig | - | 基准配置 |
| `variants` | list | [] | 变体配置列表 |
| `test_dataset_path` | str | - | 测试数据集路径 |
| `max_concurrency` | int | 5 | 最大并发数 |
| `timeout_seconds` | int | 120 | 单查询超时时间 |

## 🔒 稳定性保障

### 1. 并发控制

使用 `asyncio.Semaphore` 限制并发数，避免过载：

```python
semaphore = asyncio.Semaphore(max_concurrency)

async with semaphore:
    result = await execute_query()
```

### 2. 超时保护

每个查询设置超时限制：

```python
result = await asyncio.wait_for(
    rag_executor.execute(query),
    timeout=timeout_seconds,
)
```

### 3. 异常处理

捕获并记录异常，不中断整体评测：

```python
try:
    result = await evaluate_test_case()
except Exception as exc:
    result = {"status": "error", "error": str(exc)}
```

### 4. 失败统计

记录成功/失败数量，确保无失败样本：

```python
success_count = sum(1 for r in results if r["status"] == "success")
failure_count = len(results) - success_count
```

## 📦 依赖安装

评测系统需要以下额外依赖：

```bash
pip install ragas  # RAGAS 评测库（可选，我们实现了自定义版本）
```

## 🎓 使用示例

### 场景 1：快速验证回归测试

```bash
# 使用 30 条回归样本快速验证
python -m app.evaluation.cli quick

# 检查成功率是否达到 100%
```

### 场景 2：优化检索参数

```bash
# 对比不同 Top-K 和 Chunk Size
python -m app.evaluation.cli full

# 查看 HTML 报告，选择最优配置
```

### 场景 3：CI/CD 集成

```bash
# 在 CI 流程中运行评测
python -m app.evaluation.run_evaluation

# 检查退出码，失败则阻止部署
```

## 📊 预期结果

运行 30 条回归样本后，预期结果：

- ✅ **成功率**：100%（30/30）
- ✅ **并发限制**：5
- ✅ **无失败样本**：0 个超时或错误
- ✅ **平均耗时**：根据配置不同，约 30-120 秒

## 🐛 故障排查

### 问题 1：评测超时

**解决方法**：
- 增加 `timeout_seconds`
- 降低 `max_concurrency`
- 检查 Milvus/百炼服务连接

### 问题 2：RAGAS 指标为 0

**解决方法**：
- 确保测试数据集有 `ground_truth` 和 `reference_contexts`
- 检查 LLM 调用是否成功
- 查看日志中的错误信息

### 问题 3：HTML 报告无法生成

**解决方法**：
- 确保 `evaluation_reports` 目录可写
- 检查磁盘空间
- 查看异常堆栈

## 🔧 高级用法

### 自定义指标

扩展 `metrics.py` 添加自定义指标：

```python
class CustomMetrics(RAGASMetrics):
    async def compute_custom_metric(self, ...):
        # 自定义逻辑
        return score
```

### 自定义报告模板

修改 `report.py` 的 HTML 模板：

```python
def _get_styles(self) -> str:
    return """
    /* 自定义 CSS */
    """
```

### 批量数据集评测

```python
datasets = ["dataset1.json", "dataset2.json", "dataset3.json"]

for dataset_path in datasets:
    experiment = ExperimentConfig(
        experiment_name=f"test_{Path(dataset_path).stem}",
        test_dataset_path=dataset_path,
        ...
    )
    await runner.run()
```

## 📚 参考资料

- [RAGAS 官方文档](https://docs.ragas.io/)
- [RAG 评测最佳实践](https://arxiv.org/abs/2309.15217)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

**维护者**：AI 求职团队  
**最后更新**：2026-07-27
