# RAG 评测与稳定性保障系统

## 📋 功能概述

完整的 RAG 系统评测框架，支持：

当前指标由固定版本 `ragas==0.3.9` 的官方实现计算，已替换早期自定义评分器。
每个样本最多 16 次指标 LLM 调用，每次输出最多 2048 Token（可配置），
回答相关性保持 3 次反向问题生成。调用异常、解析失败或 NaN 记为样本失败，
不补 0.5/0.6；有效零分参与四指标均值。检索、回答生成和指标评测共用样本超时。
报告记录引擎与模型，旧版自定义指标的分数不能作为新 RAGAS 实验的对照基线。

指标请求遇到断连、网络超时或可恢复的上游错误时，默认最多重试一次
（`EVALUATION_METRIC_MAX_RETRIES=1`，允许 0～2）。所有真实尝试均计入每题
`EVALUATION_METRIC_MAX_CALLS=16` 上限，不叠加 SDK/RAGAS 的传输重试；
认证失败、额度不足和输出校验错误不会通过这个机制重试。重试仍受整题总耗时约束。
文档评测脚本会打印检索、生成、各指标以及请求计数的阶段进度。

本地 80 题配置每题总耗时上限为 600 秒。每次模型请求另外受实际总时长限制
（`EVALUATION_LLM_REQUEST_TIMEOUT=90`），防止 HTTPX 分段读取超时与自动重试叠加，
导致单次请求占用整题时间。回答生成默认最多重试一次
（`EVALUATION_LLM_MAX_RETRIES=1`）；指标重试由适配器单独计数，不叠加。

`scripts/run_ragas_document.py` 会原子保存 `progress.json`：检索完成后保存上下文，
回答完成后保存回答，每项指标完成后保存分数。后续加 `--resume <progress.json>`
即可复用已完成步骤；每次续跑仍输出到新目录，保留原报告。
复用前会校验用户、文档分块、题库、模型、提示词和检索配置；不一致时拒绝复用。
`--cases 3` 的进度可以用于同题库的 `--cases 80`；并发和整题超时调整不使进度失效。
旧版本报告未保存中间进度，不能直接把 `results.json` 当作续跑文件。
续跑报告会标记复用状态；耗时仅包含当前这轮执行，不能当作完整评测延迟。
进度只供本地维护脚本使用，包含文档片段与回答，请按业务数据保管。

自有问答与反思提示词统一维护于 `app/core/prompts.py`，报告记录其版本。
它们与官方 RAGAS 的评分提示词分开管理：本次只调整被评测的回答生成方式，
没有改动官方指标规则。对比检索策略时还需固定问答提示词，避免混淆变量。

✅ **RAGAS 指标评估**：Context Precision、Context Recall、Faithfulness、Answer Relevancy  
✅ **A/B 实验对比**：支持多配置并行评测  
✅ **参数对比测试**：Top-K、检索模式（Dense/BM25/Hybrid）、Reranker、Reflection
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

### 独立评测模型

在 `backend/.env` 配置：

```dotenv
EVALUATION_LLM_PROVIDER=bailian
EVALUATION_LLM_MODEL=
```

留空会复用现有 `BAILIAN_CHAT_MODEL`，密钥和地址分别沿用 `DASHSCOPE_API_KEY`
与 `BAILIAN_BASE_URL`，不要用通用地址覆盖已有的工作空间专属地址。
回答生成、可选反思和全部 RAGAS 指标评判均使用该模型；独立评测开关不改变日常聊天模型。
如需使用小米，将提供方改为 `mimo`、模型改为 `mimo-v2.6-flash`，
并配置 `MIMO_API_KEY` 与 `MIMO_BASE_URL`。
小米请求关闭深度思考，使用 `max_completion_tokens`。两种提供方均保留请求预算、积分计量、
输出截断检查和指标调用次数上限。参数遵循[小米官方文档](https://mimo.mi.com/docs/zh-CN/api/chat/openai-api)。
Milvus 查询与回答相关性仍使用现有 `BAILIAN_EMBEDDING_MODEL`，复用缓存和已有文档向量。
报告记录实际评判模型、提供方、思考模式与向量模型。修改配置后需重启运行中的后端/评测进程。
切换提供方后，旧提供方生成的 `progress.json` 不可复用；上游失败不会自动回退到另一提供方。

### 用户范围和执行参数

- 网页/API 的 `owner_id` 只来自认证后的当前管理员，忽略客户端自报身份。
- CLI 必须显式提供 `--owner-id`；缺失、空值或无效 UUID 不允许退回全库检索。
- 一个实验固定使用该用户启动时的就绪文档，基线和变体共享同一文档列表。
- Dense、BM25、Hybrid 三条分支均传递 `owner_id`，并按快照文档 ID 进一步收窄。
- `enable_reranker` 与 `reranker_top_k` 按单个配置生效，不修改全局设置。
- `top_k` 表示检索返回上限；启用重排后，最终上限为 `min(top_k, reranker_top_k)`。
- 生成与反思使用实际返回的全部上下文，不再固定截取前 5 条。
- 检索和生成异常由运行器记录为失败，不将异常字符串或检索故障计为成功。
- Redis 仅保存实验状态；它不是可恢复任务队列，进程退出不会自动续跑后台实验。
- 报告读取核对当前用户的已完成实验；状态过期后不会通过路径绕过归属检查。
- Python 直接调用运行器前应先初始化 Milvus 与该用户的 BM25 索引；CLI 已处理此生命周期。

### 1. 命令行快速评测

```bash
# 快速评测（仅 baseline 配置）
python -m app.evaluation.cli quick --owner-id <用户UUID>

# 完整评测（6 个配置对比）
python -m app.evaluation.cli full --owner-id <用户UUID>
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
from app.evaluation.cli import run_evaluation

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
    # 此入口负责初始化与关闭连接，并生成 HTML 报告。
    await run_evaluation(experiment, owner_id="填写当前用户的 UUID")

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

| 指标                  | 说明                                     | 范围 |
|-----------------------|------------------------------------------|------|
| **Context Precision** | 检索到的上下文中有多少是真正相关的       | 0-1  |
| **Context Recall**    | 标准答案中的陈述有多少能被检索上下文覆盖 | 0-1  |
| **Faithfulness**      | 答案中的陈述有多少能被检索上下文支持     | 0-1  |
| **Answer Relevancy**  | 答案反向生成的问题与原问题的向量相似度   | 0-1  |
| **Overall Score**     | 四个指标的平均值                         | 0-1  |

Context Precision 使用 `LLMContextPrecisionWithReference`，Context Recall 使用
`LLMContextRecall`，两者均使用数据集的 `ground_truth` 作为 `reference`。
`reference_contexts` 保留为数据集标注字段，不再用文本重叠率代替 RAGAS Recall。
空检索或空回答定义为四项零分；基础设施异常则记为失败，报告同时展示失败数量。

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
        reranker_top_k=10,
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

### 3. 切块参数与现有索引

评测复用当前账号已就绪文档的索引，不会因查询配置变化而重新切块或改写线上索引。
启动前会使用保存的原始 Block 和配置重新计算分块，并与实际存储的 Chunk 内容核对。
如果 `chunk_size` / `chunk_overlap` 与已有分块不一致，评测会明确拒绝启动。

切块对照需要先建立各自的独立索引并固定语料；不能仅改变配置中的数字。
当前内置完整实验只比较查询阶段参数，不再包含无效的切块变体。

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

| 参数              | 类型 | 默认值   | 说明                        |
|-------------------|------|----------|-----------------------------|
| `top_k`           | int  | 5        | 检索返回的文档数量          |
| `search_mode`     | str  | "hybrid" | 检索模式：dense/bm25/hybrid |
| `chunk_size`      | int  | 800      | 校验现有索引的文档切块大小  |
| `chunk_overlap`   | int  | 100      | 校验现有索引的切块重叠大小  |
| `enable_reranker` | bool | True     | 是否启用 BGE 重排           |
| `reranker_top_k`  | int  | 5        | 重排后返回的数量            |

### ReflectionConfig 参数

| 参数                | 类型 | 默认值 | 说明             |
|---------------------|------|--------|------------------|
| `enable_reflection` | bool | False  | 是否启用反思机制 |
| `max_iterations`    | int  | 2      | 最大反思次数     |
| `reflection_prompt` | str  | ...    | 反思提示词       |

### ExperimentConfig 参数

| 参数                | 类型             | 默认值 | 说明           |
|---------------------|------------------|--------|----------------|
| `experiment_name`   | str              | -      | 实验名称       |
| `baseline`          | EvaluationConfig | -      | 基准配置       |
| `variants`          | list             | []     | 变体配置列表   |
| `test_dataset_path` | str              | -      | 测试数据集路径 |
| `max_concurrency`   | int              | 5      | 最大并发数     |
| `timeout_seconds`   | int              | 120    | 单查询超时时间 |

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
python -m pip install -r requirements-evaluation.txt
```

在 `backend` 目录、与后端相同的 Python 环境执行。该文件同时引用主依赖约束，
避免单独安装评测库时无约束升级应用依赖。下载的是库和依赖；评分模型仍使用已有百炼接口，
评测会产生模型调用费用。未安装时，仅评测入口返回明确错误，其他页面不依赖 RAGAS 导入。

## 🎓 使用示例

### 场景 1：快速验证回归测试

```bash
# 使用 30 条回归样本快速验证
python -m app.evaluation.cli quick --owner-id <用户UUID>

# 检查成功率是否达到 100%
```

### 场景 2：优化检索参数

```bash
# 对比不同 Top-K、检索模式、重排与反思策略
python -m app.evaluation.cli full --owner-id <用户UUID>

# 查看 HTML 报告，选择最优配置
```

### 场景 3：CI/CD 集成

```bash
# 在 CI 流程中运行评测
python -m app.evaluation.run_evaluation --owner-id <用户UUID>

# 检查退出码，失败则阻止部署
```

## 📊 预期结果

运行后应检查成功/失败数量、四项指标、实际耗时和模型配置。
预置样本数不等于成功数；准确率、指标提升与耗时均需真实运行后记录，不能预设为 100%。

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
