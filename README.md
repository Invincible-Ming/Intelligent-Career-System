# 智能求职 Multi-Agent 系统

基于 LangGraph Multi-Agent 与 RAG 的智能求职辅助平台：简历与岗位多维匹配、工具增强型 RAG 智能对话（联网搜索 + 百度地图）、面试真题预测、定制学习计划，并内置管理员专属的 RAG 评测模块。

## 功能总览

| 页面 | 说明 |
|------|------|
| 智能对话 | 工具增强型 ReAct Agent：受限联网搜索 + 百度地图（定位 / POI 检索 / 路线规划 / 天气 / 通勤矩阵），流式回答并展示工具调用进度 |
| 知识库 | 文档上传（PDF / DOCX / XLSX / TXT）、沙箱解析、切块、向量化入库；多路混合检索验证（管理员） |
| 岗位匹配 | 并发分析简历和 JD，五维加权打分；自动校验通过即完成，仅未通过时暂停人工审核 |
| 面试准备 | 基于匹配任务的岗位要求与能力缺口，生成专属面试预测题与要点 |
| 学习计划 | 基于匹配任务中的缺失技能，定制分周学习路径 |
| 评测（管理员） | RAG 检索与回答质量的 A/B 对比评测，自动生成 HTML 报告 |

## 系统架构

```
浏览器（Streamlit :8501）
        │ cookie 会话 + REST / SSE
        ▼
FastAPI 后端 :8000 ──── Redis（分析缓存）
        │
        ├── PostgreSQL      业务数据、会话、LangGraph 检查点
        ├── MinIO           原始文档存储
        ├── Milvus + BM25   向量 + 倒排双索引
        ├── 阿里云百炼       对话 / 向量化 / OCR
        └── 百度地图 MCP     定位 / POI / 路线 / 天气（远程 Streamable HTTP）

安全沙箱：沙箱 Runner（持有 Docker 权限，Unix socket 通信）
        └── 临时容器：文档解析（含 OCR）/ MCP 联网搜索（无网络、只读、用后即删）
```

MCP 工具按信任等级分两条接入路径：

- **沙箱路径**：本机执行的联网搜索工具，经 Runner 在一次性容器中运行（无网络、只读、用后即删）
- **直连路径**：可信官方远程服务（百度地图 MCP），后端进程经 Streamable HTTP 直连，AK 由 `backend/.env` 提供

- 所有模型调用统一由后端执行，密钥不进入沙箱容器
- 业务数据按用户 `owner_id` 强制隔离；配置唯一来源为 `backend/.env`

### 文档处理、评分与评测边界

- **文档复用**：同一用户、类型、解析版本下按整文件 SHA-256 去重；变更文件仍完整
  解析和分块，Embedding 按模型、维度和分块文本复用缓存，仅未命中的分块调用向量模型。
  这属于向量计算复用，不是 PDF/DOCX 的二进制增量解析。
- **PDF 阅读顺序**：根据文本块坐标做递归空白分割，优先按栏阅读，并分隔跨栏标题；
  模糊或重叠版面回退到几何顺序，低质量文本继续走 OCR。复杂表格、无明显栏间空白等版面
  不保证正确恢复；新解析版本为 `structure-v3`。
- **五维匹配**：技能 40%、经验 25%、职责 20%、学历 10%、加分项 5%，总分由代码计算。
- **有限纠错**：默认最多 2 轮 Verify 调用（允许修正后再复核），全部轮次共用 60 秒；
  单轮输出上限 3072 Token，实际还受全局模型输出上限约束。校验关闭 Schema 与网络隐式重试，
  达到上限、修正版不变或调用异常时保留报告并暂停人工审核。自动通过后直接输出最终报告。
  人工审核可接受当前报告，或填写意见重新生成与校验；重新校验不通过会再次暂停。
  人工接受与 AI 校验通过分别记录，修正后不直接视为校验通过。
- **公开检索授权**：开始匹配时可选“补充检索公开岗位资料”（API 字段 `research_approved`，
  默认 false）。仅明确授权后使用受限搜索工具，AI 校验通过和人工审核本身均不会授予该权限。
- **官方 RAGAS**：评测模块可选安装 `backend/requirements-evaluation.txt`，固定使用 `ragas==0.3.9`
  四指标实现，并通过现有百炼服务调用模型。评测异常不补默认分，真实零分参与平均值；
  报告记录评测引擎和模型。旧版自定义指标报告与新 RAGAS 分数不可直接横向比较。

## 项目结构

```
backend/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── core/                # 配置、数据库、限流、数据模型、日志与链路追踪
│   ├── security/            # 认证（密码哈希、会话校验、角色注入）
│   ├── api/                 # 业务 / 对话 / 认证路由
│   ├── services/            # 文档、MinIO、Milvus、BM25、Reranker、
│   │                        # 混合检索、百炼、缓存、MCP、沙箱客户端等服务
│   ├── agents/              # LangGraph Multi-Agent 工作流与对话 Agent
│   └── evaluation/          # 评测模块（数据集 / 运行器 / 报告）
├── scripts/                 # 管理与运维脚本（含Milvus owner_id 迁移）
└── tests/                   # 测试
frontend/
├── app.py                   # 页面编排（只做布局与状态协调）
├── career_ui/               # 前端可复用模块
│   ├── config.py            # 配置常量
│   ├── api.py               # 后端客户端（鉴权 / 超时 / SSE）
│   ├── session.py           # 会话状态与浏览器登录桥接
│   ├── endpoints.py         # 业务接口薄封装
│   └── render.py            # 视图展示组件
├── styles.css
└── auth_browser/            # 登录 iframe 静态资源
deploy/
├── sandbox/                 # 文档解析沙箱（Runner / Worker / 审计）
└── docker-compose.*.yml     # Milvus / Redis / MCP 中间件编排
docs/                        # 安全设计文档
```

## 可观测性

每个请求都会分配 `trace_id`（或沿用上游 `X-Request-ID`），并回写到响应头：

```bash
curl -i http://localhost:8000/api/documents -H "X-Request-ID: my-trace-001"
# 响应头含X-Request-ID: my-trace-001，日志中同一 trace 下可看到全部节点
```

日志行尾统一带 `[trace=...]`；用户报障时提供该 ID 即可检索完整链路。
`LOG_JSON=true` 切换为单行 JSON 输出，便于接入日志采集系统。

## 快速开始

### 环境要求

- Python 3.11+，Docker Desktop
- 阿里云百炼 API Key（模型广场领取各模型免费额度）
- 百度地图开放平台 AK（可选，用于对话 Agent 的地图能力）

### 1. 启动中间件（Docker）

```bash
docker compose -f deploy/docker-compose.milvus.yml up -d
docker compose -f deploy/docker-compose.redis.yml up -d
```

### 2. 配置后端

```bash
cd backend
cp .env.example .env   # 若无模板则按 .env 内字段说明手动创建
```

所有配置**仅从 `backend/.env` 读取**，不受终端环境变量影响。

### 3. 初始化管理员账号

```bash
python scripts/manage_security.py --bootstrap-admin
# 凭据生成于 backend/.local/initial_login.json（0600 权限）
```

### 4. 启动文档解析沙箱（必须先于后端）

```bash
docker build -f deploy/sandbox/Dockerfile -t career-document-sandbox:1 .
cd backend
python ../deploy/sandbox/start.py
```

### 5. 启动后端与前端

```bash
# 后端（backend/ 目录下）
uvicorn app.main:app --reload

# 前端（frontend/ 目录下）
streamlit run app.py
```

访问 `http://localhost:8501`，API 文档见 `http://localhost:8000/docs`。

## 关键配置项（backend/.env）

| 配置 | 说明 | 当前值 |
|------|------|--------|
| `MAX_UPLOAD_SIZE_MB` | 单文件上传上限 | 50 |
| `MAX_CHUNKS_PER_DOCUMENT` | 单文档分块数上限 | 512 |
| `MAX_MODEL_CALLS_PER_OPERATION` | 单次操作模型调用上限（**管理员不限**） | 200 |
| `MATCH_VERIFY_MAX_ROUNDS` | 所有用户的校验轮数上限（可设 1～4） | 2 |
| `MATCH_VERIFY_TIMEOUT` | 整个校验阶段耗时上限（秒） | 60 |
| `MATCH_VERIFY_MAX_OUTPUT_TOKENS` | 单轮校验输出上限 | 3072 |
| `EVALUATION_METRIC_MAX_CALLS` | 每个样本四项指标合计的 LLM 调用上限，管理员也生效 | 16 |
| `EVALUATION_METRIC_MAX_OUTPUT_TOKENS` | 每次指标 LLM 调用输出上限 | 2048 |
| `UPLOAD_TOTAL_TIMEOUT` | 上传解析总耗时上限（秒） | 600 |
| `BAILIAN_OCR_MODEL` | 文档 OCR 使用的视觉模型 | qwen-vl-ocr-latest |
| `BAIDU_MAP_MCP_URL` | 百度地图 MCP 接入地址（含 AK），留空则不加载 | 见 .env |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 切块长度与重叠 | 800 / 100 |
| `EMBED_CACHE_TTL` | Embedding 向量缓存时长（秒） | 604800（7 天） |
| `EVALUATION_STORE_REDIS_ENABLED` | 评测实验状态落 Redis | true |
| `LOG_LEVEL` | 日志级别 | INFO |
| `LOG_JSON` | 切换单行 JSON 日志 | false |
| `LOG_ACCESS_ENABLED` | 记录访问日志 | true |

## 缓存与状态存储

| 数据 | 存储位置 | 失效策略 |
|------|---------|---------|
| Embedding 向量 | Redis + 进程内 LRU | `EMBED_CACHE_TTL`（默认 7 天） |
| 分析结果 | Redis | `ANALYSIS_CACHE_TTL`（默认 1 天） |
| 评测实验状态 | Redis（按 `owner_id` 命名空间） | 终态 7 天TTL |
| BM25 倒排索引 | 进程内存 | 上传/删除串行更新语料，构建后整体发布 |

Redis 不可用时，Embedding 与评测状态会自动降级：缓存直接调用模型，评测状态暂存进程内，均不影响功能正确性。

三个缓存服务共用一个进程内 Redis 连接池。启动预热逐条建立并检查连接，
每条连接保持独占，全部检查后统一归还，避免将同一连接的多次 PING 当成多条连接。
预热数量不超过池容量；日志区分全部成功、部分成功和失败。预热中断时也会归还
已借出的连接，关闭临时客户端不会关闭共享池。预热只反映启动时的检查结果，
运行期间连接仍可能断开并按现有策略重连或降级。

BM25 使用独立的写入锁和快照锁：写入锁覆盖读取当前语料、后台构建和发布，
避免并发上传或删除覆盖其他操作；检索只短暂取得快照锁，构建期间继续读取旧索引。
构建失败时保留完整旧索引。`rebuild(rows)` 用于启动或独立维护时的全量替换，
在线文档更新使用 `add_document` / `delete_document`。索引与锁均为进程内状态，
多 worker 或多副本之间仍需额外的索引同步机制。

## 角色与限流

| 能力 | 管理员 | 普通用户 |
|------|--------|----------|
| 模型调用次数 | 不限 | 200 次 / 操作 |
| 生成频率 | 不限 | 10 次 / 分，200 次 / 天 |
| 评测模块 / 检索验证 / 知识资料维护 | ✅ | ❌ |
| 对话 / 简历上传 / 岗位匹配 | ✅ | ✅ |

限流报错会以具体原因透出至前端（如「本次操作已达到模型调用次数上限（200 次）」）。

## 安全设计

- **配置隔离**：配置只读 `backend/.env`，终端环境变量（代理、旧密钥）在启动时被主动清除
- **解析沙箱**：文档不在后端进程内解析，统一交给临时容器；容器无网络、UID 10001、只读、无 capabilities
- **数据隔离**：文档、会话、任务、向量全链路按 `owner_id` 过滤，管理员无跨用户数据权限
- **能力白名单**：MCP 工具按能力白名单校验；模型不可指定镜像、挂载或 Docker 参数
- **角色化限流**：模型调用计数与频率限制对普通用户生效，管理员豁免（总耗时上限仍生效）
- 敏感文件（`.env`、`.local/`）已在 `.gitignore` 中排除

## 切换 Embedding 模型

```bash
# backend/.env
BAILIAN_EMBEDDING_MODEL=qwen3.7-text-embedding-flash
EMBEDDING_DIMENSION=1024
```

| 模型 | 支持维度 | 说明 |
|------|---------|------|
| `qwen3.7-text-embedding` | 256 ~ 2560（默认 1024） | 标准版，效果优先 |
| `qwen3.7-text-embedding-flash` | 256 ~ **1024**（默认 1024） | 成本与吞吐更优，单价 0.125 元/百万 Token |

两者都支持 201 种语言、128K 长文本；flash 在语义、跨语言及代码检索上与标准版基本持平，
AIRBench-zh 略优约 3%。**代码检索能力两者都支持**，换 flash 不会损失。

**换模型无需重建 Collection**（前提维度不变）。但注意：

- `EMBEDDING_DIMENSION` 必须与 Milvus Collection 的 `dim` 一致，否则向量写入失败；
- 向量缓存的 key 含模型名，换模型后旧缓存自动失效，不会返回另一模型的向量；
- flash 版维度上限是 1024，配成 1536+ 会在启动时报错（已加校验）。

## 已知边界

- 图片内绘制的文字（非页面文字层）不会进入知识库，OCR 不可靠的图片页将跳过并占位
- IP 定位仅精确到城市级别；知识库覆盖范围内的地点与路线查询不受影响
- 评测实验列表保存在 Redis，重启后仍可查询；已生成的 HTML 报告保留在 `backend/evaluation_reports/`
- 免费模型额度按系列各 100 万 Token、90 天有效，批量上传扫描版大文件前请留意控制台余量

## 升级到含owner_id 的 Milvus Schema

若 Milvus 中的 Collection 是旧版本（无 `owner_id` 字段），后端会拒绝启动以避免在无隔离的索引上提供检索。迁移步骤：

```bash
cd backend
python scripts/migrate_milvus_owner.py --dry-run   # 预检，不写入
python scripts/migrate_milvus_owner.py             # 正式执行
```

脚本会将旧 Collection **重命名**为 `_backup_<时间戳>`（保留而非删除），新建带 `owner_id` 标量索引的
Collection，并按 PostgreSQL 的解析文本重新向量化回填。回填失败时把 `MILVUS_COLLECTION` 改回备份名即可回滚。
