# 智能求职 Multi-Agent 系统

基于 LangGraph Multi-Agent 与 RAG 的智能求职辅助平台：简历与岗位多维匹配、工具增强型 RAG 智能对话（联网搜索 + 百度地图）、面试真题预测、定制学习计划，并内置管理员专属的 RAG 评测模块。

## 功能总览

| 页面 | 说明 |
|------|------|
| 智能对话 | 工具增强型 ReAct Agent：受限联网搜索 + 百度地图（定位 / POI 检索 / 路线规划 / 天气 / 通勤矩阵），流式回答并展示工具调用进度 |
| 知识库 | 文档上传（PDF / DOCX / XLSX / TXT）、沙箱解析、切块、向量化入库；多路混合检索验证（管理员） |
| 岗位匹配 | Resume Agent 与 JD Agent 并发分析，Match Agent 六维加权打分，人工审核确认后生成匹配报告 |
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

## 项目结构

```
backend/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── core/                # 配置、数据库、限流、数据模型、Schema
│   ├── security/            # 认证（密码哈希、会话校验、角色注入）
│   ├── api/                 # 业务 / 对话 / 认证路由
│   ├── services/            # 文档、MinIO、Milvus、BM25、Reranker、
│   │                        # 混合检索、百炼、MCP、沙箱客户端等服务
│   ├── agents/              # LangGraph Multi-Agent 工作流与对话 Agent
│   └── evaluation/          # 评测模块（数据集 / 运行器 / 报告）
├── scripts/                 # 管理与运维脚本
└── tests/                   # 测试
frontend/                    # Streamlit 前端
deploy/
├── sandbox/                 # 文档解析沙箱（Runner / Worker / 审计）
└── docker-compose.*.yml     # Milvus / Redis / MCP 中间件编排
docs/                        # 安全设计文档
```

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
| `UPLOAD_TOTAL_TIMEOUT` | 上传解析总耗时上限（秒） | 600 |
| `BAILIAN_OCR_MODEL` | 文档 OCR 使用的视觉模型 | qwen-vl-ocr-latest |
| `BAIDU_MAP_MCP_URL` | 百度地图 MCP 接入地址（含 AK），留空则不加载 | 见 .env |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 切块长度与重叠 | 800 / 100 |

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

## 已知边界

- 图片内绘制的文字（非页面文字层）不会进入知识库，OCR 不可靠的图片页将跳过并占位
- IP 定位仅精确到城市级别；知识库覆盖范围内的地点与路线查询不受影响
- 评测实验列表保存在内存中，后端重启后清空（已生成的 HTML 报告保留在 `backend/evaluation_reports/`）
- 免费模型额度按系列各 100 万 Token、90 天有效，批量上传扫描版大文件前请留意控制台余量
