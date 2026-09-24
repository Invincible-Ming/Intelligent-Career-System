# 登录、数据归属与资源限制

运行环境：Python 3.11+，FastAPI 0.121+，Streamlit 1.60+。依赖分别安装 backend/requirements.txt 与 frontend/requirements.txt。后端在 backend 目录启动，读取 backend/.env；前端通过 API_BASE_URL 指定后端地址。

当前百炼模型配置：对话模型 `qwen3.8-flash`，上下文摘要模型 `deepseek-v4-flash-0731`。两个模型需要在百炼控制台开启并确认额度有效。

## 登录与归属

前端先展示登录/注册页面。登录表单使用受信任的 Streamlit v2 浏览器组件，直接 POST 到 `/api/auth/browser-login`，后端设置 host-only 的 `career_session` Cookie，属性为 HttpOnly、SameSite=Lax、Path=/，有效期与数据库会话一致。浏览器组件仅收到成功标志，不收到 Token；成功后刷新页面建立新的 WebSocket 连接。浏览器刷新时，Streamlit 从初始请求的 `st.context.cookies` 读取凭证，调用 `/auth/me` 验证后恢复身份，再为普通与 SSE 请求携带 Authorization: Bearer。业务 API 仍只接受 Bearer，不通过 Cookie 直接鉴权。

退出或收到 401 时清空私有页面与会话状态，由浏览器 POST `/api/auth/browser-logout` 清除 Cookie 并撤销数据库会话，再返回登录页；撤销后的凭证不能恢复身份。浏览器登录与退出验证 Origin，返回地址只能是 `CORS_ORIGINS` 中的同域前端，避免跨站登录、跨站退出和开放重定向。登录响应禁止缓存；密码及 Token 不写入 URL、localStorage、全局缓存或日志。

本机 HTTP 开发支持 `localhost` 与 `127.0.0.1`，浏览器访问前后端必须使用同一个主机名；Cookie 不按端口区分。前端会自动对齐这两个本机主机名。线上需使用同域 HTTPS，后端设置 `AUTH_COOKIE_SECURE=true`，并配置 `CORS_ORIGINS` 为实际前端地址。若浏览器访问后端的地址不同于 Streamlit 服务端的 `API_BASE_URL`，为前端设置 `BROWSER_API_BASE_URL`（同域反向代理的 API 地址）。生产使用 HTTPS 时 Cookie 自动启用 Secure；反向代理终止 TLS 时应显式配置。

用户名仅允许 3–32 位 ASCII 字母、数字及下划线，统一小写。密码 12–128 位；数据库仅保存带随机盐的 scrypt 散列。登录得到随机、不透明的会话 Token，数据库仅保存 Token 的 SHA-256 散列；会话默认 12 小时到期，退出立即撤销当前会话。注册只能创建普通用户。

身份来自服务端会话，不能通过请求体设置 user_id、owner_id 或 is_admin。文档、对话、任务有 owner_id 外键且归属必填。列表只返回本人记录，读取/修改/删除他人记录与读取不存在记录均返回 404。匹配、面试、学习计划的输入文档/报告均先校验归属。恢复工作流时，thread_id 必须等于本人 run_id，且只允许原子地将 paused 的 match 任务切换为 running；已运行、已完成或其他类型任务不能重放。

相同文件的去重仅在同一用户和文档类型中进行。Dense、BM25、Hybrid 检索先从 PostgreSQL 取得本人 ready 文档 ID，再在检索和排序前过滤；API 返回时再次过滤。没有本人文档时直接返回空结果，不发送 Embedding 请求。

同一对话的生成使用单独连接持有 PostgreSQL advisory lock，锁直到保存结果/流结束才释放。历史从数据库按页读取，关系不自动加载全部消息。中断回复标记 incomplete，不加入后续模型上下文。

评测 API 仅限管理员；评测状态只返回启动该实验的管理员记录。数据集限制为 backend/app/evaluation 目录内的 JSON 文件（最大 2 MiB），变体最多 3 个；评测继续共享同一请求的调用预算与总耗时限制。评测 CLI 是本地运维工具。

## 当前默认限制

以下设置可写入 backend/.env，修改后重启后端。前端从 /api/auth/options 获取显示限制，后端始终是最终裁决方。

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| AUTH_REGISTRATION_ENABLED | true | 是否开放注册 |
| AUTH_SESSION_HOURS | 12 | 会话有效期 |
| API_REQUESTS_PER_MINUTE | 120 | 每用户每固定分钟的受保护 API 请求 |
| AUTH_REQUESTS_PER_MINUTE | 30 | 每来源 IP 每分钟的注册/登录请求；另限制每用户名登录 5 次/分钟 |
| GENERATION_REQUESTS_PER_MINUTE | 10 | 每用户每分钟的生成、上传或检索请求 |
| GENERATION_REQUESTS_PER_DAY | 200 | 每用户每日上述请求 |
| GLOBAL_GENERATION_REQUESTS_PER_DAY | 1000 | 全站每日上述请求 |
| MAX_CONCURRENT_OPERATIONS | 2 | 每用户同时执行的上述操作 |
| CHAT_MAX_INPUT_CHARS | 4000 | 单条聊天输入字符数 |
| CHAT_HISTORY_MESSAGES | 40 | 历史候选消息数，再加当前消息 |
| CHAT_CONTEXT_BUDGET | 18000 | 聊天文本上下文预算：UTF-8 字节数 + 每消息 32 字节 |
| CHAT_SUMMARY_TRIGGER_MESSAGES | 24 | 未摘要消息达到此数量时调用摘要模型 |
| CHAT_SUMMARY_TRIGGER_BYTES | 10000 | 未摘要文本达到此 UTF-8 字节数时调用摘要模型 |
| CHAT_SUMMARY_KEEP_MESSAGES | 8 | 生成摘要后仍保留的最近原始消息数 |
| CHAT_SUMMARY_MAX_OUTPUT_TOKENS | 800 | 摘要模型最大输出 Token 数 |
| CHAT_MAX_OUTPUT_TOKENS | 2048 | 单次聊天模型输出 Token 上限 |
| MODEL_MAX_OUTPUT_TOKENS | 4096 | 分析、结构化输出、OCR 等单次模型输出上限 |
| MODEL_INPUT_BUDGET | 200000 | 单次模型文本输入预算：UTF-8 字节数 + 消息开销 |
| MAX_MODEL_CALLS_PER_OPERATION | 24 | 一次操作中模型、Embedding、OCR、调研工具调用合计上限；重试/JSON 修复/并行节点共享 |
| CHAT_TOTAL_TIMEOUT | 90 | 聊天包括整个 SSE 流的总工作耗时，秒 |
| ANALYSIS_TOTAL_TIMEOUT | 180 | 面试、学习计划与评测总工作耗时，秒 |
| MATCH_TOTAL_TIMEOUT | 600 | 岗位匹配首次生成及每次恢复的总工作耗时上限，秒；人工等待不计入 |
| MATCH_MODEL_TIMEOUT | 120 | 岗位匹配中单次模型请求的超时上限，秒；同时受剩余总耗时约束 |
| UPLOAD_TOTAL_TIMEOUT | 120 | 上传、解析、向量化总工作耗时，秒 |
| API_TOTAL_TIMEOUT | 30 | 其他 API 总工作耗时，秒 |
| JSON_BODY_MAX_BYTES | 160000 | 普通请求体最大字节数 |
| MAX_UPLOAD_SIZE_MB | 20 | 单文件最大大小；multipart 另允许 64 KiB 开销 |
| MAX_DOCUMENTS_PER_USER | 200 | 用户文档数量上限 |
| MAX_DOCUMENT_TEXT_CHARS | 30000 | 分析文档及粘贴 JD 的字符上限 |
| MAX_PARSED_DOCUMENT_CHARS | 200000 | 单文档解析文本上限 |
| MAX_CHUNKS_PER_DOCUMENT | 256 | 单文档分块上限；调用预算可能更早结束上传 |

搜索与人工反馈各最多 2000 字；任务列表最多 100 条，对话消息每页最多 200 条。解析还限制 DOCX/XLSX 解压合计 32 MiB、ZIP 条目 4096、PDF 500 页、OCR 渲染 2000 万像素、工作簿 50 张表/每表 10000 行/200 列。

上下文预算采用保守的 UTF-8 文本预算，**不是模型精确的 Token 数**。未摘要历史达到消息数或字节阈值时，服务端使用 `BAILIAN_SUMMARY_MODEL` 生成滚动摘要，保存摘要覆盖边界，再将摘要、最近原始消息和当前问题发送给 `BAILIAN_CHAT_MODEL`。摘要调用不开放 MCP 工具；摘要失败时回退到原有的完整轮次截断。所有模型请求通过服务端设置 max_tokens，不允许客户端覆盖。

数据库原子计数让请求频率和并发限制跨后端进程生效。按 UTC 固定分钟/日计数，被拒绝的计数事务回滚；合法进入的请求即使后续失败或归属校验拒绝也计入频率限额。并发占用使用有期限的数据库租约，正常完成、异常、超时与断开连接时释放；崩溃留下的租约到期后恢复。全站日限额限制开放注册带来的总调用量，但不是每日费用的精确计量。

到限时停止后续调用、取消并等待工作流子任务、关闭上游流；使用受保护的短清理步骤释放锁和占用。未开始输出时返回 413（请求体）、422（字段）、429（次数/并发/预算）或 504（总耗时）；SSE 已开始时通过 error 事件/错误数据通知，聊天再发送 [DONE]。清理步骤有短时限，网络故障下可能稍晚于工作耗时上限返回。已经发送给云服务的请求无法保证撤销服务端计算或计费；同步 SDK/解析器当前正在执行的单次调用也不等同于可强制终止的独立进程。

## 本机账号与数据清理

首次管理员已由本地命令生成，用户名与随机密码保存在 backend/.local/initial_login.json，权限 0600；不写进仓库、日志或 MCP 容器。也可在前端注册普通账号。

```bash
cd backend
python scripts/manage_security.py --bootstrap-admin
```

已有管理员时不修改账号。此命令不会让公开注册的第一个用户自动获得管理员权限。

按照用户明确要求，本次已经删除所有无归属的旧文档、聊天和任务，并同步删除项目 MinIO 原文件、Milvus/BM25 索引和旧工作流检查点。摘要保存在 backend/.local/legacy_cleanup.json。不会在启动时自动删除数据；清理命令用于获得明确授权后的维护，保留有归属的数据：

```bash
cd backend
python scripts/manage_security.py --delete-unowned
```

MCP 调研仍只开放公开搜索，并且位于岗位匹配的人工审核闸门之后：首次匹配只根据简历和 JD 生成待审报告，工作流在任何 MCP 调研前暂停；只有用户确认或提交反馈后才恢复并执行一次搜索。恢复/校验重试复用已保存的调研结果，不重复调用工具。文件工具目录仅存放经审核的共享资料，不放用户简历；数据库工具仅能查询既有聚合统计视图。用户数据归属由应用 API 校验，应用不给模型传入客户端指定的身份或数据库凭据。

上传文档由独立 Runner 启动一次性无网络容器解析。MCP 搜索、文件、数据库容器也由 Runner 按固定权限启动，经本机受限 Unix socket 与后端通信；岗位搜索容器标记用户哈希和任务 ID。Runner 或对应容器不可用时禁用该能力，不回退到后端本机执行。模型与 LangGraph 编排仍在可信后端；生产环境需将 Runner 与后端分属不同 OS 账号，并禁止后端账号访问 Docker socket。详见 [沙箱部署](../deploy/sandbox/README.md)。

## 岗位匹配缓存与进度

匹配报告直接用一次结构化模型调用生成评分、匹配技能、技能差距、优势、风险和建议；前端负责排版，综合分仍由服务端按固定权重计算。正常情况下，首次匹配从四次模型调用减少为三次；简历与岗位分析都命中缓存时，审核前只需一次报告调用。输出格式修复及用户批准后的报告更新、校验仍可能产生额外调用。

人工确认后的搜索由应用构造固定查询，直接调用一次 `search_web`，不再使用模型决定搜索步骤，也不额外调用模型总结搜索结果。`MATCH_SEARCH_TIMEOUT=20` 秒；失败、超时或无有效结果时跳过调研。只有规范的标题、摘要、来源链接会作为不可信资料输入模型，步骤不足等错误不会变成情报。没有新增反馈且没有有效调研资料时复用初评报告；有新增资料或反馈时最多执行一次报告更新。报告生成与格式修复总计受 `MATCH_REPORT_TIMEOUT=75` 秒限制。

确认后的事实校验受 `MATCH_VERIFY_TIMEOUT=60` 秒限制，包括格式修复。模型在同一次校验中返回修正后的完整分析，服务端重新计算综合分；不再走“校验、重写、再校验”的循环。未通过且未给出可用修正时，明确添加人工复核风险，不宣称校验通过。确认更新通过 `as_node="human_review"` 明确从审核节点继续，避免默认推断最近节点导致跳过搜索；参见 [LangGraph 状态更新说明](https://reference.langchain.com/python/langgraph/pregel/main/Pregel/aupdate_state)。模型耗时仍取决于服务提供方，超时会显示降级提示，不能保证固定完成秒数。

Redis 只缓存简历分析与岗位分析，不缓存最终报告。缓存键包含服务端认证的用户身份、完整内容、模型、实际提示词、提示词版本和输出结构；用户之间不能共用缓存，内容变化会重新分析。相同分析的并发请求通过带过期时间的 Redis 锁复用计算。损坏缓存会重算；Redis 不可用时退回正常模型分析，不阻止匹配。没有认证用户的 Studio 调试不使用共享缓存。

默认 `ANALYSIS_CACHE_ENABLED=true`、`ANALYSIS_CACHE_TTL=86400`（24 小时）、`REDIS_SOCKET_TIMEOUT=0.5` 秒，连接地址由 `REDIS_URL` 配置。首次在新环境启动本地缓存：

```bash
cd backend
python scripts/setup_redis.py
cd ..
docker compose -f deploy/docker-compose.redis.yml up -d
```

配置脚本生成随机 Redis 密码，保存到被 Git 忽略的 `deploy/.local/redis.conf`，并更新 `backend/.env` 中的连接地址，不在终端打印密码。容器以非 root 用户运行、只读文件系统、移除全部 capabilities，只在 `127.0.0.1:6379` 开放带密码的连接。缓存限制为 128 MB，采用 LRU 淘汰，不写持久化文件；容器重启后正常重新分析。

前端用加载动画与真实阶段进度代替报告逐字展示，显示简历分析、岗位分析、报告生成以及缓存复用状态。后端每 5 秒发送等待心跳，前端更新已等待时间；进度表示完成的阶段，不估计模型剩余时间。完整待审报告在 MCP 调研之前显示，恢复后继续展示调研及校验状态。

匹配的简历分析、岗位分析、报告生成和事实校验，对 `qwen3.8-flash` 系列显式设置 `reasoning_effort="none"`，减少默认高推理力度造成的等待。可用 `MATCH_REASONING_EFFORT` 切换为 `low`、`medium` 或 `xhigh`；其他模型不传此参数，通用聊天和摘要调用保留现有设置。推理模式也纳入缓存键，切换时重新分析。参数含义参见 [百炼 Chat API 文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)。

## 验证

新增缓存测试需要本地 Redis 运行，只操作随机临时键，不访问真实用户缓存；实际工作流测试使用替身模型核对调用次数、跨用户隔离与进度事件。

```bash
cd backend
python -m unittest discover -s tests -v
```

认证集成测试会创建/删除专用临时 PostgreSQL 数据库，要求本地数据库运行且测试账号具有 CREATEDB。测试不会读写真实用户记录或调用收费云模型。覆盖双账号越权、搜索过滤、按用户去重、任务线程绑定与重放、并发原子限流、对话锁、并行模型调用预算、输入/输出预算、SSE 超时清理、文档解析与 MCP 隔离，以及实际 Streamlit 登录/退出页面。

## 本次真实服务检查

本机 HTTP 登录、归属过滤、退出后的 Token 撤销、前端健康检查均已验证。百炼真实流式请求返回 403 / AllocationQuota.FreeTierOnly；根据[官方免费额度说明](https://help.aliyun.com/zh/model-studio/new-free-quota)，这是免费额度耗尽后“用完即停”触发的拒绝。页面会显示明确的额度提示；不会替用户调整付费开关。模型生成的限制通过 SDK 请求参数检查、替身流和实际 LangGraph 并行节点验证；因提供方额度拒绝，尚不能宣称真实生成链路通过。
