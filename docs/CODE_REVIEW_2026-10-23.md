# 智能求职系统代码审查记录

审查日期：2026-10-23（北京时间）。对象：当前工作区，包括尚未提交的修改。

## 结论

项目已具备可以展示的完整主链路，但目前不能认定为“没有问题”或“可直接用于公网多租户生产环境”。
本次发现 1 项需要优先处理的部署安全问题、4 项功能或安全边界问题，以及 1 项维护脚本问题。
本次只做审查、模拟复现和文档整理，未修改业务代码、真实用户数据或生产配置。

对应的架构、模块实现与面试讲解见 [总体技术报告](/Users/crazyzm/pythonCode/Intelligent_Career_System/docs/PROJECT_TECHNICAL_REPORT.md)。

## 验证范围

- 阅读前后端、模型服务、积分、任务工作流、检索、文档解析、MCP、沙箱 Runner、评测与部署配置。
- 在专用的临时 PostgreSQL、Redis 中执行测试，没有使用真实业务数据库作为测试库。
- 现有测试：**373 passed，48.09 秒**。
- `backend/app`、`frontend`、`backend/tests` 的 Ruff 检查通过；`git diff --check` 通过。
- 扩大到部署脚本后，`deploy/mcp/verify_runtime.py` 出现导入位置检查问题；实际执行还发现旧模块路径错误。
- 新增模拟故障验证了并行错误状态冲突、调用预算重复计数以及积分结算取消等待问题。
- 检查了仓库使用的 Milvus 镜像默认配置，没有启动真实 Milvus 服务，也没有查询用户向量。
- 本次没有调用真实大模型、执行付费评测、进行公网攻击测试或压测。因此不提供模型效果、线上可用性和并发性能的保证。

现有测试全部通过，只能证明现有用例覆盖的行为通过；以下故障复现表明仍存在测试盲区。

## F01 · P1：Milvus 对所有网卡开放，默认配置未启用鉴权

**位置**：[deploy/docker-compose.milvus.yml:52](/Users/crazyzm/pythonCode/Intelligent_Career_System/deploy/docker-compose.milvus.yml:52)。

Compose 将 `19530:19530`、`9091:9091` 发布到所有网卡，没有限制为回环地址，也没有提供开启 Milvus 鉴权的配置覆盖。
本次实际检查 `milvusdb/milvus:v2.5.14` 镜像内 `/milvus/configs/milvus.yaml`，其中 `common.security.authorizationEnabled: false`。

如果其他设备能够连到宿主机这些端口，就可能直接访问向量服务，绕过 FastAPI 的登录和 `owner_id` 过滤。
向量集合中不只有向量，还保存了分块正文、文档 ID 和所有者 ID，因此风险包括私人简历、岗位和知识文档内容的越权读取或修改。

**已确认**：仓库的端口绑定和镜像默认配置。
**未确认**：实际宿主机防火墙、云安全组和公网可达性；不能据此断言目前已经发生泄露。

**建议**：本机开发至少绑定 `127.0.0.1`；服务同网部署时优先不向宿主机发布数据库端口；跨主机访问须配置内网、鉴权和最小网络放行。仅在客户端填写 Token 不会自动开启服务端鉴权。

## F02 · P2：两个并行抽取节点同时失败时，兜底链路发生状态冲突

**位置**：[backend/app/agents/workflow.py:147](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/agents/workflow.py:147)、[工作流并行边:803](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/agents/workflow.py:803)。

Resume Agent 与 JD Agent 在同一步并行执行。异常包装器在任一节点失败时都返回 `error`、`is_degraded`、`degrade_reason`。
`CareerState` 对这些字段没有声明并发归并规则。

当模型认证失败、额度不足或上游故障同时影响两个抽取节点时，两边会在同一步写相同字段。
本次将两个结构化抽取调用都模拟为失败，实际得到：

```text
InvalidUpdateError: At key 'error': Can receive only one value per step.
```

结果是工作流在状态归并阶段就失败，不能按原设计进入 `fallback_node` 和人工审核。

**建议**：让并行分支分别写 `resume_error`、`jd_error`，汇合后统一生成全局错误；或为错误列表等字段设计明确、可重放的 reducer。须增加“两边同时失败”的回归用例。

## F03 · P2：工具对话的一次模型请求被调用预算计数两次

**位置**：[模型回调:969](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/agents/workflow.py:969)、[非流式调用配置:232](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/api/chat_api.py:232)、[流式调用配置:293](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/api/chat_api.py:293)。

`ChatOpenAI` 自身挂载 `ModelBudgetCallback()`，调用 Agent 时又在 `config.callbacks` 传入另一个实例。
同一次模型启动事件会触发两份计数回调。

本次使用本地模拟 HTTP 响应和真实 LangChain 回调分发验证：

```text
实际上游请求数：1
InvocationBudget.calls：2
```

这会使普通用户比配置的调用次数更早触达上限，尤其影响多轮工具调用。
**这是调用预算重复计数，不是已经证明积分重复扣费**；积分回调只有一处挂载，不能混为一谈。

**建议**：统一回调挂载层，只保留一份模型预算回调，同时明确“工具调用次数”和“模型请求次数”是否需要分别计数。

## F04 · P2：积分结算的取消保护没有独立截止时间

**位置**：[backend/app/services/credits.py:236](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/services/credits.py:236)。

`ModelCharge.finish()` 使用 AnyIO shield 和 `asyncio.shield` 保护结算；收到取消后继续 `await task`，但没有为结算添加单独超时。
结算过程需要数据库连接和钱包行锁。数据库连接、锁等待异常时，请求即使已达到总耗时上限，也可能仍在等待结算。

本次将结算操作模拟为被锁阻塞，取消外层任务后，它仍保持运行，直到主动解除模拟阻塞才退出。
这验证了取消不能及时终止等待的机制；没有对真实数据库制造故障。

**建议**：对结算清理设置短而明确的独立预算，并配合数据库 `lock_timeout` / `statement_timeout`；超时后留下可恢复的持久化记录，通过补偿任务处理。不能简单地把失败的结算默认为成功扣费。

## F05 · P2：网页读取工具的 DNS 校验和实际连接没有绑定

**位置**：[backend/app/services/web_fetch.py:41](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/services/web_fetch.py:41)、[实际 HTTP 请求:64](/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/app/services/web_fetch.py:64)。

代码先通过 `socket.getaddrinfo` 判断域名解析结果是否为公网地址，再将原始域名 URL 交给 HTTPX 连接。
HTTP 客户端建立连接时仍可能重新解析域名，没有强制使用刚刚验证过的 IP。

攻击者若控制域名解析，在校验与连接之间改变解析结果，就存在 DNS rebinding / TOCTOU 风险。
此外，DNS 校验本身是同步调用，位于异步请求链路中。

**已确认**：两次操作分离、网页读取直接在可信后端执行。
**未执行**：真实 DNS rebinding 利用、内网探测或云元数据读取。

**建议**：使用校验后固定连接地址并正确保留 Host/TLS 校验的受控传输，或者将网页抓取迁入具备统一出口策略的代理/容器。
现有的协议、端口、逐跳重定向和大小限制仍有价值，但不能表述为完整的网络隔离。

## F06 · P3：MCP 运行时检查脚本引用已不存在的模块

**位置**：[deploy/mcp/verify_runtime.py:10](/Users/crazyzm/pythonCode/Intelligent_Career_System/deploy/mcp/verify_runtime.py:10)。

脚本仍导入 `app.mcp_tools`，实际模块已位于 `app.services.mcp_tools`。
直接执行脚本立即报 `ModuleNotFoundError`，还没有运行到容器边界验证。

**建议**：修正模块路径及 import 组织，检查脚本中其它旧服务清单假设，并将维护脚本纳入静态检查和可执行冒烟测试。

## 其它应如实说明的边界

以下属于当前设计范围或后续工程任务，不等于本次已证实的漏洞：

1. Docker 隔离的是文档解析和本地搜索工具任务，不是整个用户会话的常驻独立运行环境。
2. 网页抓取、远程百度地图 MCP、模型调用和 LangGraph 编排仍在可信后端执行。
3. 意图识别是正则规则加模型兜底，没有对每一条消息做独立模型安全审查，也没有完整的输出审查或外发隐私过滤。
4. 积分优先使用供应商 `usage`；缺失时会估算。进程崩溃后没有保存下来的用量不能保证精确对账。
5. 开放注册且每账号赠送积分，不等于已经实现防批量注册、防刷赠送的完整机制。
6. 评测采用进程内后台任务，不是持久化分布式任务队列；80 条题目仅覆盖当前 LangChain 概述资料，不能代表所有岗位或全部知识库。
7. 当前缺少生产压测、跨副本故障恢复验证、真实检索效果对照和持续安全评测证据，不能编写相应提升百分比。

## 建议处理顺序

先收紧 Milvus 网络入口；随后修复并行失败归并、预算回调和结算截止时间；再完善网页抓取出口及维护脚本。
修复后补齐本次复现的故障测试，再执行完整回归和针对性的运行时验证。

## 处理进展

2026-10-27 处理了以下三项，其余仍待处理。

| 编号 | 状态 | 处理方式 |
|---|---|---|
| F01 | 已修复 | `deploy/docker-compose.milvus.yml` 端口改为 `127.0.0.1:19530` / `127.0.0.1:9091`。已运行的 `milvus-standalone` 仍是旧绑定，需 `docker compose -f deploy/docker-compose.milvus.yml up -d` 重建后生效 |
| F02 | 已修复 | 并行分支分别写 `resume_error` / `jd_error`，新增 `merge_extraction` 汇合节点统一生成 `error`、`is_degraded`、`degrade_reason`；`tests/test_match_confirmation.py` 增加两边同时失败的回归用例 |
| F06 | 已修复 | 导入改为 `app.services.mcp_tools`；脚本改为只验证实际加载成功的容器模式，移除已下线的 filesystem / postgres 分支；实际执行通过（容器边界、broker、真实搜索） |

F03（调用预算重复计数）、F04（积分结算无独立截止时间）、F05（网页抓取 DNS 校验与实际连接未绑定）未处理。
