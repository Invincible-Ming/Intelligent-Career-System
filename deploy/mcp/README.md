# 受限 MCP 部署

调研 Agent 仅获得 `search_web`，没有任意 URL、网页 JavaScript、Shell、SQL 或文件工具。
普通聊天仍直接调用百炼，不会因此获得工具权限。

## 启动

从项目根目录执行（Mac 先启动 Docker Desktop）：

```bash
docker compose -f deploy/docker-compose.mcp.yml up -d --build --wait
```

先按照 [Runner 启动说明](../sandbox/README.md) 构建并启动沙箱 Runner。后端通过权限受限的本机 Unix socket 请求 Runner 启动 MCP 容器；每次工具连接结束，容器即删除。
搜索服务常驻，使用 Unix socket 通信，无宿主机端口。离线或镜像未构建时禁用对应工具，
绝不回退到本机 `npx`、宽权限 Puppeteer 或业务数据库账号。

## 数据库只读账号

数据库启动且业务表已存在后，使用项目 Python 环境执行：

```bash
python deploy/mcp/provision_db.py
```

默认读取 `backend/.env` 的数据库配置，仅初始化脚本使用业务/管理员连接。
若该账号没有 CREATEROLE，通过环境变量 `MCP_ADMIN_DATABASE_URL` 指定管理员连接；
不要把管理员凭据写入 MCP 配置或镜像。初始化不会打印密码。
远程数据库可通过 `--container-host` 指定工具容器连接的主机。

初始化新增 `career_mcp_reader` 和 `mcp_safe` schema，不修改业务表：

- `knowledge_inventory`：仅 knowledge 文档按状态聚合的文档数和分块数。
- `evaluation_summary`：实验数、测试数、成功数、失败数的整体统计。

不包含简历、文件名、文档 ID、聊天消息、原文、实验名称或密钥。
账号无角色继承、无管理权限，仅授予上述视图 SELECT。每次查询还会验证实际权限，
包括 PUBLIC 继承来的权限、建 schema 权限与角色成员关系；检测到多余权限就拒绝查询。
工具不接受 SQL，只接受视图枚举和至多 50 行。

凭据保存到 `deploy/mcp/secrets/database.json`，权限 0600；只读挂载到数据库容器。
该目录已忽略 Git，Docker 构建上下文也排除了凭据、文档和 `.env`。
已有账号或凭据不会自动覆盖。连接为只读事务，查询 3 秒超时。
若旧数据库向 PUBLIC 授予 CREATE 或业务表权限，初始化会回滚并失败；
不会通过修改所有用户的 PUBLIC 权限来偷偷绕过校验，需管理员单独审计处理。

## 文件边界

只读挂载 `backend/mcp_workspace`，只提供列目录和读 UTF-8 文本工具。
该目录应保留可供 UID 10001 读取的权限（目录 0755、共享文本 0644）。
隐藏文件、路径穿越、符号链接、非普通文件与超过 128 KiB 文件均拒绝。
路径逐级使用 openat/O_NOFOLLOW，避免符号链接替换竞态。
目录是明确批准的共享资料区，不是私人简历存储，不提供用户间身份隔离。
未来给用户任务接入文件工具时，需要每任务单独挂载并先验证归属。

## 容器和网络边界

- 搜索 MCP：UID/GID 10001、只读根文件系统、只读搜索 socket 卷、`--network=none`、无 capabilities。
- 文件 MCP：相同限制，仅专用目录只读挂载、`--network=none`。
- 搜索服务：只允许 `www.bing.com` 启动时解析出的公网 IPv4 的 TCP 443；拒绝私网、
  环回、链路本地、组播、保留地址和 IPv6，运行时不允许 DNS。
  HTTPS 固定 Host/SNI 并校验证书，直接连接已验证 IP，不跟随重定向，不访问搜索结果链接。
  无通用浏览器，因此网页 JavaScript 不会执行。
- 数据库 MCP：仅允许专用数据库主机解析出的固定 IPv4 和数据库端口，
  使用 libpq hostaddr 连接，不允许其他网络出口。允许这一明确的数据库地址，是数据库工具唯一的内网例外。
- 搜索服务和数据库服务初始化时短暂使用 root/NET_ADMIN 安装 OUTPUT 默认 DROP 规则；
  防火墙失败立即退出。数据库初始化还需 DAC_OVERRIDE 读取 0600 挂载凭据。
  启动服务前 setpriv 永久切换 UID/GID 10001，清除 capabilities/bounding set，开启 no-new-privileges。
- 不注入百炼、MinIO、管理员数据库等无关凭据；不挂载项目根目录或 Docker socket；
  无 privileged/host networking；限制 CPU、内存、进程数，tmpfs 为 noexec/nosuid。

Docker CLI 由可信 Runner 发起，参数来自固定应用配置；模型和用户无法选择镜像、挂载或 Docker 参数。生产环境的后端账号不得直接访问 Docker socket。
`app/requirements.txt` 统一引用 `backend/requirements.txt`；LangGraph/MCP 依赖已对齐到此次验证的 API 系列。
这些是工具执行隔离，不能替代业务接口登录/数据归属校验，也不能保证模型不受提示词注入影响。

## 可用性与配置

搜索使用固定 Bing HTML 搜索页，无需额外 API 密钥。页面结构变化、验证码、IP 变更或网络不通
会导致搜索不可用并退回基础匹配；不会扩大权限或伪造搜索结果。IP 更新后重启搜索服务刷新白名单。
若全局代理把 `www.bing.com` 解析成 `198.18.0.0/15` 等虚拟映射地址，搜索代理会按非公网地址拒绝启动；应在梯子中为该域名配置真实 DNS/直连路由，而不是放宽内网访问规则。

后端支持：`MCP_ENABLED`、`MCP_FILES_ENABLED`、`MCP_DATABASE_ENABLED`、
`MCP_STARTUP_TIMEOUT`、`MCP_RESEARCH_TIMEOUT`、`MCP_RESEARCH_MAX_STEPS`。
默认调研最多 6 个图步骤、总耗时 45 秒；输入最多 300 字符、结果最多 5 条。

## 验证

```bash
cd backend
python -m unittest discover -s tests -v
```

部署后需要验证真实容器中的 UID、capabilities、只读挂载和网络拒绝，不能只看 Compose 配置。
从项目根目录、无其他 MCP 任务运行时执行 `python deploy/mcp/verify_runtime.py`，
它验证真实 stdio 工具、网络阻断和一次公开搜索，不调用百炼模型。
不要读取/输出 `docker inspect` 的完整环境变量或凭据文件内容。
