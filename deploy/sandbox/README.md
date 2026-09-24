# 任务沙箱 Runner

用户上传的 PDF、DOCX、XLSX、TXT 不再在 FastAPI 进程内解析。可信 Sandbox Runner 持有 Docker 权限，通过仅本机可访问的 Unix socket 接受已验证归属的文档字节；每份文档启动一次临时容器，结束即删除。容器没有网络、数据库、MinIO 或百炼密钥，也没有任何用户目录挂载。扫描 PDF 的 OCR 图片通过受控协议交给 FastAPI 调用百炼，模型密钥始终留在后端。

## 快速启动（本地开发）

在项目根目录构建镜像并启动 Runner：

```bash
docker build -f deploy/sandbox/Dockerfile -t career-document-sandbox:1 .
cd backend
python ../deploy/sandbox/start.py
```

另开终端启动 FastAPI。Runner 必须先启动；后端默认 `SANDBOX_DOCUMENTS_ENABLED=true`，Runner 或 Docker 不可用时文件上传会明确失败，不会在后端进程回退解析。MCP 工具也通过 Runner 启动；Runner 不可用时对应工具禁用。`SANDBOX_RUNNER_SOCKET` 默认为 `deploy/.local/sandbox.sock`。默认目录和 socket 分别为 0700、0600，仅本机同一受信用户可访问。生产 Linux 部署时让 Runner 使用单独的 OS 账号持有 Docker 权限，将 Runner 与后端账号加入专用 socket 组，并以 `SANDBOX_RUNNER_GROUP=组名` 启动 Runner；此时目录和 socket 分别为 0770、0660。后端账号不得加入 Docker 组，也不得直接访问 Docker socket。Mac 上的 Docker Desktop 适合本地验证，生产多租户应放在专用 Linux 主机。

## 生产部署

在 Linux 主机上准备可被 Runner 账号读取的项目和 Python 环境。以 root 身份运行 `deploy/sandbox/setup.sh /项目绝对路径`；必须设置 `SANDBOX_BACKEND_USER` 为后端服务账号，可通过 `SANDBOX_RUNNER_USER`、`SANDBOX_RUNNER_GROUP`、`SANDBOX_PYTHON` 指定 Runner 账号、专用 socket 组和 Python 解释器。脚本仅安装系统账号与 systemd 单元，不改动 `deploy/.local` 的所有权。将后端 `backend/.env` 中的 `SANDBOX_RUNNER_SOCKET` 设为 `/run/career/sandbox.sock`，再启动服务：

```bash
sudo systemctl enable --now sandbox-runner
```

后端账号由安装脚本加入专用 socket 组；需重新登录或重启后端进程使组成员资格生效。后端不得加入 Docker 组，也不得直接访问 Docker socket。`setup.sh` 不会启动服务，以便先检查账号和目录权限。服务通过 systemd 的 `RuntimeDirectory` 和 `LogsDirectory` 创建独立目录；本地开发仍使用 `deploy/.local`。Runner 仅从自身进程环境读取 `MCP_IMAGE` 与 `MCP_SEARCH_SOCKET_VOLUME`（缺省值分别为 `career-mcp:1`、`career-mcp-search-socket`），不读取后端 `.env`。

## 审计日志

Runner 在本地开发时将 JSON 行审计日志写入 `deploy/.local/audit.log`，systemd 部署时写入 `/var/log/career/sandbox-audit.log`。记录的事件包括：

- `parse_start` / `parse_result` / `parse_error` / `parse_cancel` — 文档解析任务的生命周期与耗时
- `ocr_request` — OCR 请求（仅记录图片 SHA256 摘要，不记录原文）
- `mcp_start` / `mcp_result` / `mcp_cancel` — MCP 工具调用

审计日志不包含简历原文、文档字节、OCR 图片、百炼密钥或 Docker 参数。

## 安全边界

文档解析接口限制文件格式、大小、输出文本及 OCR 回复。固定镜像用 `--pull=never`，容器使用 UID 10001、只读根文件系统、`--network=none`、默认 seccomp、`no-new-privileges`、无 capabilities、CPU/内存/进程数限制；输入和结果仅经 stdio，不挂载项目目录或 Docker socket。Runner 取消任务时强制删除容器，进程重启会回收遗留容器。`owner_id` 的哈希及任务 ID 仅作为审计标签，不代替后端的数据归属校验。

MCP 调用经本机 Unix socket 转发至 Runner，由 Runner 按固定配置启动搜索、文件或数据库短生命周期容器；模型和客户端不能指定镜像、挂载或 Docker 参数。岗位匹配的搜索容器带任务 ID 和用户哈希标签，连接断开或达到 120 秒上限时删除。文件工具仅能访问已批准的共享资料目录，数据库工具仅能查询受限视图，二者不接触私人简历。私人文件 MCP 尚未接入；在完成 API 归属校验和专属目录权限设计前，Runner 拒绝按任务挂载私人目录。

匹配工作流和模型调用目前仍在可信 FastAPI 后端；这不是每个登录用户常驻一个容器，也不表示整个 LangGraph 执行已迁入容器。生产环境需要把 Runner 与 FastAPI 分成不同 OS 身份，并确保 FastAPI 身份无法直接访问 Docker socket；本机同一 macOS 用户运行只验证容器的任务隔离，不构成 Docker 控制面的进程权限隔离。
