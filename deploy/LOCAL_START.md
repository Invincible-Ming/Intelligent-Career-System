# 本机一键启动

适用于当前 macOS + Docker Desktop 开发环境。项目已有 PostgreSQL/MinIO 容器、沙箱/MCP 镜像、
`backend/.env` 以及已安装前后端依赖的 `career-agent` Conda 环境。
cpolar 需提前安装并配置好账号；启动器不读取或打印 cpolar 的账号凭证。

## 启动

在终端执行：

```bash
cd /Users/crazyzm/pythonCode/Intelligent_Career_System
./start.sh
```

也可在 Finder 中双击项目根目录的 `start.command`。脚本自动定位项目目录和 `career-agent` 的 Python，
无需提前激活环境。需要选择其他解释器时：

```bash
CAREER_PYTHON=/你的环境/bin/python ./start.sh
```

启动器会：

1. 检查 Docker 引擎，未运行时启动 Docker Desktop 并等待就绪。
2. 启动现有 `career-postgres` 和 `career-minio`，通过 Compose 启动 Redis、Milvus 及其依赖、MCP 搜索 broker 和同域网关。
3. 在私有 Unix socket 上启动文档/MCP Runner，等待健康检查通过。
4. 启动本机 `127.0.0.1:8501` 的 Streamlit，并检查 `127.0.0.1:8080` 网关转发。
5. 优先复用小程序配置中仍可用的 cpolar HTTPS 隧道；否则启动 cpolar，验证公网地址后自动更新 `backend/.env` 的 `CORS_ORIGINS`。
6. 启动本机 `127.0.0.1:8000` 的后端，验证数据库健康、登录来源校验和公网连通性，将同一个地址写入小程序 `config.local.ts`，再输出地址。

完成后可以关闭启动终端，服务仍在后台运行。启动过程及各服务日志保存在：

```text
deploy/.local/launcher/
├── state.json
└── logs/
    ├── setup.log
    ├── sandbox-时间.log
    ├── frontend-时间.log
    ├── backend-时间.log
    └── tunnel-时间.log
```

状态文件不保存环境变量、模型密钥或登录凭证；状态与日志文件采用私有权限，目录已被 Git 忽略。
启动不执行模型生成或评测。后端正常启动时仍会初始化数据库、索引连接和已配置的 MCP 等扩展。

## 常用命令

```bash
./start.sh status             # 查看脚本记录的服务状态和当前公网地址
./start.sh stop               # 停止脚本启动的隧道、前端、后端和沙箱；Docker 容器继续运行
./start.sh stop --containers  # 同时停止本项目的 Docker 容器，保留容器和数据卷
./start.sh restart            # 重启脚本管理的应用进程和隧道，加载新代码和配置
./start.sh --local            # 仅本地运行，不开隧道；关闭此前由此脚本启动的隧道
```

默认启动不使用 Uvicorn 自动重载，避免文件变化时中断服务。修改后端代码/配置后用 `restart` 加载。
cpolar 免费随机域名可能发生变化；重新运行启动器会探测旧地址，失效时创建新隧道。
新建隧道可能改变公网地址；使用启动结果输出的地址登录，不同域名的浏览器 Cookie 不共享。
仍在运行的脚本隧道会被复用，重复运行 `./start.sh` 不会重复创建进程或域名。

## 已有服务与失败处理

健康的手工服务可复用，但不归启动器管理，`stop` 不会结束它们。端口被不健康服务占用时明确报错，
不会按端口直接杀进程。停止托管服务前会同时检查 PID、进程创建时间、命令和进程组，防止 PID 复用时误杀。

如果复用的手工后端没有加载新公网域名，启动器会明确提示。请在原终端停止该手工后端，再运行 `./start.sh`，
之后即可由脚本统一管理。手工开启的其他隧道也不会被自动关闭。

失败时先查看提示中的日志；已成功启动的进程保留，以便修复后继续运行启动命令。
也可用 `./start.sh stop` 清理本次托管进程。不要手动删除正在运行服务的 `state.json`，否则会失去托管记录。

首次准备缺失时会报告缺少的容器、镜像或 Python 依赖。沙箱和 MCP 镜像可按项目现有说明构建：

```bash
docker build -f deploy/sandbox/Dockerfile -t career-document-sandbox:1 .
docker build -f deploy/mcp/Dockerfile -t career-mcp:1 deploy/mcp
```

`career-postgres`、`career-minio` 继续使用现有部署及其凭证和数据卷，启动器不会重建或覆盖它们。
Redis 配置不存在时调用项目已有 `setup_redis.py` 生成；已有配置会复用。

本机页面可访问 `http://127.0.0.1:8501`。默认通过 `8080` 为公网提供同域页面与浏览器登录路径，
详细转发边界见 [网关说明](gateway/README.md)。

## cpolar 与小程序

启动器只使用 cpolar，不再启动 Cloudflare。优先复用小程序当前配置且健康检查通过的
cpolar 地址；手工启动的 cpolar 始终保留，不归脚本停止。脚本新建的隧道由脚本管理，
请求检查器仅绑定本机回环地址，不对公网开放。

默认使用 `cn` 节点，指定其他地区可使用：

```bash
CAREER_CPOLAR_REGION=hk ./start.sh
```

生成新地址时自动更新 Web 登录允许来源和小程序开发地址。微信开发者工具重新编译即可加载；
不会填写或覆盖 `config.ts` 的正式版地址。正式使用仍需固定 HTTPS 合法域名。

## 小程序后台任务并发

`backend/.env` 可设置 `MOBILE_JOB_MAX_WORKERS=16`（默认 16，允许 1～128）。
这个上限由 Redis 原子分配，多个后端进程共享；完成或取消即释放，异常退出的名额最多
45 秒回收。达到上限返回 429，不会偷偷超额执行，也不自动重试付费任务。
每用户原有的 `MAX_CONCURRENT_OPERATIONS=2`、调用预算及积分检查继续生效。
此设置限制小程序匹配/面试/学习后台任务，不代表整台机器能并发解析同样数量的文档。
修改配置后重启后端生效。
