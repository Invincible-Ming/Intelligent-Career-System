# cpolar 同域登录与小程序入口

公网入口必须同时转发 Streamlit 页面与浏览器登录路径。只把隧道指向 8501 时，
浏览器无法在这个域名访问 8000 的登录接口。此网关在本机 8080 提供统一入口：

```text
浏览器 → HTTPS 隧道 → 127.0.0.1:8080
                      ├─ 页面、静态资源、WebSocket → Streamlit :8501
                      ├─ /api/auth/browser-login  → FastAPI :8000
                      ├─ /api/auth/browser-logout → FastAPI :8000
                      └─ /pricing                 → FastAPI :8000
```

小程序通过显式允许的 `/mini-api/` 路由携带 Bearer 会话访问同一后端；
管理员与评测 API 不在该允许列表。其他 `/api/` 路径返回 404。业务 API 继续由 Streamlit 服务端在本机携带 Bearer 凭证调用。
网关不读取 `.env`，不持有模型/数据库密钥，不挂载 Docker socket。

## 本地启动

推荐使用项目根目录的 `./start.sh`，自动启动全部本机服务、生成隧道域名和更新允许列表。
macOS 也可双击 `start.command`，详见 [一键启动说明](../LOCAL_START.md)。下方保留手动启动方式。

先启动 Docker Desktop 和现有 PostgreSQL、Redis、MinIO，以及项目所用的 Milvus/MCP 依赖。
以下命令中的 `python` 使用 `career-agent` 环境。

在项目根目录的不同终端依次运行：

```bash
# 终端 1：沙箱 Runner，已有进程时不要重复启动
python deploy/sandbox/start.py
```

```bash
# 终端 2：后端
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# 终端 3：前端，API_BASE_URL 始终是服务端能访问的本机后端
cd frontend
API_BASE_URL=http://127.0.0.1:8000 python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

```bash
# 终端 4：同域网关，再开隧道
docker compose -f deploy/docker-compose.gateway.yml up -d
cpolar http -proto=https -inspect-addr=127.0.0.1:0 127.0.0.1:8080
```

看到 cpolar 输出的 HTTPS 地址后，在项目根目录另开终端，替换下面的示例地址：

```bash
python deploy/gateway/configure_origin.py https://你的实际地址.r9.cpolar.top
```

脚本只更新 `backend/.env` 的 `CORS_ORIGINS`，保留原有来源并加入本地网关和当前准确域名。
之后重启后端；仅修改 `.env` 不保证触发 Uvicorn 自动重载。再访问新的公网地址登录。
无须把后端 `API_BASE_URL` 改成公网域名，也不要把 `CORS_ORIGINS` 设置成 `*`。

前端默认自动区分公网与本机访问：公网使用当前站点的登录 URL，本机直连 8501 使用 8000。
如果希望本机也从 8080 登录，可在启动前端时添加 `BROWSER_API_BASE_URL=same-origin`，
然后访问 `http://127.0.0.1:8080`。这个变量是前端进程环境变量，不会从 `backend/.env` 自动加载。
原来设置过错误的 `BROWSER_API_BASE_URL` 时先 `unset BROWSER_API_BASE_URL`。

## 检查与停止

```bash
curl --noproxy '*' http://127.0.0.1:8080/_stcore/health
docker compose -f deploy/docker-compose.gateway.yml exec gateway nginx -t
docker compose -f deploy/docker-compose.gateway.yml logs --tail 30 gateway
```

健康检查应返回 `ok`；登录接口只接受 POST，GET 返回 405 是正常的。
页面打开后长时间空白时检查 WebSocket 是否建立；此配置转发 Upgrade/Connection，
不需要关闭 Streamlit 的 XSRF 或 CORS 保护。

临时隧道重启会获得新地址，需要再次运行来源配置脚本、重启后端，并从新地址登录。
不同域名的 Cookie 不共享，旧地址登录态不会迁移。过期隧道地址可从 `CORS_ORIGINS` 中移除。
按 Ctrl+C 停止 cpolar；网关用下面的命令停止，业务存储保持独立：

```bash
docker compose -f deploy/docker-compose.gateway.yml down
```

这是 Docker Desktop 上的本机部署配置，使用 `host.docker.internal` 连接宿主服务。
生产部署应使用固定域名/固定隧道；HTTPS 专用环境设置 `AUTH_COOKIE_SECURE=true`。
当前网关转发外部 HTTPS 协议，后端确认 HTTPS 后会自动设置 Secure Cookie；
应保留后端的准确来源、Origin、同域返回地址和 HttpOnly Cookie 校验。

参考：[cpolar 官方文档](https://www.cpolar.com/docs)、
[NGINX WebSocket 代理](https://nginx.org/en/docs/http/websocket.html)。
