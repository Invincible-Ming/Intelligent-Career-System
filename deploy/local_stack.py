"""Start/stop this Mac development stack without putting credentials in state or output."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values
from deploy.gateway.configure_origin import allow_origin, normalize_origin

LOCAL = ROOT / "deploy/.local/launcher"
PUBLIC_URL = re.compile(r"https://(?:[a-z0-9-]+\.)+cpolar\.(?:top|cn|io|com)\b")
CONTAINERS = ("career-web-gateway", "milvus-standalone", "career-milvus-minio", "career-milvus-etcd",
              "career-mcp-search-broker-1", "career-redis", "career-minio", "career-postgres")
HTTP = build_opener(ProxyHandler({}))  # Local checks must not go through shell proxy variables.


def sync_miniprogram_origin(origin):
    """Only public development routing, never backend keys or session tokens."""
    target = ROOT / "miniprogram/miniprogram/config.local.ts"
    if target.parent.exists():
        source = ("// 开发连接地址，由项目启动脚本同步；这里不放任何密钥。\n"
                  f"export const developmentApiBase = {json.dumps(origin.rstrip('/') + '/mini-api')}\n")
        target.write_text(source, encoding="utf-8")


class StartError(RuntimeError):
    pass


def say(message):
    print(message, flush=True)


def normalized_identity(identity):
    # cpolar intentionally replaces argv with "cpolar: master process" after
    # connecting. Preserve birth time while normalizing just its known titles.
    # This also recognizes records created before the title changed.
    return re.sub(r"(?<=\d{4} )(?:\S*/)?cpolar(?:\s+http\b.*|: (?:master|worker|single) process.*)$",
                  "cpolar", identity or "")


def process_identity(pid):
    """PID alone is insufficient: validate creation time, command and process group."""
    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        # ps localizes lstart (Chinese dates contain a different number of
        # words). Read independent columns and never include the mutable stat
        # value in identity. Force one locale across different startup shells.
        environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        born = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                              capture_output=True, text=True, timeout=3, env=environment)
        details = subprocess.run(["ps", "-p", str(pid), "-o", "stat=,args="],
                                 capture_output=True, text=True, timeout=3, env=environment)
        if born.returncode or details.returncode or os.getpgid(pid) != pid:
            return None
        fields = details.stdout.strip().split(None, 1)
        if not born.stdout.strip() or len(fields) != 2 or fields[0].startswith("Z"):
            return None
        return normalized_identity(" ".join(born.stdout.split()) + " " + fields[1])
    except (OSError, subprocess.TimeoutExpired):
        return None


def owned_alive(record):
    return bool(record and record.get("managed") and record.get("identity")
                and process_identity(record.get("pid")) == normalized_identity(record["identity"]))


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.5):
            return True
    except OSError:
        return False


def fetch(url, *, data=None, headers=None):
    request = Request(url, data=data, headers=headers or {})
    try:
        with HTTP.open(request, timeout=4) as response:
            return response.status, response.read(4096)
    except HTTPError as exc:
        return exc.code, exc.read(4096)
    except (OSError, URLError, TimeoutError, http.client.HTTPException):
        return 0, b""


def ready_http(url, *, json_status=False):
    status, body = fetch(url)
    if status != 200:
        return False
    if json_status:
        try:
            return json.loads(body).get("status") == "ok"
        except (ValueError, AttributeError):
            return False
    return body.strip() == b"ok"


def runner_connected(path):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(1)
            stream.connect(str(path))
        return True
    except OSError:
        return False


def runner_ready(path):
    connection = http.client.HTTPConnection("localhost", timeout=4)
    try:
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(4)
        connection.sock.connect(str(path))
        connection.request("GET", "/health")
        response = connection.getresponse()
        return response.status == 200 and json.loads(response.read(1024)).get("status") == "ok"
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def login_origin_ready(origin):
    from urllib.parse import urlsplit
    status, body = fetch(
        "http://127.0.0.1:8080/api/auth/browser-logout",
        data=urlencode({"return_to": origin + "/"}).encode(),
        headers={"Origin": origin, "Host": urlsplit(origin).netloc,
                 "X-Forwarded-Proto": urlsplit(origin).scheme,
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    # No Cookie is sent; this validates the origin without logging out any user.
    try:
        return status == 200 and json.loads(body).get("cleared") is True
    except (ValueError, AttributeError):
        return False


def atomic_state(path, data):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class Stack:
    def __init__(self, local=LOCAL, timeout=180):
        self.local = Path(local)
        self.local.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.logs = self.local / "logs"
        self.logs.mkdir(mode=0o700, exist_ok=True)
        self.path = self.local / "state.json"
        self.timeout = timeout
        self.children = {}
        try:
            self.state = json.loads(self.path.read_text()) if self.path.exists() else {"services": {}}
            if not isinstance(self.state.get("services"), dict):
                raise ValueError("invalid services")
        except (ValueError, OSError) as exc:
            raise StartError(f"启动状态文件无法读取，请检查 {self.path}") from exc

    @contextmanager
    def lock(self):
        with (self.local / "launcher.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StartError("另一条启动/停止命令正在执行，请等待它结束") from exc
            # Another launcher may have completed after this instance was constructed.
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
            yield

    def save(self):
        atomic_state(self.path, self.state)

    def wait(self, predicate, label, service=None, timeout=None):
        deadline = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < deadline:
            if service and not owned_alive(self.state["services"].get(service)):
                raise StartError(f"{label}进程已退出，请查看 {self.state['services'][service]['log']}")
            if predicate():
                return
            time.sleep(1)
        raise StartError(f"等待{label}超时，请检查 {self.logs}")

    def run(self, command, label, *, timeout=240):
        log = self.logs / "setup.log"
        with log.open("a") as output:
            log.chmod(0o600)
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise StartError(f"{label}超时，详情见 {log}") from exc
        if result.returncode:
            raise StartError(f"{label}失败，详情见 {log}")

    def spawn(self, name, command, cwd, environment=None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        log = self.logs / f"{name}-{stamp}.log"
        with log.open("w") as output:
            log.chmod(0o600)
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = process
        identity = None
        for _ in range(20):
            identity = process_identity(process.pid)
            if identity or process.poll() is not None:
                break
            time.sleep(.05)
        if identity is None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)
            raise StartError(f"{name}启动失败，请查看 {log}")
        self.state["services"][name] = {"managed": True, "pid": process.pid, "identity": identity, "log": str(log)}
        self.save()
        return self.state["services"][name]

    def stop_service(self, name):
        record = self.state["services"].get(name)
        if owned_alive(record):
            pid = record["pid"]
            say(f"停止 {name}…")
            try:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 12
                while owned_alive(record) and time.monotonic() < deadline:
                    if name in self.children:
                        self.children[name].poll()
                    time.sleep(.1)
                if owned_alive(record):
                    os.killpg(pid, signal.SIGKILL)
                if name in self.children:
                    self.children[name].wait(timeout=3)
            except ProcessLookupError:
                pass
        elif record and not record.get("managed"):
            say(f"保留手工启动的 {name}")
        self.state["services"].pop(name, None)
        if name == "tunnel":
            previous = self.state.pop("public_url", "")
            if PUBLIC_URL.fullmatch(previous):
                self.state["cpolar_url"] = previous
        self.save()

    def ensure(self, name, command, cwd, ready, occupied, environment=None):
        record = self.state["services"].get(name)
        if owned_alive(record):
            say(f"复用 {name}")
            self.wait(ready, name, service=name)
            return
        if occupied():
            if not ready():
                raise StartError(f"{name}地址已被占用且健康检查失败，请先检查手工启动的服务；未终止该进程")
            say(f"复用手工启动的 {name}（停止脚本不会结束它）")
            self.state["services"][name] = {"managed": False}
            self.save()
            return
        say(f"启动 {name}…")
        self.spawn(name, command, cwd, environment)
        self.wait(ready, name, service=name)

    def docker_ready(self):
        try:
            return subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def prepare_docker(self):
        if not shutil.which("docker"):
            raise StartError("没有找到 docker，请先安装并启动 Docker Desktop")
        if not self.docker_ready():
            say("启动 Docker Desktop，首次启动可能需要等待…")
            self.run(["docker", "desktop", "start", "--timeout", str(self.timeout)], "启动 Docker Desktop")
            self.wait(self.docker_ready, "Docker 引擎")
        for name in ("career-postgres", "career-minio"):
            result = subprocess.run(["docker", "container", "inspect", name],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                raise StartError(f"缺少已有容器 {name}，请先按项目说明完成首次数据库/对象存储部署")
        for image, dockerfile in (("career-document-sandbox:1", "deploy/sandbox/Dockerfile"),
                                  ("career-mcp:1", "deploy/mcp/Dockerfile")):
            result = subprocess.run(["docker", "image", "inspect", image],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                raise StartError(f"缺少镜像 {image}，请先构建 {dockerfile} 对应的镜像")
        say("启动 PostgreSQL、MinIO、Redis、Milvus、MCP 和同域网关…")
        self.run(["docker", "start", "career-postgres", "career-minio"], "启动业务存储")
        if not (ROOT / "deploy/.local/redis.conf").is_file():
            self.run([sys.executable, "backend/scripts/setup_redis.py"], "准备 Redis 配置")
        for file in ("redis", "milvus", "mcp", "gateway"):
            self.run(["docker", "compose", "-f", f"deploy/docker-compose.{file}.yml", "up", "-d", "--wait",
                      "--wait-timeout", str(self.timeout)], f"启动 {file}")
        self.run(["docker", "compose", "-f", "deploy/docker-compose.gateway.yml", "exec", "-T", "gateway",
                  "nginx", "-t"], "校验网关")

    def tunnel_online(self, origin=None):
        """Only reuse a HTTPS gateway that actually reaches this application."""
        url = origin or self.state.get("public_url", "")
        return bool(PUBLIC_URL.fullmatch(url)) and ready_http(url + "/_stcore/health")

    def cpolar_log_origin(self, record):
        with Path(record["log"]).open("rb") as log:
            log.seek(max(0, log.seek(0, 2) - 65536))
            urls = PUBLIC_URL.findall(log.read().decode(errors="replace"))
        for origin in reversed(list(dict.fromkeys(urls))):
            if self.tunnel_online(origin):
                return normalize_origin(origin)
        return None

    def tunnel(self):
        # Reuse an existing manual cpolar tunnel without taking ownership of it.
        # This preserves today's address and never stops a user-started process.
        record = self.state["services"].get("tunnel")
        if record and record.get("provider") == "cpolar" and self.tunnel_online():
            if owned_alive(record) or not record.get("managed"):
                say("复用已有 cpolar 公网隧道")
                return self.state["public_url"]
        if record and record.get("provider") == "cpolar" and owned_alive(record):
            # Recover a launch interrupted after cpolar connected but before
            # the public URL was saved. Do not create another random hostname.
            origin = self.cpolar_log_origin(record)
            if origin:
                self.state["public_url"] = origin
                self.save()
                say("恢复已有 cpolar 公网地址")
                return origin
        if record and record.get("managed"):
            self.stop_service("tunnel")
        candidates = []
        source = ROOT / "miniprogram/miniprogram/config.local.ts"
        if source.is_file():
            candidates = PUBLIC_URL.findall(source.read_text(encoding="utf-8"))
        previous = self.state.get("cpolar_url", "")
        if PUBLIC_URL.fullmatch(previous):
            candidates.append(previous)
        for origin in dict.fromkeys(candidates):
            if self.tunnel_online(origin):
                self.state["services"]["tunnel"] = {"managed": False, "provider": "cpolar"}
                self.state["public_url"] = origin
                self.save()
                say("复用手工启动的 cpolar（停止脚本不会结束它）")
                return origin
        if not shutil.which("cpolar"):
            raise StartError("没有找到 cpolar，请先安装并配置账号；仅本地启动可用 ./start.sh --local")
        self.state.pop("public_url", None)
        self.state.pop("ready_at", None)
        environment = {key: value for key, value in os.environ.items() if key in {
            "PATH", "HOME", "USER", "TMPDIR", "TMP", "TEMP", "SSL_CERT_FILE", "SSL_CERT_DIR",
        }}
        # cpolar reads its own account config. Do not pass application keys, or
        # expose its inspector outside the loopback interface.
        region = os.environ.get("CAREER_CPOLAR_REGION", "cn")
        if not re.fullmatch(r"[a-z_]+", region):
            raise StartError("CAREER_CPOLAR_REGION 格式错误，例如 cn、hk 或 us")
        say("创建 cpolar HTTPS 隧道…")
        record = self.spawn("tunnel", ["cpolar", "http", "-daemon=off", "-processMode=single",
                                     "-log=stdout", "-log-level=INFO",
                                     "-inspect-addr=127.0.0.1:0", "-proto=https", "-tunnelName=career-web",
                                     "-region=" + region, "127.0.0.1:8080"], ROOT, environment)
        record["provider"] = "cpolar"
        self.save()

        def found():
            # Read only our freshly created log, never unrelated dashboard or
            # documentation URLs, and validate the public route before using it.
            origin = self.cpolar_log_origin(record)
            if origin:
                self.state["public_url"] = origin
                self.save()
                return True
            return False

        self.wait(found, "cpolar 公网隧道（请确认 cpolar 账号已配置）", service="tunnel", timeout=90)
        return self.state["public_url"]

    def start(self, local_only=False):
        if not (ROOT / "backend/.env").is_file():
            raise StartError("backend/.env 不存在，请先完成后端配置")
        for module in ("streamlit", "uvicorn", "httpx"):
            import importlib.util
            if importlib.util.find_spec(module) is None:
                raise StartError(f"当前 Python 缺少 {module}，请使用 career-agent 环境或设置 CAREER_PYTHON")
        self.prepare_docker()
        config = dotenv_values(ROOT / "backend/.env")
        runner_socket = Path(config.get("SANDBOX_RUNNER_SOCKET") or ROOT / "deploy/.local/sandbox.sock")
        runner_env = os.environ.copy()
        runner_env["SANDBOX_RUNNER_SOCKET"] = str(runner_socket)
        for name in ("MCP_IMAGE", "MCP_SEARCH_SOCKET_VOLUME"):
            if config.get(name):
                runner_env[name] = config[name]
        self.ensure("sandbox", [sys.executable, str(ROOT / "deploy/sandbox/start.py")], ROOT,
                    lambda: runner_ready(runner_socket), lambda: runner_connected(runner_socket), runner_env)
        frontend_env = os.environ.copy()
        frontend_env.update(API_BASE_URL="http://127.0.0.1:8000", BROWSER_API_BASE_URL="auto")
        bypass = frontend_env.get("NO_PROXY") or frontend_env.get("no_proxy") or ""
        frontend_env["NO_PROXY"] = ",".join(filter(None, (bypass, "localhost", "127.0.0.1", "::1")))
        self.ensure("frontend", [sys.executable, "-m", "streamlit", "run", "app.py", "--server.address",
                                  "127.0.0.1", "--server.port", "8501", "--server.headless", "true",
                                  "--browser.gatherUsageStats", "false"], ROOT / "frontend",
                    lambda: ready_http("http://127.0.0.1:8501/_stcore/health"), lambda: port_open(8501), frontend_env)
        self.wait(lambda: ready_http("http://127.0.0.1:8080/_stcore/health"), "网关转发")
        if local_only:
            self.stop_service("tunnel")
            origin = "http://127.0.0.1:8080"
        else:
            origin = self.tunnel()
        old_origins = (dotenv_values(ROOT / "backend/.env").get("CORS_ORIGINS") or "").split(",")
        allow_origin(origin)
        backend = self.state["services"].get("backend")
        if owned_alive(backend) and origin not in old_origins:
            self.stop_service("backend")
        self.ensure("backend", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                 "--port", "8000"], ROOT / "backend",
                    lambda: ready_http("http://127.0.0.1:8000/health", json_status=True), lambda: port_open(8000))
        if not login_origin_ready(origin):
            if owned_alive(self.state["services"].get("backend")):
                self.stop_service("backend")
                self.ensure("backend", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                         "--port", "8000"], ROOT / "backend",
                            lambda: ready_http("http://127.0.0.1:8000/health", json_status=True), lambda: port_open(8000))
                self.wait(lambda: login_origin_ready(origin), "登录来源校验")
            else:
                raise StartError("手工启动的后端尚未加载新域名。请在原终端停止它，再运行 ./start.sh；未终止该进程")
        if not local_only:
            say("等待公网入口连通…")
            self.wait(lambda: ready_http(origin + "/_stcore/health"), "公网入口", service="tunnel" if self.state["services"].get("tunnel", {}).get("managed") else None, timeout=90)
        sync_miniprogram_origin(origin)
        self.state["ready_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        say("\n启动完成")
        say("本地页面：http://127.0.0.1:8501")
        say("统一入口：http://127.0.0.1:8080")
        if not local_only:
            say(f"公网页面：{origin}")
        say(f"日志目录：{self.logs}\n查看状态：./start.sh status\n停止服务：./start.sh stop")

    def stop(self, containers=False):
        for name in ("tunnel", "frontend", "backend", "sandbox"):
            self.stop_service(name)
        self.state.pop("ready_at", None)
        self.save()
        if containers:
            for name in CONTAINERS:
                exists = subprocess.run(["docker", "container", "inspect", name],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if exists.returncode == 0:
                    self.run(["docker", "stop", name], f"停止 {name}")
            say("项目容器已停止，数据卷保留")
        else:
            say("脚本管理的进程已停止，Docker 容器保留运行")

    def status(self):
        for name in ("sandbox", "backend", "frontend", "tunnel"):
            record = self.state["services"].get(name)
            value = "运行中" if owned_alive(record) else "未运行"
            if record and not record.get("managed"):
                value = "手工服务（不由脚本管理）"
            say(f"{name}: {value}")
        if self.state["services"].get("tunnel") and self.state.get("public_url"):
            say(f"公网页面：{self.state.get('public_url', '等待域名')}")
        say(f"日志目录：{self.logs}")


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0].startswith("-"):
        values.insert(0, "start")
    parser = argparse.ArgumentParser(description="智能求职系统本地一键启动/停止")
    parser.add_argument("action", choices=("start", "stop", "status", "restart"))
    parser.add_argument("--local", action="store_true", help="仅本地启动，关闭脚本管理的公网隧道")
    parser.add_argument("--containers", action="store_true", help="stop 时也停止项目 Docker 容器，保留数据卷")
    args = parser.parse_args(values)
    if args.containers and args.action != "stop":
        parser.error("--containers 仅用于 stop")
    if args.local and args.action not in ("start", "restart"):
        parser.error("--local 仅用于 start 或 restart")
    os.umask(0o077)
    try:
        stack = Stack()
        with stack.lock():
            if args.action == "status":
                stack.status()
            elif args.action == "stop":
                stack.stop(args.containers)
            else:
                if args.action == "restart":
                    stack.stop()
                stack.start(args.local)
    except (StartError, OSError, subprocess.SubprocessError) as exc:
        say(f"\n启动器：{exc}\n日志：{LOCAL / 'logs'}\n已启动的服务保留，可修正后重试或运行 ./start.sh stop")
        return 1
    except KeyboardInterrupt:
        say("\n启动检查已中止，已启动服务保留；可运行 ./start.sh stop")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
