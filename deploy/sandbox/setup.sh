#!/usr/bin/env bash
# Install a dedicated Linux Runner account and render its systemd service.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "请以 root 身份运行" >&2
    exit 1
fi

RUNNER_GROUP="${SANDBOX_RUNNER_GROUP:-career-sandbox}"
RUNNER_USER="${SANDBOX_RUNNER_USER:-career-runner}"
BACKEND_USER="${SANDBOX_BACKEND_USER:?请设置 SANDBOX_BACKEND_USER 为后端服务账号}"
PROJECT_ROOT="$(realpath "${1:-/opt/career-system}")"
PYTHON_BIN="$(realpath "${SANDBOX_PYTHON:-$(command -v python3)}")"

for value in "$RUNNER_GROUP" "$RUNNER_USER" "$BACKEND_USER" "$PROJECT_ROOT" "$PYTHON_BIN"; do
    if [[ "$value" == *[\ \|\&\\]* ]]; then
        echo "账号名称和路径不能包含空格或模板分隔符" >&2
        exit 1
    fi
done
if [[ ! -d "$PROJECT_ROOT/backend" || ! -f "$PROJECT_ROOT/deploy/sandbox/start.py" || ! -x "$PYTHON_BIN" ]]; then
    echo "项目路径或 Python 解释器不可用" >&2
    exit 1
fi

if ! getent group "$RUNNER_GROUP" >/dev/null; then
    groupadd --system "$RUNNER_GROUP"
fi
if ! id "$RUNNER_USER" >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin \
        --gid "$RUNNER_GROUP" "$RUNNER_USER"
fi
id "$BACKEND_USER" >/dev/null
if [[ "$BACKEND_USER" == "$RUNNER_USER" ]]; then
    echo "Runner 与后端必须使用不同账号" >&2
    exit 1
fi
usermod -aG "$RUNNER_GROUP" "$BACKEND_USER"

# Only the Runner may control Docker; the backend joins the socket group above,
# never the Docker group. Do not transfer ownership of deploy/.local: it also
# contains unrelated application configuration such as the Redis password.
if [[ -S /var/run/docker.sock ]]; then
    getent group docker >/dev/null || { echo "Docker 组不存在" >&2; exit 1; }
    usermod -aG docker "$RUNNER_USER"
fi

sed -e "s|@RUNNER_USER@|$RUNNER_USER|g" \
    -e "s|@RUNNER_GROUP@|$RUNNER_GROUP|g" \
    -e "s|@PROJECT_ROOT@|$PROJECT_ROOT|g" \
    -e "s|@PYTHON@|$PYTHON_BIN|g" \
    "$PROJECT_ROOT/deploy/sandbox/sandbox-runner.service" \
    > /etc/systemd/system/sandbox-runner.service
systemctl daemon-reload

echo "已安装 sandbox-runner.service。将 backend/.env 中的 SANDBOX_RUNNER_SOCKET 设为 /run/career/sandbox.sock。"
echo "确认 $RUNNER_USER 可读取项目代码和 Python 环境后，运行: systemctl enable --now sandbox-runner"
