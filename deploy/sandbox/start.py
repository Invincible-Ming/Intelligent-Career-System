"""Start the Docker-owning Runner on a local, private Unix socket."""
from pathlib import Path
import grp
import os
import signal
import socket
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
# Local shells may export application credentials; the Docker-owning Runner
# has no use for them. Its Docker subprocesses get a separate allowlisted env.
for secret_name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY", "DATABASE_URL",
                    "MCP_ADMIN_DATABASE_URL", "REDIS_URL", "MINIO_ACCESS_KEY",
                    "MINIO_SECRET_KEY", "MILVUS_TOKEN"):
    os.environ.pop(secret_name, None)
path = Path(os.environ.get("SANDBOX_RUNNER_SOCKET", str(ROOT / "deploy" / ".local" / "sandbox.sock")))
if not path.is_absolute():
    raise SystemExit("SANDBOX_RUNNER_SOCKET must be an absolute path")
directory = path.parent
directory.mkdir(mode=0o700, parents=True, exist_ok=True)
if directory.is_symlink():
    raise SystemExit("Sandbox socket directory must not be a symlink")
group_name = os.environ.get("SANDBOX_RUNNER_GROUP")
if group_name:
    group_id = grp.getgrnam(group_name).gr_gid
    os.chown(directory, -1, group_id)
    directory.chmod(0o770)
else:
    directory.chmod(0o700)
if path.exists():
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(str(path)) == 0:
            raise SystemExit("Sandbox Runner is already running")
    path.unlink()
os.umask(0o077)
os.chdir(ROOT)
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(str(path))
if group_name:
    os.chown(path, -1, group_id)
    path.chmod(0o660)
else:
    path.chmod(0o600)
listener.listen(128)


def exit_on_term(_signum, _frame):
    # Uvicorn re-raises a captured SIGTERM after shutdown. Convert that final
    # signal into an exception so the socket cleanup below still runs.
    raise SystemExit(0)


signal.signal(signal.SIGTERM, exit_on_term)
try:
    uvicorn.run("deploy.sandbox.runner:app", fd=listener.fileno(), log_level="info")
finally:
    listener.close()
    path.unlink(missing_ok=True)
