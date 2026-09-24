"""Fixed MCP container arguments for the trusted Runner; no backend secrets."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT_ROOT / "backend" / "mcp_workspace"
DATABASE_SECRET = PROJECT_ROOT / "deploy" / "mcp" / "secrets" / "database.json"
DOCKER_ENV = {key: os.environ[key] for key in
              ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "XDG_RUNTIME_DIR")
              if key in os.environ}


def docker_connection(mode: str, *, owner_id: str | None = None, run_id: str | None = None) -> dict:
    """Only fixed, operator-owned argv; the MCP client cannot choose mounts or image."""
    args = ["run", "--rm", "-i", "--pull=never", "--read-only", "--cap-drop=ALL",
            f"--label=career.mcp.scope={mode}",
            "--security-opt=no-new-privileges:true", "--memory=256m", "--cpus=1",
            "--pids-limit=64", "--tmpfs=/tmp:rw,noexec,nosuid,size=16m"]
    if mode == "search":
        if owner_id is not None or run_id is not None:
            owner = uuid.UUID(str(owner_id))
            run = uuid.UUID(str(run_id))
            args += ["--label", f"career.owner={hashlib.sha256(owner.bytes).hexdigest()[:24]}",
                     "--label", f"career.run={run}"]
        volume = os.environ.get("MCP_SEARCH_SOCKET_VOLUME", "career-mcp-search-socket")
        args += ["--network=none", "--user=10001:10001", "--mount",
                 f"type=volume,src={volume},dst=/run/search,readonly"]
    elif mode == "filesystem":
        if owner_id is not None or run_id is not None:
            raise ValueError("Filesystem MCP only serves approved shared materials")
        WORKSPACE.mkdir(mode=0o755, exist_ok=True)
        if WORKSPACE.is_symlink():
            raise ValueError("MCP 共享目录不能是符号链接")
        args += ["--network=none", "--user=10001:10001", "--mount",
                 f"type=bind,src={WORKSPACE},dst=/workspace,readonly"]
    elif mode == "postgres":
        if owner_id is not None or run_id is not None:
            raise ValueError("Database MCP only exposes aggregate views")
        if not DATABASE_SECRET.is_file() or DATABASE_SECRET.is_symlink():
            raise ValueError("未配置专用 MCP 只读账号；先运行 deploy/mcp/provision_db.py")
        args += ["--network=bridge", "--user=0:0", "--cap-add=NET_ADMIN",
                 "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=SETPCAP", "--cap-add=DAC_OVERRIDE",
                 "--add-host=host.docker.internal:host-gateway", "--mount",
                 f"type=bind,src={DATABASE_SECRET},dst=/run/secrets/database.json,readonly"]
    else:
        raise ValueError("Unknown MCP service")
    program = "entrypoint.py" if mode == "postgres" else "server.py"
    image = os.environ.get("MCP_IMAGE", "career-mcp:1")
    args += [image, "python", "-u", f"/app/{program}", mode]
    return {"transport": "stdio", "command": "docker", "args": args,
            "env": DOCKER_ENV}
