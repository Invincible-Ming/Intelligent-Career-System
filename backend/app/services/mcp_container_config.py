"""Fixed MCP container arguments for the trusted Runner; no backend secrets."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
    else:
        raise ValueError("Unknown MCP service")
    program = "server.py"
    image = os.environ.get("MCP_IMAGE", "career-mcp:1")
    args += [image, "python", "-u", f"/app/{program}", mode]
    return {"transport": "stdio", "command": "docker", "args": args,
            "env": DOCKER_ENV}
