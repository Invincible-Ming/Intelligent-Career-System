"""Container-only MCP tools, with explicit capabilities for each caller."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import settings
from app.services.mcp_container_config import docker_connection

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_TOOLS = {
    "search": frozenset({"search_web"}),
    # 远程官方服务，工具清单由服务端决定，不做强等校验
    "baidu_map": None,
}


def runner_connection(mode: str, *, owner_id: str | None = None, run_id: str | None = None) -> dict:
    """Backend talks to local Runner; only Runner starts the fixed Docker command."""
    if mode not in ALLOWED_TOOLS:
        raise ValueError("Unknown MCP service")
    if owner_id is not None or run_id is not None:
        if mode != "search":
            raise ValueError("Per-run MCP scope is search only")
        uuid.UUID(str(owner_id))
        uuid.UUID(str(run_id))
    bridge = PROJECT_ROOT / "deploy" / "sandbox" / "mcp_bridge.py"
    return {"transport": "stdio", "command": sys.executable,
            "args": [str(bridge), mode, settings.SANDBOX_RUNNER_SOCKET,
                     owner_id or "", run_id or ""],
            "env": {"PATH": os.environ.get("PATH", os.defpath)}}


class MCPService:
    def __init__(self):
        self.client: MultiServerMCPClient | None = None
        self.tools: list = []
        self.tools_by_service: dict[str, list] = {}
        self.startup_errors: list[str] = []

    async def initialize(self) -> None:
        self.tools = []
        self.tools_by_service = {}
        self.startup_errors = []
        if not settings.MCP_ENABLED:
            return
        connections = {}
        modes = ["search"]
        for mode in modes:
            try:
                connections[mode] = runner_connection(mode)
            except Exception:
                error = f"MCP {mode} 配置不可用；该服务已禁用"
                self.startup_errors.append(error)
                logger.warning(error)
        if settings.BAIDU_MAP_MCP_URL:
            connections["baidu_map"] = {
                "transport": "streamable_http", "url": settings.BAIDU_MAP_MCP_URL,
            }
        self.client = MultiServerMCPClient(connections)
        for mode in connections:
            try:
                tools = await asyncio.wait_for(self.client.get_tools(server_name=mode), settings.MCP_STARTUP_TIMEOUT)
                names = {tool.name for tool in tools}
                allowed = ALLOWED_TOOLS.get(mode)
                if allowed is not None and names != allowed:
                    raise PermissionError("MCP tool inventory differs from the capability allowlist")
                self.tools_by_service[mode] = tools
                self.tools.extend(tools)
                logger.info("已加载受限 MCP 服务 %s: %s", mode, sorted(names))
            except Exception:
                # A driver exception may embed credentials; don't log it.
                error = f"MCP {mode} 容器不可用或工具校验失败；无本机工具回退"
                self.startup_errors.append(error)
                logger.warning(error)

    async def close(self) -> None:
        self.client = None
        self.tools = []
        self.tools_by_service = {}

    def get_tools(self, capability: str = "search") -> list:
        """Default is search-only; callers must explicitly request another scope."""
        if capability not in ALLOWED_TOOLS:
            raise ValueError("Unknown MCP capability")
        return list(self.tools_by_service.get(capability, []))

    async def get_job_search_tools(self, *, owner_id: str, run_id: str) -> list:
        """Fresh, search-only MCP connection tied to one authenticated run."""
        if not settings.MCP_ENABLED:
            return []
        connection = runner_connection("search", owner_id=owner_id, run_id=run_id)
        client = MultiServerMCPClient({"search": connection})
        try:
            tools = await asyncio.wait_for(client.get_tools(server_name="search"), settings.MCP_STARTUP_TIMEOUT)
            if {tool.name for tool in tools} != ALLOWED_TOOLS["search"]:
                raise PermissionError("Unexpected search tool inventory")
            return tools
        except Exception:
            logger.warning("按任务隔离的 MCP 搜索不可用，跳过调研")
            return []


mcp_service = MCPService()
