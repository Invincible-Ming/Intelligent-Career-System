"""Exercise actual container boundaries without cloud models or secret output."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.mcp_tools import ALLOWED_TOOLS, MCPService  # noqa: E402

# 只有由固定 argv 启动本地容器的模式才存在 label、只读挂载和 NetworkMode 可断言。
# baidu_map 走远程 Streamable HTTP，没有本地容器；filesystem / postgres 的容器启动参数
# 已随 docker_connection() 下线，后端也不再加载，因此不在此清单内。
CONTAINER_MODES = ("search",)


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


PROBE = r'''
import os,socket,json,sys
status=dict(line.split(':',1) for line in open('/proc/1/status') if ':' in line)
assert status['Uid'].split()==['10001']*4
for field in ('CapEff','CapPrm','CapBnd','CapAmb'):
 assert int(status[field].strip(),16)==0,field
assert status['NoNewPrivs'].strip()=='1'
environment=dict(item.split('=',1) for item in open('/proc/1/environ','rb').read().decode().split('\0') if '=' in item)
for key in ('DASHSCOPE_API_KEY','MINIO_SECRET_KEY','MCP_ADMIN_DATABASE_URL','DATABASE_URL'):
 assert key not in environment
assert not os.path.exists('/var/run/docker.sock')
assert not os.path.exists('/Users/crazyzm/pythonCode/Intelligent_Career_System/backend/.env')
targets=[('127.0.0.1',8000),('10.0.0.1',80),('192.168.1.1',80),('169.254.169.254',80),('8.8.8.8',443),('::1',80)]
targets += [(address,8000) for address in json.loads(sys.argv[1])]
for host,port in targets:
 try:
  stream=socket.create_connection((host,port),timeout=.3)
 except OSError:
  continue
 else:
  stream.close();raise AssertionError('Forbidden network target reachable')
try:
 open('/app/write-probe','w').close()
except OSError:
 pass
else:
 raise AssertionError('Root filesystem is writable')
print('uid=10001 capabilities=0 readonly-root=true forbidden-network=blocked')
'''


async def verify():
    unknown = set(CONTAINER_MODES) - set(ALLOWED_TOOLS)
    assert not unknown, f"脚本验证的模式不在后端能力白名单中: {sorted(unknown)}"

    service = MCPService()
    await service.initialize()
    # 只验证真正加载成功的容器模式，不再要求加载清单与 ALLOWED_TOOLS 完全相等：
    # baidu_map 未配置时不会出现在 tools_by_service 中，但那不构成启动失败。
    loaded = [mode for mode in CONTAINER_MODES if mode in service.tools_by_service]
    if not loaded:
        raise RuntimeError("本地 MCP 容器未启动: " + "; ".join(service.startup_errors))
    if service.client is None:
        raise RuntimeError("MCP 客户端未初始化，请确认 MCP_ENABLED 与沙箱 Runner 已就绪")

    for mode in loaded:
        async with service.client.session(mode) as session:
            inventory = await session.list_tools()
            names = {tool.name for tool in inventory.tools}
            allowed = ALLOWED_TOOLS[mode]
            assert allowed is None or names == allowed, f"{mode} 工具清单与白名单不一致: {sorted(names)}"
            ids = docker("ps", "--filter", f"label=career.mcp.scope={mode}", "--format", "{{.ID}}").splitlines()
            assert len(ids) == 1, "Run verification when no other MCP tasks are active"
            container = ids[0]
            config = json.loads(docker("inspect", container, "--format", "{{json .HostConfig}}"))
            assert config["ReadonlyRootfs"] and not config["Privileged"]
            assert config["NetworkMode"] == "none"
            mounts = json.loads(docker("inspect", container, "--format", "{{json .Mounts}}"))
            assert mounts and all(not mount["RW"] for mount in mounts)
            assert all("docker.sock" not in mount["Source"] for mount in mounts)
            networks = json.loads(docker("inspect", container, "--format", "{{json .NetworkSettings.Networks}}"))
            gateways = [item["Gateway"] for item in networks.values() if item.get("Gateway")]
            print(mode, docker("exec", "--user", "10001:10001", container, "python", "-c", PROBE, json.dumps(gateways)))
            denied = await session.call_tool("search_web", {"query": "a" * 301})
            assert denied.isError

    networks = json.loads(
        docker("inspect", "career-mcp-search-broker-1", "--format", "{{json .NetworkSettings.Networks}}"))
    gateways = [item["Gateway"] for item in networks.values() if item.get("Gateway")]
    print("broker", docker("exec", "--user", "10001:10001", "career-mcp-search-broker-1", "python", "-c", PROBE,
                           json.dumps(gateways)))
    result = await service.get_tools("search")[0].ainvoke(
        {"query": "Python backend developer requirements", "count": 2})
    text = " ".join(block.get("text", "") for block in result) if isinstance(result, list) else str(result)
    evidence = json.loads(text)
    assert evidence.get("results") and evidence.get("untrusted_content") is True
    print("live_search_ok", len(evidence["results"]))
    await service.close()


if __name__ == "__main__":
    asyncio.run(verify())
