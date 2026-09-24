"""Exercise actual container boundaries without cloud models or secret output."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.mcp_tools import ALLOWED_TOOLS, MCPService


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
# A DB target is the only permitted private endpoint, and only on its DB port.
if environment.get('MCP_DB_ADDRESS'):
 targets.append((environment['MCP_DB_ADDRESS'],8000))
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
if os.path.isdir('/workspace'):
 try:
  open('/workspace/write-probe','w').close()
 except OSError:
  pass
 else:
  raise AssertionError('Workspace mount is writable')
print('uid=10001 capabilities=0 readonly-root=true forbidden-network=blocked')
'''


async def verify():
    service = MCPService()
    await service.initialize()
    if set(service.tools_by_service) != set(ALLOWED_TOOLS):
        raise RuntimeError("Some MCP services failed startup: " + "; ".join(service.startup_errors))
    for mode in ALLOWED_TOOLS:
        async with service.client.session(mode) as session:
            inventory = await session.list_tools()
            assert {tool.name for tool in inventory.tools} == ALLOWED_TOOLS[mode]
            ids = docker("ps", "--filter", f"label=career.mcp.scope={mode}", "--format", "{{.ID}}").splitlines()
            assert len(ids) == 1, "Run verification when no other MCP tasks are active"
            container = ids[0]
            config = json.loads(docker("inspect", container, "--format", "{{json .HostConfig}}"))
            assert config["ReadonlyRootfs"] and not config["Privileged"]
            assert config["NetworkMode"] == ("bridge" if mode == "postgres" else "none")
            mounts = json.loads(docker("inspect", container, "--format", "{{json .Mounts}}"))
            assert mounts and all(not mount["RW"] for mount in mounts)
            assert all("docker.sock" not in mount["Source"] for mount in mounts)
            networks = json.loads(docker("inspect", container, "--format", "{{json .NetworkSettings.Networks}}"))
            gateways = [item["Gateway"] for item in networks.values() if item.get("Gateway")]
            print(mode, docker("exec", "--user", "10001:10001", container, "python", "-c", PROBE, json.dumps(gateways)))
            if mode == "filesystem":
                allowed = await session.call_tool("read_text_file", {"path": "README.txt"})
                assert not allowed.isError
                for path in ("../.env", "/etc/passwd", ".env"):
                    denied = await session.call_tool("read_text_file", {"path": path})
                    assert denied.isError
            if mode == "postgres":
                for view in ("knowledge_inventory", "evaluation_summary"):
                    allowed = await session.call_tool("read_statistics", {"view": view, "limit": 1})
                    assert not allowed.isError
                denied = await session.call_tool("read_statistics", {"view": "documents", "limit": 1})
                assert denied.isError
                denied = await session.call_tool("read_statistics",
                                                 {"view": "knowledge_inventory; DROP TABLE documents", "limit": 1})
                assert denied.isError
            if mode == "search":
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
