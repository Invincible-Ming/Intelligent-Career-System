"""Security boundaries; no cloud model or running database required."""
import asyncio
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

MCP_DIR = Path(__file__).resolve().parents[2] / "deploy/mcp"
sys.path.insert(0, str(MCP_DIR))
from policy import MAX_FILE_BYTES, public_address, read_workspace_file, result_url, validate_query
from db_guard import check_identity
from broker import SearchResults
from app.services.mcp_tools import MCPService, docker_connection, runner_connection
from app.services.mcp_container_config import DOCKER_ENV


class NetworkPolicyTests(unittest.TestCase):
    def test_search_parses_evidence_and_drops_local_links(self):
        parser = SearchResults()
        parser.feed('<li class="b_algo"><h2><a href="https://example.com">Python 岗位</a></h2><p>技能要求</p></li>'
                    '<li class="b_algo"><h2><a href="http://127.0.0.1">恶意链接</a></h2><p>内容</p></li>')
        self.assertEqual(parser.results,
                         [{"title": "Python 岗位", "snippet": "技能要求", "url": "https://example.com"}])

    def test_nonpublic_addresses_denied(self):
        for address in ("127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254",
                        "0.0.0.0", "100.64.0.1", "192.0.2.1", "224.0.0.1", "::1", "::ffff:127.0.0.1"):
            self.assertFalse(public_address(address), address)
        self.assertTrue(public_address("8.8.8.8"))

    def test_query_bounded(self):
        self.assertEqual(validate_query(" Python 招聘 ", 5), "Python 招聘")
        for query, count in (("", 1), ("a" * 301, 1), ("a\n", 1), ("a\x00b", 1), ("a", 6), ("a", True)):
            with self.assertRaises(ValueError):
                validate_query(query, count)

    def test_citation_urls_do_not_accept_local_targets(self):
        for url in ("file:///etc/passwd", "javascript:alert(1)", "http://localhost", "http://host.docker.internal",
                    "http://127.0.0.1", "http://169.254.169.254", "http://user:password@example.com",
                    "http://example.com:5432"):
            self.assertIsNone(result_url(url), url)
        self.assertEqual(result_url("https://example.com/jobs"), "https://example.com/jobs")


class FilePolicyTests(unittest.TestCase):
    def test_read_and_traversal_symlinks(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            root = Path(directory)
            (root / "notes.txt").write_text("公开资料", encoding="utf-8")
            (Path(other) / "secret.txt").write_text("private")
            (root / "link").symlink_to(other, target_is_directory=True)
            (root / "alias.txt").symlink_to(Path(other) / "secret.txt")
            self.assertEqual(read_workspace_file(root, "notes.txt"), "公开资料")
            for name in ("../secret.txt", "/etc/passwd", ".env", "link/secret.txt", "alias.txt", "..\\secret.txt"):
                with self.assertRaises((ValueError, OSError)):
                    read_workspace_file(root, name)

    def test_large_and_binary_files_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "large.txt").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
            (root / "binary.bin").write_bytes(b"\xff\xfe")
            for name in ("large.txt", "binary.bin"):
                with self.assertRaises((ValueError, UnicodeError)):
                    read_workspace_file(root, name)

    def test_fifo_cannot_block_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            os.mkfifo(Path(directory) / "pipe")
            with self.assertRaises(ValueError):
                read_workspace_file(directory, "pipe")


class CapabilityTests(unittest.TestCase):
    def test_research_never_gets_files_or_database(self):
        service = MCPService()
        search = SimpleNamespace(name="search_web")
        service.tools_by_service = {"search": [search], "filesystem": [SimpleNamespace(name="read_text_file")],
                                    "postgres": [SimpleNamespace(name="read_statistics")]}
        self.assertEqual(service.get_tools(), [search])
        self.assertEqual(service.get_tools("search"), [search])
        with self.assertRaises(ValueError):
            service.get_tools("all")

    def test_search_and_files_have_no_network_or_host_secrets(self):
        for mode in ("search", "filesystem"):
            connection = docker_connection(mode)
            args = connection["args"]
            self.assertEqual(connection["command"], "docker")
            self.assertIn("--network=none", args)
            self.assertIn("--user=10001:10001", args)
            self.assertIn("--read-only", args)
            self.assertIn("--cap-drop=ALL", args)
            self.assertFalse(any("docker.sock" in arg or "database.json" in arg or ".env" in arg for arg in args))
            self.assertTrue(all(arg.endswith("readonly") for arg in args if arg.startswith("type=")))
            self.assertNotIn("DASHSCOPE_API_KEY", connection["env"])

    def test_backend_mcp_transport_only_connects_to_runner(self):
        owner, run = uuid.uuid4(), uuid.uuid4()
        for mode in ("search", "filesystem", "postgres"):
            connection = runner_connection(mode)
            self.assertEqual(connection["command"], sys.executable)
            self.assertTrue(connection["args"][0].endswith("deploy/sandbox/mcp_bridge.py"))
            self.assertNotIn("docker", connection["args"])
            self.assertNotIn("DASHSCOPE_API_KEY", connection["env"])
        search = runner_connection("search", owner_id=str(owner), run_id=str(run))
        self.assertEqual(search["args"][-2:], [str(owner), str(run)])
        with self.assertRaises(ValueError):
            runner_connection("filesystem", owner_id=str(owner), run_id=str(run))

    def test_runner_container_config_does_not_load_backend_secrets(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; import app.services.mcp_container_config; "
                                 "assert 'app.core.config' not in sys.modules"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertTrue(set(DOCKER_ENV) <= {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT",
                                            "DOCKER_CONFIG", "XDG_RUNTIME_DIR"})

    def test_unexpected_tool_inventory_fails_closed(self):
        client = SimpleNamespace(get_tools=AsyncMock(return_value=[SimpleNamespace(name="execute_shell")]))
        service = MCPService()
        with patch("app.services.mcp_tools.MultiServerMCPClient", return_value=client), patch(
                "app.services.mcp_tools.settings.MCP_FILES_ENABLED", False), patch(
                "app.services.mcp_tools.settings.MCP_DATABASE_ENABLED", False):
            asyncio.run(service.initialize())
        self.assertEqual(service.get_tools(), [])
        self.assertTrue(service.startup_errors)

    def test_job_search_container_has_only_hashed_owner_and_run_labels(self):
        owner, run = uuid.uuid4(), uuid.uuid4()
        args = docker_connection("search", owner_id=str(owner), run_id=str(run))["args"]
        self.assertIn(f"career.owner={hashlib.sha256(owner.bytes).hexdigest()[:24]}", args)
        self.assertIn(f"career.run={run}", args)
        self.assertNotIn(str(owner), args)
        self.assertIn("--network=none", args)
        with self.assertRaises(ValueError):
            docker_connection("search", owner_id="../../etc", run_id=str(run))


class DatabasePolicyTests(unittest.TestCase):
    def test_privileged_identity_and_inherited_access_denied(self):
        role = dict(rolname="career_mcp_reader", rolsuper=False, rolcreaterole=False, rolcreatedb=False,
                    rolreplication=False, rolbypassrls=False)
        access = dict(extra_access=False, membership=False, schema_create=False, database_create=False)
        check_identity(role, access, "on")
        for key in access:
            with self.assertRaises(PermissionError):
                check_identity(role, {**access, key: True}, "on")
        with self.assertRaises(PermissionError):
            check_identity({**role, "rolsuper": True}, access, "on")
        with self.assertRaises(PermissionError):
            check_identity(role, access, "off")


if __name__ == "__main__":
    unittest.main()
