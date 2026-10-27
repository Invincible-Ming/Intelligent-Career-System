"""Launcher tests use temporary files and disposable processes, never real Docker or models."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "local_stack_launcher", Path(__file__).resolve().parents[2] / "deploy/local_stack.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_state_is_private_and_atomic(tmp_path):
    path = tmp_path / "state.json"
    launcher.atomic_state(path, {"services": {}, "public_url": "https://sample.trycloudflare.com"})
    assert json.loads(path.read_text())["services"] == {}
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".state-*"))


def test_stop_never_signals_a_reused_pid_or_unmanaged_service(tmp_path):
    stack = launcher.Stack(tmp_path)
    for record in ({"managed": False, "pid": 123}, {"managed": True, "pid": 123, "identity": "old-process"}):
        stack.state["services"]["backend"] = record
        with patch.object(launcher, "process_identity", return_value="new-process"), \
             patch.object(launcher.os, "killpg") as kill:
            stack.stop_service("backend")
            kill.assert_not_called()


def test_spawn_reuse_and_stop_only_owned_process(tmp_path):
    stack = launcher.Stack(tmp_path, timeout=3)
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    record = stack.spawn("synthetic", command, tmp_path)
    try:
        assert launcher.owned_alive(record)
        assert Path(record["log"]).stat().st_mode & 0o777 == 0o600
        with patch.object(stack, "spawn") as spawn:
            stack.ensure("synthetic", command, tmp_path, lambda: True, lambda: True)
            spawn.assert_not_called()
        stack.stop_service("synthetic")
        assert not launcher.owned_alive(record)
    finally:
        process = stack.children["synthetic"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def test_healthy_hand_started_service_is_reused_and_unhealthy_port_is_rejected(tmp_path):
    stack = launcher.Stack(tmp_path)
    with patch.object(stack, "spawn") as spawn:
        stack.ensure("frontend", [], tmp_path, lambda: True, lambda: True)
        assert stack.state["services"]["frontend"] == {"managed": False}
        with pytest.raises(launcher.StartError, match="健康检查失败"):
            stack.ensure("backend", [], tmp_path, lambda: False, lambda: True)
        spawn.assert_not_called()


def test_cpolar_log_discovery_reuse_and_secret_free_environment(tmp_path):
    stack = launcher.Stack(tmp_path / "state")
    log = tmp_path / "tunnel.log"
    log.write_text("docs https://www.cpolar.com/docs\nold https://old.trycloudflare.com\nurl=https://example.r9.cpolar.top\n")

    def spawn(name, command, cwd, environment):
        assert command[0:2] == ["cpolar", "http"]
        assert "127.0.0.1:8080" in command
        assert "-inspect-addr=127.0.0.1:0" in command
        assert "-proto=https" in command
        assert "DASHSCOPE_API_KEY" not in environment
        assert "HTTPS_PROXY" not in environment
        assert "TUNNEL_TOKEN" not in environment
        stack.state["services"][name] = {"managed": True, "log": str(log)}
        return stack.state["services"][name]

    with patch.object(launcher, "ROOT", tmp_path), \
         patch.object(stack, "spawn", side_effect=spawn) as start, \
         patch.object(launcher, "owned_alive", side_effect=lambda record: bool(record)), \
         patch.object(launcher, "ready_http", return_value=True), \
         patch.object(launcher.shutil, "which", return_value="/synthetic/cpolar"), \
         patch.dict(os.environ, {"DASHSCOPE_API_KEY": "synthetic-only", "TUNNEL_TOKEN": "synthetic-only", "HTTPS_PROXY": "synthetic-only"}):
        assert stack.tunnel() == "https://example.r9.cpolar.top"
        assert stack.tunnel() == "https://example.r9.cpolar.top"
        assert start.call_count == 1
        assert stack.state["services"]["tunnel"]["provider"] == "cpolar"


def test_origin_probe_does_not_send_any_cookie():
    with patch.object(launcher, "fetch", return_value=(200, b'{"cleared":true}')) as fetch:
        assert launcher.login_origin_ready("https://career.example")
        kwargs = fetch.call_args.kwargs
        assert kwargs["headers"]["Host"] == "career.example"
        assert "Cookie" not in kwargs["headers"]
        assert b"return_to=https%3A%2F%2Fcareer.example%2F" == kwargs["data"]


def test_simultaneous_launchers_cannot_mutate_state(tmp_path):
    first, second = launcher.Stack(tmp_path), launcher.Stack(tmp_path)
    with first.lock():
        with pytest.raises(launcher.StartError, match="正在执行"):
            with second.lock():
                pass


def test_failed_child_is_reported_with_log_path(tmp_path):
    stack = launcher.Stack(tmp_path, timeout=2)
    with patch.object(launcher, "owned_alive", return_value=False):
        stack.state["services"]["backend"] = {"log": str(tmp_path / "synthetic.log")}
        with pytest.raises(launcher.StartError, match="进程已退出"):
            stack.wait(lambda: False, "backend", service="backend")


def test_public_origin_is_configured_before_backend_starts(tmp_path):
    root = tmp_path / "project"
    (root / "backend").mkdir(parents=True)
    (root / "backend/.env").write_text("CORS_ORIGINS=http://localhost:8501\n")
    stack = launcher.Stack(tmp_path / "state")
    order = []

    def ensure(name, *args):
        order.append(name)
        stack.state["services"][name] = {"managed": False}

    with patch.object(launcher, "ROOT", root), \
         patch.object(stack, "prepare_docker", side_effect=lambda: order.append("docker")), \
         patch.object(stack, "ensure", side_effect=ensure), \
         patch.object(stack, "wait"), \
         patch.object(stack, "tunnel", side_effect=lambda: order.append("tunnel") or "https://sample.trycloudflare.com"), \
         patch.object(launcher, "allow_origin", side_effect=lambda value: order.append("configure-origin")), \
         patch.object(launcher, "login_origin_ready", return_value=True):
        stack.start()
    assert order == ["docker", "sandbox", "frontend", "tunnel", "configure-origin", "backend"]


def test_normal_stop_leaves_docker_containers_running(tmp_path):
    stack = launcher.Stack(tmp_path)
    with patch.object(stack, "run") as docker:
        stack.stop()
        docker.assert_not_called()


def test_identity_does_not_depend_on_shell_language_or_mutable_process_status():
    command = "cloudflared tunnel --url http://127.0.0.1:8080"
    calls = []
    status = ["Rs"]

    def ps(args, **kwargs):
        calls.append(kwargs["env"])
        if args[-1] == "lstart=":
            return subprocess.CompletedProcess(args, 0, "Mon Oct 26 16:38:57 2026\n", "")
        return subprocess.CompletedProcess(args, 0, status[0] + " " + command + "\n", "")

    with patch.object(launcher.subprocess, "run", side_effect=ps), \
         patch.object(launcher.os, "getpgid", return_value=1234), \
         patch.dict(os.environ, {"LANG": "zh_CN.UTF-8", "LC_ALL": "zh_CN.UTF-8"}):
        running = launcher.process_identity(1234)
        status[0] = "Ss"
        sleeping = launcher.process_identity(1234)
    assert running == sleeping == "Mon Oct 26 16:38:57 2026 " + command
    assert all(environment["LC_ALL"] == "C" for environment in calls)


def test_real_process_identity_survives_a_different_launch_locale(tmp_path):
    stack = launcher.Stack(tmp_path)
    with patch.dict(os.environ, {"LANG": "zh_CN.UTF-8", "LC_ALL": "zh_CN.UTF-8"}):
        record = stack.spawn("synthetic", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path)
    try:
        assert launcher.owned_alive(record)
        stack.stop_service("synthetic")
        assert not launcher.owned_alive(record)
    finally:
        process = stack.children["synthetic"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def test_resume_live_cpolar_without_creating_another_hostname(tmp_path):
    stack = launcher.Stack(tmp_path)
    stack.state["public_url"] = "https://existing.r9.cpolar.top"
    stack.state["services"]["tunnel"] = {"managed": True, "provider": "cpolar"}
    with patch.object(launcher, "owned_alive", side_effect=lambda record: bool(record)), \
         patch.object(launcher, "ready_http", return_value=True), \
         patch.object(stack, "spawn") as spawn:
        assert stack.tunnel() == "https://existing.r9.cpolar.top"
        spawn.assert_not_called()


def test_reuse_manual_cpolar_from_miniprogram_without_stopping_it(tmp_path):
    stack = launcher.Stack(tmp_path / "state")
    config = tmp_path / "miniprogram/miniprogram/config.local.ts"
    config.parent.mkdir(parents=True)
    config.write_text('export const developmentApiBase = "https://manual.r9.cpolar.top/mini-api"')
    with patch.object(launcher, "ROOT", tmp_path), \
         patch.object(launcher, "ready_http", return_value=True), \
         patch.object(stack, "spawn") as spawn, \
         patch.object(launcher.os, "killpg") as kill:
        assert stack.tunnel() == "https://manual.r9.cpolar.top"
        assert stack.state["services"]["tunnel"] == {"managed": False, "provider": "cpolar"}
        stack.stop_service("tunnel")
        spawn.assert_not_called()
        kill.assert_not_called()


def test_unreachable_manual_tunnel_is_not_reused_and_cloudflared_is_not_started(tmp_path):
    stack = launcher.Stack(tmp_path / "state")
    config = tmp_path / "miniprogram/miniprogram/config.local.ts"
    config.parent.mkdir(parents=True)
    config.write_text('export const developmentApiBase = "https://expired.r9.cpolar.top/mini-api"')
    with patch.object(launcher, "ROOT", tmp_path), \
         patch.object(launcher, "ready_http", return_value=False), \
         patch.object(launcher.shutil, "which", return_value=None) as which, \
         patch.object(stack, "spawn") as spawn:
        with pytest.raises(launcher.StartError, match="cpolar"):
            stack.tunnel()
        which.assert_called_once_with("cpolar")
        spawn.assert_not_called()


def test_origin_sync_only_updates_the_development_address(tmp_path):
    source = tmp_path / "miniprogram/miniprogram/config.local.ts"
    source.parent.mkdir(parents=True)
    release = source.with_name("config.ts")
    release.write_text("const releaseApiBase = ''")
    with patch.object(launcher, "ROOT", tmp_path):
        launcher.sync_miniprogram_origin("https://new.r9.cpolar.top/")
    assert 'https://new.r9.cpolar.top/mini-api' in source.read_text()
    assert release.read_text() == "const releaseApiBase = ''"


def test_cpolar_process_title_change_keeps_identity_but_pid_reuse_does_not():
    born = "Tue Oct 27 14:52:15 2026 "
    original = born + "cpolar http -daemon=off -processMode=single 127.0.0.1:8080"
    renamed = born + "cpolar: master process"
    assert launcher.normalized_identity(original) == launcher.normalized_identity(renamed)
    assert launcher.normalized_identity(original) != launcher.normalized_identity(
        "Tue Oct 27 14:52:16 2026 cpolar: master process")
    assert launcher.normalized_identity(born + "python app.py") == born + "python app.py"


def test_interrupted_launch_recovers_healthy_cpolar_from_its_own_log(tmp_path):
    stack = launcher.Stack(tmp_path / "state")
    log = tmp_path / "cpolar.log"
    log.write_text('Tunnel established at https://recovered.r9.cpolar.top\n')
    stack.state["services"]["tunnel"] = {"managed": True, "provider": "cpolar", "log": str(log)}
    with patch.object(launcher, "owned_alive", return_value=True), \
         patch.object(launcher, "ready_http", return_value=True), \
         patch.object(stack, "spawn") as spawn:
        assert stack.tunnel() == "https://recovered.r9.cpolar.top"
        spawn.assert_not_called()


def test_local_stop_remembers_manual_cpolar_for_next_public_start(tmp_path):
    stack = launcher.Stack(tmp_path)
    stack.state["services"]["tunnel"] = {"managed": False, "provider": "cpolar"}
    stack.state["public_url"] = "https://manual.r9.cpolar.top"
    stack.stop_service("tunnel")
    with patch.object(launcher, "ROOT", tmp_path), \
         patch.object(launcher, "ready_http", return_value=True), \
         patch.object(stack, "spawn") as spawn:
        assert stack.tunnel() == "https://manual.r9.cpolar.top"
        spawn.assert_not_called()
