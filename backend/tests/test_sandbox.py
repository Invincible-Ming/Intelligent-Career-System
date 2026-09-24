"""Real Docker boundary with synthetic documents; no private data or cloud calls."""
import asyncio
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import fitz
import httpx
from PIL import Image

from app.core.config import settings
from app.services.sandbox_client import parse_document


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


class SandboxTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        try:
            docker("image", "inspect", "career-document-sandbox:1", "--format", "{{.Id}}")
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("Docker sandbox image not available")

    async def test_unavailable_runner_has_no_in_process_fallback(self):
        with patch.object(settings, "SANDBOX_RUNNER_SOCKET", "/tmp/nonexistent-career-runner.sock"):
            with self.assertRaises(RuntimeError):
                await parse_document(owner_id=uuid.uuid4(), run_id=uuid.uuid4(),
                                     data=b"Synthetic text", extension=".txt", ocr=None, max_chars=100)

    async def test_custom_socket_is_removed_on_termination(self):
        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="career-sandbox-test-") as directory:
            path = Path(directory) / "runner.sock"
            environment = {**os.environ, "SANDBOX_RUNNER_SOCKET": str(path)}
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(project_root / "deploy/sandbox/start.py"),
                cwd=project_root, env=environment,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                for _ in range(80):
                    try:
                        transport = httpx.AsyncHTTPTransport(uds=str(path))
                        async with httpx.AsyncClient(transport=transport, base_url="http://sandbox",
                                                     timeout=1) as client:
                            if (await client.get("/health")).status_code == 200:
                                break
                    except (OSError, httpx.HTTPError):
                        pass
                    await asyncio.sleep(.1)
                else:
                    self.fail("Runner did not start on the custom socket")
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            finally:
                process.terminate()
                await asyncio.wait_for(process.wait(), 10)
            self.assertFalse(path.exists())

    async def test_two_users_have_distinct_ephemeral_containers(self):
        pdf = fitz.open()
        page = pdf.new_page()
        image = Image.new("RGB", (100, 100), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        page.insert_image(fitz.Rect(10, 10, 110, 110), stream=output.getvalue())
        source = pdf.tobytes()
        pdf.close()

        owners = [uuid.uuid4(), uuid.uuid4()]
        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()

        async def parse(index):
            async def ocr(image):
                self.assertTrue(image)
                entered[index].set()
                await release.wait()
                return f"Synthetic OCR result for user {index} with Python skills"

            return await parse_document(owner_id=owners[index], run_id=uuid.uuid4(), data=source,
                                        extension=".pdf", ocr=ocr, max_chars=200000)

        tasks = [asyncio.create_task(parse(index)) for index in range(2)]
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 30)
            ids = docker("ps", "--filter", "label=career.sandbox=document", "--format", "{{.ID}}").splitlines()
            self.assertEqual(len(ids), 2)
            labels = set()
            for container in ids:
                import json
                info = json.loads(docker("inspect", container, "--format", "{{json .}}"))
                config, host = info["Config"], info["HostConfig"]
                labels.add(config["Labels"]["career.owner"])
                self.assertEqual(config["User"], "10001:10001")
                self.assertEqual(host["NetworkMode"], "none")
                self.assertTrue(host["ReadonlyRootfs"])
                self.assertEqual(host["CapDrop"], ["ALL"])
                self.assertEqual(info["Mounts"], [])
                self.assertFalse(any("SECRET" in item or "DASHSCOPE" in item for item in config["Env"]))
                self.assertFalse(any("docker.sock" in item for item in config["Env"]))
            self.assertEqual(labels, {hashlib.sha256(owner.bytes).hexdigest()[:24] for owner in owners})
            self.assertEqual(Path(settings.SANDBOX_RUNNER_SOCKET).stat().st_mode & 0o777, 0o600)
        finally:
            release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 30)
        self.assertIn("user 0", results[0][0].text)
        self.assertIn("user 1", results[1][0].text)
        await asyncio.sleep(.5)
        self.assertEqual(docker("ps", "--filter", "label=career.sandbox=document", "--format", "{{.ID}}"), "")

    async def test_cancel_during_ocr_removes_container(self):
        pdf = fitz.open()
        page = pdf.new_page()
        image = Image.new("RGB", (50, 50), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        page.insert_image(fitz.Rect(10, 10, 60, 60), stream=output.getvalue())
        data = pdf.tobytes()
        pdf.close()
        entered = asyncio.Event()

        async def ocr(_):
            entered.set()
            await asyncio.sleep(100)

        task = asyncio.create_task(parse_document(owner_id=uuid.uuid4(), run_id=uuid.uuid4(),
                                                  data=data, extension=".pdf", ocr=ocr, max_chars=200000))
        await asyncio.wait_for(entered.wait(), 20)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        for _ in range(30):
            if not docker("ps", "--filter", "label=career.sandbox=document", "--format", "{{.ID}}"):
                break
            await asyncio.sleep(.1)
        self.assertEqual(docker("ps", "--filter", "label=career.sandbox=document", "--format", "{{.ID}}"), "")
