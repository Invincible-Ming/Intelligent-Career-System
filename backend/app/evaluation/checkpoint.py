"""Local, explicitly resumed evaluation progress, bound to corpus and settings."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


class EvaluationCheckpoint:
    def __init__(self, path: Path, manifest: dict, *, resume: Path | None = None):
        self.path = path
        self.resumed = resume is not None
        self.data = {"version": 1, "signature": fingerprint(manifest), "manifest": manifest, "cases": {}}
        if resume is not None:
            self.data["cases"] = self.read_resume(resume, manifest)
        self._write()

    @staticmethod
    def read_resume(path: Path, manifest: dict):
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("评测进度文件过大")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(previous, dict) or previous.get("version") != 1
                or previous.get("signature") != fingerprint(manifest) or previous.get("manifest") != manifest
                or not isinstance(previous.get("cases"), dict)):
            raise ValueError("进度与当前用户、文档、题库、模型或评测配置不一致，不能复用")
        return previous["cases"]

    def case(self, config: dict, test_case: dict):
        key = fingerprint({"config": config, "test_case": test_case})
        return CaseCheckpoint(self, key)

    def _write(self):
        # No await between mutation and replacement: concurrent evaluation tasks
        # cannot interleave writes. An interrupted write leaves the last file intact.
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)


class CaseCheckpoint:
    def __init__(self, checkpoint: EvaluationCheckpoint, key: str):
        self.checkpoint, self.key = checkpoint, key

    def get(self, stage: str):
        return copy.deepcopy(self.checkpoint.data["cases"].get(self.key, {}).get(stage))

    def save(self, stage: str, value):
        self.checkpoint.data["cases"].setdefault(self.key, {})[stage] = copy.deepcopy(value)
        self.checkpoint._write()
