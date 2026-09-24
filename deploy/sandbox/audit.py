"""Audit helper for the sandbox Runner. JSON lines; never records resume text, keys, or images."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_AUDIT_PATH = os.environ.get(
    "SANDBOX_AUDIT_LOG",
    str(Path(__file__).resolve().parents[2] / "deploy" / ".local" / "audit.log"),
)
# Ensure the audit log directory exists.
Path(_AUDIT_PATH).parent.mkdir(parents=True, exist_ok=True)

_start_times: dict[str, float] = {}


def _write(event: str, fields: dict) -> None:
    fields = {k: v for k, v in fields.items() if v is not None and v != ""}
    fields["event"] = event
    fields["ts"] = time.time()
    try:
        with open(_AUDIT_PATH, "a") as f:
            f.write(json.dumps(fields, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        pass  # Audit must not crash the Runner.


def parse_start(session_id: str, run_id: str, owner_hash: str, extension: str) -> None:
    _start_times[session_id] = time.time()
    _write("parse_start", {"session_id": session_id, "run_id": run_id,
                           "owner_hash": owner_hash, "extension": extension})


def parse_result(session_id: str, run_id: str, owner_hash: str, extension: str,
                 exit_code: int, exit_reason: str, duration_ms: int) -> None:
    _write("parse_result", {"session_id": session_id, "run_id": run_id,
                            "owner_hash": owner_hash, "extension": extension,
                            "exit_code": exit_code, "exit_reason": exit_reason,
                            "duration_ms": duration_ms})
    _start_times.pop(session_id, None)


def parse_error(session_id: str, run_id: str, owner_hash: str, extension: str,
                error_type: str, duration_ms: int) -> None:
    _write("parse_error", {"session_id": session_id, "run_id": run_id,
                           "owner_hash": owner_hash, "extension": extension,
                           "error_type": error_type, "duration_ms": duration_ms})
    _start_times.pop(session_id, None)


def parse_cancel(session_id: str, run_id: str, owner_hash: str, reason: str) -> None:
    _write("parse_cancel", {"session_id": session_id, "run_id": run_id,
                            "owner_hash": owner_hash, "reason": reason})
    _start_times.pop(session_id, None)


def ocr_request(session_id: str, image_hash: str) -> None:
    _write("ocr_request", {"session_id": session_id, "image_sha256": image_hash})


def mcp_start(session_id: str, mode: str, owner_hash: str | None, run_id: str | None) -> None:
    _start_times[session_id] = time.time()
    _write("mcp_start", {"session_id": session_id, "mode": mode,
                         "owner_hash": owner_hash, "run_id": run_id})


def mcp_result(session_id: str, mode: str, exit_code: int, exit_reason: str, duration_ms: int) -> None:
    _write("mcp_result", {"session_id": session_id, "mode": mode,
                          "exit_code": exit_code, "exit_reason": exit_reason,
                          "duration_ms": duration_ms})
    _start_times.pop(session_id, None)


def mcp_cancel(session_id: str, mode: str, reason: str) -> None:
    _write("mcp_cancel", {"session_id": session_id, "mode": mode, "reason": reason})
    _start_times.pop(session_id, None)
