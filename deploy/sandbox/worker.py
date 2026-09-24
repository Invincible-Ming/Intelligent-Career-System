"""One document per container. JSON lines over stdio; no network or credentials."""
import asyncio
import base64
import json
import sys
from contextlib import redirect_stdout

from document_pipeline import parse_blocks, serialize_blocks

PROTOCOL_OUT = sys.stdout


def send(message):
    PROTOCOL_OUT.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    PROTOCOL_OUT.flush()


async def main():
    request = json.loads(sys.stdin.buffer.readline())
    extension = request["extension"]
    if extension not in (".pdf", ".docx", ".xlsx", ".txt"):
        raise ValueError("Unsupported extension")
    data = base64.b64decode(request["data"], validate=True)
    # 与 Runner 的 SANDBOX_MAX_DOCUMENT_MB 上限对齐（当前 50MB）。
    if not data or len(data) > 50 * 1024 * 1024:
        raise ValueError("Document size exceeded")
    max_chars = int(request["max_chars"])
    if not 0 < max_chars <= 200000:
        raise ValueError("Invalid text limit")

    async def ocr(image):
        send({"type": "ocr", "image": image})
        reply = json.loads(await asyncio.to_thread(sys.stdin.buffer.readline))
        if reply.get("type") != "ocr_result" or not isinstance(reply.get("text"), str):
            raise ValueError("OCR reply unavailable")
        return reply["text"]

    with redirect_stdout(sys.stderr):
        blocks = await parse_blocks(data, extension, ocr, max_chars=max_chars)
    send({"type": "result", "blocks": serialize_blocks(blocks)})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        # 仅本地可信链路使用；带出真实原因便于排障。
        send({"type": "error", "message": f"文档解析失败：{type(exc).__name__}: {exc}"})
