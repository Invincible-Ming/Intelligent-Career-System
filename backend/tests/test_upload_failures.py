"""OCR and Runner failures remain actionable and never expose source payloads."""
import io
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from openai import AsyncOpenAI

from app.api import api
from app.core.config import settings
from app.core.limits import InvocationBudget, invocation_budget
from app.core.models import Document
from app.services import document_service, sandbox_client
from app.services.bailian import BailianService, ModelServiceError, OcrServiceError


@pytest.mark.anyio
async def test_mimo_ocr_uses_separate_credentials_and_keeps_request_budget(monkeypatch):
    monkeypatch.setattr(settings, "OCR_PROVIDER", "mimo")
    monkeypatch.setattr(settings, "MIMO_API_KEY", "synthetic-mimo-key")
    monkeypatch.setattr(settings, "MIMO_BASE_URL", "https://mimo.example/v1")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "synthetic-bailian-key")
    monkeypatch.setattr(settings, "BAILIAN_BASE_URL", "https://bailian.example/v1")
    calls = []

    def handle(request):
        calls.append((request.url.host, request.headers["authorization"], json.loads(request.content)))
        return httpx.Response(200, json={
            "id": "synthetic", "model": "synthetic", "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OCR TEST 12345"}}],
        })

    def create_client(**kwargs):
        if "http_client" not in kwargs:
            kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        else:
            # Replace only the transport; retain the production client's ownership/close path.
            kwargs["http_client"]._transport = httpx.MockTransport(handle)
        return AsyncOpenAI(**kwargs)

    service = BailianService()
    budget = InvocationBudget(time.monotonic() + 30, 2)
    token = invocation_budget.set(budget)
    try:
        with patch("app.services.bailian.AsyncOpenAI", side_effect=create_client):
            assert await service.ocr_image("synthetic-image") == "OCR TEST 12345"
            assert await service.chat([{"role": "user", "content": "hello"}]) == "OCR TEST 12345"
        assert budget.calls == 2
        assert calls[0][0:2] == ("mimo.example", "Bearer synthetic-mimo-key")
        params = calls[0][2]
        assert params["model"] == "mimo-v2.6-flash"
        assert params["thinking"] == {"type": "disabled"}
        assert params["max_completion_tokens"] == settings.MODEL_MAX_OUTPUT_TOKENS
        assert "max_tokens" not in params
        assert params["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,synthetic-image"
        assert calls[1][0:2] == ("bailian.example", "Bearer synthetic-bailian-key")
        ocr_client, chat_client = service._ocr_client, service._chat_client
    finally:
        invocation_budget.reset(token)
        await service.close()
    assert ocr_client.is_closed() and chat_client.is_closed()


@pytest.mark.anyio
async def test_mimo_ocr_missing_key_does_not_fall_back_to_bailian(monkeypatch):
    monkeypatch.setattr(settings, "OCR_PROVIDER", "mimo")
    monkeypatch.setattr(settings, "MIMO_API_KEY", "")
    service = BailianService()
    with patch.object(service, "_chat_retry", AsyncMock()) as request:
        with pytest.raises(OcrServiceError, match="MIMO_API_KEY"):
            await service.ocr_image("synthetic-image")
    request.assert_not_awaited()
    assert service._chat_client is None


@pytest.mark.anyio
@pytest.mark.parametrize("finish_reason,content,message", [
    ("length", "partial page", "输出上限"),
    ("stop", "", "内容为空"),
])
async def test_ocr_rejects_incomplete_page(finish_reason, content, message, monkeypatch):
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "synthetic-key")
    service = BailianService()
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                      message=SimpleNamespace(content=content))])
    try:
        with patch.object(service, "_chat_retry", AsyncMock(return_value=response)):
            with pytest.raises(OcrServiceError, match=message):
                await service.ocr_image("synthetic-image")
    finally:
        await service.close()


@pytest.mark.anyio
@pytest.mark.parametrize("code,status,message", [
    ("model_not_found", 404, "BAILIAN_OCR_MODEL"),
    ("insufficient_quota", 403, "额度不足"),
    ("AllocationQuota.FreeTierOnly", 403, "额度不足"),
    ("AccessDenied.Unpurchased", 403, "权限"),
    ("invalid_api_key", 401, "认证失败"),
    ("rate_limit", 429, "频率受限"),
])
async def test_ocr_errors_are_specific_and_safe(code, status, message):
    service = BailianService()
    with patch.object(settings, "DASHSCOPE_API_KEY", "synthetic-key"), patch.object(
        service, "_chat_retry", AsyncMock(side_effect=ModelServiceError(code, status)),
    ):
        with pytest.raises(OcrServiceError) as exc:
            await service.ocr_image("synthetic-private-image-base64")
    await service.close()
    assert message in str(exc.value)
    assert "synthetic-private-image-base64" not in str(exc.value)


@pytest.mark.anyio
async def test_upload_preserves_ocr_error_in_document_and_api_response():
    owner = uuid.uuid4()
    document = Document(id=uuid.uuid4(), owner_id=owner, filename="scan.pdf", document_type="knowledge",
                        minio_object_key="synthetic", status="processing")
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[
        MagicMock(scalars=lambda: MagicMock(first=lambda: None)),
        MagicMock(scalar_one=lambda: 0),
    ])
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.get = AsyncMock(return_value=document)
    reason = "文档需要 OCR 识别，但当前接口不支持配置的模型；请检查后端 BAILIAN_OCR_MODEL 配置。"
    with patch.object(settings, "KNOWLEDGE_PROVIDER", "ima"), patch.object(
        document_service.minio_service, "upload", AsyncMock(return_value="synthetic"),
    ), patch.object(document_service, "parse_document", AsyncMock(side_effect=OcrServiceError(reason))), patch.object(
        document_service, "record_upload_error",
    ):
        with pytest.raises(OcrServiceError):
            await document_service.document_service.upload_document(
                session=session, owner_id=owner, document_type="knowledge",
                file=UploadFile(filename="scan.pdf", file=io.BytesIO(b"synthetic PDF")),
            )
    assert document.status == "failed"
    assert document.error_message == reason
    with patch.object(api.document_service, "upload_document", AsyncMock(side_effect=OcrServiceError(reason))):
        with pytest.raises(HTTPException) as exc:
            await api.upload_document(user=SimpleNamespace(id=owner), session=session,
                                      file=UploadFile(filename="scan.pdf", file=io.BytesIO(b"test")),
                                      document_type="knowledge")
    assert exc.value.status_code == 503
    assert exc.value.detail == reason


@pytest.mark.anyio
@pytest.mark.parametrize("status,exception", [(422, sandbox_client.SandboxUnavailable), (413, ValueError)])
async def test_streaming_error_response_does_not_raise_response_not_read(status, exception):
    def handle(request):
        return httpx.Response(status, json={"detail": "synthetic-sensitive-input"})

    with patch("app.services.sandbox_client.Path.exists", return_value=True), patch.object(
        sandbox_client.httpx, "AsyncHTTPTransport", return_value=httpx.MockTransport(handle),
    ):
        with pytest.raises(exception) as exc:
            await sandbox_client.parse_document(owner_id=uuid.uuid4(), run_id=uuid.uuid4(), data=b"test",
                                                extension=".pdf", ocr=None, max_chars=100)
    assert "synthetic-sensitive-input" not in str(exc.value)

