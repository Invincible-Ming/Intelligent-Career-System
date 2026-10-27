"""Cloud retrieval contract, credential boundaries, and operation without a vector DB."""
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.config import Settings, settings
from app.core.models import Document
from app.core.schemas import SearchRequest
from app.services.ima_service import ImaService, ImaUnavailable, validate_media_url
from app.services.knowledge_service import KnowledgeService
from app.services.web_fetch import FetchBlocked

OWNER = uuid.uuid4()


@pytest.fixture
def ima_config(monkeypatch, tmp_path):
    client_id = tmp_path / "client_id"
    api_key = tmp_path / "api_key"
    client_id.write_text("synthetic-client\n")
    api_key.write_text("synthetic-key+/=\n")
    monkeypatch.setattr(settings, "KNOWLEDGE_PROVIDER", "ima")
    monkeypatch.setattr(settings, "IMA_CLIENT_ID_FILE", str(client_id))
    monkeypatch.setattr(settings, "IMA_API_KEY_FILE", str(api_key))
    monkeypatch.setattr(settings, "IMA_KNOWLEDGE_BASE_ID", "shared-career-library")


@pytest.mark.anyio
async def test_credentials_headers_and_library_scope(ima_config):
    calls = []

    def handle(request):
        import json

        body = json.loads(request.content)
        calls.append((request.url.path, body))
        assert request.headers["ima-openapi-apikey"] == "synthetic-key+/="
        assert request.headers["ima-openapi-clientid"] == "synthetic-client"
        assert request.method == "POST"
        assert "apikey" not in str(request.url)
        assert body["knowledge_base_id"] == "shared-career-library"
        return httpx.Response(200, json={"retcode": 0, "data": {"info_list": [{
            "media_id": "media-1", "title": "RAG 入门", "highlight_content": "<em>RAG</em> 检索增强生成",
        }]}})

    service = ImaService()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        service._client = client
        results = await service.search(query="RAG", owner_id=str(OWNER), top_k=5)
    assert len(calls) == 1
    assert results[0]["source"] == "ima"
    assert "RAG 检索增强生成" in results[0]["content"]
    assert "<em>" not in results[0]["content"]
    assert results[0]["chunk_metadata"]["score_kind"] == "rank"


@pytest.mark.anyio
async def test_titles_and_denied_content_are_not_evidence(ima_config):
    def handle(request):
        if request.url.path.endswith("search_knowledge"):
            return httpx.Response(200, json={"code": 0, "data": {"info_list": [
                {"media_id": "denied-note", "title": "不能据此编造内容", "highlight_content": ""},
            ]}})
        return httpx.Response(200, json={"code": 220030, "msg": "no permission", "data": {}})

    service = ImaService()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        service._client = client
        assert await service.search(query="RAG", owner_id=str(OWNER), top_k=5) == []


@pytest.mark.anyio
async def test_private_note_plaintext_is_read_after_scoped_search(ima_config):
    def handle(request):
        import json

        if request.url.path.endswith("search_knowledge"):
            data = {"info_list": [{"media_id": "m", "title": "基础知识"}]}
        elif request.url.path.endswith("get_media_info"):
            data = {"media_type": 11, "notebook_ext_info": {"notebook_id": "note-1"}}
        else:
            assert request.url.path == "/openapi/note/v1/get_doc_content"
            assert json.loads(request.content) == {"note_id": "note-1", "target_content_format": 0}
            data = {"content": "检索增强生成依据外部资料回答"}
        return httpx.Response(200, json={"code": 0, "data": data})

    service = ImaService()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        service._client = client
        results = await service.search(query="检索", owner_id=str(OWNER), top_k=5)
    assert "外部资料" in results[0]["content"]
    assert results[0]["chunk_metadata"]["evidence"] == "original_excerpt"


@pytest.mark.anyio
async def test_signed_download_does_not_receive_api_credentials_and_uses_sandbox(ima_config):
    def handle(request):
        if request.method == "GET":
            assert "ima-openapi-apikey" not in request.headers
            assert "ima-openapi-clientid" not in request.headers
            return httpx.Response(200, content=b"RAG document")
        if request.url.path.endswith("search_knowledge"):
            data = {"info_list": [{"media_id": "m", "title": "RAG.txt"}]}
        else:
            data = {"media_type": 13, "url_info": {"url": "https://docs.ima.myqcloud.com/file.txt"}}
        return httpx.Response(200, json={"code": 0, "data": data})

    service = ImaService()
    parse = AsyncMock(return_value=[SimpleNamespace(text="RAG document")])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        service._client = client
        with patch("app.services.ima_service.validate_media_url"), patch(
            "app.services.sandbox_client.parse_document", parse,
        ):
            results = await service.search(query="RAG", owner_id=str(OWNER), top_k=5)
    assert "RAG document" in results[0]["content"]
    assert parse.await_args.kwargs["owner_id"] == OWNER
    assert parse.await_args.kwargs["extension"] == ".txt"


@pytest.mark.anyio
@pytest.mark.parametrize("payload,status", [
    ({"code": 110030, "msg": "synthetic-key+/="}, 200),
    ({"code": 220004, "msg": "synthetic-key+/="}, 200),
    ({"retcode": 9, "errmsg": "synthetic-key+/="}, 200),
    ({"data": {}}, 200),
    ({}, 401),
])
async def test_fail_closed_without_echoing_upstream_data(ima_config, payload, status):
    service = ImaService()
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json=payload),
    )) as client:
        service._client = client
        with pytest.raises(ImaUnavailable) as exc:
            await service.library_info()
    assert "synthetic-key" not in str(exc.value)
    if payload.get("code") == 220004:
        assert "IMA_KNOWLEDGE_BASE_ID" in str(exc.value)


@pytest.mark.parametrize("url", [
    "http://docs.ima.myqcloud.com/a", "https://127.0.0.1/a",
    "https://169.254.169.254/a", "https://example.com/a",
    "https://ima.qq.com.evil.example/a", "https://docs.ima.myqcloud.com:8443/a",
    "https://user:password@ima.qq.com/a",
])
def test_media_url_restrictions(url):
    with pytest.raises(FetchBlocked):
        validate_media_url(url)


@pytest.mark.anyio
async def test_remote_empty_library_is_searchable_without_local_documents(ima_config):
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(all=lambda: []))
    remote = [{"document_id": "ima:1", "content": "RAG", "source": "ima"}]
    with patch("app.services.knowledge_service.ima_service.search", AsyncMock(return_value=remote)) as search:
        assert await KnowledgeService().search(session=session, owner_id=OWNER, query="RAG") == remote
    assert search.await_args.kwargs["owner_id"] == str(OWNER)
    stmt = session.execute.await_args.args[0]
    params = stmt.compile().params
    assert OWNER in params.values()
    assert "ready" in params.values()
    assert "documents.owner_id =" in str(stmt)


@pytest.mark.anyio
async def test_private_document_query_does_not_call_cloud(ima_config):
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(all=lambda: [
        (uuid.uuid4(), "resume", "简历.txt", [{"content": "Python 项目开发"}], []),
    ]))
    with patch("app.services.knowledge_service.ima_service.search", AsyncMock()) as search:
        results = await KnowledgeService().search(session=session, owner_id=OWNER, query="Python", document_type="resume")
    search.assert_not_awaited()
    assert results[0]["source"] == "local_keyword"
    assert "resume" in session.execute.await_args.args[0].compile().params.values()


@pytest.mark.anyio
async def test_ima_upload_and_delete_work_without_embedding_or_milvus(ima_config):
    from io import BytesIO

    from fastapi import UploadFile

    from app.services import document_service as module

    document = Document(id=uuid.uuid4(), owner_id=OWNER, filename="简历.txt", document_type="resume",
                        minio_object_key="object", status="ready")
    session = MagicMock()
    # 一次上传会依次执行三条 SQL：整文件去重（scalars().first）、文档数量上限
    # （scalar_one）、DocumentParse 的 upsert；删除时还要再按行加锁取回文档。
    # 固定长度的 side_effect 列表会随实现演进报 StopIteration，这里统一返回
    # 同一个结果对象：无重复、数量为 0、加锁查询命中同一份文档。
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalar_one.return_value = 0
    result.scalar_one_or_none.return_value = document
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.get = AsyncMock(return_value=document)
    session.delete = AsyncMock()
    session.rollback = AsyncMock()
    with patch.object(module.minio_service, "upload", AsyncMock(return_value="object")), \
         patch.object(module.minio_service, "delete", AsyncMock()), \
         patch.object(module, "parse_document", AsyncMock(return_value=[module.Block("Python 项目开发")])), \
         patch.object(module, "embed_chunks", AsyncMock(side_effect=AssertionError("no embedding"))), \
         patch.object(module.milvus_service, "insert", AsyncMock(side_effect=AssertionError("no vectors"))), \
         patch.object(module.milvus_service, "delete_document", AsyncMock(side_effect=AssertionError("no vectors"))):
        saved, duplicate = await module.document_service.upload_document(
            session=session, owner_id=OWNER,
            file=UploadFile(filename="简历.txt", file=BytesIO(b"Python")), document_type="resume",
        )
        assert not duplicate and saved.status == "ready" and saved.chunk_count > 0
        await module.document_service.delete_document(session=session, owner_id=OWNER, document_id=document.id)
    session.delete.assert_awaited_once_with(document)


@pytest.mark.anyio
async def test_ima_endpoint_bypasses_local_has_documents_guard(ima_config):
    from app.api import api

    with patch.object(api.knowledge_service, "search", AsyncMock(return_value=[
        {"document_id": "ima:m", "document_type": "knowledge", "content": "RAG", "score": 1, "source": "ima"},
    ])), patch.object(api, "has_ready_document", AsyncMock(side_effect=AssertionError("must not gate cloud"))):
        results = await api.search_knowledge_hybrid(MagicMock(), SimpleNamespace(id=OWNER), SearchRequest(query="RAG"))
    assert results[0].source == "ima"


def test_default_keeps_milvus_and_untrusted_api_host_is_rejected():
    assert Settings(_env_file=None).KNOWLEDGE_PROVIDER == "credits"
    with pytest.raises(ValueError):
        Settings(_env_file=None, IMA_BASE_URL="https://evil.example")


@pytest.mark.anyio
async def test_chat_tool_binds_owner_outside_model_arguments(ima_config):
    from app.agents import workflow

    tool = next(t for t in workflow.get_chat_agent_tools(owner_id=str(OWNER)) if t.name == "search_career_knowledge")
    assert set(tool.args) == {"query", "document_type"}
    with patch("app.core.database.AsyncSessionLocal"), patch(
        "app.services.knowledge_service.knowledge_service.search", AsyncMock(return_value=[]),
    ) as search:
        await tool.ainvoke({"query": "RAG"})
    assert search.await_args.kwargs["owner_id"] == OWNER


@pytest.mark.anyio
async def test_ima_lifecycle_never_initializes_vector_stack(ima_config, monkeypatch):
    from app import main

    for name in ("warm_up_redis_pool", "check_runner", "init_database", "init_workflow", "close_pool", "close_database"):
        monkeypatch.setattr(main, name, AsyncMock(return_value=True))
    for name in ("analysis_cache", "embed_cache", "experiment_store", "mcp_service", "bailian_service"):
        monkeypatch.setattr(main, name, SimpleNamespace(
            initialize=AsyncMock(), close=AsyncMock(), startup_errors=[], tools=[],
        ))
    vectors = SimpleNamespace(initialize=AsyncMock(side_effect=AssertionError("no vectors")),
                              list_chunks=AsyncMock(side_effect=AssertionError("no vectors")), close=AsyncMock())
    monkeypatch.setattr(main, "milvus_service", vectors)
    monkeypatch.setattr(main, "minio_service", SimpleNamespace(initialize=AsyncMock()))
    monkeypatch.setattr(main, "ima_service", SimpleNamespace(
        library_info=AsyncMock(return_value={"name": "Career"}), close=AsyncMock(),
    ))
    app = SimpleNamespace(state=SimpleNamespace())
    async with main.lifespan(app):
        assert not app.state.startup_errors
    vectors.initialize.assert_not_awaited()
    vectors.list_chunks.assert_not_awaited()
    main.embed_cache.initialize.assert_not_awaited()


@pytest.mark.anyio
async def test_local_ablation_cannot_run_with_ima(ima_config):
    from fastapi import HTTPException

    from app.evaluation.api import start_experiment

    with pytest.raises(HTTPException) as exc:
        await start_experiment(SimpleNamespace(id=OWNER), None, None)
    assert exc.value.status_code == 409


@pytest.mark.anyio
async def test_rollback_rebuild_preserves_chunk_layout_and_writes_a_new_collection(ima_config):
    from scripts import reindex_local_documents as script

    doc = SimpleNamespace(id=uuid.uuid4(), owner_id=OWNER, document_type="resume")
    stored = SimpleNamespace(chunks=[{"content": "原有分块一"}, {"content": "原有分块二"}])
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(all=lambda: [(doc, stored)]))
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = session
    target = SimpleNamespace(collection="", client=SimpleNamespace(has_collection=MagicMock(return_value=False)),
                             initialize=AsyncMock(), insert=AsyncMock(), close=AsyncMock())
    with patch.object(script, "AsyncSessionLocal", factory), patch.object(
        script, "MilvusService", return_value=target,
    ), patch.object(script, "close_database", AsyncMock()), patch.object(
        script.bailian_service, "embed", AsyncMock(return_value=[[1], [2]]),
    ), patch.object(script.bailian_service, "close", AsyncMock()):
        await script.rebuild(execute=False)
        target.initialize.assert_not_awaited()
        target.insert.assert_not_awaited()
        await script.rebuild(execute=True)
    assert target.collection.startswith(settings.MILVUS_COLLECTION + "_restore_")
    assert target.collection != settings.MILVUS_COLLECTION
    assert target.insert.await_args.kwargs["chunks"] == ["原有分块一", "原有分块二"]

