"""Evaluation routes only LLM requests to MiMo and retains shared safeguards."""
import asyncio
import json
import time
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.limits import BudgetExceeded, InvocationBudget, invocation_budget
from app.evaluation.executor import RAGExecutor
from app.evaluation.checkpoint import EvaluationCheckpoint
from app.evaluation.llm import EvaluationLLM
from app.services.bailian import BailianService, ModelServiceError, StructuredOutputError


@pytest.fixture
def mimo(monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_LLM_PROVIDER", "mimo")
    monkeypatch.setattr(settings, "EVALUATION_LLM_MODEL", "mimo-v2.6-flash")
    monkeypatch.setattr(settings, "MIMO_API_KEY", "synthetic-mimo-key")
    monkeypatch.setattr(settings, "MIMO_BASE_URL", "https://mimo.example/v1")
    # LLM evaluation must not need a valid DashScope key.
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "")


def mock_clients(monkeypatch, *, status=200, finish_reason="stop"):
    requests = []

    def handle(request):
        requests.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "synthetic", "code": "denied"}})
        return httpx.Response(200, json={
            "id": "synthetic", "model": "mimo-v2.6-flash", "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        })

    def create_client(**kwargs):
        http_client = kwargs.get("http_client")
        if http_client is None:
            kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        else:
            http_client._transport = httpx.MockTransport(handle)
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr("app.evaluation.llm.AsyncOpenAI", create_client)
    monkeypatch.setattr("app.services.bailian.AsyncOpenAI", create_client)
    return requests


@pytest.fixture
def adapter(monkeypatch):
    compat = Path(__file__).resolve().parents[1] / ".local" / "ragas_compat"
    if compat.is_dir():
        monkeypatch.syspath_prepend(str(compat))
    pytest.importorskip("ragas", exc_type=ImportError)
    return import_module("app.evaluation.ragas_adapter")


@pytest.mark.anyio
async def test_mimo_evaluation_credentials_parameters_and_budget(mimo, monkeypatch):
    requests = mock_clients(monkeypatch)
    service = EvaluationLLM()
    budget = InvocationBudget(time.monotonic() + 30, 1)
    token = invocation_budget.set(budget)
    try:
        assert await service.chat([{"role": "user", "content": "hello"}],
                                  max_output_tokens=128, max_retries=0) == "OK"
        assert budget.calls == 1
        request = requests[0]
        assert request.url.host == "mimo.example"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-mimo-key"
        body = json.loads(request.content)
        assert body["model"] == "mimo-v2.6-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_completion_tokens"] == 128
        assert "max_tokens" not in body and "enable_thinking" not in body
        with pytest.raises(BudgetExceeded):
            await service.chat([{"role": "user", "content": "again"}], max_retries=0)
        assert len(requests) == 1
        client = service.chat_client
    finally:
        invocation_budget.reset(token)
        await service.close()
    assert client.is_closed()


@pytest.mark.anyio
async def test_missing_mimo_key_does_not_fall_back(mimo, monkeypatch):
    monkeypatch.setattr(settings, "MIMO_API_KEY", "")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "synthetic-bailian-key")
    requests = mock_clients(monkeypatch)
    with pytest.raises(RuntimeError, match="MIMO_API_KEY"):
        await EvaluationLLM().chat([{"role": "user", "content": "hello"}])
    assert requests == []


@pytest.mark.anyio
async def test_mimo_auth_error_is_not_retried_or_sent_to_bailian(mimo, monkeypatch):
    requests = mock_clients(monkeypatch, status=401)
    service = EvaluationLLM()
    try:
        with pytest.raises(ModelServiceError) as error:
            await service.chat([{"role": "user", "content": "hello"}])
        assert error.value.status_code == 401
        assert len(requests) == 1
        assert requests[0].url.host == "mimo.example"
    finally:
        await service.close()


@pytest.mark.anyio
async def test_mimo_truncated_output_is_rejected(mimo, monkeypatch):
    mock_clients(monkeypatch, finish_reason="length")
    service = EvaluationLLM()
    try:
        with pytest.raises(StructuredOutputError, match="output_truncated"):
            await service.chat([{"role": "user", "content": "hello"}], max_retries=0)
    finally:
        await service.close()


@pytest.mark.anyio
async def test_bailian_evaluation_keeps_existing_protocol(monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_LLM_PROVIDER", "bailian")
    monkeypatch.setattr(settings, "EVALUATION_LLM_MODEL", "")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "synthetic-bailian-key")
    monkeypatch.setattr(settings, "BAILIAN_BASE_URL", "https://bailian.example/v1")
    requests = mock_clients(monkeypatch)
    service = EvaluationLLM()
    try:
        await service.chat([{"role": "user", "content": "hello"}], max_output_tokens=128, max_retries=0)
        assert requests[0].url.host == "bailian.example"
        body = json.loads(requests[0].content)
        assert body["model"] == settings.BAILIAN_CHAT_MODEL
        assert body["max_tokens"] == 128
        assert "thinking" not in body and "max_completion_tokens" not in body
    finally:
        await service.close()


@pytest.mark.anyio
async def test_executor_uses_evaluation_llm_and_keeps_existing_embeddings():
    executor = RAGExecutor()
    with patch("app.evaluation.executor.evaluation_llm.chat", AsyncMock(return_value="answer")) as model, \
            patch("app.evaluation.executor.bailian_service.chat", AsyncMock()) as business_model:
        assert await executor._generate_answer(query="question", contexts=["evidence"]) == "answer"
    model.assert_awaited_once()
    business_model.assert_not_awaited()

    with patch("app.evaluation.executor.bailian_service.embed_query", AsyncMock(return_value=[0.1])) as embed, \
            patch("app.evaluation.executor.milvus_service.search", AsyncMock(return_value=[])):
        await executor._dense_search(query="question", owner_id="synthetic", allowed_document_ids=[],
                                     document_type=None, top_k=5)
    embed.assert_awaited_once_with("question")


@pytest.mark.anyio
async def test_all_four_ragas_metrics_use_mimo_but_embeddings_stay_on_bailian(mimo, adapter):
    # Synthetic judge responses verify routing/JSON parsing, not evaluation quality.
    statement = "LangChain builds LLM applications."
    question = "What does LangChain do?"
    responses = iter([
        {"reason": "supported", "verdict": 1},
        {"classifications": [{"statement": statement, "reason": "supported", "attributed": 1}]},
        {"statements": [statement]},
        {"statements": [{"statement": statement, "reason": "supported", "verdict": 1}]},
        *[{"question": question, "noncommittal": 0} for _ in range(3)],
    ])
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "synthetic", "model": "mimo-v2.6-flash", "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(next(responses))}}],
        })

    service = EvaluationLLM()
    service._chat_client = AsyncOpenAI(api_key="synthetic-mimo-key", base_url="https://mimo.example/v1",
                                     max_retries=0,
                                     http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    try:
        with patch.object(adapter, "evaluation_llm", service), \
                patch.object(adapter.bailian_service, "embed", AsyncMock(return_value=[[1.0, 0.0]])) as embed, \
                patch.object(adapter.bailian_service, "chat", AsyncMock()) as business_model:
            scores = await adapter.evaluate_sample(query=question, answer=statement,
                                                   contexts=[statement], reference=statement)
        assert set(scores) == {"context_precision", "context_recall", "faithfulness", "answer_relevancy"}
        assert all(value == pytest.approx(1.0) for value in scores.values())
        assert len(requests) == 7  # One context, recall, two faithfulness calls, three relevancy calls.
        assert all(request["model"] == "mimo-v2.6-flash" for request in requests)
        assert all(request["max_completion_tokens"] == settings.EVALUATION_METRIC_MAX_OUTPUT_TOKENS
                   for request in requests)
        embed.assert_awaited_once_with([question])
        business_model.assert_not_awaited()
    finally:
        await service.close()


@pytest.mark.anyio
async def test_ragas_call_limit_still_applies_after_switching(mimo, adapter, monkeypatch):
    from langchain_core.prompt_values import StringPromptValue

    monkeypatch.setattr(settings, "EVALUATION_METRIC_MAX_CALLS", 2)
    model = adapter.BailianRagasLLM()
    with patch.object(adapter.evaluation_llm, "chat", AsyncMock(return_value="synthetic")) as call:
        with pytest.raises(adapter.MetricCallLimit):
            await model.agenerate_text(StringPromptValue(text="synthetic"), n=3)
    assert call.await_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("broken_requests,call_limit,retries,expected_error,expected_calls", [
    (1, 16, 1, None, 2),
    (2, 16, 1, ModelServiceError, 2),
    (1, 1, 1, "limit", 1),
    (1, 16, 0, ModelServiceError, 1),
])
async def test_ragas_incomplete_http_body_recovery_is_bounded(
        mimo, adapter, monkeypatch, broken_requests, call_limit, retries, expected_error, expected_calls):
    from langchain_core.prompt_values import StringPromptValue

    monkeypatch.setattr(settings, "EVALUATION_METRIC_MAX_CALLS", call_limit)
    monkeypatch.setattr(settings, "EVALUATION_METRIC_MAX_RETRIES", retries)
    requests = []

    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"id":"incomplete"'
            raise httpx.RemoteProtocolError("incomplete chunked read")

    def handle(request):
        requests.append(request)
        if len(requests) <= broken_requests:
            return httpx.Response(200, stream=BrokenBody())
        return httpx.Response(200, json={
            "id": "synthetic", "model": "mimo-v2.6-flash", "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OK"}}],
        })

    service = EvaluationLLM()
    service._chat_client = AsyncOpenAI(api_key="synthetic", base_url="https://mimo.example/v1",
                                     max_retries=0,
                                     http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    model = adapter.BailianRagasLLM()
    try:
        with patch.object(adapter, "evaluation_llm", service), patch.object(adapter.asyncio, "sleep", AsyncMock()):
            error_type = adapter.MetricCallLimit if expected_error == "limit" else expected_error
            if error_type:
                with pytest.raises(error_type):
                    await model.agenerate_text(StringPromptValue(text="synthetic"))
            else:
                response = await model.agenerate_text(StringPromptValue(text="synthetic"))
                assert response.generations[0][0].text == "OK"
        assert len(requests) == expected_calls
        assert model.calls == expected_calls
    finally:
        await service.close()


@pytest.mark.anyio
@pytest.mark.parametrize("code,status,retry", [
    (None, 401, False), (None, 403, False), (None, 400, False),
    ("insufficient_quota", 429, False), ("AllocationQuota.FreeTierOnly", 403, False),
    (None, 429, True), (None, 500, True), (None, 503, True), (None, None, False),
])
async def test_ragas_retries_only_transient_errors(mimo, adapter, monkeypatch, code, status, retry):
    from langchain_core.prompt_values import StringPromptValue

    monkeypatch.setattr(settings, "EVALUATION_METRIC_MAX_RETRIES", 1)
    model = adapter.BailianRagasLLM()
    error = ModelServiceError(code, status)
    with patch.object(adapter.evaluation_llm, "chat", AsyncMock(side_effect=error)) as request, \
            patch.object(adapter.asyncio, "sleep", AsyncMock()):
        with pytest.raises(ModelServiceError):
            await model.agenerate_text(StringPromptValue(text="synthetic"))
    assert request.await_count == (2 if retry else 1)
    assert model.calls == request.await_count


@pytest.mark.anyio
async def test_ragas_retry_backoff_respects_cancellation(mimo, adapter):
    from langchain_core.prompt_values import StringPromptValue

    model = adapter.BailianRagasLLM()
    with patch.object(adapter.evaluation_llm, "chat", AsyncMock(side_effect=ModelServiceError(status_code=503))) as call:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(model.agenerate_text(StringPromptValue(text="synthetic")), timeout=0.01)
    assert call.await_count == model.calls == 1


@pytest.mark.anyio
async def test_evaluation_request_has_real_wall_clock_timeout(mimo, monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_LLM_REQUEST_TIMEOUT", 0.02)
    cancelled = []

    async def slow_response(*args, **kwargs):
        assert kwargs["max_retries"] == 0
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    started = time.monotonic()
    with patch.object(BailianService, "chat", slow_response):
        with pytest.raises(ModelServiceError) as error:
            await EvaluationLLM().chat([{"role": "user", "content": "synthetic"}], max_retries=0)
    assert error.value.code == "evaluation_request_timeout"
    assert error.value.status_code == 408
    assert time.monotonic() - started < 1
    assert cancelled == [True]


def test_evaluation_checkpoint_resume_preserves_original_and_is_case_scoped(tmp_path):
    manifest = {"owner": "owner-a", "document": "doc-a", "model": "mimo-v2.6-flash", "corpus": "hash"}
    path = tmp_path / "progress.json"
    checkpoint = EvaluationCheckpoint(path, manifest)
    case = checkpoint.case({"name": "hybrid"}, {"query": "question-a"})
    case.save("contexts", ["evidence"])
    case.save("metrics", {"context_precision": 0.0})
    original = path.read_bytes()
    resumed = EvaluationCheckpoint(tmp_path / "next.json", manifest, resume=path)
    assert resumed.case({"name": "hybrid"}, {"query": "question-a"}).get("metrics") == {"context_precision": 0.0}
    assert resumed.case({"name": "dense"}, {"query": "question-a"}).get("contexts") is None
    assert resumed.case({"name": "hybrid"}, {"query": "question-b"}).get("contexts") is None
    resumed.case({"name": "hybrid"}, {"query": "question-a"}).save("answer", "new")
    assert path.read_bytes() == original
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("field", ["owner", "document", "model", "corpus", "prompts", "configs"])
def test_evaluation_checkpoint_rejects_incompatible_run(tmp_path, field):
    manifest = {key: "original" for key in ("owner", "document", "model", "corpus", "prompts", "configs")}
    path = tmp_path / "progress.json"
    EvaluationCheckpoint(path, manifest)
    with pytest.raises(ValueError, match="不一致"):
        EvaluationCheckpoint(tmp_path / "next.json", {**manifest, field: "changed"}, resume=path)
    assert not (tmp_path / "next.json").exists()


@pytest.mark.anyio
async def test_executor_reuses_checkpoint_contexts_without_new_retrieval(tmp_path):
    from app.evaluation.config import EvaluationConfig

    checkpoint = EvaluationCheckpoint(tmp_path / "progress.json", {})
    progress = checkpoint.case({}, {})
    progress.save("contexts", ["evidence"])
    executor = RAGExecutor()
    with patch.object(executor, "_retrieve", AsyncMock()) as retrieve, \
            patch.object(executor, "_generate_answer", AsyncMock(return_value="answer")) as answer:
        result = await executor.execute(query="question", config=EvaluationConfig(name="hybrid"),
                                        owner_id="76cd0f0c-eb00-4eb0-b62e-9730fee1150b",
                                        allowed_document_ids=["1bbabb2c-7535-495a-ae8e-319f1ccd7607"],
                                        progress=progress)
    assert result["contexts"] == ["evidence"]
    retrieve.assert_not_awaited()
    answer.assert_awaited_once()


@pytest.mark.anyio
async def test_runner_preserves_answer_on_metric_timeout_and_reuses_it(tmp_path):
    from app.evaluation.config import EvaluationConfig, ExperimentConfig
    from app.evaluation.dataset import TestCase
    from app.evaluation.runner import EvaluationRunner
    from app.evaluation.scope import EvaluationScope

    owner, document = "76cd0f0c-eb00-4eb0-b62e-9730fee1150b", "1bbabb2c-7535-495a-ae8e-319f1ccd7607"
    config = EvaluationConfig(name="hybrid")
    experiment = ExperimentConfig(experiment_name="test", baseline=config, test_dataset_path="unused")
    test_case = TestCase(query="question", ground_truth="answer")
    checkpoint = EvaluationCheckpoint(tmp_path / "progress.json", {"owner": owner, "document": document})
    runner = EvaluationRunner(experiment, owner_id=owner, scope=EvaluationScope(owner, (document,)), checkpoint=checkpoint)
    generated = {"answer": "answer", "contexts": ["evidence"], "retrieved_count": 1, "reflection_iterations": 0}

    async def fail_after_one_metric(**kwargs):
        kwargs["progress"].save("metrics", {"context_precision": 0.5})
        raise TimeoutError()

    with patch("app.evaluation.runner.rag_executor.execute", AsyncMock(return_value=generated)), \
            patch("app.evaluation.runner.ragas_metrics.evaluate", fail_after_one_metric):
        result = await runner._evaluate_test_case(test_case=test_case, config=config, semaphore=asyncio.Semaphore(1))
    assert result["status"] == "timeout"
    assert result["completed_metrics"] == {"context_precision": 0.5}
    restored = EvaluationCheckpoint(tmp_path / "next.json", checkpoint.data["manifest"], resume=checkpoint.path)
    runner.checkpoint = restored
    with patch("app.evaluation.runner.rag_executor.execute", AsyncMock()) as generate, \
            patch("app.evaluation.runner.ragas_metrics.evaluate", AsyncMock(return_value={"overall_score": 0.5})):
        result = await runner._evaluate_test_case(test_case=test_case, config=config, semaphore=asyncio.Semaphore(1))
    assert result["status"] == "success"
    assert result["answer"] == "answer"
    generate.assert_not_awaited()


@pytest.mark.anyio
async def test_ragas_resume_reuses_completed_metrics_after_later_failure(mimo, adapter, tmp_path):
    from types import SimpleNamespace

    progress = EvaluationCheckpoint(tmp_path / "progress.json", {}).case({}, {})
    calls = [AsyncMock(return_value=0.5), AsyncMock(return_value=0.75),
             AsyncMock(side_effect=ModelServiceError(status_code=503)), AsyncMock(return_value=0.8)]
    classes = ["LLMContextPrecisionWithReference", "LLMContextRecall", "Faithfulness", "AsyncResponseRelevancy"]
    from contextlib import ExitStack

    with ExitStack() as stack:
        for name, call in zip(classes, calls, strict=True):
            stack.enter_context(patch.object(adapter, name, return_value=SimpleNamespace(single_turn_ascore=call)))
        with pytest.raises(ModelServiceError):
            await adapter.evaluate_sample(query="question", answer="answer", contexts=["evidence"],
                                          reference="answer", progress=progress)
        assert progress.get("metrics") == {"context_precision": 0.5, "context_recall": 0.75}
        calls[2].side_effect = None
        calls[2].return_value = 0.9
        scores = await adapter.evaluate_sample(query="question", answer="answer", contexts=["evidence"],
                                              reference="answer", progress=progress)
    assert scores == {"context_precision": 0.5, "context_recall": 0.75, "faithfulness": 0.9, "answer_relevancy": 0.8}
    assert [call.await_count for call in calls] == [1, 1, 2, 1]
