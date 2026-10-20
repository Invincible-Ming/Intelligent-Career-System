"""Approval recovery latency limits and correction; no cloud/tools or user data."""

import asyncio
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from app.agents import workflow
from app.core.config import settings
from app.core.schemas import JobAnalysis, MatchEvaluation, MatchScores, ResumeAnalysis, VerificationResult
from app.services.bailian import BailianService


def evaluation(education=80):
    return MatchEvaluation(
        scores=MatchScores(skill=80, experience=75, responsibility=70, education=education, bonus=10),
        matched_skills=["Python"],
        risks=["Synthetic year mismatch"] if education < 80 else [],
    )


def state():
    resume = {"skills": ["Python"]}
    jd = {"job_title": "Synthetic developer", "required_skills": ["Python"]}
    report = workflow.build_match_report(evaluation(), resume, jd)
    return dict(
        resume_analysis=resume,
        job_analysis=jd,
        match_report=report,
        result=report,
        human_reviewed=True,
        research_approved=True,
        research_completed=False,
        human_feedback="",
        verification_feedback="",
        retry_count=0,
    )


class ConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_direct_answer_setting_survives_json_repair(self):
        invalid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
        valid = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"is_faithful": True, "reason": "Synthetic facts checked"})
                    )
                )
            ]
        )
        create = AsyncMock(side_effect=[invalid, valid, valid])
        service = BailianService()
        service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(settings, "DASHSCOPE_API_KEY", "synthetic-test-key"):
            review = await service.structured_chat(
                system_prompt="Synthetic review",
                user_prompt="Synthetic report",
                response_model=VerificationResult,
                reasoning_effort="none",
            )
            await service.chat([{"role": "user", "content": "Synthetic general chat"}])
        self.assertTrue(review.is_faithful)
        for call in create.await_args_list[:2]:
            self.assertEqual(call.kwargs["reasoning_effort"], "none")
            self.assertEqual(call.kwargs["response_format"], {"type": "json_object"})
        self.assertNotIn("reasoning_effort", create.await_args_list[2].kwargs)

    async def test_no_new_evidence_reuses_report_without_model_call(self):
        original = state()
        with (
            patch.object(workflow.mcp_service, "get_tools", return_value=[]),
            patch.object(workflow.bailian_service, "structured_chat") as model,
        ):
            result = await workflow.match_agent(original, {"configurable": {}})
        model.assert_not_called()
        self.assertEqual(result["match_report"], original["match_report"])
        self.assertTrue(result["research_completed"])

    async def test_search_timeout_reuses_report_and_is_not_repeated(self):
        canceled = asyncio.Event()

        async def search(_, **kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                canceled.set()

        tool = SimpleNamespace(name="search_web", ainvoke=AsyncMock(side_effect=search))
        original = state()
        with (
            patch.object(settings, "MATCH_SEARCH_TIMEOUT", 0.02),
            patch.object(workflow.mcp_service, "get_tools", return_value=[tool]),
            patch.object(workflow.bailian_service, "structured_chat") as model,
        ):
            result = await asyncio.wait_for(workflow.match_agent(original, {"configurable": {}}), 0.5)
            await workflow.match_agent({**original, **result}, {"configurable": {}})
        model.assert_not_called()
        tool.ainvoke.assert_awaited_once()
        self.assertTrue(canceled.is_set())
        self.assertTrue(result["research_completed"])

    async def test_invalid_agent_message_is_not_evidence(self):
        tool = SimpleNamespace(
            name="search_web", ainvoke=AsyncMock(return_value="Sorry, need more steps to process this request.")
        )
        with (
            patch.object(workflow.mcp_service, "get_tools", return_value=[tool]),
            patch.object(workflow.bailian_service, "structured_chat") as model,
        ):
            result = await workflow.match_agent(state(), {"configurable": {}})
        model.assert_not_called()
        self.assertEqual(result["research_context"], "")

    async def test_feedback_regenerates_but_never_repeats_completed_search(self):
        original = {**state(), "research_completed": True, "human_feedback": "Check synthetic graduation year"}
        with (
            patch.object(workflow.mcp_service, "get_tools") as tools,
            patch.object(workflow.bailian_service, "structured_chat", return_value=evaluation(20)) as model,
        ):
            result = await workflow.match_agent(original, {"configurable": {}})
        tools.assert_not_called()
        model.assert_awaited_once()
        self.assertEqual(result["match_report"]["scores"]["education"], 20)

    async def test_review_rechecks_correction_and_recalculates_total(self):
        original = state()
        review = VerificationResult(
            is_faithful=False, reason="Synthetic graduation mismatch", corrected_evaluation=evaluation(20)
        )
        with patch.object(
            workflow.bailian_service,
            "structured_chat",
            side_effect=[review, VerificationResult(is_faithful=True, reason="Corrected facts checked")],
        ) as model:
            result = await workflow.verify_agent(original, {"configurable": {}})
        self.assertEqual(model.await_count, 2)
        self.assertEqual(result["retry_count"], 2)
        for call in model.await_args_list:
            self.assertEqual(call.kwargs["schema_attempts"], 1)
            self.assertEqual(call.kwargs["max_retries"], 0)
        self.assertTrue(result["verification_corrected"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["result"]["scores"]["education"], 20)
        self.assertEqual(result["result"]["total_score"], workflow.calculate_total_score(evaluation(20)))
        self.assertEqual(workflow.check_verification(result), workflow.END)

    async def test_uncorrectable_review_marks_risk_and_stops_generation_loop(self):
        review = VerificationResult(is_faithful=False, reason="Synthetic issue")
        with patch.object(workflow.bailian_service, "structured_chat", return_value=review):
            result = await workflow.verify_agent(state(), {"configurable": {}})
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["is_degraded"])
        self.assertIn("需人工复核", result["result"]["risks"][-1])
        self.assertEqual(workflow.check_verification(result), "human_review")

    async def test_timeouts_cancel_models_and_route_to_fallback_or_manual_review(self):
        async def slow(**_):
            await asyncio.sleep(10)

        original = {**state(), "research_completed": True, "human_feedback": "Synthetic correction"}
        with (
            patch.object(settings, "MATCH_REPORT_TIMEOUT", 0.02),
            patch.object(settings, "MATCH_VERIFY_TIMEOUT", 0.02),
            patch.object(workflow.bailian_service, "structured_chat", side_effect=slow),
            patch.object(workflow.logger, "error"),
        ):
            match = await asyncio.wait_for(workflow.match_agent(original, {"configurable": {}}), 0.5)
            verify = await asyncio.wait_for(workflow.verify_agent(original, {"configurable": {}}), 0.5)
        self.assertEqual(workflow.route_after_match(match), "fallback_node")
        self.assertEqual(workflow.check_verification(verify), "human_review")

    async def test_real_graph_bounds_corrections_then_manual_accepts_without_model(self):
        graph = workflow.get_match_workflow_builder().compile(
            checkpointer=InMemorySaver(), interrupt_before=["human_review"]
        )
        schemas = []

        async def model(**kwargs):
            schema = kwargs["response_model"]
            schemas.append(schema)
            if schema is ResumeAnalysis:
                return ResumeAnalysis(skills=["Python"])
            if schema is JobAnalysis:
                return JobAnalysis(job_title="Synthetic developer")
            if schema is MatchEvaluation:
                return evaluation()
            if schema is VerificationResult:
                return VerificationResult(
                    is_faithful=False, reason="Synthetic issue", corrected_evaluation=evaluation(20)
                )
            raise AssertionError(schema)

        with (
            patch.object(workflow, "match_workflow", graph),
            patch.object(settings, "ANALYSIS_CACHE_ENABLED", False),
            patch.object(workflow.mcp_service, "get_job_search_tools", return_value=[]) as tools,
            patch.object(workflow.bailian_service, "structured_chat", side_effect=model),
        ):
            initial = [
                event
                async for event in workflow.run_match_stream(
                    resume_text="Synthetic resume", jd_text="Synthetic JD", owner_id="synthetic-owner"
                )
            ]
            tools.assert_not_called()
            thread = json.loads(initial[-1].splitlines()[1].split(": ", 1)[1])["thread_id"]
            self.assertEqual(schemas.count(VerificationResult), 2)
            self.assertTrue(initial[-1].startswith("event: interrupt"))
            schemas.clear()
            resumed = [
                event async for event in workflow.resume_match_stream(thread_id=thread, owner_id="synthetic-owner")
            ]
        self.assertEqual(schemas, [])
        tools.assert_not_called()
        final = json.loads(resumed[-1].splitlines()[1].split(": ", 1)[1])
        self.assertEqual(final["report"]["scores"]["education"], 20)
        self.assertTrue(final["manual_accepted"])
        self.assertFalse(final["is_valid"])
        self.assertTrue(resumed[-1].startswith("event: complete"))

    async def test_authorized_search_precedes_report_and_auto_review_in_nonstream_graph(self):
        graph = workflow.get_match_workflow_builder().compile(
            checkpointer=InMemorySaver(), interrupt_before=["human_review"]
        )
        trace = []

        async def search(args, **kwargs):
            trace.append("search")
            self.assertNotIn("resume", args["query"].lower())
            return {
                "results": [
                    {"title": "Synthetic job", "url": "https://example.com/job", "snippet": "Synthetic role evidence"}
                ]
            }

        tool = SimpleNamespace(name="search_web", ainvoke=AsyncMock(side_effect=search))

        async def model(**kwargs):
            schema = kwargs["response_model"]
            if schema is ResumeAnalysis:
                return ResumeAnalysis(skills=["Python"])
            if schema is JobAnalysis:
                return JobAnalysis(job_title="Synthetic developer")
            if schema is MatchEvaluation:
                trace.append("report")
                return evaluation()
            if schema is VerificationResult:
                trace.append("review")
                return VerificationResult(is_faithful=True, reason="Synthetic facts checked")
            raise AssertionError(schema)

        with (
            patch.object(workflow, "match_workflow", graph),
            patch.object(settings, "ANALYSIS_CACHE_ENABLED", False),
            patch.object(workflow.mcp_service, "get_job_search_tools", return_value=[tool]) as tools,
            patch.object(workflow.bailian_service, "structured_chat", side_effect=model),
        ):
            initial = await workflow.run_match(
                resume_text="Synthetic resume",
                jd_text="Synthetic JD",
                owner_id=str(uuid.uuid4()),
                research_approved=True,
            )
            tools.assert_awaited_once()
            final = initial
        self.assertEqual(trace, ["search", "report", "review"])
        tool.ainvoke.assert_awaited_once()
        self.assertEqual(final["current_report"]["scores"]["education"], 80)
        self.assertEqual(final["status"], "completed")
        self.assertTrue(final["is_valid"])

    async def test_two_unverified_corrections_require_manual_review(self):
        with patch.object(
            workflow.bailian_service,
            "structured_chat",
            side_effect=[
                VerificationResult(is_faithful=False, reason="First issue", corrected_evaluation=evaluation(20)),
                VerificationResult(is_faithful=False, reason="Second issue", corrected_evaluation=evaluation(40)),
            ],
        ) as model:
            result = await workflow.verify_agent(state(), {"configurable": {}})
        self.assertEqual(model.await_count, 2)
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(result["result"]["scores"]["education"], 40)
        self.assertFalse(result["is_valid"])
        self.assertEqual(workflow.check_verification(result), "human_review")

    async def test_verification_rounds_share_one_timeout(self):
        canceled = asyncio.Event()
        calls = 0

        async def model(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return VerificationResult(is_faithful=False, reason="Issue", corrected_evaluation=evaluation(20))
            try:
                await asyncio.sleep(10)
            finally:
                canceled.set()

        with (
            patch.object(settings, "MATCH_VERIFY_TIMEOUT", 0.04),
            patch.object(workflow.asyncio, "timeout", wraps=asyncio.timeout) as deadline,
            patch.object(workflow.bailian_service, "structured_chat", side_effect=model),
        ):
            result = await asyncio.wait_for(workflow.verify_agent(state(), {"configurable": {}}), 0.5)
        self.assertEqual(calls, 2)
        deadline.assert_called_once_with(0.04)
        self.assertTrue(canceled.is_set())
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["verification_feedback"], "校验超时")
        self.assertEqual(workflow.check_verification(result), "human_review")

    async def test_feedback_regenerates_then_passes_or_pauses_again(self):
        for passes in (True, False):
            with self.subTest(passes=passes):
                graph = workflow.get_match_workflow_builder().compile(
                    checkpointer=InMemorySaver(), interrupt_before=["human_review"]
                )
                reviews = []
                reports = []

                async def model(*, _reports=reports, _reviews=reviews, _passes=passes, **kwargs):
                    schema = kwargs["response_model"]
                    if schema is ResumeAnalysis:
                        return ResumeAnalysis(skills=["Python"])
                    if schema is JobAnalysis:
                        return JobAnalysis(job_title="Synthetic developer")
                    if schema is MatchEvaluation:
                        _reports.append(kwargs["user_prompt"])
                        return evaluation(20 if len(_reports) > 1 else 80)
                    if schema is VerificationResult:
                        _reviews.append(kwargs["user_prompt"])
                        return VerificationResult(is_faithful=_passes and len(_reviews) > 1, reason="Synthetic review")
                    raise AssertionError(schema)

                owner = str(uuid.uuid4())
                with (
                    patch.object(workflow, "match_workflow", graph),
                    patch.object(settings, "ANALYSIS_CACHE_ENABLED", False),
                    patch.object(workflow.bailian_service, "structured_chat", side_effect=model),
                    patch.object(workflow.mcp_service, "get_job_search_tools") as tools,
                ):
                    initial = await workflow.run_match(
                        resume_text="Synthetic resume", jd_text="Synthetic JD", owner_id=owner
                    )
                    self.assertEqual(initial["status"], "paused")
                    result = await workflow.resume_match(
                        thread_id=initial["thread_id"], owner_id=owner, human_feedback="Recheck graduation year"
                    )
                self.assertEqual(len(reports), 2)
                self.assertIn("Recheck graduation year", reports[-1])
                self.assertEqual(len(reviews), 2)
                self.assertFalse(result["manual_accepted"])
                self.assertEqual(result["is_valid"], passes)
                self.assertEqual(result["status"], "completed" if passes else "paused")
                tools.assert_not_called()
