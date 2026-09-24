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
from app.services.bailian import BailianService
from app.core.schemas import ResumeAnalysis, JobAnalysis, MatchEvaluation, MatchScores, VerificationResult


def evaluation(education=80):
    return MatchEvaluation(
        scores=MatchScores(skill=80, experience=75, responsibility=70, education=education, bonus=10),
        matched_skills=['Python'], risks=['Synthetic year mismatch'] if education < 80 else [])


def state():
    resume = {'skills': ['Python']}
    jd = {'job_title': 'Synthetic developer', 'required_skills': ['Python']}
    report = workflow.build_match_report(evaluation(), resume, jd)
    return dict(resume_analysis=resume, job_analysis=jd, match_report=report, result=report,
                human_reviewed=True, research_completed=False, human_feedback='', verification_feedback='',
                retry_count=0)


class ConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_direct_answer_setting_survives_json_repair(self):
        invalid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
        valid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            'is_faithful': True, 'reason': 'Synthetic facts checked'})))])
        create = AsyncMock(side_effect=[invalid, valid, valid])
        service = BailianService()
        service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(settings, 'DASHSCOPE_API_KEY', 'synthetic-test-key'):
            review = await service.structured_chat(system_prompt='Synthetic review', user_prompt='Synthetic report',
                                                   response_model=VerificationResult, reasoning_effort='none')
            await service.chat([{'role': 'user', 'content': 'Synthetic general chat'}])
        self.assertTrue(review.is_faithful)
        for call in create.await_args_list[:2]:
            self.assertEqual(call.kwargs['reasoning_effort'], 'none')
            self.assertEqual(call.kwargs['response_format'], {'type': 'json_object'})
        self.assertNotIn('reasoning_effort', create.await_args_list[2].kwargs)

    async def test_no_new_evidence_reuses_report_without_model_call(self):
        original = state()
        with patch.object(workflow.mcp_service, 'get_tools', return_value=[]), \
                patch.object(workflow.bailian_service, 'structured_chat') as model:
            result = await workflow.match_agent(original, {'configurable': {}})
        model.assert_not_called()
        self.assertEqual(result['match_report'], original['match_report'])
        self.assertTrue(result['research_completed'])

    async def test_search_timeout_reuses_report_and_is_not_repeated(self):
        canceled = asyncio.Event()

        async def search(_, **kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                canceled.set()

        tool = SimpleNamespace(name='search_web', ainvoke=AsyncMock(side_effect=search))
        original = state()
        with patch.object(settings, 'MATCH_SEARCH_TIMEOUT', .02), \
                patch.object(workflow.mcp_service, 'get_tools', return_value=[tool]), \
                patch.object(workflow.bailian_service, 'structured_chat') as model:
            result = await asyncio.wait_for(workflow.match_agent(original, {'configurable': {}}), .5)
            await workflow.match_agent({**original, **result}, {'configurable': {}})
        model.assert_not_called();
        tool.ainvoke.assert_awaited_once()
        self.assertTrue(canceled.is_set())
        self.assertTrue(result['research_completed'])

    async def test_invalid_agent_message_is_not_evidence(self):
        tool = SimpleNamespace(name='search_web',
                               ainvoke=AsyncMock(return_value='Sorry, need more steps to process this request.'))
        with patch.object(workflow.mcp_service, 'get_tools', return_value=[tool]), \
                patch.object(workflow.bailian_service, 'structured_chat') as model:
            result = await workflow.match_agent(state(), {'configurable': {}})
        model.assert_not_called();
        self.assertEqual(result['research_context'], '')

    async def test_feedback_regenerates_but_never_repeats_completed_search(self):
        original = {**state(), 'research_completed': True, 'human_feedback': 'Check synthetic graduation year'}
        with patch.object(workflow.mcp_service, 'get_tools') as tools, \
                patch.object(workflow.bailian_service, 'structured_chat', return_value=evaluation(20)) as model:
            result = await workflow.match_agent(original, {'configurable': {}})
        tools.assert_not_called();
        model.assert_awaited_once()
        self.assertEqual(result['match_report']['scores']['education'], 20)

    async def test_review_applies_correction_and_recalculates_total_once(self):
        original = state()
        review = VerificationResult(is_faithful=False, reason='Synthetic graduation mismatch',
                                    corrected_evaluation=evaluation(20))
        with patch.object(workflow.bailian_service, 'structured_chat', return_value=review) as model:
            result = await workflow.verify_agent(original, {'configurable': {}})
        model.assert_awaited_once()
        self.assertTrue(result['verification_corrected']);
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['result']['scores']['education'], 20)
        self.assertEqual(result['result']['total_score'], workflow.calculate_total_score(evaluation(20)))
        self.assertEqual(workflow.check_verification(result), workflow.END)

    async def test_uncorrectable_review_marks_risk_and_stops_generation_loop(self):
        review = VerificationResult(is_faithful=False, reason='Synthetic issue')
        with patch.object(workflow.bailian_service, 'structured_chat', return_value=review):
            result = await workflow.verify_agent(state(), {'configurable': {}})
        self.assertFalse(result['is_valid']);
        self.assertTrue(result['is_degraded'])
        self.assertIn('需人工复核', result['result']['risks'][-1])
        self.assertEqual(workflow.check_verification(result), workflow.END)

    async def test_report_and_review_timeouts_cancel_model_and_route_to_fallback(self):
        async def slow(**_): await asyncio.sleep(10)

        original = {**state(), 'research_completed': True, 'human_feedback': 'Synthetic correction'}
        with patch.object(settings, 'MATCH_REPORT_TIMEOUT', .02), patch.object(settings, 'MATCH_VERIFY_TIMEOUT', .02), \
                patch.object(workflow.bailian_service, 'structured_chat', side_effect=slow), \
                patch.object(workflow.logger, 'error'):
            match = await asyncio.wait_for(workflow.match_agent(original, {'configurable': {}}), .5)
            verify = await asyncio.wait_for(workflow.verify_agent(original, {'configurable': {}}), .5)
        self.assertEqual(workflow.route_after_match(match), 'fallback_node')
        self.assertEqual(workflow.check_verification(verify), 'fallback_node')

    async def test_real_graph_confirmation_uses_one_review_call_with_correction(self):
        graph = workflow.get_match_workflow_builder().compile(checkpointer=InMemorySaver(),
                                                              interrupt_before=['human_review'])
        schemas = []

        async def model(**kwargs):
            schema = kwargs['response_model'];
            schemas.append(schema)
            if schema is ResumeAnalysis: return ResumeAnalysis(skills=['Python'])
            if schema is JobAnalysis: return JobAnalysis(job_title='Synthetic developer')
            if schema is MatchEvaluation: return evaluation()
            if schema is VerificationResult: return VerificationResult(is_faithful=False, reason='Synthetic issue',
                                                                       corrected_evaluation=evaluation(20))
            raise AssertionError(schema)

        with patch.object(workflow, 'match_workflow', graph), patch.object(settings, 'ANALYSIS_CACHE_ENABLED', False), \
                patch.object(workflow.mcp_service, 'get_job_search_tools', return_value=[]) as tools, \
                patch.object(workflow.bailian_service, 'structured_chat', side_effect=model):
            initial = [event async for event in
                       workflow.run_match_stream(resume_text='Synthetic resume', jd_text='Synthetic JD',
                                                 owner_id='synthetic-owner')]
            tools.assert_not_called()
            thread = json.loads(initial[-1].splitlines()[1].split(': ', 1)[1])['thread_id']
            schemas.clear()
            resumed = [event async for event in
                       workflow.resume_match_stream(thread_id=thread, owner_id=str(uuid.uuid4()))]
        self.assertEqual(schemas, [VerificationResult])
        tools.assert_awaited_once()
        final = json.loads(resumed[-1].splitlines()[1].split(': ', 1)[1])
        self.assertEqual(final['report']['scores']['education'], 20)
        self.assertFalse(final['is_degraded'])
        self.assertTrue(any('在本次校验中修正' in event for event in resumed))

    async def test_confirmed_search_precedes_update_and_review_in_nonstream_graph(self):
        graph = workflow.get_match_workflow_builder().compile(checkpointer=InMemorySaver(),
                                                              interrupt_before=['human_review'])
        trace = []

        async def search(args, **kwargs):
            trace.append('search')
            self.assertNotIn('resume', args['query'].lower())
            return {'results': [
                {'title': 'Synthetic job', 'url': 'https://example.com/job', 'snippet': 'Synthetic role evidence'}]}

        tool = SimpleNamespace(name='search_web', ainvoke=AsyncMock(side_effect=search))

        async def model(**kwargs):
            schema = kwargs['response_model']
            if schema is ResumeAnalysis: return ResumeAnalysis(skills=['Python'])
            if schema is JobAnalysis: return JobAnalysis(job_title='Synthetic developer')
            if schema is MatchEvaluation:
                trace.append('report')
                return evaluation()
            if schema is VerificationResult:
                trace.append('review')
                return VerificationResult(is_faithful=True, reason='Synthetic facts checked')
            raise AssertionError(schema)

        with patch.object(workflow, 'match_workflow', graph), patch.object(settings, 'ANALYSIS_CACHE_ENABLED', False), \
                patch.object(workflow.mcp_service, 'get_job_search_tools', return_value=[tool]) as tools, \
                patch.object(workflow.bailian_service, 'structured_chat', side_effect=model):
            initial = await workflow.run_match(resume_text='Synthetic resume', jd_text='Synthetic JD',
                                               owner_id='synthetic-owner')
            tools.assert_not_called()
            trace.clear()
            final = await workflow.resume_match(thread_id=initial['thread_id'], owner_id=str(uuid.uuid4()))
        self.assertEqual(trace, ['search', 'report', 'review'])
        tool.ainvoke.assert_awaited_once()
        self.assertEqual(final.scores.education, 80)
