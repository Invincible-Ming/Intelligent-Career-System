"""Actual LangGraph + Redis, synthetic documents and fake models."""
import asyncio
import json
import unittest
from collections import Counter
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app import workflow
from app.core.schemas import JobAnalysis, ResumeAnalysis, MatchEvaluation, MatchScores, VerificationResult
import test_analysis_cache as cache_tests


class MatchSpeedTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = cache_tests.AnalysisCacheTests.asyncSetUp
    asyncTearDown = cache_tests.AnalysisCacheTests.asyncTearDown
    async def test_real_graph_cache_hits_single_report_call_and_heartbeat(self):
        calls=Counter()
        async def model(**kwargs):
            schema=kwargs['response_model']
            calls[schema.__name__]+=1
            if schema is ResumeAnalysis:return ResumeAnalysis(skills=['Python'])
            if schema is JobAnalysis:return JobAnalysis(job_title='Synthetic developer',required_skills=['Python'])
            if schema is MatchEvaluation:
                await asyncio.sleep(.06)
                return MatchEvaluation(scores=MatchScores(skill=80,experience=70,responsibility=75,education=60,bonus=10),
                                       matched_skills=['Python'],suggestions=['Synthetic improvement'])
            if schema is VerificationResult:return VerificationResult(is_faithful=True,reason='Synthetic facts match')
            raise AssertionError('unexpected schema')

        graph=workflow.get_match_workflow_builder().compile(checkpointer=InMemorySaver(),interrupt_before=['human_review'])
        async def initial(owner,jd='Synthetic JD'):
            chunks=[chunk async for chunk in workflow.run_match_stream(
                resume_text='Synthetic resume',jd_text=jd,owner_id=owner)]
            events=[]
            for chunk in chunks:
                lines=chunk.splitlines()
                events.append((lines[0].split(': ',1)[1],json.loads(lines[1].split(': ',1)[1])))
            return events

        with patch.object(workflow,'match_workflow',graph),patch.object(workflow,'analysis_cache',self.cache), \
             patch.object(workflow,'MATCH_HEARTBEAT_SECONDS',.02), \
             patch.object(workflow.bailian_service,'structured_chat',side_effect=model), \
             patch.object(workflow.bailian_service,'stream_chat') as stream, \
             patch.object(workflow.mcp_service,'get_job_search_tools',return_value=[]) as tools:
            events=await initial(self.owners[0])
            self.assertEqual(calls,{'ResumeAnalysis':1,'JobAnalysis':1,'MatchEvaluation':1})
            self.assertEqual(events[-1][0],'interrupt')
            self.assertTrue(any(kind=='heartbeat' for kind,_ in events))
            self.assertTrue(any(kind=='progress' and data['node']=='match_agent' for kind,data in events))
            tools.assert_not_called()
            calls.clear()
            second=await initial(self.owners[0])
            self.assertEqual(calls,{'MatchEvaluation':1})
            hits=[data['cache_hit'] for kind,data in second if kind=='node_update' and data['node'] in ('resume_agent','jd_agent')]
            self.assertEqual(hits,[True,True])
            calls.clear()
            await initial(self.owners[0],jd='Changed synthetic JD')
            self.assertEqual(calls,{'JobAnalysis':1,'MatchEvaluation':1})
            calls.clear()
            await initial(self.owners[1])
            self.assertEqual(calls,{'ResumeAnalysis':1,'JobAnalysis':1,'MatchEvaluation':1})
            resumed=[chunk async for chunk in workflow.resume_match_stream(thread_id=events[-1][1]['thread_id'],owner_id=self.owners[0])]
            self.assertTrue(resumed[-1].startswith('event: complete'))
            self.assertEqual(calls['VerificationResult'],1)
            stream.assert_not_called()
