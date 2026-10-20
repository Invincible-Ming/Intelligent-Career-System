"""MCP approval gate tests; no cloud model or live MCP calls required."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.agents import workflow
from app.core.schemas import MatchEvaluation, MatchScores


class WorkflowApprovalTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _state(*, reviewed: bool, research_approved: bool = False) -> dict:
        return {
            "resume_analysis": {"skills": ["Python"]},
            "job_analysis": {
                "job_title": "后端工程师",
                "required_skills": ["Python"],
            },
            "human_reviewed": reviewed,
            "research_approved": research_approved,
            "human_feedback": "",
            "research_context": "",
            "research_completed": False,
        }

    async def _run_match(self, state: dict):
        evaluation = MatchEvaluation(
            scores=MatchScores(
                skill=80,
                experience=70,
                responsibility=75,
                education=60,
                bonus=10,
            ),
            matched_skills=["Python"],
        )
        with (
            patch.object(workflow.bailian_service, "stream_chat") as stream_chat,
            patch.object(workflow.bailian_service, "structured_chat", AsyncMock(return_value=evaluation)) as structured,
        ):
            result = await workflow.match_agent(state, {"configurable": {}})
            stream_chat.assert_not_called()
            structured.assert_awaited_once()
            self.assertIn("简历分析", structured.call_args.kwargs["user_prompt"])
            self.assertIn("岗位分析", structured.call_args.kwargs["user_prompt"])
            return result

    async def test_initial_report_never_loads_mcp_tools(self):
        with patch.object(workflow.mcp_service, "get_tools") as get_tools:
            result = await self._run_match(self._state(reviewed=False))

        get_tools.assert_not_called()
        self.assertFalse(result["research_completed"])
        self.assertEqual(result["research_context"], "")

    async def test_mcp_runs_only_after_explicit_authorization_and_is_persisted(self):
        search_tool = SimpleNamespace(
            name="search_web",
            ainvoke=AsyncMock(
                return_value=[
                    {
                        "type": "text",
                        "text": '{"results":[{"title":"公开岗位情报","url":"https://example.com/job","snippet":"Python"}]}',
                    }
                ]
            ),
        )
        with patch.object(workflow.mcp_service, "get_tools", return_value=[search_tool]) as get_tools:
            result = await self._run_match(self._state(reviewed=False, research_approved=True))

        get_tools.assert_called_once_with("search")
        self.assertTrue(result["research_completed"])
        self.assertIn("公开岗位情报", result["research_context"])
        search_tool.ainvoke.assert_awaited_once()
        self.assertEqual(search_tool.ainvoke.call_args.args[0], {"query": "后端工程师 岗位 技能要求 面试", "count": 3})
        self.assertIsInstance(
            search_tool.ainvoke.call_args.kwargs["config"]["callbacks"][0], workflow.ModelBudgetCallback
        )

    async def test_human_review_does_not_authorize_research(self):
        with patch.object(workflow.mcp_service, "get_tools") as tools:
            result = await self._run_match(self._state(reviewed=True))
        tools.assert_not_called()
        self.assertFalse(result["research_completed"])

    def test_manual_acceptance_ends_without_regeneration(self):
        self.assertEqual(workflow.route_after_human_review({"manual_accepted": True}), workflow.END)

    def test_feedback_routes_to_match_node(self):
        self.assertEqual(
            workflow.route_after_human_review({"human_reviewed": True}),
            "match_agent",
        )


if __name__ == "__main__":
    unittest.main()
