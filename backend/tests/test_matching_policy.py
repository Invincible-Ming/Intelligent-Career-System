"""Pure business policy tests; no cloud models, storage or user documents."""

import unittest

from app.core.matching_policy import apply_match_policies, can_continue_unverified, unresolved_report_issues


class MatchingPolicyTests(unittest.TestCase):
    def test_unknown_availability_moves_to_hr_advice(self):
        risk = "实习时长需确认：简历显示2026.08至今，需核实是否满足岗位要求的连续实习三个月及以上。"
        original = {"risks": [risk], "suggestions": []}
        report, corrections = apply_match_policies(
            original, resume_text="AI Agent 实习生\n2026.08 - 至今", jd_text="能连续实习三个月及以上者优先。",
        )
        self.assertEqual(report["risks"], [])
        self.assertEqual(report["suggestions"], ["面试时可与 HR 沟通到岗时间和实习安排。"])
        self.assertTrue(corrections)
        self.assertEqual(original["risks"], [risk])
        again, notes = apply_match_policies(report, resume_text="2026.08至今", jd_text="连续实习三个月")
        self.assertEqual(again, report)
        self.assertEqual(notes, [])

    def test_explicit_candidate_availability_is_not_silently_removed(self):
        original = {"risks": ["需确认连续实习安排是否符合岗位要求。"], "suggestions": []}
        report, corrections = apply_match_policies(
            original, resume_text="只能连续实习两个月，每周可到岗三天。", jd_text="需连续实习三个月，每周到岗五天。",
        )
        self.assertEqual(report, original)
        self.assertEqual(corrections, [])

    def test_explicit_conflict_and_existing_experience_requirement_are_preserved(self):
        risks = ["只能到岗两天，与岗位每周五天要求冲突。", "需确认既往连续实习经验是否满足要求。"]
        report, corrections = apply_match_policies(
            {"risks": risks}, resume_text="Synthetic resume", jd_text="Synthetic JD",
        )
        self.assertEqual(report["risks"], risks)
        self.assertEqual(corrections, [])

    def test_no_hr_advice_is_invented_for_a_jd_without_arrangements(self):
        report, _ = apply_match_policies(
            {"risks": ["需确认连续实习安排。"], "suggestions": []},
            resume_text="Synthetic resume", jd_text="Python developer",
        )
        self.assertEqual(report["risks"], [])
        self.assertEqual(report["suggestions"], [])

    def test_known_remaining_claim_prevents_unverified_continuation(self):
        issue = {"field": "matched_skills", "claim": "Invented skill"}
        report = {
            "matched_skills": ["Invented skill"],
            "verification": {"status": "service_error", "issues": [issue]},
        }
        self.assertEqual(unresolved_report_issues([issue], report), [issue])
        self.assertFalse(can_continue_unverified(report))
        report["matched_skills"] = []
        self.assertEqual(unresolved_report_issues([issue], report), [])
        self.assertTrue(can_continue_unverified(report))

    def test_unverified_continuation_is_limited_to_incomplete_review_statuses(self):
        for status in ("timeout", "service_error", "correction_unverified"):
            with self.subTest(status=status):
                self.assertTrue(can_continue_unverified({"verification": {"status": status, "issues": []}}))
        for status in ("passed", "manual_accepted", "needs_review", None):
            with self.subTest(status=status):
                self.assertFalse(can_continue_unverified({"verification": {"status": status, "issues": []}}))
