import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from clarification import RECORD_STORE, build_inline_clarification
from evaluator import RESULTS_JSON, run_evaluation
from main import app


class TestClarification(unittest.TestCase):
    def test_inline_clarification_never_generated_for_auto_accept_or_none(self):
        self.assertIsNone(
            build_inline_clarification(
                text="Block my Visa card",
                ambiguity_type="none",
                status="AUTO_ACCEPT",
                top_reading="Block my Visa card",
                alternatives=[],
            )
        )
        self.assertIsNone(
            build_inline_clarification(
                text="Block my card",
                ambiguity_type="lexical",
                status="AUTO_ACCEPT",
                top_reading="Block card",
                alternatives=[{"reading": "Close account", "probability": 0.1}],
            )
        )

    def test_inline_clarification_and_post_clarify_endpoint(self):
        client = TestClient(app)
        proc = client.post(
            "/process",
            json={"raw_text": "Cancel all active items now\nWhere is my order ORD_5501?", "format": "text"},
        )
        self.assertEqual(proc.status_code, 200)
        results = proc.json()["results"]
        self.assertEqual(len(results), 2)

        review_rec = results[0]
        clear_rec = results[1]
        self.assertIn(review_rec["status"], ("REVIEW", "ACCEPT_WITH_ALTERNATIVES"))
        self.assertIsNotNone(review_rec["clarification"])
        self.assertTrue(any(o["is_other"] for o in review_rec["clarification"]["options"]))
        self.assertEqual(clear_rec["status"], "AUTO_ACCEPT")
        self.assertIsNone(clear_rec["clarification"])

        chosen = review_rec["clarification"]["options"][1]["reading"]
        clar = client.post(
            "/clarify",
            json={"record_id": review_rec["id"], "chosen_option": chosen},
        )
        self.assertEqual(clar.status_code, 200)
        updated = clar.json()
        self.assertTrue(updated["clarified_by_user"])
        self.assertEqual(updated["status"], "AUTO_ACCEPT")
        self.assertEqual(updated["top_reading"], chosen)
        self.assertIsNone(updated["clarification"])
        self.assertIn(updated["id"], RECORD_STORE)

    def test_simulated_user_eval_and_results_json(self):
        res = run_evaluation()
        sim = res["simulated_user_clarification"]
        for user_mode in ("always_correct", "ten_percent_wrong"):
            for policy in ("never_ask", "selective_ask", "always_ask"):
                m = sim[user_mode][policy]
                self.assertIn("accuracy_before", m)
                self.assertIn("accuracy_after", m)
                self.assertIn("questions_per_100", m)
                self.assertIn("share_questions_needed", m)
        self.assertGreater(
            sim["always_correct"]["selective_ask"]["accuracy_after"],
            sim["always_correct"]["never_ask"]["accuracy_after"],
        )
        self.assertTrue(Path(RESULTS_JSON).exists())
        saved = json.loads(Path(RESULTS_JSON).read_text(encoding="utf-8"))
        self.assertIn("simulated_user_clarification", saved)


if __name__ == "__main__":
    unittest.main()
