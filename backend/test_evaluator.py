import unittest

from fastapi.testclient import TestClient

from evaluator import run_evaluation
from main import app


class TestEvaluator(unittest.TestCase):
    def test_run_evaluation_on_held_out_test_set(self):
        res = run_evaluation(max_rows=5)
        self.assertEqual(res["dataset"]["total"], 5)
        for sys_name in ("most_frequent_sense", "plain_llm", "llm_only", "uncertainty_resolver"):
            m = res["systems"][sys_name]
            self.assertIn("detection_f1", m)
            self.assertIn("false_positive_rate", m)
            self.assertIn("resolution_accuracy", m)
            self.assertIn("per_type", m)
            self.assertIn("ece", m)
            self.assertGreater(len(m["accuracy_vs_coverage"]), 0)

        client = TestClient(app)
        http_res = client.get("/evaluate?max_rows=5")
        self.assertEqual(http_res.status_code, 200)
        self.assertEqual(http_res.json()["dataset"]["total"], 5)


if __name__ == "__main__":
    unittest.main()
