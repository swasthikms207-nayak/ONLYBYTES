import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from evaluator import EVAL_FILE, run_evaluation
from main import app


class TestEvaluator(unittest.TestCase):
    def test_eval_dataset_has_40_sentences_and_three_categories(self):
        data = json.loads(Path(EVAL_FILE).read_text(encoding="utf-8"))
        self.assertEqual(len(data), 40)
        cats = {row["category"] for row in data}
        self.assertEqual(cats, {"ambiguous", "context_conflicting", "unambiguous"})

    def test_run_evaluation_and_endpoint(self):
        res = run_evaluation()
        self.assertEqual(res["dataset"]["total"], 40)
        for sys_name in ("most_frequent_sense", "plain_llm", "uncertainty_resolver"):
            m = res["systems"][sys_name]
            self.assertIn("detection_f1", m)
            self.assertIn("false_positive_rate", m)
            self.assertIn("resolution_accuracy", m)
            self.assertGreater(len(m["accuracy_vs_coverage"]), 0)

        client = TestClient(app)
        http_res = client.get("/evaluate")
        self.assertEqual(http_res.status_code, 200)
        self.assertEqual(http_res.json()["dataset"]["total"], 40)


if __name__ == "__main__":
    unittest.main()
