import tempfile
import unittest
from pathlib import Path

from evaluate import compute_ece, evaluate_system, save_calibration_plot


class TestEvaluate(unittest.TestCase):
    def test_four_systems_metrics_and_calibration(self):
        rows = [
            {
                "query": "Where is my order?",
                "customer_id": "CUST_1001",
                "is_ambiguous": "false",
                "ambiguity_type": "none",
                "correct_intent": "order_status",
                "annotator2_label": "order_status",
            },
            {
                "query": "Sab band karo",
                "customer_id": "CUST_1003",
                "is_ambiguous": "true",
                "ambiguity_type": "lexical",
                "correct_intent": "card_block",
                "annotator2_label": "account_closure",
            },
        ]
        predictions = [
            {
                "analysis": {
                    "is_ambiguous": False,
                    "ambiguity_type": "none",
                    "candidate_intents": [
                        {"intent": "order_status", "confidence": 0.9},
                        {"intent": "cancel_order", "confidence": 0.1},
                    ],
                },
                "top_intent": "order_status",
                "top_confidence": 0.9,
                "margin": 0.2,
            },
            {
                "analysis": {
                    "is_ambiguous": True,
                    "ambiguity_type": "lexical",
                    "candidate_intents": [
                        {"intent": "account_closure", "confidence": 0.6},
                        {"intent": "card_block", "confidence": 0.4},
                    ],
                },
                "top_intent": "account_closure",
                "top_confidence": 0.6,
                "margin": 0.02,
            },
        ]
        policies = {
            "order_status": {"risk_tier": "low", "cost_weight": 1, "confidence_threshold": 0.6},
            "card_block": {"risk_tier": "high", "cost_weight": 10, "confidence_threshold": 0.95},
            "account_closure": {"risk_tier": "high", "cost_weight": 10, "confidence_threshold": 0.95},
        }

        always_act = evaluate_system(rows, predictions, policies, "always_act")
        always_ask = evaluate_system(rows, predictions, policies, "always_ask")
        risk_aware = evaluate_system(rows, predictions, policies, "risk_aware_margin")

        self.assertEqual(always_act["unnecessary_question_rate"], 0.0)
        self.assertEqual(always_act["cost_weighted_error"], 5.0)
        self.assertEqual(always_ask["unnecessary_question_rate"], 1.0)
        self.assertEqual(risk_aware["detection_f1"], 1.0)
        self.assertEqual(risk_aware["cost_weighted_error"], 0.0)

        cal = compute_ece([0.9, 0.6], [1, 0])
        self.assertGreaterEqual(cal["ece"], 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "cal.png"
            save_calibration_plot(cal, output_path=plot_path)
            self.assertTrue(plot_path.exists())


if __name__ == "__main__":
    unittest.main()
