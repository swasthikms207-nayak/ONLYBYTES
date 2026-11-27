import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from evaluate import (
    PRIMARY_METRICS,
    check_quality_preserved,
    check_steering_control,
    check_visualizer_faithfulness,
    compute_ece,
    evaluate_system,
    run_ablation,
    save_calibration_plot,
)


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

    def test_three_evaluation_checks(self):
        fake_centroids = {
            "card_block": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "order_status": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "account_closure": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        }
        fake_model = MagicMock()
        fake_model.encode.side_effect = lambda texts, normalize_embeddings=True: np.tile(
            np.array([[0.2, 0.2, 0.8, 0.1]], dtype=np.float32), (len(texts), 1)
        )
        rows = [
            {
                "query": "Sab band karo",
                "correct_intent": "card_block",
                "annotator2_label": "account_closure",
            }
        ]
        fake_proj = {
            "points": [
                {"text": f"t{i}", "x": float(i), "y": float(i * 0.5)}
                for i in range(12)
            ]
        }
        fake_model_vis = MagicMock()
        fake_model_vis.encode.return_value = np.array(
            [[float(i), float(i * 0.5), 0.0, 0.0] for i in range(12)],
            dtype=np.float64,
        )

        with patch("evaluate.compute_and_cache_centroids", return_value=fake_centroids), patch(
            "evaluate.get_model", return_value=fake_model
        ):
            steer_res = check_steering_control(rows)
            self.assertGreater(steer_res["similarity_gain"], 0.0)
            self.assertGreater(steer_res["correlation"], 0.9)

            qual_res = check_quality_preserved(rows)
            self.assertGreaterEqual(qual_res["accuracy_after"], qual_res["accuracy_before"])

        with patch("evaluate.fit_or_load_projection", return_value=fake_proj), patch(
            "evaluate.get_model", return_value=fake_model_vis
        ):
            vis_res = check_visualizer_faithfulness()
            self.assertGreaterEqual(vis_res["trustworthiness"], 0.9)

    def test_ablation_reports_all_primary_metrics(self):
        rows = [
            {
                "query": "Sab band karo",
                "customer_id": "CUST_1003",
                "is_ambiguous": "true",
                "ambiguity_type": "lexical",
                "correct_intent": "card_block",
                "annotator2_label": "account_closure",
            }
        ]
        predictions = [
            {
                "analysis": {
                    "is_ambiguous": True,
                    "ambiguity_type": "lexical",
                    "candidate_intents": [
                        {"intent": "account_closure", "confidence": 0.96},
                        {"intent": "card_block", "confidence": 0.04},
                    ],
                },
                "top_intent": "account_closure",
                "top_confidence": 0.96,
                "margin": 0.01,
            }
        ]
        policies = {
            "card_block": {"risk_tier": "high", "cost_weight": 10, "confidence_threshold": 0.95},
            "account_closure": {"risk_tier": "high", "cost_weight": 10, "confidence_threshold": 0.95},
            "order_status": {"risk_tier": "low", "cost_weight": 1, "confidence_threshold": 0.6},
        }
        fake_centroids = {
            "card_block": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "account_closure": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "order_status": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.3, 0.6, 0.1]], dtype=np.float32)

        with patch("evaluate.compute_and_cache_centroids", return_value=fake_centroids), patch(
            "evaluate.get_model", return_value=fake_model
        ):
            ablation = run_ablation(rows, predictions, policies)

        for metric in PRIMARY_METRICS:
            self.assertIn(metric, ablation["difference"])
        self.assertGreater(ablation["difference"]["detection_f1"], 0.0)
        self.assertLess(ablation["difference"]["cost_weighted_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
