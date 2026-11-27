import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app, decisions_log


class TestDecideEndpoint(unittest.TestCase):
    def setUp(self):
        decisions_log.clear()
        self.client = TestClient(app)

    def test_decide_act_and_logs_decision(self):
        mock_analysis = {
            "is_ambiguous": False,
            "ambiguity_type": "none",
            "ambiguous_spans": [],
            "readings": [
                {
                    "paraphrase": "Check the status of my order.",
                    "probability": 1.0,
                    "intent": "order_status",
                }
            ],
            "candidate_intents": [
                {"intent": "order_status", "confidence": 0.9},
                {"intent": "cancel_order", "confidence": 0.1},
            ],
        }
        with patch("main.classify_intent", new=AsyncMock(return_value=mock_analysis)), patch(
            "main.compute_similarity_margin",
            return_value={"margin": 0.2, "top_intent": "order_status", "second_intent": "cancel_order"},
        ):
            res = self.client.post(
                "/decide",
                json={"customer_id": "CUST_1001", "query": "Where is my order?"},
            )

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["decision"], "ACT")
        self.assertFalse(body["is_ambiguous"])
        self.assertEqual(body["type"], "none")
        self.assertEqual(body["risk_tier"], "low")
        self.assertEqual(body["threshold"], 0.6)
        self.assertIsNone(body["clarification"])
        self.assertIn("Success:", body["action_result"])
        self.assertEqual(body["margin"], 0.2)
        self.assertIn("ambiguity_score", body)
        self.assertEqual(len(decisions_log), 1)
        self.assertEqual(decisions_log[0]["customer_id"], "CUST_1001")

    def test_decide_ask_then_escalate(self):
        ambiguous_analysis = {
            "is_ambiguous": True,
            "ambiguity_type": "lexical",
            "candidate_intents": [
                {"intent": "card_block", "confidence": 0.65},
                {"intent": "account_closure", "confidence": 0.35},
            ],
        }
        with patch("main.classify_intent", new=AsyncMock(return_value=ambiguous_analysis)), patch(
            "main.compute_similarity_margin",
            return_value={"margin": 0.02, "top_intent": "card_block", "second_intent": "account_closure"},
        ):
            ask_res = self.client.post(
                "/decide",
                json={"customer_id": "CUST_1003", "query": "Sab band karo"},
            )
            self.assertEqual(ask_res.status_code, 200)
            ask_body = ask_res.json()
            self.assertEqual(ask_body["decision"], "ASK")
            self.assertIsNotNone(ask_body["clarification"])
            self.assertEqual(len(ask_body["clarification"]["options"]), 2)
            self.assertIsNone(ask_body["action_result"])

            esc_res = self.client.post(
                "/decide",
                json={
                    "customer_id": "CUST_1003",
                    "query": "Sab band karo",
                    "clarification_choice": "Block card",
                },
            )
            self.assertEqual(esc_res.status_code, 200)
            esc_body = esc_res.json()
            self.assertEqual(esc_body["decision"], "ESCALATE")
            self.assertIsNone(esc_body["clarification"])

        self.assertEqual(len(decisions_log), 2)

    def test_decide_unknown_customer_returns_404(self):
        res = self.client.post(
            "/decide",
            json={"customer_id": "CUST_UNKNOWN", "query": "Where is my order?"},
        )
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
