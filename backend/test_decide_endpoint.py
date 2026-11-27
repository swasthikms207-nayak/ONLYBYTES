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
            "main.compute_reading_ambiguity",
            return_value={"score": 0.0, "entropy": 0.0, "divergence": 0.0},
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
        self.assertEqual(body["score"], 0.0)
        self.assertEqual(body["entropy"], 0.0)
        self.assertEqual(body["divergence"], 0.0)
        self.assertIn("context_used", body)
        self.assertEqual(len(decisions_log), 1)
        self.assertEqual(decisions_log[0]["customer_id"], "CUST_1001")

    def test_decide_ask_then_escalate(self):
        ambiguous_analysis = {
            "is_ambiguous": True,
            "ambiguity_type": "lexical",
            "ambiguous_spans": [{"text": "Sab band", "type": "lexical", "reason": "Unclear"}],
            "readings": [
                {"paraphrase": "Block my debit card.", "probability": 0.65, "intent": "card_block"},
                {"paraphrase": "Close my account.", "probability": 0.35, "intent": "account_closure"},
            ],
            "candidate_intents": [
                {"intent": "card_block", "confidence": 0.65},
                {"intent": "account_closure", "confidence": 0.35},
            ],
        }
        mock_steer = {
            "before_similarities": [{"paraphrase": "Block my debit card.", "intent": "card_block", "similarity": 0.5}],
            "similarities": [{"paraphrase": "Block my debit card.", "intent": "card_block", "similarity": 0.85}],
        }
        with patch("main.classify_intent", new=AsyncMock(return_value=ambiguous_analysis)), patch(
            "main.compute_reading_ambiguity",
            return_value={"score": 0.62, "entropy": 0.93, "divergence": 0.67},
        ), patch("main.steer_query", return_value=mock_steer):
            ask_res = self.client.post(
                "/decide",
                json={"customer_id": "CUST_1003", "query": "Sab band karo"},
            )
            self.assertEqual(ask_res.status_code, 200)
            ask_body = ask_res.json()
            self.assertEqual(ask_body["decision"], "ASK")
            self.assertIsNotNone(ask_body["clarification"])
            self.assertIsNone(ask_body["before_similarities"])
            self.assertEqual(len(ask_body["clarification"]["options"]), 2)
            self.assertIsNone(ask_body["action_result"])

            esc_res = self.client.post(
                "/decide",
                json={
                    "customer_id": "CUST_1003",
                    "query": "Sab band karo",
                    "clarification_choice": "card_block",
                },
            )
            self.assertEqual(esc_res.status_code, 200)
            esc_body = esc_res.json()
            self.assertEqual(esc_body["decision"], "ESCALATE")
            self.assertIsNone(esc_body["clarification"])
            self.assertIsNotNone(esc_body["before_similarities"])
            self.assertIsNotNone(esc_body["after_similarities"])

        self.assertEqual(len(decisions_log), 2)

    def test_decide_unknown_customer_returns_404(self):
        res = self.client.post(
            "/decide",
            json={"customer_id": "CUST_UNKNOWN", "query": "Where is my order?"},
        )
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
