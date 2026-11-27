import unittest

from context_filter import filter_readings_with_context
from decision_engine import decide_action


class TestContextFilter(unittest.TestCase):
    def test_rules_out_shipped_order_actions_and_resolves(self):
        readings = [
            {
                "paraphrase": "Cancel my order before it ships.",
                "probability": 0.45,
                "intent": "cancel_order",
            },
            {
                "paraphrase": "Change the shipping address of my order.",
                "probability": 0.25,
                "intent": "change_address",
            },
            {
                "paraphrase": "Check where my package is right now.",
                "probability": 0.30,
                "intent": "order_status",
            },
        ]
        customer_data = {
            "pending_orders": [{"order_id": "ORD_5490", "status": "in_transit"}],
            "pending_transactions": [],
        }
        res = filter_readings_with_context(readings, customer_data, chat_history=[])

        self.assertEqual(len(res["readings"]), 1)
        self.assertEqual(res["readings"][0]["intent"], "order_status")
        self.assertEqual(res["readings"][0]["probability"], 1.0)
        self.assertTrue(res["resolved_by_context"])
        self.assertIn("ORD_5490", res["context_used"])

        decision = decide_action(
            {"candidate_intents": res["candidate_intents"]},
            clarification_count=0,
            ambiguity_score=0.0,
            resolved_by_context=res["resolved_by_context"],
        )
        self.assertEqual(decision["decision"], "ACT")
        self.assertIn("Resolved by customer context", decision["reason"])

    def test_keeps_readings_if_all_would_be_removed(self):
        readings = [
            {"paraphrase": "Cancel order", "probability": 0.6, "intent": "cancel_order"},
            {"paraphrase": "Track order", "probability": 0.4, "intent": "order_status"},
        ]
        res = filter_readings_with_context(
            readings,
            customer_data={"pending_orders": [], "pending_transactions": []},
            chat_history=[],
        )
        self.assertEqual(len(res["readings"]), 2)
        self.assertFalse(res["resolved_by_context"])


if __name__ == "__main__":
    unittest.main()
