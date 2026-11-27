import unittest

from decision_engine import decide_action


class TestDecisionEngine(unittest.TestCase):
    def test_act_when_above_threshold(self):
        analysis = {
            "is_ambiguous": False,
            "ambiguity_type": "none",
            "candidate_intents": [
                {"intent": "order_status", "confidence": 0.85},
                {"intent": "cancel_order", "confidence": 0.15},
            ],
        }
        res = decide_action(analysis, clarification_count=0)
        self.assertEqual(res["decision"], "ACT")
        self.assertEqual(res["intent"], "order_status")
        self.assertIsNone(res["clarification_question"])
        self.assertEqual(res["options"], [])

    def test_ask_when_below_threshold_first_time(self):
        analysis = {
            "is_ambiguous": True,
            "ambiguity_type": "lexical",
            "candidate_intents": [
                {"intent": "card_block", "confidence": 0.6},
                {"intent": "account_closure", "confidence": 0.3},
                {"intent": "transaction_dispute", "confidence": 0.1},
            ],
        }
        res = decide_action(analysis, clarification_count=0)
        self.assertEqual(res["decision"], "ASK")
        self.assertTrue(res["clarification_question"])
        self.assertGreaterEqual(len(res["options"]), 2)
        self.assertLessEqual(len(res["options"]), 3)
        self.assertEqual(res["options"][0]["intent"], "card_block")

    def test_escalate_after_one_clarification(self):
        analysis = {
            "is_ambiguous": True,
            "ambiguity_type": "underspecified",
            "candidate_intents": [
                {"intent": "payment_reversal", "confidence": 0.72},
                {"intent": "refund_status", "confidence": 0.28},
            ],
        }
        res = decide_action(analysis, clarification_count=1)
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIsNone(res["clarification_question"])
        self.assertEqual(res["options"], [])

    def test_high_reading_ambiguity_score_triggers_ask(self):
        analysis = {
            "is_ambiguous": True,
            "ambiguity_type": "lexical",
            "candidate_intents": [
                {"intent": "order_status", "confidence": 0.70},
                {"intent": "cancel_order", "confidence": 0.30},
            ],
        }
        res = decide_action(analysis, clarification_count=0, ambiguity_score=0.45)
        self.assertEqual(res["decision"], "ASK")
        self.assertIn("ambiguity score", res["reason"].lower())


if __name__ == "__main__":
    unittest.main()
