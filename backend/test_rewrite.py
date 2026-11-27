import unittest

from rewrite import rewrite_sentence


class TestRewrite(unittest.TestCase):
    def test_unambiguous_returns_unchanged(self):
        out = rewrite_sentence(
            text="Water boils at 100 degrees Celsius.",
            ambiguity_type="none",
            top_reading="Water boils at 100 degrees Celsius.",
            alternatives=[],
            confidence=0.95,
        )
        self.assertEqual(out["resolved_text"], "Water boils at 100 degrees Celsius.")
        self.assertFalse(out["assumed"])
        self.assertEqual(out["changes"], [])

    def test_low_confidence_outputs_both_and_marks_assumed(self):
        out = rewrite_sentence(
            text="Cancel all active items",
            ambiguity_type="scope",
            top_reading="Apply the action to all active items on the account.",
            alternatives=[{"reading": "Apply only to the most recent item.", "probability": 0.32}],
            confidence=0.42,
        )
        self.assertTrue(out["assumed"])
        self.assertEqual(
            out["resolved_text"],
            "[Apply the action to all active items on the account.] / [Apply only to the most recent item.]",
        )
        self.assertEqual(len(out["changes"]), 1)

    def test_normalizes_date_time_and_amount(self):
        out = rewrite_sentence(
            text="Pay 500 on 05/06/2026 at 5",
            ambiguity_type="temporal_numeric",
            top_reading="Pay USD 500 on 2026-06-05 at 5:00 PM",
            alternatives=[],
            confidence=0.85,
        )
        self.assertIn("2026-06-05", out["resolved_text"])
        self.assertIn("5:00 PM", out["resolved_text"])
        self.assertIn("USD 500", out["resolved_text"])
        self.assertEqual(len(out["changes"]), 3)
        for ch in out["changes"]:
            self.assertIn("span", ch)
            self.assertIn("before", ch)
            self.assertIn("after", ch)
            self.assertIn("reason", ch)

    def test_replaces_pronoun_with_named_entity(self):
        out = rewrite_sentence(
            text="Cancel it before dispatch",
            ambiguity_type="referential",
            top_reading="Cancel headphones order ORD_5519 before dispatch.",
            alternatives=[],
            confidence=0.72,
            context_items=["I ordered headphones ORD_5519 yesterday."],
        )
        self.assertIn("ORD_5519", out["resolved_text"])
        self.assertNotIn(" it ", f" {out['resolved_text']} ")
        self.assertFalse(out["assumed"])
        self.assertGreaterEqual(len(out["changes"]), 1)

    def test_syntactic_rewrite_uses_top_reading(self):
        out = rewrite_sentence(
            text="I saw the man with the telescope",
            ambiguity_type="syntactic",
            top_reading="I used a telescope to see the man",
            alternatives=[],
            confidence=0.84,
        )
        self.assertEqual(out["resolved_text"], "I used a telescope to see the man")
        self.assertFalse(out["assumed"])
        self.assertEqual(len(out["changes"]), 1)


if __name__ == "__main__":
    unittest.main()
