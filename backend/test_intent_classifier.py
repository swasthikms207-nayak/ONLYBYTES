import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from intent_classifier import IntentAnalysisResult, classify_intent


class TestIntentClassifier(unittest.TestCase):
    def test_clear_sentence_returns_empty_spans_and_single_reading(self):
        result = IntentAnalysisResult(
            ambiguous_spans=[],
            readings=[
                {
                    "paraphrase": "Check the delivery status of my order.",
                    "probability": 0.85,
                    "intent": "order_status",
                },
                {
                    "paraphrase": "Cancel my order.",
                    "probability": 0.15,
                    "intent": "cancel_order",
                },
            ],
        )
        self.assertEqual(result.ambiguous_spans, [])
        self.assertEqual(len(result.readings), 1)
        self.assertEqual(result.readings[0].probability, 1.0)
        self.assertFalse(result.is_ambiguous)
        self.assertEqual(result.ambiguity_type, "none")

    def test_ambiguous_readings_normalize_to_one(self):
        result = IntentAnalysisResult(
            ambiguous_spans=[
                {
                    "text": "band kar do",
                    "type": "lexical",
                    "reason": "Can mean blocking a card or closing an account.",
                }
            ],
            readings=[
                {
                    "paraphrase": "Block my debit card immediately.",
                    "probability": 0.5,
                    "intent": "card_block",
                },
                {
                    "paraphrase": "Permanently close my user account.",
                    "probability": 0.3,
                    "intent": "account_closure",
                },
                {
                    "paraphrase": "Cancel my pending order.",
                    "probability": 0.1,
                    "intent": "cancel_order",
                },
            ],
        )
        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.ambiguity_type, "lexical")
        self.assertAlmostEqual(sum(r.probability for r in result.readings), 1.0, places=4)
        self.assertAlmostEqual(sum(c.confidence for c in result.candidate_intents), 1.0, places=4)

    def test_classify_intent_hinglish_mocked_optional_args(self):
        mock_llm_json = json.dumps(
            {
                "ambiguous_spans": [
                    {
                        "text": "Sab band kar do",
                        "type": "scope",
                        "reason": "'Sab' could refer to card, account, or pending order.",
                    }
                ],
                "readings": [
                    {
                        "paraphrase": "Block my debit card due to fraud.",
                        "probability": 0.5,
                        "intent": "card_block",
                    },
                    {
                        "paraphrase": "Close my entire OnlyBytes account.",
                        "probability": 0.3,
                        "intent": "account_closure",
                    },
                    {
                        "paraphrase": "Dispute the unauthorized transaction.",
                        "probability": 0.2,
                        "intent": "transaction_dispute",
                    },
                ],
            }
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": mock_llm_json}}]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"LLM_API_KEY": "sk-test-key"}), patch(
            "intent_classifier.httpx.AsyncClient", return_value=mock_client
        ):
            out = asyncio.run(classify_intent("Sab band kar do"))

        self.assertTrue(out["is_ambiguous"])
        self.assertEqual(len(out["ambiguous_spans"]), 1)
        self.assertEqual(len(out["readings"]), 3)
        self.assertAlmostEqual(sum(r["probability"] for r in out["readings"]), 1.0, places=4)

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {"LLM_API_KEY": "your_llm_api_key_here"}):
            with self.assertRaises(RuntimeError):
                asyncio.run(classify_intent("Where is my order?"))


if __name__ == "__main__":
    unittest.main()
