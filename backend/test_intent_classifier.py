import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from intent_classifier import IntentAnalysisResult, classify_intent


class TestIntentClassifier(unittest.TestCase):
    def test_normalization_sums_to_one(self):
        result = IntentAnalysisResult(
            is_ambiguous=True,
            ambiguity_type="lexical",
            candidate_intents=[
                {"intent": "card_block", "confidence": 0.6},
                {"intent": "account_closure", "confidence": 0.3},
            ],
        )
        total = sum(c.confidence for c in result.candidate_intents)
        self.assertAlmostEqual(total, 1.0, places=4)
        self.assertEqual(result.candidate_intents[0].intent, "card_block")

    def test_unambiguous_forces_none_type(self):
        result = IntentAnalysisResult(
            is_ambiguous=False,
            ambiguity_type="referential",
            candidate_intents=[{"intent": "order_status", "confidence": 1.0}],
        )
        self.assertEqual(result.ambiguity_type, "none")

    def test_classify_intent_hinglish_mocked(self):
        mock_llm_json = json.dumps(
            {
                "is_ambiguous": True,
                "ambiguity_type": "lexical",
                "candidate_intents": [
                    {"intent": "card_block", "confidence": 0.55},
                    {"intent": "account_closure", "confidence": 0.44},
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
            out = asyncio.run(
                classify_intent(
                    query="Mera sab kuch band kar do abhi",
                    customer_data={"customer_id": "CUST_1003"},
                    chat_history=[{"role": "user", "message": "Card pe fraud lag raha hai"}],
                )
            )

        self.assertTrue(out["is_ambiguous"])
        self.assertEqual(out["ambiguity_type"], "lexical")
        self.assertAlmostEqual(
            sum(c["confidence"] for c in out["candidate_intents"]), 1.0, places=4
        )

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {"LLM_API_KEY": "your_llm_api_key_here"}):
            with self.assertRaises(RuntimeError):
                asyncio.run(classify_intent("Where is my order?", {}, []))


if __name__ == "__main__":
    unittest.main()
