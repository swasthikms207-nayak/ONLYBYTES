import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from main import app
from resolver import parse_csv_records, resolve_batch, route_record


class TestResolver(unittest.TestCase):
    def test_routing_rules(self):
        self.assertEqual(route_record(0.85, conflict=False), "AUTO_ACCEPT")
        self.assertEqual(route_record(0.85, conflict=True), "REVIEW")
        self.assertEqual(route_record(0.65, conflict=False), "ACCEPT_WITH_ALTERNATIVES")
        self.assertEqual(route_record(0.40, conflict=False), "REVIEW")

    def test_parse_csv_with_header(self):
        csv_data = 'text,note\n"Block my card",high\n"Cancel all items",ambiguous\n'
        self.assertEqual(parse_csv_records(csv_data), ["Block my card", "Cancel all items"])

    def test_resolve_batch_mock_fallback(self):
        with patch.dict("os.environ", {"LLM_API_KEY": "your_llm_api_key_here"}):
            out = asyncio.run(
                resolve_batch([
                    "Where is my order ORD_5501?",
                    "Cancel all active items",
                    "Cancel it now",
                ])
            )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["ambiguity_type"], "none")
        self.assertEqual(out[0]["top_reading"], "Where is my order ORD_5501?")
        self.assertEqual(out[0]["status"], "AUTO_ACCEPT")
        self.assertEqual(out[1]["status"], "REVIEW")
        self.assertTrue(out[2]["conflict"])
        self.assertEqual(out[2]["status"], "REVIEW")

    def test_llm_5_sample_self_consistency_and_conflict(self):
        alone_payload = {
            "ambiguity_type": "lexical",
            "top_reading": "Cancel the pending order.",
            "alternatives": [{"reading": "Block card", "probability": 0.3}],
            "top_probability": 0.7,
        }
        ctx_block = {
            "ambiguity_type": "lexical",
            "top_reading": "Block the lost debit card.",
            "alternatives": [{"reading": "Close account", "probability": 0.2}],
            "top_probability": 0.8,
        }
        ctx_close = {
            "ambiguity_type": "lexical",
            "top_reading": "Close the user account.",
            "alternatives": [{"reading": "Block card", "probability": 0.4}],
            "top_probability": 0.6,
        }
        # First record ("Close it"): 1 alone + 5 context samples
        # Second record ("My card was stolen"): 1 alone + 5 context samples (clear)
        clear_payload = {
            "ambiguity_type": "none",
            "top_reading": "My card was stolen",
            "alternatives": [],
            "top_probability": 0.98,
        }
        responses = [
            alone_payload,
            ctx_block,
            ctx_block,
            ctx_block,
            ctx_block,
            ctx_close,
            clear_payload,
            clear_payload,
            clear_payload,
            clear_payload,
            clear_payload,
            clear_payload,
        ]

        def make_resp(payload):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
            return r

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(side_effect=[make_resp(p) for p in responses])

        with patch.dict("os.environ", {"LLM_API_KEY": "sk-test-key"}), patch(
            "resolver.httpx.AsyncClient", return_value=mock_client
        ):
            out = asyncio.run(resolve_batch(["Close it", "My card was stolen"]))

        self.assertEqual(len(out), 2)
        first = out[0]
        self.assertEqual(first["text"], "Close it")
        self.assertTrue(first["conflict"])
        self.assertAlmostEqual(first["disagreement"], 0.2, places=4)
        self.assertEqual(first["status"], "REVIEW")
        self.assertEqual(len(first["resolution_steps"]), 7)
        self.assertEqual([s["step"] for s in first["resolution_steps"]], [1, 2, 3, 4, 5, 6, 7])
        for s in first["resolution_steps"]:
            self.assertIn("action", s)
            self.assertIn("result", s)

        second = out[1]
        self.assertEqual(second["ambiguity_type"], "none")
        self.assertEqual(second["top_reading"], "My card was stolen")
        self.assertEqual(second["status"], "AUTO_ACCEPT")

    def test_process_endpoint_text_and_csv(self):
        client = TestClient(app)
        with patch.dict("os.environ", {"LLM_API_KEY": "your_llm_api_key_here"}):
            res = client.post(
                "/process",
                json={"raw_text": "Block my card\nCancel all items", "format": "text"},
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["count"], 2)

            csv_res = client.post(
                "/process",
                json={"raw_text": "text\nWhere is my order?\nFix it", "format": "csv"},
            )
            self.assertEqual(csv_res.status_code, 200)
            self.assertEqual(csv_res.json()["count"], 2)

    def test_process_empty_returns_400(self):
        client = TestClient(app)
        res = client.post("/process", json={"raw_text": "   ", "format": "text"})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
