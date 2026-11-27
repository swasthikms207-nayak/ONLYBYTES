import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from intent_map import fit_or_load_projection, get_map_data
from main import app


class TestIntentMap(unittest.TestCase):
    def test_fit_once_and_project_query_readings_and_steered(self):
        fake_model = MagicMock()

        def fake_encode(texts, normalize_embeddings=True):
            vecs = []
            for t in texts:
                tl = t.lower()
                if "card" in tl:
                    vecs.append([0.9, 0.1, 0.0, 0.0])
                elif "account" in tl:
                    vecs.append([0.1, 0.9, 0.0, 0.0])
                else:
                    vecs.append([0.5, 0.5, 0.1, 0.0])
            return np.array(vecs, dtype=np.float32)

        fake_model.encode.side_effect = fake_encode

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            past_csv = tmp / "past.csv"
            cust_json = tmp / "customers.json"
            cache_file = tmp / "query_readings_pca.json"
            past_csv.write_text(
                "query\nBlock my card\nClose my account\nWhere is my order\n",
                encoding="utf-8",
            )
            cust_json.write_text(json.dumps({"customers": []}), encoding="utf-8")

            with patch("intent_map.get_model", return_value=fake_model):
                fit_or_load_projection(
                    past_csv=past_csv,
                    customers_path=cust_json,
                    cache_path=cache_file,
                    force_recompute=True,
                )
                self.assertTrue(cache_file.exists())

                res = get_map_data(
                    query="Sab band karo",
                    readings=["Block my debit card", "Close my account"],
                    target_reading="Block my debit card",
                    alpha=0.5,
                    cache_path=cache_file,
                )
                self.assertNotIn("points", res)
                self.assertIsNotNone(res["query_point"])
                self.assertEqual(len(res["reading_points"]), 2)
                self.assertIsNotNone(res["steered_point"])
                self.assertNotEqual(
                    (res["query_point"]["x"], res["query_point"]["y"]),
                    (res["steered_point"]["x"], res["steered_point"]["y"]),
                )

    def test_get_map_endpoint(self):
        client = TestClient(app)
        mock_data = {
            "query_point": {"text": "Sab band", "x": 0.0, "y": 0.0},
            "reading_points": [
                {"paraphrase": "Block my card", "x": 0.5, "y": 0.2},
                {"paraphrase": "Close my account", "x": -0.4, "y": -0.1},
            ],
            "steered_point": {"text": "Block my card", "alpha": 0.5, "x": 0.25, "y": 0.1},
        }
        with patch("main.get_map_data", return_value=mock_data):
            res = client.get(
                "/map",
                params={
                    "query": "Sab band",
                    "readings": ["Block my card", "Close my account"],
                    "target_reading": "Block my card",
                    "alpha": 0.5,
                },
            )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertNotIn("points", body)
        self.assertIsNotNone(body["query_point"])
        self.assertEqual(len(body["reading_points"]), 2)
        self.assertIsNotNone(body["steered_point"])

    def test_steer_and_corrections_endpoints(self):
        client = TestClient(app)
        mock_steer = {
            "query": "Sab band karo",
            "target_reading": "Block my debit card immediately.",
            "alpha": 0.5,
            "old_point": {"x": 0.1, "y": 0.2},
            "new_point": {"x": 0.3, "y": 0.1},
            "before_similarities": [
                {"paraphrase": "Block my debit card immediately.", "intent": "card_block", "similarity": 0.45},
                {"paraphrase": "Close my account.", "intent": "account_closure", "similarity": 0.40},
            ],
            "similarities": [
                {"paraphrase": "Block my debit card immediately.", "intent": "card_block", "similarity": 0.85},
                {"paraphrase": "Close my account.", "intent": "account_closure", "similarity": 0.30},
            ],
        }
        with patch("main.steer_query", return_value=mock_steer):
            steer_res = client.post(
                "/steer",
                json={
                    "query": "Sab band karo",
                    "target_reading": "Block my debit card immediately.",
                    "alpha": 0.5,
                    "readings": [
                        {"paraphrase": "Block my debit card immediately.", "intent": "card_block"},
                        {"paraphrase": "Close my account.", "intent": "account_closure"},
                    ],
                },
            )
        self.assertEqual(steer_res.status_code, 200)
        self.assertEqual(steer_res.json()["new_point"], {"x": 0.3, "y": 0.1})
        self.assertEqual(len(steer_res.json()["similarities"]), 2)

        corr_res = client.post(
            "/corrections",
            json={
                "query": "Sab band karo",
                "target_intent": "card_block",
                "alpha": 0.5,
            },
        )
        self.assertEqual(corr_res.status_code, 200)
        self.assertGreaterEqual(corr_res.json()["count"], 1)


if __name__ == "__main__":
    unittest.main()
