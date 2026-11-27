import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from intent_map import get_map_data
from main import app


class TestIntentMap(unittest.TestCase):
    def test_fit_once_and_project_before_after(self):
        fake_model = MagicMock()

        def fake_encode(texts, normalize_embeddings=True):
            vecs = []
            for t in texts:
                tl = t.lower()
                if "card" in tl:
                    vecs.append([0.9, 0.1, 0.0, 0.0])
                elif "order" in tl:
                    vecs.append([0.1, 0.9, 0.0, 0.0])
                else:
                    vecs.append([0.5, 0.5, 0.1, 0.0])
            return np.array(vecs, dtype=np.float32)

        fake_model.encode.side_effect = fake_encode

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            intents_file = tmp / "intents.json"
            cache_file = tmp / "map_pca.json"
            intents_file.write_text(
                json.dumps(
                    {
                        "intents": [
                            {"intent": "card_block", "examples": ["Block my card", "Freeze card"]},
                            {"intent": "order_status", "examples": ["Where is order", "Track order"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("intent_map.get_model", return_value=fake_model):
                first = get_map_data(
                    query="Band karo",
                    clarification_choice="card_block",
                    intents_path=intents_file,
                    cache_path=cache_file,
                )
                self.assertEqual(len(first["points"]), 4)
                self.assertTrue(all("color" in p and "x" in p and "y" in p for p in first["points"]))
                self.assertIsNotNone(first["query_before"])
                self.assertIsNotNone(first["query_after"])
                self.assertNotEqual(
                    (first["query_before"]["x"], first["query_before"]["y"]),
                    (first["query_after"]["x"], first["query_after"]["y"]),
                )

                # Subsequent call should reuse cached PCA without re-encoding all examples
                fake_model.encode.reset_mock()
                second = get_map_data(
                    intents_path=intents_file,
                    cache_path=cache_file,
                )
                self.assertEqual(fake_model.encode.call_count, 0)
                self.assertEqual(len(second["points"]), 4)

    def test_get_map_endpoint(self):
        client = TestClient(app)
        mock_data = {
            "points": [{"intent": "order_status", "color": "#22c55e", "text": "Where?", "x": 0.1, "y": 0.2}],
            "query_before": {"text": "Sab band", "x": 0.0, "y": 0.0},
            "query_after": {"text": "Sab band (card_block)", "x": 0.3, "y": -0.1},
        }
        with patch("main.get_map_data", return_value=mock_data):
            res = client.get("/map", params={"query": "Sab band", "clarification_choice": "card_block"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["points"]), 1)
        self.assertIsNotNone(body["query_before"])
        self.assertIsNotNone(body["query_after"])

    def test_steer_and_corrections_endpoints(self):
        client = TestClient(app)
        mock_steer = {
            "query": "Sab band karo",
            "source_intent": "account_closure",
            "target_intent": "card_block",
            "alpha": 0.5,
            "old_point": {"x": 0.1, "y": 0.2},
            "new_point": {"x": 0.3, "y": 0.1},
            "probabilities": [
                {"intent": "card_block", "probability": 0.7},
                {"intent": "account_closure", "probability": 0.3},
            ],
        }
        with patch("main.steer_query", return_value=mock_steer):
            steer_res = client.post(
                "/steer",
                json={
                    "query": "Sab band karo",
                    "source_intent": "account_closure",
                    "target_intent": "card_block",
                    "alpha": 0.5,
                },
            )
        self.assertEqual(steer_res.status_code, 200)
        self.assertEqual(steer_res.json()["new_point"], {"x": 0.3, "y": 0.1})

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
