import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from intent_embeddings import (
    MODEL_NAME,
    compute_and_cache_centroids,
    query_centroid_similarities,
)


class TestIntentEmbeddings(unittest.TestCase):
    def test_compute_cache_and_similarity_with_mock_model(self):
        fake_model = MagicMock()

        def fake_encode(texts, normalize_embeddings=True):
            out = []
            for t in texts:
                if "order" in t.lower():
                    vec = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    vec = np.array([0.0, 1.0], dtype=np.float32)
                out.append(vec)
            return np.array(out)

        fake_model.encode.side_effect = fake_encode

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            intents_file = tmp / "intents.json"
            cache_file = tmp / "centroids.json"
            intents_file.write_text(
                json.dumps(
                    {
                        "intents": [
                            {"intent": "order_status", "examples": ["Where is my order?"]},
                            {"intent": "card_block", "examples": ["Block my card"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("intent_embeddings.get_model", return_value=fake_model):
                centroids = compute_and_cache_centroids(
                    intents_path=intents_file,
                    cache_path=cache_file,
                    force_recompute=True,
                )
                self.assertTrue(cache_file.exists())
                self.assertEqual(set(centroids.keys()), {"order_status", "card_block"})

                sims = query_centroid_similarities(
                    "Track my order",
                    intents_path=intents_file,
                    cache_path=cache_file,
                )
                self.assertAlmostEqual(sims["order_status"], 1.0, places=4)
                self.assertAlmostEqual(sims["card_block"], 0.0, places=4)

            # Verify cache loads from disk without calling encode again
            fake_model.encode.reset_mock()
            with patch("intent_embeddings.get_model", return_value=fake_model):
                cached_centroids = compute_and_cache_centroids(
                    intents_path=intents_file,
                    cache_path=cache_file,
                )
                self.assertEqual(fake_model.encode.call_count, 0)
                self.assertEqual(set(cached_centroids.keys()), {"order_status", "card_block"})
                saved = json.loads(cache_file.read_text(encoding="utf-8"))
                self.assertEqual(saved["model_name"], MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
