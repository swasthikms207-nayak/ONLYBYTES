import csv
import unittest
from pathlib import Path

from classifier import LABELS, blend_scores

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestClassifierAndDataset(unittest.TestCase):
    def test_dataset_600_rows_and_splits(self):
        full_rows = list(csv.DictReader((DATA_DIR / "text_dataset.csv").open(encoding="utf-8")))
        train_rows = list(csv.DictReader((DATA_DIR / "train.csv").open(encoding="utf-8")))
        val_rows = list(csv.DictReader((DATA_DIR / "val.csv").open(encoding="utf-8")))
        test_rows = list(csv.DictReader((DATA_DIR / "test.csv").open(encoding="utf-8")))

        self.assertEqual(len(full_rows), 600)
        self.assertEqual(len(train_rows), 420)
        self.assertEqual(len(val_rows), 90)
        self.assertEqual(len(test_rows), 90)

        unambig = sum(1 for r in full_rows if r["is_ambiguous"] == "false")
        self.assertEqual(unambig, 150)

    def test_blend_scores_agreement(self):
        spacy_res = {lbl: {"score": 0.1, "span": "", "reason": ""} for lbl in LABELS}
        spacy_res["lexical"]["score"] = 0.85
        deb_probs = {lbl: 0.05 for lbl in LABELS}
        deb_probs["lexical"] = 0.90

        out = blend_scores(spacy_res, deb_probs)
        self.assertTrue(out["deberta_used"])
        self.assertAlmostEqual(out["blended_scores"]["lexical"], 0.6 * 0.90 + 0.4 * 0.85, places=4)
        self.assertEqual(out["confidence_penalty"], 0.0)

    def test_blend_scores_disagreement_lowers_confidence(self):
        spacy_res = {lbl: {"score": 0.05, "span": "", "reason": ""} for lbl in LABELS}
        spacy_res["syntactic"]["score"] = 0.85
        deb_probs = {lbl: 0.05 for lbl in LABELS}
        deb_probs["lexical"] = 0.85

        out = blend_scores(spacy_res, deb_probs)
        self.assertGreater(out["detector_disagreement"], 0.5)
        self.assertGreater(out["confidence_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
