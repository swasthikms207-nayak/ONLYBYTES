import unittest

from detectors import detect_all


class TestDetectors(unittest.TestCase):
    def test_all_11_keys_present(self):
        res = detect_all("I saw the man with the telescope")
        expected = {
            "lexical", "syntactic", "referential", "scope", "semantic",
            "pragmatic", "underspecified", "ellipsis", "temporal_numeric",
            "cultural_domain", "none",
        }
        self.assertEqual(set(res.keys()), expected)
        for key, val in res.items():
            self.assertGreaterEqual(val["score"], 0.0, key)
            self.assertLessEqual(val["score"], 1.0, key)
            self.assertIn("span", val)
            self.assertIn("reason", val)

    def test_syntactic_pp_coordination_and_ing(self):
        self.assertGreaterEqual(detect_all("I saw the man with the telescope")["syntactic"]["score"], 0.7)
        self.assertGreaterEqual(detect_all("Old men and women sat outside")["syntactic"]["score"], 0.7)
        self.assertGreaterEqual(detect_all("Flying planes can be dangerous")["syntactic"]["score"], 0.7)

    def test_referential_two_antecedents(self):
        res = detect_all("John told Bill that he would come")
        self.assertGreaterEqual(res["referential"]["score"], 0.7)
        self.assertEqual(res["referential"]["span"].lower(), "he")

    def test_scope_quantifiers_and_negation(self):
        self.assertGreaterEqual(detect_all("Every student read a book")["scope"]["score"], 0.7)
        self.assertGreaterEqual(detect_all("He did not leave because he was angry")["scope"]["score"], 0.7)

    def test_semantic_plural_action(self):
        res = detect_all("The students carried the piano")
        self.assertGreaterEqual(res["semantic"]["score"], 0.65)

    def test_pragmatic(self):
        self.assertGreaterEqual(detect_all("Can you pass the salt?")["pragmatic"]["score"], 0.7)
        self.assertGreaterEqual(detect_all("It's cold in here")["pragmatic"]["score"], 0.65)

    def test_underspecified(self):
        self.assertGreaterEqual(detect_all("Cancel please")["underspecified"]["score"], 0.65)
        self.assertGreaterEqual(detect_all("Send the report soon")["underspecified"]["score"], 0.6)

    def test_ellipsis(self):
        self.assertGreaterEqual(detect_all("Priya likes Ravi more than Meera")["ellipsis"]["score"], 0.75)
        self.assertGreaterEqual(detect_all("Alice can swim and so can Bob")["ellipsis"]["score"], 0.75)

    def test_temporal_numeric(self):
        self.assertGreaterEqual(detect_all("05/06/2026")["temporal_numeric"]["score"], 0.8)
        self.assertGreaterEqual(detect_all("Meet me next Monday")["temporal_numeric"]["score"], 0.7)
        self.assertGreaterEqual(detect_all("Meet me at 5")["temporal_numeric"]["score"], 0.7)

    def test_cultural_domain(self):
        self.assertGreaterEqual(detect_all("Table the motion")["cultural_domain"]["score"], 0.7)
        self.assertGreaterEqual(detect_all("Check the kernel settings")["cultural_domain"]["score"], 0.7)

    def test_unambiguous_scores_low_on_all_types(self):
        for sent in [
            "Water boils at 100 degrees Celsius.",
            "The cat sat on the mat.",
            "Track shipment for order ORD_5490.",
        ]:
            res = detect_all(sent)
            for k, v in res.items():
                if k == "none":
                    self.assertGreaterEqual(v["score"], 0.65, f"{sent} none={v}")
                else:
                    self.assertLess(v["score"], 0.35, f"{sent} {k}={v}")


if __name__ == "__main__":
    unittest.main()
