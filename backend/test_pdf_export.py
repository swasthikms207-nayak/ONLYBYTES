import unittest

from fastapi.testclient import TestClient

from main import app


class TestPdfExport(unittest.TestCase):
    def test_evaluation_pdf_download(self):
        client = TestClient(app)
        res = client.get("/export/evaluation.pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/pdf")
        self.assertTrue(res.content.startswith(b"%PDF"))

    def test_resolved_pdf_download(self):
        client = TestClient(app)
        sample = [
            {
                "text": "Cancel it before dispatch",
                "resolved_text": "Cancel headphones ORD_5519 before dispatch",
                "ambiguity_type": "referential",
                "top_reading": "Refers to headphones ORD_5519.",
                "alternatives": [{"reading": "Cancel billing profile", "probability": 0.2}],
                "confidence": 0.68,
                "conflict": True,
                "assumed": False,
                "status": "REVIEW",
                "changes": [{"span": "it", "before": "it", "after": "ORD_5519", "reason": "Replaced pronoun"}],
                "context_used": "pasted context text",
            }
        ]
        res = client.post("/export/resolved.pdf", json={"results": sample})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/pdf")
        self.assertTrue(res.content.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
