import unittest

from fastapi.testclient import TestClient

from main import app
from resolver import parse_csv_records, resolve_batch


class TestResolver(unittest.TestCase):
    def test_parse_csv_with_header(self):
        csv_data = 'text,note\n"Block my card",high\n"Sab band karo",ambiguous\n'
        self.assertEqual(parse_csv_records(csv_data), ["Block my card", "Sab band karo"])

    def test_resolve_batch_schema(self):
        out = resolve_batch([
            "Where is my order ORD_5501?",
            "Sab band kar do",
            "Cancel it now",
        ])
        self.assertEqual(len(out), 3)
        for item in out:
            self.assertIn("ambiguity_type", item)
            self.assertIn("top_reading", item)
            self.assertIn("alternatives", item)
            self.assertIn("confidence", item)
            self.assertIn("status", item)
        self.assertEqual(out[0]["status"], "resolved")
        self.assertEqual(out[1]["ambiguity_type"], "scope")
        self.assertEqual(out[2]["ambiguity_type"], "referential")

    def test_process_endpoint_text_and_csv(self):
        client = TestClient(app)
        res = client.post(
            "/process",
            json={"raw_text": "Block my card\nSab band karo", "format": "text"},
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
