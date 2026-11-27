import io
import json
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from main import app


class TestFileUpload(unittest.TestCase):
    def test_upload_csv_json_jsonl_xlsx_parquet(self):
        client = TestClient(app)
        df = pd.DataFrame({"text": ["Where is my order ORD_5501?", "Cancel it before dispatch"]})

        xlsx_buf = io.BytesIO()
        df.to_excel(xlsx_buf, index=False, engine="openpyxl")

        parquet_buf = io.BytesIO()
        df.to_parquet(parquet_buf, index=False, engine="pyarrow")

        files = [
            ("sample.csv", df.to_csv(index=False).encode("utf-8"), "text/csv"),
            ("sample.json", json.dumps(df.to_dict(orient="records")).encode("utf-8"), "application/json"),
            ("sample.jsonl", "\n".join(json.dumps({"text": t}) for t in df["text"]).encode("utf-8"), "application/jsonl"),
            ("sample.xlsx", xlsx_buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("sample.parquet", parquet_buf.getvalue(), "application/octet-stream"),
        ]

        with patch.dict("os.environ", {"LLM_API_KEY": "", "GROQ_API_KEY": "", "GEMINI_API_KEY": ""}):
            for fname, content, mime in files:
                res = client.post("/process/file", files={"file": (fname, content, mime)})
                self.assertEqual(res.status_code, 200, f"{fname}: {res.text}")
                self.assertEqual(res.json()["count"], 2, fname)


if __name__ == "__main__":
    unittest.main()
