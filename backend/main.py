from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from evaluator import run_evaluation
from resolver import ResolvedRecord, parse_csv_records, resolve_batch

app = FastAPI(title="Ambiguity Resolver API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    records: Optional[list[str]] = None
    raw_text: Optional[str] = None
    format: Literal["text", "csv"] = "text"


class ProcessResponse(BaseModel):
    count: int = Field(..., ge=0)
    results: list[ResolvedRecord]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process_records(request: ProcessRequest):
    """Process a batch of text records (from pasted text, CSV string, or records list)."""
    items: list[str] = []
    if request.records:
        items = [r.strip() for r in request.records if r and r.strip()]
    elif request.raw_text and request.raw_text.strip():
        if request.format == "csv":
            items = parse_csv_records(request.raw_text)
        else:
            items = [line.strip() for line in request.raw_text.splitlines() if line.strip()]

    if not items:
        raise HTTPException(status_code=400, detail="No valid text records provided.")

    results = await resolve_batch(items)
    return {"count": len(results), "results": results}


@app.get("/evaluate")
def evaluate_systems():
    """Run 3-system evaluation across the 40 gold-labeled sentences."""
    return run_evaluation()
