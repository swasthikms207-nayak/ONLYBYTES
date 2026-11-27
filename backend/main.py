from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from clarification import RECORD_STORE, apply_inline_clarification, apply_mcq_answers, generate_mcqs
from evaluator import run_evaluation
from pdf_export import build_evaluation_pdf, build_resolved_pdf
from resolver import ResolvedRecord, parse_csv_records, resolve_batch

app = FastAPI(title="Ambiguity Resolver API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    records: Optional[list[str]] = None
    raw_text: Optional[str] = None
    format: Literal["text", "csv"] = "text"
    # Context sources ─────────────────────────────────────────────────────────
    # Pasted text, chat history, or previous records (newline-separated).
    # Also accepted as a list of strings.
    context: Optional[str] = None
    context_records: Optional[list[str]] = None
    # Document-level metadata key-value pairs
    metadata: Optional[dict[str, Any]] = None


class ProcessResponse(BaseModel):
    count: int = Field(..., ge=0)
    results: list[ResolvedRecord]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process_records(request: ProcessRequest):
    """Process a batch of text records (pasted text, CSV, or list of strings).

    Context accepted via:
    - ``context``         – free-form pasted text / chat history / doc excerpt
    - ``context_records`` – explicit list of previous records
    - ``metadata``        – document-level key-value pairs
    """
    # ── Parse primary records ─────────────────────────────────────────────────
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

    # ── Merge explicit context sources ────────────────────────────────────────
    explicit_parts: list[str] = []
    if request.context:
        explicit_parts.append(request.context.strip())
    if request.context_records:
        explicit_parts.extend(r.strip() for r in request.context_records if r.strip())
    explicit_context = "\n".join(explicit_parts) if explicit_parts else None

    results = await resolve_batch(
        items,
        explicit_context=explicit_context,
        metadata=request.metadata,
    )
    return {"count": len(results), "results": results}


class ClarifyInlineRequest(BaseModel):
    record_id: str
    chosen_option: str
    custom_text: Optional[str] = None
    record: Optional[dict[str, Any]] = None


@app.post("/clarify")
def clarify_record(request: ClarifyInlineRequest):
    """Apply user's chosen clarification option to a record and return the updated record."""
    rec = RECORD_STORE.get(request.record_id) or request.record
    if not rec:
        raise HTTPException(status_code=404, detail=f"Record '{request.record_id}' not found.")
    return apply_inline_clarification(
        rec,
        chosen_option=request.chosen_option,
        custom_text=request.custom_text,
    )


class ClarifyMcqRequest(BaseModel):
    results: list[dict[str, Any]]
    max_questions: int = Field(default=5, ge=1, le=5)


class ClarifySubmitRequest(BaseModel):
    results: list[dict[str, Any]]
    answers: list[dict[str, Any]]


@app.post("/clarify/mcqs")
def get_clarification_mcqs(request: ClarifyMcqRequest):
    """Generate 4-5 MCQs related to the processed batch records."""
    return {"mcqs": generate_mcqs(request.results, max_questions=request.max_questions)}


@app.post("/clarify/submit")
def submit_clarification_mcqs(request: ClarifySubmitRequest):
    """Apply user MCQ answers to update resolutions and compute clarification evaluation scores."""
    return apply_mcq_answers(request.results, request.answers)


@app.get("/evaluate")
def evaluate_systems():
    """Run 3-system evaluation across the 40 gold-labeled sentences."""
    return run_evaluation()


class ExportResolvedRequest(BaseModel):
    results: list[dict[str, Any]]


@app.get("/export/evaluation.pdf")
def export_evaluation_pdf():
    """Download the final evaluation report as a PDF."""
    pdf_bytes = build_evaluation_pdf(run_evaluation())
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="evaluation_results.pdf"'},
    )


@app.post("/export/resolved.pdf")
def export_resolved_pdf(request: ExportResolvedRequest):
    """Download resolved text records as a PDF."""
    if not request.results:
        raise HTTPException(status_code=400, detail="No resolved records provided.")
    pdf_bytes = build_resolved_pdf(request.results)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="resolved_results.pdf"'},
    )
