import csv
import io
from typing import Literal

from pydantic import BaseModel, Field

AmbiguityType = Literal[
    "lexical",
    "referential",
    "scope",
    "syntactic",
    "underspecified",
    "none",
]

StatusType = Literal["resolved", "ambiguous", "needs_review"]


class AlternativeReading(BaseModel):
    reading: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ResolvedRecord(BaseModel):
    text: str
    ambiguity_type: AmbiguityType
    top_reading: str
    alternatives: list[AlternativeReading]
    confidence: float = Field(..., ge=0.0, le=1.0)
    status: StatusType


def parse_csv_records(csv_text: str) -> list[str]:
    """Extract text records from a CSV string (supports 'text'/'query' header or first column)."""
    reader = csv.reader(io.StringIO(csv_text.strip()))
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        return []

    header = [col.strip().lower() for col in rows[0]]
    col_idx = 0
    start_row = 0
    for candidate in ("text", "query", "record", "sentence", "input"):
        if candidate in header:
            col_idx = header.index(candidate)
            start_row = 1
            break

    return [
        row[col_idx].strip()
        for row in rows[start_row:]
        if len(row) > col_idx and row[col_idx].strip()
    ]


def _resolve_single_mock(text: str) -> ResolvedRecord:
    """Deterministic mock ambiguity resolver for text records."""
    lower = text.lower()

    if any(k in lower for k in ("sab", "all", "everything", "both", "saare")):
        return ResolvedRecord(
            text=text,
            ambiguity_type="scope",
            top_reading="Apply the action to all active items on the account.",
            alternatives=[
                AlternativeReading(
                    reading="Apply the action only to the most recent item.",
                    probability=0.31,
                ),
                AlternativeReading(
                    reading="Apply the action across both orders and payment methods.",
                    probability=0.17,
                ),
            ],
            confidence=0.52,
            status="needs_review",
        )

    if any(w in lower.split() for w in ("it", "that", "this", "isko", "uska", "woh")):
        return ResolvedRecord(
            text=text,
            ambiguity_type="referential",
            top_reading="Refers to the most recent pending transaction or order.",
            alternatives=[
                AlternativeReading(
                    reading="Refers to an earlier completed order from chat history.",
                    probability=0.24,
                ),
                AlternativeReading(
                    reading="Refers to the saved billing profile.",
                    probability=0.12,
                ),
            ],
            confidence=0.64,
            status="ambiguous",
        )

    if any(k in lower for k in ("band", "close", "charge", "bank", "freeze", "cancel")):
        if any(k in lower for k in ("card", "order", "account", "ord_", "txn_")):
            return ResolvedRecord(
                text=text,
                ambiguity_type="none",
                top_reading=f"Clear request: {text}",
                alternatives=[],
                confidence=0.94,
                status="resolved",
            )
        return ResolvedRecord(
            text=text,
            ambiguity_type="lexical",
            top_reading="Block or freeze the active payment card.",
            alternatives=[
                AlternativeReading(
                    reading="Permanently close the user account.",
                    probability=0.28,
                ),
                AlternativeReading(
                    reading="Cancel the latest pending order.",
                    probability=0.11,
                ),
            ],
            confidence=0.61,
            status="ambiguous",
        )

    if len(text.split()) <= 3:
        return ResolvedRecord(
            text=text,
            ambiguity_type="underspecified",
            top_reading=f"Request for status or assistance regarding '{text}'.",
            alternatives=[
                AlternativeReading(
                    reading="Request to modify or cancel an active item.",
                    probability=0.35,
                ),
                AlternativeReading(
                    reading="General support inquiry.",
                    probability=0.16,
                ),
            ],
            confidence=0.49,
            status="needs_review",
        )

    return ResolvedRecord(
        text=text,
        ambiguity_type="none",
        top_reading=f"Unambiguous intent: {text}",
        alternatives=[],
        confidence=0.92,
        status="resolved",
    )


def resolve_batch(records: list[str]) -> list[dict]:
    """Resolve a batch of text records using mock ambiguity analysis."""
    results = []
    for raw in records:
        cleaned = raw.strip()
        if not cleaned:
            continue
        results.append(_resolve_single_mock(cleaned).model_dump())
    return results
