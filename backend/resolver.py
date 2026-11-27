import asyncio
import csv
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV if ROOT_ENV.exists() else None)

AmbiguityType = Literal[
    "lexical",
    "syntactic",
    "referential",
    "scope",
    "pragmatic",
    "underspecified",
    "none",
]

RoutingStatus = Literal["AUTO_ACCEPT", "ACCEPT_WITH_ALTERNATIVES", "REVIEW"]

PLACEHOLDER_KEYS = {"", "your_llm_api_key_here", "sk-your-key-here"}
SELF_CONSISTENCY_SAMPLES = 5


class AlternativeReading(BaseModel):
    reading: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ResolvedRecord(BaseModel):
    text: str
    ambiguity_type: AmbiguityType
    top_reading: str
    alternatives: list[AlternativeReading]
    confidence: float = Field(..., ge=0.0, le=1.0)
    disagreement: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict: bool = False
    status: RoutingStatus


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


def route_record(confidence: float, conflict: bool) -> RoutingStatus:
    """Route record based on confidence and sentence-vs-context conflict."""
    if conflict or confidence < 0.5:
        return "REVIEW"
    if confidence <= 0.8:
        return "ACCEPT_WITH_ALTERNATIVES"
    return "AUTO_ACCEPT"


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _build_llm_system_prompt() -> str:
    return (
        "You are a linguistic ambiguity analyzer.\n"
        "For the given target sentence (and optional surrounding context records):\n"
        "1. Detect ambiguity_type: one of lexical, syntactic, referential, scope, pragmatic, underspecified, or none.\n"
        "2. List candidate readings with probabilities summing to 1.0.\n"
        "3. Pick top_reading using the sentence plus surrounding records (if provided).\n"
        "If there is no ambiguity, return ambiguity_type: 'none', top_reading equal to the original sentence untouched, "
        "and alternatives: [].\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "ambiguity_type": "lexical|syntactic|referential|scope|pragmatic|underspecified|none",\n'
        '  "top_reading": "<string>",\n'
        '  "alternatives": [{"reading": "<string>", "probability": <float 0..1>}],\n'
        '  "top_probability": <float 0..1>\n'
        "}"
    )


async def _call_llm_once(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    model: str,
    text: str,
    context_records: list[str],
    temperature: float,
) -> dict[str, Any]:
    user_payload: dict[str, Any] = {"sentence": text}
    if context_records:
        user_payload["surrounding_context"] = context_records

    response = await client.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _build_llm_system_prompt()},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(_strip_code_fences(content))


async def _resolve_single_llm(
    text: str,
    surrounding_context: list[str],
    api_key: str,
    base_url: str,
    model: str,
    n_samples: int = SELF_CONSISTENCY_SAMPLES,
) -> ResolvedRecord:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Score sentence alone
        alone_task = _call_llm_once(
            client, api_key, base_url, model, text, [], temperature=0.0
        )
        # 2. Sample LLM 5 times with surrounding context for self-consistency
        context_tasks = [
            _call_llm_once(
                client,
                api_key,
                base_url,
                model,
                text,
                surrounding_context,
                temperature=0.7 if i > 0 else 0.0,
            )
            for i in range(n_samples)
        ]
        alone_res, *context_samples = await asyncio.gather(alone_task, *context_tasks)

    # Measure self-consistency disagreement across the 5 context samples
    top_readings = [
        str(s.get("top_reading", text)).strip() or text for s in context_samples
    ]
    counts = Counter(r.lower() for r in top_readings)
    most_common_lower, top_count = counts.most_common(1)[0]
    agreement = top_count / len(context_samples)
    disagreement = round(1.0 - agreement, 4)

    # Pick representative sample matching the consensus top_reading
    chosen = next(
        (s for s in context_samples if str(s.get("top_reading", "")).strip().lower() == most_common_lower),
        context_samples[0],
    )

    amb_type = str(chosen.get("ambiguity_type", "none")).lower()
    if amb_type not in {"lexical", "syntactic", "referential", "scope", "pragmatic", "underspecified", "none"}:
        amb_type = "underspecified"

    if amb_type == "none":
        top_reading = text
        alternatives: list[AlternativeReading] = []
    else:
        top_reading = str(chosen.get("top_reading", text)).strip() or text
        alternatives = [
            AlternativeReading(
                reading=str(alt.get("reading", "")).strip(),
                probability=max(0.0, min(1.0, float(alt.get("probability", 0.0)))),
            )
            for alt in chosen.get("alternatives", [])
            if str(alt.get("reading", "")).strip()
        ]

    # Conflict if sentence-alone top reading disagrees with context top reading
    alone_top = str(alone_res.get("top_reading", text)).strip().lower()
    conflict = alone_top != top_reading.strip().lower() and amb_type != "none"

    raw_probs = [float(s.get("top_probability", 0.7)) for s in context_samples]
    avg_prob = sum(max(0.0, min(1.0, p)) for p in raw_probs) / len(raw_probs)
    confidence = round(max(0.0, min(1.0, avg_prob * agreement)), 4)

    status = route_record(confidence=confidence, conflict=conflict)

    return ResolvedRecord(
        text=text,
        ambiguity_type=amb_type,  # type: ignore[arg-type]
        top_reading=top_reading,
        alternatives=alternatives,
        confidence=confidence,
        disagreement=disagreement,
        conflict=conflict,
        status=status,
    )


def _resolve_single_mock(text: str, surrounding_context: list[str]) -> ResolvedRecord:
    """Deterministic mock fallback when LLM_API_KEY is not configured."""
    lower = text.lower()
    has_context = len(surrounding_context) > 0

    if any(k in lower for k in ("sab", "all", "everything", "both", "saare")):
        conf = 0.48
        return ResolvedRecord(
            text=text,
            ambiguity_type="scope",
            top_reading="Apply the action to all active items on the account.",
            alternatives=[
                AlternativeReading(reading="Apply only to the most recent item.", probability=0.32),
                AlternativeReading(reading="Apply across both orders and payment methods.", probability=0.20),
            ],
            confidence=conf,
            disagreement=0.40,
            conflict=False,
            status=route_record(conf, False),
        )

    if any(w in lower.split() for w in ("it", "that", "this", "isko", "uska", "woh")):
        conf = 0.68
        conflict = has_context
        return ResolvedRecord(
            text=text,
            ambiguity_type="referential",
            top_reading="Refers to the item mentioned in surrounding context." if has_context else "Refers to an unspecified recent item.",
            alternatives=[
                AlternativeReading(reading="Refers to an earlier completed order.", probability=0.20),
                AlternativeReading(reading="Refers to the saved billing profile.", probability=0.12),
            ],
            confidence=conf,
            disagreement=0.20,
            conflict=conflict,
            status=route_record(conf, conflict),
        )

    if any(k in lower for k in ("band", "close", "charge", "bank", "freeze", "cancel")):
        if any(k in lower for k in ("card", "order", "account", "ord_", "txn_")):
            conf = 0.94
            return ResolvedRecord(
                text=text,
                ambiguity_type="none",
                top_reading=text,
                alternatives=[],
                confidence=conf,
                disagreement=0.0,
                conflict=False,
                status=route_record(conf, False),
            )
        conf = 0.64
        return ResolvedRecord(
            text=text,
            ambiguity_type="lexical",
            top_reading="Block or freeze the active payment card.",
            alternatives=[
                AlternativeReading(reading="Permanently close the user account.", probability=0.25),
                AlternativeReading(reading="Cancel the latest pending order.", probability=0.11),
            ],
            confidence=conf,
            disagreement=0.20,
            conflict=False,
            status=route_record(conf, False),
        )

    if len(text.split()) <= 3:
        conf = 0.45
        return ResolvedRecord(
            text=text,
            ambiguity_type="underspecified",
            top_reading=f"Request for status or assistance regarding '{text}'.",
            alternatives=[
                AlternativeReading(reading="Request to modify or cancel an active item.", probability=0.35),
                AlternativeReading(reading="General support inquiry.", probability=0.20),
            ],
            confidence=conf,
            disagreement=0.40,
            conflict=False,
            status=route_record(conf, False),
        )

    conf = 0.92
    return ResolvedRecord(
        text=text,
        ambiguity_type="none",
        top_reading=text,
        alternatives=[],
        confidence=conf,
        disagreement=0.0,
        conflict=False,
        status=route_record(conf, False),
    )


async def resolve_batch(records: list[str]) -> list[dict[str, Any]]:
    """Resolve a batch of text records using LLM self-consistency & conflict detection (or mock fallback)."""
    cleaned_records = [r.strip() for r in records if r and r.strip()]
    if not cleaned_records:
        return []

    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    use_llm = bool(api_key and api_key not in PLACEHOLDER_KEYS)

    results: list[dict[str, Any]] = []
    for idx, text in enumerate(cleaned_records):
        surrounding = [
            r for i, r in enumerate(cleaned_records) if i != idx and abs(i - idx) <= 2
        ]
        if use_llm:
            resolved = await _resolve_single_llm(
                text=text,
                surrounding_context=surrounding,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
        else:
            resolved = _resolve_single_mock(text=text, surrounding_context=surrounding)
        results.append(resolved.model_dump())

    return results
