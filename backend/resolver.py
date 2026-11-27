"""
resolver.py – Ambiguity Resolver backend logic.

Context sources accepted:
  • surrounding_context  – nearby records from the same batch (auto-derived)
  • explicit_context     – pasted text / previous messages / document metadata
    passed explicitly by the caller

Conflict detection:
  • sentence-alone score  (temperature=0)
  • context score (n_samples samples at higher temperatures for self-consistency)
  • conflict = True  when  alone-top-reading ≠ context-consensus-top-reading
                            AND  the two probability distributions differ by > CONFLICT_THRESHOLD

Confidence calibration:
  • raw_confidence = avg_top_probability × agreement_rate
  • temperature_penalty  = mean_sample_temperature / TEMP_SCALE_FACTOR
  • calibrated_confidence = raw_confidence × (1 − temperature_penalty)

Routing:
  >0.8              → AUTO_ACCEPT
  0.5–0.8           → ACCEPT_WITH_ALTERNATIVES
  <0.5 or conflict  → REVIEW

Original text and all alternatives are always preserved in the output.
"""

import asyncio
import csv
import io
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from classifier import blend_scores, predict_deberta
from clarification import RECORD_STORE, build_inline_clarification
from detectors import detect_all
from rewrite import rewrite_sentence

ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV if ROOT_ENV.exists() else None)

AmbiguityType = Literal[
    "lexical",
    "syntactic",
    "referential",
    "scope",
    "semantic",
    "pragmatic",
    "underspecified",
    "ellipsis",
    "temporal_numeric",
    "cultural_domain",
    "none",
]

VALID_TYPES = {
    "lexical",
    "syntactic",
    "referential",
    "scope",
    "semantic",
    "pragmatic",
    "underspecified",
    "ellipsis",
    "temporal_numeric",
    "cultural_domain",
    "none",
}

RoutingStatus = Literal["AUTO_ACCEPT", "ACCEPT_WITH_ALTERNATIVES", "REVIEW"]

PLACEHOLDER_KEYS = {"", "your_llm_api_key_here", "your_groq_api_key_here", "your_gemini_api_key_here", "sk-your-key-here"}
SELF_CONSISTENCY_SAMPLES = 5
CONFLICT_THRESHOLD = 0.3   # min probability difference to call a conflict
TEMP_SCALE_FACTOR = 2.0    # divides mean sample temperature for penalty


class AlternativeReading(BaseModel):
    reading: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ResolutionStep(BaseModel):
    step: int
    action: str
    result: Any


class RewriteChange(BaseModel):
    span: str
    before: str
    after: str
    reason: str


class ResolvedRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    ambiguity_type: AmbiguityType
    detected_types: list[str] = Field(default_factory=list)
    top_reading: str
    resolved_text: str = ""
    assumed: bool = False
    changes: list[RewriteChange] = Field(default_factory=list)
    alternatives: list[AlternativeReading]
    confidence: float = Field(..., ge=0.0, le=1.0)
    disagreement: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict: bool = False
    conflict_reason: Optional[str] = None
    context_used: Optional[str] = None      # one-line evidence summary
    status: RoutingStatus
    clarification: Optional[dict[str, Any]] = None
    clarified_by_user: bool = False
    resolution_steps: list[ResolutionStep] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------

def parse_csv_records(csv_text: str) -> list[str]:
    """Extract text records from a CSV string (supports 'text'/'query' header or first column)."""
    reader = csv.reader(io.StringIO(csv_text.strip()))
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        return []
    header = [col.strip().lower() for col in rows[0]]
    col_idx, start_row = 0, 0
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


def build_context_list(
    *,
    surrounding: list[str],
    explicit: Optional[str],
    metadata: Optional[dict],
) -> tuple[list[str], str]:
    """
    Merge all context sources into a flat list of strings and produce a
    human-readable one-line summary of which sources were used.
    """
    parts: list[str] = []
    used: list[str] = []

    if surrounding:
        parts.extend(surrounding)
        used.append(f"{len(surrounding)} surrounding record(s)")

    if explicit and explicit.strip():
        # Split by newlines; each non-empty line becomes a context item
        lines = [l.strip() for l in explicit.splitlines() if l.strip()]
        parts.extend(lines)
        used.append("pasted context text")

    if metadata:
        for k, v in metadata.items():
            parts.append(f"{k}: {v}")
        used.append("document metadata")

    summary = "Context used: " + ", ".join(used) if used else "No external context"
    return parts, summary


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_record(confidence: float, conflict: bool) -> RoutingStatus:
    if conflict or confidence < 0.5:
        return "REVIEW"
    if confidence <= 0.8:
        return "ACCEPT_WITH_ALTERNATIVES"
    return "AUTO_ACCEPT"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()
    return text


def _build_llm_system_prompt() -> str:
    return (
        "You are a linguistic ambiguity analyzer.\n"
        "For the given target sentence (optional surrounding context, and optional linguistic detector hints):\n"
        "1. Detect ambiguity_type: lexical | syntactic | referential | scope | semantic | pragmatic | underspecified | ellipsis | temporal_numeric | cultural_domain | none.\n"
        "   - Watch carefully for PP-attachment ('I saw the man with the telescope'), quantifier/negation scope ('Every student read a book'), "
        "pronouns with multiple antecedents, comparative ellipsis ('more than Meera'), ambiguous dates/times/amounts ('05/06/2026', 'next Monday', 'at 5'), "
        "domain homonyms ('Table the motion', 'bank', 'kernel'), and indirect requests.\n"
        "   - If detected_ambiguity_hints are provided, evaluate them carefully; only return 'none' if the sentence truly has a single unambiguous reading.\n"
        "2. List ALL candidate readings (2-4) with probabilities that sum to 1.0.\n"
        "3. Pick top_reading using the sentence PLUS any provided context (if present). "
        "When ambiguity_type is NOT 'none', top_reading MUST be a clear, disambiguated paraphrase of the intended meaning (never repeat the ambiguous sentence verbatim).\n"
        "4. Provide top_probability: the probability you assign to top_reading.\n"
        "If there is no ambiguity, return ambiguity_type='none', top_reading=original sentence, alternatives=[].\n"
        "NEVER ask the user questions. Return STRICT JSON only — no markdown fences:\n"
        "{\n"
        '  "ambiguity_type": "<type>",\n'
        '  "top_reading": "<string>",\n'
        '  "alternatives": [{"reading": "<string>", "probability": <float>}],\n'
        '  "top_probability": <float>\n'
        "}"
    )


async def _call_llm_once(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    model: str,
    text: str,
    context_items: list[str],
    temperature: float,
    detected_hints: Optional[list[str]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"sentence": text}
    if context_items:
        payload["context"] = context_items
    if detected_hints:
        payload["detected_ambiguity_hints"] = detected_hints

    response = await client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _build_llm_system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(_strip_code_fences(content))


# ---------------------------------------------------------------------------
# Core resolution logic (LLM path)
# ---------------------------------------------------------------------------

def _build_steps(
    *,
    detected_types: list[str],
    top_reading: str,
    alternatives: list[AlternativeReading],
    alone_top: str,
    alone_prob: float,
    ctx_top: str,
    ctx_prob: float,
    conflict: bool,
    conflict_reason: Optional[str],
    n_samples: int,
    agreement: float,
    disagreement: float,
    raw_conf: float,
    confidence: float,
    status: RoutingStatus,
) -> list[ResolutionStep]:
    """Record each of the 7 resolution steps as {step, action, result}."""
    return [
        ResolutionStep(
            step=1,
            action="list_candidate_readings",
            result={
                "detected_types": detected_types,
                "candidates": [a.model_dump() for a in alternatives],
            },
        ),
        ResolutionStep(
            step=2,
            action="score_sentence_alone",
            result={"top_reading": alone_top, "probability": round(alone_prob, 4)},
        ),
        ResolutionStep(
            step=3,
            action="score_with_context",
            result={"top_reading": ctx_top, "probability": round(ctx_prob, 4)},
        ),
        ResolutionStep(
            step=4,
            action="check_ranking_conflict",
            result={"conflict": conflict, "reason": conflict_reason},
        ),
        ResolutionStep(
            step=5,
            action="self_consistency_sample",
            result={
                "samples": n_samples,
                "agreement": round(agreement, 4),
                "disagreement": disagreement,
            },
        ),
        ResolutionStep(
            step=6,
            action="calibrate_temperature_scaling",
            result={"raw_confidence": round(raw_conf, 4), "calibrated_confidence": confidence},
        ),
        ResolutionStep(
            step=7,
            action="route_record",
            result={"status": status, "top_reading": top_reading},
        ),
    ]


async def _resolve_single_llm(
    text: str,
    context_items: list[str],
    context_summary: str,
    api_key: str,
    base_url: str,
    model: str,
    n_samples: int = SELF_CONSISTENCY_SAMPLES,
    detected_types: Optional[list[str]] = None,
    confidence_penalty: float = 0.0,
    strong_spacy_types: Optional[list[str]] = None,
) -> ResolvedRecord:
    """
    1. Score sentence alone (temp=0, no context).
    2. Score sentence with full context, n_samples times.
       Temperatures: [0.0, 0.3, 0.5, 0.7, 0.9] for calibration diversity.
    3. Detect conflict if alone-top ≠ context-consensus AND probability gap > CONFLICT_THRESHOLD.
    4. Calibrate confidence with temperature scaling.
    """
    sample_temps = [0.0, 0.3, 0.5, 0.7, 0.9][:n_samples]
    hints = [t for t in (detected_types or []) if t != "none"] or None

    async with httpx.AsyncClient(timeout=30.0) as client:
        alone_task = _call_llm_once(client, api_key, base_url, model, text, [], temperature=0.0, detected_hints=hints)
        context_tasks = [
            _call_llm_once(client, api_key, base_url, model, text, context_items, temperature=t, detected_hints=hints)
            for t in sample_temps
        ]
        alone_res, *ctx_samples = await asyncio.gather(alone_task, *context_tasks)

    # ── Self-consistency across context samples ──────────────────────────────
    top_readings_ctx = [str(s.get("top_reading", text)).strip() or text for s in ctx_samples]
    counts = Counter(r.lower() for r in top_readings_ctx)
    most_common_lower, top_count = counts.most_common(1)[0]
    agreement = top_count / len(ctx_samples)
    disagreement = round(1.0 - agreement, 4)

    # Representative context sample
    chosen = next(
        (s for s in ctx_samples if str(s.get("top_reading", "")).strip().lower() == most_common_lower),
        ctx_samples[0],
    )

    amb_type = str(chosen.get("ambiguity_type", "none")).lower()
    if amb_type not in VALID_TYPES:
        amb_type = "underspecified"

    # If high-precision spaCy linguistic detector found a strong ambiguity (>= 0.70)
    # that the LLM missed ('none'), honor the spaCy linguistic detection.
    if amb_type == "none" and strong_spacy_types:
        amb_type = strong_spacy_types[0]

    # ── Build top_reading and alternatives (always preserved) ─────────────────
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
        if not alternatives:
            alternatives = [
                AlternativeReading(
                    reading=f"Alternative {amb_type} interpretation of: {text}",
                    probability=0.35,
                )
            ]

    # ── Conflict detection ───────────────────────────────────────────────────
    alone_top_raw = str(alone_res.get("top_reading", text)).strip() or text
    alone_top = alone_top_raw.lower()
    alone_prob = float(alone_res.get("top_probability", 0.5))
    ctx_prob = float(chosen.get("top_probability", 0.5))
    if alone_top == top_reading.strip().lower():
        prob_gap = 0.0
    else:
        alone_alts = {
            str(a.get("reading", "")).strip().lower(): float(a.get("probability", 0.0))
            for a in alone_res.get("alternatives", [])
        }
        alone_prob_for_ctx = alone_alts.get(top_reading.strip().lower(), 0.0)
        prob_gap = abs(ctx_prob - alone_prob_for_ctx)

    conflict = (
        alone_top != top_reading.strip().lower()
        and amb_type != "none"
        and prob_gap > CONFLICT_THRESHOLD
    )
    conflict_reason: Optional[str] = None
    if conflict:
        conflict_reason = (
            f"Sentence-alone reading: '{alone_res.get('top_reading', '')}' "
            f"(p={alone_prob:.2f}) differs from context reading: '{top_reading}' "
            f"(p={ctx_prob:.2f}); gap={prob_gap:.2f})"
        )

    # ── Confidence calibration with temperature scaling ──────────────────────
    raw_probs = [float(s.get("top_probability", 0.7)) for s in ctx_samples]
    avg_prob = sum(max(0.0, min(1.0, p)) for p in raw_probs) / len(raw_probs)
    mean_temp = sum(sample_temps) / len(sample_temps)
    cal_temp = 1.0 + (mean_temp / TEMP_SCALE_FACTOR)
    raw_conf = max(1e-6, min(1.0 - 1e-6, avg_prob * agreement))
    logit = math.log(raw_conf / (1.0 - raw_conf))
    cal_conf = 1.0 / (1.0 + math.exp(-logit / cal_temp))
    effective_penalty = 0.0 if (amb_type == "none" and (not detected_types or detected_types == ["none"])) else confidence_penalty
    confidence = round(max(0.0, min(1.0, cal_conf - effective_penalty)), 4)
    if str(chosen.get("ambiguity_type", "none")).lower() == "none" and amb_type != "none":
        confidence = min(confidence, 0.70)

    status = route_record(confidence=confidence, conflict=conflict)
    det_types = detected_types if detected_types is not None else ([amb_type] if amb_type != "none" else ["none"])
    steps = _build_steps(
        detected_types=det_types,
        top_reading=top_reading,
        alternatives=alternatives,
        alone_top=alone_top_raw,
        alone_prob=alone_prob,
        ctx_top=top_reading,
        ctx_prob=ctx_prob,
        conflict=conflict,
        conflict_reason=conflict_reason,
        n_samples=len(ctx_samples),
        agreement=agreement,
        disagreement=disagreement,
        raw_conf=raw_conf,
        confidence=confidence,
        status=status,
    )

    rw = rewrite_sentence(
        text=text,
        ambiguity_type=amb_type,
        top_reading=top_reading,
        alternatives=[a.model_dump() for a in alternatives],
        confidence=confidence,
        context_items=context_items,
    )

    clarification = build_inline_clarification(
        text=text,
        ambiguity_type=amb_type,
        status=status,
        top_reading=top_reading,
        alternatives=[a.model_dump() for a in alternatives],
    )

    return ResolvedRecord(
        text=text,
        ambiguity_type=amb_type,  # type: ignore[arg-type]
        detected_types=det_types,
        top_reading=top_reading,
        resolved_text=rw["resolved_text"],
        assumed=rw["assumed"],
        changes=[RewriteChange(**c) for c in rw["changes"]],
        alternatives=alternatives,
        confidence=confidence,
        disagreement=disagreement,
        conflict=conflict,
        conflict_reason=conflict_reason,
        context_used=context_summary,
        status=status,
        clarification=clarification,
        clarified_by_user=False,
        resolution_steps=steps,
    )


# ---------------------------------------------------------------------------
# Mock fallback (no API key)
# ---------------------------------------------------------------------------

def _resolve_single_mock(
    text: str,
    context_items: list[str],
    context_summary: str,
    detected_types: Optional[list[str]] = None,
    confidence_penalty: float = 0.0,
    strong_spacy_types: Optional[list[str]] = None,
) -> ResolvedRecord:
    """Deterministic mock used when LLM_API_KEY is absent or placeholder."""
    lower = text.lower()
    has_context = bool(context_items)

    def _finish(
        amb_type: AmbiguityType,
        top_reading: str,
        alternatives: list[AlternativeReading],
        conf: float,
        disagreement: float,
        conflict: bool,
        conflict_reason: Optional[str] = None,
    ) -> ResolvedRecord:
        adj_conf = round(max(0.0, min(1.0, conf - confidence_penalty)), 4)
        status = route_record(adj_conf, conflict)
        det_types = detected_types if detected_types is not None else ([amb_type] if amb_type != "none" else ["none"])
        steps = _build_steps(
            detected_types=det_types,
            top_reading=top_reading,
            alternatives=alternatives,
            alone_top=top_reading if not conflict else (alternatives[0].reading if alternatives else text),
            alone_prob=adj_conf,
            ctx_top=top_reading,
            ctx_prob=adj_conf,
            conflict=conflict,
            conflict_reason=conflict_reason,
            n_samples=SELF_CONSISTENCY_SAMPLES,
            agreement=1.0 - disagreement,
            disagreement=disagreement,
            raw_conf=conf,
            confidence=adj_conf,
            status=status,
        )
        rw = rewrite_sentence(
            text=text,
            ambiguity_type=amb_type,
            top_reading=top_reading,
            alternatives=[a.model_dump() for a in alternatives],
            confidence=adj_conf,
            context_items=context_items,
        )
        clarification = build_inline_clarification(
            text=text,
            ambiguity_type=amb_type,
            status=status,
            top_reading=top_reading,
            alternatives=[a.model_dump() for a in alternatives],
        )
        return ResolvedRecord(
            text=text,
            ambiguity_type=amb_type,
            detected_types=det_types,
            top_reading=top_reading,
            resolved_text=rw["resolved_text"],
            assumed=rw["assumed"],
            changes=[RewriteChange(**c) for c in rw["changes"]],
            alternatives=alternatives,
            confidence=adj_conf,
            disagreement=disagreement,
            conflict=conflict,
            conflict_reason=conflict_reason,
            context_used=context_summary,
            status=status,
            clarification=clarification,
            clarified_by_user=False,
            resolution_steps=steps,
        )

    if any(k in lower for k in ("all", "everything", "both")):
        return _finish(
            "scope",
            "Apply the action to all active items on the account.",
            [
                AlternativeReading(reading="Apply only to the most recent item.", probability=0.32),
                AlternativeReading(reading="Apply across both orders and payment methods.", probability=0.20),
            ],
            0.48,
            0.40,
            has_context,
            "Scope interpretation changes with context." if has_context else None,
        )

    if any(w in lower.split() for w in ("it", "that", "this")):
        return _finish(
            "referential",
            (
                "Refers to the item mentioned in surrounding context."
                if has_context
                else "Refers to an unspecified recent item."
            ),
            [
                AlternativeReading(reading="Refers to an earlier completed order.", probability=0.20),
                AlternativeReading(reading="Refers to the saved billing profile.", probability=0.12),
            ],
            0.68,
            0.20,
            has_context,
            "Referent changes with context." if has_context else None,
        )

    if any(k in lower for k in ("close", "charge", "bank", "freeze", "cancel")):
        if any(k in lower for k in ("card", "order", "account", "ord_", "txn_")):
            return _finish("none", text, [], 0.94, 0.0, False)
        return _finish(
            "lexical",
            "Block or freeze the active payment card.",
            [
                AlternativeReading(reading="Permanently close the user account.", probability=0.25),
                AlternativeReading(reading="Cancel the latest pending order.", probability=0.11),
            ],
            0.64,
            0.20,
            False,
        )

    if len(text.split()) <= 3:
        return _finish(
            "underspecified",
            f"Request for status or assistance regarding '{text}'.",
            [
                AlternativeReading(reading="Request to modify or cancel an active item.", probability=0.35),
                AlternativeReading(reading="General support inquiry.", probability=0.20),
            ],
            0.45,
            0.40,
            False,
        )

    if strong_spacy_types:
        st = strong_spacy_types[0]
        return _finish(
            st,  # type: ignore[arg-type]
            f"Resolved {st} interpretation of '{text}'.",
            [
                AlternativeReading(reading=f"Primary {st} interpretation of '{text}'.", probability=0.65),
                AlternativeReading(reading=f"Alternative {st} interpretation of '{text}'.", probability=0.35),
            ],
            0.70,
            0.20,
            False,
        )

    return _finish("none", text, [], 0.92, 0.0, False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _get_llm_providers() -> list[tuple[str, str, str, str]]:
    """
    Return ordered list of (name, api_key, base_url, model).
    Prioritizes Groq, with Gemini as fallback (and optional generic LLM_API_KEY).
    """
    providers: list[tuple[str, str, str, str]] = []

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and groq_key not in PLACEHOLDER_KEYS:
        providers.append((
            "groq",
            groq_key,
            os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        ))

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key not in PLACEHOLDER_KEYS:
        providers.append((
            "gemini",
            gemini_key,
            os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/"),
            os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip(),
        ))

    llm_key = os.getenv("LLM_API_KEY", "").strip()
    if llm_key and llm_key not in PLACEHOLDER_KEYS:
        providers.append((
            "custom",
            llm_key,
            os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        ))

    return providers


async def resolve_batch(
    records: list[str],
    *,
    explicit_context: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """
    Resolve a batch of text records.

    Parameters
    ----------
    records          Primary sentences / records to analyze.
    explicit_context Pasted text, chat history, or previous records as a
                     single string (newline-separated). Merged with the
                     automatic surrounding-record context.
    metadata         Document-level key-value pairs (e.g. {"doc_title": "...",
                     "author": "..."}).  Added to context automatically.
    """
    cleaned = [r.strip() for r in records if r and r.strip()]
    if not cleaned:
        return []

    providers = _get_llm_providers()
    sem = asyncio.Semaphore(4)

    async def _process_one(idx: int, text: str) -> dict[str, Any]:
        surrounding = [r for i, r in enumerate(cleaned) if i != idx and abs(i - idx) <= 2]
        context_items, context_summary = build_context_list(
            surrounding=surrounding,
            explicit=explicit_context,
            metadata=metadata,
        )

        ctx_str = "\n".join(context_items)
        spacy_det = detect_all(text, context=ctx_str)
        deb_probs = predict_deberta(text, context=ctx_str)
        blended = blend_scores(spacy_det, deb_probs)
        scores = blended["blended_scores"]
        conf_penalty = blended["confidence_penalty"]

        detected_types = [k for k, v in scores.items() if k != "none" and v >= 0.5]
        if not detected_types:
            detected_types = ["none"]
        strong_spacy = [k for k, v in spacy_det.items() if k != "none" and v.get("score", 0.0) >= 0.70]

        resolved: Optional[ResolvedRecord] = None
        async with sem:
            for _, api_key, base_url, model in providers:
                try:
                    resolved = await _resolve_single_llm(
                        text=text,
                        context_items=context_items,
                        context_summary=context_summary,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        detected_types=detected_types,
                        confidence_penalty=conf_penalty,
                        strong_spacy_types=strong_spacy,
                    )
                    break
                except Exception:
                    continue

        if resolved is None:
            resolved = _resolve_single_mock(
                text=text,
                context_items=context_items,
                context_summary=context_summary,
                detected_types=detected_types,
                confidence_penalty=conf_penalty,
                strong_spacy_types=strong_spacy,
            )
        dumped = resolved.model_dump()
        RECORD_STORE[dumped["id"]] = dumped
        return dumped

    return list(await asyncio.gather(*[_process_one(idx, text) for idx, text in enumerate(cleaned)]))
