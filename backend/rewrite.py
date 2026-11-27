"""
rewrite.py – Deterministic span-aware rewrite stage for Ambiguity Resolver.

Produces `resolved_text`, `changes` ([{span, before, after, reason}]), and
`assumed` (bool) from the original text, top_reading, confidence, detected spans,
and context.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from detectors import detect_all, get_nlp

_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|1[0-2])[/\-](\d{2,4})\b")
_TIME_RE = re.compile(
    r"\b(at|by|around|before|after)\s+([1-9]|1[0-2])(?::([0-5]\d))?\b(?!\s*(?:am|pm|a\.m\.|p\.m\.))",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"\b(send|pay|transfer|charge|cost|costs|owe|worth|about)\s+(\d+(?:\.\d+)?)\b"
    r"(?!\s*(?:dollars|usd|inr|rs|rupees|eur|gbp|kg|km|hours|mins|minutes|%))",
    re.IGNORECASE,
)

_PRONOUN_SET = {"he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "this", "that"}


def _extract_named_entity(text: str, context_items: list[str], top_reading: str) -> Optional[str]:
    """Find a concrete named entity or noun phrase from context, top_reading, or text."""
    nlp = get_nlp()
    # 1. Prefer named entities from context items (most recent first)
    for ctx in reversed(context_items):
        cdoc = nlp(ctx)
        for ent in cdoc.ents:
            if ent.label_ in {"PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART"}:
                return ent.text
        # Fallback to explicit IDs like ORD_5519 / TXN_90421 or noun chunks
        m_id = re.search(r"\b(?:ORD|TXN|CARD)_[A-Za-z0-9]+\b", ctx)
        if m_id:
            return m_id.group(0)
        for chunk in cdoc.noun_chunks:
            if chunk.root.pos_ in {"NOUN", "PROPN"} and chunk.root.lower_ not in _PRONOUN_SET:
                return chunk.text

    # 2. Try entities in text itself
    tdoc = nlp(text)
    for ent in tdoc.ents:
        if ent.label_ in {"PERSON", "ORG", "PRODUCT"}:
            return ent.text

    # 3. Fallback to top_reading if it names a concrete noun
    rdoc = nlp(top_reading)
    for ent in rdoc.ents:
        return ent.text
    return None


def rewrite_sentence(
    *,
    text: str,
    ambiguity_type: str,
    top_reading: str,
    alternatives: list[dict[str, Any]],
    confidence: float,
    context_items: Optional[list[str]] = None,
    detections: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Rewrite `text` according to `top_reading` and detected spans.

    Returns:
        {
            "resolved_text": str,
            "assumed": bool,
            "changes": [{"span": str, "before": str, "after": str, "reason": str}],
        }
    """
    ctx = context_items or []

    # Rule 5: If no ambiguity, return the original text unchanged
    if ambiguity_type == "none":
        return {"resolved_text": text, "assumed": False, "changes": []}

    # Rule 4: If confidence < 0.5, output both versions and mark "assumed"
    if confidence < 0.5:
        alt_reading = (
            alternatives[0]["reading"]
            if alternatives and alternatives[0].get("reading")
            else text
        )
        dual = f"[{top_reading}] / [{alt_reading}]"
        return {
            "resolved_text": dual,
            "assumed": True,
            "changes": [
                {
                    "span": text,
                    "before": text,
                    "after": dual,
                    "reason": f"Low confidence ({confidence:.2f} < 0.50); output both readings and marked assumed",
                }
            ],
        }

    det = detections if detections is not None else detect_all(text, context="\n".join(ctx))
    rewritten = text
    changes: list[dict[str, str]] = []

    # Rule 2a: Normalize ambiguous dates (dd/mm vs mm/dd) -> ISO YYYY-MM-DD
    def _replace_date(m: re.Match[str]) -> str:
        p1, p2, yr = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(yr) == 2:
            yr = f"20{yr}"
        # Use MM/DD if top_reading explicitly favors US/month-first; else DD/MM
        if "mm/dd" in top_reading.lower() or "month/day" in top_reading.lower():
            month, day = p1, p2
        else:
            day, month = p1, p2
        iso = f"{yr}-{month:02d}-{day:02d}"
        changes.append({
            "span": m.group(0),
            "before": m.group(0),
            "after": iso,
            "reason": "Normalized ambiguous date to ISO 8601 (YYYY-MM-DD)",
        })
        return iso

    rewritten = _DATE_RE.sub(_replace_date, rewritten)

    # Rule 2b: Normalize times without AM/PM
    def _replace_time(m: re.Match[str]) -> str:
        prep, hr, mins = m.group(1), m.group(2), m.group(3) or "00"
        meridiem = "AM" if "am" in top_reading.lower() or "morning" in top_reading.lower() else "PM"
        norm = f"{prep} {hr}:{mins} {meridiem}"
        changes.append({
            "span": m.group(0),
            "before": m.group(0),
            "after": norm,
            "reason": f"Normalized time with explicit {meridiem}",
        })
        return norm

    rewritten = _TIME_RE.sub(_replace_time, rewritten)

    # Rule 2c: Normalize bare numeric amounts with currency
    def _replace_amount(m: re.Match[str]) -> str:
        verb, amt = m.group(1), m.group(2)
        currency = "INR" if any(k in (top_reading + " " + " ".join(ctx)).lower() for k in ("inr", "rupee", "rs", "upi")) else "USD"
        norm = f"{verb} {currency} {amt}"
        changes.append({
            "span": m.group(0),
            "before": m.group(0),
            "after": norm,
            "reason": f"Normalized bare amount with explicit currency ({currency})",
        })
        return norm

    rewritten = _AMOUNT_RE.sub(_replace_amount, rewritten)

    # Rule 3: Replace pronouns with the named entity (if referential or pronoun span detected)
    ref_info = det.get("referential", {})
    if ambiguity_type == "referential" or ref_info.get("score", 0.0) >= 0.5:
        pron_span = ref_info.get("span") or next(
            (w for w in rewritten.split() if w.lower().strip(".,!?") in _PRONOUN_SET),
            "",
        )
        entity = _extract_named_entity(text, ctx, top_reading)
        if pron_span and entity:
            pattern = re.compile(rf"\b{re.escape(pron_span)}\b", re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(entity, rewritten, count=1)
                changes.append({
                    "span": pron_span,
                    "before": pron_span,
                    "after": entity,
                    "reason": f"Replaced ambiguous pronoun '{pron_span}' with named entity '{entity}'",
                })

    # Rule 1: Resolve any remaining detected ambiguity span using top_reading
    # while keeping surrounding text unchanged where possible.
    if not changes:
        primary_det = det.get(ambiguity_type, {})
        span = primary_det.get("span") or text
        # If top_reading is a full disambiguated paraphrase of the sentence, use it
        rewritten = top_reading
        changes.append({
            "span": span,
            "before": text,
            "after": top_reading,
            "reason": f"Rewrote '{span}' according to top {ambiguity_type} reading",
        })

    return {
        "resolved_text": rewritten,
        "assumed": False,
        "changes": changes,
    }
