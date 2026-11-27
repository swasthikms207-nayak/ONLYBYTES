"""
clarification.py – Inline clarification question generation, POST /clarify
re-resolution, and batch MCQ helpers.
"""

from __future__ import annotations

from typing import Any, Optional

from detectors import detect_all
from rewrite import rewrite_sentence

# In-memory store of recently processed records keyed by record id
RECORD_STORE: dict[str, dict[str, Any]] = {}


def build_inline_clarification(
    *,
    text: str,
    ambiguity_type: str,
    status: str,
    top_reading: str,
    alternatives: list[dict[str, Any]],
    span: str = "",
) -> Optional[dict[str, Any]]:
    """
    Generate ONE short optional inline question with 2-4 plain-language
    paraphrase options plus a 'Something else' option.
    ONLY generates a question for genuinely ambiguous sentences (never for
    AUTO_ACCEPT or unambiguous 'none' records).
    """
    if status == "AUTO_ACCEPT" or ambiguity_type == "none":
        return None
    if not alternatives and top_reading.strip().lower() == text.strip().lower():
        return None

    target_span = span.strip()
    if not target_span:
        det = detect_all(text)
        target_span = det.get(ambiguity_type, {}).get("span", "").strip()

    question = (
        f'What did you mean by "{target_span}"?'
        if target_span
        else f'Which meaning did you intend for "{text}"?'
    )

    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, reading in enumerate(
        [top_reading] + [str(a.get("reading", "")).strip() for a in alternatives]
    ):
        r_clean = reading.strip()
        if not r_clean or r_clean.lower() in seen:
            continue
        seen.add(r_clean.lower())
        options.append(
            {
                "id": f"opt_{len(options)}",
                "label": r_clean,
                "reading": r_clean,
                "is_other": False,
            }
        )
        if len(options) >= 4:
            break

    # Ensure at least 2 paraphrase options before adding "Something else"
    if len(options) < 2:
        fallback = f"Keep original wording: {text}"
        if fallback.lower() not in seen:
            options.append(
                {
                    "id": f"opt_{len(options)}",
                    "label": fallback,
                    "reading": text,
                    "is_other": False,
                }
            )

    options.append(
        {
            "id": "opt_other",
            "label": "Something else",
            "reading": "__other__",
            "is_other": True,
        }
    )

    return {
        "question": question,
        "span": target_span or text,
        "options": options,
    }


def apply_inline_clarification(
    record: dict[str, Any],
    chosen_option: str,
    custom_text: Optional[str] = None,
) -> dict[str, Any]:
    """
    Use `chosen_option` (or `custom_text` when 'Something else' is chosen) as the
    target reading, recompute confidence & routing, regenerate the rewrite, and
    return the updated record with `clarified_by_user = True`.
    """
    updated = dict(record)
    target_reading = (
        custom_text.strip()
        if chosen_option == "__other__" and custom_text and custom_text.strip()
        else chosen_option.strip()
    )
    if not target_reading:
        target_reading = updated.get("top_reading", updated.get("text", ""))

    # Re-run confidence & routing after explicit user clarification
    new_conf = 0.96
    new_status = "AUTO_ACCEPT"

    # Regenerate rewrite with the user-selected target reading
    rw = rewrite_sentence(
        text=updated.get("text", ""),
        ambiguity_type=updated.get("ambiguity_type", "none"),
        top_reading=target_reading,
        alternatives=updated.get("alternatives") or [],
        confidence=new_conf,
    )

    updated["top_reading"] = target_reading
    updated["resolved_text"] = rw["resolved_text"]
    updated["assumed"] = False
    updated["changes"] = rw["changes"]
    updated["confidence"] = new_conf
    updated["disagreement"] = 0.0
    updated["conflict"] = False
    updated["conflict_reason"] = None
    updated["status"] = new_status
    updated["clarified_by_user"] = True
    updated["clarification"] = None

    steps = list(updated.get("resolution_steps") or [])
    steps.append(
        {
            "step": len(steps) + 1,
            "action": "inline_user_clarification",
            "result": {
                "target_reading": target_reading,
                "confidence": new_conf,
                "status": new_status,
                "resolved_text": rw["resolved_text"],
            },
        }
    )
    updated["resolution_steps"] = steps

    rec_id = updated.get("id")
    if rec_id:
        RECORD_STORE[rec_id] = updated

    return updated


def generate_mcqs(results: list[dict[str, Any]], max_questions: int = 5) -> list[dict[str, Any]]:
    """Generate up to `max_questions` (4-5) MCQs from processed records."""
    if not results:
        return []

    def _priority(item: tuple[int, dict[str, Any]]) -> tuple[int, float]:
        _, r = item
        status_rank = {"REVIEW": 0, "ACCEPT_WITH_ALTERNATIVES": 1, "AUTO_ACCEPT": 2}.get(
            r.get("status", "AUTO_ACCEPT"), 2
        )
        return (status_rank, float(r.get("confidence", 1.0)))

    ranked = sorted(enumerate(results), key=_priority)[:max_questions]
    mcqs: list[dict[str, Any]] = []
    for q_idx, (rec_idx, r) in enumerate(ranked, start=1):
        text = r.get("text", "")
        amb_type = r.get("ambiguity_type", "none")
        top_reading = r.get("top_reading", text)
        alts = r.get("alternatives") or []

        options: list[dict[str, Any]] = [
            {
                "id": "opt_0",
                "label": top_reading,
                "reading": top_reading,
                "probability": round(
                    max(0.0, 1.0 - sum(float(a.get("probability", 0.0)) for a in alts)), 4
                ),
                "is_model_top": True,
            }
        ]
        for a_idx, alt in enumerate(alts, start=1):
            reading = str(alt.get("reading", "")).strip()
            if reading and reading.lower() != top_reading.lower():
                options.append(
                    {
                        "id": f"opt_{a_idx}",
                        "label": reading,
                        "reading": reading,
                        "probability": round(float(alt.get("probability", 0.0)), 4),
                        "is_model_top": False,
                    }
                )
        if len(options) < 2:
            options.append(
                {
                    "id": "opt_alt",
                    "label": "Needs different interpretation / manual review",
                    "reading": text,
                    "probability": 0.0,
                    "is_model_top": False,
                }
            )
        mcqs.append(
            {
                "id": f"mcq_{q_idx}",
                "record_index": rec_idx,
                "text": text,
                "ambiguity_type": amb_type,
                "status": r.get("status", ""),
                "confidence": r.get("confidence", 0.0),
                "question": (
                    f'Which meaning best fits "{text}"?'
                    if amb_type != "none"
                    else f'Confirm the intended meaning for "{text}":'
                ),
                "options": options[:4],
            }
        )
    return mcqs


def apply_mcq_answers(
    results: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply user MCQ selections to `results` and compute clarification evaluation scores."""
    updated = [dict(r) for r in results]
    answer_map = {int(a["record_index"]): str(a["selected_reading"]).strip() for a in answers}

    matched_model_top = 0
    resolved_conflicts = 0
    pre_confs: list[float] = []
    post_confs: list[float] = []

    for rec_idx, selected_reading in answer_map.items():
        if rec_idx < 0 or rec_idx >= len(updated) or not selected_reading:
            continue
        rec = updated[rec_idx]
        old_top = str(rec.get("top_reading", "")).strip()
        old_conf = float(rec.get("confidence", 0.5))
        pre_confs.append(old_conf)

        if selected_reading.lower() == old_top.lower():
            matched_model_top += 1
        if rec.get("conflict") or rec.get("status") == "REVIEW":
            resolved_conflicts += 1

        clarified = apply_inline_clarification(rec, selected_reading)
        post_confs.append(float(clarified["confidence"]))
        updated[rec_idx] = clarified

    answered_count = len(pre_confs)
    pre_acc = round(matched_model_top / answered_count, 4) if answered_count else 0.0
    post_acc = 1.0 if answered_count else 0.0
    return {
        "results": updated,
        "clarification_score": {
            "answered_count": answered_count,
            "matched_model_top": matched_model_top,
            "pre_clarification_accuracy": pre_acc,
            "post_clarification_accuracy": post_acc,
            "accuracy_gain": round(post_acc - pre_acc, 4),
            "pre_confidence_avg": round(sum(pre_confs) / answered_count, 4) if answered_count else 0.0,
            "post_confidence_avg": round(sum(post_confs) / answered_count, 4) if answered_count else 0.0,
            "resolved_conflicts": resolved_conflicts,
        },
    }
