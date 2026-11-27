"""
classifier.py – Multi-label DeBERTa-v3-small inference & spaCy blending.

Loads the fine-tuned model from `backend/models/deberta-ambiguity`,
predicts per-type probabilities, blends them with spaCy
detector scores using a weighted average, and computes a confidence penalty
when DeBERTa and spaCy disagree.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

MODEL_DIR = Path(__file__).resolve().parent / "models" / "deberta-ambiguity"

LABELS = [
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
]

_TOKENIZER: Any = None
_MODEL: Any = None


def is_deberta_available() -> bool:
    return (MODEL_DIR / "config.json").exists()


def _load_model() -> tuple[Any, Any]:
    global _TOKENIZER, _MODEL
    if _MODEL is None and is_deberta_available():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _MODEL = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def predict_deberta(text: str, context: str = "") -> Optional[dict[str, float]]:
    """Return {type: prob} for all 10 types + 'none', or None if model not trained yet."""
    tokenizer, model = _load_model()
    if tokenizer is None or model is None:
        return None

    import torch

    enc = tokenizer(
        text,
        context if context else None,
        truncation=True,
        padding=True,
        max_length=96,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(0).tolist()

    probs = {lbl: round(1.0 / (1.0 + math.exp(-float(logit))), 4) for lbl, logit in zip(LABELS, logits)}
    probs["none"] = round(max(0.0, 1.0 - max(probs.values(), default=0.0)), 4)
    return probs


def blend_scores(
    spacy_results: dict[str, dict[str, Any]],
    deberta_probs: Optional[dict[str, float]],
    *,
    deberta_weight: float = 0.6,
    spacy_weight: float = 0.4,
    disagreement_threshold: float = 0.30,
) -> dict[str, Any]:
    """
    Blend spaCy detector scores and DeBERTa probabilities (weighted average).
    Lowers confidence when they disagree strongly.
    """
    if not deberta_probs:
        blended = {k: v["score"] for k, v in spacy_results.items()}
        return {
            "blended_scores": blended,
            "deberta_used": False,
            "detector_disagreement": 0.0,
            "confidence_penalty": 0.0,
        }

    blended: dict[str, float] = {}
    diffs: list[float] = []
    for lbl in LABELS:
        s_score = float(spacy_results.get(lbl, {}).get("score", 0.0))
        d_score = float(deberta_probs.get(lbl, 0.0))
        blended[lbl] = round(deberta_weight * d_score + spacy_weight * s_score, 4)
        diffs.append(abs(d_score - s_score))

    blended["none"] = round(max(0.0, 1.0 - max(blended.values(), default=0.0)), 4)

    max_diff = max(diffs, default=0.0)
    spacy_top = max(LABELS, key=lambda l: spacy_results.get(l, {}).get("score", 0.0))
    deb_top = max(LABELS, key=lambda l: deberta_probs.get(l, 0.0))
    top_mismatch = (
        spacy_top != deb_top
        and spacy_results.get(spacy_top, {}).get("score", 0.0) >= 0.5
        and deberta_probs.get(deb_top, 0.0) >= 0.5
    )

    disagreement = round(max_diff, 4)
    # Lower confidence proportionally when DeBERTa and spaCy disagree
    penalty = 0.0
    if max_diff > disagreement_threshold or top_mismatch:
        penalty = round(min(0.15, 0.25 * max_diff + (0.05 if top_mismatch else 0.0)), 4)

    return {
        "blended_scores": blended,
        "deberta_used": True,
        "detector_disagreement": disagreement,
        "confidence_penalty": penalty,
    }
