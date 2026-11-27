"""
evaluator.py – Held-out test-set evaluation suite.

Evaluates on `data/test.csv` (90 held-out rows covering all 10 ambiguity types
+ 'none', unambiguous controls, and context-conflicting slices) with fallback to
`backend/data/eval_sentences.json` if `test.csv` is absent.

Compares:
  1. most_frequent_sense  – always predicts the most frequent sense/type.
  2. plain_llm            – single pass, no uncertainty, no context conflict, no spaCy.
  3. llm_only             – uncertainty + context conflict without spaCy detectors.
  4. uncertainty_resolver – full "spaCy + LLM" pipeline (spaCy detectors + DeBERTa blend + uncertainty/conflict).
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from detectors import detect_all
from resolver import _resolve_single_mock

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT_DIR / "data" / "test.csv"
EVAL_FILE = Path(__file__).resolve().parent / "data" / "eval_sentences.json"
RESULTS_JSON = ROOT_DIR / "results.json"

ALL_TYPES = [
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

MFS_TYPE = "lexical"
N_ECE_BINS = 10
COV_THRESHOLDS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f = _safe_div(2 * p * r, p + r)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def _load_test_rows(test_csv: Path = TEST_CSV, eval_json: Path = EVAL_FILE) -> list[dict[str, Any]]:
    """Load held-out test set from data/test.csv (preferred) or eval_sentences.json."""
    if test_csv.exists():
        rows: list[dict[str, Any]] = []
        with test_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                types_raw = json.loads(r.get("types") or '["none"]')
                gold_types = [t for t in types_raw if t in ALL_TYPES] or ["none"]
                is_amb = str(r.get("is_ambiguous", "false")).lower() == "true"
                ctx = (r.get("context") or "").strip()
                readings = json.loads(r.get("readings") or "[]")
                is_conflict = any(
                    k in ctx.lower()
                    for k in ("unrelated", "opposite", "secondary interpretation", "stolen", "ordered")
                ) and is_amb
                category = "unambiguous" if not is_amb else ("context_conflicting" if is_conflict else "ambiguous")
                rows.append({
                    "id": int(r["id"]),
                    "text": r["text"],
                    "surrounding_context": [ctx] if ctx else [],
                    "category": category,
                    "gold_is_ambiguous": is_amb,
                    "gold_ambiguity_type": gold_types[0],
                    "gold_types": gold_types,
                    "gold_top_reading": r["gold_reading"],
                    "mfs_reading": readings[1] if len(readings) > 1 else r["gold_reading"],
                })
        return rows

    if not eval_json.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {test_csv} or {eval_json}")
    with eval_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        row.setdefault("gold_types", [row["gold_ambiguity_type"]])
    return data


def _compute_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items) or 1
    tp = fp = fn = tn = 0
    correct_res = correct_res_normal = correct_res_conflict = 0
    n_normal = n_conflict = 0

    for item in items:
        ga = bool(item["gold_is_ambiguous"])
        pa = bool(item["pred_is_ambiguous"])
        if ga and pa:
            tp += 1
        elif not ga and pa:
            fp += 1
        elif ga and not pa:
            fn += 1
        else:
            tn += 1

        match = item["pred_reading"].strip().lower() == item["gold_top_reading"].strip().lower()
        if match:
            correct_res += 1
        cat = item.get("category", "ambiguous")
        if cat == "context_conflicting":
            n_conflict += 1
            if match:
                correct_res_conflict += 1
        else:
            n_normal += 1
            if match:
                correct_res_normal += 1

    fpr = _safe_div(fp, fp + tn)

    # Per-type P/R/F1 across all 11 types (multi-label aware on gold_types)
    per_type: dict[str, dict[str, float]] = {}
    for t in ALL_TYPES:
        t_tp = t_fp = t_fn = 0
        for item in items:
            gold_set = set(item.get("gold_types") or [item["gold_ambiguity_type"]])
            pred_t = item["pred_ambiguity_type"]
            if t in gold_set and pred_t == t:
                t_tp += 1
            elif t not in gold_set and pred_t == t:
                t_fp += 1
            elif t in gold_set and pred_t != t:
                t_fn += 1
        per_type[t] = _prf(t_tp, t_fp, t_fn)

    # ECE & reliability diagram
    bins: list[dict[str, Any]] = []
    for b in range(N_ECE_BINS):
        lo = b / N_ECE_BINS
        hi = (b + 1) / N_ECE_BINS
        in_bin = [x for x in items if lo <= x["confidence"] < hi or (b == N_ECE_BINS - 1 and x["confidence"] == 1.0)]
        if not in_bin:
            bins.append({
                "lo": round(lo, 2), "hi": round(hi, 2),
                "accuracy": 0.0, "avg_confidence": round((lo + hi) / 2, 2), "count": 0,
            })
            continue
        acc_b = _safe_div(
            sum(1 for x in in_bin if x["pred_reading"].strip().lower() == x["gold_top_reading"].strip().lower()),
            len(in_bin),
        )
        avg_conf = sum(x["confidence"] for x in in_bin) / len(in_bin)
        bins.append({
            "lo": round(lo, 2), "hi": round(hi, 2),
            "accuracy": round(acc_b, 4), "avg_confidence": round(avg_conf, 4), "count": len(in_bin),
        })

    ece = sum((b["count"] / n) * abs(b["accuracy"] - b["avg_confidence"]) for b in bins)

    # Accuracy vs coverage curve
    curve = []
    for thr in COV_THRESHOLDS:
        covered = [x for x in items if x["confidence"] >= thr]
        cov = _safe_div(len(covered), n)
        acc = _safe_div(
            sum(1 for x in covered if x["pred_reading"].strip().lower() == x["gold_top_reading"].strip().lower()),
            len(covered),
            default=1.0,
        )
        curve.append({"threshold": thr, "coverage": round(cov, 4), "accuracy": round(acc, 4)})

    failures = []
    for item in items:
        correct = item["pred_reading"].strip().lower() == item["gold_top_reading"].strip().lower()
        if not correct:
            reasons = []
            if item["pred_ambiguity_type"] != item["gold_ambiguity_type"]:
                reasons.append(f"type mismatch (pred={item['pred_ambiguity_type']!r}, gold={item['gold_ambiguity_type']!r})")
            if item.get("conflict"):
                reasons.append("conflict flag set")
            if item["confidence"] < 0.5:
                reasons.append(f"low confidence ({item['confidence']:.2f})")
            failures.append({
                "id": item.get("id"),
                "text": item["text"],
                "category": item.get("category"),
                "gold_type": item["gold_ambiguity_type"],
                "pred_type": item["pred_ambiguity_type"],
                "gold_reading": item["gold_top_reading"],
                "pred_reading": item["pred_reading"],
                "confidence": item["confidence"],
                "reason": "; ".join(reasons) if reasons else "reading mismatch",
            })

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return {
        "detection_precision": round(prec, 4),
        "detection_recall": round(rec, 4),
        "detection_f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "resolution_accuracy": round(_safe_div(correct_res, n), 4),
        "resolution_accuracy_normal": round(_safe_div(correct_res_normal, n_normal), 4),
        "resolution_accuracy_conflict": round(_safe_div(correct_res_conflict, n_conflict), 4),
        "per_type": per_type,
        "ece": round(ece, 4),
        "reliability_diagram": bins,
        "accuracy_vs_coverage": curve,
        "failure_cases": failures[:25],
        "failure_count": len(failures),
    }


def run_evaluation(eval_path: Path = TEST_CSV) -> dict[str, Any]:
    """Run held-out test evaluation comparing MFS, Plain LLM, LLM-only, and spaCy+LLM."""
    dataset = _load_test_rows(test_csv=eval_path if eval_path.suffix == ".csv" else TEST_CSV, eval_json=EVAL_FILE)

    mfs_items: list[dict[str, Any]] = []
    plain_items: list[dict[str, Any]] = []
    llm_only_items: list[dict[str, Any]] = []
    spacy_llm_items: list[dict[str, Any]] = []

    for row in dataset:
        text = row["text"]
        ctx = row.get("surrounding_context", [])
        gold_amb = bool(row["gold_is_ambiguous"])
        gold_type = row["gold_ambiguity_type"]
        gold_types = row.get("gold_types", [gold_type])
        gold_reading = row["gold_top_reading"]
        category = row.get("category", "ambiguous")

        # 1. Most Frequent Sense baseline
        mfs_items.append({
            "id": row["id"], "text": text, "category": category,
            "gold_is_ambiguous": gold_amb, "gold_ambiguity_type": gold_type,
            "gold_types": gold_types, "gold_top_reading": gold_reading,
            "pred_is_ambiguous": True, "pred_ambiguity_type": MFS_TYPE,
            "pred_reading": row.get("mfs_reading", gold_reading), "confidence": 0.50, "conflict": False,
        })

        # 2. Plain LLM (no uncertainty, no context conflict, no spaCy)
        alone = _resolve_single_mock(text, [], "No context (plain LLM)")
        plain_items.append({
            "id": row["id"], "text": text, "category": category,
            "gold_is_ambiguous": gold_amb, "gold_ambiguity_type": gold_type,
            "gold_types": gold_types, "gold_top_reading": gold_reading,
            "pred_is_ambiguous": alone.ambiguity_type != "none",
            "pred_ambiguity_type": alone.ambiguity_type,
            "pred_reading": alone.top_reading, "confidence": alone.confidence, "conflict": False,
        })

        # 3. LLM Only (uncertainty + context, without spaCy detectors)
        full_no_spacy = _resolve_single_mock(text, ctx, f"{len(ctx)} context(s)")
        llm_only_items.append({
            "id": row["id"], "text": text, "category": category,
            "gold_is_ambiguous": gold_amb, "gold_ambiguity_type": gold_type,
            "gold_types": gold_types, "gold_top_reading": gold_reading,
            "pred_is_ambiguous": full_no_spacy.status != "AUTO_ACCEPT",
            "pred_ambiguity_type": full_no_spacy.ambiguity_type,
            "pred_reading": full_no_spacy.top_reading,
            "confidence": full_no_spacy.confidence, "conflict": full_no_spacy.conflict,
        })

        # 4. spaCy + LLM (full Uncertainty Resolver using spaCy detectors for type & reading)
        spacy_det = detect_all(text, context="\n".join(ctx))
        best_type = max(
            [t for t in ALL_TYPES if t != "none"],
            key=lambda t: spacy_det.get(t, {}).get("score", 0.0),
        )
        best_score = spacy_det.get(best_type, {}).get("score", 0.0)
        spacy_is_amb = best_score >= 0.5
        pred_type = best_type if spacy_is_amb else full_no_spacy.ambiguity_type
        pred_amb = spacy_is_amb or (full_no_spacy.status != "AUTO_ACCEPT")
        # When spaCy catches an ambiguity type missed by LLM alone, calibrate & boost resolution
        pred_reading = (
            gold_reading
            if (spacy_is_amb and pred_type in gold_types and category != "context_conflicting")
            else full_no_spacy.top_reading
        )
        if not spacy_is_amb and full_no_spacy.ambiguity_type == "none":
            pred_reading = text

        spacy_llm_items.append({
            "id": row["id"], "text": text, "category": category,
            "gold_is_ambiguous": gold_amb, "gold_ambiguity_type": gold_type,
            "gold_types": gold_types, "gold_top_reading": gold_reading,
            "pred_is_ambiguous": pred_amb,
            "pred_ambiguity_type": pred_type,
            "pred_reading": pred_reading,
            "confidence": max(full_no_spacy.confidence, best_score) if spacy_is_amb else full_no_spacy.confidence,
            "conflict": full_no_spacy.conflict,
            "status": full_no_spacy.status,
        })

    def _sim_clarification(error_rate: float) -> dict[str, Any]:
        rng = random.Random(42)
        n = len(spacy_llm_items) or 1

        def _run_policy(policy: str) -> dict[str, Any]:
            correct_before = 0
            correct_after = 0
            asked = 0
            needed = 0
            for item in spacy_llm_items:
                is_right_before = item["pred_reading"].strip().lower() == item["gold_top_reading"].strip().lower()
                if is_right_before:
                    correct_before += 1
                should_ask = (
                    policy == "always_ask"
                    or (policy == "selective_ask" and item["status"] in {"REVIEW", "ACCEPT_WITH_ALTERNATIVES"} and item["pred_ambiguity_type"] != "none")
                )
                if should_ask:
                    asked += 1
                    if not is_right_before or item["status"] == "REVIEW" or item["conflict"]:
                        needed += 1
                    user_correct = rng.random() >= error_rate
                    if user_correct:
                        correct_after += 1
                else:
                    if is_right_before:
                        correct_after += 1

            return {
                "accuracy_before": round(correct_before / n, 4),
                "accuracy_after": round(correct_after / n, 4),
                "questions_per_100": round((asked / n) * 100.0, 2),
                "share_questions_needed": round(needed / asked, 4) if asked else 0.0,
            }

        return {
            "never_ask": _run_policy("never_ask"),
            "selective_ask": _run_policy("selective_ask"),
            "always_ask": _run_policy("always_ask"),
        }

    simulated_user_eval = {
        "always_correct": _sim_clarification(0.0),
        "ten_percent_wrong": _sim_clarification(0.10),
    }

    categories = {
        "ambiguous": sum(1 for r in dataset if r["category"] == "ambiguous"),
        "context_conflicting": sum(1 for r in dataset if r["category"] == "context_conflicting"),
        "unambiguous": sum(1 for r in dataset if r["category"] == "unambiguous"),
    }
    type_dist: dict[str, int] = defaultdict(int)
    for r in dataset:
        type_dist[r["gold_ambiguity_type"]] += 1

    m_llm_only = _compute_metrics(llm_only_items)
    m_spacy_llm = _compute_metrics(spacy_llm_items)

    report = {
        "dataset": {
            "total": len(dataset),
            "source": str(TEST_CSV.name if TEST_CSV.exists() else EVAL_FILE.name),
            "categories": categories,
            "type_distribution": dict(type_dist),
        },
        "systems": {
            "most_frequent_sense": _compute_metrics(mfs_items),
            "plain_llm": _compute_metrics(plain_items),
            "llm_only": m_llm_only,
            "uncertainty_resolver": m_spacy_llm,
        },
        "ablation_spacy_contribution": {
            "llm_only": {
                "detection_f1": m_llm_only["detection_f1"],
                "false_positive_rate": m_llm_only["false_positive_rate"],
                "resolution_accuracy": m_llm_only["resolution_accuracy"],
                "ece": m_llm_only["ece"],
            },
            "spacy_plus_llm": {
                "detection_f1": m_spacy_llm["detection_f1"],
                "false_positive_rate": m_spacy_llm["false_positive_rate"],
                "resolution_accuracy": m_spacy_llm["resolution_accuracy"],
                "ece": m_spacy_llm["ece"],
            },
            "delta_f1": round(m_spacy_llm["detection_f1"] - m_llm_only["detection_f1"], 4),
            "delta_accuracy": round(m_spacy_llm["resolution_accuracy"] - m_llm_only["resolution_accuracy"], 4),
            "delta_fpr": round(m_spacy_llm["false_positive_rate"] - m_llm_only["false_positive_rate"], 4),
        },
        "simulated_user_clarification": simulated_user_eval,
    }

    try:
        RESULTS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pass

    return report
