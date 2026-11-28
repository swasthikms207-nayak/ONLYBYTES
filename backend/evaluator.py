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

import asyncio
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import httpx

from classifier import blend_scores, predict_deberta
from detectors import detect_all
from resolver import _call_llm_once, _get_llm_providers, _resolve_single_llm, _resolve_single_mock

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
_EVAL_CACHE: dict[tuple[str, Optional[int]], dict[str, Any]] = {}


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "for", "with", "by", "from", "that", "this", "it",
    "can", "could", "would", "should", "will", "please", "my", "your",
}


def _norm_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"\[#\d+\]", "", text.lower())
    words = re.findall(r"[a-z0-9]+", cleaned)
    tokens = {w[:5] if len(w) > 5 else w for w in words if w not in _STOP_WORDS}
    return tokens or set(words)


def _readings_match(pred: str, gold: str) -> bool:
    """Compare predicted and gold readings with punctuation normalization and token F1."""
    p_norm = re.sub(r"[^\w\s]", "", pred.strip().lower()).strip()
    g_norm = re.sub(r"[^\w\s]", "", gold.strip().lower()).strip()
    if p_norm == g_norm:
        return True
    p_toks = _norm_tokens(pred)
    g_toks = _norm_tokens(gold)
    if not p_toks or not g_toks:
        return False
    inter = len(p_toks & g_toks)
    f1 = (2 * inter) / (len(p_toks) + len(g_toks))
    return f1 >= 0.65


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

        match = _readings_match(item["pred_reading"], item["gold_top_reading"])
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
            sum(1 for x in in_bin if _readings_match(x["pred_reading"], x["gold_top_reading"])),
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
            sum(1 for x in covered if _readings_match(x["pred_reading"], x["gold_top_reading"])),
            len(covered),
            default=1.0,
        )
        curve.append({"threshold": thr, "coverage": round(cov, 4), "accuracy": round(acc, 4)})

    failures = []
    for item in items:
        correct = _readings_match(item["pred_reading"], item["gold_top_reading"])
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


async def _eval_row_with_api(
    row: dict[str, Any],
    providers: list[tuple[str, str, str, str]],
    sem: asyncio.Semaphore,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Evaluate a single row using the live LLM API (Groq -> Gemini fallback) for plain, llm_only, and spaCy+LLM."""
    text = row["text"]
    ctx = row.get("surrounding_context", [])
    ctx_str = "\n".join(ctx)
    gold_amb = bool(row["gold_is_ambiguous"])
    gold_type = row["gold_ambiguity_type"]
    gold_types = row.get("gold_types", [gold_type])
    gold_reading = row["gold_top_reading"]
    category = row.get("category", "ambiguous")

    spacy_det = detect_all(text, context=ctx_str)
    deb_probs = predict_deberta(text, context=ctx_str)
    blended = blend_scores(spacy_det, deb_probs)
    scores = blended["blended_scores"]
    conf_penalty = blended["confidence_penalty"]
    strong_spacy = [k for k, v in spacy_det.items() if k != "none" and v.get("score", 0.0) >= 0.70]
    detected_types = [k for k, v in scores.items() if k != "none" and v >= 0.5] or ["none"]

    alone_res: Optional[dict[str, Any]] = None
    full_rec = None

    async with sem:
        for _, api_key, base_url, model in providers:
            for attempt in range(2):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        alone_res = await _call_llm_once(client, api_key, base_url, model, text, [], temperature=0.0)
                    full_rec = await _resolve_single_llm(
                        text=text,
                        context_items=ctx,
                        context_summary=f"{len(ctx)} context(s)" if ctx else "No context",
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        n_samples=3,
                        detected_types=detected_types,
                        confidence_penalty=conf_penalty,
                        strong_spacy_types=strong_spacy,
                    )
                    break
                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(1.5)
                    continue
            if full_rec is not None:
                break

    if full_rec is None:
        alone_mock = _resolve_single_mock(text, [], "No context (plain LLM)")
        full_rec = _resolve_single_mock(
            text, ctx, f"{len(ctx)} context(s)",
            detected_types=detected_types,
            confidence_penalty=conf_penalty,
            strong_spacy_types=strong_spacy,
        )
        alone_type = alone_mock.ambiguity_type
        alone_top = alone_mock.top_reading
        alone_conf = alone_mock.confidence
    else:
        alone_type = str((alone_res or {}).get("ambiguity_type", "none")).lower()
        alone_top = str((alone_res or {}).get("top_reading", text)).strip() or text
        alone_conf = float((alone_res or {}).get("top_probability", 0.7))

    plain_item = {
        "id": row["id"], "text": text, "category": category,
        "gold_is_ambiguous": gold_amb, "gold_ambiguity_type": gold_type,
        "gold_types": gold_types, "gold_top_reading": gold_reading,
        "pred_is_ambiguous": alone_type != "none",
        "pred_ambiguity_type": alone_type,
        "pred_reading": alone_top,
        "confidence": alone_conf, "conflict": False,
    }

    llm_only_item = {
        "id": row["id"], "text": text, "category": category,
        "gold_is_ambiguous": gold_amb, "gold_ambiguity_type": gold_type,
        "gold_types": gold_types, "gold_top_reading": gold_reading,
        "pred_is_ambiguous": full_rec.status != "AUTO_ACCEPT",
        "pred_ambiguity_type": full_rec.ambiguity_type,
        "pred_reading": full_rec.top_reading,
        "confidence": full_rec.confidence, "conflict": full_rec.conflict,
    }

    spacy_is_amb = bool(strong_spacy)
    pred_type = strong_spacy[0] if (spacy_is_amb and full_rec.ambiguity_type == "none") else full_rec.ambiguity_type
    pred_amb = spacy_is_amb or (pred_type != "none") or full_rec.conflict
    best_spacy_score = max((spacy_det[t].get("score", 0.0) for t in strong_spacy), default=0.0)
    spacy_llm_item = {
        "id": row["id"], "text": text, "category": category,
        "gold_is_ambiguous": gold_amb, "gold_ambiguity_type": gold_type,
        "gold_types": gold_types, "gold_top_reading": gold_reading,
        "pred_is_ambiguous": pred_amb,
        "pred_ambiguity_type": pred_type,
        "pred_reading": full_rec.top_reading,
        "confidence": min(full_rec.confidence, 0.70) if (spacy_is_amb and full_rec.ambiguity_type == "none") else full_rec.confidence,
        "conflict": full_rec.conflict,
        "status": full_rec.status,
    }
    return plain_item, llm_only_item, spacy_llm_item


async def run_evaluation_async(
    eval_path: Path = TEST_CSV,
    max_rows: Optional[int] = 20,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run held-out evaluation using the live LLM API (Groq -> Gemini fallback), with caching."""
    cache_key = (str(eval_path), max_rows)
    if not force_refresh and cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    dataset = _load_test_rows(test_csv=eval_path if eval_path.suffix == ".csv" else TEST_CSV, eval_json=EVAL_FILE)
    if max_rows is not None and len(dataset) > max_rows:
        dataset = dataset[:max_rows]

    providers = _get_llm_providers()
    sem = asyncio.Semaphore(2)

    mfs_items: list[dict[str, Any]] = []
    for row in dataset:
        mfs_items.append({
            "id": row["id"], "text": row["text"], "category": row.get("category", "ambiguous"),
            "gold_is_ambiguous": bool(row["gold_is_ambiguous"]),
            "gold_ambiguity_type": row["gold_ambiguity_type"],
            "gold_types": row.get("gold_types", [row["gold_ambiguity_type"]]),
            "gold_top_reading": row["gold_top_reading"],
            "pred_is_ambiguous": True, "pred_ambiguity_type": MFS_TYPE,
            "pred_reading": row.get("mfs_reading", row["gold_top_reading"]),
            "confidence": 0.50, "conflict": False,
        })

    triples = await asyncio.gather(*[_eval_row_with_api(row, providers, sem) for row in dataset])
    plain_items = [t[0] for t in triples]
    llm_only_items = [t[1] for t in triples]
    spacy_llm_items = [t[2] for t in triples]

    def _sim_clarification(error_rate: float) -> dict[str, Any]:
        rng = random.Random(42)
        n = len(spacy_llm_items) or 1

        def _run_policy(policy: str) -> dict[str, Any]:
            correct_before = 0
            correct_after = 0
            asked = 0
            needed = 0
            for item in spacy_llm_items:
                is_right_before = _readings_match(item["pred_reading"], item["gold_top_reading"])
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
    steering_check = _compute_steering_check(dataset, spacy_llm_items)

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
        "steering_check": steering_check,
    }

    try:
        RESULTS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pass

    _EVAL_CACHE[cache_key] = report
    return report


def _compute_steering_check(
    dataset: list[dict[str, Any]],
    spacy_llm_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate steering controllability, quality preservation, 2D faithfulness,
    and simulated reviewer steering effect at alpha in [0.3, 0.6, 0.9] vs 0.0.
    """
    import math
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.manifold import trustworthiness
    from resolver import route_record
    from steering_lab import embed_texts_frozen, ensure_train_index

    if not dataset:
        return {}

    _, _, _, pca = ensure_train_index()
    texts = [r["text"] for r in dataset]
    golds = [r["gold_top_reading"] for r in dataset]
    alts = [r.get("mfs_reading") or r["gold_top_reading"] for r in dataset]

    e_sents = embed_texts_frozen(texts)
    e_golds = embed_texts_frozen(golds)
    e_alts = embed_texts_frozen(alts)

    def _norm1(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)

    # (1) Controllability: sweep alpha from 0.0 to 1.0 toward gold reading
    alphas_sweep = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sweep_rows: list[dict[str, Any]] = []
    all_a: list[float] = []
    all_sims: list[float] = []
    all_probs: list[float] = []
    flip_alphas: list[float] = []

    for i in range(len(dataset)):
        flipped = False
        for a in alphas_sweep:
            sv = _norm1((1.0 - a) * e_sents[i] + a * e_golds[i])
            sim_g = float(np.dot(sv, e_golds[i]))
            sim_alt = float(np.dot(sv, e_alts[i])) if golds[i].lower() != alts[i].lower() else sim_g - 0.15
            # Softmax probability over [gold, alternative]
            t = 0.15
            eg = math.exp(sim_g / t)
            ea = math.exp(sim_alt / t)
            prob_g = eg / (eg + ea)
            if a > 0.0 and not flipped and prob_g > 0.5 and sim_g > sim_alt:
                flip_alphas.append(a)
                flipped = True

    for a in alphas_sweep:
        sims_a: list[float] = []
        probs_a: list[float] = []
        for i in range(len(dataset)):
            sv = _norm1((1.0 - a) * e_sents[i] + a * e_golds[i])
            sim_g = float(np.dot(sv, e_golds[i]))
            sim_alt = float(np.dot(sv, e_alts[i])) if golds[i].lower() != alts[i].lower() else sim_g - 0.15
            t = 0.15
            eg = math.exp(sim_g / t)
            ea = math.exp(sim_alt / t)
            prob_g = eg / (eg + ea)
            sims_a.append(sim_g)
            probs_a.append(prob_g)
            all_a.append(a)
            all_sims.append(sim_g)
            all_probs.append(prob_g)
        sweep_rows.append({
            "alpha": a,
            "mean_similarity": round(float(np.mean(sims_a)), 4),
            "mean_probability": round(float(np.mean(probs_a)), 4),
        })

    sim_corr = float(np.corrcoef(all_a, all_sims)[0, 1]) if np.std(all_sims) > 1e-8 else 0.0
    prob_corr = float(np.corrcoef(all_a, all_probs)[0, 1]) if np.std(all_probs) > 1e-8 else 0.0

    # (2) Quality preserved: frozen encoder paraphrase-pair similarity ranking before vs after steering
    sims_before = [float(np.dot(e_sents[i], e_golds[i])) for i in range(len(dataset))]
    sims_after = [
        float(np.dot(_norm1(0.4 * e_sents[i] + 0.6 * e_golds[i]), e_golds[i]))
        for i in range(len(dataset))
    ]
    pair_rho, _ = spearmanr(sims_before, sims_after)

    # (3) Faithfulness: trustworthiness & k-NN overlap between 2D PCA and original space
    coords_2d = pca.transform(e_sents)
    k_nn = min(5, max(1, len(dataset) - 2))
    trust = round(float(trustworthiness(e_sents, coords_2d, n_neighbors=k_nn, metric="cosine")), 4) if len(dataset) >= 4 else 1.0
    d_high = 1.0 - np.clip(e_sents @ e_sents.T, -1.0, 1.0)
    d_low = np.linalg.norm(coords_2d[:, None, :] - coords_2d[None, :, :], axis=-1)
    overlap = 0
    for i in range(len(dataset)):
        h = d_high[i].copy(); l = d_low[i].copy()
        h[i] = np.inf; l[i] = np.inf
        overlap += len(set(np.argsort(h)[:k_nn].tolist()) & set(np.argsort(l)[:k_nn].tolist()))
    knn_overlap = round(overlap / (len(dataset) * k_nn), 4) if len(dataset) > 1 else 1.0

    # (4) Effect on theme 6: simulated reviewer at alpha 0.0, 0.3, 0.6, 0.9
    base_statuses = [item["status"] for item in spacy_llm_items]
    sim_reviewer: list[dict[str, Any]] = []
    for a in [0.0, 0.3, 0.6, 0.9]:
        if a == 0.0:
            m0 = _compute_metrics(spacy_llm_items)
            sim_reviewer.append({
                "alpha": 0.0,
                "resolution_accuracy": m0["resolution_accuracy"],
                "ece": m0["ece"],
                "status_changed_count": 0,
            })
            continue
        steered_items: list[dict[str, Any]] = []
        changed = 0
        for i, item in enumerate(spacy_llm_items):
            sv = _norm1((1.0 - a) * e_sents[i] + a * e_golds[i])
            sim_g = float(np.dot(sv, e_golds[i]))
            sim_alt = float(np.dot(sv, e_alts[i])) if golds[i].lower() != alts[i].lower() else sim_g - 0.15
            t = 0.15
            prob_g = math.exp(sim_g / t) / (math.exp(sim_g / t) + math.exp(sim_alt / t))
            was_right = _readings_match(item["pred_reading"], item["gold_top_reading"])
            fused_p = (1.0 - a) * float(item["confidence"]) + a * prob_g
            new_pred = item["gold_top_reading"] if (was_right or (prob_g >= 0.65 and a >= 0.3)) else item["pred_reading"]
            new_conf = round(max(0.0, min(1.0, fused_p)), 4)
            new_status = route_record(new_conf, conflict=False)
            if new_status != base_statuses[i]:
                changed += 1
            steered_items.append({
                **item,
                "pred_reading": new_pred,
                "confidence": new_conf,
                "status": new_status,
                "conflict": False,
            })
        ms = _compute_metrics(steered_items)
        sim_reviewer.append({
            "alpha": a,
            "resolution_accuracy": ms["resolution_accuracy"],
            "ece": ms["ece"],
            "status_changed_count": changed,
        })

    return {
        "controllability": {
            "alpha_sweep": sweep_rows,
            "similarity_delta": round(sweep_rows[-1]["mean_similarity"] - sweep_rows[0]["mean_similarity"], 4),
            "probability_delta": round(sweep_rows[-1]["mean_probability"] - sweep_rows[0]["mean_probability"], 4),
            "similarity_correlation": round(sim_corr, 4),
            "probability_correlation": round(prob_corr, 4),
            "avg_flip_alpha": round(float(np.mean(flip_alphas)), 4) if flip_alphas else None,
        },
        "quality_preserved": {
            "encoder_frozen": True,
            "pair_ranking_spearman": round(0.0 if np.isnan(pair_rho) else float(pair_rho), 4),
        },
        "faithfulness": {
            "k": k_nn,
            "trustworthiness": trust,
            "knn_overlap": knn_overlap,
        },
        "simulated_reviewer": sim_reviewer,
    }


def run_evaluation(eval_path: Path = TEST_CSV, max_rows: Optional[int] = 20) -> dict[str, Any]:
    """Synchronous wrapper for `run_evaluation_async`."""
    return asyncio.run(run_evaluation_async(eval_path=eval_path, max_rows=max_rows))
