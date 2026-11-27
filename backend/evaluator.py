import json
from pathlib import Path
from typing import Any

from resolver import _resolve_single_mock

EVAL_FILE = Path(__file__).resolve().parent / "data" / "eval_sentences.json"


def _compute_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    correct_resolutions = 0

    for item in items:
        gt_amb = bool(item["gold_is_ambiguous"])
        pred_amb = bool(item["pred_is_ambiguous"])
        if gt_amb and pred_amb:
            tp += 1
        elif not gt_amb and pred_amb:
            fp += 1
        elif gt_amb and not pred_amb:
            fn += 1
        else:
            tn += 1

        if item["pred_reading"].strip().lower() == item["gold_top_reading"].strip().lower():
            correct_resolutions += 1

    n = len(items) or 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = correct_resolutions / n

    # Accuracy vs. coverage curve across confidence thresholds
    thresholds = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9]
    curve = []
    for thr in thresholds:
        covered = [x for x in items if x["confidence"] >= thr]
        cov = len(covered) / n
        acc = (
            sum(
                1
                for x in covered
                if x["pred_reading"].strip().lower() == x["gold_top_reading"].strip().lower()
            )
            / len(covered)
            if covered
            else 1.0
        )
        curve.append(
            {
                "threshold": thr,
                "coverage": round(cov, 4),
                "accuracy": round(acc, 4),
            }
        )

    return {
        "detection_precision": round(precision, 4),
        "detection_recall": round(recall, 4),
        "detection_f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "resolution_accuracy": round(accuracy, 4),
        "accuracy_vs_coverage": curve,
    }


def run_evaluation(eval_path: Path = EVAL_FILE) -> dict[str, Any]:
    """Evaluate Most Frequent Sense, Plain LLM, and Uncertainty Resolver on 40 gold-labeled sentences."""
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")

    with eval_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    mfs_items: list[dict[str, Any]] = []
    plain_llm_items: list[dict[str, Any]] = []
    uncertainty_items: list[dict[str, Any]] = []

    for row in dataset:
        text = row["text"]
        ctx = row.get("surrounding_context", [])
        gold_amb = bool(row["gold_is_ambiguous"])
        gold_reading = row["gold_top_reading"]

        # 1. Most Frequent Sense baseline (never flags ambiguity, uses default sense)
        mfs_items.append(
            {
                "gold_is_ambiguous": gold_amb,
                "gold_top_reading": gold_reading,
                "pred_is_ambiguous": False,
                "pred_reading": row["mfs_reading"],
                "confidence": 0.50,
            }
        )

        # 2. Plain LLM (no uncertainty / no self-consistency / no conflict check)
        alone = _resolve_single_mock(text, [])
        plain_amb = alone.ambiguity_type != "none"
        plain_llm_items.append(
            {
                "gold_is_ambiguous": gold_amb,
                "gold_top_reading": gold_reading,
                "pred_is_ambiguous": plain_amb,
                "pred_reading": alone.top_reading,
                "confidence": alone.confidence,
            }
        )

        # 3. Our Uncertainty Resolver (self-consistency + sentence-vs-context conflict + routing)
        full = _resolve_single_mock(text, ctx)
        full_amb = full.status != "AUTO_ACCEPT"
        uncertainty_items.append(
            {
                "gold_is_ambiguous": gold_amb,
                "gold_top_reading": gold_reading,
                "pred_is_ambiguous": full_amb,
                "pred_reading": full.top_reading,
                "confidence": full.confidence,
            }
        )

    categories = {
        "ambiguous": sum(1 for r in dataset if r["category"] == "ambiguous"),
        "context_conflicting": sum(1 for r in dataset if r["category"] == "context_conflicting"),
        "unambiguous": sum(1 for r in dataset if r["category"] == "unambiguous"),
    }

    return {
        "dataset": {
            "total": len(dataset),
            "categories": categories,
        },
        "systems": {
            "most_frequent_sense": _compute_metrics(mfs_items),
            "plain_llm": _compute_metrics(plain_llm_items),
            "uncertainty_resolver": _compute_metrics(uncertainty_items),
        },
    }
