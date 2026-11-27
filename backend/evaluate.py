import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decision_engine import decide_action
from intent_classifier import classify_intent
from intent_embeddings import (
    compute_ambiguity_score,
    compute_similarity_margin,
    query_centroid_similarities,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_CSV = DATA_DIR / "test_queries.csv"
CUSTOMERS_FILE = DATA_DIR / "mock_customers.json"
POLICY_FILE = DATA_DIR / "policy.json"
RESULTS_FILE = DATA_DIR / "results.json"
CALIBRATION_PLOT_FILE = DATA_DIR / "calibration_plot.png"

FIXED_CONFIDENCE_THRESHOLD = 0.80
MARGIN_MIN_FOR_ACT = 0.05


def load_customers_map() -> dict[str, dict[str, Any]]:
    with CUSTOMERS_FILE.open("r", encoding="utf-8") as f:
        customers = json.load(f).get("customers", [])
    return {c["customer_id"]: c for c in customers}


def load_policies() -> dict[str, dict[str, Any]]:
    with POLICY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f).get("policies", {})


def _embedding_fallback_analysis(query: str) -> dict[str, Any]:
    """Deterministic fallback classifier using multilingual centroid similarities when no LLM key is set."""
    sims = query_centroid_similarities(query)
    intents = list(sims.keys())
    values = np.array([sims[i] for i in intents], dtype=np.float64)
    scaled = values * 12.0
    exp_v = np.exp(scaled - np.max(scaled))
    probs = exp_v / np.sum(exp_v)

    candidates = [
        {"intent": intent, "confidence": round(float(prob), 4)}
        for intent, prob in zip(intents, probs)
    ]
    margin_info = compute_similarity_margin(query)
    is_amb = margin_info["margin"] < 0.06 or candidates[0]["confidence"] < 0.70
    return {
        "is_ambiguous": is_amb,
        "ambiguity_type": "underspecified" if is_amb else "none",
        "candidate_intents": candidates,
    }


async def predict_query(
    query: str,
    customer: dict[str, Any],
    use_llm: bool,
) -> dict[str, Any]:
    chat_history = customer.get("chat_history", [])
    customer_data = {k: v for k, v in customer.items() if k != "chat_history"}

    if use_llm:
        analysis = await classify_intent(
            query=query,
            customer_data=customer_data,
            chat_history=chat_history,
        )
    else:
        analysis = _embedding_fallback_analysis(query)

    margin_info = compute_similarity_margin(query)
    margin = margin_info["margin"]
    top = analysis["candidate_intents"][0]
    ambiguity_score = compute_ambiguity_score(top["confidence"], margin)

    return {
        "analysis": analysis,
        "top_intent": top["intent"],
        "top_confidence": float(top["confidence"]),
        "margin": margin,
        "ambiguity_score": ambiguity_score,
    }


def compute_ece(confidences: list[float], correct: list[int], n_bins: int = 10) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and reliability curve bins."""
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(confs)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs >= low) & (confs < high)
        count = int(np.sum(mask))
        bin_centers.append(round((low + high) / 2.0, 2))
        bin_counts.append(count)
        if count > 0:
            avg_acc = float(np.mean(accs[mask]))
            avg_conf = float(np.mean(confs[mask]))
            ece += (count / n) * abs(avg_acc - avg_conf)
            bin_accs.append(round(avg_acc, 4))
            bin_confs.append(round(avg_conf, 4))
        else:
            bin_accs.append(None)
            bin_confs.append(None)

    return {
        "ece": round(float(ece), 4),
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def evaluate_system(
    rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    system_name: str,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    intent_correct = 0
    clear_total = 0
    unnecessary_questions = 0
    total_cost_error = 0.0
    confidences = []
    correct_flags = []

    for row, pred in zip(rows, predictions):
        gt_ambiguous = row["is_ambiguous"].strip().lower() == "true"
        gt_intent = row["correct_intent"].strip()
        top_intent = pred["top_intent"]
        top_conf = pred["top_confidence"]
        margin = pred["margin"]

        if system_name == "always_act":
            decision = "ACT"
        elif system_name == "always_ask":
            decision = "ASK"
        elif system_name == "confidence_only":
            decision = "ACT" if top_conf >= FIXED_CONFIDENCE_THRESHOLD else "ASK"
        elif system_name == "risk_aware_margin":
            dec = decide_action(pred["analysis"], clarification_count=0)
            decision = dec["decision"]
            risk_tier = dec["risk_tier"]
            if decision == "ACT" and risk_tier != "low" and margin < MARGIN_MIN_FOR_ACT:
                decision = "ASK"
        else:
            raise ValueError(f"Unknown system: {system_name}")

        pred_ambiguous = decision != "ACT"
        if gt_ambiguous and pred_ambiguous:
            tp += 1
        elif not gt_ambiguous and pred_ambiguous:
            fp += 1
        elif gt_ambiguous and not pred_ambiguous:
            fn += 1
        else:
            tn += 1

        is_correct = int(top_intent == gt_intent)
        intent_correct += is_correct
        confidences.append(top_conf)
        correct_flags.append(is_correct)

        if not gt_ambiguous:
            clear_total += 1
            if decision == "ASK":
                unnecessary_questions += 1

        if decision == "ACT" and top_intent != gt_intent:
            cost_weight = float(policies.get(gt_intent, {}).get("cost_weight", 1))
            total_cost_error += cost_weight

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    n = len(rows)
    intent_accuracy = intent_correct / n if n > 0 else 0.0
    uqr = unnecessary_questions / clear_total if clear_total > 0 else 0.0
    mean_cost_error = total_cost_error / n if n > 0 else 0.0
    calibration = compute_ece(confidences, correct_flags)

    return {
        "system": system_name,
        "detection_precision": round(precision, 4),
        "detection_recall": round(recall, 4),
        "detection_f1": round(f1, 4),
        "intent_accuracy": round(intent_accuracy, 4),
        "unnecessary_question_rate": round(uqr, 4),
        "cost_weighted_error": round(mean_cost_error, 4),
        "total_cost_weighted_error": round(total_cost_error, 4),
        "expected_calibration_error": calibration["ece"],
        "calibration_bins": calibration,
    }


def save_calibration_plot(
    calibration: dict[str, Any],
    output_path: Path = CALIBRATION_PLOT_FILE,
) -> None:
    centers = []
    accs = []
    for c, a in zip(calibration["bin_centers"], calibration["bin_accuracies"]):
        if a is not None:
            centers.append(c)
            accs.append(a)

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=140)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#71717a", label="Perfect calibration")
    if centers:
        ax.plot(centers, accs, marker="o", linewidth=2, color="#22c55e", label=f"Model (ECE={calibration['ece']:.4f})")
        ax.bar(centers, accs, width=0.08, alpha=0.2, color="#22c55e")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Intent Classifier Calibration Plot")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


async def run_evaluation(limit: int | None = None) -> dict[str, Any]:
    with TEST_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]

    customers = load_customers_map()
    policies = load_policies()

    api_key = os.getenv("LLM_API_KEY", "").strip()
    use_llm = bool(api_key and api_key not in {"your_llm_api_key_here", "sk-your-key-here"})

    predictions = []
    for row in rows:
        customer = customers.get(row["customer_id"], {})
        pred = await predict_query(row["query"], customer, use_llm=use_llm)
        predictions.append(pred)

    systems = ["always_act", "always_ask", "confidence_only", "risk_aware_margin"]
    system_results = {
        name: evaluate_system(rows, predictions, policies, name)
        for name in systems
    }

    clear_count = sum(1 for r in rows if r["is_ambiguous"].strip().lower() == "false")
    ambiguous_count = len(rows) - clear_count
    annotator_agree = (
        sum(1 for r in rows if r["correct_intent"].strip() == r["annotator2_label"].strip())
        / len(rows)
        if rows
        else 0.0
    )

    results = {
        "mode": "llm" if use_llm else "embedding_fallback",
        "dataset": {
            "num_queries": len(rows),
            "clear_queries": clear_count,
            "ambiguous_queries": ambiguous_count,
            "inter_annotator_agreement": round(annotator_agree, 4),
        },
        "systems": system_results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    save_calibration_plot(system_results["risk_aware_margin"]["calibration_bins"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 4 intent decision systems.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to evaluate")
    args = parser.parse_args()
    results = asyncio.run(run_evaluation(limit=args.limit))
    print(f"Mode: {results['mode']} | Queries: {results['dataset']['num_queries']}")
    for name, m in results["systems"].items():
        print(
            f"{name:18s} | F1={m['detection_f1']:.3f} | Acc={m['intent_accuracy']:.3f} "
            f"| UQR={m['unnecessary_question_rate']:.3f} | CostErr={m['cost_weighted_error']:.3f} "
            f"| ECE={m['expected_calibration_error']:.3f}"
        )


if __name__ == "__main__":
    main()
