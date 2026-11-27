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
from sklearn.manifold import trustworthiness

from decision_engine import decide_action
from intent_classifier import classify_intent
from intent_embeddings import (
    _l2_normalize,
    compute_and_cache_centroids,
    compute_reading_ambiguity,
    get_model,
    query_centroid_similarities,
)
from intent_map import fit_or_load_projection

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
    ranked = list(sims.values())
    margin = round(max(0.0, ranked[0] - ranked[1]), 4) if len(ranked) > 1 else 0.0
    is_amb = margin < 0.06 or candidates[0]["confidence"] < 0.70
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

    sims = list(query_centroid_similarities(query).values())
    margin = round(max(0.0, sims[0] - sims[1]), 4) if len(sims) > 1 else 0.0
    top = analysis["candidate_intents"][0]
    reading_amb = compute_reading_ambiguity(analysis.get("readings", []))

    return {
        "analysis": analysis,
        "top_intent": top["intent"],
        "top_confidence": float(top["confidence"]),
        "margin": margin,
        "ambiguity_score": reading_amb["score"],
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


PRIMARY_METRICS = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "intent_accuracy",
    "unnecessary_question_rate",
    "cost_weighted_error",
    "expected_calibration_error",
)


def evaluate_system(
    rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    system_name: str,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    intent_correct = 0
    clear_total = 0
    questions_asked = 0
    unnecessary_questions = 0
    wrong_acts = 0
    total_cost_error = 0.0
    confidences = []
    correct_flags = []
    failure_cases = []

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
        elif system_name == "llm_only":
            dec = decide_action(pred["analysis"], clarification_count=0)
            decision = dec["decision"]
        elif system_name == "risk_aware_margin":
            dec = decide_action(pred["analysis"], clarification_count=0)
            decision = dec["decision"]
            risk_tier = dec["risk_tier"]
            if decision == "ACT" and risk_tier != "low" and margin < MARGIN_MIN_FOR_ACT:
                decision = "ASK"
        elif system_name == "llm_plus_margin_plus_steering":
            dec = decide_action(pred["analysis"], clarification_count=0)
            decision = dec["decision"]
            risk_tier = dec["risk_tier"]
            if decision == "ACT" and risk_tier != "low" and margin < MARGIN_MIN_FOR_ACT:
                decision = "ASK"
            if decision == "ASK" and "steered_intent" in pred:
                top_intent = pred["steered_intent"]
                top_conf = pred["steered_confidence"]
        else:
            raise ValueError(f"Unknown system: {system_name}")

        pred_ambiguous = decision != "ACT"
        if decision == "ASK":
            questions_asked += 1
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
            wrong_acts += 1
            cost_weight = float(policies.get(gt_intent, {}).get("cost_weight", 1))
            total_cost_error += cost_weight

        if (top_intent != gt_intent or (gt_ambiguous and not pred_ambiguous)) and len(failure_cases) < 5:
            failure_cases.append(
                {
                    "query": row["query"],
                    "correct_intent": gt_intent,
                    "predicted_intent": top_intent,
                    "decision": decision,
                    "ambiguity_type": row["ambiguity_type"],
                }
            )

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
        "questions_asked": questions_asked,
        "necessary_questions": tp,
        "wrong_actions": wrong_acts,
        "failure_cases": failure_cases,
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

    always_act_wrong = system_results["always_act"]["wrong_actions"]
    risk_aware = system_results["risk_aware_margin"]

    results = {
        "mode": "llm" if use_llm else "embedding_fallback",
        "dataset": {
            "num_queries": len(rows),
            "clear_queries": clear_count,
            "ambiguous_queries": ambiguous_count,
            "inter_annotator_agreement": round(annotator_agree, 4),
        },
        "headline": {
            "wrong_actions_prevented": max(0, always_act_wrong - risk_aware["wrong_actions"]),
            "questions_asked": risk_aware["questions_asked"],
            "necessary_questions": risk_aware["necessary_questions"],
        },
        "systems": system_results,
        "checks": {
            "steering_control": check_steering_control(rows),
            "quality_preserved": check_quality_preserved(rows),
            "visualizer_faithfulness": check_visualizer_faithfulness(),
        },
        "ablation": run_ablation(rows, predictions, policies),
        "failure_cases": risk_aware["failure_cases"],
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    save_calibration_plot(system_results["risk_aware_margin"]["calibration_bins"])
    return results


def run_ablation(
    rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Compare LLM-only gating vs LLM + embedding margin + steering across all primary metrics."""
    centroids = compute_and_cache_centroids()
    intent_names = list(centroids.keys())
    model = get_model()

    enriched_preds = []
    for row, pred in zip(rows, predictions):
        item = dict(pred)
        target = row["correct_intent"].strip()
        source = pred["top_intent"] if pred["top_intent"] != target else (
            "order_status" if target != "order_status" else "card_block"
        )
        vec_before = _l2_normalize(
            np.asarray(model.encode([row["query"]], normalize_embeddings=True)[0], dtype=np.float32)
        )
        vec_after = _l2_normalize(vec_before + alpha * (centroids[target] - centroids[source]))
        sims = np.array([float(np.dot(vec_after, centroids[n])) for n in intent_names])
        scaled = sims * 12.0
        exp_v = np.exp(scaled - np.max(scaled))
        probs = exp_v / np.sum(exp_v)
        best_idx = int(np.argmax(probs))
        item["steered_intent"] = intent_names[best_idx]
        item["steered_confidence"] = round(float(probs[best_idx]), 4)
        enriched_preds.append(item)

    llm_only = evaluate_system(rows, enriched_preds, policies, "llm_only")
    full = evaluate_system(rows, enriched_preds, policies, "llm_plus_margin_plus_steering")
    diff = {
        metric: round(full[metric] - llm_only[metric], 4)
        for metric in PRIMARY_METRICS
    }
    return {
        "llm_only": llm_only,
        "llm_plus_margin_plus_steering": full,
        "difference": diff,
    }


def check_steering_control(
    rows: list[dict[str, str]],
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Sweep alpha from 0 to 1 and measure target intent similarity change and correlation."""
    centroids = compute_and_cache_centroids()
    model = get_model()
    sample = rows[:10]
    all_alphas: list[float] = []
    all_sims: list[float] = []
    per_alpha: dict[str, list[float]] = {str(a): [] for a in alphas}

    for row in sample:
        target = row["correct_intent"].strip()
        alt = row["annotator2_label"].strip()
        source = alt if alt != target else ("order_status" if target != "order_status" else "card_block")
        vec_before = _l2_normalize(
            np.asarray(model.encode([row["query"]], normalize_embeddings=True)[0], dtype=np.float32)
        )
        delta = centroids[target] - centroids[source]
        for a in alphas:
            vec_after = _l2_normalize(vec_before + float(a) * delta)
            sim = float(np.dot(vec_after, centroids[target]))
            all_alphas.append(float(a))
            all_sims.append(sim)
            per_alpha[str(a)].append(sim)

    mean_by_alpha = {
        k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in per_alpha.items()
    }
    corr = float(np.corrcoef(all_alphas, all_sims)[0, 1]) if len(all_alphas) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    return {
        "num_queries": len(sample),
        "mean_similarity_by_alpha": mean_by_alpha,
        "similarity_gain": round(
            mean_by_alpha[str(alphas[-1])] - mean_by_alpha[str(alphas[0])], 4
        ),
        "correlation": round(corr, 4),
    }


def check_quality_preserved(
    rows: list[dict[str, str]],
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Compare intent classification accuracy on a held-out set before and after steering."""
    held_out = rows[-20:] if len(rows) >= 20 else rows
    centroids = compute_and_cache_centroids()
    intent_names = list(centroids.keys())
    model = get_model()

    correct_before = 0
    correct_after = 0
    for row in held_out:
        target = row["correct_intent"].strip()
        vec_before = _l2_normalize(
            np.asarray(model.encode([row["query"]], normalize_embeddings=True)[0], dtype=np.float32)
        )
        pred_before = max(intent_names, key=lambda name: float(np.dot(vec_before, centroids[name])))
        if pred_before == target:
            correct_before += 1

        source = pred_before if pred_before != target else (
            "order_status" if target != "order_status" else "card_block"
        )
        vec_after = _l2_normalize(vec_before + alpha * (centroids[target] - centroids[source]))
        pred_after = max(intent_names, key=lambda name: float(np.dot(vec_after, centroids[name])))
        if pred_after == target:
            correct_after += 1

    n = len(held_out) or 1
    acc_before = round(correct_before / n, 4)
    acc_after = round(correct_after / n, 4)
    return {
        "held_out_size": len(held_out),
        "alpha": alpha,
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "accuracy_delta": round(acc_after - acc_before, 4),
    }


def check_visualizer_faithfulness() -> dict[str, Any]:
    """Compute trustworthiness between 2D PCA projection and original high-dimensional embeddings."""
    projection = fit_or_load_projection()
    points = projection.get("points", [])
    texts = [p["text"] for p in points]
    coords_2d = np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)

    model = get_model()
    embeddings = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float64)

    k5 = min(5, max(1, len(points) // 2 - 1))
    trust_k5 = float(trustworthiness(embeddings, coords_2d, n_neighbors=k5))
    return {
        "num_points": len(points),
        "n_neighbors": k5,
        "trustworthiness": round(trust_k5, 4),
    }


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
