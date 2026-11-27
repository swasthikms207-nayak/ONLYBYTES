from typing import Any


SHIPPED_STATUSES = {"shipped", "in_transit", "delivered", "cancelled"}


def filter_readings_with_context(
    readings: list[dict[str, Any]],
    customer_data: dict[str, Any],
    chat_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rule out impossible readings using customer data and chat history, then renormalize."""
    if not readings:
        return {
            "readings": [],
            "candidate_intents": [],
            "context_used": "No readings provided for context filtering.",
            "resolved_by_context": False,
        }

    pending_orders = customer_data.get("pending_orders", [])
    pending_txns = customer_data.get("pending_transactions", [])
    account_status = str(customer_data.get("account_status", "")).lower()

    ruled_out_reasons: list[str] = []
    ruled_out_intents: set[str] = set()

    if len(pending_orders) == 0:
        ruled_out_intents.update({"cancel_order", "order_status", "change_address"})
        ruled_out_reasons.append("no pending orders exist")
    else:
        all_shipped = all(
            str(o.get("status", "")).lower() in SHIPPED_STATUSES for o in pending_orders
        )
        if all_shipped:
            ruled_out_intents.update({"cancel_order", "change_address"})
            order_ids = ", ".join(str(o.get("order_id", "order")) for o in pending_orders)
            ruled_out_reasons.append(f"{order_ids} already shipped/in transit")

    if len(pending_txns) == 0:
        ruled_out_intents.update({"payment_reversal", "transaction_dispute"})
        ruled_out_reasons.append("no pending transactions exist")

    if account_status == "closed":
        ruled_out_intents.add("account_closure")
        ruled_out_reasons.append("account is already closed")

    kept = [dict(r) for r in readings if r.get("intent") not in ruled_out_intents]
    removed = [r.get("intent") for r in readings if r.get("intent") in ruled_out_intents]

    # Never drop all readings; fall back to original readings if every reading was ruled out
    if not kept:
        kept = [dict(r) for r in readings]
        removed = []
        ruled_out_reasons = []

    total = sum(max(0.0, float(r.get("probability", 0.0))) for r in kept)
    if total <= 0.0:
        equal_p = round(1.0 / len(kept), 4)
        for r in kept:
            r["probability"] = equal_p
    else:
        for r in kept:
            r["probability"] = round(max(0.0, float(r["probability"])) / total, 4)

    kept.sort(key=lambda r: r["probability"], reverse=True)
    drift = round(1.0 - sum(r["probability"] for r in kept), 4)
    if drift != 0.0:
        kept[0]["probability"] = round(kept[0]["probability"] + drift, 4)

    # Aggregate into candidate_intents
    intent_probs: dict[str, float] = {}
    for r in kept:
        intent = str(r["intent"])
        intent_probs[intent] = round(intent_probs.get(intent, 0.0) + r["probability"], 4)

    candidate_intents = [
        {"intent": intent, "confidence": conf}
        for intent, conf in sorted(intent_probs.items(), key=lambda kv: kv[1], reverse=True)
    ]
    if candidate_intents:
        cand_drift = round(1.0 - sum(c["confidence"] for c in candidate_intents), 4)
        if cand_drift != 0.0:
            candidate_intents[0]["confidence"] = round(
                candidate_intents[0]["confidence"] + cand_drift, 4
            )

    top_prob = kept[0]["probability"]
    resolved_by_context = bool(removed) and (len(kept) == 1 or top_prob >= 0.80)

    if removed:
        removed_str = ", ".join(dict.fromkeys(str(i) for i in removed))
        reason_str = "; ".join(ruled_out_reasons)
        history_note = f" (checked {len(chat_history)} chat turns)" if chat_history else ""
        context_used = (
            f"Ruled out {removed_str} ({reason_str}){history_note}; "
            f"top reading '{kept[0]['intent']}' at {top_prob:.2f}."
        )
    else:
        context_used = "Checked customer data and chat history; no readings ruled out."

    return {
        "readings": kept,
        "candidate_intents": candidate_intents,
        "context_used": context_used,
        "resolved_by_context": resolved_by_context,
    }
