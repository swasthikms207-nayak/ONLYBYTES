import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

POLICY_FILE = Path(__file__).resolve().parent / "data" / "policy.json"

INTENT_LABELS: dict[str, str] = {
    "payment_reversal": "Reverse a duplicate or failed payment",
    "cancel_order": "Cancel an existing order",
    "refund_status": "Check refund status",
    "account_closure": "Close or deactivate my account",
    "change_address": "Update delivery or billing address",
    "card_block": "Block or freeze my card",
    "order_status": "Track or check order status",
    "transaction_dispute": "Dispute an unauthorized transaction",
}

DecisionType = Literal["ACT", "ASK", "ESCALATE"]


class ClarificationOption(BaseModel):
    intent: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class DecisionResult(BaseModel):
    decision: DecisionType
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    risk_tier: Literal["low", "medium", "high"]
    reason: str
    clarification_question: Optional[str] = None
    options: list[ClarificationOption] = Field(default_factory=list)


def _load_policies(policy_path: Path = POLICY_FILE) -> dict[str, dict[str, Any]]:
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    with policy_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("policies", {})


def decide_action(
    analysis: dict[str, Any],
    clarification_count: int = 0,
    policy_path: Path = POLICY_FILE,
) -> dict[str, Any]:
    """Decide whether to ACT, ASK, or ESCALATE based on top intent confidence vs policy threshold."""
    candidates = analysis.get("candidate_intents") or []
    if not candidates:
        raise ValueError("analysis must contain at least one candidate intent")

    sorted_candidates = sorted(
        candidates, key=lambda c: float(c["confidence"]), reverse=True
    )
    top = sorted_candidates[0]
    top_intent = str(top["intent"])
    top_confidence = float(top["confidence"])

    policies = _load_policies(policy_path)
    if top_intent not in policies:
        raise ValueError(f"No policy found for intent '{top_intent}'")

    policy = policies[top_intent]
    threshold = float(policy["confidence_threshold"])
    risk_tier = policy["risk_tier"]

    if top_confidence >= threshold:
        return DecisionResult(
            decision="ACT",
            intent=top_intent,
            confidence=top_confidence,
            threshold=threshold,
            risk_tier=risk_tier,
            reason=(
                f"Top intent '{top_intent}' confidence ({top_confidence:.2f}) "
                f"meets {risk_tier}-risk threshold ({threshold:.2f})."
            ),
        ).model_dump()

    if clarification_count >= 1:
        return DecisionResult(
            decision="ESCALATE",
            intent=top_intent,
            confidence=top_confidence,
            threshold=threshold,
            risk_tier=risk_tier,
            reason=(
                f"Confidence ({top_confidence:.2f}) for '{top_intent}' remains below "
                f"threshold ({threshold:.2f}) after {clarification_count} clarification."
            ),
        ).model_dump()

    options = [
        ClarificationOption(
            intent=str(c["intent"]),
            label=INTENT_LABELS.get(
                str(c["intent"]), str(c["intent"]).replace("_", " ").title()
            ),
            confidence=float(c["confidence"]),
        )
        for c in sorted_candidates[:3]
    ]
    if len(options) == 1:
        options.append(
            ClarificationOption(
                intent="escalate_agent",
                label="Something else / Speak to support",
                confidence=round(max(0.0, 1.0 - top_confidence), 4),
            )
        )

    return DecisionResult(
        decision="ASK",
        intent=top_intent,
        confidence=top_confidence,
        threshold=threshold,
        risk_tier=risk_tier,
        reason=(
            f"Confidence ({top_confidence:.2f}) for '{top_intent}' is below "
            f"{risk_tier}-risk threshold ({threshold:.2f}); asking one clarification."
        ),
        clarification_question="Just to be sure, what would you like help with?",
        options=options,
    ).model_dump()
