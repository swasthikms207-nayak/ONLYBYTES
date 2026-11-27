import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV if ROOT_ENV.exists() else None)

INTENTS_FILE = Path(__file__).resolve().parent / "data" / "intents.json"

AmbiguityType = Literal[
    "referential",
    "underspecified",
    "scope",
    "lexical",
    "pragmatic",
    "none",
]

IntentName = Literal[
    "payment_reversal",
    "cancel_order",
    "refund_status",
    "account_closure",
    "change_address",
    "card_block",
    "order_status",
    "transaction_dispute",
]


class CandidateIntent(BaseModel):
    intent: IntentName
    confidence: float = Field(..., ge=0.0, le=1.0)


class IntentAnalysisResult(BaseModel):
    is_ambiguous: bool
    ambiguity_type: AmbiguityType
    candidate_intents: list[CandidateIntent] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_and_normalize(self) -> "IntentAnalysisResult":
        if not self.is_ambiguous:
            self.ambiguity_type = "none"

        total = sum(item.confidence for item in self.candidate_intents)
        if total <= 0:
            equal_weight = round(1.0 / len(self.candidate_intents), 4)
            for item in self.candidate_intents:
                item.confidence = equal_weight
        else:
            for item in self.candidate_intents:
                item.confidence = round(item.confidence / total, 4)

        # Adjust rounding drift on the highest-confidence candidate so sum == 1.0
        drift = round(1.0 - sum(item.confidence for item in self.candidate_intents), 4)
        if drift != 0.0:
            top_idx = max(
                range(len(self.candidate_intents)),
                key=lambda i: self.candidate_intents[i].confidence,
            )
            self.candidate_intents[top_idx].confidence = round(
                self.candidate_intents[top_idx].confidence + drift, 4
            )

        self.candidate_intents.sort(key=lambda c: c.confidence, reverse=True)
        return self


def _load_intent_catalog() -> list[dict[str, str]]:
    if not INTENTS_FILE.exists():
        return []
    with INTENTS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {"intent": item["intent"], "description": item["description"]}
        for item in data.get("intents", [])
    ]


def _build_system_prompt() -> str:
    catalog = _load_intent_catalog()
    intents_block = "\n".join(
        f"- {item['intent']}: {item['description']}" for item in catalog
    )
    return (
        "You are a bilingual (English and Hinglish) fintech and e-commerce support intent classifier.\n"
        "Analyze the user's latest query using the provided customer data (pending transactions, pending orders, "
        "account details) and recent chat history.\n\n"
        "Allowed intents:\n"
        f"{intents_block}\n\n"
        "Ambiguity types:\n"
        "- referential: pronoun or reference (e.g. 'it', 'that', 'uska', 'ye wala') could point to multiple orders/transactions.\n"
        "- underspecified: missing key target or detail needed to act safely.\n"
        "- scope: unclear how many items/transactions/accounts the request applies to.\n"
        "- lexical: a word or Hinglish phrase has multiple meanings (e.g. 'band karo' -> card_block vs account_closure).\n"
        "- pragmatic: indirect/sarcastic/implicit request whose literal meaning differs from the user's goal.\n"
        "- none: query is clear and unambiguous (use only when is_ambiguous is false).\n\n"
        "Return STRICT JSON only (no markdown, no extra text) matching this exact schema:\n"
        "{\n"
        '  "is_ambiguous": boolean,\n'
        '  "ambiguity_type": "referential" | "underspecified" | "scope" | "lexical" | "pragmatic" | "none",\n'
        '  "candidate_intents": [\n'
        '    {"intent": "<allowed_intent>", "confidence": <float between 0 and 1>}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "1. Understand both English and Hinglish (Romanized Hindi + English mix).\n"
        "2. All confidence values in candidate_intents MUST sum to 1.0.\n"
        "3. If is_ambiguous is false, set ambiguity_type to 'none'."
    )


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


async def classify_intent(
    query: str,
    customer_data: dict[str, Any],
    chat_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify an English or Hinglish support query into strict intent & ambiguity JSON."""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

    if not api_key or api_key in {"your_llm_api_key_here", "sk-your-key-here"}:
        raise RuntimeError("LLM_API_KEY is not configured in .env")

    user_payload = {
        "query": query,
        "customer_data": customer_data,
        "chat_history": chat_history,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

    parsed = json.loads(_strip_code_fences(content))
    validated = IntentAnalysisResult(**parsed)
    return validated.model_dump()
