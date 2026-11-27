import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV if ROOT_ENV.exists() else None)

INTENTS_FILE = Path(__file__).resolve().parent / "data" / "intents.json"

SpanAmbiguityType = Literal[
    "lexical",
    "referential",
    "scope",
    "syntactic",
    "underspecified",
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


class AmbiguousSpan(BaseModel):
    text: str = Field(..., min_length=1)
    type: SpanAmbiguityType
    reason: str = Field(..., min_length=1)


class Reading(BaseModel):
    paraphrase: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    intent: IntentName


class CandidateIntent(BaseModel):
    intent: IntentName
    confidence: float = Field(..., ge=0.0, le=1.0)


class IntentAnalysisResult(BaseModel):
    ambiguous_spans: list[AmbiguousSpan] = Field(default_factory=list)
    readings: list[Reading] = Field(..., min_length=1, max_length=5)
    # Derived compatibility fields for /decide and decide_action
    is_ambiguous: Optional[bool] = None
    ambiguity_type: Optional[str] = None
    candidate_intents: Optional[list[CandidateIntent]] = None

    @model_validator(mode="after")
    def validate_and_normalize(self) -> "IntentAnalysisResult":
        if len(self.ambiguous_spans) == 0:
            # Clear sentence: empty spans list and a single reading with probability 1.0
            self.readings = [self.readings[0]]
            self.readings[0].probability = 1.0
            self.is_ambiguous = False
            self.ambiguity_type = "none"
        else:
            self.is_ambiguous = True
            self.ambiguity_type = self.ambiguous_spans[0].type
            total = sum(r.probability for r in self.readings)
            if total <= 0:
                equal_p = round(1.0 / len(self.readings), 4)
                for r in self.readings:
                    r.probability = equal_p
            else:
                for r in self.readings:
                    r.probability = round(r.probability / total, 4)

            drift = round(1.0 - sum(r.probability for r in self.readings), 4)
            if drift != 0.0:
                top_idx = max(range(len(self.readings)), key=lambda i: self.readings[i].probability)
                self.readings[top_idx].probability = round(
                    self.readings[top_idx].probability + drift, 4
                )
            self.readings.sort(key=lambda r: r.probability, reverse=True)

        # Aggregate readings into candidate_intents for /decide compatibility
        intent_probs: dict[str, float] = {}
        for r in self.readings:
            intent_probs[r.intent] = round(intent_probs.get(r.intent, 0.0) + r.probability, 4)

        cand_list = [
            CandidateIntent(intent=intent, confidence=prob)
            for intent, prob in sorted(intent_probs.items(), key=lambda kv: kv[1], reverse=True)
        ]
        cand_drift = round(1.0 - sum(c.confidence for c in cand_list), 4)
        if cand_drift != 0.0 and cand_list:
            cand_list[0].confidence = round(cand_list[0].confidence + cand_drift, 4)
        self.candidate_intents = cand_list
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
        "You are a bilingual (English and Hinglish) linguistic ambiguity detector for fintech and e-commerce support.\n"
        "Analyze the user's query at the language level (using optional customer data and chat history if provided).\n\n"
        "Allowed implied intents:\n"
        f"{intents_block}\n\n"
        "Span ambiguity types:\n"
        "- lexical: a word or Hinglish phrase has multiple meanings (e.g., 'band karo' -> card_block vs account_closure).\n"
        "- referential: a pronoun or deictic phrase (e.g., 'it', 'that', 'uska', 'ye wala') could refer to multiple entities.\n"
        "- scope: unclear quantifier or scope of action (e.g., 'all', 'sab', 'everything').\n"
        "- syntactic: structural/attachment ambiguity in how phrases modify each other.\n"
        "- underspecified: missing critical target or detail needed to determine one clear meaning.\n\n"
        "Return STRICT JSON only (no markdown, no extra text) matching this exact schema:\n"
        "{\n"
        '  "ambiguous_spans": [\n'
        '    {"text": "<exact span from query>", "type": "lexical|referential|scope|syntactic|underspecified", "reason": "<why it is ambiguous>"}\n'
        "  ],\n"
        '  "readings": [\n'
        '    {"paraphrase": "<explicit paraphrase of one meaning>", "probability": <float 0..1>, "intent": "<allowed_intent>"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "1. Handle both English and Hinglish (Romanized Hindi + English mix).\n"
        "2. For clear sentences: return an empty ambiguous_spans list [] and a SINGLE reading with probability 1.0.\n"
        "3. For ambiguous sentences: return 1+ ambiguous_spans and 3 to 5 explicit paraphrase readings whose probabilities sum to 1.0."
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
    customer_data: Optional[dict[str, Any]] = None,
    chat_history: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Detect language-level ambiguity spans and explicit readings for an English or Hinglish query."""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

    if not api_key or api_key in {"your_llm_api_key_here", "sk-your-key-here"}:
        raise RuntimeError("LLM_API_KEY is not configured in .env")

    user_payload = {
        "query": query,
        "customer_data": customer_data or {},
        "chat_history": chat_history or [],
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
