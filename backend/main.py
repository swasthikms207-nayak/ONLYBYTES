import json
import os
from pathlib import Path
from typing import Any, Dict, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from decision_engine import DecisionResult, decide_action
from intent_classifier import IntentAnalysisResult, classify_intent

# Load environment variables from root .env (fallback to backend/.env)
ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV if ROOT_ENV.exists() else None)

POLICY_FILE = Path(__file__).resolve().parent / "data" / "policy.json"

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

PLACEHOLDER_KEYS = {"", "your_llm_api_key_here", "sk-your-key-here"}

app = FastAPI(title="ONLYBYTES API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt for the LLM")


class ChatResponse(BaseModel):
    reply: str
    model: str
    configured: bool


class IntentPolicy(BaseModel):
    risk_tier: Literal["low", "medium", "high"]
    cost_weight: Literal[1, 3, 10]
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)


class PolicyConfig(BaseModel):
    policies: Dict[str, IntentPolicy]


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1)
    customer_data: Dict[str, Any] = Field(default_factory=dict)
    chat_history: list[Dict[str, Any]] = Field(default_factory=list)


class DecideRequest(BaseModel):
    analysis: IntentAnalysisResult
    clarification_count: int = Field(default=0, ge=0)


def load_policy() -> PolicyConfig:
    if not POLICY_FILE.exists():
        raise HTTPException(status_code=404, detail="Policy file not found")
    with POLICY_FILE.open("r", encoding="utf-8") as f:
        return PolicyConfig(**json.load(f))


def save_policy(config: PolicyConfig) -> PolicyConfig:
    POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with POLICY_FILE.open("w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
        f.write("\n")
    return config


def is_llm_configured() -> bool:
    return LLM_API_KEY not in PLACEHOLDER_KEYS


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "llm_configured": is_llm_configured(),
        "model": LLM_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not is_llm_configured():
        return ChatResponse(
            reply=(
                "LLM_API_KEY is not configured yet. "
                "Add your API key to the root .env file and restart the backend."
            ),
            model=LLM_MODEL,
            configured=False,
        )

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": request.prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LLM_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            return ChatResponse(reply=reply, model=LLM_MODEL, configured=True)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"LLM provider error: {exc.response.text}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Request failed: {exc}") from exc


@app.get("/api/policy", response_model=PolicyConfig)
def get_policy():
    """Return the current intent risk, cost, and confidence threshold policy."""
    return load_policy()


@app.put("/api/policy", response_model=PolicyConfig)
def update_policy(config: PolicyConfig):
    """Validate and persist an updated intent policy configuration."""
    return save_policy(config)


@app.post("/api/classify", response_model=IntentAnalysisResult)
async def classify(request: ClassifyRequest):
    """Classify a customer query (English or Hinglish) into strict intent & ambiguity JSON."""
    try:
        return await classify_intent(
            query=request.query,
            customer_data=request.customer_data,
            chat_history=request.chat_history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"LLM provider error: {exc.response.text}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc


@app.post("/api/decide", response_model=DecisionResult)
def decide(request: DecideRequest):
    """Return ACT, ASK, or ESCALATE based on top candidate confidence vs policy threshold."""
    try:
        return decide_action(
            analysis=request.analysis.model_dump(),
            clarification_count=request.clarification_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
