import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from decision_engine import DecisionResult, decide_action
from intent_classifier import IntentAnalysisResult, classify_intent
from intent_embeddings import compute_ambiguity_score, compute_similarity_margin
from intent_map import get_map_data, steer_query

# Load environment variables from root .env (fallback to backend/.env)
ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV if ROOT_ENV.exists() else None)

POLICY_FILE = Path(__file__).resolve().parent / "data" / "policy.json"
CUSTOMERS_FILE = Path(__file__).resolve().parent / "data" / "mock_customers.json"

decisions_log: list[Dict[str, Any]] = []
corrections_log: list[Dict[str, Any]] = []

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


class DecideEndpointRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    clarification_choice: Optional[str] = None


class SteerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    source_intent: str = Field(..., min_length=1)
    target_intent: str = Field(..., min_length=1)
    alpha: float = Field(default=0.5, ge=-1.0, le=2.0)


class CorrectionRequest(BaseModel):
    query: str = Field(..., min_length=1)
    target_intent: str = Field(..., min_length=1)
    alpha: float = Field(..., ge=-1.0, le=2.0)
    source_intent: Optional[str] = None


def load_customer(customer_id: str) -> Dict[str, Any]:
    if not CUSTOMERS_FILE.exists():
        raise HTTPException(status_code=404, detail="Mock customers file not found")
    with CUSTOMERS_FILE.open("r", encoding="utf-8") as f:
        customers = json.load(f).get("customers", [])
    for customer in customers:
        if customer.get("customer_id") == customer_id:
            return customer
    raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")


def execute_mock_action(intent: str, customer_id: str) -> str:
    """Execute a mocked backend tool action for an approved intent."""
    return f"Success: '{intent}' action completed for customer {customer_id}."


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


@app.post("/decide")
async def decide_pipeline(request: DecideEndpointRequest):
    """Classify customer query, apply risk policy, log decision, and run mock action if ACT."""
    customer = load_customer(request.customer_id)
    chat_history = customer.get("chat_history", [])
    customer_data = {k: v for k, v in customer.items() if k != "chat_history"}

    effective_query = request.query
    clarification_count = 0
    if request.clarification_choice and request.clarification_choice.strip():
        clarification_count = 1
        effective_query = (
            f"{request.query} (User selected clarification: {request.clarification_choice.strip()})"
        )

    try:
        analysis = await classify_intent(
            query=effective_query,
            customer_data=customer_data,
            chat_history=chat_history,
        )
        decision_res = decide_action(
            analysis=analysis,
            clarification_count=clarification_count,
        )
        margin_info = compute_similarity_margin(effective_query)
        margin = margin_info["margin"]
        ambiguity_score = compute_ambiguity_score(
            top_llm_confidence=decision_res["confidence"],
            margin=margin,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"LLM provider error: {exc.response.text}",
        ) from exc

    clarification = None
    if decision_res["decision"] == "ASK":
        clarification = {
            "question": decision_res["clarification_question"],
            "options": decision_res["options"],
        }

    action_result = None
    if decision_res["decision"] == "ACT":
        action_result = execute_mock_action(decision_res["intent"], request.customer_id)

    response_payload = {
        "is_ambiguous": analysis["is_ambiguous"],
        "type": analysis["ambiguity_type"],
        "candidates": analysis["candidate_intents"],
        "risk_tier": decision_res["risk_tier"],
        "threshold": decision_res["threshold"],
        "decision": decision_res["decision"],
        "reason": decision_res["reason"],
        "clarification": clarification,
        "action_result": action_result,
        "margin": margin,
        "ambiguity_score": ambiguity_score,
    }

    decisions_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_id": request.customer_id,
            "query": request.query,
            "clarification_choice": request.clarification_choice,
            **response_payload,
        }
    )

    return response_payload


@app.get("/decisions")
def get_decisions():
    """Return all logged decisions from POST /decide calls."""
    return {"decisions": decisions_log}


@app.get("/customers")
def get_customers():
    """Return mock customers for the Live UI."""
    if not CUSTOMERS_FILE.exists():
        return {"customers": []}
    with CUSTOMERS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/steer")
def steer(request: SteerRequest):
    """Steer a query embedding from source_intent toward target_intent and return 2D coords + probabilities."""
    try:
        return steer_query(
            query=request.query,
            source_intent=request.source_intent,
            target_intent=request.target_intent,
            alpha=request.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/corrections")
def save_correction(request: CorrectionRequest):
    """Store a query steering correction in the corrections list."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": request.query,
        "source_intent": request.source_intent,
        "target_intent": request.target_intent,
        "alpha": request.alpha,
    }
    corrections_log.append(entry)
    return {"saved": entry, "count": len(corrections_log)}


@app.get("/corrections")
def get_corrections():
    """Return all saved steering corrections."""
    return {"corrections": corrections_log}


@app.get("/results")
def get_results():
    """Return evaluation results from backend/data/results.json."""
    results_path = Path(__file__).resolve().parent / "data" / "results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="results.json not found")
    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/calibration-plot")
def get_calibration_plot():
    """Serve the generated calibration_plot.png image."""
    plot_path = Path(__file__).resolve().parent / "data" / "calibration_plot.png"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="calibration_plot.png not found")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/map")
def get_intent_map(
    query: Optional[str] = None,
    steered_query: Optional[str] = None,
    clarification_choice: Optional[str] = None,
):
    """Return 2D PCA projection of example queries colored by intent plus before/after query points."""
    return get_map_data(
        query=query,
        steered_query=steered_query,
        clarification_choice=clarification_choice,
    )
