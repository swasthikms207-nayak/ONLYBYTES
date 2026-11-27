import json
import os
from pathlib import Path
from typing import Dict, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
