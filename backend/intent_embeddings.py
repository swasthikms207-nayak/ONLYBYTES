import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = Path(__file__).resolve().parent / "data"
INTENTS_FILE = DATA_DIR / "intents.json"
CACHE_FILE = DATA_DIR / "intent_centroids.json"

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load and reuse the multilingual sentence-transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def compute_and_cache_centroids(
    intents_path: Path = INTENTS_FILE,
    cache_path: Path = CACHE_FILE,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Embed all example queries per intent, compute L2-normalized centroids, and cache to disk."""
    if cache_path.exists() and not force_recompute:
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("model_name") == MODEL_NAME and "centroids" in cached:
            return {
                intent: np.array(vec, dtype=np.float32)
                for intent, vec in cached["centroids"].items()
            }

    if not intents_path.exists():
        raise FileNotFoundError(f"Intents file not found: {intents_path}")

    with intents_path.open("r", encoding="utf-8") as f:
        intents_data = json.load(f).get("intents", [])

    model = get_model()
    centroids: dict[str, np.ndarray] = {}

    for item in intents_data:
        intent = item["intent"]
        examples = item.get("examples", [])
        if not examples:
            continue
        embeddings = model.encode(examples, normalize_embeddings=True)
        centroid = _l2_normalize(np.mean(embeddings, axis=0))
        centroids[intent] = centroid.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "centroids": {k: v.tolist() for k, v in centroids.items()},
            },
            f,
        )

    return centroids


def query_centroid_similarities(
    query: str,
    intents_path: Path = INTENTS_FILE,
    cache_path: Path = CACHE_FILE,
) -> dict[str, float]:
    """Embed a query and return cosine similarity to each intent centroid, sorted descending."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    centroids = compute_and_cache_centroids(
        intents_path=intents_path,
        cache_path=cache_path,
    )
    model = get_model()
    query_vec = _l2_normalize(
        np.asarray(model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
    )

    scores = {
        intent: round(float(np.dot(query_vec, centroid)), 4)
        for intent, centroid in centroids.items()
    }
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def compute_similarity_margin(
    query: str,
    intents_path: Path = INTENTS_FILE,
    cache_path: Path = CACHE_FILE,
) -> dict[str, Any]:
    """Return centroid similarities and the gap (margin) between the top two intents."""
    scores = query_centroid_similarities(
        query=query,
        intents_path=intents_path,
        cache_path=cache_path,
    )
    ranked = list(scores.items())
    top_intent, top_sim = ranked[0] if ranked else ("", 0.0)
    second_intent, second_sim = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = round(max(0.0, top_sim - second_sim), 4)
    return {
        "top_intent": top_intent,
        "second_intent": second_intent,
        "margin": margin,
        "similarities": scores,
    }


def compute_ambiguity_score(
    top_llm_confidence: float,
    margin: float,
    margin_scale: float = 0.25,
) -> float:
    """Combine LLM uncertainty (1 - conf) and embedding margin into a [0, 1] ambiguity score."""
    llm_uncertainty = max(0.0, min(1.0, 1.0 - float(top_llm_confidence)))
    normalized_margin = max(0.0, min(1.0, float(margin) / margin_scale))
    margin_ambiguity = 1.0 - normalized_margin
    score = 0.6 * llm_uncertainty + 0.4 * margin_ambiguity
    return round(max(0.0, min(1.0, score)), 4)
