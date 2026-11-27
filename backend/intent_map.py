import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA

from intent_embeddings import (
    DATA_DIR,
    INTENTS_FILE,
    MODEL_NAME,
    _l2_normalize,
    compute_and_cache_centroids,
    get_model,
)

MAP_CACHE_FILE = DATA_DIR / "intent_map_pca.json"

INTENT_COLORS: dict[str, str] = {
    "payment_reversal": "#ef4444",
    "cancel_order": "#f97316",
    "refund_status": "#eab308",
    "account_closure": "#8b5cf6",
    "change_address": "#06b6d4",
    "card_block": "#ec4899",
    "order_status": "#22c55e",
    "transaction_dispute": "#3b82f6",
}

_cached_projection: Optional[dict[str, Any]] = None


def _project_array(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (vectors - mean) @ components.T


def fit_or_load_projection(
    intents_path: Path = INTENTS_FILE,
    cache_path: Path = MAP_CACHE_FILE,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Fit 2D PCA once on all example queries and cache the projection + points to disk."""
    global _cached_projection

    if _cached_projection is not None and not force_recompute and cache_path == MAP_CACHE_FILE:
        return _cached_projection

    if cache_path.exists() and not force_recompute:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("model_name") == MODEL_NAME and "components" in data and "mean" in data:
            if cache_path == MAP_CACHE_FILE:
                _cached_projection = data
            return data

    if not intents_path.exists():
        raise FileNotFoundError(f"Intents file not found: {intents_path}")

    with intents_path.open("r", encoding="utf-8") as f:
        intents_data = json.load(f).get("intents", [])

    texts: list[str] = []
    intents: list[str] = []
    for item in intents_data:
        intent = item["intent"]
        for ex in item.get("examples", []):
            texts.append(ex)
            intents.append(intent)

    if len(texts) < 2:
        raise ValueError("At least 2 example queries are required to fit 2D PCA.")

    model = get_model()
    embeddings = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    points = [
        {
            "intent": intent,
            "color": INTENT_COLORS.get(intent, "#a1a1aa"),
            "text": text,
            "x": round(float(coord[0]), 4),
            "y": round(float(coord[1]), 4),
        }
        for intent, text, coord in zip(intents, texts, coords)
    ]

    payload = {
        "model_name": MODEL_NAME,
        "mean": pca.mean_.astype(float).tolist(),
        "components": pca.components_.astype(float).tolist(),
        "points": points,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    if cache_path == MAP_CACHE_FILE:
        _cached_projection = payload
    return payload


def project_query_2d(query: str, projection: dict[str, Any]) -> dict[str, Any]:
    """Project a single query into the already-fitted 2D PCA space."""
    model = get_model()
    vec = np.asarray(model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
    mean = np.asarray(projection["mean"], dtype=np.float32)
    components = np.asarray(projection["components"], dtype=np.float32)
    xy = _project_array(vec, mean, components)
    return {
        "text": query,
        "x": round(float(xy[0]), 4),
        "y": round(float(xy[1]), 4),
    }


def get_map_data(
    query: Optional[str] = None,
    steered_query: Optional[str] = None,
    clarification_choice: Optional[str] = None,
    intents_path: Path = INTENTS_FILE,
    cache_path: Path = MAP_CACHE_FILE,
) -> dict[str, Any]:
    """Return 2D projected example points and optional before/after steering query positions."""
    projection = fit_or_load_projection(intents_path=intents_path, cache_path=cache_path)

    query_before = None
    query_after = None

    if query and query.strip():
        clean_query = query.strip()
        query_before = project_query_2d(clean_query, projection)

        effective_steered = None
        if steered_query and steered_query.strip():
            effective_steered = steered_query.strip()
        elif clarification_choice and clarification_choice.strip():
            effective_steered = (
                f"{clean_query} (User selected clarification: {clarification_choice.strip()})"
            )

        if effective_steered:
            query_after = project_query_2d(effective_steered, projection)

    return {
        "points": projection["points"],
        "query_before": query_before,
        "query_after": query_after,
    }


def steer_query(
    query: str,
    source_intent: str,
    target_intent: str,
    alpha: float,
    intents_path: Path = INTENTS_FILE,
    cache_path: Path = MAP_CACHE_FILE,
) -> dict[str, Any]:
    """Steer a query embedding from source_intent toward target_intent by alpha and return 2D points + probabilities."""
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    centroids = compute_and_cache_centroids(intents_path=intents_path)
    if source_intent not in centroids:
        raise ValueError(f"Unknown source_intent: {source_intent}")
    if target_intent not in centroids:
        raise ValueError(f"Unknown target_intent: {target_intent}")

    projection = fit_or_load_projection(intents_path=intents_path, cache_path=cache_path)
    mean = np.asarray(projection["mean"], dtype=np.float32)
    components = np.asarray(projection["components"], dtype=np.float32)

    model = get_model()
    vec_before = _l2_normalize(
        np.asarray(model.encode([query.strip()], normalize_embeddings=True)[0], dtype=np.float32)
    )
    delta = centroids[target_intent] - centroids[source_intent]
    vec_after = _l2_normalize(vec_before + float(alpha) * delta)

    xy_before = _project_array(vec_before, mean, components)
    xy_after = _project_array(vec_after, mean, components)

    # Softmax over cosine similarities (temperature=10) for updated intent probabilities
    intent_names = list(centroids.keys())
    sims = np.array([float(np.dot(vec_after, centroids[name])) for name in intent_names])
    scaled = sims * 10.0
    exp_scores = np.exp(scaled - np.max(scaled))
    probs = exp_scores / np.sum(exp_scores)

    probabilities = sorted(
        [
            {"intent": name, "probability": round(float(prob), 4)}
            for name, prob in zip(intent_names, probs)
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "query": query.strip(),
        "source_intent": source_intent,
        "target_intent": target_intent,
        "alpha": float(alpha),
        "old_point": {"x": round(float(xy_before[0]), 4), "y": round(float(xy_before[1]), 4)},
        "new_point": {"x": round(float(xy_after[0]), 4), "y": round(float(xy_after[1]), 4)},
        "probabilities": probabilities,
    }
