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
    target_reading: str,
    alpha: float,
    readings: Optional[list[Any]] = None,
    intents_path: Path = INTENTS_FILE,
    cache_path: Path = MAP_CACHE_FILE,
) -> dict[str, Any]:
    """Steer query_embedding toward target_reading_embedding by alpha and return similarities to every reading."""
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    if not target_reading or not target_reading.strip():
        raise ValueError("target_reading must be non-empty")

    projection = fit_or_load_projection(intents_path=intents_path, cache_path=cache_path)
    mean = np.asarray(projection["mean"], dtype=np.float32)
    components = np.asarray(projection["components"], dtype=np.float32)

    model = get_model()
    clean_query = query.strip()
    clean_target = target_reading.strip()

    q_vec = _l2_normalize(
        np.asarray(model.encode([clean_query], normalize_embeddings=True)[0], dtype=np.float32)
    )
    t_vec = _l2_normalize(
        np.asarray(model.encode([clean_target], normalize_embeddings=True)[0], dtype=np.float32)
    )
    steered_vec = _l2_normalize(q_vec + float(alpha) * (t_vec - q_vec))

    xy_before = _project_array(q_vec, mean, components)
    xy_after = _project_array(steered_vec, mean, components)

    reading_items: list[dict[str, Any]] = []
    if readings:
        for r in readings:
            if isinstance(r, dict):
                reading_items.append(
                    {
                        "paraphrase": str(r.get("paraphrase", "")).strip(),
                        "intent": r.get("intent"),
                    }
                )
            else:
                reading_items.append({"paraphrase": str(r).strip(), "intent": None})
    else:
        reading_items.append({"paraphrase": clean_target, "intent": None})

    paraphrases = [item["paraphrase"] for item in reading_items if item["paraphrase"]]
    r_vecs = np.asarray(model.encode(paraphrases, normalize_embeddings=True), dtype=np.float32)

    before_similarities = []
    after_similarities = []
    for item, r_vec in zip(reading_items, r_vecs):
        rv = _l2_normalize(r_vec)
        before_sim = round(float(np.dot(q_vec, rv)), 4)
        after_sim = round(float(np.dot(steered_vec, rv)), 4)
        before_similarities.append(
            {
                "paraphrase": item["paraphrase"],
                "intent": item["intent"],
                "similarity": before_sim,
            }
        )
        after_similarities.append(
            {
                "paraphrase": item["paraphrase"],
                "intent": item["intent"],
                "similarity": after_sim,
            }
        )

    after_similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "query": clean_query,
        "target_reading": clean_target,
        "alpha": float(alpha),
        "old_point": {"x": round(float(xy_before[0]), 4), "y": round(float(xy_before[1]), 4)},
        "new_point": {"x": round(float(xy_after[0]), 4), "y": round(float(xy_after[1]), 4)},
        "before_similarities": before_similarities,
        "similarities": after_similarities,
    }
