import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA

from intent_embeddings import DATA_DIR, INTENTS_FILE, MODEL_NAME, _l2_normalize, get_model

MAP_CACHE_FILE = DATA_DIR / "query_readings_pca.json"
PAST_QUERIES_CSV = DATA_DIR / "test_queries.csv"
CUSTOMERS_FILE = DATA_DIR / "mock_customers.json"

_cached_projection: Optional[dict[str, Any]] = None


def _project_array(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (vectors - mean) @ components.T


def _collect_past_pool(
    past_csv: Path = PAST_QUERIES_CSV,
    customers_path: Path = CUSTOMERS_FILE,
) -> list[str]:
    """Collect a pool of past queries and chat/reading texts to fit the 2D projection."""
    texts: list[str] = []
    if past_csv.exists():
        with past_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = str(row.get("query", "")).strip()
                if q:
                    texts.append(q)
    if customers_path.exists():
        with customers_path.open("r", encoding="utf-8") as f:
            for cust in json.load(f).get("customers", []):
                for turn in cust.get("chat_history", []):
                    msg = str(turn.get("message", "")).strip()
                    if msg:
                        texts.append(msg)
    return list(dict.fromkeys(texts))


def fit_or_load_projection(
    past_csv: Path = PAST_QUERIES_CSV,
    customers_path: Path = CUSTOMERS_FILE,
    cache_path: Path = MAP_CACHE_FILE,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Fit 2D PCA once on a pool of past queries/readings and cache the projection to disk."""
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

    texts = _collect_past_pool(past_csv=past_csv, customers_path=customers_path)
    if len(texts) < 2:
        raise ValueError("At least 2 past queries/readings are required to fit 2D PCA.")

    model = get_model()
    embeddings = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(embeddings)

    payload = {
        "model_name": MODEL_NAME,
        "pool_size": len(texts),
        "mean": pca.mean_.astype(float).tolist(),
        "components": pca.components_.astype(float).tolist(),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    if cache_path == MAP_CACHE_FILE:
        _cached_projection = payload
    return payload


def _project_vec_2d(vec: np.ndarray, projection: dict[str, Any]) -> dict[str, float]:
    mean = np.asarray(projection["mean"], dtype=np.float32)
    components = np.asarray(projection["components"], dtype=np.float32)
    xy = _project_array(vec, mean, components)
    return {
        "x": round(float(xy[0]), 4),
        "y": round(float(xy[1]), 4),
    }


def get_map_data(
    query: Optional[str] = None,
    readings: Optional[list[str]] = None,
    target_reading: Optional[str] = None,
    alpha: float = 0.5,
    cache_path: Path = MAP_CACHE_FILE,
) -> dict[str, Any]:
    """Project the query, its readings, and the steered query into the fitted 2D space."""
    projection = fit_or_load_projection(cache_path=cache_path)
    model = get_model()

    query_point = None
    reading_points: list[dict[str, Any]] = []
    steered_point = None

    q_vec = None
    if query and query.strip():
        clean_query = query.strip()
        q_vec = _l2_normalize(
            np.asarray(model.encode([clean_query], normalize_embeddings=True)[0], dtype=np.float32)
        )
        xy = _project_vec_2d(q_vec, projection)
        query_point = {"text": clean_query, **xy}

    if readings:
        clean_readings = [r.strip() for r in readings if r and r.strip()]
        if clean_readings:
            r_vecs = np.asarray(
                model.encode(clean_readings, normalize_embeddings=True), dtype=np.float32
            )
            for text, rv in zip(clean_readings, r_vecs):
                xy = _project_vec_2d(_l2_normalize(rv), projection)
                reading_points.append({"paraphrase": text, **xy})

    if q_vec is not None and target_reading and target_reading.strip():
        clean_target = target_reading.strip()
        t_vec = _l2_normalize(
            np.asarray(model.encode([clean_target], normalize_embeddings=True)[0], dtype=np.float32)
        )
        steered_vec = _l2_normalize(q_vec + float(alpha) * (t_vec - q_vec))
        xy = _project_vec_2d(steered_vec, projection)
        steered_point = {"text": clean_target, "alpha": float(alpha), **xy}

    return {
        "query_point": query_point,
        "reading_points": reading_points,
        "steered_point": steered_point,
    }


def steer_query(
    query: str,
    target_reading: str,
    alpha: float,
    readings: Optional[list[Any]] = None,
    cache_path: Path = MAP_CACHE_FILE,
) -> dict[str, Any]:
    """Steer query_embedding toward target_reading_embedding by alpha and return similarities to every reading."""
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    if not target_reading or not target_reading.strip():
        raise ValueError("target_reading must be non-empty")

    projection = fit_or_load_projection(cache_path=cache_path)
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

    old_xy = _project_vec_2d(q_vec, projection)
    new_xy = _project_vec_2d(steered_vec, projection)

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
        "old_point": old_xy,
        "new_point": new_xy,
        "before_similarities": before_similarities,
        "similarities": after_similarities,
    }
