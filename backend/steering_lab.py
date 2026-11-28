"""
steering_lab.py – Frozen sentence-transformer steering lab over the training split.

- Loads a frozen sentence-transformer (`sentence-transformers/all-MiniLM-L6-v2`).
- Embeds `data/train.csv` once and caches embeddings + 2D PCA projection on disk.
- `/lab/init`: analyzes sentence + optional context, computes 2D points, reading
  similarities, nearby training examples, and automatic alpha.
- `/lab/steer`: computes `steered = normalize((1 - alpha) * e_sentence + alpha * e_target)`,
  retrieves nearest training examples & votes, fuses reading probabilities, calibrates
  confidence, routes with `route_record`, and rewrites deterministically (NO LLM call).
"""

from __future__ import annotations

import csv
import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from detectors import detect_all
from resolver import resolve_batch, route_record
from rewrite import rewrite_sentence

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT_DIR / "data" / "train.csv"
CACHE_DIR = Path(__file__).resolve().parent / "lab_cache"
EMB_CACHE = CACHE_DIR / "train_embeddings.npy"
COORDS_CACHE = CACHE_DIR / "train_coords_2d.npy"
PROJ_CACHE = CACHE_DIR / "pca_2d.joblib"
META_CACHE = CACHE_DIR / "train_meta.json"

FROZEN_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STEER_SOFTMAX_TEMP = 0.5

_FROZEN_TOKENIZER: Any = None
_FROZEN_MODEL: Any = None
_FALLBACK_TFIDF: Optional[TfidfVectorizer] = None

_TRAIN_EMB: Optional[np.ndarray] = None
_TRAIN_COORDS: Optional[np.ndarray] = None
_TRAIN_META: Optional[list[dict[str, Any]]] = None
_PROJ_2D: Optional[PCA] = None

_LAB_STORE: dict[str, dict[str, Any]] = {}


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(mat).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def _l2_normalize_1d(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return (vec / n).astype(np.float32) if n > 1e-9 else vec.astype(np.float32)


def _load_frozen_encoder() -> tuple[Any, Any]:
    """Load frozen sentence-transformer weights once (no gradients, eval mode)."""
    global _FROZEN_TOKENIZER, _FROZEN_MODEL
    if _FROZEN_TOKENIZER is None or _FROZEN_MODEL is None:
        from transformers import AutoModel, AutoTokenizer

        _FROZEN_TOKENIZER = AutoTokenizer.from_pretrained(FROZEN_MODEL_NAME)
        _FROZEN_MODEL = AutoModel.from_pretrained(FROZEN_MODEL_NAME)
        _FROZEN_MODEL.eval()
        for p in _FROZEN_MODEL.parameters():
            p.requires_grad_(False)
    return _FROZEN_TOKENIZER, _FROZEN_MODEL


def embed_texts_frozen(texts: list[str]) -> np.ndarray:
    """Embed texts using the frozen sentence-transformer (or cached TF-IDF fallback)."""
    global _FALLBACK_TFIDF
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    try:
        import torch

        tokenizer, model = _load_frozen_encoder()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                pooled = torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                chunks.append(pooled.cpu().numpy())
        return _l2_normalize(np.vstack(chunks))
    except Exception:
        if _FALLBACK_TFIDF is None:
            _FALLBACK_TFIDF = TfidfVectorizer(max_features=384)
            _FALLBACK_TFIDF.fit(texts)
        return _l2_normalize(_FALLBACK_TFIDF.transform(texts).toarray())


def _load_train_rows() -> list[dict[str, Any]]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training split not found: {TRAIN_CSV}")
    rows: list[dict[str, Any]] = []
    with TRAIN_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            types = json.loads(r.get("types") or '["none"]')
            rows.append({
                "id": int(r["id"]),
                "text": r["text"],
                "context": (r.get("context") or "").strip(),
                "types": types,
                "ambiguity_type": types[0] if types else "none",
                "gold_reading": r["gold_reading"],
                "is_ambiguous": str(r.get("is_ambiguous", "false")).lower() == "true",
            })
    return rows


def ensure_train_index() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], PCA]:
    """Embed `data/train.csv` once, fit 2D PCA once, and cache both on disk."""
    global _TRAIN_EMB, _TRAIN_COORDS, _TRAIN_META, _PROJ_2D
    if _TRAIN_EMB is not None and _TRAIN_COORDS is not None and _TRAIN_META is not None and _PROJ_2D is not None:
        return _TRAIN_EMB, _TRAIN_COORDS, _TRAIN_META, _PROJ_2D

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if EMB_CACHE.exists() and COORDS_CACHE.exists() and PROJ_CACHE.exists() and META_CACHE.exists():
        _TRAIN_EMB = np.load(str(EMB_CACHE))
        _TRAIN_COORDS = np.load(str(COORDS_CACHE))
        _PROJ_2D = joblib.load(str(PROJ_CACHE))
        _TRAIN_META = json.loads(META_CACHE.read_text(encoding="utf-8"))
        return _TRAIN_EMB, _TRAIN_COORDS, _TRAIN_META, _PROJ_2D

    rows = _load_train_rows()
    texts = [r["text"] for r in rows]
    emb = embed_texts_frozen(texts)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb).astype(np.float32)

    np.save(str(EMB_CACHE), emb)
    np.save(str(COORDS_CACHE), coords)
    joblib.dump(pca, str(PROJ_CACHE))
    META_CACHE.write_text(json.dumps(rows), encoding="utf-8")

    _TRAIN_EMB, _TRAIN_COORDS, _TRAIN_META, _PROJ_2D = emb, coords, rows, pca
    return _TRAIN_EMB, _TRAIN_COORDS, _TRAIN_META, _PROJ_2D


def _project_2d(vec_or_mat: np.ndarray, pca: PCA) -> list[list[float]]:
    pts = pca.transform(np.atleast_2d(vec_or_mat))
    return [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in pts]


def _nearest_train(
    vec: np.ndarray,
    train_emb: np.ndarray,
    train_coords: np.ndarray,
    train_meta: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    sims = train_emb @ _l2_normalize_1d(vec)
    order = np.argsort(-sims)[:top_k]
    out: list[dict[str, Any]] = []
    for idx in order:
        i = int(idx)
        m = train_meta[i]
        out.append({
            "id": m["id"],
            "text": m["text"],
            "types": m["types"],
            "ambiguity_type": m["ambiguity_type"],
            "gold_reading": m["gold_reading"],
            "is_ambiguous": m["is_ambiguous"],
            "similarity": round(float(sims[i]), 4),
            "point_2d": [round(float(train_coords[i, 0]), 4), round(float(train_coords[i, 1]), 4)],
        })
    return out


async def lab_init(sentence: str, context: Optional[str] = None) -> dict[str, Any]:
    """Initialize steering lab state for a user sentence + optional context."""
    train_emb, train_coords, train_meta, pca = ensure_train_index()

    resolved_list = await resolve_batch([sentence], explicit_context=context)
    rec = resolved_list[0]
    ctx_items = [l.strip() for l in (context or "").splitlines() if l.strip()]

    # Extract ambiguous span from linguistic detectors or rewrite changes
    det = detect_all(sentence, context=context or "")
    amb_type = rec["ambiguity_type"]
    span = ""
    if amb_type != "none" and amb_type in det and det[amb_type].get("span"):
        span = str(det[amb_type]["span"])
    elif rec.get("changes"):
        span = str(rec["changes"][0].get("span") or "")

    # Build candidate readings + normalized probabilities
    raw_readings: list[dict[str, Any]] = []
    seen_r: set[str] = set()
    top_prob = max(0.05, float(rec["confidence"]))
    raw_readings.append({"reading": rec["top_reading"], "probability": top_prob})
    seen_r.add(rec["top_reading"].strip().lower())
    for alt in rec.get("alternatives", []):
        alt_r = str(alt.get("reading", "")).strip()
        key = alt_r.lower()
        if key and key not in seen_r:
            seen_r.add(key)
            raw_readings.append({"reading": alt_r, "probability": max(0.01, float(alt.get("probability", 0.1)))})

    total_p = sum(r["probability"] for r in raw_readings) or 1.0
    for r in raw_readings:
        r["probability"] = round(r["probability"] / total_p, 4)

    reading_texts = [r["reading"].strip() for r in raw_readings]
    vecs = embed_texts_frozen([sentence] + reading_texts)
    e_sentence = vecs[0]
    e_readings = vecs[1:]

    sentence_2d = _project_2d(e_sentence, pca)[0]
    readings_2d = _project_2d(e_readings, pca)

    readings_out: list[dict[str, Any]] = []
    for r, er, pt in zip(raw_readings, e_readings, readings_2d):
        readings_out.append({
            "reading": r["reading"],
            "probability": r["probability"],
            "embedding_similarity": round(float(np.dot(e_sentence, er)), 4),
            "point_2d": pt,
        })

    nearby = _nearest_train(e_sentence, train_emb, train_coords, train_meta, top_k=5)
    auto_alpha = 0.0 if (amb_type == "none" and rec.get("status") == "AUTO_ACCEPT") else round(max(0.15, min(0.85, 1.0 - float(rec["confidence"]))), 3)

    sentence_id = uuid.uuid4().hex[:12]
    _LAB_STORE[sentence_id] = {
        "sentence": sentence,
        "context_items": ctx_items,
        "ambiguity_type": amb_type,
        "conflict": bool(rec.get("conflict", False)),
        "base_confidence": float(rec["confidence"]),
        "readings": readings_out,
        "e_sentence": e_sentence,
        "e_readings": e_readings,
    }

    return {
        "sentence_id": sentence_id,
        "sentence": sentence,
        "ambiguity_type": amb_type,
        "ambiguous_span": span,
        "readings": readings_out,
        "sentence_point": sentence_2d,
        "reading_points": [r["point_2d"] for r in readings_out],
        "nearby_training_examples": nearby,
        "auto_alpha": auto_alpha,
    }


def lab_steer(sentence_id: str, target_reading: str, alpha: float) -> dict[str, Any]:
    """
    Steer sentence embedding toward `target_reading` with strength `alpha`.
    No LLM calls; retrieves only from the cached training split.
    """
    state = _LAB_STORE.get(sentence_id)
    if state is None:
        raise KeyError(f"Unknown sentence_id '{sentence_id}'. Call POST /lab/init first.")

    train_emb, train_coords, train_meta, pca = ensure_train_index()
    e_sentence: np.ndarray = state["e_sentence"]
    e_readings: np.ndarray = state["e_readings"]
    readings: list[dict[str, Any]] = state["readings"]

    # Resolve target embedding from cached readings or frozen embedder
    target_idx = next(
        (i for i, r in enumerate(readings) if r["reading"].strip().lower() == target_reading.strip().lower()),
        None,
    )
    e_target = e_readings[target_idx] if target_idx is not None else embed_texts_frozen([target_reading])[0]

    a = max(0.0, min(1.0, float(alpha)))
    steered = _l2_normalize_1d((1.0 - a) * e_sentence + a * e_target)

    # Similarities to all candidate readings
    sims_to_readings = [float(np.dot(steered, er)) for er in e_readings]
    reading_sims = [
        {"reading": r["reading"], "similarity": round(s, 4)}
        for r, s in zip(readings, sims_to_readings)
    ]

    # Nearest resolved training examples and their votes
    nearest_train = _nearest_train(steered, train_emb, train_coords, train_meta, top_k=5)
    vote_counts: dict[str, float] = {}
    for ex in nearest_train:
        t = ex["ambiguity_type"]
        vote_counts[t] = vote_counts.get(t, 0.0) + max(0.0, ex["similarity"])
    votes = [
        {"ambiguity_type": k, "weight": round(v, 4)}
        for k, v in sorted(vote_counts.items(), key=lambda kv: -kv[1])
    ]

    # Z-score cosine similarities across readings, then apply softmax with STEER_SOFTMAX_TEMP (0.5)
    sims_arr = np.array(sims_to_readings, dtype=np.float64)
    if len(sims_arr) > 1:
        std = float(np.std(sims_arr))
        z_sims = (sims_arr - float(np.mean(sims_arr))) / std if std > 1e-8 else (sims_arr - float(np.mean(sims_arr)))
    else:
        z_sims = np.zeros_like(sims_arr)

    logits = z_sims / STEER_SOFTMAX_TEMP
    logits -= np.max(logits)
    sim_probs = np.exp(logits)
    sim_probs /= np.sum(sim_probs)

    llm_probs = np.array([r["probability"] for r in readings], dtype=np.float64)
    fused = (1.0 - a) * llm_probs + a * sim_probs
    fused /= np.sum(fused)

    fused_readings = [
        {"reading": r["reading"], "probability": round(float(p), 4), "_is_target": (i == target_idx)}
        for i, (r, p) in enumerate(zip(readings, fused))
    ]
    fused_readings.sort(key=lambda x: (-x["probability"], not x["_is_target"]))
    for item in fused_readings:
        item.pop("_is_target", None)

    winning_reading = fused_readings[0]["reading"]
    win_prob = float(fused_readings[0]["probability"])

    # Recompute confidence and routing status directly from fused values
    calibrated_conf = round(max(0.0, min(1.0, win_prob)), 4)
    status = route_record(confidence=calibrated_conf, conflict=bool(state["conflict"] and a < 0.5))
    rw = rewrite_sentence(
        text=state["sentence"],
        ambiguity_type=state["ambiguity_type"],
        top_reading=winning_reading,
        alternatives=fused_readings[1:],
        confidence=calibrated_conf,
        context_items=state["context_items"],
    )
    steered_2d = _project_2d(steered, pca)[0]

    return {
        "sentence_id": sentence_id,
        "alpha": round(a, 4),
        "target_reading": target_reading,
        "similarities_to_readings": reading_sims,
        "nearest_training_examples": nearest_train,
        "training_votes": votes,
        "fused_reading_probabilities": fused_readings,
        "winning_reading": winning_reading,
        "calibrated_confidence": calibrated_conf,
        "status": status,
        "rewrite": rw,
        "steered_point_2d": steered_2d,
    }
