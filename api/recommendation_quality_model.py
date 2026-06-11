"""
Recommendation quality prediction model - inference wrapper.

Loads the pre-trained XGBoost / LightGBM model and exposes a simple
``predict_quality`` function that returns a dict with:

    ml_quality_score   : float  0-1, lower = better quality
    ml_weak_probability: float  0-1, probability the recommendation is weak/bad
    ml_label           : str    'good' | 'weak' | 'uncertain'

The model lives at output/ml_models/recommendation_quality_model.pkl.
If the file is missing the module degrades gracefully - all calls return
a sentinel dict so the caller can safely ignore ML signals.

Phase: Filter Mode (default)
  - Score is computed and weak recommendations can be suppressed.
  - Set RECOMMENDATION_QUALITY_ML_PHASE = "shadow" to log scores without filtering.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _REPO_ROOT / "output" / "ml_models" / "recommendation_quality_model.pkl"

# ---------------------------------------------------------------------------
# Phase control (env-var override)
# ---------------------------------------------------------------------------

RECOMMENDATION_QUALITY_ML_PHASE: str = os.environ.get(
    "RECOMMENDATION_QUALITY_ML_PHASE", "filter"
).lower()

# Probability threshold above which a recommendation is flagged as weak.
DEFAULT_WEAK_PROBABILITY_THRESHOLD: float = 0.834
_ENV_WEAK_PROBABILITY_THRESHOLD = os.environ.get("RECOMMENDATION_QUALITY_WEAK_THRESHOLD")
WEAK_PROBABILITY_THRESHOLD: float = float(
    _ENV_WEAK_PROBABILITY_THRESHOLD or DEFAULT_WEAK_PROBABILITY_THRESHOLD
)

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

_MODEL_ARTIFACT: Optional[dict] = None
_MODEL_LOAD_ATTEMPTED: bool = False


def _load_model() -> Optional[dict]:
    global _MODEL_ARTIFACT, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_ARTIFACT

    _MODEL_LOAD_ATTEMPTED = True

    if not _MODEL_PATH.exists():
        logger.warning(
            "Quality model not found at %s - ML scoring disabled. "
            "Run: python scripts/train_recommendation_quality_model.py",
            _MODEL_PATH,
        )
        return None

    try:
        import joblib

        artifact = joblib.load(_MODEL_PATH)
        _MODEL_ARTIFACT = artifact
        logger.info(
            "Loaded recommendation quality model (%s, trained on %d rows)",
            artifact.get("model_type", "unknown"),
            artifact.get("train_rows", 0),
        )
        return artifact
    except Exception as exc:
        logger.warning("Failed to load quality model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Feature engineering (mirrors scripts/train_recommendation_quality_model.py)
# ---------------------------------------------------------------------------

_TOP_ADJUSTMENT_TYPES = [
    "single_key_divergent_target_score_capped",
    "core_match_boost",
    "function_sim_boost",
    "cross_category_penalty",
    "low_overlap_penalty",
    "sparse_exact_score_cap",
    "semantic_core_coverage_multiplier",
    "oral_single_core_broad_target_score_cap",
    "lipid_lecithin_single_core_broad_target_score_cap",
    "no_core_weak_shared_with_extra_score_cap",
]


def _parse_json_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        result = json.loads(str(val))
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _threshold_for_artifact(artifact: dict) -> float:
    if _ENV_WEAK_PROBABILITY_THRESHOLD is not None:
        return WEAK_PROBABILITY_THRESHOLD
    try:
        return float(artifact.get("filter_threshold", DEFAULT_WEAK_PROBABILITY_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_WEAK_PROBABILITY_THRESHOLD


def _build_feature_vector(row: dict, artifact: dict) -> np.ndarray:
    """Convert a recommendation score-row dict into a feature vector."""
    feature_names: List[str] = artifact["feature_names"]
    category_maps: dict = artifact.get("category_maps", {})

    feat: Dict[str, float] = {}

    # --- Numeric base features ---
    sim = float(row.get("similarity_score") or 0.0)
    old_sim = float(row.get("old_similarity_score") or sim)
    func_sim = float(row.get("function_similarity_score") or 0.0)
    core_match = float(row.get("core_match_score") or 0.0)
    score_delta = float(row.get("score_delta") or (sim - old_sim))
    rank = int(row.get("rank") or 1)
    judge_conf = float(row.get("judge_confidence") or 0.75)

    feat["similarity_score"] = sim
    feat["old_similarity_score"] = old_sim
    feat["score_delta"] = score_delta
    feat["function_similarity_score"] = func_sim
    feat["core_match_score"] = core_match
    feat["rank"] = float(rank)
    feat["judge_confidence"] = judge_conf

    # --- Pattern numeric features (may be absent) ---
    pat_fields = [
        "shared_count", "shared_core_count", "same_primary_set",
        "primary_primary_overlap_count", "base_primary_count", "target_primary_count",
        "base_primary_in_target_secondary_count", "target_primary_in_base_secondary_count",
        "no_primary_primary_overlap_cross_role", "base_only_semantic_count",
        "target_only_semantic_count",
    ]
    for f in pat_fields:
        feat[f] = float(row.get(f) or 0.0)

    # --- Derived ---
    feat["score_above_08"] = float(sim >= 0.80)
    feat["score_above_07"] = float(sim >= 0.70)
    feat["score_below_05"] = float(sim < 0.50)
    feat["rank_1"] = float(rank == 1)
    feat["rank_gt5"] = float(rank > 5)
    feat["sim_x_funcsim"] = sim * func_sim

    base_pc = max(feat["base_primary_count"], 1.0)
    tgt_pc = max(feat["target_primary_count"], 1.0)
    avg_primary_count = (base_pc + tgt_pc) / 2
    feat["primary_overlap_ratio"] = feat["primary_primary_overlap_count"] / (
        avg_primary_count
    )

    shared_total = max(feat["shared_count"], 1.0)
    feat["core_to_shared_ratio"] = feat["shared_core_count"] / shared_total

    cross_role_overlap = (
        feat["base_primary_in_target_secondary_count"]
        + feat["target_primary_in_base_secondary_count"]
    )
    total_primary_count = max(base_pc + tgt_pc, 1.0)
    feat["primary_cross_role_overlap_count"] = cross_role_overlap
    feat["primary_cross_role_overlap_ratio"] = cross_role_overlap / total_primary_count
    feat["primary_role_mismatch_ratio"] = cross_role_overlap / shared_total
    feat["primary_overlap_gap"] = max(0.0, min(1.0, 1.0 - feat["primary_overlap_ratio"]))
    feat["primary_count_delta_abs"] = abs(feat["base_primary_count"] - feat["target_primary_count"])

    semantic_unshared_total = feat["base_only_semantic_count"] + feat["target_only_semantic_count"]
    semantic_total = max(semantic_unshared_total + feat["shared_count"], 1.0)
    feat["semantic_unshared_total"] = semantic_unshared_total
    feat["semantic_unshared_ratio"] = semantic_unshared_total / semantic_total
    feat["semantic_unshared_gap_abs"] = abs(
        feat["base_only_semantic_count"] - feat["target_only_semantic_count"]
    )

    core_missing = float(feat["shared_core_count"] <= 0 and feat["shared_count"] > 0)
    low_function = float(func_sim < 0.40)
    no_primary_overlap_cross_role = min(max(feat["no_primary_primary_overlap_cross_role"], 0.0), 1.0)
    feat["core_missing_flag"] = core_missing
    feat["high_score_no_core"] = feat["score_above_07"] * core_missing
    feat["high_score_low_function"] = feat["score_above_07"] * low_function
    feat["high_score_cross_role_no_primary"] = feat["score_above_07"] * no_primary_overlap_cross_role
    feat["same_primary_no_core"] = feat["same_primary_set"] * core_missing
    feat["same_primary_low_function"] = feat["same_primary_set"] * low_function
    feat["function_core_gap_abs"] = abs(func_sim - core_match)
    feat["sim_function_gap_positive"] = max(0.0, sim - func_sim)
    feat["sim_core_gap_positive"] = max(0.0, sim - core_match)

    # --- JSON: shared_categories ---
    shared_cats = _parse_json_list(row.get("shared_categories_json"))
    feat["shared_category_count"] = float(len(shared_cats))

    # --- JSON: score_adjustment_types ---
    adj_types = _parse_json_list(row.get("score_adjustment_types_json"))
    top_adjustment_types = artifact.get("top_adjustment_types") or _TOP_ADJUSTMENT_TYPES
    feat["adjustment_count"] = float(len(adj_types))
    for adj in top_adjustment_types:
        feat[f"adj_{adj}"] = float(adj in adj_types)

    # --- Same-category flag ---
    base_cat = str(row.get("base_main_category") or "")
    tgt_cat = str(row.get("target_main_category") or "")
    feat["same_main_category"] = float(base_cat == tgt_cat)

    # --- Categorical encoding ---
    for col in ["base_main_category", "target_main_category"]:
        cmap = category_maps.get(col, {})
        raw_val = str(row.get(col) or "")
        feat[f"{col}_enc"] = float(cmap.get(raw_val, -1))

    # Build ordered array aligned to training feature names
    return np.array([feat.get(name, 0.0) for name in feature_names], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SENTINEL = {
    "ml_quality_score": None,
    "ml_weak_probability": None,
    "ml_label": "unavailable",
    "ml_phase": "disabled",
}


def predict_quality(row: dict) -> dict:
    """Return ML quality prediction for a single recommendation row.

    ``row`` must contain at minimum:
        similarity_score, function_similarity_score, core_match_score,
        rank, base_main_category, target_main_category

    Returns a dict (never raises).  When the model is unavailable all values
    are None and ml_label is 'unavailable'.
    """
    artifact = _load_model()
    if artifact is None:
        return dict(_SENTINEL)

    try:
        X = _build_feature_vector(row, artifact)
        model = artifact["model"]
        proba = float(model.predict_proba(X.reshape(1, -1))[0, 1])
        threshold = _threshold_for_artifact(artifact)

        if proba >= threshold:
            label = "weak"
        elif proba >= threshold * 0.6:
            label = "uncertain"
        else:
            label = "good"

        return {
            "ml_quality_score": round(1.0 - proba, 4),   # higher = better
            "ml_weak_probability": round(proba, 4),
            "ml_label": label,
            "ml_phase": RECOMMENDATION_QUALITY_ML_PHASE,
        }
    except Exception as exc:
        logger.debug("ML quality prediction failed for row: %s", exc)
        return dict(_SENTINEL)


def predict_quality_batch(rows: List[dict]) -> List[dict]:
    """Batch version of predict_quality. Faster for many rows at once."""
    if not rows:
        return []

    artifact = _load_model()
    if artifact is None:
        return [dict(_SENTINEL) for _ in rows]

    try:
        model = artifact["model"]
        X = np.stack([_build_feature_vector(row, artifact) for row in rows])
        probas = model.predict_proba(X)[:, 1]
        threshold = _threshold_for_artifact(artifact)

        results = []
        for proba in probas:
            proba = float(proba)
            if proba >= threshold:
                label = "weak"
            elif proba >= threshold * 0.6:
                label = "uncertain"
            else:
                label = "good"

            results.append({
                "ml_quality_score": round(1.0 - proba, 4),
                "ml_weak_probability": round(proba, 4),
                "ml_label": label,
                "ml_phase": RECOMMENDATION_QUALITY_ML_PHASE,
            })
        return results
    except Exception as exc:
        logger.debug("ML batch quality prediction failed: %s", exc)
        return [dict(_SENTINEL) for _ in rows]


def should_filter_recommendation(ml_result: dict) -> bool:
    """Return True when the recommendation should be suppressed.

    Only acts in 'filter' or 'stacking' phase; in 'shadow' mode always
    returns False so results are never changed.
    """
    phase = str(ml_result.get("ml_phase") or "shadow").lower()
    if phase == "shadow":
        return False

    proba = ml_result.get("ml_weak_probability")
    if proba is None:
        return False

    artifact = _load_model()
    threshold = _threshold_for_artifact(artifact) if artifact is not None else WEAK_PROBABILITY_THRESHOLD
    return float(proba) >= threshold
