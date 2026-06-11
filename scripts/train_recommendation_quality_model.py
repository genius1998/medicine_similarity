"""
Train XGBoost / LightGBM recommendation quality prediction model.

Binary classification task:
  - Positive (1): judge_judgment in {'weak', 'bad'}   -> low quality recommendation
  - Negative (0): 'reasonable' | 'acceptable_adjacent' -> acceptable recommendation

Output: output/ml_models/recommendation_quality_model.pkl

Usage:
    python scripts/train_recommendation_quality_model.py
    python scripts/train_recommendation_quality_model.py --model lightgbm
    python scripts/train_recommendation_quality_model.py --model xgboost --eval
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = (
    REPO_ROOT
    / "output"
    / "recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060719_oralxylitol_p260_livermilk_p300"
)
MAIN_CSV = JUDGE_DIR / "openai_chunk_judge_results.csv"
PATTERN_CSV = JUDGE_DIR / "patterns" / "judge_pattern_features.csv"
MODEL_DIR = REPO_ROOT / "output" / "ml_models"
MODEL_PATH = MODEL_DIR / "recommendation_quality_model.pkl"
FEATURE_ENGINEERING_VERSION = "role_semantic_v2"
DEFAULT_FILTER_THRESHOLD = 0.834
TRAIN_DEDUP_KEYS = ["base_report_no", "target_report_no", "rank"]

# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "similarity_score",
    "old_similarity_score",
    "score_delta",
    "function_similarity_score",
    "core_match_score",
    "rank",
    "judge_confidence",
]

PATTERN_NUMERIC_FEATURES = [
    "shared_count",
    "shared_core_count",
    "same_primary_set",
    "primary_primary_overlap_count",
    "base_primary_count",
    "target_primary_count",
    "base_primary_in_target_secondary_count",
    "target_primary_in_base_secondary_count",
    "no_primary_primary_overlap_cross_role",
    "base_only_semantic_count",
    "target_only_semantic_count",
]

CATEGORICAL_FEATURES = ["base_main_category", "target_main_category"]

# Top adjustment types observed in the data; will be one-hot encoded.
TOP_ADJUSTMENT_TYPES = [
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

RICH_DERIVED_FEATURES = [
    "primary_cross_role_overlap_count",
    "primary_cross_role_overlap_ratio",
    "primary_role_mismatch_ratio",
    "primary_overlap_gap",
    "primary_count_delta_abs",
    "semantic_unshared_total",
    "semantic_unshared_ratio",
    "semantic_unshared_gap_abs",
    "core_missing_flag",
    "high_score_no_core",
    "high_score_low_function",
    "high_score_cross_role_no_primary",
    "same_primary_no_core",
    "same_primary_low_function",
    "function_core_gap_abs",
    "sim_function_gap_positive",
    "sim_core_gap_positive",
]


def _parse_json_list(val) -> list:
    if pd.isna(val):
        return []
    try:
        result = json.loads(str(val))
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix (all numeric) from raw dataframe."""
    feat = pd.DataFrame(index=df.index)

    # --- Numeric columns ---
    for col in NUMERIC_FEATURES:
        feat[col] = _numeric_series(df, col)

    # --- Pattern-CSV numeric columns (may be absent if not merged yet) ---
    for col in PATTERN_NUMERIC_FEATURES:
        feat[col] = _numeric_series(df, col)

    # --- Derived numeric features ---
    sim = feat["similarity_score"]
    feat["score_above_08"] = (sim >= 0.80).astype(float)
    feat["score_above_07"] = (sim >= 0.70).astype(float)
    feat["score_below_05"] = (sim < 0.50).astype(float)
    feat["rank_1"] = (feat["rank"] == 1).astype(float)
    feat["rank_gt5"] = (feat["rank"] > 5).astype(float)

    # Cross-feature: similarity * function_similarity
    feat["sim_x_funcsim"] = feat["similarity_score"] * feat["function_similarity_score"]

    # Core overlap ratio
    base_primary_count = feat["base_primary_count"]
    target_primary_count = feat["target_primary_count"]
    base_pc = base_primary_count.clip(lower=1)
    tgt_pc = target_primary_count.clip(lower=1)
    avg_primary_count = (base_pc + tgt_pc) / 2
    feat["primary_overlap_ratio"] = feat["primary_primary_overlap_count"] / (
        avg_primary_count
    )

    # Shared-core ratio vs shared total
    shared_total = feat["shared_count"].clip(lower=1)
    feat["core_to_shared_ratio"] = feat["shared_core_count"] / shared_total

    # Role/semantic mismatch signals for high-score weak recommendations.
    cross_role_overlap = (
        feat["base_primary_in_target_secondary_count"]
        + feat["target_primary_in_base_secondary_count"]
    )
    total_primary_count = (base_pc + tgt_pc).clip(lower=1)
    feat["primary_cross_role_overlap_count"] = cross_role_overlap
    feat["primary_cross_role_overlap_ratio"] = cross_role_overlap / total_primary_count
    feat["primary_role_mismatch_ratio"] = cross_role_overlap / shared_total
    feat["primary_overlap_gap"] = (1.0 - feat["primary_overlap_ratio"]).clip(lower=0.0, upper=1.0)
    feat["primary_count_delta_abs"] = (base_primary_count - target_primary_count).abs()

    semantic_unshared_total = feat["base_only_semantic_count"] + feat["target_only_semantic_count"]
    semantic_total = (semantic_unshared_total + feat["shared_count"]).clip(lower=1)
    feat["semantic_unshared_total"] = semantic_unshared_total
    feat["semantic_unshared_ratio"] = semantic_unshared_total / semantic_total
    feat["semantic_unshared_gap_abs"] = (
        feat["base_only_semantic_count"] - feat["target_only_semantic_count"]
    ).abs()

    core_missing = ((feat["shared_core_count"] <= 0) & (feat["shared_count"] > 0)).astype(float)
    low_function = (feat["function_similarity_score"] < 0.40).astype(float)
    no_primary_overlap_cross_role = feat["no_primary_primary_overlap_cross_role"].clip(lower=0, upper=1)
    feat["core_missing_flag"] = core_missing
    feat["high_score_no_core"] = feat["score_above_07"] * core_missing
    feat["high_score_low_function"] = feat["score_above_07"] * low_function
    feat["high_score_cross_role_no_primary"] = feat["score_above_07"] * no_primary_overlap_cross_role
    feat["same_primary_no_core"] = feat["same_primary_set"] * core_missing
    feat["same_primary_low_function"] = feat["same_primary_set"] * low_function
    feat["function_core_gap_abs"] = (
        feat["function_similarity_score"] - feat["core_match_score"]
    ).abs()
    feat["sim_function_gap_positive"] = (
        feat["similarity_score"] - feat["function_similarity_score"]
    ).clip(lower=0.0)
    feat["sim_core_gap_positive"] = (
        feat["similarity_score"] - feat["core_match_score"]
    ).clip(lower=0.0)

    # --- JSON column: shared_categories_json -> count ---
    if "shared_categories_json" in df.columns:
        feat["shared_category_count"] = df["shared_categories_json"].apply(
            lambda v: len(_parse_json_list(v))
        )
    else:
        feat["shared_category_count"] = 0

    # --- JSON column: score_adjustment_types_json -> count + one-hot flags ---
    if "score_adjustment_types_json" in df.columns:
        adj_lists = df["score_adjustment_types_json"].apply(_parse_json_list)
    else:
        adj_lists = pd.Series([[] for _ in range(len(df))], index=df.index)

    feat["adjustment_count"] = adj_lists.apply(len)
    for adj_type in TOP_ADJUSTMENT_TYPES:
        feat[f"adj_{adj_type}"] = adj_lists.apply(lambda lst: int(adj_type in lst))

    # --- Same-category flag ---
    base_cat = df.get("base_main_category", pd.Series([""] * len(df), index=df.index)).fillna("")
    tgt_cat = df.get("target_main_category", pd.Series([""] * len(df), index=df.index)).fillna("")
    feat["same_main_category"] = (base_cat == tgt_cat).astype(float)

    # --- Categorical: label-encode category columns ---
    for col in CATEGORICAL_FEATURES:
        raw = df.get(col, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
        # Simple ordinal encoding (stable for inference via shared encoder)
        cat = raw.astype("category")
        feat[f"{col}_enc"] = cat.cat.codes.astype(float)

    return feat


# ----------------------------------------------------------------------------
# Loading & merging
# ----------------------------------------------------------------------------


def _judge_csv_for_dir(judge_dir: Path) -> Path:
    for name in ("openai_chunk_judge_results.csv", "gemini_judge_results.csv"):
        path = judge_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"judge results CSV not found in {judge_dir}")


def _load_judge_dir(judge_dir: Path) -> pd.DataFrame:
    main_csv = _judge_csv_for_dir(judge_dir)
    pattern_csv = judge_dir / "patterns" / "judge_pattern_features.csv"
    logger.info("Loading main judge CSV: %s", main_csv)
    main = pd.read_csv(main_csv, low_memory=False)
    logger.info("  %d rows loaded from main CSV", len(main))

    merge_keys = list(TRAIN_DEDUP_KEYS)

    if pattern_csv.exists():
        logger.info("Loading pattern features CSV: %s", pattern_csv)
        pat = pd.read_csv(pattern_csv, low_memory=False)
        logger.info("  %d rows in pattern CSV", len(pat))

        pat_cols = list(merge_keys) + [
            c
            for c in PATTERN_NUMERIC_FEATURES + ["score_adjustment_types_json"]
            if c in pat.columns
        ]
        pat_sub = pat[pat_cols].drop_duplicates(subset=merge_keys)

        for key in merge_keys:
            main[key] = main[key].astype(str)
            pat_sub = pat_sub.copy()
            pat_sub[key] = pat_sub[key].astype(str)

        df = main.merge(pat_sub, on=merge_keys, how="left", suffixes=("", "_pattern"))
        if "score_adjustment_types_json_pattern" in df.columns:
            if "score_adjustment_types_json" in df.columns:
                df["score_adjustment_types_json"] = df["score_adjustment_types_json"].fillna(
                    df["score_adjustment_types_json_pattern"]
                )
            else:
                df["score_adjustment_types_json"] = df["score_adjustment_types_json_pattern"]
        logger.info("  After merge: %d rows", len(df))
    else:
        logger.warning("Pattern CSV not found; proceeding without pattern features.")
        df = main

    df["training_source_dir"] = str(judge_dir)
    return df


def load_data(extra_judge_dirs: list[str | Path] | None = None) -> pd.DataFrame:
    judge_dirs = [JUDGE_DIR] + [Path(path) for path in (extra_judge_dirs or [])]
    frames = [_load_judge_dir(Path(judge_dir)) for judge_dir in judge_dirs]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if len(frames) > 1:
        before = len(df)
        for key in TRAIN_DEDUP_KEYS:
            if key in df.columns:
                df[key] = df[key].astype(str)
        df = df.drop_duplicates(subset=TRAIN_DEDUP_KEYS, keep="last").reset_index(drop=True)
        logger.info("Training deduplication: %d -> %d rows", before, len(df))
    return df


def build_label(df: pd.DataFrame) -> np.ndarray:
    """1 = weak/bad (low quality), 0 = reasonable/acceptable_adjacent."""
    return (df["judge_judgment"].isin({"weak", "bad"})).astype(int).values


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def train(
    model_type: str = "xgboost",
    run_cv: bool = False,
    filter_threshold: float = DEFAULT_FILTER_THRESHOLD,
    extra_judge_dirs: list[str] | None = None,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(extra_judge_dirs)
    X = engineer_features(df)
    y = build_label(df)

    feature_names = list(X.columns)
    logger.info("Feature count: %d", len(feature_names))
    logger.info("Label distribution - positive (weak/bad): %d / %d (%.2f%%)",
                y.sum(), len(y), 100 * y.mean())

    scale_pos_weight = float((y == 0).sum()) / float(max((y == 1).sum(), 1))
    logger.info("scale_pos_weight: %.2f", scale_pos_weight)

    if model_type == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model_name = "LightGBM"
    else:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model_name = "XGBoost"

    if run_cv:
        logger.info("Running 5-fold CV (ROC-AUC) with %s", model_name)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X.values, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        logger.info("CV ROC-AUC: %.4f +/- %.4f", cv_scores.mean(), cv_scores.std())

    logger.info("Training final %s model on full data", model_name)
    model.fit(X.values, y)

    # Evaluate on training set (sanity check)
    proba = model.predict_proba(X.values)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    logger.info("Train ROC-AUC: %.4f  |  Average Precision: %.4f", auc, ap)
    logger.info("\n%s", classification_report(y, y_pred, target_names=["good", "weak/bad"]))

    # Feature importance (top 15)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        logger.info("Top-15 features by importance:")
        for rank_i, idx in enumerate(top_idx, 1):
            logger.info("  %2d. %-45s %.4f", rank_i, feature_names[idx], importances[idx])

    # Persist model + metadata
    category_maps = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            cat = df[col].fillna("").astype("category")
            category_maps[col] = {v: i for i, v in enumerate(cat.cat.categories)}

    artifact = {
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names,
        "category_maps": category_maps,
        "top_adjustment_types": TOP_ADJUSTMENT_TYPES,
        "rich_derived_features": RICH_DERIVED_FEATURES,
        "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
        "filter_threshold": float(filter_threshold),
        "base_judge_dir": str(JUDGE_DIR),
        "extra_judge_dirs": [str(path) for path in (extra_judge_dirs or [])],
        "training_dedup_keys": TRAIN_DEDUP_KEYS,
        "label_description": "1=weak_or_bad  0=reasonable_or_acceptable_adjacent",
        "train_rows": len(df),
        "positive_rate": float(y.mean()),
    }
    joblib.dump(artifact, MODEL_PATH)
    logger.info("Model saved: %s", MODEL_PATH)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recommendation quality model")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm"],
        default="xgboost",
        help="Gradient boosting backend (default: xgboost)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run 5-fold cross-validation before final training",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=DEFAULT_FILTER_THRESHOLD,
        help="Saved weak-probability threshold metadata; does not change training labels.",
    )
    parser.add_argument(
        "--extra-judge-dir",
        action="append",
        default=[],
        help="Additional finalized validation directory to append to the training data. Can be repeated.",
    )
    args = parser.parse_args()
    train(
        model_type=args.model,
        run_cv=args.eval,
        filter_threshold=args.filter_threshold,
        extra_judge_dirs=args.extra_judge_dir,
    )


if __name__ == "__main__":
    main()
