import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from api.recommendation_quality_model import _build_feature_vector  # noqa: E402
from scripts.train_recommendation_quality_model import engineer_features  # noqa: E402


def test_engineer_features_builds_role_semantic_mismatch_signals_with_missing_legacy_columns():
    df = pd.DataFrame(
        [
            {
                "similarity_score": 0.9,
                "function_similarity_score": 0.3,
                "core_match_score": 0.0,
                "rank": 2,
                "judge_confidence": 0.8,
                "shared_count": 2,
                "shared_core_count": 0,
                "same_primary_set": 0,
                "primary_primary_overlap_count": 0,
                "base_primary_count": 1,
                "target_primary_count": 1,
                "base_primary_in_target_secondary_count": 1,
                "target_primary_in_base_secondary_count": 1,
                "no_primary_primary_overlap_cross_role": 1,
                "base_only_semantic_count": 3,
                "target_only_semantic_count": 1,
                "score_adjustment_types_json": '["sparse_exact_score_cap"]',
                "base_main_category": "A",
                "target_main_category": "A",
            }
        ]
    )

    features = engineer_features(df).iloc[0]

    assert features["old_similarity_score"] == 0.0
    assert features["score_delta"] == 0.0
    assert features["primary_cross_role_overlap_count"] == 2
    assert features["primary_cross_role_overlap_ratio"] == 1.0
    assert features["primary_role_mismatch_ratio"] == 1.0
    assert features["primary_overlap_gap"] == 1.0
    assert features["semantic_unshared_total"] == 4
    assert round(float(features["semantic_unshared_ratio"]), 4) == 0.6667
    assert features["semantic_unshared_gap_abs"] == 2
    assert features["core_missing_flag"] == 1
    assert features["high_score_no_core"] == 1
    assert features["high_score_low_function"] == 1
    assert features["high_score_cross_role_no_primary"] == 1
    assert round(float(features["sim_function_gap_positive"]), 4) == 0.6
    assert features["sim_core_gap_positive"] == 0.9
    assert features["adj_sparse_exact_score_cap"] == 1


def test_api_feature_vector_uses_artifact_adjustment_types_and_rich_features():
    artifact = {
        "feature_names": [
            "primary_cross_role_overlap_ratio",
            "primary_role_mismatch_ratio",
            "high_score_no_core",
            "high_score_low_function",
            "adj_sparse_exact_score_cap",
            "base_main_category_enc",
            "target_main_category_enc",
        ],
        "category_maps": {
            "base_main_category": {"A": 3},
            "target_main_category": {"B": 7},
        },
        "top_adjustment_types": ["sparse_exact_score_cap"],
    }
    row = {
        "similarity_score": 0.91,
        "function_similarity_score": 0.2,
        "core_match_score": 0.0,
        "rank": 1,
        "shared_count": 2,
        "shared_core_count": 0,
        "base_primary_count": 1,
        "target_primary_count": 1,
        "base_primary_in_target_secondary_count": 1,
        "target_primary_in_base_secondary_count": 1,
        "no_primary_primary_overlap_cross_role": 1,
        "score_adjustment_types_json": '["sparse_exact_score_cap"]',
        "base_main_category": "A",
        "target_main_category": "B",
    }

    vector = _build_feature_vector(row, artifact)

    assert vector.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 7.0]
