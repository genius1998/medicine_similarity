import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.enhance_similarity_with_explanation import (
    calculate_weighted_jaccard_with_idf,
    effective_ingredient_weight,
    ensure_cache_table,
    load_cached_rows,
    refresh_cache_rows,
)


def profile(main_category, primary=None, secondary=None, support=None, ingredient_categories=None):
    primary = primary or []
    secondary = secondary or []
    support = support or []
    role_by_ingredient = {
        **{name: "primary" for name in primary},
        **{name: "secondary" for name in secondary},
        **{name: "support" for name in support},
    }
    return {
        "product_main_category": main_category,
        "primary_ingredients": primary,
        "secondary_ingredients": secondary,
        "support_ingredients": support,
        "role_by_ingredient": role_by_ingredient,
        "category_by_ingredient": ingredient_categories or {},
        "category_scores": {main_category: 5.0},
    }


def test_same_category_without_shared_ingredients_scores_zero():
    base_profile = profile("joint", primary=["core"], ingredient_categories={"core": "joint"})
    target_profile = profile("joint", primary=["other"], ingredient_categories={"other": "joint"})

    score, shared = calculate_weighted_jaccard_with_idf(
        {"core": 1.0},
        {"other": 1.0},
        {"core": 10, "other": 10},
        100,
        base_profile,
        target_profile,
    )

    assert shared == []
    assert score == 0.0


def test_rare_off_category_secondary_does_not_outrank_primary():
    base_profile = profile(
        "joint",
        primary=["core"],
        secondary=["rare"],
        ingredient_categories={"core": "joint", "rare": "other"},
    )
    frequency = {"core": 1000, "rare": 2}
    total_products = 44341

    primary_weight = effective_ingredient_weight(0.95, "core", base_profile, frequency, total_products)
    secondary_weight = effective_ingredient_weight(0.95, "rare", base_profile, frequency, total_products)

    assert primary_weight > secondary_weight
    assert secondary_weight < primary_weight * 0.2


def test_secondary_only_overlap_is_capped_by_missing_primary_overlap():
    base_profile = profile(
        "joint",
        primary=["core"],
        secondary=["rare"],
        ingredient_categories={"core": "joint", "rare": "other"},
    )
    target_profile = profile(
        "other",
        primary=["different"],
        secondary=["rare"],
        ingredient_categories={"different": "other", "rare": "other"},
    )

    score, shared = calculate_weighted_jaccard_with_idf(
        {"core": 0.95, "rare": 0.95},
        {"different": 0.95, "rare": 0.95},
        {"core": 1000, "different": 1000, "rare": 2},
        44341,
        base_profile,
        target_profile,
    )

    assert shared == ["rare"]
    assert 0 < score <= 0.35


def test_cache_uses_current_similarity_algorithm_version():
    conn = sqlite3.connect(":memory:")
    ensure_cache_table(conn)
    refresh_cache_rows(
        conn,
        "base",
        [
            {
                "base_product_id": "base",
                "target_product_id": "target",
                "base_product_name": "base name",
                "target_product_name": "target name",
                "similarity_score": 0.5,
                "function_similarity_score": 0.4,
                "core_match_score": 0.3,
                "shared_ingredients_json": "[]",
                "base_only_ingredients_json": "[]",
                "target_only_ingredients_json": "[]",
                "shared_categories_json": "[]",
                "different_categories_json": "[]",
                "substitutability": "test",
                "reason": "reason",
                "caution": "caution",
                "explanation_json": "{}",
            }
        ],
    )

    rows = load_cached_rows(conn, "base", 10)

    assert len(rows) == 1
    assert rows[0]["target_product_id"] == "target"
