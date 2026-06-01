import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.enhance_similarity_with_explanation import (
    SIMILARITY_ALGORITHM_V1,
    SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_VERSION,
    calculate_semantic_weighted_jaccard_v2,
    calculate_weighted_jaccard_with_idf,
    build_semantic_weight_vector,
    effective_ingredient_weight,
    ensure_cache_table,
    load_cached_rows,
    normalize_similarity_algorithm,
    refresh_cache_rows,
)
from api.upload_recommendation_service import UploadRecommendationService


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
        "ingredient_scores": [
            {
                "ingredient": name,
                "weight": 0.95,
                "role": role,
                "category_main": (ingredient_categories or {}).get(name, main_category),
                "category_sub": "",
            }
            for name, role in role_by_ingredient.items()
        ],
    }


def test_similarity_algorithm_default_is_semantic_v2_but_v1_is_still_selectable():
    assert SIMILARITY_ALGORITHM_VERSION == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v1") == SIMILARITY_ALGORITHM_V1


def test_upload_recommendation_temp_vector_uses_semantic_v2():
    base_profile = {
        **profile("관절/연골", primary=["MSM"], ingredient_categories={"MSM": "관절/연골"}),
        "product_id": "uploaded::base",
        "report_no": "",
        "product_name": "업로드 MSM",
    }
    target_profile = {
        **profile(
            "관절/연골",
            primary=["엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)"],
            ingredient_categories={"엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)": "관절/연골"},
        ),
        "product_id": "target::1",
        "report_no": "T1",
        "product_name": "타겟 MSM",
        "is_candidate_enabled": True,
    }
    ingredient_profiles = {
        "MSM": {
            "functional_ingredient_name": "MSM",
            "ingredient_main_category": "관절/연골",
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
        "엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)": {
            "functional_ingredient_name": "엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)",
            "ingredient_main_category": "관절/연골",
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    class DummyRecommendationService:
        product_vectors = {"target::1": {"엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)": 1.0}}
        profiles = {"target::1": target_profile}
        ingredient_postings = {
            "MSM": [],
            "엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)": ["target::1"],
        }
        ingredient_frequency = {"MSM": 1, "엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)": 1}
        ingredient_category_profiles = ingredient_profiles

        def ensure_loaded(self):
            return None

    upload_service = UploadRecommendationService(DummyRecommendationService())
    rows = upload_service.calculate_similar_products_for_temp_vector({"MSM": 1.0}, base_profile, top_k=1, candidate_limit=10)

    assert rows
    assert rows[0]["explanation"]["similarity_algorithm"] == SIMILARITY_ALGORITHM_V2
    assert "semantic_weighted_jaccard_v2" in rows[0]["explanation"]
    assert rows[0]["similarity_score"] > 0.0


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


def test_semantic_v2_excludes_formulation_aids():
    base_profile = profile("관절/연골", primary=["보스웰리아"], support=["히드록시프로필메틸셀룰로오스", "결정셀룰로오스"])
    target_profile = profile("관절/연골", primary=["보스웰리아"], support=["스테아린산마그네슘"])
    ingredient_profiles = {
        "보스웰리아": {
            "ingredient_main_category": "관절/연골",
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
        "히드록시프로필메틸셀룰로오스": {
            "ingredient_main_category": "기타",
            "ingredient_sub_function_categories": [],
            "ingredient_type": "excipient",
            "vector_include": False,
            "is_excipient": True,
        },
        "결정셀룰로오스": {
            "ingredient_main_category": "기타",
            "ingredient_sub_function_categories": [],
            "ingredient_type": "excipient",
            "vector_include": False,
            "is_excipient": True,
        },
        "스테아린산마그네슘": {
            "ingredient_main_category": "기타",
            "ingredient_sub_function_categories": [],
            "ingredient_type": "formulation_aid",
            "vector_include": False,
            "is_excipient": True,
        },
    }

    base_vector = build_semantic_weight_vector(base_profile, ingredient_profiles)
    target_vector = build_semantic_weight_vector(target_profile, ingredient_profiles)
    score, shared, detail = calculate_semantic_weighted_jaccard_v2(base_profile, target_profile, ingredient_profiles)

    assert "히드록시프로필메틸셀룰로오스" not in base_vector
    assert "결정셀룰로오스" not in base_vector
    assert "스테아린산마그네슘" not in target_vector
    assert score == 1.0
    assert shared == ["보스웰리아"]
    assert detail["shared_semantic_keys"][0]["base_weight"] == 1.0


def test_semantic_v2_uses_product_sub_category_overlap_without_legacy_sub_category():
    base_profile = profile("혈행", primary=["EPA 및 DHA 함유 유지"])
    target_profile = profile("눈 건강", primary=["EPA 및 DHA 함유 유지"])
    target_profile["llm_sub_function_categories"] = ["혈행"]
    ingredient_profiles = {
        "EPA 및 DHA 함유 유지": {
            "ingredient_main_category": "혈중지질",
            "ingredient_sub_function_categories": ["혈행", "눈 건강", "기억력"],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, shared, _detail = calculate_semantic_weighted_jaccard_v2(base_profile, target_profile, ingredient_profiles)

    assert shared == ["EPA 및 DHA 함유 유지"]
    assert 0.99 <= score <= 1.0


def test_semantic_v2_caps_generic_nutrient_weight():
    nutrient_profile = profile("영양보충", primary=["비타민 C"])
    ingredient_profiles = {
        "비타민 C": {
            "ingredient_main_category": "영양보충",
            "ingredient_sub_function_categories": ["항산화"],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    vector = build_semantic_weight_vector(nutrient_profile, ingredient_profiles)

    assert vector["비타민 C"] == 0.4


def test_semantic_v2_caps_generic_nutrient_total_for_specific_products():
    body_fat = "\uccb4\uc9c0\ubc29"
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    base = profile(
        body_fat,
        primary=["\uac00\ub974\uc2dc\ub2c8\uc544"],
        secondary=["\ube44\ud0c0\ubbfc C", "\uc544\uc5f0"],
        ingredient_categories={
            "\uac00\ub974\uc2dc\ub2c8\uc544": body_fat,
            "\ube44\ud0c0\ubbfc C": nutrition,
            "\uc544\uc5f0": nutrition,
        },
    )
    ingredient_profiles = {
        "\uac00\ub974\uc2dc\ub2c8\uc544": {
            "ingredient_main_category": body_fat,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
        "\ube44\ud0c0\ubbfc C": {
            "ingredient_main_category": nutrition,
            "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
        "\uc544\uc5f0": {
            "ingredient_main_category": nutrition,
            "ingredient_sub_function_categories": ["\uba74\uc5ed"],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    vector = build_semantic_weight_vector(base, ingredient_profiles)
    nutrient_total = vector["\ube44\ud0c0\ubbfc C"] + vector["\uc544\uc5f0"]

    assert vector["\uac00\ub974\uc2dc\ub2c8\uc544"] == 1.0
    assert nutrient_total <= 0.250001


def test_semantic_v2_family_signal_is_weak_not_exact():
    joint = "\uad00\uc808/\uc5f0\uace8"
    mucoprotein = "\ubba4\ucf54\ub2e4\ub2f9.\ub2e8\ubc31"
    base = profile(joint, support=[mucoprotein], ingredient_categories={mucoprotein: joint})
    base["ingredient_scores"][0]["relation_type"] = "family_signal"
    base["ingredient_scores"][0]["v2_decision"] = "family_signal"
    target = profile(joint, primary=[mucoprotein], ingredient_categories={mucoprotein: joint})
    ingredient_profiles = {
        mucoprotein: {
            "ingredient_main_category": joint,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        }
    }

    base_vector = build_semantic_weight_vector(base, ingredient_profiles)
    score, shared, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert base_vector["family::shark_cartilage"] == 0.35
    assert shared == [mucoprotein]
    assert score < 0.4
    assert detail["shared_semantic_keys"][0]["base_relation_type"] == "family_signal"


def test_semantic_v2_primary_core_overlap_beats_support_overlap():
    eye = "\ub208 \uac74\uac15"
    lipid = "\ud608\uc911\uc9c0\uc9c8"
    lutein = "\ub8e8\ud14c\uc778"
    astaxanthin = "\ud5e4\ub9c8\ud1a0\ucf54\ucfe0\uc2a4 \ucd94\ucd9c\ubb3c"
    omega = "EPA \ubc0f DHA \ud568\uc720 \uc720\uc9c0"
    minor = "\ube44\uc218\ub9ac\ucd94\ucd9c\ubd84\ub9d0"
    base = profile(eye, primary=[lutein, astaxanthin], secondary=[omega, minor])
    target_exact = profile(eye, primary=[lutein, astaxanthin])
    target_support_heavy = profile(eye, primary=[astaxanthin], secondary=[omega, minor])
    ingredient_profiles = {
        lutein: {"ingredient_main_category": eye, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        astaxanthin: {"ingredient_main_category": eye, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        omega: {"ingredient_main_category": lipid, "ingredient_sub_function_categories": [eye], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        minor: {"ingredient_main_category": "\ub0a8\uc131 \uac74\uac15", "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
    }

    exact_score, _, _ = calculate_semantic_weighted_jaccard_v2(base, target_exact, ingredient_profiles)
    support_score, _, support_detail = calculate_semantic_weighted_jaccard_v2(base, target_support_heavy, ingredient_profiles)

    assert exact_score > support_score
    assert support_detail["core_coverage"]["reason"] == "partial_primary_core_overlap"


def test_semantic_v2_caps_sparse_exact_one_point_zero():
    blood_flow = "\ud608\ud589"
    omega = "EPA \ubc0f DHA \ud568\uc720 \uc720\uc9c0"
    base = profile(blood_flow, primary=[omega])
    base["product_name"] = "base omega"
    target = profile(blood_flow, primary=[omega])
    target["product_name"] = "other omega"
    ingredient_profiles = {
        omega: {
            "ingredient_main_category": "\ud608\uc911\uc9c0\uc9c8",
            "ingredient_sub_function_categories": [blood_flow],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.97
    assert detail["raw_jaccard_score"] == 1.0
    assert detail["score_adjustments"][0]["type"] == "sparse_exact_score_cap"
