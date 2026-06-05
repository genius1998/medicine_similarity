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
    build_candidate_pool_v2,
    calculate_semantic_weighted_jaccard_v2,
    calculate_weighted_jaccard_with_idf,
    build_semantic_weight_vector,
    effective_ingredient_weight,
    ensure_cache_table,
    load_cached_rows,
    normalize_similarity_algorithm,
    recommendation_quality_metadata,
    refresh_cache_rows,
)
from api.recommendation_service import RecommendationService
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
    assert normalize_similarity_algorithm("semantic_weighted_jaccard_v2") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.1") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.2") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.3") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.4") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.5") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.6") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.7") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.8") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v2.9") == SIMILARITY_ALGORITHM_V2
    assert normalize_similarity_algorithm("v1") == SIMILARITY_ALGORITHM_V1


def test_semantic_v2_candidate_pool_is_same_main_only():
    liver = "\uac04 \uac74\uac15"
    immune = "\uba74\uc5ed"
    milk_thistle = "\ubc00\ud06c\uc528\uc2ac\ucd94\ucd9c\ubb3c"
    base = profile(liver, primary=[milk_thistle], ingredient_categories={milk_thistle: liver})
    same_main = profile(liver, primary=[milk_thistle], ingredient_categories={milk_thistle: liver})
    cross_main = profile(immune, primary=[milk_thistle], ingredient_categories={milk_thistle: liver})
    profiles = {"base": base, "same": same_main, "cross": cross_main}
    product_vectors = {product_id: {milk_thistle: 1.0} for product_id in profiles}
    ingredient_profiles = {
        milk_thistle: {
            "ingredient_main_category": liver,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        }
    }

    rows = build_candidate_pool_v2(
        "base",
        base,
        product_vectors,
        profiles,
        {milk_thistle: ["same", "cross"]},
        {milk_thistle: 3},
        candidate_limit=10,
        max_df_for_seed=3000,
        ingredient_category_profiles=ingredient_profiles,
    )

    assert [row["target_product_id"] for row in rows] == ["same"]


def test_recommendation_quality_metadata_marks_low_score_ineligible():
    high = recommendation_quality_metadata(0.65, {})
    low = recommendation_quality_metadata(0.64, {"score_adjustments": [{"type": "generic_nutrient_only_nutrition_main_score_cap"}]})
    supported_borderline = recommendation_quality_metadata(0.5, {}, core_match_score=1.0, function_similarity_score=0.4)
    low_core_borderline = recommendation_quality_metadata(0.49, {}, core_match_score=0.5, function_similarity_score=0.7)
    no_core = recommendation_quality_metadata(0.5, {}, core_match_score=0.0, function_similarity_score=0.8)

    assert high["recommendation_display_eligible"] is True
    assert high["recommendation_quality"] == "strong_match"
    assert low["recommendation_display_eligible"] is False
    assert low["recommendation_quality"] == "low_confidence_match"
    assert low["recommendation_review_reason"] == "generic_nutrient_only_or_dominant"
    assert supported_borderline["recommendation_display_eligible"] is True
    assert supported_borderline["recommendation_quality"] == "low_confidence_match"
    assert low_core_borderline["recommendation_display_eligible"] is False
    assert low_core_borderline["recommendation_review_reason"] == "low_score_low_core_overlap"
    assert no_core["recommendation_display_eligible"] is False
    assert no_core["recommendation_review_reason"] == "no_core_overlap"


def test_recommendation_quality_metadata_marks_weak_semantic_signal_ineligible():
    weak_semantic_only = recommendation_quality_metadata(
        0.55,
        {
            "shared_semantic_keys": [
                {
                    "semantic_key": "vegetable-blend",
                    "base_weight": 0.55,
                    "target_weight": 0.55,
                }
            ]
        },
        core_match_score=1.0,
        function_similarity_score=0.8,
    )

    assert weak_semantic_only["recommendation_display_eligible"] is False
    assert weak_semantic_only["recommendation_review_reason"] == "weak_signal_only"


def test_recommendation_quality_metadata_marks_sparse_review_signal_ineligible():
    sparse_review_signal = recommendation_quality_metadata(
        0.55,
        {
            "base_semantic_ingredient_count": 3,
            "target_semantic_ingredient_count": 1,
            "shared_semantic_keys": [
                {
                    "semantic_key": "vegetable-blend",
                    "base_weight": 0.95,
                    "target_weight": 0.95,
                    "target_ingredient_source": "ingredient_category_profile_rule_v1",
                    "target_legacy_category_sub": "\uC2DD\uBB3C\uCD94\uCD9C\uBB3C/\uAC80\uD1A0\uD544\uC694",
                }
            ],
        },
        core_match_score=1.0,
        function_similarity_score=0.8,
    )

    assert sparse_review_signal["recommendation_display_eligible"] is False
    assert sparse_review_signal["recommendation_review_reason"] == "sparse_review_signal_only"


def test_recommendation_quality_metadata_marks_single_core_low_shared_coverage_ineligible():
    single_core_low_coverage = recommendation_quality_metadata(
        0.526,
        {
            "base_semantic_ingredient_count": 4,
            "target_semantic_ingredient_count": 1,
            "core_coverage": {"shared_core_semantic_keys": ["\ud504\ub77d\ud1a0\uc62c\ub9ac\uace0\ub2f9"]},
            "shared_semantic_keys": [
                {
                    "semantic_key": "\ud504\ub77d\ud1a0\uc62c\ub9ac\uace0\ub2f9",
                    "base_weight": 1.0,
                    "target_weight": 1.0,
                }
            ],
        },
        core_match_score=1.0,
        function_similarity_score=0.4,
    )
    supported_borderline = recommendation_quality_metadata(
        0.526,
        {
            "base_semantic_ingredient_count": 2,
            "target_semantic_ingredient_count": 1,
            "core_coverage": {"shared_core_semantic_keys": ["\ud504\ub77d\ud1a0\uc62c\ub9ac\uace0\ub2f9"]},
            "shared_semantic_keys": [
                {
                    "semantic_key": "\ud504\ub77d\ud1a0\uc62c\ub9ac\uace0\ub2f9",
                    "base_weight": 1.0,
                    "target_weight": 1.0,
                }
            ],
        },
        core_match_score=1.0,
        function_similarity_score=0.4,
    )

    assert single_core_low_coverage["recommendation_display_eligible"] is False
    assert single_core_low_coverage["recommendation_review_reason"] == "single_core_low_shared_coverage"
    assert supported_borderline["recommendation_display_eligible"] is True


def test_review_fallback_ingredient_does_not_become_core_signal():
    vector = build_semantic_weight_vector(
        profile("\uC601\uC591\uBCF4\uCDA9", primary=["apple_concentrate"]),
        {
            "apple_concentrate": {
                "ingredient_main_category": "\uC601\uC591\uBCF4\uCDA9",
                "ingredient_sub_function_categories": ["\uD56D\uC0B0\uD654"],
                "ingredient_type": "functional",
                "vector_include": True,
                "is_excipient": False,
                "confidence": 0.65,
                "reason": "claim_text_category_fallback",
                "source": "ingredient_category_profile_rule_v1",
            }
        },
    )

    assert vector["apple_concentrate"] == 0.3


def test_recommendation_service_filters_display_ineligible_rows():
    service = RecommendationService.__new__(RecommendationService)
    rows = [
        {"rank": 1, "target_report_no": "LOW", "recommendation_display_eligible": False},
        {"rank": 2, "target_report_no": "HIGH", "recommendation_display_eligible": True},
    ]

    filtered = service._display_eligible_recommendations(rows, top_k=10)

    assert [item["target_report_no"] for item in filtered] == ["HIGH"]
    assert filtered[0]["rank"] == 1


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


def test_semantic_v2_core_overlap_uses_semantic_weight_not_legacy_role():
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

    assert exact_score == support_score
    assert omega in support_detail["core_coverage"]["base_core_semantic_keys"]
    assert omega in support_detail["core_coverage"]["target_core_semantic_keys"]
    assert support_detail["core_coverage"]["reason"] == "partial_semantic_core_overlap"


def test_semantic_v2_no_core_overlap_uses_conservative_multiplier():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    eye = "\ub208 \uac74\uac15"
    beta_carotene = "\ubca0\ud0c0\uce74\ub85c\ud2f4"
    vitamin_c = "\ube44\ud0c0\ubbfc C"
    base = profile(nutrition, primary=[beta_carotene, vitamin_c], ingredient_categories={beta_carotene: eye, vitamin_c: nutrition})
    base["llm_sub_function_categories"] = [eye]
    target = profile(nutrition, primary=[beta_carotene, vitamin_c], ingredient_categories={beta_carotene: eye, vitamin_c: nutrition})
    ingredient_profiles = {
        beta_carotene: {"ingredient_main_category": eye, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        vitamin_c: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
    }

    _score, _shared, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert detail["core_coverage"]["reason"] == "no_semantic_core_overlap"
    assert detail["core_coverage"]["multiplier"] == 0.68


def test_semantic_v2_match_confidence_is_not_legacy_role_weight():
    male = "\ub0a8\uc131 \uac74\uac15"
    arginine = "L-\uc544\ub974\uae30\ub2cc"
    base = profile(male, support=[arginine], ingredient_categories={arginine: male})
    base["ingredient_scores"][0]["weight"] = 0.25
    base["ingredient_scores"][0]["match_confidence"] = 0.99
    ingredient_profiles = {
        arginine: {
            "ingredient_main_category": male,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    vector = build_semantic_weight_vector(base, ingredient_profiles)

    assert vector[arginine] == 1.0


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


def test_semantic_v2_caps_generic_nutrient_only_cross_main_match():
    potassium = "\uce7c\ub968"
    base = profile("\uae30\ud0c0", primary=[potassium])
    base["product_name"] = "potassium base"
    same_main = profile("\uae30\ud0c0", primary=[potassium])
    same_main["product_name"] = "potassium same main"
    immune = profile("\uba74\uc5ed", primary=[potassium])
    immune["product_name"] = "immune potassium"
    gut_a = profile("\uc7a5 \uac74\uac15", primary=[potassium])
    gut_a["product_name"] = "gut potassium a"
    gut_b = profile("\uc7a5 \uac74\uac15", primary=[potassium])
    gut_b["product_name"] = "gut potassium b"
    ingredient_profiles = {
        potassium: {
            "ingredient_main_category": "\uc601\uc591\ubcf4\ucda9",
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    same_score, _, same_detail = calculate_semantic_weighted_jaccard_v2(base, same_main, ingredient_profiles)
    cross_score, _, cross_detail = calculate_semantic_weighted_jaccard_v2(base, immune, ingredient_profiles)
    same_non_nutrient_score, _, same_non_nutrient_detail = calculate_semantic_weighted_jaccard_v2(gut_a, gut_b, ingredient_profiles)

    assert same_score == 0.97
    assert same_detail["score_adjustments"][0]["type"] == "sparse_exact_score_cap"
    assert cross_score == 0.45
    assert cross_detail["score_adjustments"][0]["type"] == "generic_nutrient_only_cross_main_score_cap"
    assert same_non_nutrient_score == 0.55
    assert same_non_nutrient_detail["score_adjustments"][0]["type"] == "generic_nutrient_only_same_main_score_cap"


def test_semantic_v2_caps_no_core_weak_shared_with_extra_for_same_main():
    other = "\uae30\ud0c0"
    relaxation = "\uc218\uba74/\uae34\uc7a5\uc644\ud654"
    l_theanine = "L-\ud14c\uc544\ub2cc"
    seaweed = "\uac10\ud0dc\ucd94\ucd9c\ubb3c"
    calcium = "\uce7c\uc298"
    base = profile(other, primary=[l_theanine], secondary=[seaweed])
    base["product_name"] = "l-theanine base"
    target_with_extra = profile(other, primary=[l_theanine], secondary=[seaweed, calcium])
    target_with_extra["product_name"] = "l-theanine with calcium"
    target_exact = profile(other, primary=[l_theanine], secondary=[seaweed])
    target_exact["product_name"] = "l-theanine exact"
    ingredient_profiles = {
        l_theanine: {
            "ingredient_main_category": relaxation,
            "ingredient_sub_function_categories": ["\ud53c\ub85c\uac1c\uc120"],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
        seaweed: {
            "ingredient_main_category": other,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
        calcium: {
            "ingredient_main_category": "\ubf08 \uac74\uac15",
            "ingredient_sub_function_categories": ["\uc601\uc591\ubcf4\ucda9"],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    capped_score, _shared, capped_detail = calculate_semantic_weighted_jaccard_v2(
        base,
        target_with_extra,
        ingredient_profiles,
    )
    exact_score, _exact_shared, exact_detail = calculate_semantic_weighted_jaccard_v2(
        base,
        target_exact,
        ingredient_profiles,
    )

    assert capped_score == 0.64
    assert capped_detail["score_adjustments"][0]["type"] == "no_core_weak_shared_with_extra_score_cap"
    assert exact_score > 0.64
    assert not any(
        item["type"] == "no_core_weak_shared_with_extra_score_cap"
        for item in exact_detail["score_adjustments"]
    )


def test_semantic_v2_caps_no_core_weak_bone_subset_match():
    bone = "\ubf08 \uac74\uac15"
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    lipid = "\ud608\uc911\uc9c0\uc9c8"
    vitamin_d = "\ube44\ud0c0\ubbfc D"
    calcium = "\uce7c\uc298"
    lecithin = "\ub808\uc2dc\ud2f4"
    base = profile(bone, primary=[vitamin_d, calcium], secondary=[lecithin])
    base["product_name"] = "calcium 600mg"
    target = profile(bone, primary=[vitamin_d], secondary=[lecithin])
    target["product_name"] = "vitamin d only"
    ingredient_profiles = {
        vitamin_d: {
            "ingredient_main_category": nutrition,
            "ingredient_sub_function_categories": [bone],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
        calcium: {
            "ingredient_main_category": bone,
            "ingredient_sub_function_categories": [nutrition],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
        lecithin: {
            "ingredient_main_category": lipid,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, _shared, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "no_core_weak_shared_with_extra_score_cap"


def test_semantic_v2_caps_nutrition_main_generic_nutrient_only_match():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    vitamin_c = "\ube44\ud0c0\ubbfc C"
    zinc = "\uc544\uc5f0"
    base = profile(nutrition, primary=[vitamin_c, zinc])
    base["product_name"] = "base multivitamin"
    target = profile(nutrition, primary=[vitamin_c, zinc])
    target["product_name"] = "target multivitamin"
    ingredient_profiles = {
        vitamin_c: {
            "ingredient_main_category": nutrition,
            "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
        zinc: {
            "ingredient_main_category": nutrition,
            "ingredient_sub_function_categories": ["\uba74\uc5ed"],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "generic_nutrient_only_nutrition_main_score_cap"


def test_semantic_v2_caps_nutrition_main_generic_nutrient_dominant_match():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    lipid = "\ud608\uc911\uc9c0\uc9c8"
    ingredients = ["\ube44\ud0c0\ubbfc C", "\uc544\uc5f0", "\ub9c8\uadf8\ub124\uc298", "\uce7c\uc298", "\ub808\uc2dc\ud2f4"]
    base = profile(
        nutrition,
        primary=ingredients,
        ingredient_categories={"\ub808\uc2dc\ud2f4": lipid},
    )
    base["product_name"] = "base multi with lecithin"
    target = profile(
        nutrition,
        primary=ingredients,
        ingredient_categories={"\ub808\uc2dc\ud2f4": lipid},
    )
    target["product_name"] = "target multi with lecithin"
    ingredient_profiles = {
        "\ube44\ud0c0\ubbfc C": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\uc544\uc5f0": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\uba74\uc5ed"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ub9c8\uadf8\ub124\uc298": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ubf08 \uac74\uac15"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\uce7c\uc298": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ubf08 \uac74\uac15"], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        "\ub808\uc2dc\ud2f4": {"ingredient_main_category": lipid, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "generic_nutrient_dominant_nutrition_main_score_cap"
    assert "\ub808\uc2dc\ud2f4" in detail["score_adjustments"][0]["non_generic_nutrient_shared_keys"]


def test_semantic_v2_caps_nutrition_main_generic_nutrient_dominant_with_weak_adjuncts():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    gut = "\uc7a5 \uac74\uac15"
    fat = "\uccb4\uc9c0\ubc29"
    gos = "\uac08\ub77d\ud1a0\uc62c\ub9ac\uace0\ub2f9\ubd84\ub9d0"
    green_tea = "\ub179\ucc28\ucd94\ucd9c\ubb3c"
    fos = "\ud504\ub77d\ud1a0\uc62c\ub9ac\uace0\ub2f9"
    ingredients = [
        "\ube44\ud0c0\ubbfc B1",
        "\ube44\ud0c0\ubbfc B2",
        "\ube44\ud0c0\ubbfc B6",
        "\ube44\ud0c0\ubbfc B12",
        "\ube44\ud0c0\ubbfc C",
        "\ube44\ud0c0\ubbfc D",
        "\uc544\uc5f0",
        "\ub9c8\uadf8\ub124\uc298",
        "\uce7c\uc298",
        gos,
        green_tea,
        fos,
    ]
    base = profile(
        nutrition,
        primary=ingredients,
        ingredient_categories={gos: gut, green_tea: fat, fos: gut},
    )
    target = profile(
        nutrition,
        primary=ingredients,
        ingredient_categories={gos: gut, green_tea: fat, fos: gut},
    )
    ingredient_profiles = {
        "\ube44\ud0c0\ubbfc B1": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc B2": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc B6": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc B12": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc C": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc D": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ubf08 \uac74\uac15"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\uc544\uc5f0": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\uba74\uc5ed"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ub9c8\uadf8\ub124\uc298": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ubf08 \uac74\uac15"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\uce7c\uc298": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ubf08 \uac74\uac15"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        gos: {"ingredient_main_category": gut, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        green_tea: {"ingredient_main_category": fat, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        fos: {"ingredient_main_category": gut, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "generic_nutrient_dominant_nutrition_main_score_cap"
    assert set(detail["score_adjustments"][0]["non_generic_nutrient_shared_keys"]) == {gos, green_tea, fos}


def test_semantic_v2_caps_nutrition_generic_low_core_coverage_match():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    liver = "\uac04 \uac74\uac15"
    bone = "\ubf08 \uac74\uac15"
    male = "\ub0a8\uc131 \uac74\uac15"
    molybdenum = "\ubab0\ub9ac\ube0c\ub374"
    milk_thistle = "\ubc00\ud06c\uc528\uc2ac\ucd94\ucd9c\ubb3c"
    artichoke = "\uc544\ud2f0\ucd08\ud06c\ucd94\ucd9c\ubb3c"
    hovenia = "\ud5db\uac1c\ub098\ubb34"
    octacosanol = "\uc625\ud0c0\ucf54\uc0ac\ub180 \ud568\uc720 \uc720\uc9c0"
    vitamins = ["\ube44\ud0c0\ubbfc A", "\ube44\ud0c0\ubbfc B1", "\ube44\ud0c0\ubbfc B2", "\ube44\ud0c0\ubbfc B6", "\ube44\ud0c0\ubbfc C", "\uce7c\uc298", "\uc544\uc5f0", "\ud06c\ub86c"]
    base = profile(nutrition, primary=[molybdenum, milk_thistle, artichoke, hovenia], secondary=[octacosanol] + vitamins)
    base["llm_sub_function_categories"] = [liver, bone]
    target = profile(nutrition, primary=[molybdenum], secondary=[milk_thistle, hovenia, octacosanol] + vitamins)
    target["llm_sub_function_categories"] = [bone]
    ingredient_profiles = {
        molybdenum: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        milk_thistle: {"ingredient_main_category": liver, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        artichoke: {"ingredient_main_category": liver, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        hovenia: {"ingredient_main_category": liver, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        octacosanol: {"ingredient_main_category": male, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc A": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc B1": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc B2": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc B6": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc C": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\uce7c\uc298": {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\uc544\uc5f0": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\uba74\uc5ed"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ud06c\ub86c": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)
    metadata = recommendation_quality_metadata(score, detail)

    assert score == 0.64
    assert detail["score_adjustments"][-1]["type"] == "nutrition_generic_low_core_coverage_score_cap"
    assert metadata["recommendation_display_eligible"] is False
    assert metadata["recommendation_review_reason"] == "nutrition_generic_low_core_coverage"


def test_semantic_v2_caps_single_core_match_when_base_support_is_missing():
    immune = "\uba74\uc5ed"
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    bone = "\ubf08 \uac74\uac15"
    oral = "\uad6c\uac15 \uac74\uac15"
    red_ginseng = "\ud64d\uc0bc"
    xylitol = "\uc790\uc77c\ub9ac\ud1a8"
    vitamin_c = "\ube44\ud0c0\ubbfc C"
    vitamin_d = "\ube44\ud0c0\ubbfc D"
    zinc = "\uc544\uc5f0"
    base = profile(
        immune,
        primary=[red_ginseng],
        secondary=[vitamin_c, zinc, xylitol],
        support=[vitamin_d],
        ingredient_categories={red_ginseng: immune, xylitol: oral, vitamin_c: nutrition, vitamin_d: bone, zinc: immune},
    )
    target = profile(
        immune,
        primary=[red_ginseng],
        secondary=[xylitol],
        ingredient_categories={red_ginseng: immune, xylitol: oral},
    )
    better_target = profile(
        immune,
        primary=[red_ginseng],
        secondary=[vitamin_c, vitamin_d, zinc, xylitol],
        ingredient_categories={red_ginseng: immune, xylitol: oral, vitamin_c: nutrition, vitamin_d: bone, zinc: immune},
    )
    ingredient_profiles = {
        red_ginseng: {"ingredient_main_category": immune, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        xylitol: {"ingredient_main_category": oral, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        vitamin_c: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        vitamin_d: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        zinc: {"ingredient_main_category": immune, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)
    better_score, _, better_detail = calculate_semantic_weighted_jaccard_v2(base, better_target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "single_core_low_shared_coverage_score_cap"
    assert better_score > score
    assert not any(item["type"] == "single_core_low_shared_coverage_score_cap" for item in better_detail["score_adjustments"])


def test_semantic_v2_marks_single_core_with_weak_adjuncts_ineligible():
    hyaluronic = "\ud788\uc54c\ub8e8\ub860\uc0b0"
    lecithin = "\ub808\uc2dc\ud2f4"
    vitamin_b1 = "\ube44\ud0c0\ubbfc B1"
    vitamin_b2 = "\ube44\ud0c0\ubbfc B2"
    metadata = recommendation_quality_metadata(
        0.6236,
        {
            "base_semantic_ingredient_count": 12,
            "target_semantic_ingredient_count": 8,
            "core_coverage": {"shared_core_semantic_keys": [hyaluronic]},
            "shared_semantic_keys": [
                {"semantic_key": hyaluronic, "base_weight": 1.0, "target_weight": 1.0},
                {"semantic_key": lecithin, "base_weight": 0.3, "target_weight": 0.3},
                {"semantic_key": vitamin_b1, "base_weight": 0.046154, "target_weight": 0.05},
                {"semantic_key": vitamin_b2, "base_weight": 0.046154, "target_weight": 0.05},
            ],
        },
        core_match_score=1.0,
        function_similarity_score=0.56,
    )

    assert metadata["recommendation_display_eligible"] is False
    assert metadata["recommendation_review_reason"] == "single_core_low_shared_coverage"


def test_semantic_v2_caps_no_core_low_shared_partial_match():
    bone = "\ubf08 \uac74\uac15"
    joint = "\uad00\uc808/\uc5f0\uace8"
    immune = "\uba74\uc5ed"
    vitamin_d = "\ube44\ud0c0\ubbfc D"
    calcium = "\uce7c\uc298"
    magnesium = "\ub9c8\uadf8\ub124\uc298"
    manganese = "\ub9dd\uac04"
    zinc = "\uc544\uc5f0"
    vitamin_c = "\ube44\ud0c0\ubbfc C"
    msm = "\uc5e0\uc5d0\uc2a4\uc5e0(MSM, Methyl sulfonylmethane, \ub514\uba54\ud2f8\uc124\ud3f0)"
    base = profile(
        bone,
        primary=[vitamin_d, calcium],
        secondary=[magnesium, manganese, zinc, vitamin_c, msm],
        ingredient_categories={vitamin_d: bone, calcium: bone, magnesium: bone, manganese: bone, zinc: immune, vitamin_c: immune, msm: joint},
    )
    partial_target = profile(
        bone,
        primary=[vitamin_d],
        secondary=[msm],
        ingredient_categories={vitamin_d: bone, msm: joint},
    )
    stronger_target = profile(
        bone,
        primary=[vitamin_d, calcium],
        secondary=[magnesium, msm],
        ingredient_categories={vitamin_d: bone, calcium: bone, magnesium: bone, msm: joint},
    )
    ingredient_profiles = {
        vitamin_d: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        calcium: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        magnesium: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        manganese: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        zinc: {"ingredient_main_category": immune, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        vitamin_c: {"ingredient_main_category": immune, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        msm: {"ingredient_main_category": joint, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, partial_target, ingredient_profiles)
    stronger_score, _, stronger_detail = calculate_semantic_weighted_jaccard_v2(base, stronger_target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "no_core_low_shared_coverage_score_cap"
    assert stronger_score > score
    assert not any(item["type"] == "no_core_low_shared_coverage_score_cap" for item in stronger_detail["score_adjustments"])


def test_semantic_v2_caps_no_core_weak_shared_bone_adjunct_match():
    bone = "\ubf08 \uac74\uac15"
    joint = "\uad00\uc808/\uc5f0\uace8"
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    vitamin_d = "\ube44\ud0c0\ubbfc D"
    boswellia = "\ubcf4\uc2a4\uc6f0\ub9ac\uc544\ucd94\ucd9c\ubb3c"
    green_mussel = "\ucd08\ub85d\uc785\ud64d\ud569\ucd94\ucd9c\uc624\uc77c"
    calcium = "\uce7c\uc298"
    magnesium = "\ub9c8\uadf8\ub124\uc298"
    vitamin_k = "\ube44\ud0c0\ubbfc K"
    base = profile(
        bone,
        primary=[calcium, magnesium, vitamin_d, vitamin_k],
        secondary=[boswellia, green_mussel, "\uc544\uc5f0", "\ube44\ud0c0\ubbfc C"],
        ingredient_categories={calcium: bone, magnesium: bone, vitamin_d: bone, vitamin_k: bone, boswellia: joint, green_mussel: joint, "\uc544\uc5f0": nutrition, "\ube44\ud0c0\ubbfc C": nutrition},
    )
    target = profile(
        bone,
        primary=[vitamin_d],
        secondary=[boswellia, green_mussel],
        ingredient_categories={vitamin_d: bone, boswellia: joint, green_mussel: joint},
    )
    ingredient_profiles = {
        vitamin_d: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [nutrition], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        boswellia: {"ingredient_main_category": joint, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        green_mussel: {"ingredient_main_category": joint, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        calcium: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        magnesium: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        vitamin_k: {"ingredient_main_category": bone, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\uc544\uc5f0": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\uba74\uc5ed"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        "\ube44\ud0c0\ubbfc C": {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "no_core_low_shared_coverage_score_cap"


def test_semantic_v2_treats_nutrition_flavor_adjunct_as_weak_signal():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    oral = "\uad6c\uac15 \uac74\uac15"
    iron = "\ucca0"
    lemon = "\ub808\ubaac\ub18d\ucd95\ubd84\ub9d0"
    vitamin_c = "\ube44\ud0c0\ubbfc C"
    xylitol = "\uc790\uc77c\ub9ac\ud1a8"
    base = profile(nutrition, primary=[iron, lemon, vitamin_c, xylitol])
    base["product_name"] = "base iron"
    target = profile(nutrition, primary=[lemon, vitamin_c, xylitol])
    target["product_name"] = "target vitamin c"
    ingredient_profiles = {
        iron: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        lemon: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        vitamin_c: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        xylitol: {"ingredient_main_category": oral, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
    }

    vector = build_semantic_weight_vector(base, ingredient_profiles)
    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert vector[lemon] == 0.2
    assert vector[xylitol] == 0.2
    assert score == 0.52
    assert detail["base_nutrition_subtype"] == "iron"
    assert detail["target_nutrition_subtype"] == "vitamin_c"
    assert detail["score_adjustments"][-1]["type"] == "nutrition_subtype_mismatch_score_cap"


def test_semantic_v2_caps_nutrition_subtype_mismatch():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    iron = "\ucca0"
    vitamin_c = "\ube44\ud0c0\ubbfc C"
    vitamin_b2 = "\ube44\ud0c0\ubbfc B2"
    vitamin_b6 = "\ube44\ud0c0\ubbfc B6"
    lemon = "\ub808\ubaac\ub18d\ucd95\ubd84\ub9d0"
    base = profile(nutrition, primary=[iron, vitamin_c, vitamin_b2, vitamin_b6, lemon])
    base["product_name"] = "HerBalance Iron"
    target = profile(nutrition, primary=[vitamin_c, vitamin_b2, vitamin_b6, lemon])
    target["product_name"] = "\ub9ac\ud3ec\uc880 \ube44\ud0c0\ubbfcC"
    ingredient_profiles = {
        iron: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        vitamin_c: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        vitamin_b2: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        vitamin_b6: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
        lemon: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.52
    assert detail["base_nutrition_subtype"] == "iron"
    assert detail["target_nutrition_subtype"] == "vitamin_c"
    assert detail["score_adjustments"][-1]["type"] == "nutrition_subtype_mismatch_score_cap"


def test_semantic_v2_keeps_same_nutrition_subtype_match():
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    folate = "\uba54\ud2f8\ud14c\ud2b8\ub77c\ud788\ub4dc\ub85c\uc5fd\uc0b0\uae00\ub8e8\ucf54\uc0ac\ubbfc"
    vitamin_b6 = "\ube44\ud0c0\ubbfc B6"
    base = profile(nutrition, primary=[folate, vitamin_b6])
    base["product_name"] = "\ud761\uc218\uac00\uc798\ub418\ub294 \uc5fd\uc0b0"
    target = profile(nutrition, primary=[folate, vitamin_b6])
    target["product_name"] = "\ube0c\ub808\uc704\ub108 \ud65c\uc131\uc5fd\uc0b0"
    ingredient_profiles = {
        folate: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        vitamin_b6: {"ingredient_main_category": nutrition, "ingredient_sub_function_categories": [], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert detail["base_nutrition_subtype"] == "folate"
    assert detail["target_nutrition_subtype"] == "folate"
    assert "nutrition_subtype_mismatch_score_cap" not in [
        item["type"] for item in detail["score_adjustments"]
    ]
    assert score > 0.52


def test_semantic_v2_caps_weak_signal_only_sparse_match():
    antioxidant = "\ud56d\uc0b0\ud654"
    oral = "\uad6c\uac15 \uac74\uac15"
    xylitol = "\uc790\uc77c\ub9ac\ud1a8"
    base = profile(antioxidant, primary=[xylitol], ingredient_categories={xylitol: oral})
    base["product_name"] = "base c stick"
    target = profile(antioxidant, primary=[xylitol], ingredient_categories={xylitol: oral})
    target["product_name"] = "target c stick"
    ingredient_profiles = {
        xylitol: {
            "ingredient_main_category": oral,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "weak_signal_only_score_cap"


def test_semantic_v2_caps_cross_main_weak_signal_only_sparse_match():
    lipid = "\ud608\uc911\uc9c0\uc9c8"
    nutrition = "\uc601\uc591\ubcf4\ucda9"
    male = "\ub0a8\uc131 \uac74\uac15"
    bisuri = "\ube44\uc218\ub9ac\ucd94\ucd9c\ubd84\ub9d0"
    base = profile(lipid, primary=[bisuri], ingredient_categories={bisuri: male})
    base["product_name"] = "base blackcurrant"
    target = profile(nutrition, primary=[bisuri], ingredient_categories={bisuri: male})
    target["product_name"] = "target omega"
    ingredient_profiles = {
        bisuri: {
            "ingredient_main_category": male,
            "ingredient_sub_function_categories": [],
            "ingredient_type": "functional",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.45
    assert detail["score_adjustments"][0]["type"] == "weak_signal_only_cross_main_score_cap"


def test_semantic_v2_caps_cross_main_shared_subcategory_only_match():
    lipid = "\ud608\uc911\uc9c0\uc9c8"
    blood_flow = "\ud608\ud589"
    gut = "\uc7a5 \uac74\uac15"
    psyllium = "\ucc28\uc804\uc790\ud53c\uc2dd\uc774\uc12c\uc720"
    probiotics = "\ud504\ub85c\ubc14\uc774\uc624\ud2f1\uc2a4"
    magnesium = "\ub9c8\uadf8\ub124\uc298"
    base = profile(lipid, primary=[psyllium, probiotics], secondary=[magnesium])
    base["product_name"] = "base lipid gut adjunct"
    base["llm_sub_function_categories"] = [gut]
    target = profile(blood_flow, primary=[psyllium, probiotics], secondary=[magnesium])
    target["product_name"] = "target blood flow gut adjunct"
    target["llm_sub_function_categories"] = [gut]
    ingredient_profiles = {
        psyllium: {"ingredient_main_category": gut, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        probiotics: {"ingredient_main_category": gut, "ingredient_sub_function_categories": [], "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        magnesium: {"ingredient_main_category": "\uc601\uc591\ubcf4\ucda9", "ingredient_sub_function_categories": ["\ubf08 \uac74\uac15"], "ingredient_type": "nutrient", "vector_include": True, "is_excipient": False},
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)
    metadata = recommendation_quality_metadata(score, detail)

    assert score == 0.64
    assert detail["score_adjustments"][0]["type"] == "cross_main_shared_subcategory_only_score_cap"
    assert metadata["recommendation_display_eligible"] is False
    assert metadata["recommendation_review_reason"] == "cross_main_shared_subcategory_only"


def test_semantic_v2_caps_bone_health_generic_non_core_nutrient_only_match():
    bone = "\ubf08 \uac74\uac15"
    vitamin_c = "\ube44\ud0c0\ubbfc C"
    base = profile(bone, primary=[vitamin_c])
    base["product_name"] = "base vitamin d kids"
    target = profile(bone, primary=[vitamin_c])
    target["product_name"] = "target calcium"
    ingredient_profiles = {
        vitamin_c: {
            "ingredient_main_category": "\uc601\uc591\ubcf4\ucda9",
            "ingredient_sub_function_categories": ["\ud56d\uc0b0\ud654"],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.55
    assert detail["score_adjustments"][0]["type"] == "generic_nutrient_only_same_main_score_cap"


def test_semantic_v2_keeps_bone_health_core_nutrient_match_above_generic_cap():
    bone = "\ubf08 \uac74\uac15"
    calcium = "\uce7c\uc298"
    base = profile(bone, primary=[calcium])
    base["product_name"] = "base calcium"
    target = profile(bone, primary=[calcium])
    target["product_name"] = "target calcium"
    ingredient_profiles = {
        calcium: {
            "ingredient_main_category": "\uc601\uc591\ubcf4\ucda9",
            "ingredient_sub_function_categories": [bone],
            "ingredient_type": "nutrient",
            "vector_include": True,
            "is_excipient": False,
        },
    }

    score, _, detail = calculate_semantic_weighted_jaccard_v2(base, target, ingredient_profiles)

    assert score == 0.97
    assert detail["score_adjustments"][0]["type"] == "sparse_exact_score_cap"
