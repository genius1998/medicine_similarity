from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.recommendation_service import RecommendationService
from api.upload_recommendation_service import UploadRecommendationService
from scripts.enhance_similarity_with_explanation import classify_substitutability, generate_recommendation_reason


IMAGE_PATH = Path(os.environ.get("TEST_PRODUCT_013_IMAGE", ROOT_DIR / "input_images" / "product_013.jpg"))
CHONDROITIN = "콘드로이친황산염"
UC_II = "닭가슴연골분말(UC-II)"
NAG = "NAG(엔에이지, N-아세틸글루코사민, N-Acetylglucosamine)"


def contains_uc_ii(value: str) -> bool:
    text = str(value or "")
    return "UC-II" in text or "닭가슴연골" in text or "닭연골" in text


def contains_chondroitin(value: str) -> bool:
    return "콘드로이친" in str(value or "")


def main() -> None:
    service = UploadRecommendationService(RecommendationService())
    service._ensure_loaded()

    for normalized, raw in [
        ("콘드로이친 황산염", "콘드로이친 황산염"),
        ("콘드로이친", "콘드로이친"),
    ]:
        match = service._find_match_in_cache(normalized, raw_ingredient=raw, display_name=raw)
        assert match is not None, f"match missing for {raw}"
        assert not contains_uc_ii(match["functional_ingredient_name"]), match
        assert contains_chondroitin(match["functional_ingredient_name"]), match

    assert IMAGE_PATH.exists(), f"missing test image: {IMAGE_PATH}"
    with IMAGE_PATH.open("rb") as file:
        result = service.recommend_from_uploaded_image(file.read(), IMAGE_PATH.name, top_k=10, candidate_limit=1000)

    profile = result["estimated_profile"]
    profile_ingredients = (
        list(profile.get("primary_ingredients", []))
        + list(profile.get("secondary_ingredients", []))
        + list(profile.get("support_ingredients", []))
    )
    assert any(contains_chondroitin(value) for value in profile_ingredients), profile

    raw_ocr_ingredients = list(result["parsed"].get("raw_ingredients", []))
    raw_mentions_uc_ii = any(contains_uc_ii(value) for value in raw_ocr_ingredients)
    if not raw_mentions_uc_ii:
        assert all(not contains_uc_ii(value) for value in profile_ingredients), profile

    problematic_rows = [
        row
        for row in result["recommendations"]
        if "N-아세틸글루코사민" in str(row.get("product_name", ""))
    ]
    if problematic_rows:
        row = problematic_rows[0]
        assert all(not contains_uc_ii(value) for value in row.get("shared_ingredients", [])), row
        assert "UC-II" not in str(row.get("reason", "")), row
        assert "닭가슴연골" not in str(row.get("reason", "")), row

    base_profile = {
        "product_name": "관절엔 콘드로이친 1200",
        "product_main_category": "관절/연골",
        "primary_ingredients": [CHONDROITIN],
        "secondary_ingredients": [],
        "support_ingredients": [],
        "ingredient_scores": [{"ingredient": CHONDROITIN, "role": "primary", "category_main": "관절/연골"}],
    }
    target_profile = {
        "product_name": "관절의 힘 N-아세틸글루코사민",
        "product_main_category": "관절/연골",
        "primary_ingredients": [NAG],
        "secondary_ingredients": [UC_II],
        "support_ingredients": [],
        "ingredient_scores": [
            {"ingredient": NAG, "role": "primary", "category_main": "관절/연골"},
            {"ingredient": UC_II, "role": "secondary", "category_main": "관절/연골"},
        ],
    }
    comparison = {
        "shared_ingredients": [UC_II],
        "base_only_ingredients": [CHONDROITIN],
        "target_only_ingredients": [NAG],
        "shared_categories": ["관절/연골"],
        "different_categories": [],
    }
    reason = generate_recommendation_reason(base_profile, target_profile, comparison, {})
    assert "UC-II" not in reason, reason
    assert "닭가슴연골" not in reason, reason
    assert classify_substitutability(0.5, base_profile, target_profile) == "낮음"

    print("PASS: chondroitin mapping guard")


if __name__ == "__main__":
    main()
