from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.db import sqlite_connection
from api.ingredient_parse_service import parse_ingredients_from_ocr_text
from api.ocr_service import extract_text_from_image
from api.recommendation_service import RecommendationService, UPLOADED_PRODUCT_TABLE_NAME
from api.upload_recommendation_service import UploadRecommendationService


IMAGE_PATH = Path(os.environ.get("TEST_PRODUCT_013_IMAGE", ROOT_DIR / "input_images" / "product_013.jpg"))
EXPECTED_RAW_INGREDIENTS = [
    "콘드로이친 황산염(아르헨티나산)",
    "탄산칼슘",
    "히드록시프로필메틸셀룰로스",
    "해조분말(영국산)",
    "이산화규소",
    "스테아린산마그네슘",
    "카복시메틸셀룰로스칼슘",
    "글리세린지방산에스테르",
    "건조효모(셀렌, 미국산)",
    "비타민C",
]


def main() -> None:
    assert IMAGE_PATH.exists(), f"missing test image: {IMAGE_PATH}"

    ocr_payload = extract_text_from_image(IMAGE_PATH)
    raw_text = str(ocr_payload.get("raw_text", "") or "")
    assert "탄산칼슘" in raw_text, raw_text
    assert "비타민C" in raw_text, raw_text

    recommendation_service = RecommendationService()
    recommendation_service.ensure_loaded()
    parsed = parse_ingredients_from_ocr_text(raw_text, recommendation_service.runtime["sqlite_path"])

    raw_ingredients = list(parsed.get("raw_ingredients", []))
    assert len(raw_ingredients) >= len(EXPECTED_RAW_INGREDIENTS), raw_ingredients
    for expected in EXPECTED_RAW_INGREDIENTS:
        assert expected in raw_ingredients, raw_ingredients
    assert all("1일 1회" not in item for item in raw_ingredients), raw_ingredients
    assert all("섭취" not in item for item in raw_ingredients), raw_ingredients

    upload_service = UploadRecommendationService(recommendation_service)
    with IMAGE_PATH.open("rb") as file:
        result = upload_service.recommend_from_uploaded_image(file.read(), IMAGE_PATH.name, top_k=5, candidate_limit=200)

    saved_product = dict(result.get("saved_product") or {})
    assert saved_product.get("report_no"), saved_product

    with sqlite_connection(recommendation_service.runtime["sqlite_path"]) as conn:
        row = conn.execute(
            f"""
            SELECT raw_ingredients, parsed_json
            FROM {UPLOADED_PRODUCT_TABLE_NAME}
            WHERE report_no = ?
            """,
            (saved_product["report_no"],),
        ).fetchone()

    assert row is not None, saved_product
    stored_raw_ingredients = str(row[0] or "")
    for expected in ("콘드로이친 황산염", "탄산칼슘", "비타민C"):
        assert expected in stored_raw_ingredients, stored_raw_ingredients

    parsed_json = json.loads(str(row[1] or "{}"))
    stored_list = list(parsed_json.get("raw_ingredients", []))
    for expected in EXPECTED_RAW_INGREDIENTS:
        assert expected in stored_list, stored_list
    assert all("섭취" not in item for item in stored_list), stored_list

    print("PASS: full ingredient capture and storage")


if __name__ == "__main__":
    main()
