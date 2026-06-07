import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.ingredient_parse_service import (
    _validate_and_repair_parsed_result,
    canonicalize_ingredient_for_matching,
    classify_ingredient_role,
    rule_based_extract_ingredient_section,
    split_ingredients,
)
from api.ocr_text_sectionizer import sectionize_ocr_text


OCR_TEXT_WITH_WRAPPED_INGREDIENTS = "\n".join(
    [
        "원재료명 및 함량",
        "치커리뿌리추출분말(벨기에산), 파라다이스 그레인 추",
        "출분말(그래인오브파라다이스씨앗 95%, 말토덱스트린",
        "5%(중국산)), 브로콜리농축분말(브로콜리농축액(브로",
        "콜리농축액/국산), 덱스트린, 아라비아검), 양배추추출분",
        "말(국산), 흰강낭콩추출분말, 스테아린산마그네슘, 이산",
        "화규소",
        "제품명",
        "파라다이스 그레인 퍼닝",
        "식품의 유형",
        "고형차",
    ]
)


EXPECTED_INGREDIENTS = [
    "치커리뿌리추출분말(벨기에산)",
    "파라다이스 그레인 추출분말(그래인오브파라다이스씨앗 95%, 말토덱스트린 5%(중국산))",
    "브로콜리농축분말(브로콜리농축액(브로콜리농축액/국산), 덱스트린, 아라비아검)",
    "양배추추출분말(국산)",
    "흰강낭콩추출분말",
    "스테아린산마그네슘",
    "이산화규소",
]


def test_sectionizer_stops_ingredient_area_at_product_labels():
    sections = sectionize_ocr_text(OCR_TEXT_WITH_WRAPPED_INGREDIENTS)["sections"]

    assert "치커리뿌리추출분말" in sections["ingredient_area"]
    assert "화규소" in sections["ingredient_area"]
    assert "제품명" not in sections["ingredient_area"]
    assert "식품의 유형" not in sections["ingredient_area"]
    assert "고형차" not in sections["ingredient_area"]
    assert "파라다이스 그레인 퍼닝" in sections["product_name_area"]


def test_split_ingredients_rejoins_wrapped_korean_fragments():
    ingredients = split_ingredients(OCR_TEXT_WITH_WRAPPED_INGREDIENTS)

    assert ingredients == EXPECTED_INGREDIENTS
    assert "파라다이스 그레인 추" not in ingredients
    assert "출분말(그래인오브파라다이스씨앗 95%, 말토덱스트린 5%(중국산))" not in ingredients
    assert "양배추추출분" not in ingredients
    assert "말(국산)" not in ingredients
    assert "이산" not in ingredients
    assert "화규소" not in ingredients
    assert "제품명" not in ingredients


def test_rule_based_ocr_extract_returns_clean_product_and_ingredients():
    parsed = rule_based_extract_ingredient_section(OCR_TEXT_WITH_WRAPPED_INGREDIENTS)

    assert parsed["product_name_candidate"] == "파라다이스 그레인 퍼닝"
    assert parsed["raw_ingredients"] == EXPECTED_INGREDIENTS


def test_chicory_root_normalizes_to_existing_inulin_standard():
    assert canonicalize_ingredient_for_matching("치커리뿌리추출분말(벨기에산)") == "이눌린/치커리추출물"
    assert canonicalize_ingredient_for_matching("치커리") == "이눌린/치커리추출물"


def test_extract_powder_with_carrier_is_not_classified_as_excipient():
    assert classify_ingredient_role("파라다이스 그레인 추출분말(그래인오브파라다이스씨앗 95%, 말토덱스트린 5%(중국산))") != "excipient"
    assert classify_ingredient_role("브로콜리농축분말(브로콜리농축액(국산), 덱스트린, 아라비아검)") != "excipient"
    assert classify_ingredient_role("말토덱스트린") == "excipient"
    assert classify_ingredient_role("스테아린산마그네슘") == "excipient"


def test_validator_repairs_cached_excipient_role_for_extract_powder_with_carrier():
    parsed = {
        "ingredient_section_text": OCR_TEXT_WITH_WRAPPED_INGREDIENTS,
        "ingredient_objects": [
            {
                "raw": "파라다이스 그레인 추출분말(그래인오브파라다이스씨앗 95%, 말토덱스트린 5%(중국산))",
                "display_name": "파라다이스 그레인 추출분말",
                "normalized_for_matching": "파라다이스그레인",
                "role": "excipient",
            }
        ],
        "quality_warnings": [
            {
                "code": "excipient_in_core_role",
                "message": "부형제/첨가물 성격의 원료가 핵심 역할로 분류되어 보정했습니다: 파라다이스 그레인 추출분말",
                "severity": "critical",
            }
        ],
        "confidence": 0.7,
    }

    repaired = _validate_and_repair_parsed_result(parsed)

    assert repaired["ingredient_objects"][0]["role"] == "secondary"
    assert repaired["excluded_ingredients"] == []
    assert repaired["quality_warnings"] == []
