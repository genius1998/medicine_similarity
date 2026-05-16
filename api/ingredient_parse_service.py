from __future__ import annotations

import re
from typing import Any, Dict, List

from api.local_llm_client import call_local_llm, extract_json_from_llm_content


EXCIPIENT_KEYWORDS = [
    "히드록시프로필메틸셀룰로오스",
    "hpmc",
    "결정셀룰로스",
    "미결정셀룰로오스",
    "스테아린산마그네슘",
    "이산화규소",
    "카복시메틸셀룰로오스",
    "글리세린지방산에스테르",
    "캡슐기제",
    "젤라틴",
    "글리세린",
    "착색료",
    "이산화티타늄",
    "말토덱스트린",
    "덱스트린",
]

FUNCTIONAL_EXCEPTIONS = [
    "난소화성말토덱스트린",
]

PRODUCT_NAME_PATTERNS = [
    r"제품명\s*[:：]\s*(.+)",
    r"상품명\s*[:：]\s*(.+)",
]

INGREDIENT_SECTION_PATTERNS = [
    r"원재료명\s*[:：]\s*(.+)",
    r"원료명\s*[:：]\s*(.+)",
    r"기능성원료\s*[:：]\s*(.+)",
]

DAILY_INTAKE_PATTERNS = [
    r"섭취량\s*[:：]\s*(.+)",
    r"1일\s*섭취량\s*[:：]?\s*(.+)",
]

SECTION_STOP_KEYWORDS = [
    "섭취량",
    "섭취방법",
    "영양",
    "주의사항",
    "보관방법",
    "기준 및 규격",
]

CATEGORY_HINT_BY_INGREDIENT = {
    "홍삼": "면역",
    "진세노사이드": "면역",
    "루테인": "눈 건강",
    "지아잔틴": "눈 건강",
    "마리골드꽃추출물": "눈 건강",
    "프로바이오틱스": "장 건강",
    "유산균": "장 건강",
    "갈락토올리고당": "장 건강",
    "프락토올리고당": "장 건강",
    "난소화성말토덱스트린": "장 건강",
    "포스파티딜세린": "기억력/인지력",
    "은행잎추출물": "기억력/인지력",
    "DHA": "기억력/인지력",
    "저분자콜라겐펩타이드": "피부 건강",
    "콜라겐펩타이드": "피부 건강",
    "비오틴": "피부 건강",
    "비타민 A": "피부 건강",
    "히알루론산": "피부 건강",
    "난각막분말": "관절/연골",
    "난각막가수분해물": "관절/연골",
    "MSM": "관절/연골",
    "보스웰리아추출물": "관절/연골",
    "강황추출물": "관절/연골",
    "콘드로이친": "관절/연골",
    "뮤코다당단백": "관절/연골",
    "철갑상어연골": "관절/연골",
    "칼슘": "뼈 건강",
    "비타민 D": "뼈 건강",
    "비타민 K": "뼈 건강",
    "쏘팔메토열매추출물": "남성 건강",
    "옥타코사놀": "남성 건강",
    "녹용": "남성 건강",
    "아연": "남성 건강",
}


def normalize_spacing(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def is_excipient(name: str) -> bool:
    normalized = normalize_lookup_key(name)
    if any(normalize_lookup_key(item) in normalized for item in FUNCTIONAL_EXCEPTIONS):
        return False
    return any(normalize_lookup_key(item) in normalized for item in EXCIPIENT_KEYWORDS)


def _strip_parentheses(value: str) -> str:
    return normalize_spacing(re.sub(r"\([^)]*\)", "", value))


def canonicalize_ingredient_for_matching(name: str) -> str:
    value = normalize_spacing(name)
    if not value:
        return ""

    stripped = _strip_parentheses(value)
    collapsed = normalize_lookup_key(stripped)

    ordered_rules = [
        ("난소화성말토덱스트린", "난소화성말토덱스트린"),
        ("루테인지아잔틴", "루테인지아잔틴복합추출물20%"),
        ("루테인", "루테인"),
        ("지아잔틴", "지아잔틴"),
        ("마리골드", "마리골드꽃추출물"),
        ("진세노사이드", "진세노사이드"),
        ("홍삼농축액", "홍삼"),
        ("홍삼추출액", "홍삼"),
        ("홍삼", "홍삼"),
        ("포스파티딜세린", "포스파티딜세린"),
        ("phosphatidylserine", "포스파티딜세린"),
        ("은행잎추출물", "은행잎추출물"),
        ("은행잎", "은행잎추출물"),
        ("프로바이오틱스", "프로바이오틱스"),
        ("유산균", "프로바이오틱스"),
        ("갈락토올리고당", "갈락토올리고당"),
        ("프락토올리고당", "프락토올리고당"),
        ("dw2009", "프로바이오틱스"),
        ("난각막가수분해물", "난각막가수분해물"),
        ("난각막분말", "난각막분말"),
        ("난각막", "난각막분말"),
        ("msm", "MSM"),
        ("엠에스엠", "MSM"),
        ("보스웰리아", "보스웰리아추출물"),
        ("강황추출물", "강황추출물"),
        ("강황", "강황추출물"),
        ("콘드로이친", "콘드로이친"),
        ("뮤코다당단백", "뮤코다당단백"),
        ("철갑상어연골", "철갑상어연골"),
        ("저분자콜라겐펩타이드", "저분자콜라겐펩타이드"),
        ("콜라겐펩타이드", "콜라겐펩타이드"),
        ("콜라겐", "콜라겐펩타이드"),
        ("히알루론산", "히알루론산"),
        ("비오틴", "비오틴"),
        ("비타민a", "비타민 A"),
        ("비타민b1", "비타민 B1"),
        ("비타민b2", "비타민 B2"),
        ("비타민b6", "비타민 B6"),
        ("비타민b12", "비타민 B12"),
        ("비타민c", "비타민 C"),
        ("비타민d", "비타민 D"),
        ("비타민e", "비타민 E"),
        ("비타민k", "비타민 K"),
        ("산화아연", "아연"),
        ("아연", "아연"),
        ("셀렌", "셀레늄"),
        ("셀레늄", "셀레늄"),
        ("산화마그네슘", "마그네슘"),
        ("마그네슘", "마그네슘"),
        ("해조칼슘", "칼슘"),
        ("탄산칼슘", "칼슘"),
        ("칼슘", "칼슘"),
        ("옥타코사놀", "옥타코사놀"),
        ("쏘팔메토", "쏘팔메토열매추출물"),
        ("sawpalmetto", "쏘팔메토열매추출물"),
        ("녹용", "녹용"),
        ("l-아르기닌", "L-아르기닌"),
        ("l-글루타민", "L-글루타민"),
        ("dha", "DHA"),
    ]

    for token, replacement in ordered_rules:
        if token in collapsed:
            return replacement

    return stripped


def classify_ingredient_role(name: str) -> str:
    if is_excipient(name):
        return "excipient"

    normalized = canonicalize_ingredient_for_matching(name)
    collapsed = normalize_lookup_key(normalized)

    primary_tokens = [
        "홍삼",
        "진세노사이드",
        "루테인",
        "지아잔틴",
        "마리골드",
        "프로바이오틱스",
        "포스파티딜세린",
        "콜라겐",
        "난각막",
        "msm",
        "보스웰리아",
        "콘드로이친",
        "뮤코다당",
        "철갑상어연골",
        "쏘팔메토",
        "옥타코사놀",
        "녹용",
        "강황",
    ]
    support_tokens = [
        "비타민",
        "아연",
        "셀레늄",
        "칼슘",
        "마그네슘",
        "철",
        "엽산",
    ]

    if any(token in collapsed for token in primary_tokens):
        return "primary"
    if any(token in collapsed for token in support_tokens):
        return "support"
    return "secondary"


def split_ingredients(ingredient_text: str) -> List[str]:
    text = normalize_spacing(ingredient_text)
    if not text:
        return []

    text = re.sub(r"^(원재료명|원료명|기능성원료)\s*[:：]\s*", "", text)
    separators = ["\n", ";", "·", "ㆍ", "/"]
    for sep in separators:
        text = text.replace(sep, ",")

    parts = []
    seen = set()
    for token in text.split(","):
        cleaned = normalize_spacing(token)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            parts.append(cleaned)
    return parts


def _extract_first_match(text: str, patterns: List[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_spacing(match.group(1))
    return ""


def _guess_category_hint(normalized_name: str) -> str:
    return CATEGORY_HINT_BY_INGREDIENT.get(normalized_name, "")


def _normalize_quality_warnings(items: Any) -> List[dict]:
    warnings: List[dict] = []
    if not isinstance(items, list):
        return warnings
    for item in items:
        if isinstance(item, dict):
            warnings.append(item)
        elif isinstance(item, str) and item.strip():
            warnings.append({"code": "llm_warning", "message": item.strip()})
    return warnings


def _build_ingredient_objects(raw_ingredients: List[str]) -> List[dict]:
    objects: List[dict] = []
    for raw in raw_ingredients:
        normalized = canonicalize_ingredient_for_matching(raw)
        role = classify_ingredient_role(raw)
        display_name = _strip_parentheses(raw)
        objects.append(
            {
                "raw": normalize_spacing(raw),
                "normalized_for_matching": normalized,
                "display_name": display_name,
                "role": role,
                "category_hint": _guess_category_hint(normalized),
            }
        )
    return objects


def rule_based_extract_ingredient_section(raw_text: str) -> Dict[str, Any]:
    source_text = str(raw_text or "")
    lines = [normalize_spacing(line) for line in source_text.splitlines() if normalize_spacing(line)]

    product_name_candidate = _extract_first_match(source_text, PRODUCT_NAME_PATTERNS)
    ingredient_section_text = _extract_first_match(source_text, INGREDIENT_SECTION_PATTERNS)
    daily_intake_text = _extract_first_match(source_text, DAILY_INTAKE_PATTERNS)

    if not ingredient_section_text:
        for index, line in enumerate(lines):
            if any(label in line for label in ["원재료명", "원료명", "기능성원료"]):
                collected = [line]
                for next_line in lines[index + 1 : index + 5]:
                    if any(stop in next_line for stop in SECTION_STOP_KEYWORDS):
                        break
                    collected.append(next_line)
                ingredient_section_text = "\n".join(collected)
                break

    raw_ingredients = split_ingredients(ingredient_section_text)
    ingredient_objects = _build_ingredient_objects(raw_ingredients)
    normalized_ingredients = [
        item["normalized_for_matching"]
        for item in ingredient_objects
        if item["normalized_for_matching"] and item["role"] != "excipient"
    ]
    excluded_ingredients = [item["display_name"] for item in ingredient_objects if item["role"] == "excipient"]
    functional_candidates = [
        item["normalized_for_matching"]
        for item in ingredient_objects
        if item["role"] in {"primary", "secondary", "support"} and item["normalized_for_matching"]
    ]

    quality_warnings: List[dict] = []
    if not ingredient_section_text:
        quality_warnings.append(
            {
                "code": "ingredient_section_unclear",
                "message": "원재료명 영역이 명확하지 않습니다.",
            }
        )

    return {
        "product_name_candidate": product_name_candidate,
        "ingredient_section_text": ingredient_section_text,
        "functional_ingredient_candidates": list(dict.fromkeys(functional_candidates)),
        "raw_ingredients": raw_ingredients,
        "normalized_ingredients": list(dict.fromkeys(normalized_ingredients)),
        "ingredient_objects": ingredient_objects,
        "excluded_ingredients": list(dict.fromkeys(excluded_ingredients)),
        "quality_warnings": quality_warnings,
        "nutrition_or_active_components": [],
        "daily_intake_text": daily_intake_text,
        "confidence": 0.62 if normalized_ingredients else 0.2,
        "needs_user_review": True,
    }


def normalize_ingredients_with_llm(raw_text: str, ocr_lines: List[str]) -> Dict[str, Any]:
    system_prompt = (
        "너는 건강기능식품 라벨 OCR 텍스트에서 원재료명과 기능성원료를 추출하는 파서다.\n"
        "OCR 텍스트에는 줄바꿈 오류, 띄어쓰기 오류, 오타가 있을 수 있다.\n"
        "제품명과 원재료명이 충돌할 수 있으므로 둘을 분리해서 판단하라.\n"
        "OCR 텍스트에 여러 제품, 광고 문구, 추천 배너, 다른 제품명이 섞일 수 있다.\n"
        "제품명 주변의 원재료명 영역을 우선시하라.\n"
        "\"원재료명\", \"기능성원료\", \"영양·기능정보\" 근처 텍스트를 우선 추출하라.\n"
        "광고 문구, 다른 제품명, 배너 문구는 원재료명으로 넣지 마라.\n"
        "부형제, 캡슐기제, HPMC 등은 excipient로 분리하라.\n"
        "루테인, 홍삼, 프로바이오틱스 등 서로 다른 카테고리 성분이 섞여 있으면 needs_user_review=true로 표시하라.\n"
        "질병 치료나 효능 보장 해석은 하지 말고, 텍스트 구조화만 하라.\n"
        "반드시 JSON만 출력하라."
    )
    user_prompt = (
        "아래 OCR 텍스트에서 건강기능식품 추천에 사용할 원료 정보를 추출하라.\n\n"
        f"OCR_TEXT:\n{raw_text}\n\n"
        "OCR_LINES:\n"
        + "\n".join(ocr_lines[:80])
        + "\n\n"
        "출력 JSON 스키마:\n"
        "{\n"
        '  "product_name_candidate": "",\n'
        '  "ingredient_section_text": "",\n'
        '  "functional_ingredient_candidates": [],\n'
        '  "raw_ingredients": [],\n'
        '  "normalized_ingredients": [],\n'
        '  "ingredient_objects": [\n'
        '    {\n'
        '      "raw": "",\n'
        '      "normalized_for_matching": "",\n'
        '      "display_name": "",\n'
        '      "role": "primary|secondary|support|excipient|unknown",\n'
        '      "category_hint": ""\n'
        "    }\n"
        "  ],\n"
        '  "excluded_ingredients": [],\n'
        '  "quality_warnings": [],\n'
        '  "daily_intake_text": "",\n'
        '  "confidence": 0.0,\n'
        '  "needs_user_review": false\n'
        "}\n"
    )
    message = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    content = call_local_llm(message)
    parsed = extract_json_from_llm_content(content)
    if not parsed:
        raise RuntimeError("local LLM returned non-JSON content")
    return parsed


def parse_ingredients_from_ocr_text(raw_text: str) -> Dict[str, Any]:
    fallback = rule_based_extract_ingredient_section(raw_text)
    lines = [normalize_spacing(line) for line in str(raw_text or "").splitlines() if normalize_spacing(line)]

    try:
        llm_result = normalize_ingredients_with_llm(str(raw_text or ""), lines)
    except Exception as exc:  # noqa: BLE001
        fallback["quality_warnings"] = list(fallback.get("quality_warnings", [])) + [
            {
                "code": "llm_fallback",
                "message": f"LLM 파싱 실패로 rule-based fallback을 사용했습니다: {exc}",
            }
        ]
        llm_result = {}

    merged = {
        "product_name_candidate": str(llm_result.get("product_name_candidate") or fallback.get("product_name_candidate") or ""),
        "ingredient_section_text": str(llm_result.get("ingredient_section_text") or fallback.get("ingredient_section_text") or ""),
        "functional_ingredient_candidates": list(llm_result.get("functional_ingredient_candidates") or fallback.get("functional_ingredient_candidates") or []),
        "raw_ingredients": list(llm_result.get("raw_ingredients") or fallback.get("raw_ingredients") or []),
        "normalized_ingredients": list(llm_result.get("normalized_ingredients") or fallback.get("normalized_ingredients") or []),
        "ingredient_objects": list(llm_result.get("ingredient_objects") or fallback.get("ingredient_objects") or []),
        "excluded_ingredients": list(llm_result.get("excluded_ingredients") or fallback.get("excluded_ingredients") or []),
        "quality_warnings": _normalize_quality_warnings(llm_result.get("quality_warnings") or fallback.get("quality_warnings") or []),
        "nutrition_or_active_components": list(llm_result.get("nutrition_or_active_components") or []),
        "daily_intake_text": str(llm_result.get("daily_intake_text") or fallback.get("daily_intake_text") or ""),
        "confidence": float(llm_result.get("confidence") or fallback.get("confidence") or 0.0),
        "needs_user_review": bool(llm_result.get("needs_user_review", fallback.get("needs_user_review", False))),
    }

    if not merged["ingredient_objects"]:
        merged["ingredient_objects"] = _build_ingredient_objects(merged["raw_ingredients"])

    normalized_ingredients: List[str] = []
    excluded_ingredients = list(merged["excluded_ingredients"])

    for item in merged["ingredient_objects"]:
        raw = normalize_spacing(item.get("raw", ""))
        normalized_for_matching = canonicalize_ingredient_for_matching(item.get("normalized_for_matching") or raw)
        display_name = normalize_spacing(item.get("display_name") or _strip_parentheses(raw))
        role = str(item.get("role") or classify_ingredient_role(raw))
        if role == "unknown":
            role = classify_ingredient_role(raw)

        item.update(
            {
                "raw": raw,
                "normalized_for_matching": normalized_for_matching,
                "display_name": display_name,
                "role": role,
                "category_hint": str(item.get("category_hint") or _guess_category_hint(normalized_for_matching)),
            }
        )

        if role == "excipient":
            excluded_ingredients.append(display_name)
            continue

        if normalized_for_matching:
            normalized_ingredients.append(normalized_for_matching)

    merged["excluded_ingredients"] = list(dict.fromkeys([item for item in excluded_ingredients if item]))
    merged["normalized_ingredients"] = list(dict.fromkeys([item for item in normalized_ingredients if item]))
    merged["raw_ingredients"] = list(dict.fromkeys([normalize_spacing(item) for item in merged["raw_ingredients"] if normalize_spacing(item)]))

    if not merged["functional_ingredient_candidates"]:
        merged["functional_ingredient_candidates"] = merged["normalized_ingredients"][:]
    else:
        merged["functional_ingredient_candidates"] = list(
            dict.fromkeys(
                [
                    canonicalize_ingredient_for_matching(item)
                    for item in merged["functional_ingredient_candidates"]
                    if canonicalize_ingredient_for_matching(item)
                ]
            )
        )

    if merged["confidence"] < 0.8 or len(merged["normalized_ingredients"]) < 2 or not merged["ingredient_section_text"]:
        merged["needs_user_review"] = True
    if len(merged["normalized_ingredients"]) >= 20:
        merged["needs_user_review"] = True
        merged["quality_warnings"].append(
            {
                "code": "too_many_normalized_ingredients",
                "message": "정규화된 원료 수가 많아 OCR 결과 검토가 필요합니다.",
            }
        )
    if not merged["ingredient_section_text"]:
        merged["quality_warnings"].append(
            {
                "code": "ingredient_section_unclear",
                "message": "원재료명 영역이 명확하지 않습니다.",
            }
        )

    return merged
