from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from api.db import default_sqlite_path, sqlite_connection
from api.local_llm_client import call_local_llm, extract_json_from_llm_content
from api.ocr_text_sectionizer import sectionize_ocr_text


PARSE_PROMPT_VERSION = "ingredient_parse_v2_sectionized"
PARSE_SCHEMA_VERSION = "ingredient_parse_schema_v1"
PARSE_NORMALIZER_VERSION = "ingredient_normalizer_v2"
PARSE_SECTIONIZER_VERSION = "ocr_text_sectionizer_v1"
LLM_PARSE_CACHE_TABLE_NAME = "llm_parse_cache"


EXCIPIENT_KEYWORDS = [
    "히드록시프로필메틸셀룰로오스",
    "hpmc",
    "결정셀룰로오스",
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
    "정제수",
    "옥수수전분",
    "펙틴",
    "목화씨유분말",
    "스테비아",
    "감미료",
    "폴리에틸렌",
    "pe",
    "maltodextrin",
    "glycerin",
]

FUNCTIONAL_EXCEPTIONS = [
    "난소화성말토덱스트린",
    "난각막",
    "난각막분말",
    "난각막가수분해물",
    "nem",
    "demr",
    "난각칼슘",
    "난각분말",
    "계란껍질분말",
    "프로바이오틱스",
    "프리바이오틱스",
    "갈락토올리고당",
    "프락토올리고당",
    "이눌린",
    "치커리추출물",
]

PROBIOTIC_CANONICAL_KEYWORDS = [
    "프로바이오틱스",
    "유산균",
    "lactiplantibacillus",
    "lacticaseibacillus",
    "lactobacillus",
    "bifidobacterium",
    "bifidobacteri",
    "streptococcus thermophilus",
    "streptococcus",
    "enterococcus",
    "bacillus coagulans",
    "plantarum",
    "rhamnosus",
    "acidophilus",
    "casei",
    "longum",
    "breve",
    "lactis",
    "fermentum",
    "paracasei",
    "salivarius",
    "reuteri",
    "락티플란티바실러스",
    "리모시락토바실러스",
    "락토바실러스",
    "비피도박테리움",
    "스트렙토코커스",
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

ALLERGEN_NOTICE_PATTERNS = [
    r"함유",
    r"알레르기",
    r"알러지",
    r"본 제품은 .* 사용한 제품과 같은 제조시설",
]

ALLERGEN_WARNING_TOKENS = {
    "알류",
    "알류(가금류)",
    "알류(달걀)",
    "알류(계란)",
    "계란",
    "가금류",
    "우유",
    "메밀",
    "땅콩",
    "대두",
    "밀",
    "고등어",
    "게",
    "새우",
    "돼지고기",
    "복숭아",
    "토마토",
    "호두",
    "닭고기",
    "쇠고기",
    "소고기",
    "오징어",
    "조개",
    "조개류",
    "전복",
    "홍합",
    "굴",
    "잣",
    "아황산류",
    "가금류(알류)",
    "가금류",
    "알류",
}

NOTICE_WARNING_PATTERNS = [
    r"알레르기 체질",
    r"알레르기 반응 가능성",
    r"알러지 유발물질 안내",
    r"유발물질 안내",
    r"특정 알레르기 체질 주의",
    r"특이체질",
    r"함유",
    r"제조시설",
    r"섭취 시 주의",
    r"섭취를 피하시기",
    r"섭취를 피",
    r"임산부",
    r"수유부",
    r"어린이",
    r"전문가와 상담",
    r"이상사례",
    r"어린이의 손에 닿지",
    r"직사광선",
    r"고온다습",
    r"보관",
    r"소비기한",
    r"고객상담실",
    r"아황산류",
]

INFO_WARNING_PATTERNS = [
    r"제조원",
    r"판매원",
    r"유통전문판매원",
    r"영양정보",
    r"영양기능정보",
    r"영양성분",
    r"제품명",
    r"내용량",
]

CRITICAL_WARNING_PATTERNS = [
    r"원재료명 영역",
    r"파싱 실패",
    r"여러 제품",
    r"혼합",
    r"기능성원료 후보 없음",
]

CATEGORY_HINT_BY_INGREDIENT = {
    "홍삼": "면역",
    "진세노사이드": "면역",
    "루테인": "눈 건강",
    "루테인지아잔틴복합추출물20%": "눈 건강",
    "지아잔틴": "눈 건강",
    "마리골드꽃추출물": "눈 건강",
    "프로바이오틱스": "장 건강",
    "갈락토올리고당": "장 건강",
    "프락토올리고당": "장 건강",
    "이눌린/치커리추출물": "장 건강",
    "난소화성말토덱스트린": "장 건강",
    "바나바잎 추출물": "혈당",
    "코로솔산": "혈당",
    "구아바잎 추출물": "혈당",
    "달맞이꽃종자추출물": "혈당",
    "구아검가수분해물": "혈당",
    "식이섬유": "혈당",
    "키토산": "체지방",
    "키토올리고당": "체지방",
    "가르시니아캄보지아추출물": "체지방",
    "공액리놀레산": "체지방",
    "녹차추출물": "체지방",
    "카테킨": "체지방",
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
    "난각칼슘": "뼈 건강",
    "난각분말": "뼈 건강",
    "계란껍질분말": "뼈 건강",
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


def _contains_keyword(text: str, keyword: str) -> bool:
    source = str(text or "")
    normalized = normalize_lookup_key(source)
    target = normalize_lookup_key(keyword)
    if not target:
        return False
    if len(target) <= 2 and target.isascii():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])", source.lower()))
    return target in normalized


def _looks_like_functional_premix(name: str) -> bool:
    normalized = normalize_lookup_key(name)
    if "혼합제제" not in normalized:
        return False
    functional_tokens = [
        "비타민",
        "레티닐",
        "토코페롤",
        "아연",
        "셀렌",
        "셀레늄",
        "엽산",
        "나이아신",
        "니코틴산아미드",
        "망간",
        "구리",
        "비오틴",
        "실리마린",
        "밀크씨슬",
        "마리골드",
    ]
    return any(token in normalized for token in functional_tokens)


def is_excipient(name: str) -> bool:
    normalized = normalize_lookup_key(name)
    if not normalized:
        return False
    if _looks_like_functional_premix(name):
        return False
    if any(normalize_lookup_key(item) in normalized for item in FUNCTIONAL_EXCEPTIONS):
        return False
    return any(normalize_lookup_key(item) in normalized for item in EXCIPIENT_KEYWORDS)


def _is_probiotic_strain_text(value: str) -> bool:
    collapsed = normalize_lookup_key(value)
    if any(token in collapsed for token in PROBIOTIC_CANONICAL_KEYWORDS):
        return True
    return bool(
        re.search(
            r"\b(?:l|bi|b|s|lc)\.?\s*(?:plantarum|rhamnosus|acidophilus|casei|longum|breve|lactis|fermentum|paracasei|salivarius|reuteri|gasseri|helveticus|bulgaricus|thermophilus|animalis)\b",
            str(value or ""),
            flags=re.IGNORECASE,
        )
    )


def _strip_parentheses(value: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value or ""))
    text = re.sub(r"\[[^\]]*\]", "", text)
    return normalize_spacing(text)


def _remove_percentage_suffix(value: str) -> str:
    return normalize_spacing(re.sub(r"\s*\d+(?:\.\d+)?\s*%$", "", value))


def _looks_like_allergen_notice(value: str) -> bool:
    text = normalize_spacing(value)
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in ALLERGEN_NOTICE_PATTERNS)


def _looks_like_allergen_warning(value: str) -> bool:
    text = normalize_spacing(value)
    if not text:
        return False
    lowered = normalize_lookup_key(text)
    if _looks_like_allergen_notice(text):
        return True
    if "알러지유발물질안내" in lowered:
        return True
    if "조개류" in lowered:
        return True
    if "제조시설" in lowered:
        return True
    return lowered in {normalize_lookup_key(item) for item in ALLERGEN_WARNING_TOKENS}


def classify_warning_message(message: str, default_code: str = "llm_warning") -> dict:
    text = normalize_spacing(message)
    if not text:
        return {"code": default_code, "message": "", "severity": "notice"}
    if _looks_like_allergen_warning(text):
        return {"code": "allergen_notice", "message": text, "severity": "notice"}
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NOTICE_WARNING_PATTERNS):
        return {"code": "notice_warning", "message": text, "severity": "notice"}
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in INFO_WARNING_PATTERNS):
        return {"code": "info_warning", "message": text, "severity": "info"}
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in CRITICAL_WARNING_PATTERNS):
        return {"code": "critical_warning", "message": text, "severity": "critical"}
    return {"code": default_code, "message": text, "severity": "warning"}


def canonicalize_ingredient_for_matching(name: str) -> str:
    value = normalize_spacing(name)
    if not value:
        return ""

    stripped = _strip_parentheses(value)
    collapsed = normalize_lookup_key(stripped)

    if _is_probiotic_strain_text(stripped):
        return "프로바이오틱스"

    ordered_rules = [
        ("진세노사이드", "진세노사이드"),
        ("홍삼농축액", "홍삼"),
        ("홍삼추출액", "홍삼"),
        ("홍삼", "홍삼"),
        ("milkthistleextract", "밀크씨슬추출물"),
        ("milkthistle", "밀크씨슬"),
        ("silymarin", "실리마린"),
        ("루테인지아잔틴", "루테인지아잔틴복합추출물20%"),
        ("루테인추출물", "루테인"),
        ("루테인", "루테인"),
        ("지아잔틴", "지아잔틴"),
        ("마리골드", "마리골드꽃추출물"),
        ("dw2009", "프로바이오틱스"),
        ("lactiplantibacillus", "프로바이오틱스"),
        ("lacticaseibacillus", "프로바이오틱스"),
        ("lactobacillus", "프로바이오틱스"),
        ("bifidobacterium", "프로바이오틱스"),
        ("bifidobacteri", "프로바이오틱스"),
        ("streptococcus", "프로바이오틱스"),
        ("락티플란티바실러스", "프로바이오틱스"),
        ("리모시락토바실러스", "프로바이오틱스"),
        ("락토바실러스", "프로바이오틱스"),
        ("비피도박테리움", "프로바이오틱스"),
        ("스트렙토코커스", "프로바이오틱스"),
        ("락티스", "프로바이오틱스"),
        ("프로바이오틱스", "프로바이오틱스"),
        ("유산균", "프로바이오틱스"),
        ("갈락토올리고당", "갈락토올리고당"),
        ("프락토올리고당", "프락토올리고당"),
        ("이눌린/치커리추출물", "이눌린/치커리추출물"),
        ("이눌린", "이눌린/치커리추출물"),
        ("치커리추출물", "이눌린/치커리추출물"),
        ("난소화성말토덱스트린", "난소화성말토덱스트린"),
        ("바나바잎추출물", "바나바잎 추출물"),
        ("바나바잎추출", "바나바잎 추출물"),
        ("바나바", "바나바잎 추출물"),
        ("코로솔산", "코로솔산"),
        ("구아바잎추출물", "구아바잎 추출물"),
        ("구아바잎추출분말", "구아바잎 추출물"),
        ("달맞이꽃종자추출물", "달맞이꽃종자추출물"),
        ("구아검가수분해물", "구아검가수분해물"),
        ("식이섬유", "식이섬유"),
        ("키토올리고당", "키토올리고당"),
        ("키토산", "키토산"),
        ("가르시니아", "가르시니아캄보지아추출물"),
        ("hca", "가르시니아캄보지아추출물"),
        ("cla", "공액리놀레산"),
        ("공액리놀레산", "공액리놀레산"),
        ("카테킨", "카테킨"),
        ("demr", "난각막분말"),
        ("난각막가수분해물", "난각막가수분해물"),
        ("nem", "난각막가수분해물"),
        ("난각막분말", "난각막분말"),
        ("난각막", "난각막분말"),
        ("난각칼슘", "난각칼슘"),
        ("계란껍질분말", "계란껍질분말"),
        ("난각분말", "난각분말"),
        ("msm", "MSM"),
        ("엠에스엠", "MSM"),
        ("보스웰리아", "보스웰리아추출물"),
        ("강황추출물", "강황추출물"),
        ("강황", "강황추출물"),
        ("울금추출물", "강황추출물"),
        ("콘드로이친", "콘드로이친"),
        ("뮤코다당단백", "뮤코다당단백"),
        ("뮤코다당.단백", "뮤코다당단백"),
        ("뮤코다당·단백", "뮤코다당단백"),
        ("철갑상어연골", "철갑상어연골"),
        ("상어연골", "철갑상어연골"),
        ("초록입홍합추출오일", "초록입홍합추출오일"),
        ("초록입홍합분말", "초록입홍합분말"),
        ("포스파티딜세린", "포스파티딜세린"),
        ("phosphatidylserine", "포스파티딜세린"),
        ("은행잎추출물", "은행잎추출물"),
        ("은행잎", "은행잎추출물"),
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

    cleaned = _remove_percentage_suffix(stripped)
    return cleaned


def _build_display_name(raw: str, normalized: str) -> str:
    stripped = _remove_percentage_suffix(_strip_parentheses(raw))
    collapsed = normalize_lookup_key(raw)

    if "dw2009" in collapsed:
        return "DW2009 프로바이오틱스 복합물"
    if "lactiplantibacillus" in collapsed and "프로바이오틱스" in collapsed:
        return "프로바이오틱스 복합물"
    if any(
        token in collapsed
        for token in [
            "lactiplantibacillus",
            "lacticaseibacillus",
            "lactobacillus",
            "bifidobacteri",
            "streptococcus",
            "락티플란티바실러스",
            "리모시락토바실러스",
            "락토바실러스",
            "비피도박테리움",
            "스트렙토코커스",
            "락티스",
        ]
    ):
        return "프로바이오틱스"
    if _is_probiotic_strain_text(raw):
        return stripped
    if normalized == "프로바이오틱스":
        return "프로바이오틱스"
    if normalized in {"갈락토올리고당", "프락토올리고당", "이눌린/치커리추출물"}:
        return normalized
    if normalized == "루테인":
        return "루테인"
    if normalized == "홍삼":
        return "홍삼"
    if normalized == "진세노사이드":
        return "진세노사이드"
    if normalized == "난각막가수분해물":
        return "난각막가수분해물"
    if normalized == "난각막분말":
        return "난각막분말"
    if normalized == "난각칼슘":
        return "난각칼슘"
    if normalized == "난각분말":
        return "난각분말"
    if normalized == "계란껍질분말":
        return "계란껍질분말"
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
        "갈락토올리고당",
        "프락토올리고당",
        "이눌린",
        "포스파티딜세린",
        "콜라겐",
        "난각막",
        "msm",
        "보스웰리아",
        "콘드로이친",
        "뮤코다당",
        "철갑상어연골",
        "초록입홍합",
        "쏘팔메토",
        "옥타코사놀",
        "녹용",
        "강황",
        "키토산",
        "바나바",
        "코로솔산",
        "난소화성말토덱스트린",
        "구아바",
        "식이섬유",
        "이눌린",
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
    separators = ["\n", ";", "·", "ㆍ"]
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
    if normalized_name in {"밀크씨슬", "밀크씨슬추출물", "실리마린"}:
        return "간 건강"
    return CATEGORY_HINT_BY_INGREDIENT.get(normalized_name, "")


def _normalize_quality_warnings(items: Any) -> List[dict]:
    warnings: List[dict] = []
    if not isinstance(items, list):
        return warnings
    for item in items:
        if isinstance(item, dict):
            warning = dict(item)
            code = str(warning.get("code", "") or "").strip() or "unknown_warning"
            message = str(warning.get("message", "") or "").strip()
            severity = str(warning.get("severity", "") or "").strip().lower()
            if severity not in {"critical", "warning", "notice", "info"}:
                inferred = classify_warning_message(message, code)
                code = str(inferred.get("code", "") or code)
                message = str(inferred.get("message", "") or message)
                severity = str(inferred.get("severity", "") or "warning")
            warning["code"] = code
            warning["message"] = message
            warning["severity"] = severity or "warning"
            warnings.append(warning)
        elif isinstance(item, str) and item.strip():
            warnings.append(classify_warning_message(item.strip()))
    return warnings


def _build_sectionized_context(raw_text: str) -> Dict[str, Any]:
    context = sectionize_ocr_text(raw_text)
    sections = dict(context.get("sections", {}) or {})
    llm_blocks: List[str] = []
    for key in ["product_name_area", "ingredient_area", "functional_info_area", "intake_area"]:
        value = str(sections.get(key, "") or "").strip()
        if value:
            llm_blocks.append(f"[{key.upper()}]\n{value}")
    llm_text = "\n\n".join(llm_blocks).strip() or str(raw_text or "")
    llm_lines = [normalize_spacing(line) for line in llm_text.splitlines() if normalize_spacing(line)]
    return {
        **context,
        "llm_text": llm_text,
        "llm_lines": llm_lines,
    }


def _derive_ingredient_section_text(source_text: str, sections: Dict[str, Any]) -> str:
    ingredient_area = normalize_spacing(str(sections.get("ingredient_area", "") or ""))
    functional_area = str(sections.get("functional_info_area", "") or "")
    source_text = str(source_text or "")

    if ingredient_area and len(re.sub(r"\s+", "", ingredient_area)) >= 20 and "," in ingredient_area:
        return ingredient_area

    if functional_area:
        lines = [line.strip() for line in functional_area.splitlines() if line.strip()]
        collecting = False
        collected: List[str] = []
        for line in lines:
            normalized_line = normalize_spacing(line)
            if not normalized_line:
                continue
            if any(token in normalized_line for token in ["영양기능정보", "영양성분기준치", "%영양성분기준치", "1일 섭취량 당"]):
                break
            if not collecting and any(token in normalized_line for token in ["원료명", "원료명 및 함량", "마리골드", "밀크씨슬", "키토산분말", "캡슐기제"]):
                collecting = True
            if collecting:
                if any(token in normalized_line for token in ["제품명", "제품의유형", "내용량", "눈 건강&영양소", "건강기능식품 /"]):
                    continue
                collected.append(normalized_line)
        candidate = normalize_spacing(" ".join(collected))
        if len(re.sub(r"\s+", "", candidate)) >= 20 and "," in candidate:
            return candidate

    extracted = _extract_first_match(source_text, INGREDIENT_SECTION_PATTERNS)
    if extracted:
        return extracted

    return ingredient_area


def _validate_and_repair_parsed_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    try:
        quality_warnings = list(parsed.get("quality_warnings", []))
        corrected_fields: List[str] = []
        rebuilt_objects: List[dict] = []
        seen_by_normalized: Dict[str, dict] = {}
        role_priority = {"primary": 0, "secondary": 1, "support": 2, "excipient": 3}
        source_text = "\n".join(
            [
                str(parsed.get("ingredient_section_text", "") or ""),
                str(parsed.get("raw_text", "") or ""),
                str(parsed.get("source_text", "") or ""),
            ]
        )
        source_lookup = normalize_lookup_key(source_text)
        low_confidence_primary_count = 0
        unknown_count = 0

        for item in parsed.get("ingredient_objects", []) or []:
            rebuilt = dict(item or {})
            raw = normalize_spacing(rebuilt.get("raw", ""))
            display_name = normalize_spacing(rebuilt.get("display_name", raw))
            normalized = canonicalize_ingredient_for_matching(rebuilt.get("normalized_for_matching") or raw or display_name)
            standard_name = normalize_spacing(rebuilt.get("standard_name", normalized))
            evidence_text = normalize_spacing(rebuilt.get("evidence_text", ""))
            confidence = float(rebuilt.get("confidence", parsed.get("confidence", 0.0)) or 0.0)
            role = normalize_spacing(rebuilt.get("role", "")) or classify_ingredient_role(raw or display_name)
            original_role = role
            dedupe_key = normalize_lookup_key(normalized or standard_name or display_name or raw)

            if role not in {"primary", "secondary", "support", "excipient"}:
                role = classify_ingredient_role(raw or display_name)
                if role != original_role:
                    corrected_fields.append("ingredient_objects.role")

            excipient_like = is_excipient(raw or display_name)
            if excipient_like and role in {"primary", "secondary"}:
                quality_warnings.append(
                    {
                        "code": "excipient_in_core_role",
                        "message": f"부형제/첨가물 성격의 원료가 핵심 역할로 분류되어 보정했습니다: {display_name or raw}",
                        "severity": "critical" if role == "primary" else "warning",
                    }
                )
                role = "support" if role == "secondary" else "excipient"
                corrected_fields.append("ingredient_objects.role")
            elif excipient_like:
                role = "excipient"

            if evidence_text and source_lookup and normalize_lookup_key(evidence_text) not in source_lookup:
                quality_warnings.append(
                    {
                        "code": "evidence_text_not_found",
                        "message": f"근거 텍스트가 OCR 원문과 일치하지 않습니다: {display_name or raw}",
                        "severity": "warning",
                    }
                )

            if role == "primary" and confidence < 0.45:
                low_confidence_primary_count += 1
                quality_warnings.append(
                    {
                        "code": "low_confidence_primary_ingredient",
                        "message": f"신뢰도가 낮은 원료가 primary로 분류되었습니다: {display_name or raw}",
                        "severity": "warning",
                        "confidence": round(confidence, 4),
                    }
                )

            if role == "primary" and not normalize_lookup_key(normalized or standard_name):
                quality_warnings.append(
                    {
                        "code": "missing_normalized_primary",
                        "message": f"정규화 정보가 없는 원료가 primary로 분류되었습니다: {display_name or raw}",
                        "severity": "critical",
                    }
                )
                corrected_fields.append("ingredient_objects.normalized_for_matching")

            if not normalize_lookup_key(normalized or standard_name):
                unknown_count += 1

            rebuilt.update(
                {
                    "raw": raw,
                    "display_name": display_name,
                    "normalized_for_matching": normalized,
                    "standard_name": standard_name,
                    "evidence_text": evidence_text,
                    "confidence": round(confidence, 4),
                    "role": role,
                }
            )

            if dedupe_key and dedupe_key in seen_by_normalized:
                existing = seen_by_normalized[dedupe_key]
                existing_role = str(existing.get("role", "support") or "support")
                if role_priority.get(role, 9) < role_priority.get(existing_role, 9):
                    existing["role"] = role
                    existing["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), confidence)
                corrected_fields.append("ingredient_objects.deduped")
                continue

            if dedupe_key:
                seen_by_normalized[dedupe_key] = rebuilt
            rebuilt_objects.append(rebuilt)

        parsed["ingredient_objects"] = rebuilt_objects

        primary_fields = _collect_primary_fields(rebuilt_objects)
        parsed["primary_ingredients"] = primary_fields["primary_ingredients"]
        parsed["primary_ingredients_normalized"] = primary_fields["primary_ingredients_normalized"]
        parsed["excluded_ingredient_objects"] = [item for item in rebuilt_objects if str(item.get("role", "")) == "excipient"]
        parsed["excluded_ingredients"] = list(
            dict.fromkeys(
                [
                    normalize_spacing(item.get("display_name", "") or item.get("raw", ""))
                    for item in parsed["excluded_ingredient_objects"]
                    if normalize_spacing(item.get("display_name", "") or item.get("raw", ""))
                ]
            )
        )
        parsed["normalized_ingredients"] = list(
            dict.fromkeys(
                [
                    normalize_spacing(item.get("normalized_for_matching", ""))
                    for item in rebuilt_objects
                    if str(item.get("role", "")) != "excipient" and normalize_spacing(item.get("normalized_for_matching", ""))
                ]
            )
        )
        parsed["raw_ingredients"] = list(
            dict.fromkeys(
                [
                    normalize_spacing(item.get("raw", ""))
                    for item in rebuilt_objects
                    if str(item.get("role", "")) != "excipient" and normalize_spacing(item.get("raw", ""))
                ]
            )
        )

        found_primary = bool(parsed.get("primary_ingredients_normalized"))
        if not found_primary and parsed.get("normalized_ingredients"):
            quality_warnings.append(
                {
                    "code": "missing_primary_ingredients",
                    "message": "primary 원료가 비어 있어 파싱 신뢰도가 낮습니다.",
                    "severity": "critical",
                }
            )
            parsed["needs_user_review"] = True

        if not parsed.get("normalized_ingredients"):
            quality_warnings.append(
                {
                    "code": "missing_functional_ingredients",
                    "message": "정규화된 기능성 원료를 추출하지 못했습니다.",
                    "severity": "critical",
                }
            )
            parsed["needs_user_review"] = True

        normalized_count = len(parsed.get("normalized_ingredients", []) or [])
        unknown_ratio = float(unknown_count / max(1, len(rebuilt_objects))) if rebuilt_objects else 1.0
        base_confidence = float(parsed.get("confidence", 0.0) or 0.0)
        critical_count = sum(1 for item in quality_warnings if str(item.get("severity", "") or "") == "critical")
        warning_count = sum(
            1
            for item in quality_warnings
            if str(item.get("severity", "") or "") == "warning"
            and str(item.get("code", "") or "") not in {"allergen_notice"}
        )
        notice_count = sum(
            1
            for item in quality_warnings
            if str(item.get("severity", "") or "") == "notice"
            and str(item.get("code", "") or "") not in {"allergen_notice"}
        )
        normalization_coverage = float(normalized_count / max(1, len(rebuilt_objects))) if rebuilt_objects else 0.0
        evidence_count = sum(
            1
            for item in rebuilt_objects
            if normalize_lookup_key(str(item.get("evidence_text", "") or "")) in source_lookup and normalize_lookup_key(str(item.get("evidence_text", "") or ""))
        )
        evidence_coverage = float(evidence_count / max(1, len(rebuilt_objects))) if rebuilt_objects else 0.0
        role_consistency_score = 1.0
        if low_confidence_primary_count:
            role_consistency_score -= 0.2
        if not found_primary and normalized_count:
            role_consistency_score -= 0.25
        role_consistency_score = max(0.0, min(1.0, role_consistency_score))
        category_confidence = 1.0 if parsed.get("primary_ingredients_normalized") else 0.55 if normalized_count else 0.0
        ocr_quality_score = 1.0 if len(source_text) >= 120 else 0.7 if len(source_text) >= 60 else 0.35

        profile_confidence = max(
            0.0,
            min(
                1.0,
                (
                    0.25 * ocr_quality_score
                    + 0.25 * normalization_coverage
                    + 0.2 * evidence_coverage
                    + 0.15 * role_consistency_score
                    + 0.15 * category_confidence
                )
                - (critical_count * 0.08)
                - (warning_count * 0.03)
                - (notice_count * 0.01)
                - (unknown_ratio * 0.15),
            ),
        )

        if unknown_ratio >= 0.5:
            quality_warnings.append(
                {
                    "code": "high_unknown_ingredient_ratio",
                    "message": "정규화가 불명확한 원료 비율이 높습니다.",
                    "severity": "warning",
                    "unknown_ratio": round(unknown_ratio, 4),
                }
            )
            corrected_fields.append("quality_grade")

        if critical_count:
            quality_grade = "D" if critical_count >= 2 or unknown_ratio >= 0.6 else "C"
        elif profile_confidence >= 0.9 and unknown_ratio < 0.15:
            quality_grade = "A"
        elif profile_confidence >= 0.7 and unknown_ratio < 0.35:
            quality_grade = "B"
        elif normalized_count >= 1:
            quality_grade = "C"
        else:
            quality_grade = "D"

        parsed["validator_result"] = {
            "warnings": _finalize_warning_list(_normalize_quality_warnings(quality_warnings)),
            "corrected_fields": list(dict.fromkeys(corrected_fields)),
            "quality_grade": quality_grade,
            "profile_confidence": round(profile_confidence, 4),
            "unknown_ratio": round(unknown_ratio, 4),
            "confidence_breakdown": {
                "ocr_quality_score": round(ocr_quality_score, 4),
                "normalization_coverage": round(normalization_coverage, 4),
                "evidence_coverage": round(evidence_coverage, 4),
                "role_consistency_score": round(role_consistency_score, 4),
                "category_confidence": round(category_confidence, 4),
                "warning_counts": {
                    "critical": critical_count,
                    "warning": warning_count,
                    "notice": notice_count,
                },
            },
        }
        parsed["quality_warnings"] = parsed["validator_result"]["warnings"]
        parsed["quality_grade"] = quality_grade
        parsed["profile_confidence"] = parsed["validator_result"]["profile_confidence"]
        parsed["parse_metadata"] = {
            "prompt_version": PARSE_PROMPT_VERSION,
            "schema_version": PARSE_SCHEMA_VERSION,
            "normalizer_version": PARSE_NORMALIZER_VERSION,
            "sectionizer_version": PARSE_SECTIONIZER_VERSION,
            "ocr_text_hash": compute_ocr_text_hash(str(parsed.get("ingredient_section_text", "") or "")),
            "parsed_signature": compute_parsed_signature(parsed),
        }
        return parsed
    except Exception as exc:  # noqa: BLE001
        quality_warnings = list(parsed.get("quality_warnings", []))
        quality_warnings.append(
            {
                "code": "validator_error",
                "message": f"validator 보정 중 오류가 발생했습니다: {exc}",
                "severity": "warning",
            }
        )
        parsed["validator_result"] = {
            "warnings": _finalize_warning_list(_normalize_quality_warnings(quality_warnings)),
            "corrected_fields": [],
            "quality_grade": str(parsed.get("quality_grade", "") or "C"),
            "profile_confidence": float(parsed.get("profile_confidence", parsed.get("confidence", 0.0)) or 0.0),
            "unknown_ratio": 0.0,
        }
        parsed["quality_warnings"] = parsed["validator_result"]["warnings"]
        parsed["quality_grade"] = parsed["validator_result"]["quality_grade"]
        parsed["profile_confidence"] = parsed["validator_result"]["profile_confidence"]
        parsed["parse_metadata"] = {
            "prompt_version": PARSE_PROMPT_VERSION,
            "schema_version": PARSE_SCHEMA_VERSION,
            "normalizer_version": PARSE_NORMALIZER_VERSION,
            "sectionizer_version": PARSE_SECTIONIZER_VERSION,
            "ocr_text_hash": compute_ocr_text_hash(str(parsed.get("ingredient_section_text", "") or "")),
            "parsed_signature": compute_parsed_signature(parsed),
        }
        return parsed


def _build_ingredient_objects(raw_ingredients: List[str]) -> List[dict]:
    objects: List[dict] = []
    for raw in raw_ingredients:
        normalized = canonicalize_ingredient_for_matching(raw)
        role = classify_ingredient_role(raw)
        display_name = _build_display_name(raw, normalized)
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


def _collect_primary_fields(ingredient_objects: List[dict]) -> Dict[str, List[str]]:
    primary_objects = [item for item in ingredient_objects if item.get("role") == "primary"]
    if not primary_objects:
        primary_objects = [item for item in ingredient_objects if item.get("role") in {"secondary", "support"}][:3]

    primary_ingredients = list(
        dict.fromkeys(
            [
                normalize_spacing(item.get("display_name", ""))
                for item in primary_objects
                if normalize_spacing(item.get("display_name", ""))
            ]
        )
    )
    primary_ingredients_normalized = list(
        dict.fromkeys(
            [
                normalize_spacing(item.get("normalized_for_matching", ""))
                for item in primary_objects
                if normalize_spacing(item.get("normalized_for_matching", ""))
            ]
        )
    )
    return {
        "primary_ingredients": primary_ingredients,
        "primary_ingredients_normalized": primary_ingredients_normalized,
    }


def _finalize_warning_list(warnings: List[dict]) -> List[dict]:
    deduped: List[dict] = []
    seen = set()
    for item in warnings:
        code = str(item.get("code", "") or "")
        message = str(item.get("message", "") or "")
        ingredients = tuple(item.get("ingredients", []) or [])
        key = (code, message, ingredients)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def compute_ocr_text_hash(raw_text: str) -> str:
    normalized = normalize_spacing(raw_text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def compute_parsed_signature(parsed: Dict[str, Any]) -> str:
    normalized_objects = []
    for item in parsed.get("ingredient_objects", []) or []:
        normalized_objects.append(
            {
                "normalized_for_matching": normalize_spacing(item.get("normalized_for_matching", "")),
                "display_name": normalize_spacing(item.get("display_name", "")),
                "role": normalize_spacing(item.get("role", "")),
            }
        )
    payload = {
        "product_name_candidate": normalize_spacing(parsed.get("product_name_candidate", "")),
        "normalized_ingredients": sorted(
            normalize_spacing(item)
            for item in parsed.get("normalized_ingredients", []) or []
            if normalize_spacing(item)
        ),
        "primary_ingredients_normalized": sorted(
            normalize_spacing(item)
            for item in parsed.get("primary_ingredients_normalized", []) or []
            if normalize_spacing(item)
        ),
        "ingredient_objects": sorted(
            normalized_objects,
            key=lambda row: (row["normalized_for_matching"], row["role"], row["display_name"]),
        ),
        "prompt_version": PARSE_PROMPT_VERSION,
        "schema_version": PARSE_SCHEMA_VERSION,
        "normalizer_version": PARSE_NORMALIZER_VERSION,
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _ensure_llm_parse_cache_table(sqlite_path: Path) -> None:
    with sqlite_connection(sqlite_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {LLM_PARSE_CACHE_TABLE_NAME} (
                cache_key TEXT PRIMARY KEY,
                ocr_text_hash TEXT,
                cleaned_text TEXT,
                prompt_version TEXT,
                schema_version TEXT,
                normalizer_version TEXT,
                parsed_signature TEXT,
                response_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{LLM_PARSE_CACHE_TABLE_NAME}_ocr_text_hash ON {LLM_PARSE_CACHE_TABLE_NAME}(ocr_text_hash)"
        )
        conn.commit()


def _build_parse_cache_key(raw_text: str) -> str:
    source = {
        "ocr_text_hash": compute_ocr_text_hash(raw_text),
        "prompt_version": PARSE_PROMPT_VERSION,
        "schema_version": PARSE_SCHEMA_VERSION,
        "normalizer_version": PARSE_NORMALIZER_VERSION,
    }
    return hashlib.sha1(json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _load_parse_cache(sqlite_path: Path, raw_text: str) -> Dict[str, Any]:
    cache_key = _build_parse_cache_key(raw_text)
    _ensure_llm_parse_cache_table(sqlite_path)
    with sqlite_connection(sqlite_path) as conn:
        conn.row_factory = None
        row = conn.execute(
            f"""
            SELECT response_json
            FROM {LLM_PARSE_CACHE_TABLE_NAME}
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    if not row or not row[0]:
        return {}
    try:
        value = json.loads(str(row[0]))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _save_parse_cache(sqlite_path: Path, raw_text: str, parsed: Dict[str, Any]) -> None:
    cache_key = _build_parse_cache_key(raw_text)
    _ensure_llm_parse_cache_table(sqlite_path)
    payload_json = json.dumps(parsed, ensure_ascii=False)
    with sqlite_connection(sqlite_path) as conn:
        conn.execute(
            f"""
            INSERT INTO {LLM_PARSE_CACHE_TABLE_NAME} (
                cache_key, ocr_text_hash, cleaned_text, prompt_version, schema_version,
                normalizer_version, parsed_signature, response_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(cache_key) DO UPDATE SET
                ocr_text_hash=excluded.ocr_text_hash,
                cleaned_text=excluded.cleaned_text,
                prompt_version=excluded.prompt_version,
                schema_version=excluded.schema_version,
                normalizer_version=excluded.normalizer_version,
                parsed_signature=excluded.parsed_signature,
                response_json=excluded.response_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                cache_key,
                compute_ocr_text_hash(raw_text),
                normalize_spacing(raw_text),
                PARSE_PROMPT_VERSION,
                PARSE_SCHEMA_VERSION,
                PARSE_NORMALIZER_VERSION,
                compute_parsed_signature(parsed),
                payload_json,
            ),
        )
        conn.commit()


def rule_based_extract_ingredient_section(raw_text: str, section_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    source_text = str(raw_text or "")
    section_context = section_context or _build_sectionized_context(source_text)
    sections = dict(section_context.get("sections", {}) or {})
    lines = [normalize_spacing(line) for line in source_text.splitlines() if normalize_spacing(line)]

    product_name_source = "\n".join(filter(None, [str(sections.get("product_name_area", "") or ""), source_text]))
    ingredient_section_text = _derive_ingredient_section_text(source_text, sections)
    functional_info_text = normalize_spacing(str(sections.get("functional_info_area", "") or ""))
    daily_intake_source = "\n".join(filter(None, [str(sections.get("intake_area", "") or ""), source_text]))

    product_name_candidate = _extract_first_match(product_name_source, PRODUCT_NAME_PATTERNS)
    if not product_name_candidate:
        product_name_lines = [normalize_spacing(line) for line in str(sections.get("product_name_area", "") or "").splitlines() if normalize_spacing(line)]
        product_name_candidate = product_name_lines[0] if product_name_lines else ""
    daily_intake_text = _extract_first_match(daily_intake_source, DAILY_INTAKE_PATTERNS)
    if not daily_intake_text:
        daily_intake_text = normalize_spacing(str(sections.get("intake_area", "") or ""))

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
    excluded_ingredient_objects = [item for item in ingredient_objects if item["role"] == "excipient"]
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
                "severity": "warning",
            }
        )

    primary_fields = _collect_primary_fields(ingredient_objects)
    return {
        "product_name_candidate": product_name_candidate,
        "ingredient_section_text": ingredient_section_text,
        "functional_ingredient_candidates": list(dict.fromkeys(functional_candidates)),
        "raw_ingredients": raw_ingredients,
        "normalized_ingredients": list(dict.fromkeys(normalized_ingredients)),
        "ingredient_objects": ingredient_objects,
        "primary_ingredients": primary_fields["primary_ingredients"],
        "primary_ingredients_normalized": primary_fields["primary_ingredients_normalized"],
        "excluded_ingredients": [item["display_name"] for item in excluded_ingredient_objects],
        "excluded_ingredient_objects": excluded_ingredient_objects,
        "quality_warnings": quality_warnings,
        "nutrition_or_active_components": [],
        "daily_intake_text": daily_intake_text,
        "confidence": 0.62 if normalized_ingredients else 0.2,
        "needs_user_review": not bool(ingredient_section_text and normalized_ingredients),
        "ocr_sections": sections,
        "ocr_section_line_counts": dict(section_context.get("section_line_counts", {}) or {}),
        "sectionized_text_for_llm": str(section_context.get("llm_text", "") or ""),
    }


def normalize_ingredients_with_llm(raw_text: str, ocr_lines: List[str], section_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    section_context = section_context or _build_sectionized_context(raw_text)
    sections = dict(section_context.get("sections", {}) or {})
    llm_text = str(section_context.get("llm_text", "") or raw_text)
    llm_lines = list(section_context.get("llm_lines", []) or ocr_lines)
    llm_ingredient_area = _derive_ingredient_section_text(raw_text, sections)
    llm_sections = dict(sections)
    if llm_ingredient_area:
        llm_sections["ingredient_area"] = llm_ingredient_area
    sectioned_payload = "\n\n".join(
        f"[{key.upper()}]\n{value}"
        for key, value in llm_sections.items()
        if str(value or "").strip() and key in {"product_name_area", "ingredient_area", "functional_info_area", "intake_area"}
    ).strip() or llm_text
    system_prompt = (
        "너는 건강기능식품 라벨 OCR 텍스트에서 원재료명과 기능성원료를 추출하는 파서다.\n"
        "OCR 텍스트에는 줄바꿈 오류, 띄어쓰기 오류, 오타가 있을 수 있다.\n"
        "제품명과 원재료명이 충돌할 수 있으므로 둘을 분리해서 판단하라.\n"
        "광고 문구, 다른 제품명, 배너 문구는 원재료명으로 넣지 마라.\n"
        "부형제, 캡슐기제, HPMC 등은 excipient로 분리하라.\n"
        "난각막, 난각막분말, 난각막가수분해물, NEM, DEMR은 부형제가 아니라 관절/연골 후보로 유지하라.\n"
        "계란 함유, 알레르기 유발 물질 안내는 원료가 아니라 quality_warnings로 넣어라.\n"
        "warning_area, company_area, unknown_area 성격의 문구는 원료로 추출하지 마라.\n"
        "긴 원료명은 display_name과 normalized_for_matching을 분리하라.\n"
        "예: DW2009 프로바이오틱스 복합물 -> normalized_for_matching=프로바이오틱스.\n"
        "반드시 JSON만 출력하라."
    )
    user_prompt = (
        "아래 OCR 텍스트에서 건강기능식품 추천에 사용할 원료 정보를 추출하라.\n"
        "우선 SECTIONED_OCR_TEXT를 기준으로 판단하고, 필요할 때만 RAW_OCR_TEXT를 참고하라.\n\n"
        f"SECTIONED_OCR_TEXT:\n{sectioned_payload}\n\n"
        f"RAW_OCR_TEXT:\n{llm_text}\n\n"
        "OCR_LINES:\n"
        + "\n".join(llm_lines[:80])
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
        '  "primary_ingredients": [],\n'
        '  "primary_ingredients_normalized": [],\n'
        '  "excluded_ingredients": [],\n'
        '  "excluded_ingredient_objects": [],\n'
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


def parse_ingredients_from_ocr_text(raw_text: str, sqlite_path: Path | None = None) -> Dict[str, Any]:
    runtime_sqlite_path = Path(sqlite_path) if sqlite_path else default_sqlite_path()
    section_context = _build_sectionized_context(str(raw_text or ""))
    cached = _load_parse_cache(runtime_sqlite_path, str(raw_text or ""))
    if cached:
        cached["raw_text"] = str(raw_text or "")
        cached["source_text"] = str(raw_text or "")
        cached["ocr_sections"] = dict(section_context.get("sections", {}) or {})
        cached["ocr_section_line_counts"] = dict(section_context.get("section_line_counts", {}) or {})
        cached["sectionized_text_for_llm"] = str(section_context.get("llm_text", "") or "")
        cached = _validate_and_repair_parsed_result(cached)
        cached["parse_metadata"] = {
            **dict(cached.get("parse_metadata", {}) or {}),
            "cache_hit": True,
            "prompt_version": PARSE_PROMPT_VERSION,
            "schema_version": PARSE_SCHEMA_VERSION,
            "normalizer_version": PARSE_NORMALIZER_VERSION,
            "sectionizer_version": PARSE_SECTIONIZER_VERSION,
            "ocr_text_hash": compute_ocr_text_hash(str(raw_text or "")),
            "parsed_signature": compute_parsed_signature(cached),
        }
        return cached

    fallback = rule_based_extract_ingredient_section(raw_text, section_context)
    lines = list(section_context.get("llm_lines", []) or [normalize_spacing(line) for line in str(raw_text or "").splitlines() if normalize_spacing(line)])

    try:
        llm_result = normalize_ingredients_with_llm(str(raw_text or ""), lines, section_context)
    except Exception as exc:  # noqa: BLE001
        fallback["quality_warnings"] = list(fallback.get("quality_warnings", [])) + [
            {
                "code": "llm_fallback",
                "message": f"LLM 파싱 실패로 rule-based fallback을 사용했습니다: {exc}",
                "severity": "warning",
            }
        ]
        llm_result = {}

    merged = {
        "product_name_candidate": str(llm_result.get("product_name_candidate") or fallback.get("product_name_candidate") or ""),
        "raw_text": str(raw_text or ""),
        "source_text": str(raw_text or ""),
        "ingredient_section_text": str(llm_result.get("ingredient_section_text") or fallback.get("ingredient_section_text") or ""),
        "functional_ingredient_candidates": list(llm_result.get("functional_ingredient_candidates") or fallback.get("functional_ingredient_candidates") or []),
        "raw_ingredients": list(llm_result.get("raw_ingredients") or fallback.get("raw_ingredients") or []),
        "normalized_ingredients": list(llm_result.get("normalized_ingredients") or fallback.get("normalized_ingredients") or []),
        "ingredient_objects": list(llm_result.get("ingredient_objects") or fallback.get("ingredient_objects") or []),
        "primary_ingredients": list(llm_result.get("primary_ingredients") or fallback.get("primary_ingredients") or []),
        "primary_ingredients_normalized": list(
            llm_result.get("primary_ingredients_normalized") or fallback.get("primary_ingredients_normalized") or []
        ),
        "excluded_ingredients": list(llm_result.get("excluded_ingredients") or fallback.get("excluded_ingredients") or []),
        "excluded_ingredient_objects": list(
            llm_result.get("excluded_ingredient_objects") or fallback.get("excluded_ingredient_objects") or []
        ),
        "quality_warnings": _normalize_quality_warnings(llm_result.get("quality_warnings") or fallback.get("quality_warnings") or []),
        "nutrition_or_active_components": list(llm_result.get("nutrition_or_active_components") or []),
        "daily_intake_text": str(llm_result.get("daily_intake_text") or fallback.get("daily_intake_text") or ""),
        "confidence": float(llm_result.get("confidence") or fallback.get("confidence") or 0.0),
        "needs_user_review": bool(llm_result.get("needs_user_review", fallback.get("needs_user_review", False))),
        "ocr_sections": dict(llm_result.get("ocr_sections") or fallback.get("ocr_sections") or {}),
        "ocr_section_line_counts": dict(llm_result.get("ocr_section_line_counts") or fallback.get("ocr_section_line_counts") or {}),
        "sectionized_text_for_llm": str(llm_result.get("sectionized_text_for_llm") or fallback.get("sectionized_text_for_llm") or section_context.get("llm_text") or ""),
    }

    if not merged["ingredient_objects"]:
        merged["ingredient_objects"] = _build_ingredient_objects(merged["raw_ingredients"])

    normalized_ingredients: List[str] = []
    filtered_raw_ingredients: List[str] = []
    excluded_ingredients: List[str] = []
    excluded_ingredient_objects: List[dict] = []
    quality_warnings = list(merged["quality_warnings"])
    rebuilt_objects: List[dict] = []

    for item in merged["ingredient_objects"]:
        raw = normalize_spacing(item.get("raw", ""))
        if not raw:
            continue
        if _looks_like_allergen_notice(raw):
            quality_warnings.append(classify_warning_message(raw, "allergen_notice"))
            continue
        if _looks_like_allergen_warning(raw):
            quality_warnings.append(classify_warning_message(raw, "allergen_notice"))
            continue

        normalized_for_matching = canonicalize_ingredient_for_matching(item.get("normalized_for_matching") or raw)
        display_name = normalize_spacing(item.get("display_name") or _build_display_name(raw, normalized_for_matching))
        calculated_role = classify_ingredient_role(raw)
        role = str(item.get("role") or calculated_role)
        if role in {"excluded", "allergen", "notice"}:
            quality_warnings.append(classify_warning_message(raw, "allergen_notice"))
            continue
        if role == "unknown" or (role == "excipient" and calculated_role != "excipient"):
            role = calculated_role

        rebuilt = {
            "raw": raw,
            "normalized_for_matching": normalized_for_matching,
            "display_name": display_name,
            "role": role,
            "category_hint": str(item.get("category_hint") or _guess_category_hint(normalized_for_matching)),
        }
        rebuilt_objects.append(rebuilt)
        filtered_raw_ingredients.append(raw)

        if role == "excipient":
            excluded_ingredients.append(display_name)
            excluded_ingredient_objects.append(rebuilt)
            continue

        if normalized_for_matching:
            normalized_ingredients.append(normalized_for_matching)

    merged["ingredient_objects"] = rebuilt_objects
    merged["excluded_ingredient_objects"] = excluded_ingredient_objects
    merged["excluded_ingredients"] = list(dict.fromkeys([item for item in excluded_ingredients if item]))
    merged["normalized_ingredients"] = list(dict.fromkeys([item for item in normalized_ingredients if item]))
    merged["raw_ingredients"] = list(dict.fromkeys(filtered_raw_ingredients))

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

    primary_fields = _collect_primary_fields(merged["ingredient_objects"])
    merged["primary_ingredients"] = primary_fields["primary_ingredients"]
    merged["primary_ingredients_normalized"] = primary_fields["primary_ingredients_normalized"]

    if len(merged["normalized_ingredients"]) >= 20:
        merged["needs_user_review"] = True
        quality_warnings.append(
            {
                "code": "too_many_normalized_ingredients",
                "message": "정규화된 원료 수가 많아 OCR 결과 검토가 필요합니다.",
                "severity": "notice",
            }
        )
    if not merged["ingredient_section_text"]:
        merged["needs_user_review"] = True
        quality_warnings.append(
            {
                "code": "ingredient_section_unclear",
                "message": "원재료명 영역이 명확하지 않습니다.",
                "severity": "critical",
            }
        )

    merged["quality_warnings"] = _finalize_warning_list(quality_warnings)
    merged = _validate_and_repair_parsed_result(merged)
    merged["parse_metadata"] = {
        **dict(merged.get("parse_metadata", {}) or {}),
        "cache_hit": False,
        "sectionizer_version": PARSE_SECTIONIZER_VERSION,
        "ocr_text_hash": compute_ocr_text_hash(str(raw_text or "")),
        "parsed_signature": compute_parsed_signature(merged),
    }
    _save_parse_cache(runtime_sqlite_path, str(raw_text or ""), merged)
    return merged
