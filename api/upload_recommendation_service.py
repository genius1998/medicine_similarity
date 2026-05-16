from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from api.config import get_settings
from api.db import sqlite_connection
from api.ingredient_parse_service import (
    canonicalize_ingredient_for_matching,
    classify_ingredient_role,
    is_excipient,
    parse_ingredients_from_ocr_text,
    split_ingredients,
)
from api.ocr_service import extract_text_from_image, save_temp_upload
from api.recommendation_service import RecommendationService
from scripts.enhance_similarity_with_explanation import (
    build_candidate_pool,
    build_explanation_json,
    calculate_core_match_score,
    calculate_function_similarity,
    calculate_weighted_jaccard_with_idf,
    classify_substitutability,
    compare_product_profiles,
    generate_caution,
    generate_recommendation_reason,
    safe_json_loads,
)


CATEGORY_HINT_RULES = {
    "면역": {
        "product_keywords": ["홍삼", "고려홍삼", "6년근", "홍삼정", "진세노사이드", "홍삼스틱", "정관장"],
        "ingredient_keywords": ["홍삼", "홍삼농축액", "진세노사이드", "홍삼근", "홍삼분말"],
    },
    "눈 건강": {
        "product_keywords": ["루테인", "지아잔틴", "눈", "아이", "마리골드", "빌베리", "아스타잔틴", "베타카로틴"],
        "ingredient_keywords": ["루테인", "지아잔틴", "마리골드꽃추출물", "베타카로틴", "비타민 A", "빌베리추출물", "블루베리", "헤마토코쿠스추출물"],
    },
    "관절/연골": {
        "product_keywords": ["관절", "연골", "콘드로이친", "뮤코다당", "뮤코다당단백", "철갑상어", "철갑상어연골", "MSM", "보스웰리아", "NEM", "난각막", "강황"],
        "ingredient_keywords": ["콘드로이친", "뮤코다당단백", "철갑상어연골", "MSM", "엠에스엠", "보스웰리아", "난각막", "난각막분말", "난각막가수분해물", "강황추출물", "초록입홍합"],
    },
    "장 건강": {
        "product_keywords": ["유산균", "프로바이오틱스", "락토핏", "장", "대장", "프리바이오틱스", "올리고당"],
        "ingredient_keywords": ["프로바이오틱스", "유산균", "갈락토올리고당", "프락토올리고당", "난소화성말토덱스트린", "이눌린"],
    },
    "혈당": {
        "product_keywords": ["혈당", "당당자유", "당케어", "고투플러스"],
        "ingredient_keywords": ["바나바", "바나바잎", "바나바잎추출물", "코로솔산", "난소화성말토덱스트린", "구아바잎추출물", "달맞이꽃종자추출물", "구아검", "이눌린", "식이섬유"],
    },
    "체지방": {
        "product_keywords": ["체지방", "다이어트", "슬림", "컷", "컷팅"],
        "ingredient_keywords": ["키토산", "키토올리고당", "가르시니아", "hca", "공액리놀레산", "cla", "녹차추출물", "카테킨", "콜레우스포스콜리", "잔티젠"],
    },
    "피부 건강": {
        "product_keywords": ["콜라겐", "피부", "레티놀", "히알루론", "세라마이드", "비오틴"],
        "ingredient_keywords": ["콜라겐", "저분자콜라겐펩타이드", "콜라겐펩타이드", "비오틴", "비타민 A", "히알루론산", "세라마이드"],
    },
    "기억력/인지력": {
        "product_keywords": ["인지", "기억", "포스파티딜세린", "브레인", "두뇌"],
        "ingredient_keywords": ["포스파티딜세린", "은행잎", "테아닌", "DHA"],
    },
    "뼈 건강": {
        "product_keywords": ["뼈", "칼슘", "칼마디", "비타민D", "K2"],
        "ingredient_keywords": ["칼슘", "비타민 D", "비타민 K", "마그네슘"],
    },
    "남성 건강": {
        "product_keywords": ["전립선", "남성", "쏘팔메토", "옥타코사놀", "활력"],
        "ingredient_keywords": ["쏘팔메토", "쏘팔메토열매추출물", "로르산", "옥타코사놀", "아연", "녹용", "마카", "아르기닌"],
    },
    "간 건강": {
        "product_keywords": ["밀크씨슬", "밀크시슬", "실리마린", "간", "liver", "milk thistle"],
        "ingredient_keywords": ["밀크씨슬", "밀크시슬", "밀크씨슬추출물", "실리마린", "silymarin", "milk thistle"],
    },
}

EXCLUDED_PRIMARY_KEYWORDS = [
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
]

NON_EXCLUDED_PRIMARY_KEYWORDS = [
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
]

ROLE_WEIGHT = {
    "primary": 1.0,
    "secondary": 0.72,
    "support": 0.45,
}

LOW_OCR_CONFIDENCE_THRESHOLD = 0.6
LOW_PARSE_CONFIDENCE_THRESHOLD = 0.65
LOW_TOP_SIMILARITY_THRESHOLD = 0.25
MIN_SUFFICIENT_OCR_TEXT_LENGTH = 80
TOO_MANY_INGREDIENTS_THRESHOLD = 20

STRONG_JOINT_PRODUCT_KEYWORDS = [
    "관절",
    "연골",
    "콘드로이친",
    "뮤코다당",
    "뮤코다당단백",
    "철갑상어",
    "철갑상어연골",
    "MSM",
    "보스웰리아",
    "NEM",
    "난각막",
]

STRONG_JOINT_INGREDIENT_KEYWORDS = [
    "콘드로이친",
    "뮤코다당단백",
    "철갑상어연골",
    "MSM",
    "엠에스엠",
    "글루코사민",
    "NAG",
    "보스웰리아",
    "난각막",
    "난각막분말",
    "난각막가수분해물",
    "초록입홍합",
    "UC-II",
    "비변성콜라겐",
]

NUTRITION_SUPPLEMENT_PRODUCT_KEYWORDS = [
    "멀티",
    "멀티밸런스",
    "멀티비타민",
    "비타민",
    "미네랄",
    "요오드",
]

NUTRITION_SUPPLEMENT_INGREDIENT_KEYWORDS = [
    "비타민 A",
    "비타민 B1",
    "비타민 B2",
    "비타민 B6",
    "비타민 B12",
    "비타민 C",
    "비타민 D",
    "비타민 E",
    "비타민 K",
    "엽산",
    "나이아신",
    "비오틴",
    "판토텐산",
    "셀레늄",
    "아연",
    "칼슘",
    "마그네슘",
    "철분",
    "요오드",
    "망간",
    "크롬",
    "구리",
    "몰리브덴",
    "dl-a-토코페롤",
]

CRITICAL_WARNING_CODES = {
    "critical_warning",
    "ingredient_section_unclear",
    "ocr_error",
    "category_conflict_product_name_vs_ingredients",
    "generic_category_with_named_product",
    "low_parse_confidence",
    "low_ocr_confidence",
    "low_ocr_text_quality",
    "excipient_in_primary",
    "mixed_primary_categories",
    "low_top_similarity_score",
    "missing_recommendations",
    "top_reason_category_mismatch",
    "too_many_normalized_ingredients",
}

NOTICE_WARNING_CODES = {
    "notice_warning",
    "allergen_notice",
}

INFO_WARNING_CODES = {
    "info_warning",
    "ocr_confidence_unavailable",
}


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def contains_keyword(text: str, keywords: List[str]) -> bool:
    normalized = normalize_lookup_key(text)
    return any(normalize_lookup_key(keyword) in normalized for keyword in keywords)


def is_excluded_primary(name: str) -> bool:
    normalized = normalize_lookup_key(name)
    if any(normalize_lookup_key(keyword) in normalized for keyword in NON_EXCLUDED_PRIMARY_KEYWORDS):
        return False
    return any(normalize_lookup_key(keyword) in normalized for keyword in EXCLUDED_PRIMARY_KEYWORDS)


def has_sufficient_ocr_text(raw_text: str, ingredient_section_text: str = "") -> bool:
    collapsed_raw = re.sub(r"\s+", "", str(raw_text or ""))
    collapsed_section = re.sub(r"\s+", "", str(ingredient_section_text or ""))
    if len(collapsed_raw) >= MIN_SUFFICIENT_OCR_TEXT_LENGTH:
        return True
    return len(collapsed_section) >= 20


def append_warning(warnings: List[dict], code: str, message: str, severity: str = "warning", **extra) -> None:
    item = {"code": code, "message": message, "severity": severity}
    item.update({key: value for key, value in extra.items() if value not in (None, "", [], {})})
    warnings.append(item)


def dedupe_warnings(warnings: List[dict]) -> List[dict]:
    seen = set()
    deduped: List[dict] = []
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


def normalize_warning_severity(warning: dict) -> dict:
    item = dict(warning)
    code = str(item.get("code", "") or "")
    severity = str(item.get("severity", "") or "")
    message = str(item.get("message", "") or "")
    llm_notice_terms = [
        "알레르기",
        "알러지",
        "알류",
        "달걀",
        "계란",
        "우유",
        "대두",
        "밀",
        "땅콩",
        "메밀",
        "토마토",
        "복숭아",
        "아황산",
        "고등어",
        "게",
        "새우",
        "돼지고기",
        "쇠고기",
        "닭고기",
        "오징어",
        "조개류",
        "홍합",
        "잣",
        "가금류",
        "함유",
        "제조시설",
        "주의",
        "보관",
        "섭취",
        "임산부",
        "수유부",
        "어린이",
        "젤라틴",
        "우피",
        "색소",
        "먹물색소",
    ]
    if code == "too_many_normalized_ingredients" and severity in {"notice", "critical"}:
        item["severity"] = severity
    elif code == "llm_warning" and (
        any(term in message for term in llm_notice_terms)
        or "cross-contamination" in message.lower()
        or "shared manufacturing" in message.lower()
        or "allergic reaction" in message.lower()
        or "discontinue use" in message.lower()
        or "consult a professional" in message.lower()
    ):
        item["severity"] = "notice"
    elif code in INFO_WARNING_CODES:
        item["severity"] = "info"
    elif code in NOTICE_WARNING_CODES:
        item["severity"] = "notice"
    elif code in CRITICAL_WARNING_CODES:
        item["severity"] = "critical"
    elif severity in {"info", "notice", "critical"}:
        item["severity"] = severity
    elif "알레르기" in message or "알러지" in message or "함유" in message or "제조시설" in message or "주의" in message or "보관" in message:
        item["severity"] = "notice"
    else:
        item["severity"] = "critical"
    return item


def split_warning_groups(warnings: List[dict]) -> dict:
    normalized = [normalize_warning_severity(item) for item in dedupe_warnings(warnings)]
    return {
        "quality_warnings": [item for item in normalized if item.get("severity") == "critical"],
        "critical_warnings": [item for item in normalized if item.get("severity") == "critical"],
        "notices": [item for item in normalized if item.get("severity") == "notice"],
        "info_warnings": [item for item in normalized if item.get("severity") == "info"],
    }


def contains_any_keyword(values: List[str], keywords: List[str]) -> bool:
    for value in values:
        if contains_keyword(value, keywords):
            return True
    return False


def count_keyword_matches(values: List[str], keywords: List[str]) -> int:
    matched = set()
    for value in values:
        for keyword in keywords:
            if contains_keyword(value, [keyword]):
                matched.add(keyword)
    return len(matched)


class UploadRecommendationService:
    def __init__(self, recommendation_service: RecommendationService) -> None:
        self.recommendation_service = recommendation_service
        self._category_map: Optional[Dict[str, dict]] = None

    def _ensure_loaded(self) -> None:
        self.recommendation_service.ensure_loaded()
        if self._category_map is not None:
            return
        runtime = self.recommendation_service.runtime
        category_map: Dict[str, dict] = {}
        with sqlite_connection(runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT functional_ingredient_name, category_main, category_sub, claim_text, source, confidence, notes, categories_json
                FROM functional_category_map
                """
            ).fetchall()
        for row in rows:
            item = dict(row)
            item["categories"] = safe_json_loads(item.get("categories_json", "[]"), [])
            category_map[normalize_lookup_key(item["functional_ingredient_name"])] = item
        self._category_map = category_map

    def _lookup_category_row(self, functional_name: str) -> Optional[dict]:
        self._ensure_loaded()
        return self._category_map.get(normalize_lookup_key(functional_name))

    def infer_category_from_product_name(self, product_name: str) -> dict:
        name = str(product_name or "").strip()
        for category, rule in CATEGORY_HINT_RULES.items():
            if contains_keyword(name, rule["product_keywords"]):
                return {"category_main": category, "confidence": 0.95, "matched_keywords": rule["product_keywords"]}
        return {"category_main": "", "confidence": 0.0, "matched_keywords": []}

    def _collect_category_diversity(self, estimated_profile: dict) -> int:
        categories = {
            str(item.get("category_main", "") or "")
            for item in estimated_profile.get("ingredient_scores", [])
            if str(item.get("category_main", "") or "") not in {"", "기타", "영양보충"}
        }
        return len(categories)

    def _collect_primary_category_diversity(self, estimated_profile: dict) -> int:
        primary_names = {
            str(name or "")
            for name in estimated_profile.get("primary_ingredients", [])
            if str(name or "")
        }
        categories = {
            str(item.get("category_main", "") or "")
            for item in estimated_profile.get("ingredient_scores", [])
            if str(item.get("ingredient", "") or "") in primary_names
            and str(item.get("category_main", "") or "") not in {"", "기타", "영양보충"}
        }
        return len(categories)

    def _has_stable_primary_focus(self, estimated_profile: dict) -> bool:
        primary_ingredients = [str(name or "") for name in estimated_profile.get("primary_ingredients", []) if str(name or "")]
        if not primary_ingredients:
            return False
        main_category = str(estimated_profile.get("product_main_category", "") or "")
        if main_category == "장 건강":
            normalized = [normalize_lookup_key(name) for name in primary_ingredients]
            if any("프로바이오틱스" in name or "유산균" in name for name in normalized):
                return True
        return len(primary_ingredients) <= 3

    def _is_liver_health_candidate(self, product_name_candidate: str, normalized_ingredients: List[str]) -> bool:
        keywords = CATEGORY_HINT_RULES["간 건강"]["product_keywords"]
        ingredient_keywords = CATEGORY_HINT_RULES["간 건강"]["ingredient_keywords"]
        return contains_keyword(product_name_candidate, keywords) or contains_any_keyword(normalized_ingredients, ingredient_keywords)

    def _is_eye_health_candidate(self, product_name_candidate: str, normalized_ingredients: List[str]) -> bool:
        product_keywords = CATEGORY_HINT_RULES["눈 건강"]["product_keywords"]
        strong_ingredient_keywords = ["루테인", "지아잔틴", "마리골드꽃추출물", "베타카로틴", "빌베리추출물", "블루베리", "헤마토코쿠스추출물"]
        has_product_hint = contains_keyword(product_name_candidate, product_keywords)
        has_strong_ingredient = contains_any_keyword(normalized_ingredients, strong_ingredient_keywords)
        return has_product_hint and has_strong_ingredient

    def _is_blood_sugar_candidate(self, product_name_candidate: str, normalized_ingredients: List[str]) -> bool:
        product_keywords = CATEGORY_HINT_RULES["혈당"]["product_keywords"]
        strong_ingredient_keywords = ["바나바", "바나바잎", "바나바잎추출물", "코로솔산", "난소화성말토덱스트린", "구아바잎추출물", "달맞이꽃종자추출물"]
        body_keywords = ["키토산", "키토올리고당"]
        matches = [value for value in normalized_ingredients if contains_any_keyword([value], strong_ingredient_keywords)]
        has_glucose_keyword = bool(matches)
        has_body_keyword = contains_any_keyword(normalized_ingredients, body_keywords)
        explicit_blood_sugar_marker = any(
            contains_any_keyword([value], ["바나바", "바나바잎", "바나바잎추출물", "코로솔산", "구아바잎추출물", "달맞이꽃종자추출물"])
            for value in normalized_ingredients
        )
        return (
            contains_keyword(product_name_candidate, product_keywords)
            or explicit_blood_sugar_marker
            or (has_body_keyword and len(matches) >= 2)
        )

    def _is_body_fat_candidate(self, product_name_candidate: str, normalized_ingredients: List[str]) -> bool:
        product_keywords = CATEGORY_HINT_RULES["체지방"]["product_keywords"]
        ingredient_keywords = CATEGORY_HINT_RULES["체지방"]["ingredient_keywords"]
        return contains_keyword(product_name_candidate, product_keywords) or contains_any_keyword(normalized_ingredients, ingredient_keywords)

    def _is_joint_candidate(self, product_name_candidate: str, normalized_ingredients: List[str]) -> bool:
        return contains_keyword(product_name_candidate, STRONG_JOINT_PRODUCT_KEYWORDS) or contains_any_keyword(normalized_ingredients, STRONG_JOINT_INGREDIENT_KEYWORDS)

    def _is_nutrition_supplement_candidate(self, product_name_candidate: str, normalized_ingredients: List[str]) -> bool:
        nutrient_match_count = count_keyword_matches(normalized_ingredients, NUTRITION_SUPPLEMENT_INGREDIENT_KEYWORDS)
        has_product_hint = contains_keyword(product_name_candidate, NUTRITION_SUPPLEMENT_PRODUCT_KEYWORDS)
        has_strong_joint_signal = self._is_joint_candidate(product_name_candidate, normalized_ingredients)
        if has_strong_joint_signal:
            return False
        return nutrient_match_count >= 4 or (has_product_hint and nutrient_match_count >= 2)

    def _reprioritize_primary_ingredients(self, estimated_profile: dict, keywords: List[str], forced_category: str) -> dict:
        ingredient_scores = list(estimated_profile.get("ingredient_scores", []))
        preferred = [item["ingredient"] for item in ingredient_scores if contains_keyword(item.get("ingredient", ""), keywords)]
        non_preferred = [item["ingredient"] for item in ingredient_scores if item["ingredient"] not in preferred]
        if preferred:
            estimated_profile["primary_ingredients"] = list(dict.fromkeys(preferred))[:3]
            estimated_profile["secondary_ingredients"] = [item for item in list(dict.fromkeys(non_preferred)) if item not in estimated_profile["primary_ingredients"]][:8]
        estimated_profile["product_main_category"] = forced_category
        return estimated_profile

    def _apply_domain_category_overrides(self, parsed_result: dict, estimated_profile: dict) -> dict:
        product_name_candidate = str(parsed_result.get("product_name_candidate", "") or "")
        normalized_ingredients = list(parsed_result.get("normalized_ingredients", []))

        if self._is_liver_health_candidate(product_name_candidate, normalized_ingredients):
            estimated_profile = self._reprioritize_primary_ingredients(
                estimated_profile,
                CATEGORY_HINT_RULES["간 건강"]["ingredient_keywords"],
                "간 건강",
            )

        if self._is_eye_health_candidate(product_name_candidate, normalized_ingredients):
            estimated_profile = self._reprioritize_primary_ingredients(
                estimated_profile,
                CATEGORY_HINT_RULES["눈 건강"]["ingredient_keywords"],
                "눈 건강",
            )

        if self._is_joint_candidate(product_name_candidate, normalized_ingredients):
            estimated_profile = self._reprioritize_primary_ingredients(
                estimated_profile,
                CATEGORY_HINT_RULES["관절/연골"]["ingredient_keywords"],
                "관절/연골",
            )

        if not self._is_liver_health_candidate(product_name_candidate, normalized_ingredients) and self._is_blood_sugar_candidate(product_name_candidate, normalized_ingredients):
            estimated_profile = self._reprioritize_primary_ingredients(
                estimated_profile,
                CATEGORY_HINT_RULES["혈당"]["ingredient_keywords"] + ["키토산", "키토올리고당"],
                "혈당",
            )
        elif self._is_body_fat_candidate(product_name_candidate, normalized_ingredients):
            estimated_profile = self._reprioritize_primary_ingredients(
                estimated_profile,
                CATEGORY_HINT_RULES["체지방"]["ingredient_keywords"],
                "체지방",
            )

        if (
            not self._is_liver_health_candidate(product_name_candidate, normalized_ingredients)
            and not self._is_eye_health_candidate(product_name_candidate, normalized_ingredients)
            and not self._is_blood_sugar_candidate(product_name_candidate, normalized_ingredients)
            and not self._is_body_fat_candidate(product_name_candidate, normalized_ingredients)
            and self._is_nutrition_supplement_candidate(product_name_candidate, normalized_ingredients)
        ):
            estimated_profile = self._reprioritize_primary_ingredients(
                estimated_profile,
                NUTRITION_SUPPLEMENT_INGREDIENT_KEYWORDS,
                "영양보충",
            )

        return estimated_profile

    def _override_recommendation_reason(self, temp_profile: dict, row: dict) -> str:
        shared_ingredients = [str(item or "") for item in row.get("shared_ingredients", [])]
        if temp_profile.get("product_main_category") == "간 건강" and contains_any_keyword(shared_ingredients, CATEGORY_HINT_RULES["간 건강"]["ingredient_keywords"]):
            return "밀크씨슬추출물 성분이 공통으로 포함되어 간 건강 기능성 측면에서 유사합니다."
        if temp_profile.get("product_main_category") == "눈 건강" and contains_any_keyword(shared_ingredients, CATEGORY_HINT_RULES["눈 건강"]["ingredient_keywords"]):
            return "마리골드꽃추출물, 루테인/지아잔틴 계열 성분이 공통으로 포함되어 눈 건강 기능성 측면에서 유사합니다."
        if temp_profile.get("product_main_category") == "혈당" and contains_any_keyword(shared_ingredients, CATEGORY_HINT_RULES["혈당"]["ingredient_keywords"] + ["키토산"]):
            return "바나바잎추출물, 키토산, 구아바잎추출물 등 대사 관리 관련 성분이 공통으로 포함되어 혈당 기능성 측면에서 유사합니다."
        if temp_profile.get("product_main_category") == "체지방" and contains_any_keyword(shared_ingredients, CATEGORY_HINT_RULES["체지방"]["ingredient_keywords"]):
            return "키토산, 가르시니아, 녹차추출물 등 체지방 관리 관련 성분이 공통으로 포함되어 유사합니다."
        if temp_profile.get("product_main_category") == "영양보충" and contains_any_keyword(shared_ingredients, NUTRITION_SUPPLEMENT_INGREDIENT_KEYWORDS):
            return "비타민·미네랄 계열 성분이 공통으로 포함되어 영양보충 측면에서 유사합니다."
        return str(row.get("reason", "") or "")

    def _detect_category_from_ingredients(self, ingredient_names: List[str]) -> str:
        scores: Dict[str, int] = {}
        for category, rule in CATEGORY_HINT_RULES.items():
            for ingredient in ingredient_names:
                if contains_keyword(ingredient, rule["ingredient_keywords"]):
                    scores[category] = scores.get(category, 0) + 1
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[0][0] if ranked else ""

    def _find_match_in_cache(self, ingredient_name: str) -> Optional[dict]:
        runtime = self.recommendation_service.runtime
        normalized = normalize_lookup_key(ingredient_name)
        direct = self._lookup_category_row(ingredient_name)
        if direct:
            return {
                "raw_ingredient": ingredient_name,
                "functional_ingredient_name": direct["functional_ingredient_name"],
                "relation_type": "same_ingredient",
                "confidence": 0.98,
                "category_row": direct,
            }

        with sqlite_connection(runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT raw_ingredient, normalized_raw, matched_standard_name, relation_type, confidence
                FROM ingredient_match_cache
                WHERE normalized_raw = ?
                   OR raw_ingredient = ?
                   OR normalized_raw LIKE ?
                   OR raw_ingredient LIKE ?
                ORDER BY confidence DESC, cache_id ASC
                LIMIT 10
                """,
                (normalized, ingredient_name, f"%{normalized}%", f"%{ingredient_name}%"),
            ).fetchall()

        for row in rows:
            matched_standard = str(row["matched_standard_name"] or "").strip()
            matched_raw = str(row["raw_ingredient"] or "").strip()
            if matched_standard:
                category_row = self._lookup_category_row(matched_standard)
                if category_row:
                    return {
                        "raw_ingredient": ingredient_name,
                        "functional_ingredient_name": category_row["functional_ingredient_name"],
                        "relation_type": str(row["relation_type"] or "cache_match"),
                        "confidence": float(row["confidence"] or 0.0),
                        "category_row": category_row,
                    }
            if matched_raw:
                category_row = self._lookup_category_row(matched_raw)
                if category_row:
                    return {
                        "raw_ingredient": ingredient_name,
                        "functional_ingredient_name": category_row["functional_ingredient_name"],
                        "relation_type": str(row["relation_type"] or "cache_match"),
                        "confidence": float(row["confidence"] or 0.0),
                        "category_row": category_row,
                    }
        return None

    def match_raw_ingredients_to_functional_ingredients(self, ingredient_objects: List[dict]) -> List[dict]:
        results = []
        seen = set()
        for item in ingredient_objects:
            if item.get("role") == "excipient":
                continue
            normalized = str(item.get("normalized_for_matching", "") or "").strip()
            if not normalized:
                continue
            match = self._find_match_in_cache(normalized)
            if not match:
                continue
            key = (match["raw_ingredient"], match["functional_ingredient_name"])
            if key in seen:
                continue
            seen.add(key)
            match["display_name"] = str(item.get("display_name", normalized) or normalized)
            match["normalized_for_matching"] = normalized
            match["input_role"] = str(item.get("role", "") or "")
            results.append(match)
        return results

    def build_temp_product_vector_from_ingredients(self, ingredient_objects: List[dict]) -> Dict[str, float]:
        vector: Dict[str, float] = {}
        for item in self.match_raw_ingredients_to_functional_ingredients(ingredient_objects):
            functional_name = str(item["functional_ingredient_name"])
            role = str(item.get("input_role", "secondary") or "secondary")
            weight = max(0.35, min(1.0, float(item.get("confidence", 0.75)))) * ROLE_WEIGHT.get(role, 0.55)
            vector[functional_name] = max(vector.get(functional_name, 0.0), weight)
        return vector

    def _build_temp_profile(self, temp_product_id: str, product_name: str, matched_ingredients: List[dict], product_name_hint: dict) -> dict:
        category_scores: Dict[str, float] = {}
        ingredient_scores: List[dict] = []
        excipient_ingredients: List[str] = []

        for item in matched_ingredients:
            display_name = str(item.get("display_name") or item["raw_ingredient"])
            functional_name = str(item["functional_ingredient_name"])
            role = str(item.get("input_role", "secondary") or "secondary")
            if is_excluded_primary(display_name) or is_excipient(display_name):
                excipient_ingredients.append(display_name)
                continue
            category_row = item.get("category_row") or {}
            category_main = str(category_row.get("category_main", "기타") or "기타")
            category_sub = str(category_row.get("category_sub", "") or "")
            weight = max(0.35, min(1.0, float(item.get("confidence", 0.75)))) * ROLE_WEIGHT.get(role, 0.55)
            category_scores[category_main] = category_scores.get(category_main, 0.0) + weight
            ingredient_scores.append(
                {
                    "ingredient": functional_name,
                    "display_name": display_name,
                    "normalized_for_matching": str(item.get("normalized_for_matching", functional_name)),
                    "weight": round(weight, 4),
                    "role": role,
                    "category_main": category_main,
                    "category_sub": category_sub,
                }
            )

        if product_name_hint.get("category_main"):
            category_scores[product_name_hint["category_main"]] = category_scores.get(product_name_hint["category_main"], 0.0) + 2.5

        sorted_categories = sorted(category_scores.items(), key=lambda item: (-item[1], item[0]))
        main_category = sorted_categories[0][0] if sorted_categories else "기타"
        sub_categories = [name for name, _score in sorted_categories[1:4]]

        primary: List[str] = []
        secondary: List[str] = []
        support: List[str] = []
        for item in sorted(ingredient_scores, key=lambda row: (-row["weight"], row["ingredient"])):
            if item["category_main"] == main_category and item["role"] in {"primary", "secondary"} and not is_excluded_primary(item["display_name"]):
                if len(primary) < 3:
                    primary.append(item["ingredient"])
                    item["role"] = "primary"
                    continue
            if item["role"] == "support":
                support.append(item["ingredient"])
            else:
                secondary.append(item["ingredient"])

        primary = list(dict.fromkeys(primary))
        secondary = [item for item in list(dict.fromkeys(secondary)) if item not in primary]
        support = [item for item in list(dict.fromkeys(support)) if item not in primary and item not in secondary]

        if product_name_hint.get("category_main") == "면역":
            for item in ingredient_scores:
                if contains_keyword(item["ingredient"], CATEGORY_HINT_RULES["면역"]["ingredient_keywords"]) and item["ingredient"] not in primary:
                    primary.insert(0, item["ingredient"])
                    main_category = "면역"
                    break
        if product_name_hint.get("category_main") == "관절/연골" and main_category == "기타":
            main_category = "관절/연골"

        role_by_ingredient = {item["ingredient"]: item["role"] for item in ingredient_scores}
        category_by_ingredient = {item["ingredient"]: item["category_main"] for item in ingredient_scores}
        return {
            "product_id": temp_product_id,
            "report_no": "",
            "product_name": product_name,
            "product_main_category": main_category,
            "product_sub_categories": sub_categories,
            "primary_ingredients": primary[:3],
            "secondary_ingredients": secondary[:8],
            "support_ingredients": support[:10],
            "excipient_ingredients": list(dict.fromkeys(excipient_ingredients)),
            "category_scores": {key: round(value, 4) for key, value in category_scores.items()},
            "ingredient_scores": ingredient_scores,
            "role_by_ingredient": role_by_ingredient,
            "category_by_ingredient": category_by_ingredient,
            "confidence": round(sorted_categories[0][1], 4) if sorted_categories else 0.0,
            "notes": "",
        }

    def detect_category_conflict(self, product_name_category: str, estimated_main_category: str, primary_ingredients: List[str]) -> dict:
        conflict = False
        primary_category_hint = self._detect_category_from_ingredients(primary_ingredients)
        if product_name_category and estimated_main_category and product_name_category != estimated_main_category:
            conflict = True
        if product_name_category and primary_category_hint and product_name_category != primary_category_hint:
            conflict = True
        return {
            "has_conflict": conflict,
            "product_name_category": product_name_category,
            "estimated_main_category": estimated_main_category,
            "primary_category_hint": primary_category_hint,
        }

    def apply_ocr_specific_profile_corrections(
        self,
        input_type: str,
        parsed_result: dict,
        matched_ingredients: List[dict],
        estimated_profile: dict,
        ocr_payload: Optional[dict],
        recommendations: List[dict],
    ) -> dict:
        estimated_profile = self._apply_domain_category_overrides(parsed_result, estimated_profile)
        base_warnings = []
        for warning in parsed_result.get("quality_warnings", []):
            normalized = normalize_warning_severity(warning)
            if normalized.get("code") in {"too_many_normalized_ingredients", "ocr_confidence_unavailable"}:
                continue
            base_warnings.append(normalized)
        quality_warnings = list(base_warnings)
        excluded_ingredients = list(dict.fromkeys(parsed_result.get("excluded_ingredients", []) + estimated_profile.get("excipient_ingredients", [])))
        product_name_candidate = str(parsed_result.get("product_name_candidate", "") or "")
        product_name_hint = self.infer_category_from_product_name(product_name_candidate)
        product_name_category = str(product_name_hint.get("category_main", "") or "")

        estimated_main_category = str(estimated_profile.get("product_main_category", "") or "")
        primary_ingredients = list(estimated_profile.get("primary_ingredients", []))
        top_similarity_score = float(recommendations[0].get("similarity_score", 0.0)) if recommendations else 0.0
        normalized_count = len(parsed_result.get("normalized_ingredients", []))
        category_diversity_count = self._collect_category_diversity(estimated_profile)
        primary_category_diversity_count = self._collect_primary_category_diversity(estimated_profile)
        primary_unclear = not primary_ingredients
        stable_primary_focus = self._has_stable_primary_focus(estimated_profile)

        offenders = [name for name in primary_ingredients if is_excluded_primary(name)]
        if offenders:
            excluded_ingredients.extend(offenders)
            estimated_profile["primary_ingredients"] = [name for name in primary_ingredients if name not in offenders]
            estimated_profile["secondary_ingredients"] = [name for name in estimated_profile.get("secondary_ingredients", []) if name not in offenders]
            estimated_profile["support_ingredients"] = [name for name in estimated_profile.get("support_ingredients", []) if name not in offenders]
            append_warning(
                quality_warnings,
                "excipient_in_primary",
                "부형제/첨가물로 보이는 성분을 추천 기준에서 제외했습니다. 원재료명을 확인해주세요.",
                severity="critical",
                ingredients=offenders,
            )

        conflict = self.detect_category_conflict(product_name_category, estimated_main_category, list(estimated_profile.get("primary_ingredients", [])))
        if conflict["has_conflict"]:
            append_warning(
                quality_warnings,
                "category_conflict_product_name_vs_ingredients",
                "제품명과 추출 원료 카테고리가 불일치합니다.",
                severity="critical",
                product_name_category=product_name_category,
                estimated_main_category=estimated_main_category,
            )

        if product_name_category and estimated_profile.get("product_main_category") == "기타" and (
            normalized_count >= 8 or len(estimated_profile.get("primary_ingredients", [])) >= 2
        ):
            append_warning(
                quality_warnings,
                "generic_category_with_named_product",
                "제품명 카테고리 힌트가 있으나 추정 카테고리가 기타입니다.",
                severity="critical",
                product_name_category=product_name_category,
            )

        parse_confidence = float(parsed_result.get("confidence", 0.0) or 0.0)
        if input_type in {"image", "ocr_text"} and parse_confidence < LOW_PARSE_CONFIDENCE_THRESHOLD:
            append_warning(
                quality_warnings,
                "low_parse_confidence",
                "LLM 파싱 신뢰도가 낮습니다. 추출된 원재료명을 확인해주세요.",
                severity="critical",
                confidence=parse_confidence,
            )

        ocr_confidence = ocr_payload.get("confidence") if ocr_payload else parsed_result.get("ocr_confidence")
        ocr_confidence_source = str((ocr_payload or {}).get("confidence_source") or parsed_result.get("ocr_confidence_source") or "unavailable")
        raw_text = str((ocr_payload or {}).get("raw_text", "") or "")
        ingredient_section_text = str(parsed_result.get("ingredient_section_text", "") or "")
        raw_text_sufficient = has_sufficient_ocr_text(raw_text, ingredient_section_text)
        if input_type == "image" and ocr_confidence is not None and float(ocr_confidence) < LOW_OCR_CONFIDENCE_THRESHOLD:
            append_warning(
                quality_warnings,
                "low_ocr_confidence",
                "OCR 인식 신뢰도가 낮습니다. 원재료명 인식 결과를 확인해주세요.",
                severity="critical",
                confidence=ocr_confidence,
                confidence_source=ocr_confidence_source,
            )
        elif input_type == "image" and ocr_confidence is None and raw_text_sufficient:
            append_warning(
                quality_warnings,
                "ocr_confidence_unavailable",
                "OCR 신뢰도 정보가 제공되지 않았습니다. 추출된 원재료명을 확인해주세요.",
                severity="info",
                confidence_source=ocr_confidence_source,
            )
        elif input_type == "image" and ocr_confidence is None and not raw_text_sufficient:
            append_warning(
                quality_warnings,
                "low_ocr_text_quality",
                "OCR 신뢰도 정보가 없고 추출 텍스트가 부족합니다. 원재료명 인식 결과를 확인해주세요.",
                severity="critical",
                confidence_source=ocr_confidence_source,
            )

        if category_diversity_count >= 3 and estimated_profile.get("product_main_category") in {"기타", "영양보충"} and not stable_primary_focus:
            append_warning(
                quality_warnings,
                "mixed_primary_categories",
                "주요 원료 카테고리가 넓게 분산되어 있어 원재료명을 확인해주세요.",
                severity="critical",
                category_diversity_count=category_diversity_count,
            )

        if input_type in {"image", "ocr_text"} and not parsed_result.get("ingredient_section_text"):
            append_warning(
                quality_warnings,
                "ingredient_section_unclear",
                "원재료명 영역이 불명확합니다. OCR 인식 결과를 확인해주세요.",
                severity="critical",
            )

        if recommendations and top_similarity_score < LOW_TOP_SIMILARITY_THRESHOLD:
            low_similarity_is_critical = primary_unclear or estimated_profile.get("product_main_category") in {"기타", "영양보충"} or category_diversity_count >= 4
            append_warning(
                quality_warnings,
                "low_top_similarity_score",
                "추천은 생성됐지만 상위 유사도가 낮습니다. 추출된 원재료명을 확인해주세요.",
                severity="critical" if low_similarity_is_critical else "notice",
                similarity_score=top_similarity_score,
            )

        if not recommendations:
            append_warning(
                quality_warnings,
                "missing_recommendations",
                "추천 결과가 충분하지 않습니다. 원재료명을 확인해주세요.",
                severity="critical",
            )

        if normalized_count >= TOO_MANY_INGREDIENTS_THRESHOLD:
            probiotic_stable_case = (
                estimated_profile.get("product_main_category") == "장 건강"
                and "프로바이오틱스" in {str(item or "") for item in parsed_result.get("primary_ingredients_normalized", [])}
                and primary_category_diversity_count <= 1
            )
            too_many_is_critical = (
                not probiotic_stable_case
                and not stable_primary_focus
                and (
                primary_category_diversity_count >= 3
                or conflict["has_conflict"]
                or top_similarity_score < 0.3
                or primary_unclear
                )
            )
            append_warning(
                quality_warnings,
                "too_many_normalized_ingredients",
                "정규화된 원료 수가 많습니다.",
                severity="critical" if too_many_is_critical else "notice",
                normalized_ingredients_count=normalized_count,
                category_diversity_count=category_diversity_count,
                primary_category_diversity_count=primary_category_diversity_count,
            )

        warning_groups = split_warning_groups(quality_warnings)
        needs_review = bool(warning_groups["critical_warnings"])
        review_message = warning_groups["critical_warnings"][0]["message"] if warning_groups["critical_warnings"] else ""

        return {
            "parsed_result": parsed_result,
            "estimated_profile": estimated_profile,
            "needs_user_review": needs_review,
            "review_message": review_message,
            "quality_warnings": warning_groups["quality_warnings"],
            "critical_warnings": warning_groups["critical_warnings"],
            "notices": warning_groups["notices"],
            "info_warnings": warning_groups["info_warnings"],
            "excluded_ingredients": list(dict.fromkeys([item for item in excluded_ingredients if item])),
            "product_name_category_hint": product_name_category,
            "category_diversity_count": category_diversity_count,
        }

    def calculate_similar_products_for_temp_vector(self, temp_vector: Dict[str, float], temp_profile: dict, top_k: int, candidate_limit: int) -> List[dict]:
        service = self.recommendation_service
        temp_product_id = str(temp_profile["product_id"])
        candidate_pool = build_candidate_pool(
            temp_product_id,
            temp_profile,
            service.product_vectors,
            service.profiles,
            service.ingredient_postings,
            service.ingredient_frequency,
            candidate_limit,
            get_settings().default_max_df_for_seed,
        )

        rows: List[dict] = []
        total_product_count = len(service.product_vectors)
        for candidate in candidate_pool:
            target_product_id = candidate["target_product_id"]
            target_profile = service.profiles[target_product_id]
            target_vector = service.product_vectors[target_product_id]
            similarity_score, _ = calculate_weighted_jaccard_with_idf(
                temp_vector,
                target_vector,
                service.ingredient_frequency,
                total_product_count,
                temp_profile,
                target_profile,
            )
            if similarity_score <= 0:
                continue
            comparison = compare_product_profiles(temp_profile, target_profile, temp_vector, target_vector)
            function_similarity_score = calculate_function_similarity(temp_profile, target_profile)
            core_match_score = calculate_core_match_score(temp_profile, target_profile)
            substitutability = classify_substitutability(similarity_score, temp_profile, target_profile)
            reason = generate_recommendation_reason(temp_profile, target_profile, comparison, service.ingredient_frequency)
            explanation = build_explanation_json(reason, comparison, substitutability)
            rows.append(
                {
                    "rank": 0,
                    "target_report_no": str(target_profile.get("report_no", "") or ""),
                    "target_product_name": str(target_profile.get("product_name", "") or ""),
                    "similarity_score": float(similarity_score),
                    "function_similarity_score": float(function_similarity_score),
                    "core_match_score": float(core_match_score),
                    "substitutability": substitutability,
                    "shared_ingredients": comparison["shared_ingredients"],
                    "shared_categories": comparison["shared_categories"],
                    "reason": reason,
                    "caution": generate_caution(),
                    "explanation": explanation,
                }
            )
        rows = sorted(rows, key=lambda item: (-item["similarity_score"], -item["core_match_score"], -item["function_similarity_score"], item["target_report_no"]))[:top_k]
        for index, item in enumerate(rows, start=1):
            item["rank"] = index
            item["reason"] = self._override_recommendation_reason(temp_profile, item)
            item["explanation"]["reason"] = item["reason"]
        return rows

    def _build_response(
        self,
        input_type: str,
        ocr_payload: Optional[dict],
        parsed: dict,
        matched_ingredients: List[dict],
        estimated_profile: dict,
        recommendations: List[dict],
        execution_seconds: float,
    ) -> dict:
        corrected = self.apply_ocr_specific_profile_corrections(input_type, parsed, matched_ingredients, estimated_profile, ocr_payload, recommendations)

        if corrected["product_name_category_hint"] and recommendations:
            top_shared_categories = recommendations[0].get("shared_categories", [])
            top_similarity_score = float(recommendations[0].get("similarity_score", 0.0) or 0.0)
            if top_shared_categories and corrected["product_name_category_hint"] not in top_shared_categories and top_similarity_score < LOW_TOP_SIMILARITY_THRESHOLD:
                corrected["needs_user_review"] = True
                if not corrected["review_message"]:
                    corrected["review_message"] = "제품명과 추천 결과 카테고리가 다를 수 있어 OCR 인식 결과를 확인해주세요."
                combined_warnings = corrected["critical_warnings"] + corrected["notices"] + corrected["info_warnings"]
                append_warning(
                    combined_warnings,
                    "top_reason_category_mismatch",
                    "제품명 카테고리와 추천 결과의 대표 카테고리가 다를 수 있습니다.",
                    severity="critical",
                    product_name_category=corrected["product_name_category_hint"],
                    shared_categories=top_shared_categories,
                )
                regrouped = split_warning_groups(combined_warnings)
                corrected["quality_warnings"] = regrouped["quality_warnings"]
                corrected["critical_warnings"] = regrouped["critical_warnings"]
                corrected["notices"] = regrouped["notices"]
                corrected["info_warnings"] = regrouped["info_warnings"]

        return {
            "input_type": input_type,
            "ocr": ocr_payload,
            "parsed": corrected["parsed_result"],
            "detected_functional_ingredients": [
                {
                    "raw_ingredient": item["raw_ingredient"],
                    "functional_ingredient_name": item["functional_ingredient_name"],
                    "relation_type": item["relation_type"],
                    "confidence": item["confidence"],
                    "display_name": item.get("display_name", item["raw_ingredient"]),
                    "normalized_for_matching": item.get("normalized_for_matching", item["functional_ingredient_name"]),
                }
                for item in matched_ingredients
            ],
            "estimated_profile": {
                "product_main_category": corrected["estimated_profile"].get("product_main_category", ""),
                "primary_ingredients": corrected["estimated_profile"].get("primary_ingredients", []),
                "secondary_ingredients": corrected["estimated_profile"].get("secondary_ingredients", []),
                "support_ingredients": corrected["estimated_profile"].get("support_ingredients", []),
            },
            "recommendations": recommendations,
            "execution_seconds": execution_seconds,
            "needs_user_review": corrected["needs_user_review"],
            "review_message": corrected["review_message"],
            "quality_warnings": corrected["quality_warnings"],
            "critical_warnings": corrected["critical_warnings"],
            "notices": corrected["notices"],
            "info_warnings": corrected["info_warnings"],
            "parsed_ingredients_for_review": corrected["parsed_result"].get("normalized_ingredients", []),
            "excluded_ingredients": corrected["excluded_ingredients"],
            "product_name_category_hint": corrected["product_name_category_hint"],
            "category_diversity_count": corrected["category_diversity_count"],
        }

    def recommend_from_ingredients(self, ingredients: List[str], top_k: int = 10, candidate_limit: int = 1000, product_name_candidate: str = "") -> dict:
        self._ensure_loaded()
        started = time.perf_counter()
        ingredient_objects = []
        for item in ingredients:
            canonical = canonicalize_ingredient_for_matching(item)
            role = classify_ingredient_role(item)
            ingredient_objects.append(
                {
                    "raw": item,
                    "normalized_for_matching": canonical,
                    "display_name": item,
                    "role": role,
                    "category_hint": "",
                }
            )
        parsed = {
            "product_name_candidate": product_name_candidate,
            "ingredient_section_text": "",
            "functional_ingredient_candidates": [item["normalized_for_matching"] for item in ingredient_objects if item["normalized_for_matching"]],
            "raw_ingredients": [item["raw"] for item in ingredient_objects],
            "normalized_ingredients": [item["normalized_for_matching"] for item in ingredient_objects if item["normalized_for_matching"] and item["role"] != "excipient"],
            "ingredient_objects": ingredient_objects,
            "primary_ingredients": [item["display_name"] for item in ingredient_objects if item["role"] == "primary"],
            "primary_ingredients_normalized": [
                item["normalized_for_matching"] for item in ingredient_objects if item["role"] == "primary" and item["normalized_for_matching"]
            ],
            "excluded_ingredients": [item["display_name"] for item in ingredient_objects if item["role"] == "excipient"],
            "excluded_ingredient_objects": [item for item in ingredient_objects if item["role"] == "excipient"],
            "quality_warnings": [],
            "daily_intake_text": "",
            "confidence": 0.95,
            "needs_user_review": False,
        }
        matched = self.match_raw_ingredients_to_functional_ingredients(ingredient_objects)
        temp_vector = self.build_temp_product_vector_from_ingredients(ingredient_objects)
        temp_hash = hashlib.sha1("|".join(sorted(parsed["normalized_ingredients"])).encode("utf-8")).hexdigest()[:16]
        name_hint = self.infer_category_from_product_name(product_name_candidate)
        temp_profile = self._build_temp_profile(f"uploaded::{temp_hash}", product_name_candidate or "업로드 입력 제품", matched, name_hint)
        temp_profile = self._apply_domain_category_overrides(parsed, temp_profile)
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit) if temp_vector else []
        return self._build_response("ingredients", None, parsed, matched, temp_profile, recommendations, round(time.perf_counter() - started, 6))

    def recommend_from_ocr_text(self, raw_text: str, top_k: int = 10, candidate_limit: int = 1000) -> dict:
        self._ensure_loaded()
        started = time.perf_counter()
        parsed = parse_ingredients_from_ocr_text(raw_text)
        matched = self.match_raw_ingredients_to_functional_ingredients(parsed.get("ingredient_objects", []))
        temp_vector = self.build_temp_product_vector_from_ingredients(parsed.get("ingredient_objects", []))
        temp_hash = hashlib.sha1((raw_text or "").encode("utf-8")).hexdigest()[:16]
        name_hint = self.infer_category_from_product_name(parsed.get("product_name_candidate", ""))
        temp_profile = self._build_temp_profile(f"uploaded::{temp_hash}", parsed.get("product_name_candidate") or "OCR 입력 제품", matched, name_hint)
        temp_profile = self._apply_domain_category_overrides(parsed, temp_profile)
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit) if temp_vector else []
        ocr_payload = {
            "raw_text": raw_text,
            "confidence": None,
            "confidence_source": "unavailable",
            "lines": [],
            "blocks": [],
            "source": "ocr_text",
        }
        return self._build_response("ocr_text", ocr_payload, parsed, matched, temp_profile, recommendations, round(time.perf_counter() - started, 6))

    def recommend_from_uploaded_image(self, image_bytes: bytes, filename: str, top_k: int = 10, candidate_limit: int = 1000) -> dict:
        self._ensure_loaded()
        started = time.perf_counter()
        suffix = Path(filename or "upload.jpg").suffix.lower() or ".jpg"
        temp_path = save_temp_upload(image_bytes, suffix)
        try:
            ocr_result = extract_text_from_image(str(temp_path))
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        if ocr_result.get("error"):
            return {
                "input_type": "image",
                "ocr": ocr_result,
                "parsed": {
                    "product_name_candidate": "",
                    "ingredient_section_text": "",
                    "functional_ingredient_candidates": [],
                    "raw_ingredients": [],
                    "normalized_ingredients": [],
                    "ingredient_objects": [],
                    "primary_ingredients": [],
                    "primary_ingredients_normalized": [],
                    "excluded_ingredients": [],
                    "excluded_ingredient_objects": [],
                    "quality_warnings": [],
                    "nutrition_or_active_components": [],
                    "daily_intake_text": "",
                    "confidence": 0.0,
                    "needs_user_review": True,
                    "warnings": [],
                },
                "detected_functional_ingredients": [],
                "estimated_profile": {
                    "product_main_category": "",
                    "primary_ingredients": [],
                    "secondary_ingredients": [],
                    "support_ingredients": [],
                },
                "recommendations": [],
                "execution_seconds": round(time.perf_counter() - started, 6),
                "needs_user_review": True,
                "review_message": "OCR 인식 결과를 확인해주세요.",
                "quality_warnings": [{"code": "ocr_error", "message": str(ocr_result.get("error"))}],
                "parsed_ingredients_for_review": [],
                "excluded_ingredients": [],
                "product_name_category_hint": "",
            }

        parsed = parse_ingredients_from_ocr_text(str(ocr_result.get("raw_text", "")))
        parsed["ocr_confidence"] = ocr_result.get("confidence")
        parsed["ocr_confidence_source"] = ocr_result.get("confidence_source", "unavailable")
        matched = self.match_raw_ingredients_to_functional_ingredients(parsed.get("ingredient_objects", []))
        temp_vector = self.build_temp_product_vector_from_ingredients(parsed.get("ingredient_objects", []))
        temp_hash = hashlib.sha1((filename + str(ocr_result.get("raw_text", ""))).encode("utf-8")).hexdigest()[:16]
        name_hint = self.infer_category_from_product_name(parsed.get("product_name_candidate", filename))
        temp_profile = self._build_temp_profile(f"uploaded::{temp_hash}", parsed.get("product_name_candidate") or filename, matched, name_hint)
        temp_profile = self._apply_domain_category_overrides(parsed, temp_profile)
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit) if temp_vector else []
        return self._build_response("image", ocr_result, parsed, matched, temp_profile, recommendations, round(time.perf_counter() - started, 6))


def coerce_ingredient_request_payload(ingredients: Optional[List[str]], raw_ingredients: Optional[str]) -> List[str]:
    if ingredients:
        return [str(item).strip() for item in ingredients if str(item).strip()]
    if raw_ingredients:
        return split_ingredients(raw_ingredients)
    return []
