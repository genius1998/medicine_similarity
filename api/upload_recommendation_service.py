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
        "product_keywords": ["루테인", "지아잔틴", "눈", "아이", "마리골드"],
        "ingredient_keywords": ["루테인", "지아잔틴", "마리골드꽃추출물", "베타카로틴", "비타민 A"],
    },
    "관절/연골": {
        "product_keywords": ["관절", "연골", "콘드로이친", "뮤코다당", "뮤코다당단백", "철갑상어", "철갑상어연골", "MSM", "보스웰리아", "NEM", "난각막", "강황"],
        "ingredient_keywords": ["콘드로이친", "뮤코다당단백", "철갑상어연골", "MSM", "엠에스엠", "보스웰리아", "난각막", "난각막분말", "난각막가수분해물", "강황추출물", "초록입홍합"],
    },
    "장 건강": {
        "product_keywords": ["유산균", "프로바이오틱스", "락토핏", "장", "대장", "프리바이오틱스", "올리고당"],
        "ingredient_keywords": ["프로바이오틱스", "유산균", "갈락토올리고당", "프락토올리고당", "난소화성말토덱스트린", "이눌린"],
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

ROLE_WEIGHT = {
    "primary": 1.0,
    "secondary": 0.72,
    "support": 0.45,
}


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def contains_keyword(text: str, keywords: List[str]) -> bool:
    normalized = normalize_lookup_key(text)
    return any(normalize_lookup_key(keyword) in normalized for keyword in keywords)


def is_excluded_primary(name: str) -> bool:
    normalized = normalize_lookup_key(name)
    if "난소화성말토덱스트린" in normalized:
        return False
    return any(normalize_lookup_key(keyword) in normalized for keyword in EXCLUDED_PRIMARY_KEYWORDS)


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

    def apply_ocr_specific_profile_corrections(self, parsed_result: dict, matched_ingredients: List[dict], estimated_profile: dict) -> dict:
        quality_warnings = list(parsed_result.get("quality_warnings", []))
        excluded_ingredients = list(dict.fromkeys(parsed_result.get("excluded_ingredients", []) + estimated_profile.get("excipient_ingredients", [])))
        product_name_candidate = str(parsed_result.get("product_name_candidate", "") or "")
        product_name_hint = self.infer_category_from_product_name(product_name_candidate)
        product_name_category = str(product_name_hint.get("category_main", "") or "")

        estimated_main_category = str(estimated_profile.get("product_main_category", "") or "")
        primary_ingredients = list(estimated_profile.get("primary_ingredients", []))

        offenders = [name for name in primary_ingredients if is_excluded_primary(name)]
        if offenders:
            excluded_ingredients.extend(offenders)
            estimated_profile["primary_ingredients"] = [name for name in primary_ingredients if name not in offenders]
            estimated_profile["secondary_ingredients"] = [name for name in estimated_profile.get("secondary_ingredients", []) if name not in offenders]
            estimated_profile["support_ingredients"] = [name for name in estimated_profile.get("support_ingredients", []) if name not in offenders]
            quality_warnings.append(
                {
                    "code": "excipient_in_primary",
                    "message": "부형제/첨가물이 primary 성분에 포함되어 제외했습니다.",
                    "ingredients": offenders,
                }
            )

        conflict = self.detect_category_conflict(product_name_category, estimated_main_category, list(estimated_profile.get("primary_ingredients", [])))
        needs_review = bool(parsed_result.get("needs_user_review", False))
        review_message = ""

        if conflict["has_conflict"]:
            needs_review = True
            if product_name_category == "면역":
                review_message = "제품명은 면역/홍삼 계열로 보이지만, OCR에서 추출된 주요 원료는 다른 카테고리로 분류되었습니다. OCR 인식 결과를 확인해주세요."
                if estimated_profile.get("product_main_category") in {"기타", "눈 건강"}:
                    estimated_profile["product_main_category"] = "면역"
            else:
                review_message = "제품명과 추출 원료 카테고리가 불일치합니다. OCR 인식 결과를 확인해주세요."
            quality_warnings.append(
                {
                    "code": "category_conflict_product_name_vs_ingredients",
                    "message": "제품명과 추출 원료 카테고리가 불일치합니다.",
                    "product_name_category": product_name_category,
                    "estimated_main_category": estimated_main_category,
                }
            )

        if product_name_category and estimated_profile.get("product_main_category") == "기타":
            needs_review = True
            quality_warnings.append(
                {
                    "code": "generic_category_with_named_product",
                    "message": "제품명 카테고리 힌트가 있으나 추정 카테고리가 기타입니다.",
                    "product_name_category": product_name_category,
                }
            )

        if float(parsed_result.get("confidence", 0.0) or 0.0) < 0.8:
            needs_review = True
            quality_warnings.append(
                {
                    "code": "low_parse_confidence",
                    "message": "OCR/LLM 파싱 confidence가 낮아 검토가 필요합니다.",
                    "confidence": parsed_result.get("confidence", 0.0),
                }
            )

        ocr_confidence = parsed_result.get("ocr_confidence")
        if ocr_confidence is not None and float(ocr_confidence) < 0.8:
            needs_review = True
            quality_warnings.append(
                {
                    "code": "low_ocr_confidence",
                    "message": "OCR confidence가 낮아 검토가 필요합니다.",
                    "confidence": ocr_confidence,
                }
            )

        if len(parsed_result.get("normalized_ingredients", [])) >= 20:
            needs_review = True

        if estimated_profile.get("product_main_category") == "기타" and len(estimated_profile.get("primary_ingredients", [])) >= 2:
            primary_categories = {
                item.get("category_main", "")
                for item in estimated_profile.get("ingredient_scores", [])
                if item.get("ingredient") in estimated_profile.get("primary_ingredients", [])
            }
            primary_categories = {item for item in primary_categories if item and item != "기타"}
            if len(primary_categories) >= 2:
                needs_review = True
                quality_warnings.append(
                    {
                        "code": "mixed_primary_categories",
                        "message": "기타 카테고리인데 primary 원료의 카테고리가 혼재되어 있습니다.",
                    }
                )

        if not parsed_result.get("ingredient_section_text"):
            needs_review = True

        top_reason = ""
        if matched_ingredients:
            pass

        if not review_message and needs_review:
            review_message = "OCR 인식 결과를 확인해주세요."

        return {
            "parsed_result": parsed_result,
            "estimated_profile": estimated_profile,
            "needs_user_review": needs_review,
            "review_message": review_message,
            "quality_warnings": quality_warnings,
            "excluded_ingredients": list(dict.fromkeys([item for item in excluded_ingredients if item])),
            "product_name_category_hint": product_name_category,
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
        corrected = self.apply_ocr_specific_profile_corrections(parsed, matched_ingredients, estimated_profile)

        if corrected["product_name_category_hint"] and recommendations:
            top_shared_categories = recommendations[0].get("shared_categories", [])
            if top_shared_categories and corrected["product_name_category_hint"] not in top_shared_categories:
                corrected["needs_user_review"] = True
                if not corrected["review_message"]:
                    corrected["review_message"] = "제품명과 추천 결과 카테고리가 다를 수 있어 OCR 인식 결과를 확인해주세요."
                corrected["quality_warnings"].append(
                    {
                        "code": "top_reason_category_mismatch",
                        "message": "제품명 카테고리와 추천 결과의 대표 카테고리가 다를 수 있습니다.",
                        "product_name_category": corrected["product_name_category_hint"],
                        "shared_categories": top_shared_categories,
                    }
                )

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
            "parsed_ingredients_for_review": corrected["parsed_result"].get("normalized_ingredients", []),
            "excluded_ingredients": corrected["excluded_ingredients"],
            "product_name_category_hint": corrected["product_name_category_hint"],
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
            "excluded_ingredients": [item["display_name"] for item in ingredient_objects if item["role"] == "excipient"],
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
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit) if temp_vector else []
        ocr_payload = {"raw_text": raw_text, "confidence": None, "lines": [], "blocks": [], "source": "ocr_text"}
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
                    "excluded_ingredients": [],
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
        matched = self.match_raw_ingredients_to_functional_ingredients(parsed.get("ingredient_objects", []))
        temp_vector = self.build_temp_product_vector_from_ingredients(parsed.get("ingredient_objects", []))
        temp_hash = hashlib.sha1((filename + str(ocr_result.get("raw_text", ""))).encode("utf-8")).hexdigest()[:16]
        name_hint = self.infer_category_from_product_name(parsed.get("product_name_candidate", filename))
        temp_profile = self._build_temp_profile(f"uploaded::{temp_hash}", parsed.get("product_name_candidate") or filename, matched, name_hint)
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit) if temp_vector else []
        return self._build_response("image", ocr_result, parsed, matched, temp_profile, recommendations, round(time.perf_counter() - started, 6))


def coerce_ingredient_request_payload(ingredients: Optional[List[str]], raw_ingredients: Optional[str]) -> List[str]:
    if ingredients:
        return [str(item).strip() for item in ingredients if str(item).strip()]
    if raw_ingredients:
        return split_ingredients(raw_ingredients)
    return []
