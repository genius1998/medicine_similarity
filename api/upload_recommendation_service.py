from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import pandas as pd

from api.config import get_settings
from api.db import sqlite_connection
from api.ingredient_embedding_index import IngredientEmbeddingIndex
from api.ingredient_family import (
    PROTECTED_JOINT_FAMILIES,
    canonical_joint_family_name,
    infer_joint_ingredient_family,
    joint_family_aliases,
)
from api.ingredient_parse_service import (
    _looks_like_functional_premix,
    canonicalize_ingredient_for_matching,
    classify_ingredient_role,
    compute_ocr_text_hash,
    compute_parsed_signature,
    is_excipient,
    normalize_product_category_label,
    parse_ingredients_from_ocr_text,
    split_ingredients,
)
from api.local_llm_client import call_local_llm, extract_json_from_llm_content
from api.ocr_service import extract_text_from_image, save_temp_upload
from api.recommendation_service import RecommendationService
from scripts.enhance_similarity_with_explanation import (
    build_candidate_pool_v2,
    build_explanation_json,
    build_semantic_weight_vector,
    calculate_core_match_score,
    calculate_function_similarity,
    calculate_semantic_weighted_jaccard_v2,
    classify_substitutability,
    compare_product_profiles,
    generate_caution,
    generate_recommendation_reason,
    is_semantic_excipient_name,
    recommendation_quality_metadata,
    safe_json_loads,
    SIMILARITY_ALGORITHM_VERSION,
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

NON_FUNCTIONAL_CACHE_RELATION_TYPES = {"excipient", "unrelated", "unknown", "error"}

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

logger = logging.getLogger(__name__)

PROGRESS_TTL_SECONDS = 900
_PROGRESS_LOCK = Lock()
_REQUEST_PROGRESS: Dict[str, dict] = {}


def update_request_progress(
    request_id: str,
    *,
    phase: str = "",
    message: str = "",
    percent: Optional[float] = None,
    detail: Optional[str] = None,
    current_ingredient: Optional[str] = None,
    current: Optional[int] = None,
    total: Optional[int] = None,
    status: str = "running",
    extra: Optional[dict] = None,
) -> None:
    request_id = str(request_id or "").strip()
    if not request_id:
        return
    now = time.time()
    with _PROGRESS_LOCK:
        stale_ids = [
            key
            for key, value in _REQUEST_PROGRESS.items()
            if now - float(value.get("updated_at_epoch", now) or now) > PROGRESS_TTL_SECONDS
        ]
        for key in stale_ids:
            _REQUEST_PROGRESS.pop(key, None)
        previous = dict(_REQUEST_PROGRESS.get(request_id, {}) or {})
        previous_phase = str(previous.get("phase", "") or "")
        next_phase = phase or previous_phase
        payload = {
            **previous,
            "request_id": request_id,
            "phase": next_phase,
            "message": message or previous.get("message", ""),
            "detail": detail if detail is not None else ("" if phase and phase != previous_phase else previous.get("detail", "")),
            "current_ingredient": current_ingredient if current_ingredient is not None else ("" if phase and phase != previous_phase else previous.get("current_ingredient", "")),
            "status": status or previous.get("status", "running"),
            "updated_at_epoch": now,
        }
        if percent is not None:
            payload["percent"] = max(0, min(100, round(float(percent), 1)))
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if extra:
            payload["extra"] = {**dict(previous.get("extra", {}) or {}), **extra}
        _REQUEST_PROGRESS[request_id] = payload


def get_request_progress(request_id: str) -> dict:
    request_id = str(request_id or "").strip()
    if not request_id:
        return {"request_id": "", "status": "missing", "percent": 0}
    with _PROGRESS_LOCK:
        payload = dict(_REQUEST_PROGRESS.get(request_id, {}) or {})
    if not payload:
        return {"request_id": request_id, "status": "unknown", "percent": 0, "message": "No progress record found."}
    return payload

ENABLE_RUNTIME_RAG_FALLBACK = True
RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME = "runtime_custom_functional_ingredient_map"
RUNTIME_RAG_TOP_K = 5
RUNTIME_RAG_NAME_SIMILARITY_MIN = 0.78

RUNTIME_LIKE_DISABLED_KEYS = {
    "hca",
    "gaba",
    "cla",
    "msm",
    "epa",
    "dha",
    "cpp",
    "bcaa",
    "coq10",
    "mk7",
    "rg1",
    "rb1",
    "rg3",
    "l테아닌",
    "테아닌",
}

VECTOR_ALLOWED_RELATION_TYPES = {
    "same_ingredient",
    "marker_compound",
    "ingredient_group",
    "nutrient_form",
}

RUNTIME_CACHE_LIKE_VECTOR_EXCLUDED_TERMS = {
    "\uc2dd\ubb3c\ud63c\ud569\ub18d\ucd95\uc561",
    "\uc2dd\ubb3c\ud63c\ud569\ucd94\ucd9c\ubb3c",
    "\ud63c\ud569\ub18d\ucd95\uc561",
    "\uc885\uc790\ucd94\ucd9c\ubb3c",
    "\uad6c\uc5f0\uc0b0",
    "\uad6c\uc5f0\uc0b0\uc0bc\ub098\ud2b8\ub968",
    "\uad6c\uc5f0\uc0b0\ub098\ud2b8\ub968",
    "\uc804\ubd84",
    "\uc625\uc218\uc218",
    "\ub9d0\ud1a0\ub371\uc2a4\ud2b8\ub9b0",
    "\ub371\uc2a4\ud2b8\ub9b0",
    "\uc815\uc81c\uc218",
    "\uc8fc\uc815",
    "\uc5d0\ud0c4\uc62c",
    "\ud5a5\ub8cc",
    "\uac10\ubbf8\ub8cc",
    "\ud63c\ud569\uc81c\uc81c",
    "200mg",
    "500mg",
    "mg",
    "g",
    "%",
    "hpmc",
    "\ud788\ub4dc\ub85d\uc2dc\ud504\ub85c\ud544\uba54\ud2f8\uc140\ub8f0\ub85c\uc2a4",
    "\uce74\ubcf5\uc2dc\uba54\ud2f8\uc140\ub8f0\ub85c\uc2a4",
    "\uc2a4\ud14c\uc544\ub9b0\uc0b0\ub9c8\uadf8\ub124\uc298",
    "\uc774\uc0b0\ud654\uaddc\uc18c",
    "\uacb0\uc815\uc140\ub8f0\ub85c\uc2a4",
    "\uae00\ub9ac\uc138\ub9b0",
    "d\uc18c\ube44\ud1a8\uc561",
    "\uc18c\ube44\ud1a8\uc561",
}

RUNTIME_EXCIPIENT_VECTOR_EXCLUDED_TERMS = {
    "\ud788\ub4dc\ub85d\uc2dc\ud504\ub85c\ud544\uba54\ud2f8\uc140\ub8f0\ub85c\uc624\uc2a4",
    "\ud788\ub4dc\ub85d\uc2dc\ud504\ub85c\ud544\uba54\ud2f8\uc140\ub8f0\ub85c\uc2a4",
    "hpmc",
    "\uacb0\uc815\uc140\ub8f0\ub85c\uc2a4",
    "\ubbf8\uacb0\uc815\uc140\ub8f0\ub85c\uc2a4",
    "\uc774\uc0b0\ud654\uaddc\uc18c",
    "\uc2a4\ud14c\uc544\ub9b0\uc0b0\ub9c8\uadf8\ub124\uc298",
    "\uc2a4\ud14c\uc544\ub9b0\uc0b0\uce7c\uc298",
    "\uce74\ubcf5\uc2dc\uba54\ud2f8\uc140\ub8f0\ub85c\uc2a4",
    "\uce74\ubcf5\uc2dc\uba54\ud2f8\uc140\ub8f0\ub85c\uc2a4\uce7c\uc298",
    "\uae00\ub9ac\uc138\ub9b0",
    "d\uc18c\ube44\ud1a8\uc561",
    "\uc18c\ube44\ud1a8\uc561",
    "\ub371\uc2a4\ud2b8\ub9b0",
    "\ub9d0\ud1a0\ub371\uc2a4\ud2b8\ub9b0",
    "\uc815\uc81c\uc218",
    "\uc544\ub77c\ube44\uc544\uac80",
    "\uc124\ud0d5",
    "\uc774\uc18c\ub9d0\ud2b8",
    "\uc5d0\ub9ac\uc2a4\ub9ac\ud1a8",
    "\uc790\uc77c\ub9ac\ud1a8",
    "\ud5a5\ub8cc",
    "\ucc29\ud5a5\ub8cc",
    "\uac10\ubbf8\ub8cc",
}

RUNTIME_RAG_FALLBACK_SKIP_TERMS = RUNTIME_EXCIPIENT_VECTOR_EXCLUDED_TERMS | {
    "\uac00\uacf5\uc720\uc9c0",
    "\uc720\ub2f9",
    "\uc720\ub2f9\ud63c\ud569\ubd84\ub9d0",
    "\uac74\uc870\ud6a8\ubaa8",
    "\ucc44\uc18c\ud63c\ud569\ubd84\ub9d0",
    "\ucc44\uc18c\ud638\ud569\ubd84\ub9d0",
    "\uc57c\ucc44\ud63c\ud569\ubd84\ub9d0",
    "\uc804\ubd84",
}

RUNTIME_LOW_SIGNAL_UPLOAD_VECTOR_EXACT_TERMS = {
    "\uac00\uacf5\uc720\uc9c0",
    "\uc720\ub2f9",
    "\uc720\ub2f9\ud63c\ud569\ubd84\ub9d0",
    "\uac74\uc870\ud6a8\ubaa8",
    "\ucc44\uc18c\ud63c\ud569\ubd84\ub9d0",
    "\ucc44\uc18c\ud638\ud569\ubd84\ub9d0",
    "\uc57c\ucc44\ud63c\ud569\ubd84\ub9d0",
    "\uc804\ubd84",
}

RUNTIME_VECTOR_EXCLUSION_REASON_MESSAGES = {
    "excluded_from_vector_by_cache_like_guard": "cache_like guard excluded this match from the upload vector",
    "excluded_from_vector_by_excipient_guard": "excipient guard excluded this match from the upload vector",
    "excluded_from_vector_by_low_signal_upload_guard": "low-signal upload ingredient excluded from the upload vector",
    "excluded_from_vector_by_relation_guard": "relation type guard excluded this match from the upload vector",
}

RUNTIME_SUSPICIOUS_MAPPING_RULES = [
    {
        "name": "hca",
        "inputs": ["hca", "hydroxycitricacid", "hydroxycitric"],
        "allowed": ["가르시니아", "garcinia", "hca"],
        "blocked": ["녹차", "카테킨", "egcg", "greentea", "greentea"],
        "safe_aliases": [
            "가르시니아 캄보지아 껍질추출물 60HCA",
            "가르시니아 캄보지아 추출물",
            "가르시니아 캄보지아 껍질추출물",
        ],
    },
    {
        "name": "silymarin",
        "inputs": ["실리마린", "silymarin"],
        "allowed": ["밀크씨슬", "밀크시슬", "milkthistle", "silymarin"],
        "blocked": [],
        "safe_aliases": ["밀크씨슬추출물", "밀크씨슬 추출물", "밀크씨슬", "milk thistle"],
    },
    {
        "name": "ginsenoside",
        "inputs": ["진세노사이드", "ginsenoside"],
        "allowed": ["홍삼", "인삼", "ginseng", "redginseng"],
        "blocked": [],
        "safe_aliases": ["홍삼", "인삼"],
    },
    {
        "name": "lutein_zeaxanthin",
        "inputs": ["루테인", "lutein", "지아잔틴", "zeaxanthin"],
        "allowed": ["루테인", "지아잔틴", "마리골드", "lutein", "zeaxanthin", "marigold"],
        "blocked": [],
        "safe_aliases": ["루테인", "지아잔틴", "마리골드꽃추출물", "마리골드꽃추출물(지아잔틴함유)"],
    },
    {
        "name": "omega3",
        "inputs": ["epa", "dha", "오메가3", "오메가-3", "omega3", "fishoil", "fish oil"],
        "allowed": ["epa", "dha", "오메가", "omega", "어유", "fishoil", "fishoil"],
        "blocked": [],
        "safe_aliases": ["EPA 및 DHA 함유 유지", "EPA 및 DHA 함유 유지(고시형 원료)", "오메가-3 지방산 함유유지"],
    },
    {
        "name": "vitamin_d",
        "inputs": ["비타민d3", "건조비타민d3", "vitamind3", "vitamind"],
        "allowed": ["비타민d", "vitamind"],
        "blocked": [],
        "safe_aliases": ["비타민 D", "비타민D"],
    },
    {
        "name": "zinc",
        "inputs": ["산화아연", "글루콘산아연", "zincoxide", "zincgluconate"],
        "allowed": ["아연", "zinc"],
        "blocked": [],
        "safe_aliases": ["아연", "zinc"],
    },
]

JOINT_FAMILY_CATEGORY_SUB = {
    "chondroitin": "콘드로이친 계열",
    "glucosamine": "글루코사민 계열",
    "nag": "NAG 계열",
    "uc_ii": "UC-II 계열",
    "msm": "MSM 계열",
    "boswellia": "보스웰리아 계열",
}


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def normalize_cache_exact_key(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", str(value or "").strip().lower())


def normalized_name_similarity(left: str, right: str) -> float:
    left_key = normalize_cache_exact_key(left)
    right_key = normalize_cache_exact_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    shorter = min(len(left_key), len(right_key))
    if shorter < 4:
        return 0.0
    return float(SequenceMatcher(None, left_key, right_key).ratio())


def dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        key = normalize_lookup_key(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def contains_normalized_fragment(value: str, fragments: List[str]) -> bool:
    normalized = normalize_cache_exact_key(value)
    return any(normalize_cache_exact_key(fragment) in normalized for fragment in fragments if str(fragment or "").strip())


def resolve_runtime_mapping_rule(*values: str) -> Optional[dict]:
    normalized_values = [normalize_cache_exact_key(value) for value in values if str(value or "").strip()]
    for rule in RUNTIME_SUSPICIOUS_MAPPING_RULES:
        for normalized_value in normalized_values:
            if any(normalize_cache_exact_key(trigger) in normalized_value for trigger in rule["inputs"]):
                return rule
    return None


def is_like_blocked_input(*values: str) -> bool:
    normalized_values = [normalize_cache_exact_key(value) for value in values if str(value or "").strip()]
    for normalized in normalized_values:
        if normalized in RUNTIME_LIKE_DISABLED_KEYS:
            return True
        if 0 < len(normalized) <= 4 and re.fullmatch(r"[a-z0-9]+", normalized):
            return True
    return False


def contains_guarded_runtime_term(value: str, guarded_terms: set[str]) -> bool:
    normalized_value = normalize_cache_exact_key(value)
    if not normalized_value:
        return False
    for term in guarded_terms:
        normalized_term = normalize_cache_exact_key(term)
        if not normalized_term:
            continue
        if normalized_value == normalized_term or normalized_term in normalized_value:
            return True
    return False


def contains_exact_guarded_runtime_term(value: str, guarded_terms: set[str]) -> bool:
    normalized_value = normalize_cache_exact_key(value)
    if not normalized_value:
        return False
    for term in guarded_terms:
        normalized_term = normalize_cache_exact_key(term)
        if normalized_term and normalized_value == normalized_term:
            return True
    return False


def is_low_signal_upload_vector_term(*values: str) -> bool:
    return any(contains_exact_guarded_runtime_term(value, RUNTIME_LOW_SIGNAL_UPLOAD_VECTOR_EXACT_TERMS) for value in values)


def looks_like_runtime_measurement_or_symbol(value: str) -> bool:
    text = str(value or "").strip()
    normalized = normalize_cache_exact_key(text)
    if not text or not normalized:
        return False
    if normalized in {"mg", "g", "%", "ml", "kg", "mcg", "μg", "ug"}:
        return True
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:mg|g|ml|kg|mcg|ug|%)?", normalized))


def resolve_joint_input_family(*values: str) -> str:
    for value in values:
        family = infer_joint_ingredient_family(str(value or ""))
        if family:
            return family
    return ""


def build_joint_family_category_row(functional_name: str, family: str) -> dict:
    return {
        "functional_ingredient_name": functional_name,
        "category_main": "관절/연골",
        "category_sub": JOINT_FAMILY_CATEGORY_SUB.get(str(family or ""), "관절/연골 보호 매핑"),
        "claim_text": "",
        "source": "protected_joint_family_fallback",
        "confidence": 0.76,
        "notes": "protected joint family fallback",
        "categories": [],
    }


def build_upload_signature(source_type: str, payload: str) -> str:
    normalized_payload = str(payload or "").strip()
    if not normalized_payload:
        return ""
    digest = hashlib.sha1(normalized_payload.encode("utf-8")).hexdigest()
    return f"{str(source_type or 'upload').strip().lower()}::{digest}"


def build_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha1(bytes(image_bytes or b"")).hexdigest() if image_bytes is not None else ""


def build_profile_signature(temp_profile: dict, temp_vector: Dict[str, float]) -> str:
    payload = {
        "product_main_category": str(temp_profile.get("product_main_category", "") or ""),
        "primary_ingredients": sorted(str(item or "") for item in temp_profile.get("primary_ingredients", []) if str(item or "")),
        "secondary_ingredients": sorted(str(item or "") for item in temp_profile.get("secondary_ingredients", []) if str(item or "")),
        "support_ingredients": sorted(str(item or "") for item in temp_profile.get("support_ingredients", []) if str(item or "")),
        "vector": {str(key): round(float(value), 6) for key, value in sorted(temp_vector.items()) if float(value or 0.0) > 0},
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def determine_upload_candidate_state(parsed: dict, estimated_profile: dict, needs_user_review: bool) -> dict:
    normalized_count = len(parsed.get("normalized_ingredients", []) or [])
    confidence = float(parsed.get("profile_confidence", parsed.get("confidence", 0.0)) or 0.0)
    quality_grade = str(parsed.get("quality_grade", "") or "")
    main_category = str(estimated_profile.get("product_main_category", "") or "")
    primary_normalized = [
        str(item or "").strip()
        for item in (
            parsed.get("primary_ingredients_normalized")
            or parsed.get("primary_ingredients")
            or estimated_profile.get("primary_ingredients", [])
            or []
        )
        if str(item or "").strip()
    ]
    primary_count = len(primary_normalized)
    critical_codes = {
        str(item.get("code", "") or "")
        for item in parsed.get("quality_warnings", []) or []
        if str(item.get("severity", "") or "") == "critical"
    }
    has_valid_category = bool(main_category and len(main_category.strip()) >= 2 and "?" not in main_category)
    soft_review_only = bool(
        needs_user_review
        and not critical_codes
        and confidence >= 0.7
        and primary_count >= 1
        and has_valid_category
    )
    candidate_disabled_reason = ""
    candidate_scope = "none"
    base_candidate_enabled = bool(
        normalized_count >= 2
        and confidence >= 0.7
        and (not needs_user_review or soft_review_only)
        and not critical_codes
        and quality_grade in {"", "A", "B"}
        and has_valid_category
    )
    candidate_enabled = bool(base_candidate_enabled)
    if candidate_enabled:
        candidate_scope = "uploaded_auto"
    elif normalized_count < 2:
        candidate_disabled_reason = "normalized_ingredients_too_few"
    elif confidence < 0.7:
        candidate_disabled_reason = "profile_confidence_below_threshold"
    elif needs_user_review and not soft_review_only:
        candidate_disabled_reason = "needs_user_review"
    elif critical_codes:
        candidate_disabled_reason = f"critical_warnings:{','.join(sorted(critical_codes))}"
    elif quality_grade not in {"", "A", "B"}:
        candidate_disabled_reason = f"quality_grade_{quality_grade.lower()}"
    elif not has_valid_category:
        candidate_disabled_reason = "missing_or_generic_main_category"
    else:
        candidate_disabled_reason = "policy_gate_unspecified"
    if (
        not candidate_enabled
        and normalized_count >= 1
        and primary_count >= 1
        and confidence >= 0.75
        and (not needs_user_review or soft_review_only)
        and not critical_codes
        and has_valid_category
    ):
        candidate_enabled = True
        candidate_scope = "uploaded_auto_notice_review" if soft_review_only else "uploaded_auto_single_core"
        candidate_disabled_reason = ""
    elif candidate_enabled and soft_review_only:
        candidate_scope = "uploaded_auto_notice_review"
    if not quality_grade:
        if candidate_enabled and confidence >= 0.9:
            quality_grade = "A"
        elif candidate_enabled:
            quality_grade = "B"
        elif normalized_count >= 1:
            quality_grade = "C"
        else:
            quality_grade = "D"
    elif candidate_enabled and quality_grade not in {"A", "B"}:
        quality_grade = "A" if confidence >= 0.9 else "B"
    elif not candidate_enabled and quality_grade in {"A", "B"}:
        quality_grade = "C" if normalized_count >= 1 else "D"
    if candidate_enabled:
        status = "verified" if normalized_count >= 2 and confidence >= 0.7 and not critical_codes and not soft_review_only else "auto_eligible"
    elif needs_user_review and not soft_review_only:
        status = "review_needed"
    else:
        status = "raw"
    return {
        "status": status,
        "quality_grade": quality_grade,
        "is_candidate_enabled": candidate_enabled,
        "candidate_scope": candidate_scope,
        "candidate_disabled_reason": candidate_disabled_reason,
    }


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
    elif code == "excipient_in_core_role" and ("비타민" in message or "혼합제" in message):
        item["severity"] = "warning"
    elif code in CRITICAL_WARNING_CODES:
        item["severity"] = "critical"
    elif severity in {"info", "notice", "warning", "critical"}:
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
        self._category_loose_map: Optional[Dict[str, dict]] = None
        self._runtime_rag_documents: Optional[List[dict]] = None
        self._runtime_rag_exact_map: Optional[Dict[str, dict]] = None
        self._embedding_index: Optional[IngredientEmbeddingIndex] = None

    def _ensure_loaded(self) -> None:
        self.recommendation_service.ensure_loaded()
        if self._category_map is not None:
            return
        runtime = self.recommendation_service.runtime
        category_map: Dict[str, dict] = {}
        category_loose_map: Dict[str, dict] = {}
        with sqlite_connection(runtime["sqlite_path"]) as conn:
            self._ensure_runtime_extension_tables(conn)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT functional_ingredient_name, category_main, category_sub, function_text, claim_text, source, confidence, notes, categories_json
                FROM functional_category_map
                """
            ).fetchall()
            custom_rows = conn.execute(
                f"""
                SELECT functional_ingredient_name, category_main, category_sub, function_text, claim_text, source, confidence, notes, categories_json
                FROM {RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME}
                """
            ).fetchall()
        for row in list(rows) + list(custom_rows):
            item = dict(row)
            item["categories"] = safe_json_loads(item.get("categories_json", "[]"), [])
            category_map[normalize_lookup_key(item["functional_ingredient_name"])] = item
            loose_key = normalize_cache_exact_key(item["functional_ingredient_name"])
            if loose_key and loose_key not in category_loose_map:
                category_loose_map[loose_key] = item
        self._category_map = category_map
        self._category_loose_map = category_loose_map

    def _expand_runtime_lookup_terms(self, *values: str) -> List[str]:
        suffixes = [
            "혼합제제",
            "복합제제",
            "혼합분말",
            "복합분말",
            "혼합물",
            "복합물",
            "제제",
            "분말",
            "농축분말",
            "추출분말",
            "추출물분말",
            "추출물",
            "추출액",
            "농축액",
        ]
        terms: List[str] = []
        for value in values:
            cleaned = str(value or "").strip()
            if not cleaned:
                continue
            terms.append(cleaned)
            canonical = canonicalize_ingredient_for_matching(cleaned)
            if canonical and canonical != cleaned:
                terms.append(canonical)
            for suffix in suffixes:
                if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                    stripped = cleaned[: -len(suffix)].strip(" ,;/")
                    if stripped:
                        terms.append(stripped)
                        canonical = canonicalize_ingredient_for_matching(stripped)
                        if canonical and canonical != stripped:
                            terms.append(canonical)
            for group in re.findall(r"\(([^)]+)\)", cleaned):
                for token in split_ingredients(group):
                    token = str(token or "").strip()
                    if not token:
                        continue
                    terms.append(token)
                    canonical = canonicalize_ingredient_for_matching(token)
                    if canonical and canonical != token:
                        terms.append(canonical)
        return dedupe_strings(terms)

    def _ensure_runtime_extension_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME} (
                functional_ingredient_name TEXT PRIMARY KEY,
                category_main TEXT,
                category_sub TEXT,
                function_text TEXT,
                claim_text TEXT,
                source TEXT,
                confidence REAL,
                notes TEXT,
                categories_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        try:
            conn.execute(f"ALTER TABLE {RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME} ADD COLUMN function_text TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME}_category_main ON {RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME}(category_main)"
        )
        conn.commit()

    def _reset_runtime_lookups(self) -> None:
        self._category_map = None
        self._category_loose_map = None
        self._runtime_rag_exact_map = None

    def _load_runtime_rag_documents(self) -> List[dict]:
        if self._runtime_rag_documents is not None:
            return self._runtime_rag_documents
        rag_path = Path(r"D:\db\deploy_ec2\data\functional_ingredient_rag_documents_merged_item_class_boosted.csv")
        synonym_path = Path(r"D:\db\deploy_ec2\data\functional_ingredient_synonym_dictionary_merged_item_class_boosted.csv")
        records: List[dict] = []
        if rag_path.exists():
            rag_df = pd.read_csv(rag_path, encoding="utf-8-sig", low_memory=False).fillna("")
            for row in rag_df.to_dict(orient="records"):
                records.append(
                    {
                        "standard_name": str(row.get("standard_name", "") or "").strip(),
                        "search_text": str(row.get("search_text", "") or ""),
                        "synonyms_joined": str(row.get("synonyms_joined", "") or ""),
                        "sources_joined": str(row.get("sources_joined", "") or ""),
                        "function_text": str(row.get("function_text", "") or ""),
                        "caution_text": str(row.get("caution_text", "") or ""),
                        "canonical_priority": float(row.get("canonical_priority", 0.0) or 0.0),
                        "specific_penalty": float(row.get("specific_penalty", 0.0) or 0.0),
                    }
                )
        if synonym_path.exists():
            synonym_df = pd.read_csv(synonym_path, encoding="utf-8-sig", low_memory=False).fillna("")
            grouped = synonym_df.groupby("standard_name", sort=False)
            for standard_name, group in grouped:
                synonyms = [
                    str(value).strip()
                    for value in group["synonym"].astype(str).tolist()
                    if str(value).strip()
                ]
                function_text = next((str(value).strip() for value in group["function_text"].astype(str).tolist() if str(value).strip()), "")
                caution_text = next((str(value).strip() for value in group["caution_text"].astype(str).tolist() if str(value).strip()), "")
                records.append(
                    {
                        "standard_name": str(standard_name or "").strip(),
                        "search_text": " ".join(synonyms),
                        "synonyms_joined": ", ".join(synonyms),
                        "sources_joined": "synonym_dictionary",
                        "function_text": function_text,
                        "caution_text": caution_text,
                        "canonical_priority": 0.0,
                        "specific_penalty": 0.0,
                    }
                )
        deduped: Dict[str, dict] = {}
        for row in records:
            standard_name = str(row.get("standard_name", "") or "").strip()
            if not standard_name:
                continue
            deduped.setdefault(standard_name, row)
        self._runtime_rag_documents = list(deduped.values())
        return self._runtime_rag_documents

    def _ensure_runtime_rag_exact_map(self) -> Dict[str, dict]:
        if self._runtime_rag_exact_map is not None:
            return self._runtime_rag_exact_map
        exact_map: Dict[str, dict] = {}
        for row in self._load_runtime_rag_documents():
            variants = [str(row.get("standard_name", "") or "").strip()]
            synonyms_joined = str(row.get("synonyms_joined", "") or "")
            if synonyms_joined:
                variants.extend([part.strip() for part in synonyms_joined.split(",") if part.strip()])
            for variant in self._expand_runtime_lookup_terms(*variants):
                key = normalize_cache_exact_key(variant)
                if not key:
                    continue
                existing = exact_map.get(key)
                if not existing:
                    exact_map[key] = row
                    continue
                existing_penalty = float(existing.get("specific_penalty", 0.0) or 0.0)
                new_penalty = float(row.get("specific_penalty", 0.0) or 0.0)
                existing_priority = float(existing.get("canonical_priority", 0.0) or 0.0)
                new_priority = float(row.get("canonical_priority", 0.0) or 0.0)
                current_rank = (existing_penalty, -existing_priority, len(str(existing.get("standard_name", "") or "")))
                new_rank = (new_penalty, -new_priority, len(str(row.get("standard_name", "") or "")))
                if new_rank < current_rank:
                    exact_map[key] = row
        self._runtime_rag_exact_map = exact_map
        return self._runtime_rag_exact_map

    def _lookup_runtime_rag_exact_candidate(self, *terms: str) -> Optional[dict]:
        exact_map = self._ensure_runtime_rag_exact_map()
        for term in self._expand_runtime_lookup_terms(*terms):
            key = normalize_cache_exact_key(term)
            if not key:
                continue
            row = exact_map.get(key)
            if row:
                return dict(row)
        return None

    def _build_runtime_rag_exact_payload(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        candidate_row: dict,
        input_family: str,
        mapping_rule: Optional[dict],
    ) -> Optional[dict]:
        standard_name = str(candidate_row.get("standard_name", "") or "").strip()
        if not standard_name or not self._is_allowed_by_mapping_rule(mapping_rule, standard_name):
            return None
        category_row = self._lookup_category_row(standard_name)
        if not category_row:
            return None
        resolved_name = str(category_row.get("functional_ingredient_name", "") or standard_name).strip()
        if not resolved_name or not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
            return None
        resolved_family = infer_joint_ingredient_family(resolved_name)
        if input_family and resolved_family and resolved_family != input_family:
            return None
        return self._build_match_payload(
            ingredient_name=ingredient_name,
            raw_ingredient=raw_ingredient,
            display_name=display_name,
            functional_ingredient_name=resolved_name,
            relation_type="same_ingredient",
            confidence=0.955,
            category_row=category_row,
            match_source="rag_synonym_exact",
            protected_family=input_family,
        )

    def _runtime_candidate_name_variants(self, candidate_row: dict) -> List[str]:
        variants = [str(candidate_row.get("standard_name", "") or "").strip()]
        synonyms_joined = str(candidate_row.get("synonyms_joined", "") or "")
        if synonyms_joined:
            variants.extend([part.strip() for part in synonyms_joined.split(",") if part.strip()])
        return dedupe_strings(variants)

    def _runtime_name_similarity(self, query: str, candidate_row: dict) -> float:
        query_terms = self._expand_runtime_lookup_terms(query)
        scores: List[float] = []
        for query_term in query_terms:
            for variant in self._runtime_candidate_name_variants(candidate_row):
                scores.append(normalized_name_similarity(query_term, variant))
        return round(max(scores or [0.0]), 6)

    def _has_runtime_lexical_overlap(self, query_terms: List[str], candidate_row: dict) -> bool:
        normalized_query_terms = [normalize_cache_exact_key(term) for term in query_terms if normalize_cache_exact_key(term)]
        if not normalized_query_terms:
            return False
        candidate_text = " ".join(
            [
                str(candidate_row.get("standard_name", "") or ""),
                str(candidate_row.get("synonyms_joined", "") or ""),
                str(candidate_row.get("search_text", "") or ""),
            ]
        )
        normalized_candidate = normalize_cache_exact_key(candidate_text)
        if not normalized_candidate:
            return False
        for term in normalized_query_terms:
            if len(term) >= 2 and term in normalized_candidate:
                return True
            if len(term) >= 4:
                for variant in self._runtime_candidate_name_variants(candidate_row):
                    if normalized_name_similarity(term, variant) >= RUNTIME_RAG_NAME_SIMILARITY_MIN:
                        return True
        query_tokens = {term for term in normalized_query_terms if len(term) >= 2}
        candidate_tokens = {
            normalize_cache_exact_key(token)
            for token in re.split(r"[^0-9A-Za-z가-힣]+", candidate_text.lower())
            if normalize_cache_exact_key(token)
        }
        return bool(query_tokens & candidate_tokens)

    def _get_embedding_index(self) -> Optional[IngredientEmbeddingIndex]:
        if self._embedding_index is not None:
            return self._embedding_index
        settings = get_settings()
        if not settings.ingredient_embedding_enabled:
            return None
        self._embedding_index = IngredientEmbeddingIndex(settings)
        return self._embedding_index

    def _build_runtime_embedding_query_text(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        input_family: str,
        mapping_rule: Optional[dict],
    ) -> str:
        term_candidates: List[str] = []
        for value in [display_name, raw_ingredient, ingredient_name]:
            cleaned = str(value or "").strip()
            if not cleaned:
                continue
            term_candidates.append(cleaned)
            canonical = canonicalize_ingredient_for_matching(cleaned)
            if canonical and canonical != cleaned:
                term_candidates.append(canonical)
        premix_source = str(display_name or raw_ingredient or ingredient_name or "").strip()
        if _looks_like_functional_premix(premix_source):
            inner_terms = re.findall(r"\(([^)]+)\)", premix_source)
            for group in inner_terms:
                for token in split_ingredients(group):
                    token = str(token or "").strip()
                    if not token:
                        continue
                    term_candidates.append(token)
                    canonical = canonicalize_ingredient_for_matching(token)
                    if canonical and canonical != token:
                        term_candidates.append(canonical)
        deduped_terms = dedupe_strings(term_candidates)
        parts = [
            f"입력 원료명: {display_name or raw_ingredient or ingredient_name}",
            f"원본 원료명: {raw_ingredient or display_name or ingredient_name}",
            f"정규화 원료명: {ingredient_name or display_name or raw_ingredient}",
            f"검색 토큰: {', '.join(deduped_terms)}",
        ]
        if input_family:
            parts.append(f"원료 계열 힌트: {input_family}")
        rule_name = str((mapping_rule or {}).get("name", "") or "").strip()
        if rule_name:
            parts.append(f"매핑 검증 규칙: {rule_name}")
        return "\n".join(part for part in parts if str(part).strip())

    def _search_runtime_embedding_candidates(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        input_family: str,
        mapping_rule: Optional[dict],
        top_k: int = RUNTIME_RAG_TOP_K,
    ) -> List[dict]:
        index = self._get_embedding_index()
        if not index:
            return []
        query_terms = self._expand_runtime_lookup_terms(display_name, raw_ingredient, ingredient_name)
        query_text = self._build_runtime_embedding_query_text(
            ingredient_name=ingredient_name,
            raw_ingredient=raw_ingredient,
            display_name=display_name,
            input_family=input_family,
            mapping_rule=mapping_rule,
        )
        try:
            results = index.search(query_text, top_k=max(top_k * 3, top_k))
        except Exception as exc:
            logger.warning("Ingredient embedding search failed: query=%r error=%s", query_text, exc)
            return []
        filtered = [row for row in results if self._has_runtime_lexical_overlap(query_terms, row)]
        return filtered[:top_k]

    def _merge_runtime_candidate_lists(self, *candidate_groups: List[dict], top_k: int = RUNTIME_RAG_TOP_K) -> List[dict]:
        merged: Dict[str, dict] = {}
        for group in candidate_groups:
            for item in group:
                standard_name = str(item.get("standard_name", "") or "").strip()
                if not standard_name:
                    continue
                existing = merged.get(standard_name)
                if not existing:
                    merged[standard_name] = dict(item)
                    continue
                existing["retrieval_score"] = max(
                    float(existing.get("retrieval_score", 0.0) or 0.0),
                    float(item.get("retrieval_score", 0.0) or 0.0),
                )
                existing["embedding_score"] = max(
                    float(existing.get("embedding_score", 0.0) or 0.0),
                    float(item.get("embedding_score", 0.0) or 0.0),
                )
                existing["name_similarity_score"] = max(
                    float(existing.get("name_similarity_score", 0.0) or 0.0),
                    float(item.get("name_similarity_score", 0.0) or 0.0),
                )
                sources = {
                    str(existing.get("retrieval_source", "") or "").strip(),
                    str(item.get("retrieval_source", "") or "").strip(),
                } - {""}
                if sources:
                    existing["retrieval_source"] = ",".join(sorted(sources))
        merged_rows = list(merged.values())
        merged_rows.sort(
            key=lambda row: (
                -float(row.get("retrieval_score", 0.0) or 0.0),
                -float(row.get("embedding_score", 0.0) or 0.0),
                -float(row.get("name_similarity_score", 0.0) or 0.0),
                float(row.get("specific_penalty", 0.0) or 0.0),
                str(row.get("standard_name", "") or ""),
            )
        )
        return merged_rows[:top_k]

    def _has_strong_runtime_rag_signal(self, candidates: List[dict]) -> bool:
        for item in candidates:
            if float(item.get("retrieval_score", 0.0) or 0.0) >= 2.0:
                return True
            if float(item.get("embedding_score", 0.0) or 0.0) >= 0.45:
                return True
            if float(item.get("name_similarity_score", 0.0) or 0.0) >= RUNTIME_RAG_NAME_SIMILARITY_MIN:
                return True
        return False

    def _log_trace(
        self,
        *,
        input_type: str,
        parsed: dict,
        estimated_profile: dict,
        recommendations: List[dict],
        execution_seconds: float,
        upload_signature: str = "",
        image_hash: str = "",
        ocr_text_hash: str = "",
        candidate_count: int = 0,
    ) -> str:
        trace_payload = {
            "input_type": input_type,
            "product_name_candidate": str(parsed.get("product_name_candidate", "") or ""),
            "upload_signature": str(upload_signature or ""),
            "image_hash": str(image_hash or ""),
            "ocr_text_hash": str(ocr_text_hash or ""),
            "parsed_signature": str((parsed.get("parse_metadata") or {}).get("parsed_signature", "") or compute_parsed_signature(parsed)),
            "profile_signature": str(estimated_profile.get("profile_signature", "") or ""),
            "top_recommendations": [
                {
                    "report_no": str(item.get("target_report_no", "") or ""),
                    "product_name": str(item.get("target_product_name", "") or ""),
                    "similarity_score": float(item.get("similarity_score", 0.0) or 0.0),
                }
                for item in recommendations[:10]
            ],
        }
        trace_id = hashlib.sha1(json.dumps(trace_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        warnings = list(parsed.get("quality_warnings", []) or [])
        metadata = {
            "needs_user_review": bool(parsed.get("needs_user_review")),
            "parse_metadata": parsed.get("parse_metadata", {}),
            "product_main_category": estimated_profile.get("product_main_category", ""),
        }
        self.recommendation_service.log_recommendation_trace(
            trace_id=trace_id,
            input_type=input_type,
            product_name=str(parsed.get("product_name_candidate", "") or estimated_profile.get("product_name", "") or ""),
            image_hash=image_hash,
            ocr_text_hash=ocr_text_hash,
            parsed_signature=str((parsed.get("parse_metadata") or {}).get("parsed_signature", "") or compute_parsed_signature(parsed)),
            profile_signature=str(estimated_profile.get("profile_signature", "") or ""),
            upload_signature=upload_signature,
            candidate_count=candidate_count,
            recommendations=recommendations[:20],
            warnings=warnings,
            metadata=metadata,
            execution_seconds=execution_seconds,
        )
        return trace_id

    def _lookup_category_row(self, functional_name: str) -> Optional[dict]:
        self._ensure_loaded()
        exact_key = normalize_lookup_key(functional_name)
        direct = self._category_map.get(exact_key)
        if direct:
            return direct
        loose_key = normalize_cache_exact_key(functional_name)
        if not loose_key:
            return None
        return self._category_loose_map.get(loose_key)

    def _score_runtime_rag_candidate(self, query: str, row: dict) -> float:
        query_text = str(query or "").strip()
        normalized_query = normalize_cache_exact_key(query_text)
        if not normalized_query:
            return 0.0
        standard_name = str(row.get("standard_name", "") or "").strip()
        search_text = str(row.get("search_text", "") or "")
        synonyms_joined = str(row.get("synonyms_joined", "") or "")
        normalized_standard = normalize_cache_exact_key(standard_name)
        normalized_search = normalize_cache_exact_key(search_text)
        normalized_synonyms = normalize_cache_exact_key(synonyms_joined)
        name_similarity_score = self._runtime_name_similarity(query_text, row)
        score = 0.0
        if normalized_standard == normalized_query:
            score += 12.0
        elif normalized_query in normalized_standard or normalized_standard in normalized_query:
            score += 7.0
        if normalized_query in normalized_search or normalized_query in normalized_synonyms:
            score += 6.0
        query_tokens = {token for token in re.split(r"[^0-9A-Za-z가-힣]+", query_text) if token}
        candidate_tokens = {
            token
            for token in re.split(r"[^0-9A-Za-z가-힣]+", f"{standard_name} {search_text} {synonyms_joined}")
            if token
        }
        shared_tokens = query_tokens & candidate_tokens
        score += float(len(shared_tokens)) * 1.7
        if len(normalized_query) >= 4 and normalized_standard and normalized_query[:4] == normalized_standard[:4]:
            score += 0.8
        if name_similarity_score >= RUNTIME_RAG_NAME_SIMILARITY_MIN:
            score += 4.8 * name_similarity_score
        score -= float(row.get("specific_penalty", 0.0) or 0.0) * 0.35
        return round(score, 6)

    def _search_runtime_rag_candidates(self, query: str, top_k: int = RUNTIME_RAG_TOP_K) -> List[dict]:
        rows = []
        for row in self._load_runtime_rag_documents():
            score = self._score_runtime_rag_candidate(query, row)
            if score <= 0:
                continue
            rows.append({**row, "retrieval_score": score, "name_similarity_score": self._runtime_name_similarity(query, row)})
        rows.sort(
            key=lambda item: (
                -float(item.get("retrieval_score", 0.0) or 0.0),
                float(item.get("specific_penalty", 0.0) or 0.0),
                str(item.get("standard_name", "") or ""),
            )
        )
        return rows[:top_k]

    def _classify_runtime_rag_candidates_with_llm(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        candidates: List[dict],
    ) -> dict:
        category_choices = sorted({str(key) for key in CATEGORY_HINT_RULES})
        payload = {
            "task": "입력 원료가 기존 표준 기능성 원료인지, 새로운 기능성 원료인지, 비기능성/판정불가인지 분류하라.",
            "rules": [
                "기존 표준 원료와 사실상 같은 뜻이면 decision=existing_match",
                "후보 중 하나로 보기 어렵지만 기능성 원료처럼 보이면 decision=new_standard",
                "부형제/광고문구/일반 문장/판정불가이면 decision=no_match",
                "existing_match일 때 matched_standard_name은 candidates 중 하나여야 한다",
                "JSON만 출력하라",
            ],
            "allowed_categories": category_choices,
            "output_schema": {
                "decision": "existing_match|new_standard|no_match",
                "matched_standard_name": "",
                "proposed_standard_name": "",
                "relation_type": "same_ingredient|ingredient_group|marker_compound|nutrient_form|same_function_only|unrelated",
                "confidence": 0.0,
                "reason": "",
                "category_main": "",
                "category_sub": "",
                "claim_text": "",
            },
            "input": {
                "ingredient_name": ingredient_name,
                "raw_ingredient": raw_ingredient,
                "display_name": display_name,
            },
            "candidates": [
                {
                    "standard_name": str(item.get("standard_name", "") or ""),
                    "retrieval_score": float(item.get("retrieval_score", 0.0) or 0.0),
                    "embedding_score": float(item.get("embedding_score", 0.0) or 0.0),
                    "retrieval_source": str(item.get("retrieval_source", "") or ""),
                    "sources_joined": str(item.get("sources_joined", "") or ""),
                    "synonyms_joined": str(item.get("synonyms_joined", "") or ""),
                    "function_text": str(item.get("function_text", "") or ""),
                    "caution_text": str(item.get("caution_text", "") or ""),
                }
                for item in candidates
            ],
        }
        content = call_local_llm(json.dumps(payload, ensure_ascii=False, indent=2))
        parsed = extract_json_from_llm_content(content)
        if not parsed:
            raise RuntimeError("runtime RAG LLM returned non-JSON content")
        parsed["_llm_raw_content"] = content
        return parsed

    def _generate_new_standard_metadata_with_llm(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        candidates: List[dict],
    ) -> dict:
        category_choices = sorted({str(key) for key in CATEGORY_HINT_RULES})
        payload = {
            "task": "Generate conservative metadata for a newly discovered functional ingredient candidate.",
            "rules": [
                "Use only information reasonably inferable from the ingredient name and nearby candidate evidence.",
                "Do not invent precise dosage or regulatory claims when uncertain.",
                "If the function is unclear, leave function_text and claim_text empty.",
                "Prefer short Korean descriptions suitable for internal review.",
                "Return JSON only.",
            ],
            "allowed_categories": category_choices,
            "output_schema": {
                "category_main": "",
                "category_sub": "",
                "function_text": "",
                "claim_text": "",
                "review_notes": "",
            },
            "input": {
                "ingredient_name": str(ingredient_name or ""),
                "raw_ingredient": str(raw_ingredient or ""),
                "display_name": str(display_name or ""),
            },
            "nearby_candidates": [
                {
                    "standard_name": str(item.get("standard_name", "") or ""),
                    "function_text": str(item.get("function_text", "") or ""),
                    "synonyms_joined": str(item.get("synonyms_joined", "") or ""),
                    "retrieval_source": str(item.get("retrieval_source", "") or ""),
                }
                for item in candidates[:5]
            ],
        }
        content = call_local_llm(json.dumps(payload, ensure_ascii=False, indent=2))
        parsed = extract_json_from_llm_content(content)
        if not parsed:
            raise RuntimeError("new standard metadata LLM returned non-JSON content")
        parsed["_llm_raw_content"] = content
        return parsed

    def _classify_runtime_rag_candidates_with_llm(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        candidates: List[dict],
    ) -> dict:
        category_choices = sorted({str(key) for key in CATEGORY_HINT_RULES})
        payload = {
            "task": "Classify whether the input ingredient is an existing standard functional ingredient, a new standard functional ingredient, or not a functional ingredient.",
            "rules": [
                "If the input is effectively the same ingredient as one of the candidates, use decision=existing_match.",
                "If none of the candidates are correct but the input still looks like a real functional ingredient, use decision=new_standard.",
                "If the input is an excipient, marketing phrase, generic sentence, dosage text, or unrelated text, use decision=no_match.",
                "When decision=existing_match, matched_standard_name must be exactly one of the candidate standard_name values.",
                "Prefer the candidate with the strongest name match and compatible function/category evidence.",
                "Return JSON only.",
            ],
            "allowed_categories": category_choices,
            "output_schema": {
                "decision": "existing_match|new_standard|no_match",
                "matched_standard_name": "",
                "proposed_standard_name": "",
                "relation_type": "same_ingredient|ingredient_group|marker_compound|nutrient_form|same_function_only|unrelated",
                "confidence": 0.0,
                "reason": "",
                "category_main": "",
                "category_sub": "",
                "claim_text": "",
            },
            "input": {
                "ingredient_name": str(ingredient_name or ""),
                "raw_ingredient": str(raw_ingredient or ""),
                "display_name": str(display_name or ""),
            },
            "candidates": [
                {
                    "standard_name": str(item.get("standard_name", "") or ""),
                    "retrieval_score": float(item.get("retrieval_score", 0.0) or 0.0),
                    "embedding_score": float(item.get("embedding_score", 0.0) or 0.0),
                    "retrieval_source": str(item.get("retrieval_source", "") or ""),
                    "sources_joined": str(item.get("sources_joined", "") or ""),
                    "synonyms_joined": str(item.get("synonyms_joined", "") or ""),
                    "function_text": str(item.get("function_text", "") or ""),
                    "caution_text": str(item.get("caution_text", "") or ""),
                }
                for item in candidates
            ],
        }
        content = call_local_llm(json.dumps(payload, ensure_ascii=False, indent=2))
        parsed = extract_json_from_llm_content(content)
        if not parsed:
            raise RuntimeError("runtime RAG LLM returned non-JSON content")
        parsed["_llm_raw_content"] = content
        return parsed

    def _save_runtime_cache_match(
        self,
        *,
        raw_ingredient: str,
        normalized_raw: str,
        matched_standard_name: str,
        relation_type: str,
        confidence: float,
        reason: str,
        rag_candidates: List[dict],
        gpt_payload: dict,
        match_method: str,
    ) -> None:
        runtime = self.recommendation_service.runtime
        with sqlite_connection(runtime["sqlite_path"]) as conn:
            self._ensure_runtime_extension_tables(conn)
            conn.execute(
                """
                INSERT INTO ingredient_match_cache (
                    raw_ingredient, normalized_raw, matched_standard_name, relation_type,
                    confidence, reason, rag_candidates_json, gpt_response_json,
                    created_at, updated_at, match_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """,
                (
                    str(raw_ingredient or ""),
                    str(normalized_raw or ""),
                    str(matched_standard_name or ""),
                    str(relation_type or ""),
                    float(confidence or 0.0),
                    str(reason or ""),
                    json.dumps(rag_candidates or [], ensure_ascii=False),
                    json.dumps(gpt_payload or {}, ensure_ascii=False),
                    str(match_method or ""),
                ),
            )
            conn.commit()

    def _save_runtime_custom_functional_ingredient(
        self,
        *,
        standard_name: str,
        category_main: str,
        category_sub: str,
        function_text: str,
        claim_text: str,
        confidence: float,
        notes: str,
    ) -> dict:
        runtime = self.recommendation_service.runtime
        normalized_standard = str(standard_name or "").strip()
        row = {
            "functional_ingredient_name": normalized_standard,
            "category_main": str(category_main or self._detect_category_from_ingredients([normalized_standard]) or "기타"),
            "category_sub": str(category_sub or ""),
            "function_text": str(function_text or ""),
            "claim_text": str(claim_text or ""),
            "source": "runtime_auto_provisional",
            "confidence": float(confidence or 0.0),
            "notes": str(notes or "runtime RAG+LLM provisional ingredient"),
            "categories_json": json.dumps([], ensure_ascii=False),
        }
        with sqlite_connection(runtime["sqlite_path"]) as conn:
            self._ensure_runtime_extension_tables(conn)
            conn.execute(
                f"""
                INSERT INTO {RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME} (
                    functional_ingredient_name, category_main, category_sub, function_text, claim_text,
                    source, confidence, notes, categories_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(functional_ingredient_name) DO UPDATE SET
                    category_main=excluded.category_main,
                    category_sub=excluded.category_sub,
                    function_text=excluded.function_text,
                    claim_text=excluded.claim_text,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    notes=excluded.notes,
                    categories_json=excluded.categories_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    row["functional_ingredient_name"],
                    row["category_main"],
                    row["category_sub"],
                    row["function_text"],
                    row["claim_text"],
                    row["source"],
                    row["confidence"],
                    row["notes"],
                    row["categories_json"],
                ),
            )
            conn.commit()
        try:
            from api.ops_service import get_ops_service

            get_ops_service().sync_pending_ingredients_from_runtime()
        except Exception:  # noqa: BLE001
            pass
        self._reset_runtime_lookups()
        self._ensure_loaded()
        return self._lookup_category_row(normalized_standard) or {
            **row,
            "categories": [],
        }

    def _is_allowed_by_mapping_rule(self, mapping_rule: Optional[dict], candidate_name: str) -> bool:
        if not mapping_rule:
            return True
        candidate = str(candidate_name or "").strip()
        if not candidate:
            return False
        if contains_normalized_fragment(candidate, mapping_rule.get("blocked", [])):
            return False
        allowed_fragments = list(mapping_rule.get("allowed", []))
        if not allowed_fragments:
            return True
        return contains_normalized_fragment(candidate, allowed_fragments)

    def _resolve_safe_alias_terms(
        self,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        mapping_rule: Optional[dict],
    ) -> List[str]:
        alias_terms: List[str] = []
        if mapping_rule:
            alias_terms.extend(mapping_rule.get("safe_aliases", []))
        alias_terms.extend(self._expand_runtime_lookup_terms(raw_ingredient, display_name, ingredient_name))
        return dedupe_strings(alias_terms)

    def _is_cache_like_vector_excluded(self, *values: str) -> bool:
        for value in values:
            if contains_guarded_runtime_term(value, RUNTIME_CACHE_LIKE_VECTOR_EXCLUDED_TERMS):
                return True
            if looks_like_runtime_measurement_or_symbol(value):
                return True
        return False

    def _is_excipient_vector_excluded(self, *values: str) -> bool:
        for value in values:
            if contains_guarded_runtime_term(value, RUNTIME_EXCIPIENT_VECTOR_EXCLUDED_TERMS):
                return True
        return False

    def _get_runtime_vector_exclusion_reason(self, item: dict) -> str:
        raw_input = str(item.get("raw_input") or item.get("raw_ingredient") or "")
        display_name = str(item.get("display_name") or raw_input or "")
        normalized_input = str(item.get("normalized_for_matching") or "")
        functional_name = str(item.get("functional_ingredient_name") or "")
        relation_type = str(item.get("relation_type") or "")
        match_source = str(item.get("match_source") or "")
        confidence = float(item.get("confidence", 0.0) or 0.0)
        functional_premix_input = _looks_like_functional_premix(raw_input) or _looks_like_functional_premix(display_name)

        if is_low_signal_upload_vector_term(raw_input, display_name, normalized_input):
            return "excluded_from_vector_by_low_signal_upload_guard"

        if self._is_excipient_vector_excluded(raw_input, display_name, normalized_input, functional_name) and not functional_premix_input:
            return "excluded_from_vector_by_excipient_guard"

        if relation_type in NON_FUNCTIONAL_CACHE_RELATION_TYPES:
            return "excluded_from_vector_by_relation_guard"

        high_confidence_exact_match = (
            confidence >= 0.90
            and relation_type in VECTOR_ALLOWED_RELATION_TYPES
            and match_source in {
                "direct_category",
                "safe_alias_exact",
                "cache_normalized_exact",
                "cache_raw_exact",
                "cache_matched_standard_exact",
                "rag_synonym_exact",
                "runtime_rag_llm_existing",
                "runtime_rag_llm_new",
            }
            and not looks_like_runtime_measurement_or_symbol(normalized_input)
            and not looks_like_runtime_measurement_or_symbol(functional_name)
        )
        if high_confidence_exact_match:
            return ""

        if self._is_cache_like_vector_excluded(raw_input, display_name, normalized_input) and not functional_premix_input:
            return "excluded_from_vector_by_cache_like_guard"

        if match_source == "cache_like":
            if confidence < 0.90:
                return "excluded_from_vector_by_cache_like_guard"
            if relation_type not in VECTOR_ALLOWED_RELATION_TYPES:
                return "excluded_from_vector_by_cache_like_guard"
            if bool(item.get("suspicious_mapping")):
                return "excluded_from_vector_by_cache_like_guard"

        return ""

    def _annotate_runtime_vector_usage(self, matches: List[dict]) -> List[dict]:
        for item in matches:
            reason = self._get_runtime_vector_exclusion_reason(item)
            item["vector_exclusion_reason"] = reason
            item["is_used_in_vector"] = not bool(reason)
            if reason:
                item["vector_exclusion_message"] = RUNTIME_VECTOR_EXCLUSION_REASON_MESSAGES.get(reason, reason)
        return matches

    def _fetch_cache_rows_by_normalized_raw(self, conn: sqlite3.Connection, normalized_raw: str) -> List[sqlite3.Row]:
        normalized = normalize_cache_exact_key(normalized_raw)
        if not normalized:
            return []
        return conn.execute(
            """
            SELECT raw_ingredient, normalized_raw, matched_standard_name, relation_type, confidence, match_method
            FROM ingredient_match_cache
            WHERE normalized_raw = ?
            ORDER BY confidence DESC, cache_id ASC
            LIMIT 20
            """,
            (normalized,),
        ).fetchall()

    def _fetch_cache_rows_by_raw(self, conn: sqlite3.Connection, raw_ingredient: str) -> List[sqlite3.Row]:
        cleaned = str(raw_ingredient or "").strip()
        if not cleaned:
            return []
        return conn.execute(
            """
            SELECT raw_ingredient, normalized_raw, matched_standard_name, relation_type, confidence, match_method
            FROM ingredient_match_cache
            WHERE raw_ingredient = ?
            ORDER BY confidence DESC, cache_id ASC
            LIMIT 20
            """,
            (cleaned,),
        ).fetchall()

    def _fetch_cache_rows_by_matched_standard(self, conn: sqlite3.Connection, standard_name: str) -> List[sqlite3.Row]:
        cleaned = str(standard_name or "").strip()
        if not cleaned:
            return []
        return conn.execute(
            """
            SELECT raw_ingredient, normalized_raw, matched_standard_name, relation_type, confidence, match_method
            FROM ingredient_match_cache
            WHERE matched_standard_name = ?
            ORDER BY confidence DESC, cache_id ASC
            LIMIT 20
            """,
            (cleaned,),
        ).fetchall()

    def _nonfunctional_cache_relation_types(self, rows: List[sqlite3.Row]) -> set[str]:
        return {
            str(row["relation_type"] or "").strip()
            for row in rows
            if str(row["relation_type"] or "").strip() in NON_FUNCTIONAL_CACHE_RELATION_TYPES
        }

    def _should_stop_after_nonfunctional_cache_hit(
        self,
        *,
        relation_types: set[str],
        normalized_input: str,
        raw_input: str,
        display_input: str,
    ) -> bool:
        if not relation_types:
            return False
        values = [raw_input, display_input, normalized_input]
        if any(looks_like_runtime_measurement_or_symbol(value) for value in values):
            return True
        if any(contains_guarded_runtime_term(value, RUNTIME_CACHE_LIKE_VECTOR_EXCLUDED_TERMS) for value in values):
            return True
        if "excipient" in relation_types:
            if any(
                is_excipient(value) or contains_guarded_runtime_term(value, RUNTIME_EXCIPIENT_VECTOR_EXCLUDED_TERMS)
                for value in values
            ):
                return True
        return False

    def _build_match_payload(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        functional_ingredient_name: str,
        relation_type: str,
        confidence: float,
        category_row: dict,
        match_source: str,
        protected_family: str = "",
        preserve_raw_in_profile: bool = False,
        suspicious_mapping: bool = False,
    ) -> dict:
        normalized_input = str(ingredient_name or "").strip()
        raw_input = str(raw_ingredient or normalized_input or display_name).strip()
        display_input = str(display_name or raw_input or normalized_input).strip()
        payload = {
            "raw_ingredient": raw_input or normalized_input,
            "functional_ingredient_name": str(functional_ingredient_name or raw_input or normalized_input),
            "relation_type": relation_type,
            "confidence": float(confidence or 0.0),
            "category_row": category_row,
            "match_source": match_source,
            "protected_family": str(protected_family or ""),
            "display_name": display_input or normalized_input,
            "normalized_for_matching": normalized_input,
            "profile_ingredient_name": str(functional_ingredient_name or normalized_input or display_input),
            "preserve_raw_in_profile": bool(preserve_raw_in_profile),
            "suspicious_mapping": bool(suspicious_mapping),
            "raw_input": raw_input,
        }
        if payload["preserve_raw_in_profile"] and normalized_input:
            payload["profile_ingredient_name"] = normalized_input
        return payload

    def _build_protected_family_fallback_match(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        input_family: str,
    ) -> Optional[dict]:
        family = str(input_family or "")
        if family not in PROTECTED_JOINT_FAMILIES:
            return None
        candidates = dedupe_strings([canonical_joint_family_name(family), *joint_family_aliases(family)])
        for candidate_name in candidates:
            category_row = self._lookup_category_row(candidate_name)
            if category_row:
                return self._build_match_payload(
                    ingredient_name=ingredient_name,
                    raw_ingredient=raw_ingredient,
                    display_name=display_name,
                    functional_ingredient_name=str(category_row["functional_ingredient_name"]),
                    relation_type="family_fallback",
                    confidence=0.76,
                    category_row=category_row,
                    match_source="protected_family_fallback",
                    protected_family=family,
                    preserve_raw_in_profile=True,
                )
        canonical_name = canonical_joint_family_name(family)
        if not canonical_name:
            return None
        return self._build_match_payload(
            ingredient_name=ingredient_name,
            raw_ingredient=raw_ingredient,
            display_name=display_name,
            functional_ingredient_name=canonical_name,
            relation_type="family_fallback",
            confidence=0.72,
            category_row=build_joint_family_category_row(canonical_name, family),
            match_source="protected_family_fallback",
            protected_family=family,
            preserve_raw_in_profile=True,
        )

    def _resolve_match_terms(self, ingredient_name: str, raw_ingredient: str, display_name: str, input_family: str) -> List[str]:
        terms = self._expand_runtime_lookup_terms(display_name, raw_ingredient, ingredient_name)
        if input_family:
            terms.extend(joint_family_aliases(input_family))
            terms.append(canonical_joint_family_name(input_family))
        return dedupe_strings(terms)

    def _fetch_cache_rows(self, conn: sqlite3.Connection, term: str, *, like: bool) -> List[sqlite3.Row]:
        cleaned = str(term or "").strip()
        if not cleaned:
            return []
        normalized = normalize_cache_exact_key(cleaned)
        if like:
            return conn.execute(
                """
                SELECT raw_ingredient, normalized_raw, matched_standard_name, relation_type, confidence, match_method
                FROM ingredient_match_cache
                WHERE normalized_raw LIKE ?
                   OR raw_ingredient LIKE ?
                ORDER BY confidence DESC, cache_id ASC
                LIMIT 20
                """,
                (f"%{normalized}%", f"%{cleaned}%"),
            ).fetchall()
        return []

    def _resolve_cache_row_match(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        row: sqlite3.Row,
        input_family: str,
        allow_like: bool,
        mapping_rule: Optional[dict],
        source_override: str = "",
    ) -> Optional[dict]:
        matched_standard = str(row["matched_standard_name"] or "").strip()
        matched_raw = str(row["raw_ingredient"] or "").strip()
        relation_type = str(row["relation_type"] or "cache_match")
        confidence = float(row["confidence"] or 0.0)
        if relation_type in NON_FUNCTIONAL_CACHE_RELATION_TYPES:
            return None

        candidate_names = []
        if matched_standard:
            candidate_names.append(matched_standard)
        if matched_raw and matched_raw not in candidate_names:
            candidate_names.append(matched_raw)

        for candidate_name in candidate_names:
            if not self._is_allowed_by_mapping_rule(mapping_rule, candidate_name):
                logger.warning(
                    "Rejected suspicious cache candidate: input=%r raw=%r candidate=%r rule=%s allow_like=%s",
                    ingredient_name,
                    raw_ingredient,
                    candidate_name,
                    str((mapping_rule or {}).get("name", "")),
                    allow_like,
                )
                continue
            family = infer_joint_ingredient_family(candidate_name)
            if input_family and family and family != input_family:
                logger.warning(
                    "Blocked suspicious ingredient mapping: input=%r raw=%r candidate=%r input_family=%s candidate_family=%s allow_like=%s",
                    ingredient_name,
                    raw_ingredient,
                    candidate_name,
                    input_family,
                    family,
                    allow_like,
                )
                continue
            if input_family and allow_like and not family:
                continue
            category_row = self._lookup_category_row(candidate_name)
            if not category_row and matched_standard and candidate_name != matched_standard:
                category_row = self._lookup_category_row(matched_standard)
            if not category_row:
                continue
            resolved_name = str(category_row["functional_ingredient_name"])
            if not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                logger.warning(
                    "Rejected suspicious resolved cache mapping: input=%r raw=%r resolved=%r rule=%s allow_like=%s",
                    ingredient_name,
                    raw_ingredient,
                    resolved_name,
                    str((mapping_rule or {}).get("name", "")),
                    allow_like,
                )
                continue
            resolved_family = infer_joint_ingredient_family(resolved_name)
            if input_family and resolved_family and resolved_family != input_family:
                logger.warning(
                    "Blocked suspicious resolved mapping: input=%r raw=%r resolved=%r input_family=%s candidate_family=%s allow_like=%s",
                    ingredient_name,
                    raw_ingredient,
                    resolved_name,
                    input_family,
                    resolved_family,
                    allow_like,
                )
                continue
            preserve_raw = bool(input_family and resolved_family == input_family and confidence < 0.8)
            return self._build_match_payload(
                ingredient_name=ingredient_name,
                raw_ingredient=raw_ingredient,
                display_name=display_name,
                functional_ingredient_name=resolved_name,
                relation_type=relation_type,
                confidence=confidence,
                category_row=category_row,
                match_source=source_override or ("cache_like" if allow_like else "cache_exact"),
                protected_family=input_family,
                preserve_raw_in_profile=preserve_raw,
            )
        return None

    def _try_runtime_rag_fallback(
        self,
        *,
        ingredient_name: str,
        raw_ingredient: str,
        display_name: str,
        input_family: str,
        mapping_rule: Optional[dict],
        allow_new_standard: bool = True,
        request_id: str = "",
    ) -> Optional[dict]:
        if not ENABLE_RUNTIME_RAG_FALLBACK:
            return None
        query = str(raw_ingredient or display_name or ingredient_name or "").strip()
        normalized_query = normalize_cache_exact_key(query)
        if not query or not normalized_query or len(normalized_query) < 2:
            return None
        if looks_like_runtime_measurement_or_symbol(query):
            return None
        if contains_guarded_runtime_term(query, RUNTIME_CACHE_LIKE_VECTOR_EXCLUDED_TERMS):
            return None
        if contains_guarded_runtime_term(query, RUNTIME_RAG_FALLBACK_SKIP_TERMS):
            return None

        update_request_progress(
            request_id,
            phase="ingredient_rag_retrieval",
            message="RAG 후보를 검색하는 중입니다.",
            detail=f"원료: {display_name or raw_ingredient or ingredient_name}",
            current_ingredient=display_name or raw_ingredient or ingredient_name,
        )
        embedding_results = self._search_runtime_embedding_candidates(
            ingredient_name=ingredient_name,
            raw_ingredient=raw_ingredient,
            display_name=display_name,
            input_family=input_family,
            mapping_rule=mapping_rule,
            top_k=max(RUNTIME_RAG_TOP_K, 8),
        )
        try:
            lexical_results = self._search_runtime_rag_candidates(query, top_k=max(RUNTIME_RAG_TOP_K, 5))
        except Exception as exc:
            logger.warning("Runtime RAG fallback retrieval failed: query=%r error=%s", query, exc)
            lexical_results = []

        results = self._merge_runtime_candidate_lists(embedding_results, lexical_results, top_k=max(RUNTIME_RAG_TOP_K, 8))
        if results and not self._has_strong_runtime_rag_signal(results):
            results = []
        if not results:
            update_request_progress(
                request_id,
                phase="ingredient_rag_no_candidate",
                message="RAG 후보가 없어 LLM 판정을 생략했습니다.",
                detail=f"원료: {display_name or raw_ingredient or ingredient_name}",
                current_ingredient=display_name or raw_ingredient or ingredient_name,
                extra={"rag_candidates": []},
            )
            return None

        candidate_names = [str(item.get("standard_name", "") or "") for item in results[:5]]
        update_request_progress(
            request_id,
            phase="ingredient_rag_llm",
            message="RAG 후보를 LLM으로 판정하는 중입니다.",
            detail=f"원료: {display_name or raw_ingredient or ingredient_name} / 후보: {', '.join(candidate_names) if candidate_names else '없음'}",
            current_ingredient=display_name or raw_ingredient or ingredient_name,
            extra={"rag_candidates": candidate_names},
        )
        try:
            llm_result = self._classify_runtime_rag_candidates_with_llm(
                ingredient_name=ingredient_name,
                raw_ingredient=raw_ingredient,
                display_name=display_name,
                candidates=results,
            )
        except Exception as exc:
            logger.warning("Runtime RAG fallback LLM classification failed: query=%r error=%s", query, exc)
            return None

        decision = str(llm_result.get("decision", "") or "").strip().lower()
        if decision == "no_match" or not decision:
            update_request_progress(
                request_id,
                phase="ingredient_rag_no_match",
                message="RAG 판정 결과 매칭 원료가 없습니다.",
                detail=f"원료: {display_name or raw_ingredient or ingredient_name}",
                current_ingredient=display_name or raw_ingredient or ingredient_name,
            )
            return None

        if decision == "existing_match":
            if not results:
                return None
            standard_name = str(llm_result.get("matched_standard_name", "") or "").strip()
            if not standard_name or not self._is_allowed_by_mapping_rule(mapping_rule, standard_name):
                return None
            category_row = self._lookup_category_row(standard_name)
            if not category_row:
                return None
            resolved_name = str(category_row["functional_ingredient_name"] or "").strip()
            if not resolved_name or not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                return None
            resolved_family = infer_joint_ingredient_family(resolved_name)
            if input_family and resolved_family and resolved_family != input_family:
                return None
            relation_type = str(llm_result.get("relation_type", "") or "same_ingredient").strip() or "same_ingredient"
            confidence = max(0.0, min(1.0, float(llm_result.get("confidence", 0.0) or 0.0)))
            if confidence < 0.6:
                confidence = 0.84
            reason = str(llm_result.get("reason", "") or "runtime RAG+LLM existing ingredient match").strip()
            update_request_progress(
                request_id,
                phase="ingredient_rag_match",
                message="RAG 판정으로 기능성 원료 매칭을 확정했습니다.",
                detail=f"{display_name or raw_ingredient or ingredient_name} -> {resolved_name}",
                current_ingredient=display_name or raw_ingredient or ingredient_name,
                extra={"matched_standard_name": resolved_name, "relation_type": relation_type, "confidence": confidence},
            )
            self._save_runtime_cache_match(
                raw_ingredient=query,
                normalized_raw=normalized_query,
                matched_standard_name=resolved_name,
                relation_type=relation_type,
                confidence=confidence,
                reason=reason,
                rag_candidates=results,
                gpt_payload=llm_result,
                match_method="runtime_rag_llm_existing",
            )
            return self._build_match_payload(
                ingredient_name=ingredient_name,
                raw_ingredient=raw_ingredient,
                display_name=display_name,
                functional_ingredient_name=resolved_name,
                relation_type=relation_type,
                confidence=confidence,
                category_row=category_row,
                match_source="runtime_rag_llm_existing",
                protected_family=input_family,
            )

        if decision != "new_standard" or not allow_new_standard:
            return None

        proposed_standard_name = str(
            llm_result.get("proposed_standard_name", "") or llm_result.get("matched_standard_name", "") or query
        ).strip()
        if not proposed_standard_name or not self._is_allowed_by_mapping_rule(mapping_rule, proposed_standard_name):
            return None
        proposed_family = infer_joint_ingredient_family(proposed_standard_name)
        if input_family and proposed_family and proposed_family != input_family:
            return None
        confidence = max(0.0, min(1.0, float(llm_result.get("confidence", 0.0) or 0.0)))
        if confidence < 0.7:
            confidence = 0.78
        reason = str(llm_result.get("reason", "") or "runtime RAG+LLM provisional new ingredient").strip()
        generated_metadata: dict = {}
        try:
            generated_metadata = self._generate_new_standard_metadata_with_llm(
                ingredient_name=ingredient_name,
                raw_ingredient=raw_ingredient,
                display_name=display_name,
                candidates=results,
            )
        except Exception as exc:
            logger.warning("New standard metadata generation failed: query=%r error=%s", query, exc)
            generated_metadata = {}
        category_main = str(llm_result.get("category_main", "") or generated_metadata.get("category_main", "") or "").strip()
        category_sub = str(llm_result.get("category_sub", "") or generated_metadata.get("category_sub", "") or "").strip()
        function_text = str(generated_metadata.get("function_text", "") or "").strip()
        claim_text = str(llm_result.get("claim_text", "") or generated_metadata.get("claim_text", "") or "").strip()
        review_notes = str(generated_metadata.get("review_notes", "") or "").strip()
        combined_notes = reason if not review_notes else f"{reason} | metadata: {review_notes}"
        llm_result["category_main"] = category_main
        llm_result["category_sub"] = category_sub
        llm_result["function_text"] = function_text
        llm_result["claim_text"] = claim_text
        if review_notes:
            llm_result["review_notes"] = review_notes
        category_row = self._save_runtime_custom_functional_ingredient(
            standard_name=proposed_standard_name,
            category_main=category_main,
            category_sub=category_sub,
            function_text=function_text,
            claim_text=claim_text,
            confidence=confidence,
            notes=combined_notes,
        )
        resolved_name = str(category_row.get("functional_ingredient_name", "") or proposed_standard_name).strip()
        self._save_runtime_cache_match(
            raw_ingredient=query,
            normalized_raw=normalized_query,
            matched_standard_name=resolved_name,
            relation_type="same_ingredient",
            confidence=confidence,
            reason=combined_notes,
            rag_candidates=results,
            gpt_payload=llm_result,
            match_method="runtime_rag_llm_new",
        )
        return self._build_match_payload(
            ingredient_name=ingredient_name,
            raw_ingredient=raw_ingredient,
            display_name=display_name,
            functional_ingredient_name=resolved_name,
            relation_type="same_ingredient",
            confidence=confidence,
            category_row=category_row,
            match_source="runtime_rag_llm_new",
            protected_family=input_family,
        )

    def infer_category_from_product_name(self, product_name: str) -> dict:
        name = str(product_name or "").strip()
        for category, rule in CATEGORY_HINT_RULES.items():
            if contains_keyword(name, rule["product_keywords"]):
                normalized_category = normalize_product_category_label(category) or category
                return {"category_main": normalized_category, "confidence": 0.95, "matched_keywords": rule["product_keywords"]}
        return {"category_main": "", "confidence": 0.0, "matched_keywords": []}

    def _infer_upload_category_hint(self, parsed_result: dict, fallback_product_name: str = "") -> dict:
        product_name = str(parsed_result.get("product_name_candidate") or fallback_product_name or "")
        product_name_hint = self.infer_category_from_product_name(product_name)
        name_category = normalize_product_category_label(product_name_hint.get("category_main", ""))
        if name_category:
            product_name_hint["category_main"] = name_category

        llm_category = normalize_product_category_label(parsed_result.get("product_main_category", ""))
        try:
            llm_confidence = max(0.0, min(1.0, float(parsed_result.get("product_category_confidence", 0.0) or 0.0)))
        except Exception:
            llm_confidence = 0.0

        if llm_category and llm_confidence >= 0.65 and (not name_category or name_category == llm_category):
            return {
                "category_main": llm_category,
                "confidence": max(float(product_name_hint.get("confidence", 0.0) or 0.0), llm_confidence),
                "matched_keywords": list(product_name_hint.get("matched_keywords", []) or []),
                "source": "llm_parse_category",
            }
        return product_name_hint

    def _has_non_joint_category_protection(self, parsed_result: dict, normalized_ingredients: List[str]) -> bool:
        llm_category = normalize_product_category_label(parsed_result.get("product_main_category", ""))
        try:
            llm_confidence = max(0.0, min(1.0, float(parsed_result.get("product_category_confidence", 0.0) or 0.0)))
        except Exception:
            llm_confidence = 0.0
        product_name = str(parsed_result.get("product_name_candidate", "") or "")
        name_category = normalize_product_category_label(self.infer_category_from_product_name(product_name).get("category_main", ""))
        nutrient_count = count_keyword_matches(normalized_ingredients, NUTRITION_SUPPLEMENT_INGREDIENT_KEYWORDS)
        protected_category = llm_category or name_category
        if not protected_category or protected_category == "관절/연골":
            if name_category == "뼈 건강" and nutrient_count >= 2:
                return True
            return False
        if llm_category and name_category and llm_category == name_category and llm_confidence >= 0.65:
            return True
        if protected_category == "뼈 건강" and nutrient_count >= 2 and name_category == "뼈 건강":
            return True
        return False

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
        protect_non_joint_category = self._has_non_joint_category_protection(parsed_result, normalized_ingredients)

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

        if self._is_joint_candidate(product_name_candidate, normalized_ingredients) and not protect_non_joint_category:
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
        if row.get("exact_same_upload"):
            return str(row.get("reason", "") or "")
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

    def _find_existing_db_match(self, ingredient_name: str, raw_ingredient: str = "", display_name: str = "") -> Optional[dict]:
        self._ensure_loaded()
        runtime = self.recommendation_service.runtime
        normalized_input = str(ingredient_name or "").strip()
        raw_input = str(raw_ingredient or normalized_input).strip()
        display_input = str(display_name or raw_input or normalized_input).strip()
        input_family = resolve_joint_input_family(display_input, raw_input, normalized_input)
        mapping_rule = resolve_runtime_mapping_rule(display_input, raw_input, normalized_input)
        direct_terms = self._resolve_match_terms(normalized_input, raw_input, display_input, input_family)
        safe_alias_terms = self._resolve_safe_alias_terms(normalized_input, raw_input, display_input, mapping_rule)
        normalized_raw_exact = normalize_cache_exact_key(normalized_input or raw_input or display_input)
        raw_normalized_exact = normalize_cache_exact_key(raw_input or display_input or normalized_input)

        def iter_unique_rows(rows: List[sqlite3.Row]) -> List[sqlite3.Row]:
            seen = set()
            unique_rows: List[sqlite3.Row] = []
            for row in rows:
                key = (
                    str(row["raw_ingredient"] or ""),
                    str(row["matched_standard_name"] or ""),
                    str(row["relation_type"] or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                unique_rows.append(row)
            return unique_rows

        with sqlite_connection(runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            for row in iter_unique_rows(self._fetch_cache_rows_by_normalized_raw(conn, normalized_raw_exact)):
                match = self._resolve_cache_row_match(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    row=row,
                    input_family=input_family,
                    allow_like=False,
                    mapping_rule=mapping_rule,
                    source_override="cache_normalized_exact",
                )
                if match:
                    return match

            if raw_normalized_exact and raw_normalized_exact != normalized_raw_exact:
                for row in iter_unique_rows(self._fetch_cache_rows_by_normalized_raw(conn, raw_normalized_exact)):
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=False,
                        mapping_rule=mapping_rule,
                        source_override="cache_raw_normalized_exact",
                    )
                    if match:
                        return match

            for row in iter_unique_rows(self._fetch_cache_rows_by_raw(conn, raw_input)):
                match = self._resolve_cache_row_match(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    row=row,
                    input_family=input_family,
                    allow_like=False,
                    mapping_rule=mapping_rule,
                    source_override="cache_raw_exact",
                )
                if match:
                    return match

            for direct_name in direct_terms:
                direct = self._lookup_category_row(direct_name)
                if not direct:
                    continue
                resolved_name = str(direct["functional_ingredient_name"] or "").strip()
                if not resolved_name or not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                    continue
                resolved_family = infer_joint_ingredient_family(resolved_name)
                if input_family and resolved_family and resolved_family != input_family:
                    continue
                return self._build_match_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    functional_ingredient_name=resolved_name,
                    relation_type="same_ingredient",
                    confidence=0.99 if input_family else 0.98,
                    category_row=direct,
                    match_source="direct_category",
                    protected_family=input_family,
                )

            for alias_term in safe_alias_terms:
                direct = self._lookup_category_row(alias_term)
                if not direct:
                    continue
                resolved_name = str(direct["functional_ingredient_name"] or "").strip()
                if not resolved_name or not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                    continue
                resolved_family = infer_joint_ingredient_family(resolved_name)
                if input_family and resolved_family and resolved_family != input_family:
                    continue
                return self._build_match_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    functional_ingredient_name=resolved_name,
                    relation_type="same_ingredient",
                    confidence=0.97 if mapping_rule else 0.96,
                    category_row=direct,
                    match_source="safe_alias_exact",
                    protected_family=input_family,
                )

            matched_standard_terms = dedupe_strings([*safe_alias_terms, *direct_terms])
            for term in matched_standard_terms:
                for row in iter_unique_rows(self._fetch_cache_rows_by_matched_standard(conn, term)):
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=False,
                        mapping_rule=mapping_rule,
                        source_override="cache_matched_standard_exact",
                    )
                    if match:
                        return match

            if not is_like_blocked_input(raw_input, display_input, normalized_input):
                like_rows: List[sqlite3.Row] = []
                seen_like = set()
                for term in direct_terms:
                    normalized_term = normalize_cache_exact_key(term)
                    if not normalized_term or is_like_blocked_input(term):
                        continue
                    if len(normalized_term) < 5:
                        continue
                    for row in self._fetch_cache_rows(conn, term, like=True):
                        key = (
                            str(row["raw_ingredient"] or ""),
                            str(row["matched_standard_name"] or ""),
                            str(row["relation_type"] or ""),
                        )
                        if key in seen_like:
                            continue
                        seen_like.add(key)
                        like_rows.append(row)
                for row in like_rows:
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=True,
                        mapping_rule=mapping_rule,
                    )
                    if match:
                        return match

        return None

    def lookup_existing_ingredient_db_match(self, ingredient_text: str) -> dict:
        raw_input = str(ingredient_text or "").strip()
        normalized_input = canonicalize_ingredient_for_matching(raw_input) if raw_input else ""
        normalized_input = normalized_input or raw_input
        ingredient_object = {
            "raw": raw_input,
            "display_name": raw_input,
            "normalized_for_matching": normalized_input,
            "role": "primary",
        }
        match = self._find_existing_db_match(normalized_input, raw_ingredient=raw_input, display_name=raw_input) if raw_input else None
        matches = [match] if match else []
        if matches:
            self._annotate_runtime_vector_usage(matches)
        status = self.build_ingredient_db_match_statuses([ingredient_object], matched_ingredients=matches)[0]
        return {
            "query": raw_input,
            "normalized_query": normalized_input,
            "matched": bool(match),
            "match": status,
        }

    def _find_match_in_cache(self, ingredient_name: str, raw_ingredient: str = "", display_name: str = "", request_id: str = "") -> Optional[dict]:
        # Match order:
        # 1) ingredient_match_cache.normalized_raw exact
        # 2) ingredient_match_cache.raw_ingredient exact
        # 3) functional_category_map exact
        # 4) safe alias exact / matched_standard_name exact
        # 5) restricted LIKE
        # 6) optional runtime RAG fallback (disabled by default)
        #
        # The previous bug with HCA came from step 5 running on a short token:
        #   normalized_raw LIKE '%hca%'
        # That pulled in a long mixed raw row already labeled as '녹차추출물',
        # so the online runtime resolved HCA -> 녹차추출물. Short abbreviations
        # are now blocked from LIKE matching and exact cache rows are checked first.
        self._ensure_loaded()
        runtime = self.recommendation_service.runtime
        normalized_input = str(ingredient_name or "").strip()
        raw_input = str(raw_ingredient or normalized_input).strip()
        display_input = str(display_name or raw_input or normalized_input).strip()
        input_family = resolve_joint_input_family(display_input, raw_input, normalized_input)
        mapping_rule = resolve_runtime_mapping_rule(display_input, raw_input, normalized_input)
        direct_terms = self._resolve_match_terms(normalized_input, raw_input, display_input, input_family)
        safe_alias_terms = self._resolve_safe_alias_terms(normalized_input, raw_input, display_input, mapping_rule)
        normalized_raw_exact = normalize_cache_exact_key(normalized_input or raw_input or display_input)
        raw_normalized_exact = normalize_cache_exact_key(raw_input or display_input or normalized_input)
        canonical_overrides_raw_cache = bool(
            normalized_raw_exact
            and raw_normalized_exact
            and normalized_raw_exact != raw_normalized_exact
            and self._lookup_category_row(normalized_input)
        )
        nonfunctional_cache_hit = False

        def iter_unique_rows(rows: List[sqlite3.Row]) -> List[sqlite3.Row]:
            seen = set()
            unique_rows: List[sqlite3.Row] = []
            for row in rows:
                key = (
                    str(row["raw_ingredient"] or ""),
                    str(row["matched_standard_name"] or ""),
                    str(row["relation_type"] or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                unique_rows.append(row)
            return unique_rows

        with sqlite_connection(runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            normalized_exact_rows = iter_unique_rows(self._fetch_cache_rows_by_normalized_raw(conn, normalized_raw_exact))
            for row in normalized_exact_rows:
                match = self._resolve_cache_row_match(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    row=row,
                    input_family=input_family,
                    allow_like=False,
                    mapping_rule=mapping_rule,
                    source_override="cache_normalized_exact",
                )
                if match:
                    return match
            normalized_negative_types = self._nonfunctional_cache_relation_types(normalized_exact_rows)
            if normalized_negative_types:
                nonfunctional_cache_hit = True
                if self._should_stop_after_nonfunctional_cache_hit(
                    relation_types=normalized_negative_types,
                    normalized_input=normalized_input,
                    raw_input=raw_input,
                    display_input=display_input,
                ):
                    logger.info(
                        "Stopping after normalized exact non-functional cache hit: input=%r raw=%r rows=%s",
                        normalized_input,
                        raw_input,
                        [str(row["relation_type"] or "").strip() for row in normalized_exact_rows],
                    )
                    return None
                logger.info(
                    "Revalidating stale normalized non-functional cache hit: input=%r raw=%r rows=%s",
                    normalized_input,
                    raw_input,
                    [str(row["relation_type"] or "").strip() for row in normalized_exact_rows],
                )

            if raw_normalized_exact and raw_normalized_exact != normalized_raw_exact:
                raw_normalized_exact_rows = iter_unique_rows(
                    self._fetch_cache_rows_by_normalized_raw(conn, raw_normalized_exact)
                )
                for row in raw_normalized_exact_rows:
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=False,
                        mapping_rule=mapping_rule,
                        source_override="cache_raw_normalized_exact",
                    )
                    if match:
                        return match
                raw_normalized_negative_types = self._nonfunctional_cache_relation_types(raw_normalized_exact_rows)
                if raw_normalized_negative_types:
                    nonfunctional_cache_hit = True
                    if canonical_overrides_raw_cache:
                        logger.info(
                            "Ignoring stale raw normalized non-functional cache because canonical input resolves directly: input=%r raw=%r rows=%s",
                            normalized_input,
                            raw_input,
                            [str(row["relation_type"] or "").strip() for row in raw_normalized_exact_rows],
                        )
                    elif self._should_stop_after_nonfunctional_cache_hit(
                        relation_types=raw_normalized_negative_types,
                        normalized_input=normalized_input,
                        raw_input=raw_input,
                        display_input=display_input,
                    ):
                        logger.info(
                            "Stopping after raw normalized non-functional cache hit: input=%r raw=%r rows=%s",
                            normalized_input,
                            raw_input,
                            [str(row["relation_type"] or "").strip() for row in raw_normalized_exact_rows],
                        )
                        return None
                    else:
                        logger.info(
                            "Revalidating stale raw normalized non-functional cache hit: input=%r raw=%r rows=%s",
                            normalized_input,
                            raw_input,
                            [str(row["relation_type"] or "").strip() for row in raw_normalized_exact_rows],
                        )

            raw_exact_rows = iter_unique_rows(self._fetch_cache_rows_by_raw(conn, raw_input))
            for row in raw_exact_rows:
                match = self._resolve_cache_row_match(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    row=row,
                    input_family=input_family,
                    allow_like=False,
                    mapping_rule=mapping_rule,
                    source_override="cache_raw_exact",
                )
                if match:
                    return match
            raw_negative_types = self._nonfunctional_cache_relation_types(raw_exact_rows)
            if raw_negative_types:
                nonfunctional_cache_hit = True
                if canonical_overrides_raw_cache:
                    logger.info(
                        "Ignoring stale raw exact non-functional cache because canonical input resolves directly: input=%r raw=%r rows=%s",
                        normalized_input,
                        raw_input,
                        [str(row["relation_type"] or "").strip() for row in raw_exact_rows],
                    )
                elif self._should_stop_after_nonfunctional_cache_hit(
                    relation_types=raw_negative_types,
                    normalized_input=normalized_input,
                    raw_input=raw_input,
                    display_input=display_input,
                ):
                    logger.info(
                        "Stopping after raw exact non-functional cache hit: input=%r raw=%r rows=%s",
                        normalized_input,
                        raw_input,
                        [str(row["relation_type"] or "").strip() for row in raw_exact_rows],
                    )
                    return None
                else:
                    logger.info(
                        "Revalidating stale raw exact non-functional cache hit: input=%r raw=%r rows=%s",
                        normalized_input,
                        raw_input,
                        [str(row["relation_type"] or "").strip() for row in raw_exact_rows],
                    )

            for direct_name in direct_terms:
                direct = self._lookup_category_row(direct_name)
                if not direct:
                    continue
                resolved_name = str(direct["functional_ingredient_name"])
                if not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                    logger.warning(
                        "Rejected suspicious direct mapping: input=%r raw=%r direct=%r resolved=%r rule=%s",
                        normalized_input,
                        raw_input,
                        direct_name,
                        resolved_name,
                        str((mapping_rule or {}).get("name", "")),
                    )
                    continue
                resolved_family = infer_joint_ingredient_family(resolved_name)
                if input_family and resolved_family and resolved_family != input_family:
                    logger.warning(
                        "Blocked direct suspicious mapping: input=%r raw=%r direct=%r input_family=%s candidate_family=%s",
                        normalized_input,
                        raw_input,
                        direct_name,
                        input_family,
                        resolved_family,
                    )
                    continue
                return self._build_match_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    functional_ingredient_name=resolved_name,
                    relation_type="same_ingredient",
                    confidence=0.99 if input_family else 0.98,
                    category_row=direct,
                    match_source="direct_category",
                    protected_family=input_family,
                )

            for alias_term in safe_alias_terms:
                direct = self._lookup_category_row(alias_term)
                if not direct:
                    continue
                resolved_name = str(direct["functional_ingredient_name"] or "")
                if not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                    continue
                resolved_family = infer_joint_ingredient_family(resolved_name)
                if input_family and resolved_family and resolved_family != input_family:
                    continue
                return self._build_match_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    functional_ingredient_name=resolved_name,
                    relation_type="same_ingredient",
                    confidence=0.97 if mapping_rule else 0.96,
                    category_row=direct,
                    match_source="safe_alias_exact",
                    protected_family=input_family,
                )

            rag_exact = self._lookup_runtime_rag_exact_candidate(display_input, raw_input, normalized_input)
            if rag_exact:
                rag_exact_match = self._build_runtime_rag_exact_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    candidate_row=rag_exact,
                    input_family=input_family,
                    mapping_rule=mapping_rule,
                )
                if rag_exact_match:
                    return rag_exact_match

            matched_standard_terms = dedupe_strings([*safe_alias_terms, *direct_terms])
            for term in matched_standard_terms:
                for row in iter_unique_rows(self._fetch_cache_rows_by_matched_standard(conn, term)):
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=False,
                        mapping_rule=mapping_rule,
                        source_override="cache_matched_standard_exact",
                    )
                    if match:
                        return match

            if not nonfunctional_cache_hit and not is_like_blocked_input(raw_input, display_input, normalized_input):
                like_rows: List[sqlite3.Row] = []
                seen_like = set()
                for term in direct_terms:
                    normalized_term = normalize_cache_exact_key(term)
                    if not normalized_term or is_like_blocked_input(term):
                        continue
                    if len(normalized_term) < 5:
                        continue
                    for row in self._fetch_cache_rows(conn, term, like=True):
                        key = (
                            str(row["raw_ingredient"] or ""),
                            str(row["matched_standard_name"] or ""),
                            str(row["relation_type"] or ""),
                        )
                        if key in seen_like:
                            continue
                        seen_like.add(key)
                        like_rows.append(row)
                for row in like_rows:
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=True,
                        mapping_rule=mapping_rule,
                    )
                    if match:
                        return match

        runtime_rag_match = self._try_runtime_rag_fallback(
            ingredient_name=normalized_input,
            raw_ingredient=raw_input,
            display_name=display_input,
            input_family=input_family,
            mapping_rule=mapping_rule,
            allow_new_standard=not nonfunctional_cache_hit,
            request_id=request_id,
        )
        if runtime_rag_match:
            return runtime_rag_match

        if input_family:
            fallback = self._build_protected_family_fallback_match(
                ingredient_name=normalized_input,
                raw_ingredient=raw_input,
                display_name=display_input,
                input_family=input_family,
            )
            if fallback:
                return fallback
        return None

    def debug_ingredient_match(self, ingredient_text: str) -> dict:
        self._ensure_loaded()
        raw_input = str(ingredient_text or "").strip()
        normalized_input = canonicalize_ingredient_for_matching(raw_input)
        display_input = raw_input
        input_family = resolve_joint_input_family(display_input, raw_input, normalized_input)
        mapping_rule = resolve_runtime_mapping_rule(display_input, raw_input, normalized_input)
        direct_terms = self._resolve_match_terms(normalized_input, raw_input, display_input, input_family)
        safe_alias_terms = self._resolve_safe_alias_terms(normalized_input, raw_input, display_input, mapping_rule)
        normalized_raw_exact = normalize_cache_exact_key(normalized_input or raw_input or display_input)

        debug = {
            "input": {
                "ingredient_text": normalized_input,
                "raw_input": raw_input,
                "normalized_input": normalized_input,
                "normalized_exact_key": normalized_raw_exact,
                "input_family": input_family,
                "mapping_rule": dict(mapping_rule or {}),
                "direct_terms": list(direct_terms),
                "safe_alias_terms": list(safe_alias_terms),
            },
            "guard_checks": {
                "looks_like_measurement_or_symbol": looks_like_runtime_measurement_or_symbol(normalized_input),
                "cache_like_excluded": self._is_cache_like_vector_excluded(normalized_input),
                "excipient_excluded": self._is_excipient_vector_excluded(normalized_input),
            },
            "exact_candidates": [],
            "direct_candidates": [],
            "safe_alias_candidates": [],
            "matched_standard_candidates": [],
            "like_candidates": [],
            "embedding_candidates": [],
            "lexical_candidates": [],
            "merged_candidates": [],
            "llm_result": None,
            "final_selection": None,
        }

        def serialize_match(match: dict, *, source: str) -> dict:
            category_row = dict(match.get("category_row", {}) or {})
            return {
                "source": source,
                "functional_ingredient_name": str(match.get("functional_ingredient_name", "") or ""),
                "relation_type": str(match.get("relation_type", "") or ""),
                "confidence": float(match.get("confidence", 0.0) or 0.0),
                "match_source": str(match.get("match_source", source) or source),
                "profile_ingredient_name": str(match.get("profile_ingredient_name", "") or ""),
                "category_main": str(category_row.get("category_main", "") or ""),
                "category_sub": str(category_row.get("category_sub", "") or ""),
                "function_text": str(category_row.get("function_text", "") or ""),
                "claim_text": str(category_row.get("claim_text", "") or ""),
            }

        def serialize_candidate(item: dict) -> dict:
            return {
                "standard_name": str(item.get("standard_name", "") or ""),
                "retrieval_score": float(item.get("retrieval_score", 0.0) or 0.0),
                "embedding_score": float(item.get("embedding_score", 0.0) or 0.0),
                "name_similarity_score": float(item.get("name_similarity_score", 0.0) or 0.0),
                "retrieval_source": str(item.get("retrieval_source", "") or ""),
                "sources_joined": str(item.get("sources_joined", "") or ""),
                "synonyms_joined": str(item.get("synonyms_joined", "") or ""),
                "function_text": str(item.get("function_text", "") or ""),
                "caution_text": str(item.get("caution_text", "") or ""),
            }

        def add_first_final(candidates: List[dict]) -> None:
            if candidates and not debug["final_selection"]:
                debug["final_selection"] = dict(candidates[0])

        def iter_unique_rows(rows: List[sqlite3.Row]) -> List[sqlite3.Row]:
            seen = set()
            unique_rows: List[sqlite3.Row] = []
            for row in rows:
                key = (
                    str(row["raw_ingredient"] or ""),
                    str(row["matched_standard_name"] or ""),
                    str(row["relation_type"] or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                unique_rows.append(row)
            return unique_rows

        runtime = self.recommendation_service.runtime
        with sqlite_connection(runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row

            normalized_exact_rows = iter_unique_rows(self._fetch_cache_rows_by_normalized_raw(conn, normalized_raw_exact))
            for row in normalized_exact_rows:
                match = self._resolve_cache_row_match(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    row=row,
                    input_family=input_family,
                    allow_like=False,
                    mapping_rule=mapping_rule,
                    source_override="cache_normalized_exact",
                )
                if match:
                    debug["exact_candidates"].append(serialize_match(match, source="cache_normalized_exact"))

            raw_exact_rows = iter_unique_rows(self._fetch_cache_rows_by_raw(conn, raw_input))
            for row in raw_exact_rows:
                match = self._resolve_cache_row_match(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    row=row,
                    input_family=input_family,
                    allow_like=False,
                    mapping_rule=mapping_rule,
                    source_override="cache_raw_exact",
                )
                if match:
                    debug["exact_candidates"].append(serialize_match(match, source="cache_raw_exact"))

            for direct_name in direct_terms:
                direct = self._lookup_category_row(direct_name)
                if not direct:
                    continue
                resolved_name = str(direct["functional_ingredient_name"] or "").strip()
                if not resolved_name or not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                    continue
                resolved_family = infer_joint_ingredient_family(resolved_name)
                if input_family and resolved_family and resolved_family != input_family:
                    continue
                payload = self._build_match_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    functional_ingredient_name=resolved_name,
                    relation_type="same_ingredient",
                    confidence=0.99 if input_family else 0.98,
                    category_row=direct,
                    match_source="direct_category",
                    protected_family=input_family,
                )
                debug["direct_candidates"].append(serialize_match(payload, source="direct_category"))

            for alias_term in safe_alias_terms:
                direct = self._lookup_category_row(alias_term)
                if not direct:
                    continue
                resolved_name = str(direct["functional_ingredient_name"] or "").strip()
                if not resolved_name or not self._is_allowed_by_mapping_rule(mapping_rule, resolved_name):
                    continue
                resolved_family = infer_joint_ingredient_family(resolved_name)
                if input_family and resolved_family and resolved_family != input_family:
                    continue
                payload = self._build_match_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    functional_ingredient_name=resolved_name,
                    relation_type="same_ingredient",
                    confidence=0.97 if mapping_rule else 0.96,
                    category_row=direct,
                    match_source="safe_alias_exact",
                    protected_family=input_family,
                )
                debug["safe_alias_candidates"].append(serialize_match(payload, source="safe_alias_exact"))

            rag_exact = self._lookup_runtime_rag_exact_candidate(display_input, raw_input, normalized_input)
            if rag_exact:
                rag_exact_match = self._build_runtime_rag_exact_payload(
                    ingredient_name=normalized_input,
                    raw_ingredient=raw_input,
                    display_name=display_input,
                    candidate_row=rag_exact,
                    input_family=input_family,
                    mapping_rule=mapping_rule,
                )
                if rag_exact_match:
                    debug["safe_alias_candidates"].append(
                        serialize_match(rag_exact_match, source="rag_synonym_exact")
                    )

            matched_standard_terms = dedupe_strings([*safe_alias_terms, *direct_terms])
            for term in matched_standard_terms:
                for row in iter_unique_rows(self._fetch_cache_rows_by_matched_standard(conn, term)):
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=False,
                        mapping_rule=mapping_rule,
                        source_override="cache_matched_standard_exact",
                    )
                    if match:
                        debug["matched_standard_candidates"].append(
                            serialize_match(match, source="cache_matched_standard_exact")
                        )

            if not is_like_blocked_input(raw_input, display_input, normalized_input):
                like_rows: List[sqlite3.Row] = []
                seen_like = set()
                for term in direct_terms:
                    normalized_term = normalize_cache_exact_key(term)
                    if not normalized_term or is_like_blocked_input(term):
                        continue
                    if len(normalized_term) < 5:
                        continue
                    for row in self._fetch_cache_rows(conn, term, like=True):
                        key = (
                            str(row["raw_ingredient"] or ""),
                            str(row["matched_standard_name"] or ""),
                            str(row["relation_type"] or ""),
                        )
                        if key in seen_like:
                            continue
                        seen_like.add(key)
                        like_rows.append(row)
                for row in like_rows:
                    match = self._resolve_cache_row_match(
                        ingredient_name=normalized_input,
                        raw_ingredient=raw_input,
                        display_name=display_input,
                        row=row,
                        input_family=input_family,
                        allow_like=True,
                        mapping_rule=mapping_rule,
                    )
                    if match:
                        debug["like_candidates"].append(serialize_match(match, source="cache_like"))

        add_first_final(debug["exact_candidates"])
        add_first_final(debug["direct_candidates"])
        add_first_final(debug["safe_alias_candidates"])
        add_first_final(debug["matched_standard_candidates"])
        add_first_final(debug["like_candidates"])

        embedding_results = self._search_runtime_embedding_candidates(
            ingredient_name=normalized_input,
            raw_ingredient=raw_input,
            display_name=display_input,
            input_family=input_family,
            mapping_rule=mapping_rule,
            top_k=max(RUNTIME_RAG_TOP_K, 8),
        )
        lexical_results = self._search_runtime_rag_candidates(normalized_input, top_k=max(RUNTIME_RAG_TOP_K, 5))
        merged_results = self._merge_runtime_candidate_lists(embedding_results, lexical_results, top_k=max(RUNTIME_RAG_TOP_K, 8))
        debug["embedding_candidates"] = [serialize_candidate(item) for item in embedding_results]
        debug["lexical_candidates"] = [serialize_candidate(item) for item in lexical_results]
        debug["merged_candidates"] = [serialize_candidate(item) for item in merged_results]

        try:
            llm_result = self._classify_runtime_rag_candidates_with_llm(
                ingredient_name=normalized_input,
                raw_ingredient=raw_input,
                display_name=display_input,
                candidates=merged_results,
            )
            debug["llm_result"] = llm_result
            if not debug["final_selection"]:
                decision = str(llm_result.get("decision", "") or "").strip().lower()
                if decision == "existing_match":
                    matched_standard_name = str(llm_result.get("matched_standard_name", "") or "").strip()
                    category_row = self._lookup_category_row(matched_standard_name) if matched_standard_name else None
                    if category_row:
                        payload = self._build_match_payload(
                            ingredient_name=normalized_input,
                            raw_ingredient=raw_input,
                            display_name=display_input,
                            functional_ingredient_name=str(category_row["functional_ingredient_name"] or ""),
                            relation_type=str(llm_result.get("relation_type", "") or "same_ingredient"),
                            confidence=float(llm_result.get("confidence", 0.0) or 0.0),
                            category_row=category_row,
                            match_source="runtime_rag_llm_existing",
                            protected_family=input_family,
                        )
                        debug["final_selection"] = serialize_match(payload, source="runtime_rag_llm_existing")
                elif decision == "new_standard":
                    final_name = str(
                        llm_result.get("proposed_standard_name", "")
                        or llm_result.get("matched_standard_name", "")
                        or normalized_input
                    )
                    metadata = {}
                    try:
                        metadata = self._generate_new_standard_metadata_with_llm(
                            ingredient_name=normalized_input,
                            raw_ingredient=raw_input,
                            display_name=display_input,
                            candidates=merged_results,
                        )
                    except Exception:
                        metadata = {}
                    debug["final_selection"] = {
                        "source": "runtime_rag_llm_new",
                        "functional_ingredient_name": final_name,
                        "relation_type": "same_ingredient",
                        "confidence": float(llm_result.get("confidence", 0.0) or 0.0),
                        "match_source": "runtime_rag_llm_new",
                        "profile_ingredient_name": final_name,
                        "category_main": str(llm_result.get("category_main", "") or metadata.get("category_main", "") or ""),
                        "category_sub": str(llm_result.get("category_sub", "") or metadata.get("category_sub", "") or ""),
                        "function_text": str(metadata.get("function_text", "") or ""),
                        "claim_text": str(llm_result.get("claim_text", "") or metadata.get("claim_text", "") or ""),
                    }
        except Exception as exc:  # noqa: BLE001
            debug["llm_result"] = {"error": str(exc)}

        return debug

    def match_raw_ingredients_to_functional_ingredients(self, ingredient_objects: List[dict], request_id: str = "") -> List[dict]:
        results = []
        seen = set()
        matchable_items = [item for item in ingredient_objects if item.get("role") != "excipient" and str(item.get("normalized_for_matching", "") or "").strip()]
        total = len(matchable_items)
        processed = 0
        for item in ingredient_objects:
            if item.get("role") == "excipient":
                continue
            normalized = str(item.get("normalized_for_matching", "") or "").strip()
            if not normalized:
                continue
            processed += 1
            raw_ingredient = str(item.get("raw", normalized) or normalized)
            display_name = str(item.get("display_name", raw_ingredient) or raw_ingredient)
            percent = 50 + (18 * (processed - 1) / max(total, 1))
            update_request_progress(
                request_id,
                phase="ingredient_matching",
                message="원료를 기능성 원료 DB와 매칭하는 중입니다.",
                percent=percent,
                detail=f"{processed}/{total}: {display_name}",
                current_ingredient=display_name,
                current=processed,
                total=total,
            )
            match = self._find_match_in_cache(normalized, raw_ingredient=raw_ingredient, display_name=display_name, request_id=request_id)
            if not match:
                update_request_progress(
                    request_id,
                    phase="ingredient_matching",
                    message="원료 매칭을 계속 진행 중입니다.",
                    percent=50 + (18 * processed / max(total, 1)),
                    detail=f"{display_name}: 매칭 없음",
                    current_ingredient=display_name,
                    current=processed,
                    total=total,
                )
                continue
            key = (str(match.get("raw_input", raw_ingredient)), match["functional_ingredient_name"])
            if key in seen:
                continue
            seen.add(key)
            match["display_name"] = display_name
            match["normalized_for_matching"] = normalized
            match["input_role"] = str(item.get("role", "") or "")
            match["raw_input"] = raw_ingredient
            results.append(match)
            update_request_progress(
                request_id,
                phase="ingredient_matching",
                message="원료 매칭을 계속 진행 중입니다.",
                percent=50 + (18 * processed / max(total, 1)),
                detail=f"{display_name} -> {match.get('functional_ingredient_name', '')} ({match.get('match_source', '')})",
                current_ingredient=display_name,
                current=processed,
                total=total,
                extra={"last_match_source": str(match.get("match_source", "") or "")},
            )
        return results

    def _semantic_detail_by_ingredient(self, estimated_profile: Optional[dict]) -> Dict[str, dict]:
        if not estimated_profile:
            return {}
        try:
            _, details = build_semantic_weight_vector(
                estimated_profile,
                self.recommendation_service.ingredient_category_profiles,
                include_details=True,
            )
        except Exception:  # noqa: BLE001
            return {}
        by_ingredient: Dict[str, dict] = {}
        for detail in details.values():
            if str(detail.get("semantic_key", "") or "").startswith("__"):
                continue
            sub_categories = detail.get("ingredient_sub_function_categories", []) or []
            if isinstance(sub_categories, str):
                sub_categories = [sub_categories] if sub_categories.strip() else []
            semantic_payload = {
                "semantic_key": str(detail.get("semantic_key", "") or ""),
                "ingredient_type": str(detail.get("ingredient_type", "") or ""),
                "profile_vector_include": bool(detail.get("vector_include", True)),
                "vector_include": bool(detail.get("vector_include", True)) and float(detail.get("weight", 0.0) or 0.0) > 0,
                "is_excipient": bool(detail.get("is_excipient", False)),
                "ingredient_main_category": str(detail.get("ingredient_main_category", "") or ""),
                "sub_function_categories": [str(item).strip() for item in sub_categories if str(item).strip()],
                "semantic_weight": float(detail.get("weight", 0.0) or 0.0),
                "semantic_weight_reason": str(detail.get("reason", "") or ""),
            }
            for ingredient in detail.get("ingredients", []) or []:
                ingredient_name = str(ingredient or "").strip()
                if ingredient_name:
                    by_ingredient[ingredient_name] = semantic_payload
        return by_ingredient

    def build_ingredient_db_match_statuses(
        self,
        ingredient_objects: List[dict],
        matched_ingredients: Optional[List[dict]] = None,
        estimated_profile: Optional[dict] = None,
    ) -> List[dict]:
        matches = matched_ingredients if matched_ingredients is not None else self.match_raw_ingredients_to_functional_ingredients(ingredient_objects)
        semantic_by_ingredient = self._semantic_detail_by_ingredient(estimated_profile)
        best_match_by_key: Dict[str, dict] = {}
        for match in matches:
            candidate_keys = {
                normalize_lookup_key(match.get("normalized_for_matching", "")),
                normalize_lookup_key(match.get("display_name", "")),
                normalize_lookup_key(match.get("raw_input", "")),
                normalize_lookup_key(match.get("raw_ingredient", "")),
            }
            candidate_keys.discard("")
            for key in candidate_keys:
                existing = best_match_by_key.get(key)
                if existing is None or float(match.get("confidence", 0.0) or 0.0) > float(existing.get("confidence", 0.0) or 0.0):
                    best_match_by_key[key] = match

        statuses: List[dict] = []
        for item in ingredient_objects:
            raw_value = str(item.get("raw", "") or "").strip()
            display_value = str(item.get("display_name", raw_value) or raw_value).strip()
            normalized_value = str(item.get("normalized_for_matching", "") or "").strip()
            input_role = str(item.get("role", "") or "").strip()
            category_hint = str(item.get("category_hint", "") or "").strip()
            keys = [
                normalize_lookup_key(normalized_value),
                normalize_lookup_key(display_value),
                normalize_lookup_key(raw_value),
            ]
            match = next((best_match_by_key[key] for key in keys if key and key in best_match_by_key), None)
            category_row = dict(match.get("category_row", {}) or {}) if match else {}
            functional_name = str(match.get("functional_ingredient_name", "") or "") if match else ""
            profile_name = str(match.get("profile_ingredient_name", "") or functional_name) if match else ""
            semantic_detail = (
                semantic_by_ingredient.get(profile_name)
                or semantic_by_ingredient.get(functional_name)
                or semantic_by_ingredient.get(display_value)
                or semantic_by_ingredient.get(normalized_value)
                or {}
            )
            if semantic_detail:
                ingredient_type = str(semantic_detail.get("ingredient_type", "") or "")
                vector_include = bool(semantic_detail.get("vector_include", True))
                is_excipient_value = bool(semantic_detail.get("is_excipient", False))
                ingredient_main_category = str(semantic_detail.get("ingredient_main_category", "") or "")
                sub_function_categories = list(semantic_detail.get("sub_function_categories", []) or [])
                semantic_weight = float(semantic_detail.get("semantic_weight", 0.0) or 0.0)
                semantic_weight_reason = str(semantic_detail.get("semantic_weight_reason", "") or "")
                semantic_key = str(semantic_detail.get("semantic_key", "") or "")
            else:
                vector_exclusion_reason = str(match.get("vector_exclusion_reason", "") or "") if match else ""
                is_excipient_value = input_role == "excipient" or is_excipient(display_value) or is_excipient(normalized_value)
                if vector_exclusion_reason == "excluded_from_vector_by_low_signal_upload_guard":
                    ingredient_type = "low_signal_food_base"
                else:
                    ingredient_type = "excipient" if is_excipient_value else ("functional" if match else "unmatched")
                vector_include = False
                ingredient_main_category = str(category_row.get("category_main", "") or "")
                sub_function_categories = []
                semantic_weight = 0.0
                semantic_weight_reason = "unmatched_ingredient" if not match else (vector_exclusion_reason or "semantic_detail_unavailable")
                if is_excipient_value:
                    semantic_weight_reason = vector_exclusion_reason or "excluded_non_functional"
                semantic_key = profile_name or functional_name or normalized_value or display_value
            statuses.append(
                {
                    "display_name": display_value or raw_value or normalized_value,
                    "raw_ingredient": raw_value or display_value or normalized_value,
                    "normalized_for_matching": normalized_value,
                    "input_role": input_role,
                    "category_hint": category_hint,
                    "is_functional_match": bool(match),
                    "functional_match": bool(match),
                    "functional_ingredient_name": functional_name,
                    "relation_type": str(match.get("relation_type", "") or "") if match else "",
                    "confidence": float(match.get("confidence", 0.0) or 0.0) if match else 0.0,
                    "category_main": str(category_row.get("category_main", "") or ""),
                    "category_sub": str(category_row.get("category_sub", "") or ""),
                    "ingredient_type": ingredient_type,
                    "vector_include": vector_include,
                    "is_excipient": is_excipient_value,
                    "ingredient_main_category": ingredient_main_category,
                    "sub_function_categories": sub_function_categories,
                    "semantic_key": semantic_key,
                    "semantic_weight": semantic_weight,
                    "semantic_weight_reason": semantic_weight_reason,
                    "function_text": str(category_row.get("function_text", "") or ""),
                    "claim_text": str(category_row.get("claim_text", "") or ""),
                    "match_source": str(match.get("match_source", "") or "") if match else "",
                }
            )
        return statuses

    def build_temp_product_vector_from_ingredients(
        self,
        ingredient_objects: List[dict],
        matched_ingredients: Optional[List[dict]] = None,
    ) -> Dict[str, float]:
        vector: Dict[str, float] = {}
        matches = matched_ingredients if matched_ingredients is not None else self.match_raw_ingredients_to_functional_ingredients(ingredient_objects)
        self._annotate_runtime_vector_usage(matches)
        for item in matches:
            if not bool(item.get("is_used_in_vector", True)):
                continue
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
            functional_name = str(item.get("profile_ingredient_name") or item["functional_ingredient_name"])
            if not bool(item.get("is_used_in_vector", True)):
                if item.get("vector_exclusion_reason") == "excluded_from_vector_by_excipient_guard":
                    excipient_ingredients.append(display_name)
                continue
            role = str(item.get("input_role", "secondary") or "secondary")
            if is_excluded_primary(display_name) or is_excipient(display_name):
                excipient_ingredients.append(display_name)
                continue
            category_row = item.get("category_row") or {}
            category_main = str(category_row.get("category_main", "기타") or "기타")
            category_sub = str(category_row.get("category_sub", "") or "")
            confidence = float(item.get("confidence", 0.75) or 0.75)
            if item.get("preserve_raw_in_profile") and str(item.get("normalized_for_matching", "") or "").strip():
                functional_name = str(item.get("normalized_for_matching") or functional_name)
            elif (item.get("suspicious_mapping") or confidence < 0.7) and str(item.get("normalized_for_matching", "") or "").strip():
                functional_name = str(item.get("normalized_for_matching") or functional_name)
            weight = max(0.35, min(1.0, confidence)) * ROLE_WEIGHT.get(role, 0.55)
            category_scores[category_main] = category_scores.get(category_main, 0.0) + weight
            ingredient_scores.append(
                {
                    "ingredient": functional_name,
                    "display_name": display_name,
                    "normalized_for_matching": str(item.get("normalized_for_matching", functional_name)),
                    "weight": round(weight, 4),
                    "match_confidence": round(confidence, 4),
                    "role": role,
                    "category_main": category_main,
                    "category_sub": category_sub,
                    "relation_type": str(item.get("relation_type", "") or ""),
                    "v2_decision": "family_signal" if str(item.get("relation_type", "") or "") == "family_fallback" else "existing_match",
                    "source_raw_ingredients": str(item.get("raw_input", "") or item.get("raw_ingredient", "") or display_name),
                    "source_match_methods": str(item.get("match_source", "") or ""),
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
            "llm_sub_function_categories": sub_categories,
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
        estimated_profile["confidence"] = float(
            parsed_result.get("profile_confidence", estimated_profile.get("confidence", parsed_result.get("confidence", 0.0))) or 0.0
        )
        base_warnings = []
        for warning in parsed_result.get("quality_warnings", []):
            normalized = normalize_warning_severity(warning)
            if normalized.get("code") in {"too_many_normalized_ingredients", "ocr_confidence_unavailable"}:
                continue
            base_warnings.append(normalized)
        quality_warnings = list(base_warnings)
        excluded_ingredients = list(dict.fromkeys(parsed_result.get("excluded_ingredients", []) + estimated_profile.get("excipient_ingredients", [])))
        product_name_candidate = str(parsed_result.get("product_name_candidate", "") or "")
        product_name_hint = self._infer_upload_category_hint(parsed_result, product_name_candidate)
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

    def calculate_similar_products_for_temp_vector(
        self,
        temp_vector: Dict[str, float],
        temp_profile: dict,
        top_k: int,
        candidate_limit: int,
        request_id: str = "",
    ) -> List[dict]:
        service = self.recommendation_service
        temp_product_id = str(temp_profile["product_id"])
        temp_upload_signature = str(temp_profile.get("upload_signature", "") or "")
        temp_profile_signature = str(temp_profile.get("profile_signature", "") or "")
        temp_ocr_text_hash = str(temp_profile.get("ocr_text_hash", "") or "")
        update_request_progress(
            request_id,
            phase="candidate_pool",
            message="후보 제품군을 추출하는 중입니다.",
            percent=74,
            detail=f"candidate_limit={candidate_limit}",
        )
        candidate_pool = build_candidate_pool_v2(
            temp_product_id,
            temp_profile,
            service.product_vectors,
            service.profiles,
            service.ingredient_postings,
            service.ingredient_frequency,
            candidate_limit,
            get_settings().default_max_df_for_seed,
            service.ingredient_category_profiles,
        )
        update_request_progress(
            request_id,
            phase="candidate_scoring",
            message="후보 제품별 semantic weighted Jaccard v2 유사도를 계산하는 중입니다.",
            percent=80,
            detail=f"후보 {len(candidate_pool)}개를 계산합니다.",
            current=0,
            total=len(candidate_pool),
        )
        candidate_target_ids = [str(candidate.get("target_product_id", "") or "") for candidate in candidate_pool]
        if temp_upload_signature:
            exact_upload_target_ids = [
                str(product_id)
                for product_id, profile in service.profiles.items()
                if str(product_id or "") != temp_product_id
                and str((profile or {}).get("upload_signature", "") or "") == temp_upload_signature
            ]
            for exact_target_id in exact_upload_target_ids:
                if exact_target_id and exact_target_id not in candidate_target_ids:
                    candidate_pool.insert(0, {"target_product_id": exact_target_id})
                    candidate_target_ids.insert(0, exact_target_id)

        rows: List[dict] = []
        total_candidates = len(candidate_pool)
        progress_interval = max(1, total_candidates // 20) if total_candidates else 1
        for index, candidate in enumerate(candidate_pool, start=1):
            if index == 1 or index == total_candidates or index % progress_interval == 0:
                update_request_progress(
                    request_id,
                    phase="candidate_scoring",
                    message="후보 제품별 semantic weighted Jaccard v2 유사도를 계산하는 중입니다.",
                    percent=80 + (12 * index / max(total_candidates, 1)),
                    detail=f"{index}/{total_candidates} 후보 처리 중",
                    current=index,
                    total=total_candidates,
                )
            target_product_id = candidate["target_product_id"]
            target_profile = service.profiles[target_product_id]
            target_vector = service.product_vectors[target_product_id]
            comparison = compare_product_profiles(temp_profile, target_profile, temp_vector, target_vector)
            target_upload_signature = str(target_profile.get("upload_signature", "") or "")
            target_profile_signature = str(target_profile.get("profile_signature", "") or "")
            target_ocr_text_hash = str(target_profile.get("ocr_text_hash", "") or "")
            exact_same_upload = bool(temp_upload_signature) and temp_upload_signature == target_upload_signature
            candidate_enabled = bool(target_profile.get("is_candidate_enabled", True))
            if not exact_same_upload and not candidate_enabled:
                continue
            semantic_explanation = {}

            uploaded_match_types: List[str] = []
            uploaded_match_labels: List[str] = []
            if str(target_profile.get("report_no", "") or "").startswith("UPLOADED-"):
                if exact_same_upload:
                    uploaded_match_types.append("exact_same_upload")
                    uploaded_match_labels.append("동일 업로드")
                if temp_profile_signature and target_profile_signature and temp_profile_signature == target_profile_signature:
                    uploaded_match_types.append("same_profile_signature")
                    uploaded_match_labels.append("동일 프로필")
                if temp_ocr_text_hash and target_ocr_text_hash and temp_ocr_text_hash == target_ocr_text_hash:
                    uploaded_match_types.append("same_ocr_text")
                    uploaded_match_labels.append("동일 OCR 텍스트")
                if not uploaded_match_types:
                    uploaded_match_types.append("similar_uploaded_case")
                    uploaded_match_labels.append("유사 업로드 사례")

            if exact_same_upload:
                all_ingredients = sorted(
                    set(comparison["shared_ingredients"])
                    | set(comparison["base_only_ingredients"])
                    | set(comparison["target_only_ingredients"])
                )
                all_categories = sorted(set(comparison["shared_categories"]) | set(comparison["different_categories"]))
                comparison = {
                    **comparison,
                    "shared_ingredients": all_ingredients,
                    "base_only_ingredients": [],
                    "target_only_ingredients": [],
                    "shared_categories": all_categories,
                    "different_categories": [],
                }
                similarity_score = 1.0
                function_similarity_score = 1.0
                core_match_score = 1.0
                substitutability = "높음"
                quality_metadata = recommendation_quality_metadata(similarity_score, {}, exact_match=True)
                reason = "이전에 업로드된 동일 이미지와 일치해 동일 제품으로 판정했습니다."
            else:
                similarity_score, shared_semantic_ingredients, semantic_explanation = calculate_semantic_weighted_jaccard_v2(
                    temp_profile,
                    target_profile,
                    service.ingredient_category_profiles,
                )
                if similarity_score <= 0:
                    continue
                if shared_semantic_ingredients:
                    comparison["shared_ingredients"] = shared_semantic_ingredients
                    comparison["shared_ingredients_raw"] = shared_semantic_ingredients
                    base_semantic_shared = set()
                    target_semantic_shared = set()
                    for detail in semantic_explanation.get("shared_semantic_keys", []) or []:
                        base_semantic_shared.update(str(name) for name in detail.get("base_ingredients", []) or [])
                        target_semantic_shared.update(str(name) for name in detail.get("target_ingredients", []) or [])
                    comparison["base_only_ingredients"] = [
                        name for name in comparison["base_only_ingredients"] if name not in base_semantic_shared
                    ]
                    comparison["target_only_ingredients"] = [
                        name for name in comparison["target_only_ingredients"] if name not in target_semantic_shared
                    ]
                    comparison["base_only_ingredients"] = [
                        name for name in comparison["base_only_ingredients"] if not is_semantic_excipient_name(name)
                    ]
                    comparison["target_only_ingredients"] = [
                        name for name in comparison["target_only_ingredients"] if not is_semantic_excipient_name(name)
                    ]
                function_similarity_score = calculate_function_similarity(temp_profile, target_profile)
                core_match_score = calculate_core_match_score(temp_profile, target_profile, semantic_explanation)
                substitutability = classify_substitutability(similarity_score, temp_profile, target_profile, semantic_explanation)
                quality_metadata = recommendation_quality_metadata(
                    similarity_score,
                    semantic_explanation,
                    core_match_score,
                    function_similarity_score,
                )
                reason = generate_recommendation_reason(temp_profile, target_profile, comparison, service.ingredient_frequency, semantic_explanation)
            explanation = build_explanation_json(reason, comparison, substitutability)
            explanation["similarity_algorithm"] = SIMILARITY_ALGORITHM_VERSION
            if semantic_explanation:
                explanation["semantic_weighted_jaccard_v2"] = semantic_explanation
            target_primary_ingredients = list(target_profile.get("primary_ingredients", []))
            target_secondary_ingredients = list(target_profile.get("secondary_ingredients", []))
            target_support_ingredients = list(target_profile.get("support_ingredients", []))
            target_all_ingredients = list(
                dict.fromkeys(target_primary_ingredients + target_secondary_ingredients + target_support_ingredients)
            )
            semantic_shared_target_ingredients = set()
            for detail in semantic_explanation.get("shared_semantic_keys", []) or []:
                semantic_shared_target_ingredients.update(str(name) for name in detail.get("target_ingredients", []) or [])
            target_other_ingredients = [
                item
                for item in target_all_ingredients
                if item not in comparison["shared_ingredients"]
                and item not in semantic_shared_target_ingredients
                and (not semantic_explanation or not is_semantic_excipient_name(item))
            ]
            rows.append(
                {
                    "rank": 0,
                    "report_no": str(target_profile.get("report_no", "") or ""),
                    "target_report_no": str(target_profile.get("report_no", "") or ""),
                    "product_name": str(target_profile.get("product_name", "") or ""),
                    "target_product_name": str(target_profile.get("product_name", "") or ""),
                    "target_product_main_category": str(target_profile.get("product_main_category", "") or ""),
                    "similarity_score": float(similarity_score),
                    "function_similarity_score": float(function_similarity_score),
                    "core_match_score": float(core_match_score),
                    "substitutability": substitutability,
                    **quality_metadata,
                    "shared_ingredients": comparison["shared_ingredients"],
                    "target_primary_ingredients": target_primary_ingredients,
                    "target_secondary_ingredients": target_secondary_ingredients,
                    "target_support_ingredients": target_support_ingredients,
                    "target_all_ingredients": target_all_ingredients,
                    "target_other_ingredients": target_other_ingredients,
                    "shared_categories": comparison["shared_categories"],
                    "reason": reason,
                    "caution": generate_caution(),
                    "explanation": explanation,
                    "exact_same_upload": exact_same_upload,
                    "profile_signature": str(target_profile.get("profile_signature", "") or ""),
                    "uploaded_match_types": uploaded_match_types,
                    "uploaded_match_labels": uploaded_match_labels,
                    "uploaded_status": str(target_profile.get("status", "") or ""),
                    "uploaded_quality_grade": str(target_profile.get("quality_grade", "") or ""),
                }
            )
        rows = sorted(
            rows,
            key=lambda item: (
                -int(bool(item.get("exact_same_upload"))),
                -item["similarity_score"],
                -item["core_match_score"],
                -item["function_similarity_score"],
                item["target_report_no"],
            ),
        )

        exact_rows: List[dict] = []
        uploaded_rows: List[dict] = []
        official_rows: List[dict] = []
        exact_profile_signatures = set()
        seen_uploaded_profile_signatures = set()
        seen_uploaded_report_nos = set()
        for item in rows:
            report_no = str(item.get("report_no", "") or "")
            if bool(item.get("exact_same_upload")):
                exact_rows.append(item)
                exact_profile_signature = str(item.get("profile_signature", "") or "")
                if exact_profile_signature:
                    exact_profile_signatures.add(exact_profile_signature)
                continue
            if report_no.startswith("UPLOADED-"):
                profile_signature = str(item.get("profile_signature", "") or "")
                dedupe_key = profile_signature or report_no
                if profile_signature and profile_signature in exact_profile_signatures:
                    continue
                if dedupe_key in seen_uploaded_profile_signatures or report_no in seen_uploaded_report_nos:
                    continue
                seen_uploaded_profile_signatures.add(dedupe_key)
                seen_uploaded_report_nos.add(report_no)
                uploaded_rows.append(item)
            else:
                if bool(item.get("recommendation_display_eligible", True)):
                    official_rows.append(item)

        uploaded_debug_rows: List[dict] = []
        seen_uploaded_debug_keys = set()
        for item in exact_rows + uploaded_rows:
            report_no = str(item.get("report_no", "") or "")
            profile_signature = str(item.get("profile_signature", "") or "")
            dedupe_key = profile_signature or report_no
            if dedupe_key in seen_uploaded_debug_keys:
                continue
            seen_uploaded_debug_keys.add(dedupe_key)
            uploaded_debug_rows.append(item)
        temp_profile["_uploaded_self_matches"] = uploaded_debug_rows[:5]

        rows = official_rows[:top_k]
        for index, item in enumerate(rows, start=1):
            item["rank"] = index
            item["reason"] = self._override_recommendation_reason(temp_profile, item)
            item["explanation"]["reason"] = item["reason"]
        update_request_progress(
            request_id,
            phase="recommendation_formatting",
            message="추천 결과를 정렬하고 화면 응답을 구성하는 중입니다.",
            percent=94,
            detail=f"공식 추천 {len(rows)}개, 업로드 참고 {len(uploaded_rows)}개",
        )
        return rows

    def _prepend_stored_exact_match_if_available(self, recommendations: List[dict], temp_profile: dict) -> List[dict]:
        upload_signature = str(temp_profile.get("upload_signature", "") or "")
        if not upload_signature:
            return recommendations
        uploaded_matches = list(temp_profile.get("_uploaded_self_matches", []) or [])
        if any(bool(item.get("exact_same_upload")) for item in uploaded_matches):
            return recommendations
        stored = self.recommendation_service.find_uploaded_record_by_upload_signature(upload_signature)
        if not stored:
            return recommendations
        synthetic = {
            "rank": 1,
            "report_no": str(stored.get("report_no", "") or ""),
            "target_report_no": str(stored.get("report_no", "") or ""),
            "product_name": str(stored.get("product_name", "") or temp_profile.get("product_name", "") or ""),
            "target_product_name": str(stored.get("product_name", "") or temp_profile.get("product_name", "") or ""),
            "target_product_main_category": str(temp_profile.get("product_main_category", "") or ""),
            "similarity_score": 1.0,
            "function_similarity_score": 1.0,
            "core_match_score": 1.0,
            "substitutability": "?믪쓬",
            **recommendation_quality_metadata(1.0, {}, exact_match=True),
            "shared_ingredients": sorted(
                set(str(item or "") for item in temp_profile.get("primary_ingredients", []))
                | set(str(item or "") for item in temp_profile.get("secondary_ingredients", []))
                | set(str(item or "") for item in temp_profile.get("support_ingredients", []))
            ),
            "target_primary_ingredients": list(temp_profile.get("primary_ingredients", [])),
            "target_secondary_ingredients": list(temp_profile.get("secondary_ingredients", [])),
            "target_support_ingredients": list(temp_profile.get("support_ingredients", [])),
            "target_all_ingredients": list(
                dict.fromkeys(
                    list(temp_profile.get("primary_ingredients", []))
                    + list(temp_profile.get("secondary_ingredients", []))
                    + list(temp_profile.get("support_ingredients", []))
                )
            ),
            "target_other_ingredients": [],
            "shared_categories": [str(temp_profile.get("product_main_category", "") or "")] if str(temp_profile.get("product_main_category", "") or "") else [],
            "reason": "?숈씪 ?낅젰 signature濡?湲곕줉??寃곗꽍?대? ?곹뭹?낅땲??",
            "caution": generate_caution(),
            "explanation": build_explanation_json(
                "?숈씪 ?낅젰 signature濡?湲곕줉??寃곗꽍?대? ?곹뭹?낅땲??",
                {
                    "shared_ingredients": sorted(
                        set(str(item or "") for item in temp_profile.get("primary_ingredients", []))
                        | set(str(item or "") for item in temp_profile.get("secondary_ingredients", []))
                        | set(str(item or "") for item in temp_profile.get("support_ingredients", []))
                    ),
                    "base_only_ingredients": [],
                    "target_only_ingredients": [],
                    "shared_categories": [str(temp_profile.get("product_main_category", "") or "")] if str(temp_profile.get("product_main_category", "") or "") else [],
                    "different_categories": [],
                },
                "?믪쓬",
            ),
            "exact_same_upload": True,
        }
        temp_profile["_uploaded_self_matches"] = [synthetic] + [
            item
            for item in uploaded_matches
            if str(item.get("report_no", "") or "") != synthetic["report_no"]
        ]
        return recommendations

    def _maybe_rerank_official_recommendations(
        self,
        temp_profile: dict,
        recommendations: List[dict],
        llm_rerank: bool,
    ) -> tuple[List[dict], bool, str]:
        official_recommendations = [
            dict(item)
            for item in recommendations
            if not str(item.get("report_no", "") or "").startswith("UPLOADED-")
        ]
        if not llm_rerank or len(official_recommendations) <= 1:
            return official_recommendations, False, ""
        reranked, applied, error = self.recommendation_service._maybe_rerank_with_llm(
            temp_profile,
            official_recommendations,
            llm_rerank,
        )
        return reranked, applied, error

    def _build_response(
        self,
        input_type: str,
        ocr_payload: Optional[dict],
        parsed: dict,
        matched_ingredients: List[dict],
        estimated_profile: dict,
        recommendations: List[dict],
        execution_seconds: float,
        saved_product: Optional[dict] = None,
        trace_id: str = "",
        image_hash: str = "",
        ocr_text_hash: str = "",
        parsed_signature: str = "",
        profile_signature: str = "",
        llm_rerank: bool = False,
    ) -> dict:
        corrected = self.apply_ocr_specific_profile_corrections(input_type, parsed, matched_ingredients, estimated_profile, ocr_payload, recommendations)
        settings = get_settings()
        uploaded_self_matches = list(corrected["estimated_profile"].get("_uploaded_self_matches", []) or [])
        exact_match_detected = any(bool(item.get("exact_same_upload")) for item in uploaded_self_matches)
        llm_rerank_applied = False
        llm_rerank_error = ""
        try:
            official_recommendations, llm_rerank_applied, llm_rerank_error = self._maybe_rerank_official_recommendations(
                corrected["estimated_profile"],
                recommendations,
                llm_rerank,
            )
        except Exception as exc:  # noqa: BLE001
            official_recommendations = [
                item for item in recommendations
                if not str(item.get("report_no", "") or "").startswith("UPLOADED-")
            ]
            llm_rerank_error = str(exc)
        uploaded_similar_cases = uploaded_self_matches
        display_recommendations = official_recommendations if official_recommendations else list(recommendations)
        parse_metadata = dict((corrected["parsed_result"] or {}).get("parse_metadata", {}) or {})
        resolved_ocr_confidence = (
            (ocr_payload or {}).get("confidence")
            if ocr_payload is not None
            else corrected["parsed_result"].get("ocr_confidence")
        )
        quality_payload = {
            "ocr_confidence": resolved_ocr_confidence,
            "profile_confidence": float(
                corrected["parsed_result"].get(
                    "profile_confidence",
                    corrected["estimated_profile"].get("confidence", corrected["parsed_result"].get("confidence", 0.0)),
                )
                or 0.0
            ),
            "quality_grade": str(
                corrected["estimated_profile"].get("quality_grade", "")
                or corrected["parsed_result"].get("quality_grade", "")
                or ""
            ),
            "warnings": corrected["quality_warnings"],
            "candidate_enabled": corrected["estimated_profile"].get("is_candidate_enabled"),
            "status": str(corrected["estimated_profile"].get("status", "") or ""),
            "candidate_scope": str(corrected["estimated_profile"].get("candidate_scope", "") or ""),
            "candidate_disabled_reason": str(corrected["estimated_profile"].get("candidate_disabled_reason", "") or ""),
        }
        debug_payload = None
        if settings.debug_response:
            debug_payload = {
                "image_hash": str(image_hash or ""),
                "ocr_text_hash": str(ocr_text_hash or parse_metadata.get("ocr_text_hash", "") or ""),
                "parsed_signature": str(parsed_signature or parse_metadata.get("parsed_signature", "") or ""),
                "profile_signature": str(profile_signature or corrected["estimated_profile"].get("profile_signature", "") or ""),
                "parse_cache_hit": bool(parse_metadata.get("cache_hit")),
                "exact_match_detected": exact_match_detected,
            }

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
            "trace_id": str(trace_id or ""),
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
            "ingredient_db_match_statuses": self.build_ingredient_db_match_statuses(
                corrected["parsed_result"].get("ingredient_objects", []),
                matched_ingredients,
                corrected["estimated_profile"],
            ),
            "estimated_profile": {
                "product_main_category": corrected["estimated_profile"].get("product_main_category", ""),
                "primary_ingredients": corrected["estimated_profile"].get("primary_ingredients", []),
                "secondary_ingredients": corrected["estimated_profile"].get("secondary_ingredients", []),
                "support_ingredients": corrected["estimated_profile"].get("support_ingredients", []),
                "upload_signature": corrected["estimated_profile"].get("upload_signature", ""),
                "profile_signature": corrected["estimated_profile"].get("profile_signature", ""),
                "status": corrected["estimated_profile"].get("status", ""),
                "quality_grade": corrected["estimated_profile"].get("quality_grade", ""),
                "is_candidate_enabled": corrected["estimated_profile"].get("is_candidate_enabled"),
                "candidate_scope": corrected["estimated_profile"].get("candidate_scope", ""),
                "candidate_disabled_reason": corrected["estimated_profile"].get("candidate_disabled_reason", ""),
                "confidence": corrected["parsed_result"].get("profile_confidence", corrected["estimated_profile"].get("confidence")),
            },
            "recommendations": display_recommendations,
            "official_recommendations": official_recommendations,
            "uploaded_self_matches": uploaded_self_matches,
            "uploaded_similar_cases": uploaded_similar_cases,
            "llm_rerank_applied": llm_rerank_applied,
            "llm_rerank_error": llm_rerank_error,
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
            "quality": quality_payload,
            "debug": debug_payload,
            "saved_product": saved_product,
        }

    def recommend_from_ingredients(
        self,
        ingredients: List[str],
        top_k: int = 10,
        candidate_limit: int = 1000,
        product_name_candidate: str = "",
        llm_rerank: bool = False,
        request_id: str = "",
    ) -> dict:
        started = time.perf_counter()
        update_request_progress(
            request_id,
            phase="service_loading",
            message="추천 DB와 원료 프로필을 로딩하는 중입니다.",
            percent=3,
            detail="초기 요청에서는 이 단계가 오래 걸릴 수 있습니다.",
        )
        self._ensure_loaded()
        update_request_progress(
            request_id,
            phase="ingredient_input",
            message="수정한 원료 목록을 검증하는 중입니다.",
            percent=10,
            detail=f"입력 원료 {len(ingredients)}개",
        )
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
        matched = self.match_raw_ingredients_to_functional_ingredients(ingredient_objects, request_id=request_id)
        update_request_progress(
            request_id,
            phase="profile_building",
            message="임시 제품 프로필과 semantic vector를 구성하는 중입니다.",
            percent=70,
            detail=f"매칭 원료 {len(matched)}개",
        )
        temp_vector = self.build_temp_product_vector_from_ingredients(ingredient_objects, matched)
        upload_signature = ""
        temp_hash = hashlib.sha1("|".join(sorted(parsed["normalized_ingredients"])).encode("utf-8")).hexdigest()[:16]
        name_hint = self._infer_upload_category_hint(parsed, product_name_candidate)
        temp_profile = self._build_temp_profile(f"uploaded::{temp_hash}", product_name_candidate or "업로드 입력 제품", matched, name_hint)
        temp_profile["upload_signature"] = upload_signature
        temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
        temp_profile = self._apply_domain_category_overrides(parsed, temp_profile)
        temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit, request_id=request_id) if temp_vector else []
        recommendations = self._prepend_stored_exact_match_if_available(recommendations, temp_profile)
        candidate_state = determine_upload_candidate_state(parsed, temp_profile, False)
        temp_profile["status"] = candidate_state["status"]
        temp_profile["quality_grade"] = candidate_state["quality_grade"]
        temp_profile["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
        temp_profile["candidate_scope"] = candidate_state["candidate_scope"]
        temp_profile["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        temp_profile["candidate_scope"] = candidate_state["candidate_scope"]
        temp_profile["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        execution_seconds = round(time.perf_counter() - started, 6)
        trace_id = self._log_trace(
            input_type="ingredients",
            parsed=parsed,
            estimated_profile=temp_profile,
            recommendations=recommendations,
            execution_seconds=execution_seconds,
            candidate_count=len(recommendations),
        )
        response = self._build_response(
            "ingredients",
            None,
            parsed,
            matched,
            temp_profile,
            recommendations,
            execution_seconds,
            trace_id=trace_id,
            parsed_signature=str((parsed.get("parse_metadata") or {}).get("parsed_signature", "") or compute_parsed_signature(parsed)),
            profile_signature=str(temp_profile.get("profile_signature", "") or ""),
            llm_rerank=llm_rerank,
        )
        update_request_progress(
            request_id,
            phase="complete",
            message="원료 기준 재추천이 완료되었습니다.",
            percent=100,
            detail=f"추천 {len(response.get('recommendations', []))}개",
            status="complete",
        )
        return response

    def recommend_from_ocr_text(
        self,
        raw_text: str,
        top_k: int = 10,
        candidate_limit: int = 1000,
        llm_rerank: bool = False,
        request_id: str = "",
    ) -> dict:
        started = time.perf_counter()
        update_request_progress(
            request_id,
            phase="service_loading",
            message="추천 DB와 원료 프로필을 로딩하는 중입니다.",
            percent=3,
            detail="초기 요청에서는 이 단계가 오래 걸릴 수 있습니다.",
        )
        self._ensure_loaded()
        update_request_progress(request_id, phase="ocr_parse", message="OCR 텍스트에서 원료 후보를 파싱하는 중입니다.", percent=35)
        parsed = parse_ingredients_from_ocr_text(raw_text, self.recommendation_service.runtime["sqlite_path"])
        matched = self.match_raw_ingredients_to_functional_ingredients(parsed.get("ingredient_objects", []), request_id=request_id)
        update_request_progress(
            request_id,
            phase="profile_building",
            message="임시 제품 프로필과 semantic vector를 구성하는 중입니다.",
            percent=70,
            detail=f"매칭 원료 {len(matched)}개",
        )
        temp_vector = self.build_temp_product_vector_from_ingredients(parsed.get("ingredient_objects", []), matched)
        upload_signature = build_upload_signature("ocr_text", raw_text or "")
        ocr_text_hash = compute_ocr_text_hash(raw_text or "")
        parsed_signature = str((parsed.get("parse_metadata") or {}).get("parsed_signature", "") or compute_parsed_signature(parsed))
        temp_hash = hashlib.sha1(upload_signature.encode("utf-8")).hexdigest()[:16]
        name_hint = self._infer_upload_category_hint(parsed)
        temp_profile = self._build_temp_profile(f"uploaded::{temp_hash}", parsed.get("product_name_candidate") or "OCR 입력 제품", matched, name_hint)
        temp_profile["upload_signature"] = upload_signature
        temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
        temp_profile = self._apply_domain_category_overrides(parsed, temp_profile)
        temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit, request_id=request_id) if temp_vector else []
        recommendations = self._prepend_stored_exact_match_if_available(recommendations, temp_profile)
        ocr_payload = {
            "raw_text": raw_text,
            "confidence": None,
            "confidence_source": "unavailable",
            "lines": [],
            "blocks": [],
            "source": "ocr_text",
        }
        candidate_state = determine_upload_candidate_state(parsed, temp_profile, False)
        temp_profile["status"] = candidate_state["status"]
        temp_profile["quality_grade"] = candidate_state["quality_grade"]
        temp_profile["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
        execution_seconds = round(time.perf_counter() - started, 6)
        trace_id = self._log_trace(
            input_type="ocr_text",
            parsed=parsed,
            estimated_profile=temp_profile,
            recommendations=recommendations,
            execution_seconds=execution_seconds,
            upload_signature=upload_signature,
            ocr_text_hash=ocr_text_hash,
            candidate_count=len(recommendations),
        )
        response = self._build_response(
            "ocr_text",
            ocr_payload,
            parsed,
            matched,
            temp_profile,
            recommendations,
            execution_seconds,
            trace_id=trace_id,
            ocr_text_hash=ocr_text_hash,
            parsed_signature=parsed_signature,
            profile_signature=str(temp_profile.get("profile_signature", "") or ""),
            llm_rerank=llm_rerank,
        )
        candidate_state = determine_upload_candidate_state(response["parsed"], temp_profile, bool(response.get("needs_user_review")))
        temp_profile["status"] = candidate_state["status"]
        temp_profile["quality_grade"] = candidate_state["quality_grade"]
        temp_profile["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
        temp_profile["candidate_scope"] = candidate_state["candidate_scope"]
        temp_profile["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        response["estimated_profile"]["status"] = candidate_state["status"]
        response["estimated_profile"]["quality_grade"] = candidate_state["quality_grade"]
        response["estimated_profile"]["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
        response["estimated_profile"]["candidate_scope"] = candidate_state["candidate_scope"]
        response["estimated_profile"]["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        response["quality"]["status"] = candidate_state["status"]
        response["quality"]["quality_grade"] = candidate_state["quality_grade"]
        response["quality"]["candidate_enabled"] = candidate_state["is_candidate_enabled"]
        response["quality"]["candidate_scope"] = candidate_state["candidate_scope"]
        response["quality"]["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        response["saved_product"] = self.recommendation_service.register_uploaded_product(
            source_type="ocr_text",
            product_name=str(temp_profile.get("product_name", "") or parsed.get("product_name_candidate", "") or "OCR 입력 상품"),
            parsed=response["parsed"],
            estimated_profile=temp_profile,
            vector=temp_vector,
            ocr_payload=ocr_payload,
            upload_signature=upload_signature,
            ocr_text_hash=ocr_text_hash,
            parsed_signature=parsed_signature,
            profile_signature=str(temp_profile.get("profile_signature", "") or ""),
            status=candidate_state["status"],
            quality_grade=candidate_state["quality_grade"],
            is_candidate_enabled=candidate_state["is_candidate_enabled"],
            notes="created from OCR text upload",
            needs_user_review=bool(response.get("needs_user_review")),
            quality_warnings=response.get("quality_warnings", []),
        )
        update_request_progress(
            request_id,
            phase="complete",
            message="OCR 텍스트 추천이 완료되었습니다.",
            percent=100,
            detail=f"추천 {len(response.get('recommendations', []))}개",
            status="complete",
        )
        return response

    def recommend_from_uploaded_image(
        self,
        image_bytes: bytes,
        filename: str,
        top_k: int = 10,
        candidate_limit: int = 1000,
        llm_rerank: bool = False,
        request_id: str = "",
    ) -> dict:
        started = time.perf_counter()
        update_request_progress(
            request_id,
            phase="service_loading",
            message="추천 DB와 원료 프로필을 로딩하는 중입니다.",
            percent=3,
            detail="초기 요청에서는 이 단계가 오래 걸릴 수 있습니다.",
        )
        self._ensure_loaded()
        update_request_progress(
            request_id,
            phase="image_prepare",
            message="업로드 이미지를 확인하고 OCR 분석을 준비하는 중입니다.",
            percent=5,
            detail=str(filename or "upload.jpg"),
        )
        image_hash = build_image_hash(image_bytes)
        upload_signature = build_upload_signature("ocr_image", image_hash)
        cached_ocr_payload = self.recommendation_service.find_cached_ocr_payload_by_upload_signature(upload_signature)
        if cached_ocr_payload:
            update_request_progress(request_id, phase="ocr_cache", message="캐시된 OCR 결과를 불러오는 중입니다.", percent=20)
            ocr_result = {
                "raw_text": str(cached_ocr_payload.get("raw_text", "") or ""),
                "confidence": cached_ocr_payload.get("confidence"),
                "confidence_source": str(cached_ocr_payload.get("confidence_source", "cached") or "cached"),
                "lines": list(cached_ocr_payload.get("lines", []) or []),
                "blocks": list(cached_ocr_payload.get("blocks", []) or []),
                "source": "ocr_cache",
            }
        else:
            ocr_result = None
        suffix = Path(filename or "upload.jpg").suffix.lower() or ".jpg"
        if ocr_result is None:
            update_request_progress(request_id, phase="ocr_extract", message="이미지에서 OCR 텍스트를 추출하는 중입니다.", percent=20)
            temp_path = save_temp_upload(image_bytes, suffix)
            try:
                ocr_result = extract_text_from_image(str(temp_path))
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        if ocr_result.get("error"):
            update_request_progress(
                request_id,
                phase="error",
                message="OCR 인식 중 오류가 발생했습니다.",
                percent=100,
                detail=str(ocr_result.get("error")),
                status="error",
            )
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
                "quality": {
                    "ocr_confidence": ocr_result.get("confidence"),
                    "profile_confidence": 0.0,
                    "quality_grade": "D",
                    "warnings": [{"code": "ocr_error", "message": str(ocr_result.get("error"))}],
                    "candidate_enabled": False,
                    "status": "review_needed",
                },
                "debug": {
                    "image_hash": "",
                    "ocr_text_hash": "",
                    "parsed_signature": "",
                    "profile_signature": "",
                    "parse_cache_hit": False,
                    "exact_match_detected": False,
                } if get_settings().debug_response else None,
                "parsed_ingredients_for_review": [],
                "excluded_ingredients": [],
                "product_name_category_hint": "",
            }

        raw_text = str(ocr_result.get("raw_text", "") or "")
        update_request_progress(
            request_id,
            phase="ocr_parse",
            message="OCR 텍스트에서 제품명과 원료 후보를 파싱하는 중입니다.",
            percent=38,
            detail=f"OCR 텍스트 {len(raw_text)}자",
        )
        parsed = parse_ingredients_from_ocr_text(raw_text, self.recommendation_service.runtime["sqlite_path"])
        parsed["ocr_confidence"] = ocr_result.get("confidence")
        parsed["ocr_confidence_source"] = ocr_result.get("confidence_source", "unavailable")
        matched = self.match_raw_ingredients_to_functional_ingredients(parsed.get("ingredient_objects", []), request_id=request_id)
        update_request_progress(
            request_id,
            phase="profile_building",
            message="임시 제품 프로필과 semantic vector를 구성하는 중입니다.",
            percent=70,
            detail=f"매칭 원료 {len(matched)}개",
        )
        temp_vector = self.build_temp_product_vector_from_ingredients(parsed.get("ingredient_objects", []), matched)
        ocr_text_hash = compute_ocr_text_hash(raw_text)
        parsed_signature = str((parsed.get("parse_metadata") or {}).get("parsed_signature", "") or compute_parsed_signature(parsed))
        temp_hash = hashlib.sha1(upload_signature.encode("utf-8")).hexdigest()[:16]
        name_hint = self._infer_upload_category_hint(parsed, filename)
        temp_profile = self._build_temp_profile(f"uploaded::{temp_hash}", parsed.get("product_name_candidate") or filename, matched, name_hint)
        temp_profile["upload_signature"] = upload_signature
        temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
        temp_profile = self._apply_domain_category_overrides(parsed, temp_profile)
        temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
        recommendations = self.calculate_similar_products_for_temp_vector(temp_vector, temp_profile, top_k, candidate_limit, request_id=request_id) if temp_vector else []
        recommendations = self._prepend_stored_exact_match_if_available(recommendations, temp_profile)
        candidate_state = determine_upload_candidate_state(parsed, temp_profile, False)
        temp_profile["status"] = candidate_state["status"]
        temp_profile["quality_grade"] = candidate_state["quality_grade"]
        temp_profile["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
        temp_profile["candidate_scope"] = candidate_state["candidate_scope"]
        temp_profile["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        execution_seconds = round(time.perf_counter() - started, 6)
        trace_id = self._log_trace(
            input_type="image",
            parsed=parsed,
            estimated_profile=temp_profile,
            recommendations=recommendations,
            execution_seconds=execution_seconds,
            upload_signature=upload_signature,
            image_hash=image_hash,
            ocr_text_hash=ocr_text_hash,
            candidate_count=len(recommendations),
        )
        response = self._build_response(
            "image",
            ocr_result,
            parsed,
            matched,
            temp_profile,
            recommendations,
            execution_seconds,
            trace_id=trace_id,
            image_hash=image_hash,
            ocr_text_hash=ocr_text_hash,
            parsed_signature=parsed_signature,
            profile_signature=str(temp_profile.get("profile_signature", "") or ""),
            llm_rerank=llm_rerank,
        )
        candidate_state = determine_upload_candidate_state(response["parsed"], temp_profile, bool(response.get("needs_user_review")))
        temp_profile["status"] = candidate_state["status"]
        temp_profile["quality_grade"] = candidate_state["quality_grade"]
        temp_profile["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
        temp_profile["candidate_scope"] = candidate_state["candidate_scope"]
        temp_profile["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        response["estimated_profile"]["status"] = candidate_state["status"]
        response["estimated_profile"]["quality_grade"] = candidate_state["quality_grade"]
        response["estimated_profile"]["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
        response["estimated_profile"]["candidate_scope"] = candidate_state["candidate_scope"]
        response["estimated_profile"]["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        response["quality"]["status"] = candidate_state["status"]
        response["quality"]["quality_grade"] = candidate_state["quality_grade"]
        response["quality"]["candidate_enabled"] = candidate_state["is_candidate_enabled"]
        response["quality"]["candidate_scope"] = candidate_state["candidate_scope"]
        response["quality"]["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]
        response["saved_product"] = self.recommendation_service.register_uploaded_product(
            source_type="ocr_image",
            product_name=str(temp_profile.get("product_name", "") or parsed.get("product_name_candidate", "") or filename),
            parsed=response["parsed"],
            estimated_profile=temp_profile,
            vector=temp_vector,
            ocr_payload=ocr_result,
            source_filename=filename,
            upload_signature=upload_signature,
            image_hash=image_hash,
            ocr_text_hash=ocr_text_hash,
            parsed_signature=parsed_signature,
            profile_signature=str(temp_profile.get("profile_signature", "") or ""),
            status=candidate_state["status"],
            quality_grade=candidate_state["quality_grade"],
            is_candidate_enabled=candidate_state["is_candidate_enabled"],
            notes=f"created from uploaded image: {filename}",
            needs_user_review=bool(response.get("needs_user_review")),
            quality_warnings=response.get("quality_warnings", []),
        )
        update_request_progress(
            request_id,
            phase="complete",
            message="이미지 분석과 추천 생성이 완료되었습니다.",
            percent=100,
            detail=f"추천 {len(response.get('recommendations', []))}개",
            status="complete",
        )
        return response


def coerce_ingredient_request_payload(ingredients: Optional[List[str]], raw_ingredients: Optional[str]) -> List[str]:
    if ingredients:
        return [str(item).strip() for item in ingredients if str(item).strip()]
    if raw_ingredients:
        return split_ingredients(raw_ingredients)
    return []
