"""
Ingredient-matching guard utilities.

Lightweight pure-Python helpers for normalising ingredient names, detecting
suspicious mappings, building cache keys, and classifying upload quality.

These functions are used by UploadRecommendationService but live in their own
module so they can be unit-tested and imported independently.
"""
from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from api.ingredient_family import infer_joint_ingredient_family
from api.ingredient_match_constants import (
    CRITICAL_WARNING_CODES,
    EXCLUDED_PRIMARY_KEYWORDS,
    INFO_WARNING_CODES,
    JOINT_FAMILY_CATEGORY_SUB,
    MIN_SUFFICIENT_OCR_TEXT_LENGTH,
    NON_EXCLUDED_PRIMARY_KEYWORDS,
    NOTICE_WARNING_CODES,
    RUNTIME_CACHE_LIKE_VECTOR_EXCLUDED_TERMS,
    RUNTIME_LIKE_DISABLED_KEYS,
    RUNTIME_LOW_SIGNAL_UPLOAD_VECTOR_EXACT_TERMS,
    RUNTIME_SUSPICIOUS_MAPPING_RULES,
)

# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------


def normalize_lookup_key(value: str) -> str:
    """Collapse whitespace and lower-case *value* for dict/set look-ups."""
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def normalize_cache_exact_key(value: str) -> str:
    """Strip everything except alphanumerics and Korean characters."""
    return re.sub(r"[^0-9a-z가-힣]", "", str(value or "").strip().lower())


def normalized_name_similarity(left: str, right: str) -> float:
    """Return a 0–1 character-level similarity between two normalised names."""
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


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def dedupe_strings(values: List[str]) -> List[str]:
    """Return *values* with duplicates removed (order preserved)."""
    seen: set[str] = set()
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
    return any(
        normalize_cache_exact_key(fragment) in normalized
        for fragment in fragments
        if str(fragment or "").strip()
    )


# ---------------------------------------------------------------------------
# Suspicious-mapping / guard checks
# ---------------------------------------------------------------------------


def resolve_runtime_mapping_rule(*values: str) -> Optional[dict]:
    """Return the first matching suspicious-mapping rule for any of *values*."""
    normalized_values = [normalize_cache_exact_key(v) for v in values if str(v or "").strip()]
    for rule in RUNTIME_SUSPICIOUS_MAPPING_RULES:
        for nv in normalized_values:
            if any(normalize_cache_exact_key(trigger) in nv for trigger in rule["inputs"]):
                return rule
    return None


def is_like_blocked_input(*values: str) -> bool:
    """Return True when *any* value is too short / ambiguous for LIKE matching."""
    for value in values:
        normalized = normalize_cache_exact_key(value)
        if not normalized:
            continue
        if normalized in RUNTIME_LIKE_DISABLED_KEYS:
            return True
        if 0 < len(normalized) <= 4 and re.fullmatch(r"[a-z0-9]+", normalized):
            return True
    return False


def contains_guarded_runtime_term(value: str, guarded_terms: set[str]) -> bool:
    """Return True when *value* contains or equals any term in *guarded_terms*."""
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
    """Return True when *value* exactly equals any term in *guarded_terms*."""
    normalized_value = normalize_cache_exact_key(value)
    if not normalized_value:
        return False
    return any(
        normalize_cache_exact_key(term) == normalized_value
        for term in guarded_terms
        if str(term or "").strip()
    )


def is_low_signal_upload_vector_term(*values: str) -> bool:
    return any(
        contains_exact_guarded_runtime_term(v, RUNTIME_LOW_SIGNAL_UPLOAD_VECTOR_EXACT_TERMS)
        for v in values
    )


def looks_like_runtime_measurement_or_symbol(value: str) -> bool:
    """Return True when *value* looks like a unit, quantity, or bare symbol."""
    text = str(value or "").strip()
    normalized = normalize_cache_exact_key(text)
    if not text or not normalized:
        return False
    if normalized in {"mg", "g", "%", "ml", "kg", "mcg", "μg", "ug"}:
        return True
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:mg|g|ml|kg|mcg|ug|%)?", normalized))


# ---------------------------------------------------------------------------
# Joint-family helpers
# ---------------------------------------------------------------------------


def resolve_joint_input_family(*values: str) -> str:
    """Return the first non-empty joint family inferred from *values*."""
    for value in values:
        family = infer_joint_ingredient_family(str(value or ""))
        if family:
            return family
    return ""


def build_joint_family_category_row(functional_name: str, family: str) -> dict:
    """Build a synthetic category-map row for a protected joint ingredient."""
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


# ---------------------------------------------------------------------------
# Hash / signature helpers
# ---------------------------------------------------------------------------


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
        "primary_ingredients": sorted(
            str(item or "") for item in temp_profile.get("primary_ingredients", []) if str(item or "")
        ),
        "secondary_ingredients": sorted(
            str(item or "") for item in temp_profile.get("secondary_ingredients", []) if str(item or "")
        ),
        "support_ingredients": sorted(
            str(item or "") for item in temp_profile.get("support_ingredients", []) if str(item or "")
        ),
        "vector": {
            str(key): round(float(value), 6)
            for key, value in sorted(temp_vector.items())
            if float(value or 0.0) > 0
        },
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Upload candidate state
# ---------------------------------------------------------------------------


def determine_upload_candidate_state(
    parsed: dict,
    estimated_profile: dict,
    needs_user_review: bool,
) -> dict:
    """Decide whether an uploaded product profile can be used as a candidate.

    Returns a dict with keys: status, quality_grade, is_candidate_enabled,
    candidate_scope, candidate_disabled_reason.
    """
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
        for item in (parsed.get("quality_warnings", []) or [])
        if str(item.get("severity", "") or "") == "critical"
    }
    has_valid_category = bool(
        main_category and len(main_category.strip()) >= 2 and "?" not in main_category
    )
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

    # Secondary eligibility path (single-core products).
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
        candidate_scope = (
            "uploaded_auto_notice_review" if soft_review_only else "uploaded_auto_single_core"
        )
        candidate_disabled_reason = ""
    elif candidate_enabled and soft_review_only:
        candidate_scope = "uploaded_auto_notice_review"

    # Derive quality grade if not already set.
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
        status = (
            "verified"
            if normalized_count >= 2 and confidence >= 0.7 and not critical_codes and not soft_review_only
            else "auto_eligible"
        )
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


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------


def contains_keyword(text: str, keywords: List[str]) -> bool:
    normalized = normalize_lookup_key(text)
    return any(normalize_lookup_key(kw) in normalized for kw in keywords)


def is_excluded_primary(name: str) -> bool:
    """Return True when *name* looks like an excipient / non-functional ingredient."""
    normalized = normalize_lookup_key(name)
    if any(normalize_lookup_key(kw) in normalized for kw in NON_EXCLUDED_PRIMARY_KEYWORDS):
        return False
    return any(normalize_lookup_key(kw) in normalized for kw in EXCLUDED_PRIMARY_KEYWORDS)


def has_sufficient_ocr_text(raw_text: str, ingredient_section_text: str = "") -> bool:
    collapsed_raw = re.sub(r"\s+", "", str(raw_text or ""))
    collapsed_section = re.sub(r"\s+", "", str(ingredient_section_text or ""))
    if len(collapsed_raw) >= MIN_SUFFICIENT_OCR_TEXT_LENGTH:
        return True
    return len(collapsed_section) >= 20


def contains_any_keyword(values: List[str], keywords: List[str]) -> bool:
    return any(contains_keyword(v, keywords) for v in values)


def count_keyword_matches(values: List[str], keywords: List[str]) -> int:
    matched: set[str] = set()
    for value in values:
        for keyword in keywords:
            if contains_keyword(value, [keyword]):
                matched.add(keyword)
    return len(matched)


# ---------------------------------------------------------------------------
# Warning helpers
# ---------------------------------------------------------------------------

_LLM_NOTICE_TERMS = [
    "알레르기", "알러지", "알류", "달걀", "계란", "우유", "대두", "밀", "땅콩",
    "메밀", "토마토", "복숭아", "아황산", "고등어", "게", "새우", "돼지고기",
    "쇠고기", "닭고기", "오징어", "조개류", "홍합", "잣", "가금류", "함유",
    "제조시설", "주의", "보관", "섭취", "임산부", "수유부", "어린이", "젤라틴",
    "우피", "색소", "먹물색소",
]


def append_warning(
    warnings: List[dict],
    code: str,
    message: str,
    severity: str = "warning",
    **extra,
) -> None:
    item: dict = {"code": code, "message": message, "severity": severity}
    item.update({k: v for k, v in extra.items() if v not in (None, "", [], {})})
    warnings.append(item)


def dedupe_warnings(warnings: List[dict]) -> List[dict]:
    seen: set = set()
    deduped: List[dict] = []
    for item in warnings:
        key = (
            str(item.get("code", "") or ""),
            str(item.get("message", "") or ""),
            tuple(item.get("ingredients", []) or []),
        )
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

    if code == "too_many_normalized_ingredients" and severity in {"notice", "critical"}:
        item["severity"] = severity
    elif code == "llm_warning" and (
        any(term in message for term in _LLM_NOTICE_TERMS)
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
    elif (
        "알레르기" in message
        or "알러지" in message
        or "함유" in message
        or "제조시설" in message
        or "주의" in message
        or "보관" in message
    ):
        item["severity"] = "notice"
    else:
        item["severity"] = "critical"
    return item


def split_warning_groups(warnings: List[dict]) -> dict:
    normalized = [normalize_warning_severity(w) for w in dedupe_warnings(warnings)]
    return {
        "quality_warnings": [w for w in normalized if w.get("severity") == "critical"],
        "critical_warnings": [w for w in normalized if w.get("severity") == "critical"],
        "notices": [w for w in normalized if w.get("severity") == "notice"],
        "info_warnings": [w for w in normalized if w.get("severity") == "info"],
    }
