from __future__ import annotations

import csv
import hashlib
import json
import os
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import api.ingredient_parse_service as ingredient_parse_service
from api.ocr_service import extract_text_from_image
from api.recommendation_service import RecommendationService
from api.upload_recommendation_service import (
    UploadRecommendationService,
    build_image_hash,
    build_profile_signature,
    build_upload_signature,
    determine_upload_candidate_state,
    normalize_cache_exact_key,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(os.environ.get("OCR_INPUT_DIR", REPO_ROOT / "input_images"))
OUTPUT_DIR = REPO_ROOT / "output" / "real_image_upload_test_v2"
SUMMARY_CSV = OUTPUT_DIR / "real_image_upload_test_summary_v2.csv"
DETAILS_JSON = OUTPUT_DIR / "real_image_upload_test_details_v2.json"
MATCHES_CSV = OUTPUT_DIR / "real_image_upload_test_matches_v2.csv"
RECOMMENDATIONS_CSV = OUTPUT_DIR / "real_image_upload_test_recommendations_v2.csv"
FAILURES_CSV = OUTPUT_DIR / "real_image_upload_test_failures_v2.csv"
TOP_K = 10
CANDIDATE_LIMIT = 1000
IMAGE_FILES = [f"product_{index:03d}.jpg" for index in range(1, 16)]
IMAGE_FALLBACK_EXTENSIONS = [".jpg", ".jpeg", ".png"]

SUMMARY_FIELDS = [
    "image_file",
    "status",
    "ocr_text_length",
    "parsed_ingredient_count",
    "matched_ingredient_count",
    "vector_size",
    "recommendation_count",
    "top1_product_name",
    "top1_similarity_score",
    "warning_count",
    "error_message",
]

MATCH_FIELDS = [
    "image_file",
    "parsed_product_name",
    "raw_ingredient",
    "normalized_raw",
    "matched_standard_name",
    "relation_type",
    "confidence",
    "match_method",
    "reason",
    "is_used_in_vector",
]

RECOMMENDATION_FIELDS = [
    "image_file",
    "parsed_product_name",
    "rank",
    "recommended_product_id",
    "recommended_product_name",
    "similarity_score",
    "similarity_label",
    "shared_ingredients",
    "evidence_strength",
    "explanation",
]

FAILURE_FIELDS = [
    "image_file",
    "stage",
    "error_message",
    "ocr_text_preview",
    "parsed_raw_ingredients_preview",
]


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(mean(values), 2) if values else 0.0


def preview_text(value: str, limit: int = 200) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:limit]


def join_preview(values: List[str], limit: int = 5) -> str:
    return " | ".join(str(item or "") for item in values[:limit])


def resolve_image_path(requested_name: str) -> Optional[Path]:
    requested_path = INPUT_DIR / requested_name
    if requested_path.exists():
        return requested_path

    stem = Path(requested_name).stem
    requested_suffix = Path(requested_name).suffix.lower()
    ordered_suffixes: List[str] = []
    if requested_suffix:
        ordered_suffixes.append(requested_suffix)
    ordered_suffixes.extend(suffix for suffix in IMAGE_FALLBACK_EXTENSIONS if suffix not in ordered_suffixes)

    for suffix in ordered_suffixes:
        candidate = INPUT_DIR / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def normalize_token(value: str) -> str:
    return normalize_cache_exact_key(str(value or ""))


def contains_any(text: str, tokens: Iterable[str]) -> bool:
    normalized = normalize_token(text)
    return any(normalize_token(token) in normalized for token in tokens if str(token or "").strip())


def warning_payload(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    payload = {"code": code, "message": message}
    payload.update(extra)
    return payload


def build_test_services() -> tuple[RecommendationService, UploadRecommendationService]:
    service = RecommendationService()
    service._ensure_catalog_table = lambda: None
    service._ensure_uploaded_product_table = lambda: None
    service._load_uploaded_products = lambda: None
    upload_service = UploadRecommendationService(service)
    upload_service._ensure_loaded()
    return service, upload_service


@contextmanager
def no_parse_cache_writes() -> Any:
    original_load = ingredient_parse_service._load_parse_cache
    original_save = ingredient_parse_service._save_parse_cache
    ingredient_parse_service._load_parse_cache = lambda *args, **kwargs: {}
    ingredient_parse_service._save_parse_cache = lambda *args, **kwargs: None
    try:
        yield
    finally:
        ingredient_parse_service._load_parse_cache = original_load
        ingredient_parse_service._save_parse_cache = original_save


def build_runtime_image_response(
    upload_service: UploadRecommendationService,
    image_path: Path,
    ocr_result: Dict[str, Any],
    parsed: Dict[str, Any],
    matched: List[Dict[str, Any]],
    temp_vector: Dict[str, float],
    recommendations: List[Dict[str, Any]],
    execution_seconds: float,
) -> Dict[str, Any]:
    raw_text = str(ocr_result.get("raw_text", "") or "")
    image_bytes = image_path.read_bytes()
    image_hash = build_image_hash(image_bytes)
    upload_signature = build_upload_signature("ocr_image", image_hash)
    ocr_text_hash = ingredient_parse_service.compute_ocr_text_hash(raw_text)
    parsed_signature = str(
        (parsed.get("parse_metadata") or {}).get("parsed_signature", "")
        or ingredient_parse_service.compute_parsed_signature(parsed)
    )
    temp_hash = hashlib.sha1(upload_signature.encode("utf-8")).hexdigest()[:16]
    product_name = str(parsed.get("product_name_candidate") or image_path.name)
    name_hint = upload_service.infer_category_from_product_name(product_name)
    temp_profile = upload_service._build_temp_profile(f"uploaded::{temp_hash}", product_name, matched, name_hint)
    temp_profile["upload_signature"] = upload_signature
    temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
    temp_profile = upload_service._apply_domain_category_overrides(parsed, temp_profile)
    temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)

    candidate_state = determine_upload_candidate_state(parsed, temp_profile, False)
    temp_profile["status"] = candidate_state["status"]
    temp_profile["quality_grade"] = candidate_state["quality_grade"]
    temp_profile["is_candidate_enabled"] = candidate_state["is_candidate_enabled"]
    temp_profile["candidate_scope"] = candidate_state["candidate_scope"]
    temp_profile["candidate_disabled_reason"] = candidate_state["candidate_disabled_reason"]

    response = upload_service._build_response(
        "image",
        ocr_result,
        parsed,
        matched,
        temp_profile,
        recommendations,
        execution_seconds,
        trace_id="",
        image_hash=image_hash,
        ocr_text_hash=ocr_text_hash,
        parsed_signature=parsed_signature,
        profile_signature=str(temp_profile.get("profile_signature", "") or ""),
    )

    final_candidate_state = determine_upload_candidate_state(response["parsed"], temp_profile, bool(response.get("needs_user_review")))
    temp_profile["status"] = final_candidate_state["status"]
    temp_profile["quality_grade"] = final_candidate_state["quality_grade"]
    temp_profile["is_candidate_enabled"] = final_candidate_state["is_candidate_enabled"]
    temp_profile["candidate_scope"] = final_candidate_state["candidate_scope"]
    temp_profile["candidate_disabled_reason"] = final_candidate_state["candidate_disabled_reason"]
    response["estimated_profile"]["status"] = final_candidate_state["status"]
    response["estimated_profile"]["quality_grade"] = final_candidate_state["quality_grade"]
    response["estimated_profile"]["is_candidate_enabled"] = final_candidate_state["is_candidate_enabled"]
    response["estimated_profile"]["candidate_scope"] = final_candidate_state["candidate_scope"]
    response["estimated_profile"]["candidate_disabled_reason"] = final_candidate_state["candidate_disabled_reason"]
    response["quality"]["status"] = final_candidate_state["status"]
    response["quality"]["quality_grade"] = final_candidate_state["quality_grade"]
    response["quality"]["candidate_enabled"] = final_candidate_state["is_candidate_enabled"]
    response["quality"]["candidate_scope"] = final_candidate_state["candidate_scope"]
    response["quality"]["candidate_disabled_reason"] = final_candidate_state["candidate_disabled_reason"]
    response["saved_product"] = None
    return response


def build_quality_warnings(
    ingredient_objects: List[Dict[str, Any]],
    matched: List[Dict[str, Any]],
    temp_vector: Dict[str, float],
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    warnings = list(response.get("quality_warnings", []) or [])
    relation_types = {str(item.get("relation_type", "") or "") for item in matched}
    if matched and relation_types.issubset({"unknown", "unrelated"}):
        warnings.append(
            warning_payload(
                "only_unknown_or_unrelated_matches",
                "매칭된 원료가 모두 unknown/unrelated로 분류되었습니다.",
            )
        )

    rule_checks = [
        (
            ("hca", "hydroxycitricacid", "hydroxycitric"),
            ("가르시니아", "garcinia"),
            "suspicious_hca_mapping",
            "HCA가 가르시니아 계열이 아닌 원료로 매칭되었습니다.",
        ),
        (
            ("실리마린", "silymarin"),
            ("밀크씨슬", "milkthistle"),
            "suspicious_silymarin_mapping",
            "실리마린이 밀크씨슬 계열이 아닌 원료로 매칭되었습니다.",
        ),
        (
            ("진세노사이드", "ginsenoside"),
            ("홍삼", "인삼", "redginseng", "ginseng"),
            "suspicious_ginsenoside_mapping",
            "진세노사이드가 홍삼/인삼 계열이 아닌 원료로 매칭되었습니다.",
        ),
        (
            ("루테인", "lutein", "지아잔틴", "zeaxanthin"),
            ("루테인", "지아잔틴", "마리골드", "marigold"),
            "suspicious_lutein_zeaxanthin_mapping",
            "루테인/지아잔틴이 루테인/마리골드 계열이 아닌 원료로 매칭되었습니다.",
        ),
        (
            ("산화아연", "글루콘산아연", "zincoxide", "zincgluconate"),
            ("아연", "zinc"),
            "suspicious_zinc_mapping",
            "산화아연/글루콘산아연이 아연 계열이 아닌 원료로 매칭되었습니다.",
        ),
    ]

    for match in matched:
        raw_input = str(match.get("raw_input") or match.get("raw_ingredient") or "")
        standard_name = str(match.get("functional_ingredient_name") or "")
        for inputs, allowed, code, message in rule_checks:
            if contains_any(raw_input, inputs) and not contains_any(standard_name, allowed):
                warnings.append(
                    warning_payload(
                        code,
                        message,
                        raw_input=raw_input,
                        matched_standard_name=standard_name,
                    )
                )

    vector_excipient_names = {"결정셀룰로스", "이산화규소", "스테아린산마그네슘"}
    for item in ingredient_objects:
        raw_name = str(item.get("raw", "") or "")
        normalized_name = str(item.get("normalized_for_matching", "") or "")
        if not contains_any(raw_name, vector_excipient_names) and not contains_any(normalized_name, vector_excipient_names):
            continue
        related_matches = [
            match for match in matched
            if str(match.get("raw_input") or match.get("raw_ingredient") or "") == raw_name
        ]
        for match in related_matches:
            functional_name = str(match.get("functional_ingredient_name", "") or "")
            if functional_name in temp_vector:
                warnings.append(
                    warning_payload(
                        "excipient_in_vector",
                        "부형제로 보이는 원료가 추천 벡터에 포함되었습니다.",
                        raw_input=raw_name,
                        functional_ingredient_name=functional_name,
                    )
                )
    return warnings


def recommendation_rows_for_image(
    image_file: str,
    parsed_product_name: str,
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in recommendations[:TOP_K]:
        rows.append(
            {
                "image_file": image_file,
                "parsed_product_name": parsed_product_name,
                "rank": item.get("rank", ""),
                "recommended_product_id": str(item.get("target_report_no", "") or ""),
                "recommended_product_name": str(item.get("target_product_name", "") or ""),
                "similarity_score": float(item.get("similarity_score", 0.0) or 0.0),
                "similarity_label": str(item.get("substitutability", "") or ""),
                "shared_ingredients": json.dumps(item.get("shared_ingredients", []), ensure_ascii=False),
                "evidence_strength": float(item.get("core_match_score", item.get("function_similarity_score", 0.0)) or 0.0),
                "explanation": str(item.get("reason", "") or ""),
            }
        )
    return rows


def match_rows_for_image(
    image_file: str,
    parsed_product_name: str,
    ingredient_objects: List[Dict[str, Any]],
    matched: List[Dict[str, Any]],
    temp_vector: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    matched_index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for match in matched:
        key = (
            str(match.get("raw_input") or match.get("raw_ingredient") or ""),
            str(match.get("normalized_for_matching", "") or ""),
        )
        matched_index[key] = match

    for item in ingredient_objects:
        raw_ingredient = str(item.get("raw", "") or "")
        normalized_raw = str(item.get("normalized_for_matching", "") or "")
        key = (raw_ingredient, normalized_raw)
        match = matched_index.get(key)
        reason_parts: List[str] = []
        if match:
            if match.get("protected_family"):
                reason_parts.append(f"protected_family={match['protected_family']}")
            if match.get("preserve_raw_in_profile"):
                reason_parts.append("preserve_raw_in_profile=true")
            if match.get("suspicious_mapping"):
                reason_parts.append("suspicious_mapping=true")
            if match.get("vector_exclusion_reason"):
                reason_parts.append(str(match.get("vector_exclusion_reason")))
        rows.append(
            {
                "image_file": image_file,
                "parsed_product_name": parsed_product_name,
                "raw_ingredient": raw_ingredient,
                "normalized_raw": normalized_raw,
                "matched_standard_name": str((match or {}).get("functional_ingredient_name", "") or ""),
                "relation_type": str((match or {}).get("relation_type", "") or ""),
                "confidence": float((match or {}).get("confidence", 0.0) or 0.0) if match else "",
                "match_method": str((match or {}).get("match_source", "") or ""),
                "reason": "; ".join(reason_parts),
                "is_used_in_vector": bool((match or {}).get("is_used_in_vector", False)),
            }
        )
    return rows


def process_image(
    upload_service: UploadRecommendationService,
    image_path: Path,
    logical_image_file: str = "",
) -> Dict[str, Any]:
    image_label = str(logical_image_file or image_path.name)
    started = perf_counter()
    errors: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []
    official_recommendations: List[Dict[str, Any]] = []
    uploaded_self_matches: List[Dict[str, Any]] = []
    temp_vector: Dict[str, float] = {}
    matched: List[Dict[str, Any]] = []
    parsed: Dict[str, Any] = {
        "product_name_candidate": "",
        "raw_ingredients": [],
        "ingredient_objects": [],
    }
    ocr_result: Dict[str, Any] = {}
    response: Optional[Dict[str, Any]] = None
    status = "error"
    error_message = ""

    try:
        ocr_result = extract_text_from_image(str(image_path))
        raw_text = str(ocr_result.get("raw_text", "") or "")
        if ocr_result.get("error") or not raw_text.strip():
            status = "no_ocr_text"
            error_message = str(ocr_result.get("error") or "OCR text is empty")
            failures.append(
                {
                    "image_file": image_label,
                    "stage": "ocr",
                    "error_message": error_message,
                    "ocr_text_preview": preview_text(raw_text),
                    "parsed_raw_ingredients_preview": "",
                }
            )
        else:
            with no_parse_cache_writes():
                parsed = ingredient_parse_service.parse_ingredients_from_ocr_text(
                    raw_text,
                    upload_service.recommendation_service.runtime["sqlite_path"],
                )
            parsed["ocr_confidence"] = ocr_result.get("confidence")
            parsed["ocr_confidence_source"] = ocr_result.get("confidence_source", "unavailable")

            ingredient_objects = list(parsed.get("ingredient_objects", []) or [])
            if not ingredient_objects:
                status = "no_ingredients_parsed"
                error_message = "No ingredient objects were parsed from OCR text"
                failures.append(
                    {
                        "image_file": image_label,
                        "stage": "parse",
                        "error_message": error_message,
                        "ocr_text_preview": preview_text(raw_text),
                        "parsed_raw_ingredients_preview": join_preview(parsed.get("raw_ingredients", [])),
                    }
                )
            else:
                matched = upload_service.match_raw_ingredients_to_functional_ingredients(ingredient_objects)
                if not matched:
                    status = "no_matched_ingredients"
                    error_message = "No functional ingredient matches were produced"
                    failures.append(
                        {
                            "image_file": image_label,
                            "stage": "match",
                            "error_message": error_message,
                            "ocr_text_preview": preview_text(raw_text),
                            "parsed_raw_ingredients_preview": join_preview(parsed.get("raw_ingredients", [])),
                        }
                    )
                temp_vector = upload_service.build_temp_product_vector_from_ingredients(ingredient_objects, matched)
                if not temp_vector:
                    status = "no_vector"
                    error_message = "Upload vector is empty"
                    failures.append(
                        {
                            "image_file": image_label,
                            "stage": "vector",
                            "error_message": error_message,
                            "ocr_text_preview": preview_text(raw_text),
                            "parsed_raw_ingredients_preview": join_preview(parsed.get("raw_ingredients", [])),
                        }
                    )
                if temp_vector:
                    temp_hash = hashlib.sha1(
                        build_upload_signature("ocr_image", build_image_hash(image_path.read_bytes())).encode("utf-8")
                    ).hexdigest()[:16]
                    name_hint = upload_service.infer_category_from_product_name(parsed.get("product_name_candidate", image_label))
                    temp_profile = upload_service._build_temp_profile(
                        f"uploaded::{temp_hash}",
                        parsed.get("product_name_candidate") or image_label,
                        matched,
                        name_hint,
                    )
                    temp_profile["upload_signature"] = build_upload_signature("ocr_image", build_image_hash(image_path.read_bytes()))
                    temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
                    temp_profile = upload_service._apply_domain_category_overrides(parsed, temp_profile)
                    temp_profile["profile_signature"] = build_profile_signature(temp_profile, temp_vector)
                    recommendations = upload_service.calculate_similar_products_for_temp_vector(
                        temp_vector,
                        temp_profile,
                        TOP_K,
                        CANDIDATE_LIMIT,
                    )
                    try:
                        recommendations = upload_service._prepend_stored_exact_match_if_available(recommendations, temp_profile)
                    except Exception as exc:  # noqa: BLE001
                        errors.append({"stage": "prepend_exact", "message": str(exc)})
                    response = build_runtime_image_response(
                        upload_service,
                        image_path,
                        ocr_result,
                        parsed,
                        matched,
                        temp_vector,
                        recommendations,
                        perf_counter() - started,
                    )
                    official_recommendations = list(response.get("official_recommendations", recommendations))
                    uploaded_self_matches = list(
                        response.get("uploaded_self_matches", response.get("uploaded_similar_cases", []))
                    )
                    if not official_recommendations:
                        status = "no_recommendations"
                        error_message = "No official recommendations were generated"
                        failures.append(
                            {
                                "image_file": image_label,
                                "stage": "recommendation",
                                "error_message": error_message,
                                "ocr_text_preview": preview_text(raw_text),
                                "parsed_raw_ingredients_preview": join_preview(parsed.get("raw_ingredients", [])),
                            }
                        )
                    elif status not in {"no_matched_ingredients", "no_vector"}:
                        status = "success"
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_message = str(exc)
        errors.append({"stage": "exception", "message": str(exc), "traceback": traceback.format_exc()})
        failures.append(
            {
                "image_file": image_label,
                "stage": "exception",
                "error_message": str(exc),
                "ocr_text_preview": preview_text(str(ocr_result.get("raw_text", "") or "")),
                "parsed_raw_ingredients_preview": join_preview(parsed.get("raw_ingredients", [])),
            }
        )

    warnings = build_quality_warnings(
        list(parsed.get("ingredient_objects", []) or []),
        matched,
        temp_vector,
        response or {"quality_warnings": []},
    )

    details = {
        "image_file": image_label,
        "image_path": str(image_path),
        "ocr_text": str((ocr_result or {}).get("raw_text", "") or ""),
        "parsed_product_name": str(parsed.get("product_name_candidate", "") or ""),
        "parsed_raw_ingredients": list(parsed.get("raw_ingredients", []) or []),
        "matched_ingredients": [
            {
                "raw_ingredient": str(item.get("raw_ingredient", "") or ""),
                "normalized_raw": str(item.get("normalized_for_matching", "") or ""),
                "matched_standard_name": str(item.get("functional_ingredient_name", "") or ""),
                "relation_type": str(item.get("relation_type", "") or ""),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "match_method": str(item.get("match_source", "") or ""),
                "protected_family": str(item.get("protected_family", "") or ""),
                "preserve_raw_in_profile": bool(item.get("preserve_raw_in_profile")),
                "suspicious_mapping": bool(item.get("suspicious_mapping")),
            }
            for item in matched
        ],
        "generated_upload_vector": {str(key): float(value) for key, value in temp_vector.items()},
        "top_recommendations": [
            {
                "rank": int(item.get("rank", 0) or 0),
                "recommended_product_id": str(item.get("target_report_no", "") or ""),
                "recommended_product_name": str(item.get("target_product_name", "") or ""),
                "similarity_score": float(item.get("similarity_score", 0.0) or 0.0),
                "similarity_label": str(item.get("substitutability", "") or ""),
                "shared_ingredients": list(item.get("shared_ingredients", []) or []),
                "explanation": str(item.get("reason", "") or ""),
            }
            for item in official_recommendations[:TOP_K]
        ],
        "official_top_recommendations": [
            {
                "rank": int(item.get("rank", 0) or 0),
                "recommended_product_id": str(item.get("target_report_no", "") or ""),
                "recommended_product_name": str(item.get("target_product_name", "") or ""),
                "similarity_score": float(item.get("similarity_score", 0.0) or 0.0),
                "similarity_label": str(item.get("substitutability", "") or ""),
                "shared_ingredients": list(item.get("shared_ingredients", []) or []),
                "explanation": str(item.get("reason", "") or ""),
            }
            for item in official_recommendations[:TOP_K]
        ],
        "uploaded_self_matches": [
            {
                "rank": int(item.get("rank", 0) or 0),
                "recommended_product_id": str(item.get("target_report_no", "") or ""),
                "recommended_product_name": str(item.get("target_product_name", "") or ""),
                "similarity_score": float(item.get("similarity_score", 0.0) or 0.0),
                "similarity_label": str(item.get("substitutability", "") or ""),
                "shared_ingredients": list(item.get("shared_ingredients", []) or []),
                "explanation": str(item.get("reason", "") or ""),
            }
            for item in uploaded_self_matches
        ],
        "errors": errors,
        "warnings": warnings,
    }

    top1 = official_recommendations[0] if official_recommendations else {}
    summary_row = {
        "image_file": image_label,
        "status": status,
        "ocr_text_length": len(str((ocr_result or {}).get("raw_text", "") or "")),
        "parsed_ingredient_count": len(list(parsed.get("raw_ingredients", []) or [])),
        "matched_ingredient_count": len(matched),
        "vector_size": len(temp_vector),
        "recommendation_count": len(official_recommendations),
        "top1_product_name": str(top1.get("target_product_name", "") or ""),
        "top1_similarity_score": float(top1.get("similarity_score", 0.0) or 0.0) if top1 else "",
        "warning_count": len(warnings),
        "error_message": error_message,
    }

    return {
        "summary": summary_row,
        "details": details,
        "match_rows": match_rows_for_image(
            image_label,
            str(parsed.get("product_name_candidate", "") or ""),
            list(parsed.get("ingredient_objects", []) or []),
            matched,
            temp_vector,
        ),
        "recommendation_rows": recommendation_rows_for_image(
            image_label,
            str(parsed.get("product_name_candidate", "") or ""),
            official_recommendations,
        ),
        "failure_rows": failures,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    service, upload_service = build_test_services()

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    recommendation_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    existing_images = [resolve_image_path(name) for name in IMAGE_FILES if resolve_image_path(name) is not None]
    print(f"처리 대상 이미지 수: {len(existing_images)} / {len(IMAGE_FILES)}")

    for index, image_file in enumerate(IMAGE_FILES, start=1):
        image_path = resolve_image_path(image_file)
        print(f"[{index}/{len(IMAGE_FILES)}] {image_file} 처리 중...")
        if image_path is None:
            summary_rows.append(
                {
                    "image_file": image_file,
                    "status": "error",
                    "ocr_text_length": 0,
                    "parsed_ingredient_count": 0,
                    "matched_ingredient_count": 0,
                    "vector_size": 0,
                    "recommendation_count": 0,
                    "top1_product_name": "",
                    "top1_similarity_score": "",
                    "warning_count": 0,
                    "error_message": "image not found",
                }
            )
            detail_rows.append(
                {
                    "image_file": image_file,
                    "image_path": str(INPUT_DIR / image_file),
                    "ocr_text": "",
                    "parsed_product_name": "",
                    "parsed_raw_ingredients": [],
                    "matched_ingredients": [],
                    "generated_upload_vector": {},
                    "top_recommendations": [],
                    "errors": [{"stage": "file_check", "message": "image not found"}],
                    "warnings": [],
                }
            )
            failure_rows.append(
                {
                    "image_file": image_file,
                    "stage": "file_check",
                    "error_message": "image not found",
                    "ocr_text_preview": "",
                    "parsed_raw_ingredients_preview": "",
                }
            )
            print("  status: error (image not found)")
            continue

        result = process_image(upload_service, image_path, logical_image_file=image_file)
        summary_rows.append(result["summary"])
        detail_rows.append(result["details"])
        match_rows.extend(result["match_rows"])
        recommendation_rows.extend(result["recommendation_rows"])
        failure_rows.extend(result["failure_rows"])
        print(f"  OCR length: {result['summary']['ocr_text_length']}")
        print(f"  parsed ingredients: {result['summary']['parsed_ingredient_count']}")
        print(f"  matched ingredients: {result['summary']['matched_ingredient_count']}")
        print(f"  vector size: {result['summary']['vector_size']}")
        print(f"  recommendations: {result['summary']['recommendation_count']}")
        print(f"  status: {result['summary']['status']}")

    write_csv(SUMMARY_CSV, SUMMARY_FIELDS, summary_rows)
    write_csv(MATCHES_CSV, MATCH_FIELDS, match_rows)
    write_csv(RECOMMENDATIONS_CSV, RECOMMENDATION_FIELDS, recommendation_rows)
    write_csv(FAILURES_CSV, FAILURE_FIELDS, failure_rows)
    DETAILS_JSON.write_text(json.dumps(detail_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    status_counts = Counter(str(row.get("status", "") or "") for row in summary_rows)
    failure_images = [row["image_file"] for row in summary_rows if row.get("status") != "success"]
    warning_counter = Counter()
    for item in detail_rows:
        for warning in item.get("warnings", []):
            code = str((warning or {}).get("code", "") or "unknown_warning")
            warning_counter[code] += 1

    print("")
    print(f"총 이미지 수: {len(IMAGE_FILES)}")
    print(f"success 수: {status_counts.get('success', 0)}")
    print(f"실패 수: {len(IMAGE_FILES) - status_counts.get('success', 0)}")
    print(f"status별 개수: {dict(status_counts)}")
    print(f"평균 OCR 길이: {safe_mean(float(row.get('ocr_text_length', 0) or 0) for row in summary_rows)}")
    print(f"평균 parsed ingredient 수: {safe_mean(float(row.get('parsed_ingredient_count', 0) or 0) for row in summary_rows)}")
    print(f"평균 matched ingredient 수: {safe_mean(float(row.get('matched_ingredient_count', 0) or 0) for row in summary_rows)}")
    print(f"평균 vector size: {safe_mean(float(row.get('vector_size', 0) or 0) for row in summary_rows)}")
    print(f"추천 결과 생성 성공 수: {sum(1 for row in summary_rows if int(row.get('recommendation_count', 0) or 0) > 0)}")
    print(f"실패 이미지 목록: {failure_images}")
    print(f"warning 요약: {dict(warning_counter)}")
    print("출력 파일 경로:")
    print(f"  {SUMMARY_CSV}")
    print(f"  {DETAILS_JSON}")
    print(f"  {MATCHES_CSV}")
    print(f"  {RECOMMENDATIONS_CSV}")
    print(f"  {FAILURES_CSV}")


if __name__ == "__main__":
    main()
