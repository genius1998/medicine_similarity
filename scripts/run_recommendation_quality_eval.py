from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.ingredient_parse_service import canonicalize_ingredient_for_matching
from api.ocr_service import extract_text_from_image
from api.recommendation_service import RecommendationService
from api.upload_recommendation_service import UploadRecommendationService


DEFAULT_DATASET = REPO_ROOT / "tests" / "fixtures" / "recommendation_eval_set.json"
OUTPUT_JSON = REPO_ROOT / "output" / "recommendation_quality_eval_report.json"
OUTPUT_CSV = REPO_ROOT / "output" / "recommendation_quality_eval_report.csv"

GENERIC_BAD_CATEGORY_MAP = {
    "혈당": ["눈 건강", "간 건강", "관절/연골", "수면/긴장완화"],
    "간 건강": ["눈 건강", "혈당", "장 건강", "관절/연골"],
    "면역": ["눈 건강", "혈당", "관절/연골"],
    "장 건강": ["눈 건강", "간 건강", "혈당", "관절/연골"],
    "관절/연골": ["눈 건강", "혈당", "장 건강", "수면/긴장완화"],
    "눈 건강": ["간 건강", "혈당", "장 건강", "관절/연골"],
}


def _canonical_set(values: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values or []:
        normalized = canonicalize_ingredient_for_matching(str(value or ""))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _recommendation_matches_category(rec: dict, expected_category: str) -> bool:
    expected = str(expected_category or "").strip()
    if not expected:
        return False
    shared_categories = [str(value or "").strip() for value in rec.get("shared_categories", []) or []]
    return expected in shared_categories


def _recommendation_matches_primary(rec: dict, expected_primary: List[str]) -> bool:
    if not expected_primary:
        return False
    shared = set(_canonical_set([str(value or "") for value in rec.get("shared_ingredients", []) or []]))
    expected = set(_canonical_set(expected_primary))
    return bool(shared & expected)


def _recommendation_matches_keywords(rec: dict, keywords: List[str]) -> bool:
    if not keywords:
        return False
    name = str(rec.get("product_name", "") or "")
    lowered = name.lower()
    return any(str(keyword or "").lower() in lowered for keyword in keywords if str(keyword or "").strip())


def _first_official(recommendations: List[dict]) -> dict:
    for rec in recommendations:
        report_no = str(rec.get("report_no", "") or "")
        if report_no and not report_no.startswith("UPLOADED-"):
            return rec
    return {}


def _official_top_k(recommendations: List[dict], top_k: int) -> List[dict]:
    official = [rec for rec in recommendations if not str(rec.get("report_no", "") or "").startswith("UPLOADED-")]
    return official[:top_k]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator / denominator), 4) if denominator else 0.0


def _normalize_expected_groups(case: dict) -> List[List[str]]:
    raw_groups = list(case.get("expected_primary_groups", []) or [])
    if raw_groups:
        groups: List[List[str]] = []
        for group in raw_groups:
            normalized = _canonical_set([str(value or "") for value in group or []])
            if normalized:
                groups.append(normalized)
        return groups
    flat = _canonical_set([str(value or "") for value in case.get("expected_primary_ingredients", []) or []])
    return [[value] for value in flat]


def _track_function_success(category_hit: Any, primary_hit: Any) -> bool:
    return bool(category_hit) or bool(primary_hit)


def _recommendation_matches_any_keywords(rec: dict, keywords: List[str]) -> bool:
    return _recommendation_matches_keywords(rec, keywords)


def _infer_bad_recommendation_label(rec: dict, expected_category: str, category_hit: Any, primary_hit: Any) -> str:
    if not rec:
        return "no_official_candidate"
    shared_categories = [str(value or "").strip() for value in rec.get("shared_categories", []) or []]
    if category_hit is False:
        for bad_category in GENERIC_BAD_CATEGORY_MAP.get(str(expected_category or "").strip(), []):
            if bad_category in shared_categories:
                return f"explicit_bad_category:{bad_category}"
    if category_hit is False and primary_hit is False:
        return "category_and_primary_miss"
    return ""


def _is_uploaded_recommendation(rec: dict) -> bool:
    report_no = str(rec.get("report_no", "") or "")
    return report_no.startswith("UPLOADED-")


def _first_official_rank(recommendations: List[dict]) -> int:
    for index, rec in enumerate(recommendations, start=1):
        if not _is_uploaded_recommendation(rec):
            return index
    return 0


def _resolve_case_image_path(value: object) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mini recommendation quality evaluation.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to evaluation dataset JSON.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations to request.")
    parser.add_argument("--candidate-limit", type=int, default=300, help="Candidate limit for recommendation search.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = list(dataset.get("cases", []) or [])

    service = UploadRecommendationService(RecommendationService())
    results: List[Dict[str, Any]] = []

    parsed_category_total = 0
    parsed_category_hits = 0
    parsed_primary_recall_sum = 0.0
    parsed_primary_all_match_hits = 0
    parsed_primary_total = 0

    official_category_hits_at_5 = 0
    official_category_total = 0
    official_primary_hits_at_5 = 0
    official_primary_total = 0
    official_keyword_hits_at_5 = 0
    official_keyword_total = 0
    acceptable_top_product_hits_at_5 = 0
    acceptable_top_product_total = 0
    top1_bad_recommendation_count = 0
    top1_bad_recommendation_total = 0
    top1_missing_official_count = 0
    top5_uploaded_count_sum = 0
    top5_official_count_sum = 0
    top5_uploaded_dominance_count = 0
    top5_exact_top1_count = 0
    first_official_rank_sum = 0
    first_official_rank_count = 0

    input_mode_counts: Dict[str, int] = {}
    keyword_sensitive_total = 0
    keyword_sensitive_success_hits = 0
    function_first_total = 0
    function_first_success_hits = 0
    function_first_keyword_expectation_total = 0
    function_first_keyword_miss_but_function_hit_count = 0

    for case in cases:
        image_path = _resolve_case_image_path(case.get("image_path", ""))
        input_mode = str(case.get("input_mode", "image") or "image").strip() or "image"
        evaluation_track = str(case.get("evaluation_track", "keyword_sensitive") or "keyword_sensitive").strip() or "keyword_sensitive"
        input_mode_counts[input_mode] = input_mode_counts.get(input_mode, 0) + 1

        if input_mode == "ocr_text_from_image":
            ocr_payload = extract_text_from_image(str(image_path))
            if ocr_payload.get("error"):
                payload = {
                    "parsed": {"primary_ingredients_normalized": []},
                    "estimated_profile": {"product_main_category": ""},
                    "recommendations": [],
                    "quality": {"status": "review_needed", "quality_grade": "D", "profile_confidence": 0.0},
                    "quality_warnings": [{"code": "ocr_error", "message": str(ocr_payload.get("error"))}],
                    "ocr_error": str(ocr_payload.get("error")),
                }
            else:
                payload = service.recommend_from_ocr_text(
                    str(ocr_payload.get("raw_text", "") or ""),
                    top_k=args.top_k,
                    candidate_limit=args.candidate_limit,
                )
        else:
            payload = service.recommend_from_uploaded_image(
                image_path.read_bytes(),
                image_path.name,
                top_k=args.top_k,
                candidate_limit=args.candidate_limit,
            )

        expected_category = str(case.get("expected_category", "") or "").strip()
        expected_primary = [str(value or "").strip() for value in case.get("expected_primary_ingredients", []) or [] if str(value or "").strip()]
        expected_primary_groups = _normalize_expected_groups(case)
        acceptable_keywords = [str(value or "").strip() for value in case.get("acceptable_product_keywords", []) or [] if str(value or "").strip()]
        acceptable_top_product_keywords = [
            str(value or "").strip() for value in case.get("acceptable_top_product_keywords", []) or [] if str(value or "").strip()
        ]

        parsed_primary = _canonical_set([str(value or "") for value in payload.get("parsed", {}).get("primary_ingredients_normalized", []) or []])
        parsed_primary_overlap: List[str] = []
        matched_group_count = 0
        for group in expected_primary_groups:
            overlap = sorted(set(parsed_primary) & set(group))
            if overlap:
                matched_group_count += 1
                parsed_primary_overlap.extend(overlap)
        parsed_primary_overlap = sorted(dict.fromkeys(parsed_primary_overlap))
        parsed_primary_recall = _safe_ratio(matched_group_count, len(expected_primary_groups)) if expected_primary_groups else 0.0

        estimated_category = str(payload.get("estimated_profile", {}).get("product_main_category", "") or "").strip()
        parsed_category_match = bool(expected_category and estimated_category == expected_category)

        recommendations = list(payload.get("recommendations", []) or [])
        official_top5 = _official_top_k(recommendations, 5)
        first_official = _first_official(recommendations)
        first_official_rank = _first_official_rank(recommendations)
        top5 = recommendations[:5]
        top5_uploaded_count = sum(1 for rec in top5 if _is_uploaded_recommendation(rec))
        top5_official_count = len(top5) - top5_uploaded_count
        exact_top1 = bool(top5 and bool(top5[0].get("exact_same_upload")))

        official_category_hit = any(_recommendation_matches_category(rec, expected_category) for rec in official_top5) if expected_category else None
        official_primary_hit = any(_recommendation_matches_primary(rec, expected_primary) for rec in official_top5) if expected_primary else None
        official_keyword_hit = any(_recommendation_matches_keywords(rec, acceptable_keywords) for rec in official_top5) if acceptable_keywords else None
        acceptable_top_product_hit = (
            any(_recommendation_matches_any_keywords(rec, acceptable_top_product_keywords) for rec in official_top5)
            if acceptable_top_product_keywords
            else None
        )
        function_first_success = _track_function_success(official_category_hit, official_primary_hit)
        keyword_sensitive_success = bool(official_keyword_hit) if acceptable_keywords else None
        top1_category_hit = _recommendation_matches_category(first_official, expected_category) if first_official and expected_category else None
        top1_primary_hit = _recommendation_matches_primary(first_official, expected_primary) if first_official and expected_primary else None
        top1_bad_recommendation_label = _infer_bad_recommendation_label(first_official, expected_category, top1_category_hit, top1_primary_hit)

        if expected_category:
            parsed_category_total += 1
            if parsed_category_match:
                parsed_category_hits += 1
            official_category_total += 1
            if official_category_hit:
                official_category_hits_at_5 += 1

        if expected_primary_groups:
            parsed_primary_total += 1
            parsed_primary_recall_sum += parsed_primary_recall
            if parsed_primary_recall == 1.0:
                parsed_primary_all_match_hits += 1
            official_primary_total += 1
            if official_primary_hit:
                official_primary_hits_at_5 += 1

        if acceptable_keywords:
            official_keyword_total += 1
            if official_keyword_hit:
                official_keyword_hits_at_5 += 1

        if acceptable_top_product_keywords:
            acceptable_top_product_total += 1
            if acceptable_top_product_hit:
                acceptable_top_product_hits_at_5 += 1

        if first_official:
            top1_bad_recommendation_total += 1
            if top1_bad_recommendation_label:
                top1_bad_recommendation_count += 1
            first_official_rank_sum += first_official_rank
            first_official_rank_count += 1
        else:
            top1_missing_official_count += 1

        top5_uploaded_count_sum += top5_uploaded_count
        top5_official_count_sum += top5_official_count
        if top5_uploaded_count > top5_official_count:
            top5_uploaded_dominance_count += 1
        if exact_top1:
            top5_exact_top1_count += 1

        if evaluation_track == "keyword_sensitive":
            keyword_sensitive_total += 1
            if keyword_sensitive_success:
                keyword_sensitive_success_hits += 1
        elif evaluation_track == "function_first":
            function_first_total += 1
            if function_first_success:
                function_first_success_hits += 1
            if acceptable_keywords:
                function_first_keyword_expectation_total += 1
            if function_first_success and acceptable_keywords and keyword_sensitive_success is False:
                function_first_keyword_miss_but_function_hit_count += 1

        results.append(
            {
                "case_id": str(case.get("case_id", "") or image_path.stem),
                "image_path": str(image_path),
                "input_mode": input_mode,
                "evaluation_track": evaluation_track,
                "expected_category": expected_category,
                "estimated_category": estimated_category,
                "parsed_category_match": parsed_category_match,
                "expected_primary_ingredients": expected_primary,
                "parsed_primary_ingredients": parsed_primary,
                "parsed_primary_overlap": parsed_primary_overlap,
                "parsed_primary_recall": parsed_primary_recall,
                "status": str(payload.get("quality", {}).get("status", "") or ""),
                "quality_grade": str(payload.get("quality", {}).get("quality_grade", "") or ""),
                "profile_confidence": float(payload.get("quality", {}).get("profile_confidence", 0.0) or 0.0),
                "first_official_report_no": str(first_official.get("report_no", "") or ""),
                "first_official_rank": first_official_rank or None,
                "first_official_product_name": str(first_official.get("product_name", "") or ""),
                "first_official_similarity_score": float(first_official.get("similarity_score", 0.0) or 0.0) if first_official else 0.0,
                "top5_uploaded_count": top5_uploaded_count,
                "top5_official_count": top5_official_count,
                "top5_uploaded_dominance": top5_uploaded_count > top5_official_count,
                "top1_exact_same_upload": exact_top1,
                "official_category_hit_at_5": official_category_hit,
                "official_primary_hit_at_5": official_primary_hit,
                "official_keyword_hit_at_5": official_keyword_hit,
                "acceptable_top_product_hit_at_5": acceptable_top_product_hit,
                "function_first_success_at_5": function_first_success,
                "keyword_sensitive_success_at_5": keyword_sensitive_success,
                "top1_bad_recommendation_label": top1_bad_recommendation_label,
                "official_top5_report_nos": [str(rec.get("report_no", "") or "") for rec in official_top5],
                "official_top5_product_names": [str(rec.get("product_name", "") or "") for rec in official_top5],
                "official_top5_shared_categories": [list(rec.get("shared_categories", []) or []) for rec in official_top5],
                "official_top5_shared_ingredients": [list(rec.get("shared_ingredients", []) or []) for rec in official_top5],
                "warnings": [str(item.get("message", "") or item) for item in payload.get("quality_warnings", []) or []],
            }
        )

    summary = {
        "dataset_path": str(dataset_path),
        "dataset_version": str(dataset.get("version", "") or ""),
        "total_cases": len(results),
        "input_mode_counts": input_mode_counts,
        "parsed_category_accuracy": _safe_ratio(parsed_category_hits, parsed_category_total),
        "parsed_primary_recall_avg": round(parsed_primary_recall_sum / parsed_primary_total, 4) if parsed_primary_total else 0.0,
        "parsed_primary_all_match_rate": _safe_ratio(parsed_primary_all_match_hits, parsed_primary_total),
        "official_category_hit_at_5": _safe_ratio(official_category_hits_at_5, official_category_total),
        "official_primary_hit_at_5": _safe_ratio(official_primary_hits_at_5, official_primary_total),
        "official_keyword_hit_at_5": _safe_ratio(official_keyword_hits_at_5, official_keyword_total),
        "acceptable_top_product_case_count": acceptable_top_product_total,
        "acceptable_top_product_recall_at_5": _safe_ratio(acceptable_top_product_hits_at_5, acceptable_top_product_total),
        "keyword_sensitive_case_count": keyword_sensitive_total,
        "keyword_sensitive_success_at_5": _safe_ratio(keyword_sensitive_success_hits, keyword_sensitive_total),
        "function_first_case_count": function_first_total,
        "function_first_success_at_5": _safe_ratio(function_first_success_hits, function_first_total),
        "function_first_keyword_expectation_case_count": function_first_keyword_expectation_total,
        "function_first_keyword_miss_but_function_hit_count": function_first_keyword_miss_but_function_hit_count,
        "top1_bad_recommendation_case_count": top1_bad_recommendation_total,
        "top1_bad_recommendation_rate": _safe_ratio(top1_bad_recommendation_count, top1_bad_recommendation_total),
        "top1_missing_official_count": top1_missing_official_count,
        "top1_missing_official_rate": _safe_ratio(top1_missing_official_count, len(results)),
        "avg_top5_uploaded_count": round(top5_uploaded_count_sum / len(results), 4) if results else 0.0,
        "avg_top5_official_count": round(top5_official_count_sum / len(results), 4) if results else 0.0,
        "top5_uploaded_dominance_rate": _safe_ratio(top5_uploaded_dominance_count, len(results)),
        "top1_exact_same_upload_rate": _safe_ratio(top5_exact_top1_count, len(results)),
        "avg_first_official_rank": round(first_official_rank_sum / first_official_rank_count, 4) if first_official_rank_count else 0.0,
    }

    OUTPUT_JSON.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "case_id",
        "image_path",
        "input_mode",
        "evaluation_track",
        "expected_category",
        "estimated_category",
        "parsed_category_match",
        "parsed_primary_recall",
        "status",
        "quality_grade",
        "profile_confidence",
        "first_official_report_no",
        "first_official_rank",
        "first_official_product_name",
        "first_official_similarity_score",
        "top5_uploaded_count",
        "top5_official_count",
        "top5_uploaded_dominance",
        "top1_exact_same_upload",
        "official_category_hit_at_5",
        "official_primary_hit_at_5",
        "official_keyword_hit_at_5",
        "acceptable_top_product_hit_at_5",
        "function_first_success_at_5",
        "keyword_sensitive_success_at_5",
        "top1_bad_recommendation_label",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key) for key in fieldnames})

    print(json.dumps({"summary": summary, "output_json": str(OUTPUT_JSON), "output_csv": str(OUTPUT_CSV)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
