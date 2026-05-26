from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.config import get_settings
from api.ocr_service import extract_text_from_image
from api.recommendation_service import CATALOG_TABLE_NAME, RecommendationService
from api.upload_recommendation_service import (
    UploadRecommendationService,
    build_candidate_pool,
    calculate_core_match_score,
    calculate_function_similarity,
    calculate_weighted_jaccard_with_idf,
    compare_product_profiles,
)


DEFAULT_DATASET = REPO_ROOT / "tests" / "fixtures" / "recommendation_eval_set.json"
DEFAULT_EVAL_REPORT = REPO_ROOT / "output" / "recommendation_quality_eval_report.json"
OUTPUT_JSON = REPO_ROOT / "output" / "official_candidate_gap_report.json"
OUTPUT_CSV = REPO_ROOT / "output" / "official_candidate_gap_report.csv"


def _is_official_profile(profile: dict) -> bool:
    report_no = str((profile or {}).get("report_no", "") or "")
    return bool(report_no) and not report_no.startswith("UPLOADED-")


def _recommendation_mode_payload(
    service: UploadRecommendationService,
    image_path: Path,
    input_mode: str,
    top_k: int,
    candidate_limit: int,
) -> dict:
    if input_mode == "ocr_text_from_image":
        ocr_payload = extract_text_from_image(str(image_path))
        if ocr_payload.get("error"):
            return {
                "parsed": {"ingredient_objects": [], "primary_ingredients_normalized": []},
                "estimated_profile": {"product_id": "", "product_name": "", "product_main_category": ""},
                "recommendations": [],
                "quality": {"status": "review_needed", "quality_grade": "D", "profile_confidence": 0.0},
                "quality_warnings": [{"code": "ocr_error", "message": str(ocr_payload.get("error"))}],
                "ocr_error": str(ocr_payload.get("error")),
            }
        return service.recommend_from_ocr_text(
            str(ocr_payload.get("raw_text", "") or ""),
            top_k=top_k,
            candidate_limit=candidate_limit,
        )
    return service.recommend_from_uploaded_image(
        image_path.read_bytes(),
        image_path.name,
        top_k=top_k,
        candidate_limit=candidate_limit,
    )


def _build_bruteforce_official_top(
    service: UploadRecommendationService,
    temp_profile: dict,
    temp_vector: Dict[str, float],
    top_k: int,
) -> List[dict]:
    rec_service = service.recommendation_service
    rows: List[dict] = []
    total_product_count = len(rec_service.product_vectors)
    for product_id, target_profile in rec_service.profiles.items():
        if not _is_official_profile(target_profile):
            continue
        target_vector = rec_service.product_vectors.get(product_id, {})
        if not target_vector:
            continue
        similarity_score, _ = calculate_weighted_jaccard_with_idf(
            temp_vector,
            target_vector,
            rec_service.ingredient_frequency,
            total_product_count,
            temp_profile,
            target_profile,
        )
        if similarity_score <= 0:
            continue
        comparison = compare_product_profiles(temp_profile, target_profile, temp_vector, target_vector)
        rows.append(
            {
                "product_id": str(product_id),
                "report_no": str(target_profile.get("report_no", "") or ""),
                "product_name": str(target_profile.get("product_name", "") or ""),
                "similarity_score": float(similarity_score),
                "function_similarity_score": float(calculate_function_similarity(temp_profile, target_profile)),
                "core_match_score": float(calculate_core_match_score(temp_profile, target_profile)),
                "shared_categories": list(comparison.get("shared_categories", []) or []),
                "shared_ingredients": list(comparison.get("shared_ingredients", []) or []),
            }
        )
    rows.sort(
        key=lambda item: (
            -item["similarity_score"],
            -item["core_match_score"],
            -item["function_similarity_score"],
            item["report_no"],
        )
    )
    return rows[:top_k]


def _rebuild_temp_profile(service: UploadRecommendationService, parsed: dict, image_path: Path, input_mode: str) -> tuple[dict, Dict[str, float]]:
    ingredient_objects = list(parsed.get("ingredient_objects", []) or [])
    matched = service.match_raw_ingredients_to_functional_ingredients(ingredient_objects)
    temp_vector = service.build_temp_product_vector_from_ingredients(ingredient_objects)
    product_name_candidate = str(parsed.get("product_name_candidate", "") or image_path.name)
    name_hint = service.infer_category_from_product_name(product_name_candidate)
    temp_hash = hashlib.sha1(f"{input_mode}:{image_path}".encode("utf-8")).hexdigest()[:16]
    temp_profile = service._build_temp_profile(  # noqa: SLF001
        f"eval::{temp_hash}",
        product_name_candidate,
        matched,
        name_hint,
    )
    temp_profile = service._apply_domain_category_overrides(parsed, temp_profile)  # noqa: SLF001
    return temp_profile, temp_vector


def _catalog_keyword_matches(rec_service: RecommendationService, keywords: List[str], limit: int = 10) -> List[dict]:
    normalized_keywords = [str(value or "").strip() for value in keywords if str(value or "").strip()]
    if not normalized_keywords or not rec_service.runtime:
        return []
    clauses = []
    params: List[str] = []
    for keyword in normalized_keywords:
        like = f"%{keyword}%"
        clauses.append("(product_name LIKE ? OR main_functionality LIKE ? OR raw_ingredients LIKE ?)")
        params.extend([like, like, like])
    query = f"""
        SELECT report_no, product_name, main_functionality
        FROM {CATALOG_TABLE_NAME}
        WHERE {" OR ".join(clauses)}
        LIMIT ?
    """
    params.append(str(limit))
    with sqlite3.connect(rec_service.runtime["sqlite_path"]) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "report_no": str(row["report_no"] or ""),
            "product_name": str(row["product_name"] or ""),
            "main_functionality": str(row["main_functionality"] or ""),
            "has_profile": str(row["report_no"] or "") in rec_service.profile_by_report_no,
        }
        for row in rows
    ]


def _classify_gap(official_in_candidate_pool: int, brute_force_top: List[dict], catalog_matches: List[dict]) -> tuple[str, str]:
    if brute_force_top:
        if official_in_candidate_pool == 0:
            return "candidate_retrieval_miss", "candidate_miss"
        return "candidate_ranking_miss", "candidate_miss"
    if any(not item.get("has_profile") for item in catalog_matches):
        return "official_catalog_profile_coverage_gap", "coverage_gap"
    if catalog_matches:
        return "official_catalog_similarity_gap", "coverage_gap"
    return "official_catalog_coverage_gap", "coverage_gap"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cases where official recommendation candidates are missing.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to evaluation dataset JSON.")
    parser.add_argument("--eval-report", default=str(DEFAULT_EVAL_REPORT), help="Path to recommendation eval report JSON.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations to request.")
    parser.add_argument("--candidate-limit", type=int, default=300, help="Candidate limit for recommendation search.")
    parser.add_argument("--bruteforce-top-k", type=int, default=10, help="Top-K official brute-force matches to collect.")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset_by_case = {str(case.get("case_id", "") or ""): case for case in dataset.get("cases", []) or []}
    eval_report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    gap_cases = [item for item in eval_report.get("results", []) or [] if not str(item.get("first_official_report_no", "") or "")]

    service = UploadRecommendationService(RecommendationService())
    service._ensure_loaded()
    rec_service = service.recommendation_service

    analyzed: List[Dict[str, Any]] = []
    for result_item in gap_cases:
        case_id = str(result_item.get("case_id", "") or "")
        case = dataset_by_case.get(case_id, {})
        image_path = Path(str(case.get("image_path", "") or ""))
        input_mode = str(case.get("input_mode", "image") or "image")
        payload = _recommendation_mode_payload(service, image_path, input_mode, args.top_k, args.candidate_limit)

        parsed = payload.get("parsed", {}) or {}
        temp_profile, temp_vector = _rebuild_temp_profile(service, parsed, image_path, input_mode)

        if temp_vector and temp_profile.get("product_id"):
            candidate_pool = build_candidate_pool(
                str(temp_profile.get("product_id", "") or ""),
                temp_profile,
                rec_service.product_vectors,
                rec_service.profiles,
                rec_service.ingredient_postings,
                rec_service.ingredient_frequency,
                args.candidate_limit,
                get_settings().default_max_df_for_seed,
            )
        else:
            candidate_pool = []

        official_in_candidate_pool = 0
        official_candidate_examples: List[dict] = []
        for candidate in candidate_pool:
            target_product_id = str(candidate.get("target_product_id", "") or "")
            target_profile = rec_service.profiles.get(target_product_id, {})
            if not _is_official_profile(target_profile):
                continue
            official_in_candidate_pool += 1
            if len(official_candidate_examples) < 5:
                official_candidate_examples.append(
                    {
                        "report_no": str(target_profile.get("report_no", "") or ""),
                        "product_name": str(target_profile.get("product_name", "") or ""),
                        "main_category": str(target_profile.get("product_main_category", "") or ""),
                    }
                )

        brute_force_top = _build_bruteforce_official_top(service, temp_profile, temp_vector, args.bruteforce_top_k) if temp_vector else []
        keywords = list(case.get("acceptable_top_product_keywords", []) or []) + list(case.get("acceptable_product_keywords", []) or [])
        catalog_matches = _catalog_keyword_matches(rec_service, keywords, limit=10)
        classification, root_type = _classify_gap(official_in_candidate_pool, brute_force_top, catalog_matches)

        analyzed.append(
            {
                "case_id": case_id,
                "input_mode": input_mode,
                "image_path": str(image_path),
                "expected_category": str(case.get("expected_category", "") or ""),
                "estimated_category": str(temp_profile.get("product_main_category", "") or ""),
                "parsed_primary_ingredients": list(parsed.get("primary_ingredients_normalized", []) or []),
                "status": str((payload.get("quality", {}) or {}).get("status", "") or ""),
                "quality_grade": str((payload.get("quality", {}) or {}).get("quality_grade", "") or ""),
                "profile_confidence": float((payload.get("quality", {}) or {}).get("profile_confidence", 0.0) or 0.0),
                "classification": classification,
                "root_type": root_type,
                "candidate_pool_size": len(candidate_pool),
                "official_candidate_pool_count": official_in_candidate_pool,
                "official_candidate_examples": official_candidate_examples,
                "brute_force_official_top_count": len(brute_force_top),
                "brute_force_official_top": brute_force_top,
                "catalog_keyword_match_count": len(catalog_matches),
                "catalog_keyword_matches": catalog_matches,
                "quality_warnings": list(payload.get("quality_warnings", []) or []),
            }
        )

    summary = {
        "dataset_version": str(dataset.get("version", "") or ""),
        "gap_case_count": len(analyzed),
        "classification_counts": {},
        "root_type_counts": {},
    }
    for item in analyzed:
        key = str(item.get("classification", "") or "unknown")
        summary["classification_counts"][key] = summary["classification_counts"].get(key, 0) + 1
        root_type = str(item.get("root_type", "") or "unknown")
        summary["root_type_counts"][root_type] = summary["root_type_counts"].get(root_type, 0) + 1

    OUTPUT_JSON.write_text(json.dumps({"summary": summary, "cases": analyzed}, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "case_id",
        "input_mode",
        "image_path",
        "expected_category",
        "estimated_category",
        "status",
        "quality_grade",
        "profile_confidence",
        "classification",
        "root_type",
        "candidate_pool_size",
        "official_candidate_pool_count",
        "brute_force_official_top_count",
        "catalog_keyword_match_count",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in analyzed:
            writer.writerow({key: item.get(key) for key in fieldnames})

    print(json.dumps({"summary": summary, "output_json": str(OUTPUT_JSON), "output_csv": str(OUTPUT_CSV)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
