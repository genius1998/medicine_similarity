from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.ocr_service import extract_text_from_image
from api.recommendation_service import RecommendationService
from api.upload_recommendation_service import UploadRecommendationService


OUTPUT_JSON = REPO_ROOT / "output" / "regression_duplicate_similarity_report.json"
OUTPUT_CSV = REPO_ROOT / "output" / "regression_duplicate_similarity_report.csv"


def _safe_top1(payload: dict) -> dict:
    recommendations = payload.get("recommendations") or []
    return recommendations[0] if recommendations else {}


def _extract_run_snapshot(payload: dict) -> dict:
    parsed = payload.get("parsed") or {}
    parse_metadata = parsed.get("parse_metadata") or {}
    estimated = payload.get("estimated_profile") or {}
    quality = payload.get("quality") or {}
    debug = payload.get("debug") or {}
    validator_result = parsed.get("validator_result") or {}
    top1 = _safe_top1(payload)
    return {
        "trace_id": payload.get("trace_id", ""),
        "top1_product_id": top1.get("report_no") or top1.get("target_report_no") or "",
        "top1_similarity_score": float(top1.get("similarity_score", 0.0) or 0.0),
        "image_hash": debug.get("image_hash", ""),
        "ocr_text_hash": debug.get("ocr_text_hash", parse_metadata.get("ocr_text_hash", "")),
        "parsed_signature": debug.get("parsed_signature", parse_metadata.get("parsed_signature", "")),
        "profile_signature": debug.get("profile_signature", estimated.get("profile_signature", "")),
        "parse_cache_hit": bool(debug.get("parse_cache_hit", parse_metadata.get("cache_hit", False))),
        "exact_match_detected": bool(debug.get("exact_match_detected", top1.get("exact_same_upload", False))),
        "status": quality.get("status", estimated.get("status", "")),
        "quality_grade": quality.get("quality_grade", estimated.get("quality_grade", "")),
        "candidate_scope": quality.get("candidate_scope", estimated.get("candidate_scope", "")),
        "candidate_disabled_reason": quality.get("candidate_disabled_reason", estimated.get("candidate_disabled_reason", "")),
        "warnings": [item.get("message", item.get("code", "")) for item in quality.get("warnings", [])],
        "saved_product": payload.get("saved_product"),
        "quality": quality,
        "validator_result": {
            "quality_grade": validator_result.get("quality_grade"),
            "profile_confidence": validator_result.get("profile_confidence"),
            "unknown_ratio": validator_result.get("unknown_ratio"),
            "corrected_fields": validator_result.get("corrected_fields", []),
            "warnings": [
                item.get("message", item.get("code", ""))
                for item in validator_result.get("warnings", [])
            ],
        },
        "debug": debug,
    }


def _run_image_duplicate(upload_service: UploadRecommendationService, image_path: Path, top_k: int, candidate_limit: int) -> List[dict]:
    image_bytes = image_path.read_bytes()
    results: List[dict] = []
    for attempt in range(2):
        payload = upload_service.recommend_from_uploaded_image(
            image_bytes,
            image_path.name,
            top_k=top_k,
            candidate_limit=candidate_limit,
        )
        snapshot = _extract_run_snapshot(payload)
        snapshot["attempt"] = attempt + 1
        results.append(snapshot)
    return results


def _run_ocr_text_duplicate(upload_service: UploadRecommendationService, raw_text: str, top_k: int, candidate_limit: int) -> List[dict]:
    results: List[dict] = []
    for attempt in range(2):
        payload = upload_service.recommend_from_ocr_text(raw_text, top_k=top_k, candidate_limit=candidate_limit)
        snapshot = _extract_run_snapshot(payload)
        snapshot["attempt"] = attempt + 1
        results.append(snapshot)
    return results


def _classify_failure(case: dict) -> List[str]:
    reasons: List[str] = []
    if case.get("second_similarity_score") != 1.0:
        reasons.append("second_similarity_not_1_0")
    if not case.get("exact_match_detected"):
        reasons.append("exact_match_not_detected")
    if not case.get("image_hash_equal"):
        reasons.append("image_hash_changed")
    if not case.get("ocr_text_hash_equal"):
        reasons.append("ocr_text_hash_changed")
    if not case.get("parsed_signature_equal"):
        reasons.append("parsed_signature_changed")
    if not case.get("profile_signature_equal"):
        reasons.append("profile_signature_changed")
    if not case.get("parse_cache_hit_on_second"):
        reasons.append("parse_cache_not_used_on_second")
    if not case.get("ocr_text_validation_pass"):
        reasons.append("ocr_text_duplicate_validation_failed")
    if case.get("status") not in {"verified", "review_needed", "raw", "auto_eligible"}:
        reasons.append("unexpected_status")
    return reasons


def _build_case_result(image_path: Path, image_runs: List[dict], ocr_text_runs: List[dict]) -> Dict[str, Any]:
    first = image_runs[0]
    second = image_runs[1]
    text_first = ocr_text_runs[0] if ocr_text_runs else {}
    text_second = ocr_text_runs[1] if len(ocr_text_runs) > 1 else {}
    image_pass = bool(second["top1_similarity_score"] == 1.0 and second["exact_match_detected"])
    text_pass = True if not ocr_text_runs else bool(text_second.get("top1_similarity_score") == 1.0)
    warnings = list(dict.fromkeys(second.get("warnings", []) + text_second.get("warnings", [])))

    case = {
        "image_path": str(image_path),
        "first_trace_id": first.get("trace_id", ""),
        "second_trace_id": second.get("trace_id", ""),
        "first_top1_product_id": first.get("top1_product_id", ""),
        "second_top1_product_id": second.get("top1_product_id", ""),
        "first_similarity_score": first.get("top1_similarity_score", 0.0),
        "second_similarity_score": second.get("top1_similarity_score", 0.0),
        "image_hash_equal": first.get("image_hash", "") == second.get("image_hash", ""),
        "ocr_text_hash_equal": first.get("ocr_text_hash", "") == second.get("ocr_text_hash", ""),
        "parsed_signature_equal": first.get("parsed_signature", "") == second.get("parsed_signature", ""),
        "profile_signature_equal": first.get("profile_signature", "") == second.get("profile_signature", ""),
        "parse_cache_hit_on_second": bool(second.get("parse_cache_hit")),
        "exact_match_detected": bool(second.get("exact_match_detected")),
        "status": second.get("status", ""),
        "quality_grade": second.get("quality_grade", ""),
        "candidate_scope": second.get("candidate_scope", ""),
        "candidate_disabled_reason": second.get("candidate_disabled_reason", ""),
        "warnings": warnings,
        "pass": bool(image_pass and text_pass),
        "ocr_text_validation_pass": text_pass,
        "ocr_text_first_trace_id": text_first.get("trace_id", ""),
        "ocr_text_second_trace_id": text_second.get("trace_id", ""),
        "ocr_text_second_exact_match_detected": bool(text_second.get("exact_match_detected", False)),
        "ocr_text_second_similarity_score": text_second.get("top1_similarity_score", 0.0),
        "details": {
            "image_runs": image_runs,
            "ocr_text_runs": ocr_text_runs,
        },
    }
    case["failure_reasons"] = _classify_failure(case)
    case["failure_summary"] = {
        "trace_ids": [item.get("trace_id", "") for item in image_runs + ocr_text_runs if item.get("trace_id")],
        "validator_corrected_fields": {
            "image_first": first.get("validator_result", {}).get("corrected_fields", []),
            "image_second": second.get("validator_result", {}).get("corrected_fields", []),
            "ocr_text_second": text_second.get("validator_result", {}).get("corrected_fields", []),
        },
        "validator_unknown_ratio": {
            "image_first": first.get("validator_result", {}).get("unknown_ratio"),
            "image_second": second.get("validator_result", {}).get("unknown_ratio"),
            "ocr_text_second": text_second.get("validator_result", {}).get("unknown_ratio"),
        },
        "validator_warnings": {
            "image_second": second.get("validator_result", {}).get("warnings", []),
            "ocr_text_second": text_second.get("validator_result", {}).get("warnings", []),
        },
    }
    return case


def _expand_inputs(images: Iterable[str], patterns: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for image in images:
        path = Path(image)
        if path.exists():
            resolved.append(path)
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            if path.exists():
                resolved.append(path)
    unique: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _write_csv(results: List[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "first_trace_id",
        "second_trace_id",
        "first_top1_product_id",
        "second_top1_product_id",
        "first_similarity_score",
        "second_similarity_score",
        "image_hash_equal",
        "ocr_text_hash_equal",
        "parsed_signature_equal",
        "profile_signature_equal",
        "parse_cache_hit_on_second",
        "exact_match_detected",
        "status",
        "quality_grade",
        "candidate_scope",
        "candidate_disabled_reason",
        "warnings",
        "pass",
        "ocr_text_validation_pass",
        "ocr_text_first_trace_id",
        "ocr_text_second_trace_id",
        "ocr_text_second_exact_match_detected",
        "ocr_text_second_similarity_score",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {key: item.get(key) for key in fieldnames}
            row["warnings"] = " | ".join(item.get("warnings", []))
            writer.writerow(row)


def _safe_print_json(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate duplicate upload similarity for multiple images.")
    parser.add_argument("images", nargs="*", help="Image paths to validate")
    parser.add_argument("--glob", action="append", default=[], dest="globs", help="Glob pattern for image files")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=200)
    args = parser.parse_args()

    image_paths = _expand_inputs(args.images, args.globs)
    if not image_paths:
        raise SystemExit("No input images found. Provide image paths or --glob.")

    service = RecommendationService()
    upload_service = UploadRecommendationService(service)

    results: List[dict] = []
    for image_path in image_paths:
        image_runs = _run_image_duplicate(upload_service, image_path, args.top_k, args.candidate_limit)
        try:
            ocr_result = extract_text_from_image(str(image_path))
            raw_text = str(ocr_result.get("raw_text", "") or "")
        except Exception as exc:  # noqa: BLE001
            raw_text = ""
            ocr_text_runs = [
                {
                    "attempt": 1,
                    "trace_id": "",
                    "top1_product_id": "",
                    "top1_similarity_score": 0.0,
                    "image_hash": "",
                    "ocr_text_hash": "",
                    "parsed_signature": "",
                    "profile_signature": "",
                    "parse_cache_hit": False,
                    "exact_match_detected": False,
                    "status": "ocr_text_unavailable",
                    "quality_grade": "D",
                    "warnings": [f"OCR raw text validation skipped: {exc}"],
                }
            ]
        else:
            ocr_text_runs = _run_ocr_text_duplicate(upload_service, raw_text, args.top_k, args.candidate_limit) if raw_text else []
        results.append(_build_case_result(image_path, image_runs, ocr_text_runs))

    summary = {
        "total": len(results),
        "passed": sum(1 for item in results if item.get("pass")),
        "failed": sum(1 for item in results if not item.get("pass")),
        "failed_cases": [item["image_path"] for item in results if not item.get("pass")],
        "failed_case_summaries": [
            {
                "image_path": item["image_path"],
                "failure_reasons": item.get("failure_reasons", []),
                "first_trace_id": item.get("first_trace_id", ""),
                "second_trace_id": item.get("second_trace_id", ""),
                "ocr_text_second_trace_id": item.get("ocr_text_second_trace_id", ""),
                "warnings": item.get("warnings", []),
                "failure_summary": item.get("failure_summary", {}),
            }
            for item in results
            if not item.get("pass")
        ],
    }
    report = {"summary": summary, "results": results}

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(results)
    _safe_print_json(report)


if __name__ == "__main__":
    main()
