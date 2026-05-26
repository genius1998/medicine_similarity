from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPORT_PATH = REPO_ROOT / "output" / "regression_duplicate_similarity_report.json"
OUTPUT_PATH = REPO_ROOT / "output" / "regression_low_quality_cases.json"


def classify_root_cause(item: Dict[str, Any]) -> str:
    warnings = [str(value or "") for value in item.get("warnings", [])]
    image_conf = float(item.get("image_profile_confidence") or 0.0)
    ocr_conf = float(item.get("ocr_text_profile_confidence") or 0.0)
    corrected_fields = list(item.get("image_validator_corrected_fields", [])) + list(item.get("ocr_text_validator_corrected_fields", []))
    candidate_enabled = bool(item.get("candidate_enabled"))
    candidate_disabled_reason = str(item.get("candidate_disabled_reason", "") or "")
    status = str(item.get("status", "") or "")

    if (
        not candidate_enabled
        and image_conf >= 0.75
        and candidate_disabled_reason in {"normalized_ingredients_too_few", "quality_grade_c", "missing_or_generic_main_category"}
    ):
        return "policy_conflict"
    if (
        not candidate_enabled
        and image_conf >= 0.65
        and candidate_disabled_reason in {"quality_grade_c", "profile_confidence_below_threshold"}
    ):
        return "over_conservative_gating"
    if any("deduped" in field for field in corrected_fields):
        return "section_mixing"
    if any("부형제" in warning or "첨가물" in warning for warning in warnings):
        return "section_mixing"
    if any("원재료명" in warning or "OCR" in warning for warning in warnings):
        return "bad_ocr"
    if not candidate_enabled and max(image_conf, ocr_conf) >= 0.75:
        return "policy_conflict"
    if not candidate_enabled and max(image_conf, ocr_conf) >= 0.6:
        return "over_conservative_gating"
    if status == "review_needed" and max(image_conf, ocr_conf) < 0.4:
        return "true_low_quality"
    if max(image_conf, ocr_conf) < 0.25:
        return "bad_ocr"
    return "alias_mapping_gap"


def is_low_quality(item: Dict[str, Any]) -> bool:
    status = str(item.get("status", "") or "")
    grade = str(item.get("quality_grade", "") or "")
    warnings = item.get("warnings", []) or []
    image_runs = ((item.get("details") or {}).get("image_runs") or [])
    latest_image = image_runs[-1] if image_runs else {}
    candidate_enabled = bool(((latest_image.get("quality") or {}).get("candidate_enabled")))
    if status in {"raw", "review_needed"}:
        return True
    if grade in {"C", "D"}:
        return True
    if warnings and (not candidate_enabled or grade not in {"A", "B"}):
        return True
    return False


def build_case_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    details = item.get("details", {}) or {}
    image_runs = details.get("image_runs") or []
    ocr_text_runs = details.get("ocr_text_runs") or []
    image_second = image_runs[1] if len(image_runs) > 1 else {}
    ocr_text_second = ocr_text_runs[1] if len(ocr_text_runs) > 1 else {}
    return {
        "image_path": item.get("image_path", ""),
        "status": item.get("status", ""),
        "quality_grade": item.get("quality_grade", ""),
        "warnings": item.get("warnings", []),
        "pass": bool(item.get("pass")),
        "first_trace_id": item.get("first_trace_id", ""),
        "second_trace_id": item.get("second_trace_id", ""),
        "ocr_text_second_trace_id": item.get("ocr_text_second_trace_id", ""),
        "image_profile_confidence": ((image_second.get("quality") or {}).get("profile_confidence")),
        "ocr_text_profile_confidence": ((ocr_text_second.get("quality") or {}).get("profile_confidence")),
        "image_validator_corrected_fields": ((image_second.get("validator_result") or {}).get("corrected_fields", [])),
        "ocr_text_validator_corrected_fields": ((ocr_text_second.get("validator_result") or {}).get("corrected_fields", [])),
        "image_validator_warnings": ((image_second.get("validator_result") or {}).get("warnings", [])),
        "ocr_text_validator_warnings": ((ocr_text_second.get("validator_result") or {}).get("warnings", [])),
        "candidate_enabled": ((image_second.get("quality") or {}).get("candidate_enabled")),
        "candidate_scope": ((image_second.get("quality") or {}).get("candidate_scope")),
        "candidate_disabled_reason": ((image_second.get("quality") or {}).get("candidate_disabled_reason")),
        "failure_reasons": item.get("failure_reasons", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract low-quality cases from duplicate regression report.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Path to regression_duplicate_similarity_report.json")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = list(report.get("results", []))
    low_quality_cases = [build_case_summary(item) for item in results if is_low_quality(item)]

    summary = {
        "total_cases": len(results),
        "low_quality_count": len(low_quality_cases),
        "status_counts": {},
        "quality_grade_counts": {},
        "low_quality_cases": low_quality_cases,
        "root_cause_counts": {},
    }

    for item in low_quality_cases:
        status = str(item.get("status", "") or "")
        grade = str(item.get("quality_grade", "") or "")
        root_cause = classify_root_cause(item)
        item["root_cause"] = root_cause
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        summary["quality_grade_counts"][grade] = summary["quality_grade_counts"].get(grade, 0) + 1
        summary["root_cause_counts"][root_cause] = summary["root_cause_counts"].get(root_cause, 0) + 1

    payload = {"summary": summary}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
