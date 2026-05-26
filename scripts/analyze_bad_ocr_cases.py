from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.ingredient_parse_service import parse_ingredients_from_ocr_text
from api.db import default_sqlite_path
from api.ocr_service import extract_text_from_image
from api.ocr_text_sectionizer import sectionize_ocr_text
from api.recommendation_service import RecommendationService


DEFAULT_REPORT_PATH = REPO_ROOT / "output" / "regression_low_quality_cases.json"
OUTPUT_JSON = REPO_ROOT / "output" / "bad_ocr_diagnostics_report.json"
OUTPUT_CSV = REPO_ROOT / "output" / "unknown_alias_priority_report.csv"


def _extract_unknown_candidates(parsed: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for item in parsed.get("ingredient_objects", []) or []:
        raw_name = str(item.get("raw_ingredient") or item.get("raw_name") or item.get("display_name") or "").strip()
        normalized = str(item.get("normalized_for_matching") or item.get("normalized_name") or item.get("standard_name") or "").strip()
        role = str(item.get("role") or "").strip().lower()
        if raw_name and not normalized and role not in {"excipient", "support"}:
            candidates.append(raw_name)
    return candidates


def _build_case(service: RecommendationService, item: Dict[str, Any]) -> Dict[str, Any]:
    image_path = Path(str(item.get("image_path", "")))
    ocr_payload = extract_text_from_image(str(image_path))
    raw_text = str(ocr_payload.get("raw_text", "") or "")
    sectionized = sectionize_ocr_text(raw_text)
    parsed = parse_ingredients_from_ocr_text(raw_text, default_sqlite_path())
    validator = parsed.get("validator_result") or {}
    unknown_candidates = _extract_unknown_candidates(parsed)
    return {
        "image_path": str(image_path),
        "status": item.get("status", ""),
        "quality_grade": item.get("quality_grade", ""),
        "candidate_disabled_reason": item.get("candidate_disabled_reason", ""),
        "warnings": item.get("warnings", []),
        "ocr_raw_text_length": len(raw_text),
        "ocr_raw_line_count": int(sectionized.get("raw_line_count", 0) or 0),
        "detected_section_names": sectionized.get("detected_section_names", []),
        "section_line_counts": sectionized.get("section_line_counts", {}),
        "sections": sectionized.get("sections", {}),
        "parse_confidence": parsed.get("confidence"),
        "profile_confidence": parsed.get("profile_confidence"),
        "normalized_count": len(parsed.get("normalized_ingredients", []) or []),
        "primary_count": len(parsed.get("primary_ingredients_normalized", []) or parsed.get("primary_ingredients", []) or []),
        "unknown_ratio": validator.get("unknown_ratio"),
        "validator_warnings": validator.get("warnings", []),
        "unknown_candidates": unknown_candidates,
    }


def _write_alias_priority(counter: Counter) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["raw_name", "count"])
        writer.writeheader()
        for raw_name, count in counter.most_common():
            writer.writerow({"raw_name": raw_name, "count": count})


def _safe_print_json(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bad OCR low-quality cases in detail.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Path to regression_low_quality_cases.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cases = list((payload.get("summary") or {}).get("low_quality_cases", []))
    bad_ocr_cases = [case for case in cases if str(case.get("root_cause", "")) == "bad_ocr"]

    service = RecommendationService()
    diagnostics: List[Dict[str, Any]] = []
    alias_counter: Counter = Counter()
    for case in bad_ocr_cases:
        diagnostic = _build_case(service, case)
        diagnostics.append(diagnostic)
        alias_counter.update(diagnostic.get("unknown_candidates", []))

    report = {
        "summary": {
            "total_bad_ocr_cases": len(diagnostics),
            "top_unknown_alias_candidates": [{"raw_name": raw_name, "count": count} for raw_name, count in alias_counter.most_common(20)],
        },
        "cases": diagnostics,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_alias_priority(alias_counter)
    _safe_print_json(report)


if __name__ == "__main__":
    main()
