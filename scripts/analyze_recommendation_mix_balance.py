from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_REPORT = REPO_ROOT / "output" / "recommendation_quality_eval_report.json"
OUTPUT_JSON = REPO_ROOT / "output" / "recommendation_mix_balance_report.json"
OUTPUT_CSV = REPO_ROOT / "output" / "recommendation_mix_balance_report.csv"


def _flag_case(item: dict) -> List[str]:
    flags: List[str] = []
    if bool(item.get("top5_uploaded_dominance")):
        flags.append("uploaded_dominance_top5")
    first_official_rank = item.get("first_official_rank")
    if isinstance(first_official_rank, int) and first_official_rank >= 5:
        flags.append("late_first_official")
    if bool(item.get("top1_exact_same_upload")):
        flags.append("exact_upload_top1")
    if not str(item.get("first_official_report_no", "") or ""):
        flags.append("missing_official")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze official/uploaded recommendation mix balance.")
    parser.add_argument("--eval-report", default=str(DEFAULT_EVAL_REPORT), help="Path to recommendation eval report JSON.")
    args = parser.parse_args()

    data = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    results = list(data.get("results", []) or [])

    analyzed: List[Dict[str, Any]] = []
    for item in results:
        flags = _flag_case(item)
        analyzed.append(
            {
                "case_id": str(item.get("case_id", "") or ""),
                "input_mode": str(item.get("input_mode", "") or ""),
                "evaluation_track": str(item.get("evaluation_track", "") or ""),
                "status": str(item.get("status", "") or ""),
                "quality_grade": str(item.get("quality_grade", "") or ""),
                "profile_confidence": float(item.get("profile_confidence", 0.0) or 0.0),
                "first_official_report_no": str(item.get("first_official_report_no", "") or ""),
                "first_official_rank": item.get("first_official_rank"),
                "top5_uploaded_count": int(item.get("top5_uploaded_count", 0) or 0),
                "top5_official_count": int(item.get("top5_official_count", 0) or 0),
                "top5_uploaded_dominance": bool(item.get("top5_uploaded_dominance")),
                "top1_exact_same_upload": bool(item.get("top1_exact_same_upload")),
                "flags": flags,
                "first_official_product_name": str(item.get("first_official_product_name", "") or ""),
            }
        )

    summary = {
        "total_cases": len(analyzed),
        "uploaded_dominance_top5_count": sum(1 for item in analyzed if item["top5_uploaded_dominance"]),
        "late_first_official_count": sum(1 for item in analyzed if isinstance(item["first_official_rank"], int) and item["first_official_rank"] >= 5),
        "exact_upload_top1_count": sum(1 for item in analyzed if item["top1_exact_same_upload"]),
        "missing_official_count": sum(1 for item in analyzed if not item["first_official_report_no"]),
    }

    OUTPUT_JSON.write_text(json.dumps({"summary": summary, "cases": analyzed}, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "case_id",
        "input_mode",
        "evaluation_track",
        "status",
        "quality_grade",
        "profile_confidence",
        "first_official_report_no",
        "first_official_rank",
        "top5_uploaded_count",
        "top5_official_count",
        "top5_uploaded_dominance",
        "top1_exact_same_upload",
        "first_official_product_name",
        "flags",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in analyzed:
            row = dict(item)
            row["flags"] = ",".join(item["flags"])
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(json.dumps({"summary": summary, "output_json": str(OUTPUT_JSON), "output_csv": str(OUTPUT_CSV)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
