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
OUTPUT_PATH = REPO_ROOT / "output" / "candidate_policy_conflicts.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find high-confidence but disabled uploaded candidates.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Path to regression_duplicate_similarity_report.json")
    parser.add_argument("--threshold", type=float, default=0.75, help="Profile confidence threshold")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    conflicts: List[Dict[str, Any]] = []
    for item in report.get("results", []):
        image_runs = (item.get("details") or {}).get("image_runs", [])
        second = image_runs[1] if len(image_runs) > 1 else {}
        quality = second.get("quality") or {}
        validator = second.get("validator_result") or {}
        profile_confidence = float((quality.get("profile_confidence") or validator.get("profile_confidence") or 0.0))
        candidate_enabled = bool(quality.get("candidate_enabled"))
        if candidate_enabled or profile_confidence < args.threshold:
            continue
        conflicts.append(
            {
                "image_path": item.get("image_path", ""),
                "status": item.get("status", ""),
                "quality_grade": item.get("quality_grade", ""),
                "profile_confidence": profile_confidence,
                "candidate_enabled": candidate_enabled,
                "candidate_scope": quality.get("candidate_scope", ""),
                "candidate_disabled_reason": quality.get("candidate_disabled_reason", ""),
                "warnings": item.get("warnings", []),
                "validator_warnings": validator.get("warnings", []),
                "trace_id": second.get("trace_id", ""),
            }
        )

    payload = {
        "summary": {
            "threshold": args.threshold,
            "total_cases": len(report.get("results", [])),
            "conflict_count": len(conflicts),
        },
        "conflicts": conflicts,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
