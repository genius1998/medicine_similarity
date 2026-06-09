from __future__ import annotations

import csv
import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "ingredient_match_v2"


LLM_FIELDS = [
    "ingredient_id",
    "raw_ingredient",
    "normalized_raw",
    "representative_raw",
    "mention_count",
    "product_count",
    "decision",
    "matched_standard_name",
    "relation_type",
    "confidence",
    "category_main",
    "category_sub",
    "reason",
    "needs_human_review",
    "match_method",
    "llm_response_json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LLM_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_llm_row(row: dict[str, str]) -> dict[str, Any]:
    result = {field: row.get(field, "") for field in LLM_FIELDS}
    result["representative_raw"] = row.get("representative_ingredient", "")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rows checked by Gemini from ingredient_match_cache_v2.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT_DIR / output_dir).resolve()
    source_path = output_dir / "ingredient_match_cache_v2.csv"
    rows = read_csv(source_path)
    llm_rows = [to_llm_row(row) for row in rows if row.get("match_method") == "gemini_batch_v2"]
    existing_rows = [row for row in llm_rows if row.get("decision") == "existing_match"]
    new_rows = [row for row in llm_rows if row.get("decision") == "new_standard_candidate"]
    unmatched_rows = [row for row in llm_rows if row.get("decision") != "existing_match"]

    checked_path = output_dir / "llm_checked_ingredients.csv"
    existing_path = output_dir / "llm_existing_match_ingredients.csv"
    unmatched_path = output_dir / "llm_unmatched_ingredients.csv"
    new_path = output_dir / "llm_new_standard_candidates.csv"

    write_csv(checked_path, llm_rows)
    write_csv(existing_path, existing_rows)
    write_csv(unmatched_path, unmatched_rows)
    write_csv(new_path, new_rows)

    summary = {
        "source_cache": str(source_path),
        "llm_checked_ingredients_csv": str(checked_path),
        "llm_existing_match_ingredients_csv": str(existing_path),
        "llm_unmatched_ingredients_csv": str(unmatched_path),
        "llm_new_standard_candidates_csv": str(new_path),
        "counts": {
            "llm_checked": len(llm_rows),
            "llm_existing_match": len(existing_rows),
            "llm_unmatched_or_review": len(unmatched_rows),
            "llm_new_standard_candidates": len(new_rows),
        },
        "llm_decision_counts": dict(Counter(row.get("decision", "") for row in llm_rows)),
        "note": "new_standard_candidate rows are provisional new-standard candidates, not existing functional_category_map matches.",
    }
    (output_dir / "llm_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
