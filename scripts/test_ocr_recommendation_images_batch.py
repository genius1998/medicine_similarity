from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app  # noqa: E402


INPUT_DIR = Path(r"D:\health_project\input_images")


def main() -> None:
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "ocr_image_batch_test_summary.json"
    results_jsonl_path = output_dir / "ocr_image_batch_test_results.jsonl"
    results_csv_path = output_dir / "ocr_image_batch_test_results.csv"
    warnings_csv_path = output_dir / "ocr_image_batch_test_warnings.csv"
    failed_csv_path = output_dir / "failed_ocr_image_batch_test.csv"

    client = TestClient(app)
    image_paths = sorted([path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])

    rows = []
    jsonl_rows = []
    warning_rows = []
    failed_rows = []
    started_all = time.perf_counter()

    for image_path in image_paths:
        started = time.perf_counter()
        with image_path.open("rb") as file:
            response = client.post(
                "/api/recommend/by-image",
                files={"file": (image_path.name, file, f"image/{image_path.suffix.lower().lstrip('.')}")},
                data={"top_k": "10", "candidate_limit": "1000"},
            )
        elapsed = round(time.perf_counter() - started, 6)
        body = response.json()
        recommendations = body.get("recommendations", [])
        parsed = body.get("parsed", {})
        ocr = body.get("ocr", {})
        estimated = body.get("estimated_profile", {})
        quality_warnings = body.get("quality_warnings", []) or []
        row = {
            "file_name": image_path.name,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "ocr_error": ocr.get("error"),
            "ocr_confidence": ocr.get("confidence"),
            "product_name_candidate": parsed.get("product_name_candidate", ""),
            "normalized_ingredients_count": len(parsed.get("normalized_ingredients", [])),
            "estimated_main_category": estimated.get("product_main_category", ""),
            "product_name_category_hint": body.get("product_name_category_hint", ""),
            "primary_ingredients": ", ".join(estimated.get("primary_ingredients", [])),
            "excluded_ingredients": ", ".join(body.get("excluded_ingredients", [])),
            "recommendation_count": len(recommendations),
            "top_target_product_name": recommendations[0].get("target_product_name", "") if recommendations else "",
            "top_similarity_score": recommendations[0].get("similarity_score", 0.0) if recommendations else 0.0,
            "top_reason": recommendations[0].get("reason", "") if recommendations else "",
            "needs_user_review": body.get("needs_user_review"),
            "review_message": body.get("review_message", ""),
            "warning_count": len(quality_warnings),
            "has_caution": bool(recommendations[0].get("caution", "")) if recommendations else False,
        }
        rows.append(row)
        jsonl_rows.append(
            {
                "file_name": image_path.name,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "response": body,
            }
        )
        for warning in quality_warnings:
            warning_rows.append(
                {
                    "file_name": image_path.name,
                    "code": warning.get("code", ""),
                    "message": warning.get("message", ""),
                    "product_name_category": warning.get("product_name_category", ""),
                    "estimated_main_category": warning.get("estimated_main_category", ""),
                    "ingredients": json.dumps(warning.get("ingredients", []), ensure_ascii=False),
                }
            )
        if response.status_code != 200 or ocr.get("error"):
            failed_rows.append(
                {
                    "file_name": image_path.name,
                    "status_code": response.status_code,
                    "ocr_error": ocr.get("error"),
                    "review_message": body.get("review_message", ""),
                }
            )

    total_elapsed = round(time.perf_counter() - started_all, 6)
    pd.DataFrame(rows).to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(warning_rows).to_csv(warnings_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(failed_csv_path, index=False, encoding="utf-8-sig")
    with results_jsonl_path.open("w", encoding="utf-8") as file:
        for row in jsonl_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dir": str(INPUT_DIR),
        "image_count": len(image_paths),
        "success_count": sum(1 for row in rows if row["status_code"] == 200 and not row["ocr_error"]),
        "failure_count": sum(1 for row in rows if row["status_code"] != 200 or row["ocr_error"]),
        "needs_user_review_count": sum(1 for row in rows if row["needs_user_review"]),
        "warning_count": len(warning_rows),
        "total_elapsed_seconds": total_elapsed,
        "results_csv": str(results_csv_path),
        "results_jsonl": str(results_jsonl_path),
        "warnings_csv": str(warnings_csv_path),
        "failed_csv": str(failed_csv_path),
        "samples": rows[:5],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
