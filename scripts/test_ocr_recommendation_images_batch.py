from __future__ import annotations

import json
import os
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
TARGET_IMAGE_NAMES = [
    "product_001.png",
    "product_002.jpg",
    "product_003.png",
    "product_004.jpg",
    "product_005.jpg",
]


def resolve_target_image_names() -> list[str]:
    raw_value = os.getenv("OCR_BATCH_TARGETS", "").strip()
    if not raw_value:
        return TARGET_IMAGE_NAMES
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_run_id() -> str:
    return os.getenv("OCR_BATCH_RUN_ID", "").strip()


def build_output_paths(output_dir: Path, run_id: str) -> dict[str, Path]:
    paths = {
        "summary": output_dir / "ocr_image_batch_test_summary.json",
        "results_jsonl": output_dir / "ocr_image_batch_test_results.jsonl",
        "results_csv": output_dir / "ocr_image_batch_test_results.csv",
        "warnings_csv": output_dir / "ocr_image_batch_test_warnings.csv",
        "failed_csv": output_dir / "failed_ocr_image_batch_test.csv",
    }
    if run_id:
        paths.update(
            {
                "summary_run": output_dir / f"ocr_image_batch_test_summary_{run_id}.json",
                "results_jsonl_run": output_dir / f"ocr_image_batch_test_results_{run_id}.jsonl",
                "results_csv_run": output_dir / f"ocr_image_batch_test_results_{run_id}.csv",
                "warnings_csv_run": output_dir / f"ocr_image_batch_test_warnings_{run_id}.csv",
                "failed_csv_run": output_dir / f"failed_ocr_image_batch_test_{run_id}.csv",
            }
        )
    return paths


def build_summary(rows: list[dict], warning_rows: list[dict], image_paths: list[Path], target_names: list[str], total_elapsed: float, output_paths: dict[str, Path]) -> dict:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dir": str(INPUT_DIR),
        "target_image_names": target_names,
        "image_count": len(image_paths),
        "success_count": sum(1 for row in rows if row["status_code"] == 200 and not row["ocr_error"]),
        "failure_count": sum(1 for row in rows if row["status_code"] != 200 or row["ocr_error"]),
        "needs_user_review_count": sum(1 for row in rows if row["needs_user_review"]),
        "warning_count": len(warning_rows),
        "total_elapsed_seconds": total_elapsed,
        "results_csv": str(output_paths["results_csv"]),
        "results_jsonl": str(output_paths["results_jsonl"]),
        "warnings_csv": str(output_paths["warnings_csv"]),
        "failed_csv": str(output_paths["failed_csv"]),
        "samples": rows[:5],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = resolve_run_id()
    output_paths = build_output_paths(output_dir, run_id)
    summary_path = output_paths["summary"]
    results_jsonl_path = output_paths["results_jsonl"]
    results_csv_path = output_paths["results_csv"]
    warnings_csv_path = output_paths["warnings_csv"]
    failed_csv_path = output_paths["failed_csv"]

    client = TestClient(app)
    target_names = resolve_target_image_names()
    available_paths = {
        path.name: path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    }
    image_paths = [available_paths[name] for name in target_names if name in available_paths]

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
        critical_warnings = body.get("critical_warnings", []) or body.get("quality_warnings", []) or []
        notices = body.get("notices", []) or []
        info_warnings = body.get("info_warnings", []) or []
        all_warnings = critical_warnings + notices + info_warnings
        warning_codes = [warning.get("code", "") for warning in all_warnings if warning.get("code")]
        row = {
            "file_name": image_path.name,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "ocr_error": ocr.get("error"),
            "ocr_confidence": ocr.get("confidence"),
            "ocr_confidence_source": ocr.get("confidence_source", "unavailable"),
            "parse_confidence": parsed.get("confidence"),
            "product_name_candidate": parsed.get("product_name_candidate", ""),
            "product_name_category_hint": body.get("product_name_category_hint", ""),
            "estimated_main_category": estimated.get("product_main_category", ""),
            "normalized_ingredients_count": len(parsed.get("normalized_ingredients", [])),
            "category_diversity_count": body.get("category_diversity_count", 0),
            "primary_ingredients": ", ".join(parsed.get("primary_ingredients", [])),
            "primary_ingredients_normalized": ", ".join(parsed.get("primary_ingredients_normalized", [])),
            "excluded_ingredients": ", ".join(parsed.get("excluded_ingredients", []) or body.get("excluded_ingredients", [])),
            "warning_codes": "|".join(warning_codes),
            "critical_warning_codes": "|".join([warning.get("code", "") for warning in critical_warnings if warning.get("code")]),
            "notice_warning_codes": "|".join([warning.get("code", "") for warning in notices if warning.get("code")]),
            "info_warning_codes": "|".join([warning.get("code", "") for warning in info_warnings if warning.get("code")]),
            "recommendation_count": len(recommendations),
            "top_target_product_name": recommendations[0].get("target_product_name", "") if recommendations else "",
            "top_similarity_score": recommendations[0].get("similarity_score", 0.0) if recommendations else 0.0,
            "top_reason": recommendations[0].get("reason", "") if recommendations else "",
            "needs_user_review": body.get("needs_user_review"),
            "review_message": body.get("review_message", ""),
            "warning_count": len(all_warnings),
            "critical_warning_count": len(critical_warnings),
            "notice_count": len(notices),
            "info_warning_count": len(info_warnings),
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
        for warning in all_warnings:
            warning_rows.append(
                {
                    "file_name": image_path.name,
                    "code": warning.get("code", ""),
                    "message": warning.get("message", ""),
                    "severity": warning.get("severity", "critical"),
                    "product_name_category": warning.get("product_name_category", ""),
                    "estimated_main_category": warning.get("estimated_main_category", ""),
                    "ingredients": json.dumps(
                        warning.get("ingredients", [])
                        or parsed.get("primary_ingredients_normalized", [])
                        or parsed.get("normalized_ingredients", []),
                        ensure_ascii=False,
                    ),
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

    summary = build_summary(rows, warning_rows, image_paths, target_names, total_elapsed, output_paths)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if run_id:
        pd.DataFrame(rows).to_csv(output_paths["results_csv_run"], index=False, encoding="utf-8-sig")
        pd.DataFrame(warning_rows).to_csv(output_paths["warnings_csv_run"], index=False, encoding="utf-8-sig")
        pd.DataFrame(failed_rows).to_csv(output_paths["failed_csv_run"], index=False, encoding="utf-8-sig")
        with output_paths["results_jsonl_run"].open("w", encoding="utf-8") as file:
            for row in jsonl_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        run_summary = dict(summary)
        run_summary["results_csv"] = str(output_paths["results_csv_run"])
        run_summary["results_jsonl"] = str(output_paths["results_jsonl_run"])
        run_summary["warnings_csv"] = str(output_paths["warnings_csv_run"])
        run_summary["failed_csv"] = str(output_paths["failed_csv_run"])
        run_summary["run_id"] = run_id
        output_paths["summary_run"].write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
