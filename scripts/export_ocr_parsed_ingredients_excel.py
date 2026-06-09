from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.db import default_sqlite_path  # noqa: E402
from api.ingredient_parse_service import parse_ingredients_from_ocr_text  # noqa: E402
from api.ocr_service import extract_text_from_image  # noqa: E402


DEFAULT_INPUT_DIR = Path(r"D:\health_project\input_images")
DEFAULT_OUTPUT_PATH = ROOT_DIR / "output" / "ocr_parsed_ingredients_export.xlsx"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def build_summary_row(image_path: Path, elapsed: float, ocr_payload: dict, parsed: dict) -> dict:
    quality_warnings = list(parsed.get("quality_warnings", []) or [])
    warning_codes = [str(item.get("code", "") or "") for item in quality_warnings if str(item.get("code", "") or "")]
    return {
        "file_name": image_path.name,
        "image_path": str(image_path),
        "elapsed_seconds": round(elapsed, 6),
        "ocr_error": str(ocr_payload.get("error", "") or ""),
        "ocr_confidence": ocr_payload.get("confidence"),
        "ocr_confidence_source": str(ocr_payload.get("confidence_source", "") or ""),
        "ocr_text_length": len(str(ocr_payload.get("raw_text", "") or "")),
        "product_name_candidate": str(parsed.get("product_name_candidate", "") or ""),
        "parse_confidence": parsed.get("confidence"),
        "needs_user_review": bool(parsed.get("needs_user_review", False)),
        "ingredient_section_text": str(parsed.get("ingredient_section_text", "") or ""),
        "raw_ingredients_joined": " | ".join(str(item).strip() for item in parsed.get("raw_ingredients", []) if str(item).strip()),
        "normalized_ingredients_joined": " | ".join(
            str(item).strip() for item in parsed.get("normalized_ingredients", []) if str(item).strip()
        ),
        "primary_ingredients_joined": " | ".join(
            str(item).strip() for item in parsed.get("primary_ingredients", []) if str(item).strip()
        ),
        "primary_ingredients_normalized_joined": " | ".join(
            str(item).strip() for item in parsed.get("primary_ingredients_normalized", []) if str(item).strip()
        ),
        "excluded_ingredients_joined": " | ".join(
            str(item).strip() for item in parsed.get("excluded_ingredients", []) if str(item).strip()
        ),
        "warning_codes": " | ".join(warning_codes),
        "warning_count": len(quality_warnings),
        "ocr_raw_text": str(ocr_payload.get("raw_text", "") or ""),
    }


def build_ingredient_rows(file_name: str, parsed: dict) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(parsed.get("ingredient_objects", []) or [], start=1):
        rows.append(
            {
                "file_name": file_name,
                "ingredient_index": idx,
                "raw": str(item.get("raw", "") or ""),
                "display_name": str(item.get("display_name", "") or ""),
                "normalized_for_matching": str(item.get("normalized_for_matching", "") or ""),
                "role": str(item.get("role", "") or ""),
                "category_hint": str(item.get("category_hint", "") or ""),
            }
        )
    return rows


def build_warning_rows(file_name: str, parsed: dict) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(parsed.get("quality_warnings", []) or [], start=1):
        rows.append(
            {
                "file_name": file_name,
                "warning_index": idx,
                "code": str(item.get("code", "") or ""),
                "severity": str(item.get("severity", "") or ""),
                "message": str(item.get("message", "") or ""),
                "payload_json": json.dumps(item, ensure_ascii=False),
            }
        )
    return rows


def run_export(input_dir: Path, output_path: Path) -> dict:
    image_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    sqlite_path = default_sqlite_path()

    summary_rows: list[dict] = []
    ingredient_rows: list[dict] = []
    warning_rows: list[dict] = []

    for image_path in image_paths:
        started = time.perf_counter()
        ocr_payload = extract_text_from_image(str(image_path))
        raw_text = str(ocr_payload.get("raw_text", "") or "")
        if ocr_payload.get("error"):
            parsed = {
                "product_name_candidate": "",
                "ingredient_section_text": "",
                "raw_ingredients": [],
                "normalized_ingredients": [],
                "ingredient_objects": [],
                "primary_ingredients": [],
                "primary_ingredients_normalized": [],
                "excluded_ingredients": [],
                "quality_warnings": [
                    {
                        "code": "ocr_error",
                        "severity": "critical",
                        "message": str(ocr_payload.get("error", "") or ""),
                    }
                ],
                "confidence": 0.0,
                "needs_user_review": True,
            }
        else:
            parsed = parse_ingredients_from_ocr_text(raw_text, sqlite_path)

        elapsed = time.perf_counter() - started
        summary_rows.append(build_summary_row(image_path, elapsed, ocr_payload, parsed))
        ingredient_rows.extend(build_ingredient_rows(image_path.name, parsed))
        warning_rows.extend(build_warning_rows(image_path.name, parsed))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(ingredient_rows).to_excel(writer, sheet_name="parsed_ingredients", index=False)
        pd.DataFrame(warning_rows).to_excel(writer, sheet_name="warnings", index=False)

    return {
        "image_count": len(image_paths),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OCR-parsed ingredient names for image batch verification.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing product images")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Excel output path")
    args = parser.parse_args()

    result = run_export(Path(args.input_dir), Path(args.output_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
