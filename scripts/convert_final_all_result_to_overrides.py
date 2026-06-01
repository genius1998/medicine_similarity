from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


MAIN_CATEGORY_MAP = {
    "영양보충": "영양보충",
    "장건강": "장 건강",
    "장 건강": "장 건강",
    "면역": "면역",
    "체지방": "체지방",
    "혈행": "혈행",
    "관절/연골": "관절/연골",
    "뼈건강": "뼈 건강",
    "뼈 건강": "뼈 건강",
    "항산화": "항산화",
    "눈건강": "눈 건강",
    "눈 건강": "눈 건강",
    "간건강": "간 건강",
    "간 건강": "간 건강",
    "혈당": "혈당",
    "피부건강": "피부 건강",
    "피부 건강": "피부 건강",
    "기억력": "기억력",
    "인지력": "인지력",
    "전립선": "남성 건강",
    "남성건강": "남성 건강",
    "남성 건강": "남성 건강",
    "갱년기": "여성 건강",
    "여성건강": "여성 건강",
    "여성 건강": "여성 건강",
    "콜레스테롤": "혈중지질",
    "혈중지질": "혈중지질",
    "기타": "기타",
    "근육": "운동/근력",
    "운동/근력": "운동/근력",
    "피로개선": "피로개선",
    "구강건강": "구강 건강",
    "구강 건강": "구강 건강",
    "위건강": "장 건강",
    "위 건강": "장 건강",
    "수면": "수면/긴장완화",
    "긴장완화": "수면/긴장완화",
    "수면/긴장완화": "수면/긴장완화",
    "혈압": "혈압",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert batch API product category results into product profile override CSV."
    )
    parser.add_argument("--input-csv", default=r"D:\data\final_all_result.csv")
    parser.add_argument("--output-csv", default="output/final_all_result_category_overrides.csv")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def normalize_label(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def map_category(value: object) -> str:
    key = normalize_label(value)
    return MAIN_CATEGORY_MAP.get(key, "")


def parse_sub_categories(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        raw_values = parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        raw_values = re.split(r"[,|;]", text)

    result: list[str] = []
    for raw in raw_values:
        mapped = map_category(raw)
        if mapped and mapped not in result:
            result.append(mapped)
    return result


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False).fillna("")
    required = {"product_id", "product_name", "main_category", "confidence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if "sub_categories" not in df.columns:
        df["sub_categories"] = ""
    if "reason" not in df.columns:
        df["reason"] = ""
    if "request_key" not in df.columns:
        df["request_key"] = ""

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    rows = []
    skipped = []
    for row in df.to_dict(orient="records"):
        report_no = str(row.get("product_id", "")).strip()
        main_category = map_category(row.get("main_category", ""))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if not report_no or not main_category or confidence < args.min_confidence:
            skipped.append(row)
            continue
        sub_function_categories = [
            item for item in parse_sub_categories(row.get("sub_categories", "")) if item != main_category
        ]
        rows.append(
            {
                "report_no": report_no,
                "product_name": str(row.get("product_name", "")).strip(),
                "predicted_main_category": main_category,
                "predicted_sub_function_categories": json.dumps(sub_function_categories, ensure_ascii=False),
                "confidence": round(confidence, 4),
                "decision": "auto_override_candidate",
                "source": "batch_api_final_all_result",
                "reason": str(row.get("reason", "")).strip(),
                "original_main_category": str(row.get("main_category", "")).strip(),
                "original_sub_categories": str(row.get("sub_categories", "")).strip(),
                "request_key": str(row.get("request_key", "")).strip(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "input_rows": int(len(df)),
        "output_rows": int(len(result_df)),
        "skipped_rows": int(len(skipped)),
        "main_category_counts": dict(Counter(result_df["predicted_main_category"].astype(str))),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
