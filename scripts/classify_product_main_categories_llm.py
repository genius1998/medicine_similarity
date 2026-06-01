from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.local_llm_client import call_local_llm


ALLOWED_CATEGORIES = [
    "영양보충",
    "장 건강",
    "뼈 건강",
    "면역",
    "기타",
    "관절/연골",
    "혈중지질",
    "눈 건강",
    "혈당",
    "남성 건강",
    "체지방",
    "구강 건강",
    "피부 건강",
    "기억력",
    "여성 건강",
    "피로개선",
    "운동/근력",
    "수면/긴장완화",
    "간 건강",
    "항산화",
    "혈행",
    "인지력",
    "혈압",
]


OUTPUT_FIELDS = [
    "batch_index",
    "sample_index",
    "report_no",
    "product_name",
    "current_product_main_category",
    "current_product_sub_categories",
    "predicted_main_category",
    "predicted_sub_categories",
    "changed",
    "sub_categories_changed",
    "confidence",
    "decision",
    "evidence_main_functionality",
    "evidence_ingredients",
    "ignored_noise",
    "reason",
    "input_main_functionality",
    "input_raw_ingredients",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify product main categories with the local LLM in fixed-size batches."
    )
    parser.add_argument("--profile-csv", default="output/product_function_profile.csv")
    parser.add_argument("--catalog-csv", default="output/cache_db_excel_export/temp_csv/catalog_product_master.csv")
    parser.add_argument("--category-map-csv", default="output/functional_category_map.csv")
    parser.add_argument("--output-csv", default="output/llm_main_category_classification_50.csv")
    parser.add_argument("--raw-output-jsonl", default="")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of rows to classify. Use 0 for all rows.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--checkpoint-every-batches", type=int, default=25)
    parser.add_argument("--backup-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument(
        "--mode",
        choices=["risk_sample", "all"],
        default="risk_sample",
        help="risk_sample prioritizes low-confidence or noted rows, all processes joined rows in file order.",
    )
    parser.add_argument(
        "--include-report-no",
        action="append",
        default=[],
        help="Force include a report_no. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap after row selection.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def truncate_text(value: Any, limit: int) -> str:
    text = normalize_text(value)
    return text if len(text) <= limit else text[: limit - 20].rstrip() + " ...[truncated]"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
            parsed = loaded if isinstance(loaded, list) else [loaded]
        except Exception:
            parsed = re.split(r"[|,]", text)
    result: list[str] = []
    for item in parsed:
        name = normalize_text(item)
        if name and name not in result:
            result.append(name)
    return result


def load_allowed_sub_categories(category_map_csv: Path) -> list[str]:
    df = pd.read_csv(category_map_csv, encoding="utf-8-sig", low_memory=False).fillna("")
    if "category_sub" not in df.columns:
        return []
    counter: Counter[str] = Counter()
    for value in df["category_sub"].astype(str):
        for item in value.split("|"):
            name = item.strip()
            if name:
                counter[name] += 1
    return [name for name, _count in counter.most_common()]


def normalize_sub_category_label(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def canonicalize_sub_categories(values: list[str], allowed_sub_categories: list[str]) -> list[str]:
    allowed_by_normalized = {
        normalize_sub_category_label(name): name
        for name in allowed_sub_categories
    }
    result: list[str] = []
    for value in values:
        candidates = [value]
        if value and not value.endswith("계열"):
            candidates.append(f"{value} 계열")
        matched = ""
        for candidate in candidates:
            matched = allowed_by_normalized.get(normalize_sub_category_label(candidate), "")
            if matched:
                break
        if matched and matched not in result:
            result.append(matched)
        elif value and value not in result:
            result.append(value)
    return result


def extract_json_payload(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced[:]
    if not candidates:
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            candidates.append(text[first : last + 1])
    if not candidates:
        first = text.find("[")
        last = text.rfind("]")
        if first >= 0 and last > first:
            candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return {"items": value}
        except Exception:
            continue
    raise ValueError(f"LLM response did not contain a JSON object or array: {text[:500]}")


def load_joined_rows(profile_csv: Path, catalog_csv: Path) -> list[dict[str, Any]]:
    profile_df = pd.read_csv(profile_csv, encoding="utf-8-sig", low_memory=False).fillna("")
    catalog_df = pd.read_csv(catalog_csv, encoding="utf-8-sig", low_memory=False).fillna("")
    profile_df["report_no"] = profile_df["report_no"].astype(str).str.strip()
    catalog_df["report_no"] = catalog_df["report_no"].astype(str).str.strip()
    joined = profile_df.merge(
        catalog_df[["report_no", "main_functionality", "raw_ingredients"]],
        on="report_no",
        how="inner",
    )
    joined = joined[joined["main_functionality"].astype(str).str.strip() != ""].copy()
    rows: list[dict[str, Any]] = []
    for row in joined.to_dict(orient="records"):
        rows.append(
            {
                "report_no": str(row.get("report_no", "")).strip(),
                "product_name": str(row.get("product_name", "")).strip(),
                "current_product_main_category": str(row.get("product_main_category", "")).strip(),
                "current_product_sub_categories": safe_json_list(row.get("product_sub_categories_json", "[]")),
                "confidence": safe_float(row.get("confidence")),
                "notes": str(row.get("notes", "")).strip(),
                "main_functionality": str(row.get("main_functionality", "")).strip(),
                "raw_ingredients": str(row.get("raw_ingredients", "")).strip(),
            }
        )
    return rows


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_report_no = {row["report_no"]: row for row in rows if row.get("report_no")}
    forced = [by_report_no[item] for item in args.include_report_no if item in by_report_no]
    forced_keys = {row["report_no"] for row in forced}
    candidates = [row for row in rows if row.get("report_no") not in forced_keys]

    if args.mode == "all":
        selected = forced + candidates
    else:
        rng = random.Random(args.seed)

        def risk_score(row: dict[str, Any]) -> tuple[int, float, str]:
            notes = str(row.get("notes", ""))
            confidence = safe_float(row.get("confidence"))
            score = 0
            if confidence < 0.45:
                score += 3
            if notes:
                score += 2
            if "대표 카테고리" in notes or "차순위" in notes:
                score += 2
            if row.get("current_product_main_category") in {"기타", "영양보충"}:
                score += 1
            return (-score, confidence, str(row.get("report_no", "")))

        risky = sorted(candidates, key=risk_score)
        sample_size = len(risky) + len(forced) if args.sample_size <= 0 else args.sample_size
        top_risky_count = max(0, min(len(risky), int(sample_size * 0.7)))
        top_risky = risky[:top_risky_count]
        rest = risky[top_risky_count:]
        rng.shuffle(rest)
        selected = forced + top_risky + rest

    if args.limit > 0:
        return selected[: args.limit]
    if args.sample_size <= 0:
        return selected
    return selected[: args.sample_size]


def build_prompt(batch: list[dict[str, Any]], allowed_sub_categories: list[str]) -> str:
    payload = {
        "task": "건강기능식품 제품별 대표 메인 카테고리와 기능성 서브카테고리를 분류한다.",
        "allowed_categories": ALLOWED_CATEGORIES,
        "allowed_sub_categories": allowed_sub_categories,
        "rules": [
            "반드시 allowed_categories 중 하나만 predicted_main_category로 선택한다.",
            "predicted_sub_categories는 allowed_sub_categories 안에서만 고른다.",
            "predicted_sub_categories는 제품의 기능성 원료 계열을 중요도 순으로 0~5개만 선택한다.",
            "정확한 기능성 원료 계열이 없거나 애매하면 predicted_sub_categories는 빈 배열로 둔다.",
            "main_functionality는 최우선 근거다. raw_ingredients는 보조 근거로만 사용한다.",
            "부형제, 제형보조, 활택제, 감미료, 향료, 색소, 캡슐기제는 대표 카테고리 근거로 쓰지 않는다.",
            "부형제, 제형보조, 활택제, 감미료, 향료, 색소, 캡슐기제는 서브카테고리 근거로도 쓰지 않는다.",
            "현재 카테고리는 참고만 한다. main_functionality와 충돌하면 현재 카테고리를 따르지 않는다.",
            "현재 서브카테고리는 참고만 한다. 표시 기능성과 기능성 원료 근거가 약하면 그대로 따르지 않는다.",
            "복합 기능 제품은 가장 직접적이고 중심적인 표시 기능성을 하나만 선택한다.",
            "근거가 애매하면 confidence를 낮게 준다.",
            "입력 제품을 하나도 누락하지 말고 같은 report_no로 출력한다.",
        ],
        "output_schema": {
            "items": [
                {
                    "report_no": "string",
                    "predicted_main_category": "allowed category string",
                    "predicted_sub_categories": ["allowed sub category string"],
                    "confidence": 0.0,
                    "evidence_main_functionality": ["short evidence strings"],
                    "evidence_ingredients": ["short ingredient names"],
                    "ignored_noise": ["short ignored ingredient names"],
                    "reason": "short Korean reason",
                }
            ]
        },
        "products": [
            {
                "report_no": row["report_no"],
                "product_name": row["product_name"],
                "current_product_main_category": row["current_product_main_category"],
                "current_product_sub_categories": row["current_product_sub_categories"],
                "main_functionality": truncate_text(row["main_functionality"], 1400),
                "raw_ingredients": truncate_text(row["raw_ingredients"], 1400),
            }
            for row in batch
        ],
    }
    return (
        "[SYSTEM]\n"
        "너는 건강기능식품 표시 기능성과 기능성 원료 기준으로 제품 대표 카테고리를 분류하는 심사자다. "
        "출력은 반드시 JSON 객체 하나만 작성한다.\n\n"
        "[USER]\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def classify_batch(
    batch: list[dict[str, Any]],
    batch_index: int,
    allowed_sub_categories: list[str],
) -> tuple[list[dict[str, Any]], str]:
    prompt = build_prompt(batch, allowed_sub_categories)
    content = call_local_llm(prompt)
    parsed = extract_json_payload(content)
    items = parsed.get("items")
    if not isinstance(items, list):
        raise ValueError("LLM JSON missing items array")

    expected = {row["report_no"] for row in batch}
    by_report_no: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        report_no = str(item.get("report_no", "")).strip()
        if report_no:
            by_report_no[report_no] = item

    missing = expected - set(by_report_no)
    extra = set(by_report_no) - expected
    if missing or extra:
        raise ValueError(f"LLM report_no mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    rows = []
    for idx, source in enumerate(batch, start=1):
        item = by_report_no[source["report_no"]]
        predicted = str(item.get("predicted_main_category", "")).strip()
        if predicted not in ALLOWED_CATEGORIES:
            raise ValueError(f"invalid category for {source['report_no']}: {predicted}")
        predicted_sub_categories = canonicalize_sub_categories(
            safe_json_list(item.get("predicted_sub_categories", [])),
            allowed_sub_categories,
        )
        invalid_sub_categories = [
            name for name in predicted_sub_categories if name not in allowed_sub_categories
        ]
        if invalid_sub_categories:
            raise ValueError(
                f"invalid sub categories for {source['report_no']}: {invalid_sub_categories}"
            )
        confidence = max(0.0, min(1.0, safe_float(item.get("confidence"))))
        if confidence >= 0.9:
            decision = "auto_override_candidate"
        elif confidence >= 0.7:
            decision = "review_candidate"
        else:
            decision = "manual_review"
        rows.append(
            {
                "batch_index": batch_index,
                "sample_index": idx,
                "report_no": source["report_no"],
                "product_name": source["product_name"],
                "current_product_main_category": source["current_product_main_category"],
                "current_product_sub_categories": json.dumps(
                    source["current_product_sub_categories"], ensure_ascii=False
                ),
                "predicted_main_category": predicted,
                "predicted_sub_categories": json.dumps(predicted_sub_categories, ensure_ascii=False),
                "changed": int(predicted != source["current_product_main_category"]),
                "sub_categories_changed": int(
                    predicted_sub_categories != source["current_product_sub_categories"]
                ),
                "confidence": round(confidence, 4),
                "decision": decision,
                "evidence_main_functionality": json.dumps(
                    item.get("evidence_main_functionality", []), ensure_ascii=False
                ),
                "evidence_ingredients": json.dumps(item.get("evidence_ingredients", []), ensure_ascii=False),
                "ignored_noise": json.dumps(item.get("ignored_noise", []), ensure_ascii=False),
                "reason": str(item.get("reason", "")).strip(),
                "input_main_functionality": source["main_functionality"],
                "input_raw_ingredients": source["raw_ingredients"],
            }
        )
    return rows, content


def build_error_rows(batch: list[dict[str, Any]], batch_index: int, error: Exception) -> list[dict[str, Any]]:
    rows = []
    for idx, source in enumerate(batch, start=1):
        current_sub_categories = list(source.get("current_product_sub_categories", []))
        rows.append(
            {
                "batch_index": batch_index,
                "sample_index": idx,
                "report_no": source["report_no"],
                "product_name": source["product_name"],
                "current_product_main_category": source["current_product_main_category"],
                "current_product_sub_categories": json.dumps(current_sub_categories, ensure_ascii=False),
                "predicted_main_category": source["current_product_main_category"],
                "predicted_sub_categories": json.dumps(current_sub_categories, ensure_ascii=False),
                "changed": 0,
                "sub_categories_changed": 0,
                "confidence": 0.0,
                "decision": "llm_error",
                "evidence_main_functionality": "[]",
                "evidence_ingredients": "[]",
                "ignored_noise": "[]",
                "reason": f"LLM batch failed: {error}",
                "input_main_functionality": source["main_functionality"],
                "input_raw_ingredients": source["raw_ingredients"],
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [{field: row.get(field, "") for field in OUTPUT_FIELDS} for row in reader]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def backup_checkpoint(output_csv: Path, raw_output_jsonl: Path, backup_dir: Path, batch_index: int) -> None:
    if not output_csv.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"batch_{batch_index:05d}_{timestamp}"
    shutil.copy2(output_csv, backup_dir / f"{output_csv.stem}_{suffix}.csv")
    if raw_output_jsonl.exists():
        shutil.copy2(raw_output_jsonl, backup_dir / f"{raw_output_jsonl.stem}_{suffix}.jsonl")


def main() -> None:
    args = parse_args()
    rows = load_joined_rows(Path(args.profile_csv), Path(args.catalog_csv))
    selected = select_rows(rows, args)
    if not selected:
        raise SystemExit("No rows selected")
    allowed_sub_categories = load_allowed_sub_categories(Path(args.category_map_csv))
    if not allowed_sub_categories:
        raise SystemExit("No allowed sub categories loaded")

    output_csv_path = Path(args.output_csv)
    output_rows = read_existing_rows(output_csv_path) if args.resume else []
    processed_report_nos = {str(row.get("report_no", "")).strip() for row in output_rows}
    selected = [row for row in selected if row["report_no"] not in processed_report_nos]
    max_existing_batch = max([safe_int(row.get("batch_index")) for row in output_rows] or [0])
    raw_output_path = Path(args.raw_output_jsonl) if args.raw_output_jsonl else Path(args.output_csv).with_suffix(".raw.jsonl")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume or not raw_output_path.exists():
        raw_output_path.write_text("", encoding="utf-8")

    backup_dir = Path(args.backup_dir) if args.backup_dir else Path(args.output_csv).parent / "llm_category_classification_backups"
    if args.resume:
        print(f"[RESUME] existing_rows={len(output_rows)} remaining_rows={len(selected)}")

    last_batch_index = max_existing_batch
    for start in range(0, len(selected), args.batch_size):
        batch_index = max_existing_batch + start // args.batch_size + 1
        last_batch_index = batch_index
        batch = selected[start : start + args.batch_size]
        print(f"[BATCH] {batch_index} size={len(batch)}")
        last_error: Exception | None = None
        batch_rows: list[dict[str, Any]] = []
        raw_content = ""
        for attempt in range(args.max_retries + 1):
            try:
                batch_rows, raw_content = classify_batch(batch, batch_index, allowed_sub_categories)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[WARN] batch={batch_index} attempt={attempt + 1} failed: {exc}")
        if last_error is not None:
            if args.stop_on_error:
                raise last_error
            batch_rows = build_error_rows(batch, batch_index, last_error)
            raw_content = f"[ERROR] {last_error}"
        output_rows.extend(batch_rows)
        write_rows(output_csv_path, output_rows)
        with raw_output_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "batch_index": batch_index,
                        "report_nos": [row["report_no"] for row in batch],
                        "raw_content": raw_content,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if args.checkpoint_every_batches > 0 and batch_index % args.checkpoint_every_batches == 0:
            backup_checkpoint(output_csv_path, raw_output_path, backup_dir, batch_index)

    changed = sum(int(row["changed"]) for row in output_rows)
    sub_categories_changed = sum(int(row.get("sub_categories_changed") or 0) for row in output_rows)
    auto_candidates = sum(1 for row in output_rows if row["decision"] == "auto_override_candidate")
    error_rows = sum(1 for row in output_rows if row["decision"] == "llm_error")
    backup_checkpoint(output_csv_path, raw_output_path, backup_dir, last_batch_index)
    print(
        json.dumps(
            {
                "output_csv": str(output_csv_path),
                "raw_output_jsonl": str(raw_output_path),
                "backup_dir": str(backup_dir),
                "rows": len(output_rows),
                "changed": changed,
                "sub_categories_changed": sub_categories_changed,
                "auto_override_candidates": auto_candidates,
                "llm_error_rows": error_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
