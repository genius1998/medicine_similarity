from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "output" / "cache_db_excel_export" / "temp_csv" / "catalog_product_master.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "ingredient_normalization"

OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "（": "）",
    "［": "］",
    "【": "】",
    "「": "」",
    "『": "』",
}
CLOSE_TO_OPEN = {close: open_ for open_, close in OPEN_TO_CLOSE.items()}
DELIMITERS = {",", "，", "、"}
LONG_TOKEN_THRESHOLD = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse product raw ingredients and build normalized unique ingredient list.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="catalog_product_master.csv path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output CSV/JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Optional product row limit for testing")
    return parser.parse_args()


def normalize_readable(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_matching_key(value: str) -> str:
    text = normalize_readable(value).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def split_ingredients(raw_ingredients: str) -> tuple[list[str], dict]:
    text = normalize_readable(raw_ingredients)
    if not text:
        return [], {"unbalanced_brackets": False}

    tokens: list[str] = []
    current: list[str] = []
    stack: list[str] = []
    unbalanced = False

    for char in text:
        if char in OPEN_TO_CLOSE:
            stack.append(char)
            current.append(char)
            continue
        if char in CLOSE_TO_OPEN:
            if stack and stack[-1] == CLOSE_TO_OPEN[char]:
                stack.pop()
            else:
                unbalanced = True
            current.append(char)
            continue
        if char in DELIMITERS and not stack:
            token = normalize_readable("".join(current))
            if token:
                tokens.append(token)
            current = []
            continue
        current.append(char)

    token = normalize_readable("".join(current))
    if token:
        tokens.append(token)

    if stack:
        unbalanced = True
    return tokens, {"unbalanced_brackets": unbalanced}


def read_products(input_path: Path, limit: int | None) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"report_no", "product_name", "raw_ingredients"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            rows.append(row)
        return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    products = read_products(input_path, args.limit)

    mention_rows: list[dict] = []
    product_warning_rows: list[dict] = []
    long_token_rows: list[dict] = []
    unique_groups: dict[str, dict] = {}
    empty_raw_product_count = 0
    unbalanced_product_count = 0
    ingredient_count_by_product: list[int] = []

    for product_row_index, product in enumerate(products, start=1):
        report_no = normalize_readable(product.get("report_no", ""))
        product_name = normalize_readable(product.get("product_name", ""))
        raw_materials = product.get("raw_ingredients", "")
        ingredients, parse_meta = split_ingredients(raw_materials)
        warnings: list[str] = []
        if not normalize_readable(raw_materials):
            empty_raw_product_count += 1
            warnings.append("empty_raw_ingredients")
        if parse_meta.get("unbalanced_brackets"):
            unbalanced_product_count += 1
            warnings.append("unbalanced_brackets")
        ingredient_count_by_product.append(len(ingredients))
        if warnings:
            product_warning_rows.append(
                {
                    "product_row_index": product_row_index,
                    "report_no": report_no,
                    "product_name": product_name,
                    "warnings": "|".join(warnings),
                    "ingredient_count": len(ingredients),
                    "raw_ingredients_length": len(normalize_readable(raw_materials)),
                    "parsed_ingredient_samples_json": json.dumps(ingredients[:10], ensure_ascii=False),
                    "raw_ingredients_sample": normalize_readable(raw_materials)[:700],
                }
            )

        seen_keys_for_product: set[str] = set()
        for ingredient_index, raw_ingredient in enumerate(ingredients, start=1):
            readable = normalize_readable(raw_ingredient)
            key = normalize_matching_key(readable)
            if not key:
                continue
            if len(readable) >= LONG_TOKEN_THRESHOLD:
                long_token_rows.append(
                    {
                        "product_row_index": product_row_index,
                        "report_no": report_no,
                        "product_name": product_name,
                        "ingredient_index": ingredient_index,
                        "token_length": len(readable),
                        "raw_ingredient": raw_ingredient,
                        "normalized_key": key,
                        "parse_warning": "unbalanced_brackets" if parse_meta.get("unbalanced_brackets") else "",
                    }
                )

            mention_rows.append(
                {
                    "product_row_index": product_row_index,
                    "report_no": report_no,
                    "product_name": product_name,
                    "ingredient_index": ingredient_index,
                    "raw_ingredient": raw_ingredient,
                    "normalized_readable": readable,
                    "normalized_key": key,
                    "parse_warning": "unbalanced_brackets" if parse_meta.get("unbalanced_brackets") else "",
                }
            )

            group = unique_groups.setdefault(
                key,
                {
                    "normalized_key": key,
                    "variant_counter": Counter(),
                    "report_nos": set(),
                    "product_names": set(),
                    "mention_count": 0,
                },
            )
            group["variant_counter"][readable] += 1
            group["mention_count"] += 1
            if key not in seen_keys_for_product:
                group["report_nos"].add(report_no)
                group["product_names"].add(product_name)
                seen_keys_for_product.add(key)

    unique_rows: list[dict] = []
    for key, group in unique_groups.items():
        variants: Counter = group["variant_counter"]
        sorted_variants = sorted(variants.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        representative = sorted_variants[0][0]
        variant_samples = [name for name, _count in sorted_variants[:8]]
        report_samples = sorted(group["report_nos"])[:8]
        product_samples = sorted(group["product_names"])[:5]
        unique_rows.append(
            {
                "normalized_key": key,
                "representative_ingredient": representative,
                "mention_count": group["mention_count"],
                "product_count": len(group["report_nos"]),
                "variant_count": len(variants),
                "variant_samples_json": json.dumps(variant_samples, ensure_ascii=False),
                "report_no_samples_json": json.dumps(report_samples, ensure_ascii=False),
                "product_name_samples_json": json.dumps(product_samples, ensure_ascii=False),
            }
        )

    unique_rows.sort(key=lambda row: (-int(row["mention_count"]), row["representative_ingredient"], row["normalized_key"]))
    mention_rows.sort(key=lambda row: (int(row["product_row_index"]), int(row["ingredient_index"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    mentions_path = output_dir / "all_product_ingredient_mentions.csv"
    unique_path = output_dir / "unique_normalized_ingredients.csv"
    summary_path = output_dir / "ingredient_normalization_summary.json"
    warnings_path = output_dir / "parse_warning_products.csv"
    long_tokens_path = output_dir / "suspicious_long_ingredient_tokens.csv"

    write_csv(
        mentions_path,
        [
            "product_row_index",
            "report_no",
            "product_name",
            "ingredient_index",
            "raw_ingredient",
            "normalized_readable",
            "normalized_key",
            "parse_warning",
        ],
        mention_rows,
    )
    write_csv(
        unique_path,
        [
            "normalized_key",
            "representative_ingredient",
            "mention_count",
            "product_count",
            "variant_count",
            "variant_samples_json",
            "report_no_samples_json",
            "product_name_samples_json",
        ],
        unique_rows,
    )
    write_csv(
        warnings_path,
        [
            "product_row_index",
            "report_no",
            "product_name",
            "warnings",
            "ingredient_count",
            "raw_ingredients_length",
            "parsed_ingredient_samples_json",
            "raw_ingredients_sample",
        ],
        product_warning_rows,
    )
    write_csv(
        long_tokens_path,
        [
            "product_row_index",
            "report_no",
            "product_name",
            "ingredient_index",
            "token_length",
            "raw_ingredient",
            "normalized_key",
            "parse_warning",
        ],
        long_token_rows,
    )

    counts = ingredient_count_by_product or [0]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "product_count": len(products),
        "empty_raw_ingredient_product_count": empty_raw_product_count,
        "unbalanced_bracket_product_count": unbalanced_product_count,
        "ingredient_mention_count": len(mention_rows),
        "unique_normalized_ingredient_count": len(unique_rows),
        "parse_warning_product_count": len(product_warning_rows),
        "suspicious_long_token_count": len(long_token_rows),
        "avg_ingredients_per_product": round(sum(counts) / len(counts), 4),
        "min_ingredients_per_product": min(counts),
        "max_ingredients_per_product": max(counts),
        "top_20_ingredients": [
            {
                "representative_ingredient": row["representative_ingredient"],
                "normalized_key": row["normalized_key"],
                "mention_count": row["mention_count"],
                "product_count": row["product_count"],
            }
            for row in unique_rows[:20]
        ],
        "outputs": {
            "mentions_csv": str(mentions_path),
            "unique_csv": str(unique_path),
            "parse_warnings_csv": str(warnings_path),
            "suspicious_long_tokens_csv": str(long_tokens_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
