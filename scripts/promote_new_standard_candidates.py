from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CATEGORY_MAP = ROOT_DIR / "output" / "cache_db_excel_export" / "temp_csv" / "functional_category_map.csv"
DEFAULT_BASE_METADATA = Path(r"D:\db\deploy_ec2\data\functional_ingredient_metadata_item_class_final_boosted.pkl")
DEFAULT_CACHE_DIR = ROOT_DIR / "output" / "ingredient_match_v2_curated"
DEFAULT_ENRICHMENT_DIR = ROOT_DIR / "output" / "ingredient_match_v2_promoted_enrichment"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "ingredient_match_v2_promoted"
DEFAULT_RUNTIME_SOURCE_SQLITE = ROOT_DIR / "output" / "ingredient_match_v2_curated" / "runtime_curated.sqlite"
DEFAULT_VECTOR_SOURCE_CSV = Path(r"D:\ec2_cache_snapshot\c003_product_functional_vectors_final_rebuilt.csv")
DEFAULT_MENTIONS_CSV = ROOT_DIR / "output" / "ingredient_normalization" / "all_product_ingredient_mentions.csv"
DEFAULT_CATALOG_CSV = ROOT_DIR / "output" / "cache_db_excel_export" / "temp_csv" / "catalog_product_master.csv"


CACHE_COLUMNS = [
    "ingredient_id",
    "raw_ingredient",
    "normalized_raw",
    "representative_ingredient",
    "mention_count",
    "product_count",
    "variant_samples_json",
    "decision",
    "matched_standard_name",
    "relation_type",
    "confidence",
    "category_main",
    "category_sub",
    "reason",
    "match_method",
    "rag_candidates_json",
    "llm_response_json",
    "needs_human_review",
    "cache_version",
    "created_at",
    "updated_at",
]

MAP_COLUMNS = [
    "functional_ingredient_name",
    "category_main",
    "category_sub",
    "claim_text",
    "source",
    "confidence",
    "notes",
    "categories_json",
    "function_text",
]

VECTOR_COLUMNS = [
    "product_id",
    "product_row_index",
    "product_name",
    "matched_standard_name",
    "weight",
    "best_relation_type",
    "best_confidence",
    "source_raw_ingredients",
    "source_match_methods",
    "품목제조번호",
    "인허가번호",
    "주된기능성",
    "섭취시주의사항",
    "기준규격",
    "보고일자",
    "소비기한",
    "성상",
    "섭취방법",
]


CONFLICT_EXISTING_OVERRIDES = {
    "ing_002161": ("비타민 C", "nutrient_form", 0.96, "아스코르빈산은 비타민 C 공급원/표기 변형이다."),
    "ing_003969": ("구기자추출물", "same_ingredient", 0.88, "구기자엑기스는 구기자추출물 표기 변형으로 본다."),
    "ing_004873": ("베타카로틴", "marker_compound", 0.86, "두나리엘라추출물 원료에 베타카로틴 함량이 명시되어 베타카로틴 지표 성분으로 연결한다."),
    "ing_011105": ("폴리코사놀 - 사탕수수 왁스 알코올", "same_ingredient", 0.96, "기존 DB에 동일한 폴리코사놀 사탕수수 왁스 알코올 표준이 있다."),
    "ing_011155": ("프로바이오틱스", "ingredient_group", 0.86, "락토바실러스 루테리 단일 균종 표기는 기존 프로바이오틱스 범주로 연결한다."),
}


def readable(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", str(value or "").strip().lower())


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
        if isinstance(parsed, list):
            return [readable(item) for item in parsed if readable(item)]
    except Exception:
        pass
    return []


def build_category_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup = {}
    for row in rows:
        name = readable(row.get("functional_ingredient_name"))
        if not name:
            continue
        lookup[name] = row
        lookup[normalize_key(name)] = row
    return lookup


def category_row_to_insert(row: dict[str, Any]) -> dict[str, Any]:
    category_main = readable(row.get("category_main")) or "기타"
    categories = [category_main] if category_main else []
    confidence = float(row.get("confidence") or 0.0)
    if confidence <= 0:
        confidence = 0.65
    return {
        "functional_ingredient_name": readable(row.get("standard_ingredient_name")),
        "category_main": category_main,
        "category_sub": readable(row.get("category_sub")),
        "claim_text": readable(row.get("claim_text") or row.get("function_description")),
        "source": "llm_rag_promoted_new_standard",
        "confidence": round(max(0.55, min(0.95, confidence)), 4),
        "notes": (
            f"promoted_by=llm_rag_final; review_status={row.get('review_status')}; "
            f"evidence_status={row.get('evidence_status')}; raw={readable(row.get('raw_ingredient'))}; "
            f"{readable(row.get('notes'))}"
        ).strip(),
        "categories_json": json.dumps(categories, ensure_ascii=False),
        "function_text": readable(row.get("function_description")),
    }


def enriched_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    name = readable(row.get("standard_ingredient_name"))
    aliases = parse_json_list(row.get("aliases_json"))
    if name and name not in aliases:
        aliases.insert(0, name)
    return {
        "id": f"functional_ingredient::{name}",
        "standard_name": name,
        "synonym_count": len(aliases),
        "synonyms_joined": ", ".join(aliases),
        "search_text": str(row.get("rag_search_text") or ""),
        "sources_joined": "llm_rag_promoted_new_standard",
        "function_text": str(row.get("function_description") or ""),
        "caution_text": "LLM/RAG로 승격된 확장 원료. 공식 고시 원료 여부는 별도 확인 필요.",
        "embedding_text": str(row.get("embedding_text") or row.get("rag_search_text") or name),
        "canonical_priority": 0,
        "specific_penalty": 0,
    }


def choose_unique_promotions(
    final_new_ids: set[str],
    enriched_rows: list[dict[str, str]],
    base_lookup: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    accepted_raw = []
    rejected_raw = []
    conflict_existing = []
    for row in enriched_rows:
        ingredient_id = readable(row.get("ingredient_id"))
        if ingredient_id not in final_new_ids:
            continue
        if row.get("review_status") == "reject_non_functional":
            rejected_raw.append(row)
            continue
        if ingredient_id in CONFLICT_EXISTING_OVERRIDES:
            conflict_existing.append(row)
            continue
        name = readable(row.get("standard_ingredient_name"))
        if not name:
            rejected_raw.append(row)
            continue
        exact_existing = base_lookup.get(name) or base_lookup.get(normalize_key(name))
        if exact_existing:
            conflict_existing.append(row)
            continue
        accepted_raw.append(row)

    def rank(row: dict[str, str]) -> tuple[int, float, int, str]:
        review_rank = {"add_ready": 0, "review_required": 1}.get(row.get("review_status"), 2)
        try:
            confidence = float(row.get("confidence") or 0)
        except ValueError:
            confidence = 0.0
        try:
            product_count = int(float(row.get("product_count") or 0))
        except ValueError:
            product_count = 0
        return (review_rank, -confidence, -product_count, readable(row.get("standard_ingredient_name")))

    by_name: dict[str, dict[str, str]] = {}
    for row in sorted(accepted_raw, key=rank):
        key = normalize_key(row.get("standard_ingredient_name"))
        if key and key not in by_name:
            by_name[key] = row
    return list(by_name.values()), rejected_raw, conflict_existing


def update_v2_cache(
    source_sqlite: Path,
    target_sqlite: Path,
    promoted_by_id: dict[str, dict[str, Any]],
    rejected_rows: list[dict[str, str]],
    conflict_rows: list[dict[str, str]],
    base_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    target_sqlite.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sqlite, target_sqlite)
    conn = sqlite3.connect(target_sqlite)
    conn.row_factory = sqlite3.Row
    now = datetime.now().isoformat(timespec="seconds")
    updated_promoted = 0
    updated_rejected = 0
    updated_conflicts = 0
    try:
        for ingredient_id, row in promoted_by_id.items():
            category = category_row_to_insert(row)
            confidence = float(category["confidence"])
            conn.execute(
                """
                UPDATE ingredient_match_cache_v2
                SET decision='existing_match',
                    matched_standard_name=?,
                    relation_type='same_ingredient',
                    confidence=?,
                    category_main=?,
                    category_sub=?,
                    reason=?,
                    match_method='llm_rag_promoted_standard_v1',
                    needs_human_review=0,
                    updated_at=?
                WHERE ingredient_id=?
                """,
                (
                    category["functional_ingredient_name"],
                    confidence,
                    category["category_main"],
                    category["category_sub"],
                    "LLM/RAG 최종 new_standard_candidate 판정 및 기능설명 생성 후 확장 표준 원료로 승격",
                    now,
                    ingredient_id,
                ),
            )
            updated_promoted += int(conn.total_changes > 0)

        for row in rejected_rows:
            ingredient_id = readable(row.get("ingredient_id"))
            conn.execute(
                """
                UPDATE ingredient_match_cache_v2
                SET decision='non_functional',
                    matched_standard_name='',
                    relation_type='excipient',
                    confidence=0.9,
                    category_main='',
                    category_sub='',
                    reason=?,
                    match_method='llm_rag_reject_non_functional_v1',
                    needs_human_review=0,
                    updated_at=?
                WHERE ingredient_id=?
                """,
                (
                    f"신규 후보 enrichment 최종 검토에서 reject_non_functional: {readable(row.get('notes'))}",
                    now,
                    ingredient_id,
                ),
            )
            updated_rejected += 1

        for row in conflict_rows:
            ingredient_id = readable(row.get("ingredient_id"))
            override = CONFLICT_EXISTING_OVERRIDES.get(ingredient_id)
            if not override:
                continue
            matched, relation, confidence, reason = override
            category = base_lookup.get(matched) or base_lookup.get(normalize_key(matched)) or {}
            conn.execute(
                """
                UPDATE ingredient_match_cache_v2
                SET decision='existing_match',
                    matched_standard_name=?,
                    relation_type=?,
                    confidence=?,
                    category_main=?,
                    category_sub=?,
                    reason=?,
                    match_method='llm_rag_conflict_resolved_existing_v1',
                    needs_human_review=0,
                    updated_at=?
                WHERE ingredient_id=?
                """,
                (
                    matched,
                    relation,
                    confidence,
                    readable(category.get("category_main")),
                    readable(category.get("category_sub")),
                    f"신규 후보 승격 전 기존 DB 충돌 해소: {reason}",
                    now,
                    ingredient_id,
                ),
            )
            updated_conflicts += 1
        conn.commit()
        rows = [dict(row) for row in conn.execute("SELECT * FROM ingredient_match_cache_v2 ORDER BY ingredient_id")]
    finally:
        conn.close()

    return {
        "updated_promoted": updated_promoted,
        "updated_rejected": updated_rejected,
        "updated_conflicts": updated_conflicts,
        "rows": rows,
        "decision_counts": dict(Counter(row.get("decision", "") for row in rows)),
        "method_counts": dict(Counter(row.get("match_method", "") for row in rows)),
    }


def create_runtime_sqlite(
    source_sqlite: Path,
    target_sqlite: Path,
    map_rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_sqlite.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sqlite, target_sqlite)
    conn = sqlite3.connect(target_sqlite)
    try:
        conn.execute("ALTER TABLE functional_category_map ADD COLUMN function_text TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    for row in map_rows:
        conn.execute(
            """
            INSERT INTO functional_category_map (
                functional_ingredient_name, category_main, category_sub, claim_text,
                source, confidence, notes, categories_json, function_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("functional_ingredient_name", ""),
                row.get("category_main", ""),
                row.get("category_sub", ""),
                row.get("claim_text", ""),
                row.get("source", ""),
                float(row.get("confidence") or 0.0),
                row.get("notes", ""),
                row.get("categories_json", "[]"),
                row.get("function_text", ""),
            ),
        )
    for row in cache_rows:
        if row.get("decision") != "existing_match":
            continue
        normalized = readable(row.get("normalized_raw"))
        raw = readable(row.get("raw_ingredient"))
        matched = readable(row.get("matched_standard_name"))
        if not normalized or not raw or not matched:
            continue
        if row.get("match_method") not in {
            "llm_rag_promoted_standard_v1",
            "llm_rag_conflict_resolved_existing_v1",
        }:
            continue
        conn.execute(
            """
            INSERT INTO ingredient_match_cache (
                raw_ingredient, normalized_raw, matched_standard_name, relation_type,
                confidence, reason, rag_candidates_json, gpt_response_json,
                created_at, updated_at, match_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            ON CONFLICT(normalized_raw) DO UPDATE SET
                raw_ingredient=excluded.raw_ingredient,
                matched_standard_name=excluded.matched_standard_name,
                relation_type=excluded.relation_type,
                confidence=excluded.confidence,
                reason=excluded.reason,
                rag_candidates_json=excluded.rag_candidates_json,
                gpt_response_json=excluded.gpt_response_json,
                updated_at=CURRENT_TIMESTAMP,
                match_method=excluded.match_method
            """,
            (
                raw,
                normalized,
                matched,
                row.get("relation_type", "same_ingredient"),
                float(row.get("confidence") or 0.0),
                row.get("reason", ""),
                row.get("rag_candidates_json", "[]"),
                row.get("llm_response_json", "{}"),
                row.get("match_method", ""),
            ),
        )
    conn.commit()
    counts = {
        "functional_category_map_count": conn.execute("SELECT COUNT(*) FROM functional_category_map").fetchone()[0],
        "ingredient_match_cache_count": conn.execute("SELECT COUNT(*) FROM ingredient_match_cache").fetchone()[0],
    }
    conn.close()
    return counts


def build_extended_vector_csv(
    source_vector_csv: Path,
    output_vector_csv: Path,
    mentions_csv: Path,
    catalog_csv: Path,
    cache_rows: list[dict[str, Any]],
    promoted_standard_names: set[str],
) -> dict[str, Any]:
    base_df = pd.read_csv(source_vector_csv, encoding="utf-8-sig", low_memory=False).fillna("")
    for column in VECTOR_COLUMNS:
        if column not in base_df.columns:
            base_df[column] = ""
    base_df = base_df[VECTOR_COLUMNS].copy()

    cache_by_key = {
        readable(row.get("normalized_raw")): row
        for row in cache_rows
        if row.get("decision") == "existing_match"
        and row.get("match_method") in {"llm_rag_promoted_standard_v1", "llm_rag_conflict_resolved_existing_v1"}
    }
    if not cache_by_key:
        output_vector_csv.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_csv(output_vector_csv, index=False, encoding="utf-8-sig")
        return {"base_vector_rows": len(base_df), "added_vector_rows": 0, "output_vector_rows": len(base_df)}

    mentions = pd.read_csv(mentions_csv, encoding="utf-8-sig", low_memory=False).fillna("")
    mentions["normalized_key"] = mentions["normalized_key"].astype(str).str.strip()
    mentions = mentions[mentions["normalized_key"].isin(cache_by_key.keys())].copy()
    catalog = pd.read_csv(catalog_csv, encoding="utf-8-sig", low_memory=False).fillna("")
    catalog_by_report = {str(row["report_no"]).strip(): row for row in catalog.to_dict(orient="records")}

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in mentions.to_dict(orient="records"):
        cache = cache_by_key.get(str(item.get("normalized_key", "")).strip())
        if not cache:
            continue
        matched = readable(cache.get("matched_standard_name"))
        if not matched:
            continue
        if cache.get("match_method") == "llm_rag_conflict_resolved_existing_v1" and matched not in promoted_standard_names:
            continue
        report_no = readable(item.get("report_no"))
        product_id = f"품목제조번호::{report_no}"
        key = (product_id, matched)
        try:
            confidence = float(cache.get("confidence") or 0.0)
        except ValueError:
            confidence = 0.0
        weight = round(max(0.55, min(0.95, confidence or 0.65)), 4)
        catalog_row = catalog_by_report.get(report_no, {})
        row = grouped.setdefault(
            key,
            {
                "product_id": product_id,
                "product_row_index": max(0, int(float(item.get("product_row_index") or 1)) - 1),
                "product_name": readable(item.get("product_name")),
                "matched_standard_name": matched,
                "weight": weight,
                "best_relation_type": readable(cache.get("relation_type")) or "same_ingredient",
                "best_confidence": confidence,
                "source_raw_ingredients": [],
                "source_match_methods": set(),
                "품목제조번호": report_no,
                "인허가번호": readable(catalog_row.get("license_no")),
                "주된기능성": readable(catalog_row.get("main_functionality")),
                "섭취시주의사항": readable(catalog_row.get("cautions")),
                "기준규격": readable(catalog_row.get("standard_spec")),
                "보고일자": readable(catalog_row.get("report_date")),
                "소비기한": readable(catalog_row.get("shelf_life")),
                "성상": readable(catalog_row.get("appearance")),
                "섭취방법": readable(catalog_row.get("intake_method")),
            },
        )
        row["weight"] = max(float(row["weight"]), weight)
        row["best_confidence"] = max(float(row["best_confidence"]), confidence)
        raw = readable(item.get("raw_ingredient"))
        if raw and raw not in row["source_raw_ingredients"]:
            row["source_raw_ingredients"].append(raw)
        row["source_match_methods"].add(readable(cache.get("match_method")))

    add_rows = []
    for row in grouped.values():
        row = dict(row)
        row["source_raw_ingredients"] = " | ".join(row["source_raw_ingredients"][:8])
        row["source_match_methods"] = " | ".join(sorted(row["source_match_methods"]))
        add_rows.append(row)
    add_df = pd.DataFrame(add_rows)
    if add_df.empty:
        out_df = base_df
    else:
        for column in VECTOR_COLUMNS:
            if column not in add_df.columns:
                add_df[column] = ""
        out_df = pd.concat([base_df, add_df[VECTOR_COLUMNS]], ignore_index=True)
        out_df["weight"] = pd.to_numeric(out_df["weight"], errors="coerce").fillna(0.0)
        out_df = (
            out_df.sort_values(["product_id", "matched_standard_name", "weight"], ascending=[True, True, False])
            .drop_duplicates(["product_id", "matched_standard_name"], keep="first")
            .reset_index(drop=True)
        )
    output_vector_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_vector_csv, index=False, encoding="utf-8-sig")
    return {
        "base_vector_rows": len(base_df),
        "added_vector_rows": len(add_rows),
        "output_vector_rows": len(out_df),
        "products_with_added_vectors": int(add_df["product_id"].nunique()) if not add_df.empty else 0,
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_category_rows = load_csv(Path(args.base_category_map))
    base_lookup = build_category_lookup(base_category_rows)
    cache_rows = load_csv(Path(args.cache_dir) / "ingredient_match_cache_v2.csv")
    final_new_ids = {row["ingredient_id"] for row in cache_rows if row.get("decision") == "new_standard_candidate"}
    enriched_rows = load_csv(Path(args.enrichment_dir) / "new_standard_candidates_enriched.csv")

    promoted_unique_rows, rejected_rows, conflict_rows = choose_unique_promotions(final_new_ids, enriched_rows, base_lookup)
    promoted_name_keys = {normalize_key(row.get("standard_ingredient_name")) for row in promoted_unique_rows}
    promoted_by_id = {
        row["ingredient_id"]: row
        for row in enriched_rows
        if row.get("ingredient_id") in final_new_ids
        and normalize_key(row.get("standard_ingredient_name")) in promoted_name_keys
        and row.get("review_status") != "reject_non_functional"
    }

    promoted_map_rows = [category_row_to_insert(row) for row in promoted_unique_rows]
    promoted_metadata_rows = [enriched_to_metadata(row) for row in promoted_unique_rows]
    promoted_standard_names = {row["functional_ingredient_name"] for row in promoted_map_rows}

    map_csv = output_dir / "functional_category_map_v2_promoted.csv"
    additions_csv = output_dir / "functional_category_map_promoted_additions.csv"
    write_csv(additions_csv, MAP_COLUMNS, promoted_map_rows)
    combined_map_rows = [dict(row, function_text=row.get("function_text", "")) for row in base_category_rows] + promoted_map_rows
    write_csv(map_csv, MAP_COLUMNS, combined_map_rows)
    shutil.copy2(map_csv, ROOT_DIR / "output" / "functional_category_map.csv")

    docs_csv = output_dir / "new_standard_candidate_rag_documents_promoted.csv"
    docs_jsonl = output_dir / "new_standard_candidate_rag_documents_promoted.jsonl"
    write_csv(
        docs_csv,
        ["id", "standard_name", "synonyms_joined", "search_text", "function_text", "sources_joined", "embedding_text"],
        promoted_metadata_rows,
    )
    write_jsonl(docs_jsonl, promoted_metadata_rows)

    base_metadata = []
    if Path(args.base_metadata).exists():
        with Path(args.base_metadata).open("rb") as handle:
            base_metadata = pickle.load(handle)
    existing_meta_keys = {normalize_key(item.get("standard_name")) for item in base_metadata if isinstance(item, dict)}
    combined_metadata = list(base_metadata)
    for row in promoted_metadata_rows:
        key = normalize_key(row.get("standard_name"))
        if key and key not in existing_meta_keys:
            combined_metadata.append(row)
            existing_meta_keys.add(key)
    metadata_pkl = output_dir / "functional_ingredient_metadata_v2_promoted.pkl"
    metadata_jsonl = output_dir / "functional_ingredient_metadata_v2_promoted.jsonl"
    with metadata_pkl.open("wb") as handle:
        pickle.dump(combined_metadata, handle)
    write_jsonl(metadata_jsonl, combined_metadata)

    v2_target_sqlite = output_dir / "ingredient_match_cache_v2_promoted.sqlite"
    v2_update = update_v2_cache(
        Path(args.cache_dir) / "ingredient_match_cache_v2.sqlite",
        v2_target_sqlite,
        promoted_by_id,
        rejected_rows,
        conflict_rows,
        base_lookup,
    )
    promoted_cache_rows = v2_update.pop("rows")
    write_csv(output_dir / "ingredient_match_cache_v2_promoted.csv", CACHE_COLUMNS, promoted_cache_rows)

    runtime_sqlite = output_dir / "runtime_promoted.sqlite"
    runtime_counts = create_runtime_sqlite(
        Path(args.runtime_source_sqlite),
        runtime_sqlite,
        promoted_map_rows,
        promoted_cache_rows,
    )

    vector_summary = build_extended_vector_csv(
        Path(args.vector_source_csv),
        ROOT_DIR / "output" / "c003_product_functional_vectors_final_rebuilt.csv",
        Path(args.mentions_csv),
        Path(args.catalog_csv),
        promoted_cache_rows,
        promoted_standard_names,
    )

    write_csv(output_dir / "promoted_rejected_non_functional.csv", list(enriched_rows[0].keys()), rejected_rows)
    write_csv(output_dir / "promoted_existing_conflicts_review.csv", list(enriched_rows[0].keys()), conflict_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "final_new_candidate_raw_count": len(final_new_ids),
        "promoted_raw_count": len(promoted_by_id),
        "promoted_unique_standard_count": len(promoted_map_rows),
        "rejected_non_functional_raw_count": len(rejected_rows),
        "conflict_raw_count": len(conflict_rows),
        "combined_category_map_count": len(combined_map_rows),
        "combined_metadata_count": len(combined_metadata),
        "v2_cache": v2_update,
        "runtime": {"runtime_sqlite": str(runtime_sqlite), **runtime_counts},
        "vectors": vector_summary,
        "outputs": {
            "promoted_category_map": str(map_csv),
            "promoted_additions": str(additions_csv),
            "runtime_sqlite": str(runtime_sqlite),
            "metadata_pkl": str(metadata_pkl),
            "rag_docs_csv": str(docs_csv),
            "vector_csv": str(ROOT_DIR / "output" / "c003_product_functional_vectors_final_rebuilt.csv"),
            "copied_runtime_category_map": str(ROOT_DIR / "output" / "functional_category_map.csv"),
        },
    }
    (output_dir / "promotion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote LLM/RAG new standard candidates into runtime category/similarity DB.")
    parser.add_argument("--base-category-map", default=str(DEFAULT_BASE_CATEGORY_MAP))
    parser.add_argument("--base-metadata", default=str(DEFAULT_BASE_METADATA))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--enrichment-dir", default=str(DEFAULT_ENRICHMENT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--runtime-source-sqlite", default=str(DEFAULT_RUNTIME_SOURCE_SQLITE))
    parser.add_argument("--vector-source-csv", default=str(DEFAULT_VECTOR_SOURCE_CSV))
    parser.add_argument("--mentions-csv", default=str(DEFAULT_MENTIONS_CSV))
    parser.add_argument("--catalog-csv", default=str(DEFAULT_CATALOG_CSV))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
