from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from api.db import sqlite_connection
from api.product_search_service import search_profile_records
from scripts.enhance_similarity_with_explanation import (
    build_vector_indexes,
    compute_topk_for_single_product,
    ensure_cache_table,
    load_cached_rows,
    load_product_function_profiles,
    load_vector_inputs,
    refresh_cache_rows,
    resolve_runtime_paths,
    safe_json_loads,
)


CATALOG_TABLE_NAME = "catalog_product_master"
CATALOG_META_TABLE_NAME = "catalog_product_master_meta"
CATALOG_COLUMN_ALIASES = {
    "license_no": ["인허가번호"],
    "company_name": ["업소명"],
    "report_no": ["품목제조번호"],
    "product_name": ["품목명"],
    "report_date": ["보고일자"],
    "product_type": ["제품형태"],
    "shelf_life": ["소비기한"],
    "appearance": ["성상"],
    "intake_method": ["섭취방법"],
    "main_functionality": ["주된기능성"],
    "cautions": ["섭취시주의사항"],
    "storage_method": ["보관방법"],
    "form_factor": ["형태"],
    "standard_spec": ["기준규격"],
    "raw_ingredients": ["원재료"],
}


class RecommendationService:
    def __init__(self) -> None:
        self._loaded = False
        self.runtime: Optional[Dict[str, Path]] = None
        self.profiles: Dict[str, dict] = {}
        self.profile_records: List[dict] = []
        self.profile_by_report_no: Dict[str, dict] = {}
        self.product_vectors: Dict[str, Dict[str, float]] = {}
        self.product_names: Dict[str, str] = {}
        self.report_to_product_ids: Dict[str, List[str]] = {}
        self.ingredient_postings: Dict[str, List[str]] = {}
        self.ingredient_frequency: Dict[str, int] = {}
        self.catalog_count: int = 0

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        self.runtime = resolve_runtime_paths()
        vector_df = load_vector_inputs(self.runtime["vector_csv_path"])
        self.profiles = load_product_function_profiles(self.runtime["product_profile_csv_path"])
        (
            self.product_vectors,
            self.product_names,
            self.report_to_product_ids,
            self.ingredient_postings,
            self.ingredient_frequency,
        ) = build_vector_indexes(vector_df)

        profile_df = pd.read_csv(self.runtime["product_profile_csv_path"], encoding="utf-8-sig", low_memory=False)
        self.profile_records = []
        for row in profile_df.to_dict(orient="records"):
            item = {
                "report_no": str(row.get("report_no", "") or ""),
                "product_id": str(row.get("product_id", "") or ""),
                "product_name": str(row.get("product_name", "") or ""),
                "product_main_category": str(row.get("product_main_category", "") or ""),
                "primary_ingredients": safe_json_loads(row.get("primary_ingredients_json", "[]"), []),
                "secondary_ingredients": safe_json_loads(row.get("secondary_ingredients_json", "[]"), []),
                "support_ingredients": safe_json_loads(row.get("support_ingredients_json", "[]"), []),
                "product_sub_categories": safe_json_loads(row.get("product_sub_categories_json", "[]"), []),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "notes": str(row.get("notes", "") or ""),
            }
            self.profile_records.append(item)
        self.profile_by_report_no = {item["report_no"]: item for item in self.profile_records if item["report_no"]}

        self._ensure_catalog_table()
        self._loaded = True

    def _find_catalog_csv_path(self) -> Optional[Path]:
        candidates = sorted(self.runtime["output_dir"].parent.glob("*C003.csv"))
        if candidates:
            return candidates[0]
        repo_candidates = sorted(Path(__file__).resolve().parents[1].glob("*C003.csv"))
        if repo_candidates:
            return repo_candidates[0]
        return None

    def _resolve_catalog_columns(self, catalog_df: pd.DataFrame) -> Dict[str, Optional[str]]:
        resolved = {}
        for key, aliases in CATALOG_COLUMN_ALIASES.items():
            resolved[key] = next((name for name in aliases if name in catalog_df.columns), None)
        return resolved

    def _ensure_catalog_table(self) -> None:
        csv_path = self._find_catalog_csv_path()
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {CATALOG_TABLE_NAME} (
                    report_no TEXT PRIMARY KEY,
                    product_name TEXT,
                    company_name TEXT,
                    license_no TEXT,
                    report_date TEXT,
                    product_type TEXT,
                    shelf_life TEXT,
                    appearance TEXT,
                    intake_method TEXT,
                    main_functionality TEXT,
                    cautions TEXT,
                    storage_method TEXT,
                    form_factor TEXT,
                    standard_spec TEXT,
                    raw_ingredients TEXT,
                    source_csv_path TEXT,
                    source_csv_mtime REAL,
                    imported_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {CATALOG_META_TABLE_NAME} (
                    meta_key TEXT PRIMARY KEY,
                    meta_value TEXT
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{CATALOG_TABLE_NAME}_product_name ON {CATALOG_TABLE_NAME}(product_name)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{CATALOG_TABLE_NAME}_company_name ON {CATALOG_TABLE_NAME}(company_name)"
            )

            row = conn.execute(
                f"SELECT COUNT(*), MAX(source_csv_mtime) FROM {CATALOG_TABLE_NAME}"
            ).fetchone()
            existing_count = int(row[0] or 0)
            existing_mtime = float(row[1] or 0.0)

            if not csv_path:
                self.catalog_count = existing_count
                return

            source_mtime = float(csv_path.stat().st_mtime)
            should_refresh = existing_count == 0 or abs(existing_mtime - source_mtime) > 0.0001
            if should_refresh:
                catalog_df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False).fillna("")
                resolved_columns = self._resolve_catalog_columns(catalog_df)
                report_no_column = resolved_columns.get("report_no")
                if not report_no_column:
                    self.catalog_count = existing_count
                    return

                catalog_df[report_no_column] = catalog_df[report_no_column].astype(str).str.strip()
                for key in ("product_name", "company_name"):
                    column_name = resolved_columns.get(key)
                    if column_name:
                        catalog_df[column_name] = catalog_df[column_name].astype(str).str.strip()

                catalog_df = catalog_df[catalog_df[report_no_column] != ""].copy()
                catalog_df = catalog_df.drop_duplicates(subset=[report_no_column], keep="first")

                def value_for(row_dict: dict, key: str) -> str:
                    column_name = resolved_columns.get(key)
                    if not column_name:
                        return ""
                    return str(row_dict.get(column_name, "") or "").strip()

                insert_rows = []
                for row_dict in catalog_df.to_dict(orient="records"):
                    insert_rows.append(
                        (
                            str(row_dict.get(report_no_column, "") or "").strip(),
                            value_for(row_dict, "product_name"),
                            value_for(row_dict, "company_name"),
                            value_for(row_dict, "license_no"),
                            value_for(row_dict, "report_date"),
                            value_for(row_dict, "product_type"),
                            value_for(row_dict, "shelf_life"),
                            value_for(row_dict, "appearance"),
                            value_for(row_dict, "intake_method"),
                            value_for(row_dict, "main_functionality"),
                            value_for(row_dict, "cautions"),
                            value_for(row_dict, "storage_method"),
                            value_for(row_dict, "form_factor"),
                            value_for(row_dict, "standard_spec"),
                            value_for(row_dict, "raw_ingredients"),
                            str(csv_path),
                            source_mtime,
                        )
                    )

                conn.execute(f"DELETE FROM {CATALOG_TABLE_NAME}")
                conn.executemany(
                    f"""
                    INSERT INTO {CATALOG_TABLE_NAME} (
                        report_no, product_name, company_name, license_no, report_date,
                        product_type, shelf_life, appearance, intake_method, main_functionality,
                        cautions, storage_method, form_factor, standard_spec, raw_ingredients,
                        source_csv_path, source_csv_mtime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_rows,
                )
                conn.execute(
                    f"""
                    INSERT INTO {CATALOG_META_TABLE_NAME}(meta_key, meta_value)
                    VALUES ('last_catalog_sync', CURRENT_TIMESTAMP)
                    ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value
                    """
                )
                conn.commit()
                self.catalog_count = len(insert_rows)
                return

            self.catalog_count = existing_count

    def _merge_catalog_with_profile(self, record: dict) -> dict:
        profile = self.profile_by_report_no.get(str(record.get("report_no", "")), {})
        result = dict(record)
        result.update(
            {
                "product_main_category": str(profile.get("product_main_category", "") or ""),
                "primary_ingredients": list(profile.get("primary_ingredients", [])),
                "secondary_ingredients": list(profile.get("secondary_ingredients", [])),
                "support_ingredients": list(profile.get("support_ingredients", [])),
                "product_sub_categories": list(profile.get("product_sub_categories", [])),
                "confidence": float(profile.get("confidence", 0.0) or 0.0),
                "notes": str(profile.get("notes", "") or ""),
            }
        )
        return result

    def health(self) -> dict:
        self.ensure_loaded()
        return {
            "loaded": self._loaded,
            "profile_count": len(self.profile_records),
            "vector_product_count": len(self.product_vectors),
            "catalog_count": self.catalog_count,
            "catalog_source": "sqlite",
        }

    def search_products(self, query: str, limit: int) -> List[dict]:
        self.ensure_loaded()
        return search_profile_records(self.profile_records, query, limit)

    def get_profile_by_report_no(self, report_no: str) -> Optional[dict]:
        self.ensure_loaded()
        return self.profile_by_report_no.get(str(report_no))

    def list_catalog_products(self, query: str, page: int, page_size: int) -> dict:
        self.ensure_loaded()
        query_text = str(query or "").strip()
        normalized = query_text.replace(" ", "")
        offset = max(0, (page - 1) * page_size)

        where_sql = ""
        params: List[object] = []
        if query_text:
            where_sql = """
            WHERE report_no = ?
               OR lower(product_name) LIKE lower(?)
               OR lower(company_name) LIKE lower(?)
               OR replace(lower(product_name), ' ', '') LIKE lower(?)
            """
            params = [query_text, f"%{query_text}%", f"%{query_text}%", f"%{normalized}%"]

        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            total_count = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM {CATALOG_TABLE_NAME} {where_sql}",
                params,
            ).fetchone()["cnt"]
            rows = conn.execute(
                f"""
                SELECT
                    report_no, product_name, company_name, license_no, report_date,
                    product_type, shelf_life, appearance, intake_method, main_functionality,
                    cautions, storage_method, form_factor, standard_spec, raw_ingredients
                FROM {CATALOG_TABLE_NAME}
                {where_sql}
                ORDER BY product_name, report_no
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            ).fetchall()

        results = [self._merge_catalog_with_profile(dict(row)) for row in rows]
        return {
            "query": query,
            "page": page,
            "page_size": page_size,
            "total_count": int(total_count or 0),
            "results": results,
        }

    def get_catalog_product_detail(self, report_no: str) -> Optional[dict]:
        self.ensure_loaded()
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT
                    report_no, product_name, company_name, license_no, report_date,
                    product_type, shelf_life, appearance, intake_method, main_functionality,
                    cautions, storage_method, form_factor, standard_spec, raw_ingredients
                FROM {CATALOG_TABLE_NAME}
                WHERE report_no = ?
                """,
                (str(report_no),),
            ).fetchone()
        if not row:
            return None
        result = self._merge_catalog_with_profile(dict(row))
        result["has_profile"] = bool(self.profile_by_report_no.get(str(report_no)))
        return result

    def resolve_product_id_by_report_no(self, report_no: str) -> str:
        self.ensure_loaded()
        matches = self.report_to_product_ids.get(str(report_no), [])
        if not matches:
            raise KeyError(f"report_no not found: {report_no}")
        return matches[0]

    def _convert_cache_row(self, row: dict, rank: int) -> dict:
        target_product_id = str(row.get("target_product_id", "") or "")
        target_report_no = target_product_id.split("::", 1)[1] if "::" in target_product_id else target_product_id
        return {
            "rank": rank,
            "target_report_no": target_report_no,
            "target_product_name": str(row.get("target_product_name", "") or ""),
            "similarity_score": float(row.get("similarity_score", 0.0) or 0.0),
            "function_similarity_score": float(row.get("function_similarity_score", 0.0) or 0.0),
            "core_match_score": float(row.get("core_match_score", 0.0) or 0.0),
            "substitutability": str(row.get("substitutability", "") or ""),
            "shared_ingredients": safe_json_loads(row.get("shared_ingredients_json", "[]"), []),
            "shared_categories": safe_json_loads(row.get("shared_categories_json", "[]"), []),
            "reason": str(row.get("reason", "") or ""),
            "caution": str(row.get("caution", "") or ""),
            "explanation": safe_json_loads(row.get("explanation_json", "{}"), {}),
        }

    def get_similar_products(self, report_no: str, top_k: int, candidate_limit: int, force_refresh: bool) -> dict:
        self.ensure_loaded()
        base_product_id = self.resolve_product_id_by_report_no(report_no)
        base_profile = self.profiles[base_product_id]
        start_time = time.perf_counter()

        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            ensure_cache_table(conn)
            cached_rows = []
            if not force_refresh:
                cached_rows = load_cached_rows(conn, base_product_id, top_k)
            if len(cached_rows) >= top_k:
                execution_seconds = round(time.perf_counter() - start_time, 6)
                return {
                    "base_product": {
                        "report_no": str(base_profile.get("report_no", "") or ""),
                        "product_name": str(base_profile.get("product_name", "") or ""),
                        "product_main_category": str(base_profile.get("product_main_category", "") or ""),
                        "primary_ingredients": list(base_profile.get("primary_ingredients", [])),
                    },
                    "recommendations": [self._convert_cache_row(row, idx) for idx, row in enumerate(cached_rows, start=1)],
                    "cache_used": True,
                    "execution_seconds": execution_seconds,
                }

            rows, failed_rows, _stats = compute_topk_for_single_product(
                base_product_id,
                top_k,
                candidate_limit,
                3000,
                self.product_vectors,
                self.product_names,
                self.profiles,
                self.ingredient_postings,
                self.ingredient_frequency,
            )
            if failed_rows:
                pass
            refresh_cache_rows(conn, base_product_id, rows)
            execution_seconds = round(time.perf_counter() - start_time, 6)
            return {
                "base_product": {
                    "report_no": str(base_profile.get("report_no", "") or ""),
                    "product_name": str(base_profile.get("product_name", "") or ""),
                    "product_main_category": str(base_profile.get("product_main_category", "") or ""),
                    "primary_ingredients": list(base_profile.get("primary_ingredients", [])),
                },
                "recommendations": [self._convert_cache_row(row, idx) for idx, row in enumerate(rows, start=1)],
                "cache_used": False,
                "execution_seconds": execution_seconds,
            }

    def recommend_by_ingredients(self, raw_ingredients: str, top_k: int, candidate_limit: int) -> dict:
        self.ensure_loaded()
        input_ingredients = [part.strip() for part in str(raw_ingredients or "").split(",") if part.strip()]
        normalized_inputs = {item.lower().replace(" ", ""): item for item in input_ingredients}
        detected = []
        for ingredient in self.ingredient_frequency:
            token = ingredient.lower().replace(" ", "")
            if token in normalized_inputs or any(token in key or key in token for key in normalized_inputs):
                detected.append(ingredient)
        return {
            "input_ingredients": input_ingredients,
            "detected_functional_ingredients": detected[:50],
            "estimated_main_category": None,
            "recommendations": [],
            "not_implemented": True,
        }
