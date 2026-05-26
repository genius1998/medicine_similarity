from __future__ import annotations

import hashlib
import json
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
UPLOADED_PRODUCT_TABLE_NAME = "uploaded_product_master"
RECOMMENDATION_TRACE_TABLE_NAME = "recommendation_trace_log"
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

UPLOADED_PRODUCT_ADDITIONAL_COLUMNS = {
    "image_hash": "TEXT",
    "ocr_text_hash": "TEXT",
    "parsed_signature": "TEXT",
    "profile_signature": "TEXT",
    "status": "TEXT",
    "quality_grade": "TEXT",
    "is_candidate_enabled": "INTEGER DEFAULT 1",
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
        self.uploaded_catalog_count: int = 0

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
        self._ensure_uploaded_product_table()
        self._load_uploaded_products()
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

    def _ensure_uploaded_product_table(self) -> None:
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {UPLOADED_PRODUCT_TABLE_NAME} (
                    report_no TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL UNIQUE,
                    product_name TEXT,
                    raw_ingredients TEXT,
                    upload_signature TEXT,
                    image_hash TEXT,
                    ocr_text_hash TEXT,
                    parsed_signature TEXT,
                    profile_signature TEXT,
                    source_type TEXT,
                    source_filename TEXT,
                    ocr_raw_text TEXT,
                    ocr_confidence REAL,
                    ocr_confidence_source TEXT,
                    needs_user_review INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'raw',
                    quality_grade TEXT DEFAULT 'C',
                    is_candidate_enabled INTEGER DEFAULT 1,
                    product_main_category TEXT,
                    product_sub_categories_json TEXT,
                    primary_ingredients_json TEXT,
                    secondary_ingredients_json TEXT,
                    support_ingredients_json TEXT,
                    category_scores_json TEXT,
                    ingredient_scores_json TEXT,
                    role_by_ingredient_json TEXT,
                    category_by_ingredient_json TEXT,
                    vector_json TEXT,
                    parsed_json TEXT,
                    quality_warnings_json TEXT,
                    confidence REAL DEFAULT 0,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{UPLOADED_PRODUCT_TABLE_NAME}_product_name ON {UPLOADED_PRODUCT_TABLE_NAME}(product_name)"
            )
            columns = {
                str(row[1] or "")
                for row in conn.execute(f"PRAGMA table_info({UPLOADED_PRODUCT_TABLE_NAME})").fetchall()
            }
            if "upload_signature" not in columns:
                conn.execute(f"ALTER TABLE {UPLOADED_PRODUCT_TABLE_NAME} ADD COLUMN upload_signature TEXT")
            for column_name, column_type in UPLOADED_PRODUCT_ADDITIONAL_COLUMNS.items():
                if column_name not in columns:
                    conn.execute(f"ALTER TABLE {UPLOADED_PRODUCT_TABLE_NAME} ADD COLUMN {column_name} {column_type}")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{UPLOADED_PRODUCT_TABLE_NAME}_upload_signature ON {UPLOADED_PRODUCT_TABLE_NAME}(upload_signature)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{UPLOADED_PRODUCT_TABLE_NAME}_candidate_enabled ON {UPLOADED_PRODUCT_TABLE_NAME}(is_candidate_enabled)"
            )
            conn.execute(
                f"""
                UPDATE {UPLOADED_PRODUCT_TABLE_NAME}
                SET status = COALESCE(NULLIF(status, ''), 'raw'),
                    quality_grade = COALESCE(NULLIF(quality_grade, ''), 'C')
                """
            )
            conn.execute(
                f"""
                UPDATE {UPLOADED_PRODUCT_TABLE_NAME}
                SET is_candidate_enabled = 0
                WHERE COALESCE(NULLIF(profile_signature, ''), '') = ''
                   OR COALESCE(NULLIF(parsed_signature, ''), '') = ''
                """
            )
            conn.commit()

    def ensure_recommendation_trace_table(self) -> None:
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {RECOMMENDATION_TRACE_TABLE_NAME} (
                    trace_id TEXT PRIMARY KEY,
                    input_type TEXT,
                    product_name TEXT,
                    image_hash TEXT,
                    ocr_text_hash TEXT,
                    parsed_signature TEXT,
                    profile_signature TEXT,
                    upload_signature TEXT,
                    candidate_count INTEGER DEFAULT 0,
                    recommendations_json TEXT,
                    warnings_json TEXT,
                    metadata_json TEXT,
                    execution_seconds REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{RECOMMENDATION_TRACE_TABLE_NAME}_created_at ON {RECOMMENDATION_TRACE_TABLE_NAME}(created_at)"
            )
            conn.commit()

    def _remove_product_from_indexes(self, product_id: str) -> None:
        old_profile = self.profiles.pop(product_id, None)
        old_vector = self.product_vectors.pop(product_id, None)
        self.product_names.pop(product_id, None)

        if old_profile:
            report_no = str(old_profile.get("report_no", "") or "")
            if report_no in self.profile_by_report_no and self.profile_by_report_no[report_no].get("product_id") == product_id:
                self.profile_by_report_no.pop(report_no, None)
            self.profile_records = [item for item in self.profile_records if str(item.get("product_id", "")) != product_id]
            if report_no in self.report_to_product_ids:
                self.report_to_product_ids[report_no] = [item for item in self.report_to_product_ids[report_no] if item != product_id]
                if not self.report_to_product_ids[report_no]:
                    self.report_to_product_ids.pop(report_no, None)

        if old_vector:
            for ingredient in old_vector:
                posting_list = [item for item in self.ingredient_postings.get(ingredient, []) if item != product_id]
                if posting_list:
                    self.ingredient_postings[ingredient] = posting_list
                else:
                    self.ingredient_postings.pop(ingredient, None)
                updated_freq = int(self.ingredient_frequency.get(ingredient, 0)) - 1
                if updated_freq > 0:
                    self.ingredient_frequency[ingredient] = updated_freq
                else:
                    self.ingredient_frequency.pop(ingredient, None)

    def _register_uploaded_profile_in_memory(self, row: dict) -> None:
        product_id = str(row.get("product_id", "") or "").strip()
        report_no = str(row.get("report_no", "") or "").strip()
        if not product_id or not report_no:
            return

        self._remove_product_from_indexes(product_id)

        vector = {
            str(key): float(value)
            for key, value in safe_json_loads(row.get("vector_json", "{}"), {}).items()
            if str(key).strip() and float(value or 0.0) > 0
        }
        profile = {
            "product_id": product_id,
            "report_no": report_no,
            "product_name": str(row.get("product_name", "") or ""),
            "upload_signature": str(row.get("upload_signature", "") or ""),
            "image_hash": str(row.get("image_hash", "") or ""),
            "ocr_text_hash": str(row.get("ocr_text_hash", "") or ""),
            "parsed_signature": str(row.get("parsed_signature", "") or ""),
            "profile_signature": str(row.get("profile_signature", "") or ""),
            "status": str(row.get("status", "") or "raw"),
            "quality_grade": str(row.get("quality_grade", "") or "C"),
            "is_candidate_enabled": bool(int(row.get("is_candidate_enabled", 1) or 0)),
            "product_main_category": str(row.get("product_main_category", "") or "기타"),
            "product_sub_categories": safe_json_loads(row.get("product_sub_categories_json", "[]"), []),
            "primary_ingredients": safe_json_loads(row.get("primary_ingredients_json", "[]"), []),
            "secondary_ingredients": safe_json_loads(row.get("secondary_ingredients_json", "[]"), []),
            "support_ingredients": safe_json_loads(row.get("support_ingredients_json", "[]"), []),
            "category_scores": safe_json_loads(row.get("category_scores_json", "{}"), {}),
            "ingredient_scores": safe_json_loads(row.get("ingredient_scores_json", "[]"), []),
            "role_by_ingredient": safe_json_loads(row.get("role_by_ingredient_json", "{}"), {}),
            "category_by_ingredient": safe_json_loads(row.get("category_by_ingredient_json", "{}"), {}),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "notes": str(row.get("notes", "") or ""),
        }
        record = {
            "report_no": report_no,
            "product_id": product_id,
            "product_name": profile["product_name"],
            "upload_signature": profile["upload_signature"],
            "image_hash": profile["image_hash"],
            "ocr_text_hash": profile["ocr_text_hash"],
            "parsed_signature": profile["parsed_signature"],
            "profile_signature": profile["profile_signature"],
            "status": profile["status"],
            "quality_grade": profile["quality_grade"],
            "is_candidate_enabled": profile["is_candidate_enabled"],
            "product_main_category": profile["product_main_category"],
            "primary_ingredients": list(profile["primary_ingredients"]),
            "secondary_ingredients": list(profile["secondary_ingredients"]),
            "support_ingredients": list(profile["support_ingredients"]),
            "confidence": profile["confidence"],
            "notes": profile["notes"],
        }

        self.profiles[product_id] = profile
        self.product_vectors[product_id] = vector
        self.product_names[product_id] = profile["product_name"]
        self.report_to_product_ids.setdefault(report_no, [])
        if product_id not in self.report_to_product_ids[report_no]:
            self.report_to_product_ids[report_no].append(product_id)
        self.profile_by_report_no[report_no] = record
        self.profile_records.append(record)

        for ingredient in vector:
            postings = self.ingredient_postings.setdefault(ingredient, [])
            if product_id not in postings:
                postings.append(product_id)
                self.ingredient_frequency[ingredient] = int(self.ingredient_frequency.get(ingredient, 0)) + 1

    def _load_uploaded_products(self) -> None:
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT
                    report_no, product_id, product_name, raw_ingredients, upload_signature,
                    image_hash, ocr_text_hash, parsed_signature, profile_signature,
                    status, quality_grade, is_candidate_enabled,
                    product_main_category, product_sub_categories_json,
                    primary_ingredients_json, secondary_ingredients_json, support_ingredients_json,
                    category_scores_json, ingredient_scores_json, role_by_ingredient_json,
                    category_by_ingredient_json, vector_json, confidence, notes
                FROM {UPLOADED_PRODUCT_TABLE_NAME}
                ORDER BY created_at ASC, report_no ASC
                """
            ).fetchall()

        self.uploaded_catalog_count = len(rows)
        for row in rows:
            self._register_uploaded_profile_in_memory(dict(row))

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
            "catalog_count": self.catalog_count + self.uploaded_catalog_count,
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
        rows = self._load_catalog_rows(query_text, normalized)
        total_count = len(rows)
        paged_rows = rows[offset : offset + page_size]
        results = [self._merge_catalog_with_profile(row) for row in paged_rows]
        return {
            "query": query,
            "page": page,
            "page_size": page_size,
            "total_count": int(total_count or 0),
            "results": results,
        }

    def get_catalog_product_detail(self, report_no: str) -> Optional[dict]:
        self.ensure_loaded()
        rows = self._load_catalog_rows(str(report_no), str(report_no).replace(" ", ""))
        target = next((row for row in rows if str(row.get("report_no", "")) == str(report_no)), None)
        if not target:
            return None
        result = self._merge_catalog_with_profile(target)
        result["has_profile"] = bool(self.profile_by_report_no.get(str(report_no)))
        return result

    def _load_catalog_rows(self, query_text: str, normalized: str) -> List[dict]:
        query_text = str(query_text or "").strip()
        normalized = str(normalized or "").strip().lower()

        def matches(row: dict) -> bool:
            if not query_text:
                return True
            report_no = str(row.get("report_no", "") or "")
            product_name = str(row.get("product_name", "") or "")
            company_name = str(row.get("company_name", "") or "")
            collapsed_name = product_name.replace(" ", "").lower()
            needle = query_text.lower()
            return (
                report_no == query_text
                or needle in product_name.lower()
                or needle in company_name.lower()
                or (normalized and normalized in collapsed_name)
            )

        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            base_rows = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT
                        report_no, product_name, company_name, license_no, report_date,
                        product_type, shelf_life, appearance, intake_method, main_functionality,
                        cautions, storage_method, form_factor, standard_spec, raw_ingredients
                    FROM {CATALOG_TABLE_NAME}
                    ORDER BY product_name, report_no
                    """
                ).fetchall()
            ]
            uploaded_rows = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT
                        report_no,
                        product_name,
                        '' AS company_name,
                        '' AS license_no,
                        '' AS report_date,
                        source_type AS product_type,
                        '' AS shelf_life,
                        '' AS appearance,
                        '' AS intake_method,
                        product_main_category AS main_functionality,
                        '' AS cautions,
                        '' AS storage_method,
                        '' AS form_factor,
                        '' AS standard_spec,
                        raw_ingredients
                    FROM {UPLOADED_PRODUCT_TABLE_NAME}
                    ORDER BY product_name, report_no
                    """
                ).fetchall()
            ]

        combined = [row for row in base_rows if matches(row)] + [row for row in uploaded_rows if matches(row)]
        combined.sort(key=lambda item: (str(item.get("product_name", "")), str(item.get("report_no", ""))))
        return combined

    def register_uploaded_product(
        self,
        *,
        source_type: str,
        product_name: str,
        parsed: dict,
        estimated_profile: dict,
        vector: Dict[str, float],
        ocr_payload: Optional[dict] = None,
        source_filename: str = "",
        upload_signature: str = "",
        image_hash: str = "",
        ocr_text_hash: str = "",
        parsed_signature: str = "",
        profile_signature: str = "",
        status: str = "raw",
        quality_grade: str = "C",
        is_candidate_enabled: bool = True,
        notes: str = "",
        needs_user_review: bool = False,
        quality_warnings: Optional[List[dict]] = None,
    ) -> Optional[dict]:
        self.ensure_loaded()
        cleaned_name = str(product_name or "").strip() or "업로드 상품"
        normalized_ingredients = [str(item).strip() for item in parsed.get("normalized_ingredients", []) if str(item).strip()]
        has_vector_payload = bool(vector)
        has_normalized_ingredients = bool(normalized_ingredients)

        upload_signature = str(upload_signature or "").strip()
        if upload_signature:
            fingerprint_source = upload_signature
        else:
            fingerprint_source = json.dumps(
                {
                    "product_name": cleaned_name,
                    "normalized_ingredients": sorted(normalized_ingredients),
                    "primary_ingredients": estimated_profile.get("primary_ingredients", []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        fingerprint = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()
        report_no = f"UPLOADED-{fingerprint[:12].upper()}"
        product_id = f"uploaded::{report_no}"

        stored_profile = {
            "product_id": product_id,
            "report_no": report_no,
            "product_name": cleaned_name,
            "upload_signature": upload_signature,
            "image_hash": str(image_hash or ""),
            "ocr_text_hash": str(ocr_text_hash or ""),
            "parsed_signature": str(parsed_signature or ""),
            "profile_signature": str(profile_signature or ""),
            "status": str(status or "raw"),
            "quality_grade": str(quality_grade or "C"),
            "is_candidate_enabled": bool(is_candidate_enabled),
            "product_main_category": str(estimated_profile.get("product_main_category", "") or "기타"),
            "product_sub_categories": list(estimated_profile.get("product_sub_categories", [])),
            "primary_ingredients": list(estimated_profile.get("primary_ingredients", [])),
            "secondary_ingredients": list(estimated_profile.get("secondary_ingredients", [])),
            "support_ingredients": list(estimated_profile.get("support_ingredients", [])),
            "category_scores": dict(estimated_profile.get("category_scores", {})),
            "ingredient_scores": list(estimated_profile.get("ingredient_scores", [])),
            "role_by_ingredient": dict(estimated_profile.get("role_by_ingredient", {})),
            "category_by_ingredient": dict(estimated_profile.get("category_by_ingredient", {})),
            "confidence": float(estimated_profile.get("confidence", 0.0) or 0.0),
            "notes": str(notes or estimated_profile.get("notes", "") or ""),
        }
        ocr_payload = ocr_payload or {}
        quality_warnings = quality_warnings or []

        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            ensure_cache_table(conn)
            conn.execute(
                f"""
                INSERT INTO {UPLOADED_PRODUCT_TABLE_NAME} (
                    report_no, product_id, product_name, raw_ingredients, upload_signature,
                    image_hash, ocr_text_hash, parsed_signature, profile_signature, status, quality_grade, is_candidate_enabled,
                    source_type, source_filename, ocr_raw_text, ocr_confidence, ocr_confidence_source,
                    needs_user_review, product_main_category, product_sub_categories_json,
                    primary_ingredients_json, secondary_ingredients_json, support_ingredients_json,
                    category_scores_json, ingredient_scores_json, role_by_ingredient_json,
                    category_by_ingredient_json, vector_json, parsed_json, quality_warnings_json,
                    confidence, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(report_no) DO UPDATE SET
                    product_id=excluded.product_id,
                    product_name=excluded.product_name,
                    raw_ingredients=excluded.raw_ingredients,
                    upload_signature=excluded.upload_signature,
                    image_hash=excluded.image_hash,
                    ocr_text_hash=excluded.ocr_text_hash,
                    parsed_signature=excluded.parsed_signature,
                    profile_signature=excluded.profile_signature,
                    status=excluded.status,
                    quality_grade=excluded.quality_grade,
                    is_candidate_enabled=excluded.is_candidate_enabled,
                    source_type=excluded.source_type,
                    source_filename=excluded.source_filename,
                    ocr_raw_text=excluded.ocr_raw_text,
                    ocr_confidence=excluded.ocr_confidence,
                    ocr_confidence_source=excluded.ocr_confidence_source,
                    needs_user_review=excluded.needs_user_review,
                    product_main_category=excluded.product_main_category,
                    product_sub_categories_json=excluded.product_sub_categories_json,
                    primary_ingredients_json=excluded.primary_ingredients_json,
                    secondary_ingredients_json=excluded.secondary_ingredients_json,
                    support_ingredients_json=excluded.support_ingredients_json,
                    category_scores_json=excluded.category_scores_json,
                    ingredient_scores_json=excluded.ingredient_scores_json,
                    role_by_ingredient_json=excluded.role_by_ingredient_json,
                    category_by_ingredient_json=excluded.category_by_ingredient_json,
                    vector_json=excluded.vector_json,
                    parsed_json=excluded.parsed_json,
                    quality_warnings_json=excluded.quality_warnings_json,
                    confidence=excluded.confidence,
                    notes=excluded.notes,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    report_no,
                    product_id,
                    cleaned_name,
                    ", ".join(str(item).strip() for item in parsed.get("raw_ingredients", []) if str(item).strip()),
                    stored_profile["upload_signature"],
                    stored_profile["image_hash"],
                    stored_profile["ocr_text_hash"],
                    stored_profile["parsed_signature"],
                    stored_profile["profile_signature"],
                    stored_profile["status"],
                    stored_profile["quality_grade"],
                    1 if stored_profile["is_candidate_enabled"] else 0,
                    str(source_type or "upload"),
                    str(source_filename or ""),
                    str(ocr_payload.get("raw_text", "") or ""),
                    ocr_payload.get("confidence"),
                    str(ocr_payload.get("confidence_source", "unavailable") or "unavailable"),
                    1 if needs_user_review else 0,
                    stored_profile["product_main_category"],
                    json.dumps(stored_profile["product_sub_categories"], ensure_ascii=False),
                    json.dumps(stored_profile["primary_ingredients"], ensure_ascii=False),
                    json.dumps(stored_profile["secondary_ingredients"], ensure_ascii=False),
                    json.dumps(stored_profile["support_ingredients"], ensure_ascii=False),
                    json.dumps(stored_profile["category_scores"], ensure_ascii=False),
                    json.dumps(stored_profile["ingredient_scores"], ensure_ascii=False),
                    json.dumps(stored_profile["role_by_ingredient"], ensure_ascii=False),
                    json.dumps(stored_profile["category_by_ingredient"], ensure_ascii=False),
                    json.dumps(vector or {}, ensure_ascii=False),
                    json.dumps(parsed, ensure_ascii=False),
                    json.dumps(quality_warnings, ensure_ascii=False),
                    stored_profile["confidence"],
                    stored_profile["notes"],
                ),
            )
            conn.execute("DELETE FROM product_similarity_explanation_cache")
            conn.commit()

        if has_vector_payload and has_normalized_ingredients:
            self._register_uploaded_profile_in_memory(
                {
                    "report_no": report_no,
                    "product_id": product_id,
                    "product_name": cleaned_name,
                    "upload_signature": stored_profile["upload_signature"],
                    "image_hash": stored_profile["image_hash"],
                    "ocr_text_hash": stored_profile["ocr_text_hash"],
                    "parsed_signature": stored_profile["parsed_signature"],
                    "profile_signature": stored_profile["profile_signature"],
                    "status": stored_profile["status"],
                    "quality_grade": stored_profile["quality_grade"],
                    "is_candidate_enabled": 1 if stored_profile["is_candidate_enabled"] else 0,
                    "product_main_category": stored_profile["product_main_category"],
                    "product_sub_categories_json": json.dumps(stored_profile["product_sub_categories"], ensure_ascii=False),
                    "primary_ingredients_json": json.dumps(stored_profile["primary_ingredients"], ensure_ascii=False),
                    "secondary_ingredients_json": json.dumps(stored_profile["secondary_ingredients"], ensure_ascii=False),
                    "support_ingredients_json": json.dumps(stored_profile["support_ingredients"], ensure_ascii=False),
                    "category_scores_json": json.dumps(stored_profile["category_scores"], ensure_ascii=False),
                    "ingredient_scores_json": json.dumps(stored_profile["ingredient_scores"], ensure_ascii=False),
                    "role_by_ingredient_json": json.dumps(stored_profile["role_by_ingredient"], ensure_ascii=False),
                    "category_by_ingredient_json": json.dumps(stored_profile["category_by_ingredient"], ensure_ascii=False),
                    "vector_json": json.dumps(vector or {}, ensure_ascii=False),
                    "confidence": stored_profile["confidence"],
                    "notes": stored_profile["notes"],
                }
            )
        self.uploaded_catalog_count = self._count_uploaded_products()
        return {
            "report_no": report_no,
            "product_name": cleaned_name,
            "included_in_recommendations": bool(stored_profile["is_candidate_enabled"]),
        }

    def _count_uploaded_products(self) -> int:
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {UPLOADED_PRODUCT_TABLE_NAME}").fetchone()
        return int(row[0] or 0)

    def find_cached_ocr_payload_by_upload_signature(self, upload_signature: str) -> Optional[dict]:
        self.ensure_loaded()
        signature = str(upload_signature or "").strip()
        if not signature:
            return None
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT report_no, product_name, upload_signature, image_hash, source_filename,
                       ocr_raw_text, ocr_confidence, ocr_confidence_source, parsed_json
                FROM {UPLOADED_PRODUCT_TABLE_NAME}
                WHERE upload_signature = ?
                  AND COALESCE(NULLIF(ocr_raw_text, ''), '') <> ''
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (signature,),
            ).fetchone()
        if not row:
            return None
        parsed_payload = safe_json_loads(row["parsed_json"], {}) if row["parsed_json"] else {}
        return {
            "report_no": str(row["report_no"] or ""),
            "product_name": str(row["product_name"] or ""),
            "upload_signature": str(row["upload_signature"] or ""),
            "image_hash": str(row["image_hash"] or ""),
            "source_filename": str(row["source_filename"] or ""),
            "raw_text": str(row["ocr_raw_text"] or ""),
            "confidence": row["ocr_confidence"],
            "confidence_source": str(row["ocr_confidence_source"] or "cached"),
            "lines": [],
            "blocks": [],
            "source": "ocr_cache",
            "parsed_json": parsed_payload,
        }

    def find_uploaded_record_by_upload_signature(self, upload_signature: str) -> Optional[dict]:
        self.ensure_loaded()
        signature = str(upload_signature or "").strip()
        if not signature:
            return None
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT report_no, product_id, product_name, upload_signature, image_hash, ocr_text_hash,
                       parsed_signature, profile_signature, status, quality_grade, is_candidate_enabled,
                       product_main_category, product_sub_categories_json,
                       primary_ingredients_json, secondary_ingredients_json, support_ingredients_json,
                       parsed_json, quality_warnings_json, confidence, notes
                FROM {UPLOADED_PRODUCT_TABLE_NAME}
                WHERE upload_signature = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "report_no": str(row["report_no"] or ""),
            "product_id": str(row["product_id"] or ""),
            "product_name": str(row["product_name"] or ""),
            "upload_signature": str(row["upload_signature"] or ""),
            "image_hash": str(row["image_hash"] or ""),
            "ocr_text_hash": str(row["ocr_text_hash"] or ""),
            "parsed_signature": str(row["parsed_signature"] or ""),
            "profile_signature": str(row["profile_signature"] or ""),
            "status": str(row["status"] or ""),
            "quality_grade": str(row["quality_grade"] or ""),
            "is_candidate_enabled": bool(row["is_candidate_enabled"]),
            "product_main_category": str(row["product_main_category"] or ""),
            "product_sub_categories": safe_json_loads(row["product_sub_categories_json"], []) if row["product_sub_categories_json"] else [],
            "primary_ingredients": safe_json_loads(row["primary_ingredients_json"], []) if row["primary_ingredients_json"] else [],
            "secondary_ingredients": safe_json_loads(row["secondary_ingredients_json"], []) if row["secondary_ingredients_json"] else [],
            "support_ingredients": safe_json_loads(row["support_ingredients_json"], []) if row["support_ingredients_json"] else [],
            "parsed_json": safe_json_loads(row["parsed_json"], {}) if row["parsed_json"] else {},
            "quality_warnings": safe_json_loads(row["quality_warnings_json"], []) if row["quality_warnings_json"] else [],
            "confidence": float(row["confidence"] or 0.0),
            "notes": str(row["notes"] or ""),
        }

    def log_recommendation_trace(
        self,
        *,
        trace_id: str,
        input_type: str,
        product_name: str,
        image_hash: str = "",
        ocr_text_hash: str = "",
        parsed_signature: str = "",
        profile_signature: str = "",
        upload_signature: str = "",
        candidate_count: int = 0,
        recommendations: Optional[List[dict]] = None,
        warnings: Optional[List[dict]] = None,
        metadata: Optional[dict] = None,
        execution_seconds: float = 0.0,
    ) -> None:
        self.ensure_loaded()
        self.ensure_recommendation_trace_table()
        with sqlite_connection(self.runtime["sqlite_path"]) as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {RECOMMENDATION_TRACE_TABLE_NAME} (
                    trace_id, input_type, product_name, image_hash, ocr_text_hash,
                    parsed_signature, profile_signature, upload_signature, candidate_count,
                    recommendations_json, warnings_json, metadata_json, execution_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(trace_id or ""),
                    str(input_type or ""),
                    str(product_name or ""),
                    str(image_hash or ""),
                    str(ocr_text_hash or ""),
                    str(parsed_signature or ""),
                    str(profile_signature or ""),
                    str(upload_signature or ""),
                    int(candidate_count or 0),
                    json.dumps(recommendations or [], ensure_ascii=False),
                    json.dumps(warnings or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    float(execution_seconds or 0.0),
                ),
            )
            conn.commit()

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
