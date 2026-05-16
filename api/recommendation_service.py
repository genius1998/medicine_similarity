from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from api.db import sqlite_connection
from api.product_search_service import search_profile_records
from scripts.enhance_similarity_with_explanation import (
    compute_topk_for_single_product,
    ensure_cache_table,
    load_cached_rows,
    load_product_function_profiles,
    load_vector_inputs,
    refresh_cache_rows,
    resolve_runtime_paths,
    safe_json_loads,
    build_vector_indexes,
)


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
        self.product_vectors: Dict[str, Dict[str, float]] = {}
        self.product_names: Dict[str, str] = {}
        self.report_to_product_ids: Dict[str, List[str]] = {}
        self.ingredient_postings: Dict[str, List[str]] = {}
        self.ingredient_frequency: Dict[str, int] = {}
        self.catalog_records: List[dict] = []
        self.catalog_by_report_no: Dict[str, dict] = {}

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
            self.profile_records.append(
                {
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
            )

        self._load_catalog_records()
        self._loaded = True

    def _resolve_catalog_columns(self, catalog_df: pd.DataFrame) -> Dict[str, Optional[str]]:
        resolved = {}
        for key, aliases in CATALOG_COLUMN_ALIASES.items():
            resolved[key] = next((name for name in aliases if name in catalog_df.columns), None)
        return resolved

    def _load_catalog_records(self) -> None:
        catalog_candidates = sorted(self.runtime["output_dir"].parent.glob("*C003.csv"))
        if not catalog_candidates:
            catalog_candidates = sorted(Path(__file__).resolve().parents[1].glob("*C003.csv"))
        if not catalog_candidates:
            self.catalog_records = []
            self.catalog_by_report_no = {}
            return

        catalog_df = pd.read_csv(catalog_candidates[0], encoding="utf-8-sig", low_memory=False).fillna("")
        resolved_columns = self._resolve_catalog_columns(catalog_df)
        report_no_column = resolved_columns.get("report_no")
        if not report_no_column:
            self.catalog_records = []
            self.catalog_by_report_no = {}
            return

        catalog_df[report_no_column] = catalog_df[report_no_column].astype(str).str.strip()
        for key in ("product_name", "company_name"):
            column_name = resolved_columns.get(key)
            if column_name:
                catalog_df[column_name] = catalog_df[column_name].astype(str).str.strip()

        catalog_df = catalog_df[catalog_df[report_no_column] != ""].copy()
        catalog_df = catalog_df.drop_duplicates(subset=[report_no_column], keep="first")

        profile_map = {str(item.get("report_no", "")): item for item in self.profile_records}
        records = []
        for row in catalog_df.to_dict(orient="records"):
            report_no = str(row.get(report_no_column, "") or "").strip()
            profile = profile_map.get(report_no, {})

            def value_for(key: str) -> str:
                column_name = resolved_columns.get(key)
                return str(row.get(column_name, "") or "").strip() if column_name else ""

            records.append(
                {
                    "report_no": report_no,
                    "product_name": value_for("product_name"),
                    "company_name": value_for("company_name"),
                    "license_no": value_for("license_no"),
                    "report_date": value_for("report_date"),
                    "product_type": value_for("product_type"),
                    "shelf_life": value_for("shelf_life"),
                    "appearance": value_for("appearance"),
                    "intake_method": value_for("intake_method"),
                    "main_functionality": value_for("main_functionality"),
                    "cautions": value_for("cautions"),
                    "storage_method": value_for("storage_method"),
                    "form_factor": value_for("form_factor"),
                    "standard_spec": value_for("standard_spec"),
                    "raw_ingredients": value_for("raw_ingredients"),
                    "product_main_category": str(profile.get("product_main_category", "") or ""),
                    "primary_ingredients": list(profile.get("primary_ingredients", [])),
                    "secondary_ingredients": list(profile.get("secondary_ingredients", [])),
                    "support_ingredients": list(profile.get("support_ingredients", [])),
                    "product_sub_categories": list(profile.get("product_sub_categories", [])),
                    "confidence": float(profile.get("confidence", 0.0) or 0.0),
                    "notes": str(profile.get("notes", "") or ""),
                }
            )

        self.catalog_records = records
        self.catalog_by_report_no = {record["report_no"]: record for record in records}

    def health(self) -> dict:
        self.ensure_loaded()
        return {
            "loaded": self._loaded,
            "profile_count": len(self.profile_records),
            "vector_product_count": len(self.product_vectors),
            "catalog_count": len(self.catalog_records),
        }

    def search_products(self, query: str, limit: int) -> List[dict]:
        self.ensure_loaded()
        return search_profile_records(self.profile_records, query, limit)

    def get_profiles_by_name(self, product_name: str) -> List[dict]:
        self.ensure_loaded()
        return [record for record in self.profile_records if str(record.get("product_name", "")) == product_name]

    def get_profile_by_report_no(self, report_no: str) -> Optional[dict]:
        self.ensure_loaded()
        for record in self.profile_records:
            if str(record.get("report_no", "")) == str(report_no):
                return record
        return None

    def list_catalog_products(self, query: str, page: int, page_size: int) -> dict:
        self.ensure_loaded()
        rows = self.catalog_records
        if query:
            query_text = str(query or "").strip().lower()
            normalized = query_text.replace(" ", "")
            rows = [
                item
                for item in rows
                if query_text in str(item.get("product_name", "")).lower()
                or query_text in str(item.get("company_name", "")).lower()
                or normalized in str(item.get("product_name", "")).lower().replace(" ", "")
                or str(item.get("report_no", "")) == query_text
            ]
        total_count = len(rows)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        return {
            "query": query,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "results": rows[start:end],
        }

    def get_catalog_product_detail(self, report_no: str) -> Optional[dict]:
        self.ensure_loaded()
        record = self.catalog_by_report_no.get(str(report_no))
        if not record:
            return None
        result = dict(record)
        result["has_profile"] = bool(self.get_profile_by_report_no(report_no))
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
                return {
                    "base_product": {
                        "report_no": str(base_profile.get("report_no", "") or ""),
                        "product_name": str(base_profile.get("product_name", "") or ""),
                        "product_main_category": str(base_profile.get("product_main_category", "") or ""),
                        "primary_ingredients": list(base_profile.get("primary_ingredients", [])),
                    },
                    "recommendations": [self._convert_cache_row(row, idx) for idx, row in enumerate(cached_rows, start=1)],
                    "cache_used": True,
                    "execution_seconds": round(time.perf_counter() - start_time, 6),
                }

            rows, _failed_rows, _stats = compute_topk_for_single_product(
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
            refresh_cache_rows(conn, base_product_id, rows)

        return {
            "base_product": {
                "report_no": str(base_profile.get("report_no", "") or ""),
                "product_name": str(base_profile.get("product_name", "") or ""),
                "product_main_category": str(base_profile.get("product_main_category", "") or ""),
                "primary_ingredients": list(base_profile.get("primary_ingredients", [])),
            },
            "recommendations": [self._convert_cache_row(row, idx) for idx, row in enumerate(rows, start=1)],
            "cache_used": False,
            "execution_seconds": round(time.perf_counter() - start_time, 6),
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
