from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from api.db import sqlite_connection
from api.product_search_service import search_profile_records
from scripts.enhance_similarity_with_explanation import (
    TABLE_NAME,
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

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.runtime = resolve_runtime_paths()
        vector_df = load_vector_inputs(self.runtime["vector_csv_path"])
        self.profiles = load_product_function_profiles(self.runtime["product_profile_csv_path"])
        self.product_vectors, self.product_names, self.report_to_product_ids, self.ingredient_postings, self.ingredient_frequency = build_vector_indexes(vector_df)

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
        self._loaded = True

    def health(self) -> dict:
        self.ensure_loaded()
        return {
            "loaded": self._loaded,
            "profile_count": len(self.profile_records),
            "vector_product_count": len(self.product_vectors),
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

    def get_profile_by_product_id(self, product_id: str) -> Optional[dict]:
        self.ensure_loaded()
        profile = self.profiles.get(product_id)
        if not profile:
            return None
        return {
            "report_no": str(profile.get("report_no", "") or ""),
            "product_id": str(profile.get("product_id", "") or ""),
            "product_name": str(profile.get("product_name", "") or ""),
            "product_main_category": str(profile.get("product_main_category", "") or ""),
            "primary_ingredients": list(profile.get("primary_ingredients", [])),
            "secondary_ingredients": list(profile.get("secondary_ingredients", [])),
            "support_ingredients": list(profile.get("support_ingredients", [])),
            "product_sub_categories": list(profile.get("product_sub_categories", [])),
            "confidence": float(profile.get("confidence", 0.0) or 0.0),
            "notes": str(profile.get("notes", "") or ""),
        }

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
