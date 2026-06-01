import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.ingredient_parse_service import canonicalize_ingredient_for_matching
from api.upload_recommendation_service import (
    UploadRecommendationService,
    normalize_cache_exact_key,
    normalize_lookup_key,
)


def _service_with_cache(tmp_path, rows, category_names):
    sqlite_path = tmp_path / "runtime.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE ingredient_match_cache (
                cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_ingredient TEXT,
                normalized_raw TEXT,
                matched_standard_name TEXT,
                relation_type TEXT,
                confidence REAL,
                match_method TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO ingredient_match_cache (
                raw_ingredient,
                normalized_raw,
                matched_standard_name,
                relation_type,
                confidence,
                match_method
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    service = UploadRecommendationService.__new__(UploadRecommendationService)
    service.recommendation_service = SimpleNamespace(runtime={"sqlite_path": sqlite_path})
    service._ensure_loaded = lambda: None
    category_rows = {
        name: {
            "functional_ingredient_name": name,
            "category_main": "nutrition",
            "category_sub": "",
            "function_text": "",
            "claim_text": "",
        }
        for name in category_names
    }
    service._category_map = {normalize_lookup_key(name): row for name, row in category_rows.items()}
    service._category_loose_map = {
        normalize_cache_exact_key(name): row
        for name, row in category_rows.items()
        if normalize_cache_exact_key(name)
    }
    return service


def test_arginine_alias_uses_standard_cache_before_stale_raw_unrelated_cache(tmp_path):
    standard = "L-\uc544\ub974\uae30\ub2cc"
    typo = "L-\uc544\ub974\uc9c0\ub2cc"
    service = _service_with_cache(
        tmp_path,
        rows=[
            (standard, normalize_cache_exact_key(standard), standard, "same_ingredient", 0.99, "trusted"),
            (typo, normalize_cache_exact_key(typo), "", "unrelated", 0.0, "old_cache"),
        ],
        category_names=[standard],
    )

    normalized = canonicalize_ingredient_for_matching(typo)
    match = service._find_match_in_cache(normalized, raw_ingredient=typo, display_name=typo)

    assert normalized == standard
    assert match
    assert match["functional_ingredient_name"] == standard
    assert match["relation_type"] == "same_ingredient"
    assert match["match_source"] == "cache_normalized_exact"


def test_non_functional_cache_relation_is_not_reported_as_match(tmp_path):
    raw = "\ubc8c\uafc0"
    service = _service_with_cache(
        tmp_path,
        rows=[
            (raw, normalize_cache_exact_key(raw), raw, "unrelated", 0.0, "old_cache"),
        ],
        category_names=[raw],
    )

    match = service._find_match_in_cache(raw, raw_ingredient=raw, display_name=raw)

    assert match is None
