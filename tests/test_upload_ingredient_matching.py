import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import api.upload_recommendation_service as upload_module
from api.ingredient_parse_service import canonicalize_ingredient_for_matching
from api.upload_recommendation_service import (
    UploadRecommendationService,
    RUNTIME_RAG_NAME_SIMILARITY_MIN,
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
    service._runtime_rag_documents = None
    service._runtime_rag_exact_map = None
    service._embedding_index = None
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
    service._get_embedding_index = lambda: None
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
    raw = "\ud788\ub4dc\ub85d\uc2dc\ud504\ub85c\ud544\uba54\ud2f8\uc140\ub8f0\ub85c\uc624\uc2a4"
    service = _service_with_cache(
        tmp_path,
        rows=[
            (raw, normalize_cache_exact_key(raw), raw, "excipient", 0.0, "old_cache"),
        ],
        category_names=[],
    )
    service._try_runtime_rag_fallback = lambda **kwargs: (_ for _ in ()).throw(AssertionError("RAG should not run for excipient cache"))

    match = service._find_match_in_cache(raw, raw_ingredient=raw, display_name=raw)

    assert match is None


def test_runtime_rag_name_similarity_promotes_korean_typo_candidate(tmp_path, monkeypatch):
    standard = "L-\uc544\ub974\uae30\ub2cc"
    typo = "L-\uc544\ub974\uc9c0\ub2cc"
    service = _service_with_cache(tmp_path, rows=[], category_names=[standard])
    service._load_runtime_rag_documents = lambda: [
        {"standard_name": "L-arabinose", "search_text": "", "synonyms_joined": "", "specific_penalty": 0.0},
        {"standard_name": standard, "search_text": "", "synonyms_joined": "", "specific_penalty": 0.0},
    ]
    monkeypatch.setattr(upload_module, "canonicalize_ingredient_for_matching", lambda value: str(value or "").strip())

    candidates = service._search_runtime_rag_candidates(typo, top_k=5)
    arginine = next(row for row in candidates if row["standard_name"] == standard)

    assert arginine["name_similarity_score"] >= RUNTIME_RAG_NAME_SIMILARITY_MIN
    assert arginine["retrieval_score"] >= 2.0


def test_unrelated_cache_is_revalidated_by_runtime_rag_existing_match_without_alias(tmp_path, monkeypatch):
    standard = "L-\uc544\ub974\uae30\ub2cc"
    typo = "L-\uc544\ub974\uc9c0\ub2cc"
    service = _service_with_cache(
        tmp_path,
        rows=[
            (typo, normalize_cache_exact_key(typo), "", "unrelated", 0.0, "old_cache"),
            (
                f"{typo}\ud63c\ub3d9\ud14d\uc2a4\ud2b8",
                normalize_cache_exact_key(f"{typo}\ud63c\ub3d9\ud14d\uc2a4\ud2b8"),
                "L-\uae00\ub8e8\ud0c0\ubbfc",
                "same_ingredient",
                0.99,
                "bad_like_cache",
            ),
        ],
        category_names=[standard, "L-\uae00\ub8e8\ud0c0\ubbfc"],
    )
    service._load_runtime_rag_documents = lambda: [
        {"standard_name": standard, "search_text": "", "synonyms_joined": "", "specific_penalty": 0.0},
    ]
    saved = {}
    service._save_runtime_cache_match = lambda **kwargs: saved.update(kwargs)

    def fake_classify(*, ingredient_name, raw_ingredient, display_name, candidates):
        assert [item["standard_name"] for item in candidates] == [standard]
        return {
            "decision": "existing_match",
            "matched_standard_name": standard,
            "relation_type": "same_ingredient",
            "confidence": 0.95,
            "reason": "typo revalidated by RAG",
        }

    service._classify_runtime_rag_candidates_with_llm = fake_classify
    monkeypatch.setattr(upload_module, "canonicalize_ingredient_for_matching", lambda value: str(value or "").strip())

    match = service._find_match_in_cache(typo, raw_ingredient=typo, display_name=typo)

    assert match
    assert match["functional_ingredient_name"] == standard
    assert match["match_source"] == "runtime_rag_llm_existing"
    assert saved["match_method"] == "runtime_rag_llm_existing"


def test_unrelated_cache_revalidation_does_not_create_new_standard(tmp_path, monkeypatch):
    raw = "\uc0ac\uc591\ubc8c\uafc0"
    service = _service_with_cache(
        tmp_path,
        rows=[
            (raw, normalize_cache_exact_key(raw), "", "unrelated", 0.0, "old_cache"),
        ],
        category_names=[],
    )
    service._load_runtime_rag_documents = lambda: []
    service._classify_runtime_rag_candidates_with_llm = lambda **kwargs: {
        "decision": "new_standard",
        "proposed_standard_name": "Honey",
        "relation_type": "same_ingredient",
        "confidence": 0.9,
    }
    service._save_runtime_cache_match = lambda **kwargs: (_ for _ in ()).throw(AssertionError("negative revalidation must not create cache"))
    service._save_runtime_custom_functional_ingredient = lambda **kwargs: (_ for _ in ()).throw(AssertionError("negative revalidation must not create standard"))
    monkeypatch.setattr(upload_module, "canonicalize_ingredient_for_matching", lambda value: str(value or "").strip())

    match = service._find_match_in_cache(raw, raw_ingredient=raw, display_name=raw)

    assert match is None
