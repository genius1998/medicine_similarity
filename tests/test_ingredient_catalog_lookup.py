import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.ops_service import OperationsService, RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME  # noqa: E402
from api.upload_recommendation_service import UploadRecommendationService  # noqa: E402


def seed_ingredient_db(sqlite_path: Path) -> None:
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE functional_category_map (
                functional_ingredient_name TEXT PRIMARY KEY,
                category_main TEXT,
                category_sub TEXT,
                function_text TEXT,
                claim_text TEXT,
                source TEXT,
                confidence REAL,
                notes TEXT,
                categories_json TEXT DEFAULT '[]'
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE {RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME} (
                functional_ingredient_name TEXT PRIMARY KEY,
                category_main TEXT,
                category_sub TEXT,
                function_text TEXT,
                claim_text TEXT,
                source TEXT,
                confidence REAL,
                notes TEXT,
                categories_json TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
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
            INSERT INTO functional_category_map (
                functional_ingredient_name, category_main, category_sub, function_text,
                claim_text, source, confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("밀크씨슬추출물", "간 건강", "간 기능", "간 건강에 도움", "간 건강", "seed", 0.99, ""),
                ("홍삼", "면역", "피로", "면역 및 피로 개선", "면역력 증진", "seed", 0.98, ""),
                ("프로바이오틱스", "장 건강", "장내 균총", "장 건강에 도움", "장 건강", "seed", 0.97, ""),
            ],
        )
        conn.execute(
            f"""
            INSERT INTO {RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME} (
                functional_ingredient_name, category_main, category_sub, function_text,
                claim_text, source, confidence, notes
            ) VALUES ('신규원료', '기타', '', '검토 중', '', 'runtime_auto_provisional', 0.72, '')
            """
        )
        conn.execute(
            """
            INSERT INTO ingredient_match_cache (
                raw_ingredient, normalized_raw, matched_standard_name, relation_type, confidence, match_method
            ) VALUES ('밀크씨슬', '밀크씨슬', '밀크씨슬추출물', 'same_ingredient', 0.94, 'seed')
            """
        )


def make_ops_service(sqlite_path: Path) -> OperationsService:
    service = OperationsService.__new__(OperationsService)
    service.sqlite_path = sqlite_path
    service.sync_pending_ingredients_from_runtime = lambda: None
    return service


class FakeRecommendationService:
    def __init__(self, sqlite_path: Path) -> None:
        self.runtime = {"sqlite_path": sqlite_path}
        self.ingredient_category_profiles = {}

    def ensure_loaded(self) -> None:
        return None


def test_list_ingredient_catalog_page_uses_count_limit_offset(tmp_path):
    sqlite_path = tmp_path / "ingredients.sqlite"
    seed_ingredient_db(sqlite_path)
    service = make_ops_service(sqlite_path)

    first_page = service.list_ingredient_catalog_page(page=1, page_size=2)
    second_page = service.list_ingredient_catalog_page(page=2, page_size=2)
    beyond_page = service.list_ingredient_catalog_page(page=9, page_size=2)

    assert first_page["total_count"] == 4
    assert first_page["total_pages"] == 2
    assert len(first_page["results"]) == 2
    assert first_page["results"][0]["functional_ingredient_name"] == "신규원료"
    assert first_page["results"][0]["origin_type"] == "new"
    assert [row["functional_ingredient_name"] for row in second_page["results"]] == ["프로바이오틱스", "홍삼"]
    assert beyond_page["page"] == 2
    assert [row["functional_ingredient_name"] for row in beyond_page["results"]] == ["프로바이오틱스", "홍삼"]


def test_list_ingredient_catalog_page_filters_query_and_origin(tmp_path):
    sqlite_path = tmp_path / "ingredients.sqlite"
    seed_ingredient_db(sqlite_path)
    service = make_ops_service(sqlite_path)

    result = service.list_ingredient_catalog_page(q="간", origin="existing", page=1, page_size=20)

    assert result["total_count"] == 1
    assert result["results"][0]["functional_ingredient_name"] == "밀크씨슬추출물"
    assert result["results"][0]["category_main"] == "간 건강"


def test_lookup_existing_ingredient_db_match_returns_cache_match(tmp_path):
    sqlite_path = tmp_path / "ingredients.sqlite"
    seed_ingredient_db(sqlite_path)
    service = UploadRecommendationService(FakeRecommendationService(sqlite_path))

    result = service.lookup_existing_ingredient_db_match("밀크씨슬")

    assert result["matched"] is True
    assert result["match"]["functional_ingredient_name"] == "밀크씨슬추출물"
    assert result["match"]["category_main"] == "간 건강"
    assert result["match"]["match_source"] == "cache_normalized_exact"


def test_lookup_existing_ingredient_db_match_uses_runtime_fallback(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "ingredients.sqlite"
    seed_ingredient_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            INSERT INTO functional_category_map (
                functional_ingredient_name, category_main, category_sub, function_text,
                claim_text, source, confidence, notes
            ) VALUES ('fallback-standard', 'fallback-main', 'fallback-sub', 'fallback function', 'fallback claim', 'test', 0.91, '')
            """
        )
    service = UploadRecommendationService(FakeRecommendationService(sqlite_path))

    captured = {}

    def fake_runtime_fallback(**kwargs):
        captured.update(kwargs)
        category_row = service._lookup_category_row("fallback-standard")
        return service._build_match_payload(
            ingredient_name=kwargs["ingredient_name"],
            raw_ingredient=kwargs["raw_ingredient"],
            display_name=kwargs["display_name"],
            functional_ingredient_name="fallback-standard",
            relation_type="same_ingredient",
            confidence=0.87,
            category_row=category_row,
            match_source="runtime_rag_llm_existing",
            protected_family=kwargs["input_family"],
        )

    monkeypatch.setattr(service, "_try_runtime_rag_fallback", fake_runtime_fallback)

    result = service.lookup_existing_ingredient_db_match("unseen-proprietary-extract")

    assert captured["raw_ingredient"] == "unseen-proprietary-extract"
    assert captured["allow_new_standard"] is True
    assert result["matched"] is True
    assert result["match"]["functional_ingredient_name"] == "fallback-standard"
    assert result["match"]["category_main"] == "fallback-main"
    assert result["match"]["match_source"] == "runtime_rag_llm_existing"
