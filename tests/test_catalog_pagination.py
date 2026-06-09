import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.recommendation_service import (  # noqa: E402
    CATALOG_TABLE_NAME,
    UPLOADED_PRODUCT_TABLE_NAME,
    RecommendationService,
)


CATALOG_COLUMNS = """
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
    raw_ingredients TEXT
"""


def make_service(sqlite_path: Path) -> RecommendationService:
    service = RecommendationService.__new__(RecommendationService)
    service._loaded = True
    service.runtime = {"sqlite_path": sqlite_path}
    service.profile_by_report_no = {
        "C3": {
            "product_main_category": "면역",
            "primary_ingredients": ["홍삼"],
            "secondary_ingredients": [],
            "support_ingredients": [],
            "llm_sub_function_categories": ["피로개선"],
            "confidence": 0.91,
            "notes": "profile note",
        }
    }
    return service


def seed_catalog_db(sqlite_path: Path) -> None:
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(f"CREATE TABLE {CATALOG_TABLE_NAME} ({CATALOG_COLUMNS})")
        conn.execute(
            f"""
            CREATE TABLE {UPLOADED_PRODUCT_TABLE_NAME} (
                report_no TEXT PRIMARY KEY,
                product_name TEXT,
                source_type TEXT,
                product_main_category TEXT,
                raw_ingredients TEXT
            )
            """
        )
        for report_no, product_name in [
            ("C1", "A One"),
            ("C2", "B Two"),
            ("C3", "C Three"),
            ("C4", "D Four"),
            ("C5", "E Five"),
        ]:
            conn.execute(
                f"""
                INSERT INTO {CATALOG_TABLE_NAME} (
                    report_no, product_name, company_name, license_no, report_date,
                    product_type, shelf_life, appearance, intake_method, main_functionality,
                    cautions, storage_method, form_factor, standard_spec, raw_ingredients
                ) VALUES (?, ?, 'Acme', '', '', '', '', '', '', '', '', '', '', '', '')
                """,
                (report_no, product_name),
            )
        conn.execute(
            f"""
            INSERT INTO {UPLOADED_PRODUCT_TABLE_NAME} (
                report_no, product_name, source_type, product_main_category, raw_ingredients
            ) VALUES ('U1', 'F Uploaded', 'ocr', '영양보충', '비타민 C')
            """
        )
        conn.execute(
            f"""
            INSERT INTO {UPLOADED_PRODUCT_TABLE_NAME} (
                report_no, product_name, source_type, product_main_category, raw_ingredients
            ) VALUES ('U2', '원재료', 'ocr', '기타', '')
            """
        )


def test_list_catalog_products_uses_db_limit_offset_and_count(tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    seed_catalog_db(sqlite_path)
    service = make_service(sqlite_path)

    result = service.list_catalog_products("", page=2, page_size=2)

    assert result["total_count"] == 6
    assert [item["report_no"] for item in result["results"]] == ["C3", "C4"]
    assert result["results"][0]["product_main_category"] == "면역"
    assert result["results"][0]["primary_ingredients"] == ["홍삼"]


def test_list_catalog_products_filters_in_sql_and_supports_collapsed_name(tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    seed_catalog_db(sqlite_path)
    service = make_service(sqlite_path)

    result = service.list_catalog_products("AOne", page=1, page_size=20)

    assert result["total_count"] == 1
    assert result["results"][0]["report_no"] == "C1"


def test_get_catalog_product_detail_uses_report_no_lookup(tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    seed_catalog_db(sqlite_path)
    service = make_service(sqlite_path)

    result = service.get_catalog_product_detail("U1")

    assert result is not None
    assert result["report_no"] == "U1"
    assert result["product_name"] == "F Uploaded"
    assert result["main_functionality"] == "영양보충"
