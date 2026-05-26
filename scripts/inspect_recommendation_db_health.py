from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.db import default_sqlite_path
from api.ingredient_parse_service import LLM_PARSE_CACHE_TABLE_NAME
from api.recommendation_service import RECOMMENDATION_TRACE_TABLE_NAME, UPLOADED_PRODUCT_TABLE_NAME


OUTPUT_PATH = REPO_ROOT / "output" / "db_health_report.json"
EXPECTED_UPLOADED_COLUMNS = [
    "report_no",
    "product_id",
    "product_name",
    "upload_signature",
    "image_hash",
    "ocr_text_hash",
    "parsed_signature",
    "profile_signature",
    "status",
    "quality_grade",
    "is_candidate_enabled",
]


def _fetchall_dict(conn: sqlite3.Connection, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    cur = conn.execute(query, params)
    columns = [item[0] for item in cur.description or []]
    return [{columns[idx]: row[idx] for idx in range(len(columns))} for row in cur.fetchall()]


def _fetch_scalar(conn: sqlite3.Connection, query: str, params: tuple = (), default: Any = 0) -> Any:
    row = conn.execute(query, params).fetchone()
    return default if row is None else row[0]


def main() -> None:
    sqlite_path = default_sqlite_path()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(sqlite_path) as conn:
        uploaded_columns = [row[1] for row in conn.execute(f"PRAGMA table_info({UPLOADED_PRODUCT_TABLE_NAME})").fetchall()]
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        uploaded_count = _fetch_scalar(conn, f"SELECT COUNT(*) FROM {UPLOADED_PRODUCT_TABLE_NAME}", default=0) if UPLOADED_PRODUCT_TABLE_NAME in table_names else 0
        status_counts = (
            _fetchall_dict(conn, f"SELECT COALESCE(NULLIF(status, ''), 'raw') AS status, COUNT(*) AS count FROM {UPLOADED_PRODUCT_TABLE_NAME} GROUP BY COALESCE(NULLIF(status, ''), 'raw') ORDER BY count DESC")
            if UPLOADED_PRODUCT_TABLE_NAME in table_names
            else []
        )
        quality_grade_counts = (
            _fetchall_dict(conn, f"SELECT COALESCE(NULLIF(quality_grade, ''), 'C') AS quality_grade, COUNT(*) AS count FROM {UPLOADED_PRODUCT_TABLE_NAME} GROUP BY COALESCE(NULLIF(quality_grade, ''), 'C') ORDER BY count DESC")
            if UPLOADED_PRODUCT_TABLE_NAME in table_names
            else []
        )
        candidate_enabled_true_count = (
            _fetch_scalar(conn, f"SELECT COUNT(*) FROM {UPLOADED_PRODUCT_TABLE_NAME} WHERE is_candidate_enabled = 1", default=0)
            if UPLOADED_PRODUCT_TABLE_NAME in table_names
            else 0
        )
        signature_missing_counts = (
            {
                "image_hash_null": _fetch_scalar(conn, f"SELECT COUNT(*) FROM {UPLOADED_PRODUCT_TABLE_NAME} WHERE COALESCE(NULLIF(image_hash, ''), '') = ''", default=0),
                "ocr_text_hash_null": _fetch_scalar(conn, f"SELECT COUNT(*) FROM {UPLOADED_PRODUCT_TABLE_NAME} WHERE COALESCE(NULLIF(ocr_text_hash, ''), '') = ''", default=0),
                "parsed_signature_null": _fetch_scalar(conn, f"SELECT COUNT(*) FROM {UPLOADED_PRODUCT_TABLE_NAME} WHERE COALESCE(NULLIF(parsed_signature, ''), '') = ''", default=0),
                "profile_signature_null": _fetch_scalar(conn, f"SELECT COUNT(*) FROM {UPLOADED_PRODUCT_TABLE_NAME} WHERE COALESCE(NULLIF(profile_signature, ''), '') = ''", default=0),
            }
            if UPLOADED_PRODUCT_TABLE_NAME in table_names
            else {}
        )
        risky_legacy_enabled_count = (
            _fetch_scalar(
                conn,
                f"""
                SELECT COUNT(*)
                FROM {UPLOADED_PRODUCT_TABLE_NAME}
                WHERE is_candidate_enabled = 1
                  AND (
                    COALESCE(NULLIF(parsed_signature, ''), '') = ''
                    OR COALESCE(NULLIF(profile_signature, ''), '') = ''
                  )
                """,
                default=0,
            )
            if UPLOADED_PRODUCT_TABLE_NAME in table_names
            else 0
        )
        recent_trace_logs = (
            _fetchall_dict(
                conn,
                f"""
                SELECT trace_id, input_type, product_name, image_hash, ocr_text_hash, parsed_signature, profile_signature, upload_signature, candidate_count, execution_seconds, created_at
                FROM {RECOMMENDATION_TRACE_TABLE_NAME}
                ORDER BY created_at DESC
                LIMIT 20
                """,
            )
            if RECOMMENDATION_TRACE_TABLE_NAME in table_names
            else []
        )
        parse_cache_count = _fetch_scalar(conn, f"SELECT COUNT(*) FROM {LLM_PARSE_CACHE_TABLE_NAME}", default=0) if LLM_PARSE_CACHE_TABLE_NAME in table_names else 0

    report = {
        "sqlite_path": str(sqlite_path),
        "tables": {
            "uploaded_product_master_exists": UPLOADED_PRODUCT_TABLE_NAME in table_names,
            "recommendation_trace_log_exists": RECOMMENDATION_TRACE_TABLE_NAME in table_names,
            "llm_parse_cache_exists": LLM_PARSE_CACHE_TABLE_NAME in table_names,
        },
        "uploaded_product_master_columns": {
            "present": uploaded_columns,
            "expected": EXPECTED_UPLOADED_COLUMNS,
            "missing": [column for column in EXPECTED_UPLOADED_COLUMNS if column not in uploaded_columns],
        },
        "counts": {
            "uploaded_product_master_total": uploaded_count,
            "candidate_enabled_true_count": candidate_enabled_true_count,
            "llm_parse_cache_total": parse_cache_count,
        },
        "status_counts": status_counts,
        "quality_grade_counts": quality_grade_counts,
        "signature_missing_counts": signature_missing_counts,
        "risky_legacy_enabled_count": risky_legacy_enabled_count,
        "recent_trace_logs": recent_trace_logs,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
