from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app  # noqa: E402


SAMPLE_OCR_TEXT = """제품명: 루테인 지아잔틴 플러스
원재료명: 마리골드꽃추출물, 비타민A, 산화아연, 대두유, 밀납, 대두레시틴
섭취량: 1일 1회, 1회 1캡슐"""


def main() -> None:
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ocr_api_test_summary.json"
    results_path = output_dir / "ocr_api_test_results.jsonl"

    client = TestClient(app)
    cases = [
        ("health", "GET", "/health", None),
        (
            "recommend_by_ocr_text",
            "POST",
            "/api/recommend/by-ocr-text",
            {"ocr_text": SAMPLE_OCR_TEXT, "top_k": 10, "candidate_limit": 1000},
        ),
        (
            "recommend_by_ingredients",
            "POST",
            "/api/recommend/by-ingredients",
            {"ingredients": ["루테인", "지아잔틴", "비타민 A", "아연"], "top_k": 10, "candidate_limit": 1000},
        ),
    ]

    rows = []
    for name, method, path, payload in cases:
        started = time.perf_counter()
        response = client.get(path) if method == "GET" else client.post(path, json=payload)
        elapsed = round(time.perf_counter() - started, 6)
        body = response.json()
        rows.append(
            {
                "test_name": name,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "body": body,
            }
        )

    by_image_route_exists = any(getattr(route, "path", "") == "/api/recommend/by-image" for route in app.routes)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tests": [
            {
                "test_name": row["test_name"],
                "status_code": row["status_code"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for row in rows
        ],
        "by_image_route_exists": by_image_route_exists,
        "ocr_text_test": {
            "normalized_ingredients": rows[1]["body"].get("parsed", {}).get("normalized_ingredients", []),
            "estimated_main_category": rows[1]["body"].get("estimated_profile", {}).get("product_main_category"),
            "recommendation_count": len(rows[1]["body"].get("recommendations", [])),
            "needs_user_review": rows[1]["body"].get("needs_user_review"),
            "has_caution": bool(rows[1]["body"].get("recommendations", [{}])[0].get("caution", "")) if rows[1]["body"].get("recommendations") else False,
        },
        "ingredients_test": {
            "estimated_main_category": rows[2]["body"].get("estimated_profile", {}).get("product_main_category"),
            "recommendation_count": len(rows[2]["body"].get("recommendations", [])),
            "top_reason": rows[2]["body"].get("recommendations", [{}])[0].get("reason", ""),
        },
    }

    with results_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
