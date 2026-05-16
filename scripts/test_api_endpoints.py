from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app  # noqa: E402


def main() -> None:
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "api_test_summary.json"
    results_path = output_dir / "api_test_results.csv"

    client = TestClient(app)
    cases = [
        ("health", "GET", "/health", None),
        ("search_joint", "GET", "/api/products/search?q=관절연골케어엔NEM", None),
        ("profile_joint", "GET", "/api/products/2018001205671/profile", None),
        ("similar_joint_first", "GET", "/api/products/2018001205671/similar?top_k=10&candidate_limit=1000&force_refresh=true", None),
        ("search_ps", "GET", "/api/products/search?q=인지력 건강엔 포스파티딜세린", None),
        ("similar_ps", "GET", "/api/products/200400200082875/similar?top_k=10&candidate_limit=1000", None),
        ("similar_joint_cache_hit", "GET", "/api/products/2018001205671/similar?top_k=10&candidate_limit=1000", None),
        (
            "recommend_by_ingredients",
            "POST",
            "/api/recommend/by-ingredients",
            {"raw_ingredients": "루테인, 지아잔틴, 비타민A, 아연", "top_k": 10, "candidate_limit": 1000},
        ),
    ]

    rows = []
    for name, method, path, payload in cases:
        start = time.perf_counter()
        if method == "GET":
            response = client.get(path)
        else:
            response = client.post(path, json=payload)
        elapsed = round(time.perf_counter() - start, 6)
        body = response.json()
        rows.append(
            {
                "test_name": name,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "result_count": len(body.get("results", body.get("recommendations", []))) if isinstance(body, dict) else 0,
                "cache_used": body.get("cache_used") if isinstance(body, dict) else None,
                "report_no": body.get("base_product", {}).get("report_no", "") if isinstance(body, dict) else "",
            }
        )

    pd.DataFrame(rows).to_csv(results_path, index=False, encoding="utf-8-sig")

    search_ps = client.get("/api/products/search?q=인지력 건강엔 포스파티딜세린").json()
    first_similar = client.get("/api/products/2018001205671/similar?top_k=10&candidate_limit=1000&force_refresh=true").json()
    second_similar = client.get("/api/products/2018001205671/similar?top_k=10&candidate_limit=1000").json()

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tests": rows,
        "duplicate_name_search_count": int(search_ps.get("count", 0)),
        "duplicate_name_report_nos": [item.get("report_no", "") for item in search_ps.get("results", [])],
        "joint_similar_first_cache_used": bool(first_similar.get("cache_used")),
        "joint_similar_second_cache_used": bool(second_similar.get("cache_used")),
        "joint_similar_first_execution_seconds": float(first_similar.get("execution_seconds", 0.0)),
        "joint_similar_second_execution_seconds": float(second_similar.get("execution_seconds", 0.0)),
        "joint_top_reason": first_similar.get("recommendations", [{}])[0].get("reason", ""),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
