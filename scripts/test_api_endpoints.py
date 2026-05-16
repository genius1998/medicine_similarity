from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = Path(r"D:\health_project\input_images\product_001.png")
OCR_WARNING_CODES = {
    "ocr_confidence_unavailable",
    "low_ocr_confidence",
    "low_ocr_text_quality",
    "ingredient_section_unclear",
    "ocr_parse_confidence_low",
    "missing_ocr_text",
}


def truncate_text(value: str, limit: int = 500) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def pick_fields(payload: Any, keys: List[str]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: payload.get(key) for key in keys if key in payload}


def summarize_response(test_name: str, body: Any) -> Any:
    if not isinstance(body, dict):
        return body

    if test_name == "health":
        return pick_fields(body, ["status", "service"])

    if test_name in {"search_joint", "search_duplicate_name"}:
        results = body.get("results", [])[:3]
        return {
            "query": body.get("query"),
            "count": body.get("count"),
            "results": [
                pick_fields(item, ["report_no", "product_name", "product_main_category", "primary_ingredients"])
                for item in results
            ],
        }

    if test_name == "profile_joint":
        return pick_fields(
            body,
            [
                "report_no",
                "product_name",
                "product_main_category",
                "primary_ingredients",
                "secondary_ingredients",
                "support_ingredients",
            ],
        )

    if test_name.startswith("similar_joint"):
        recs = body.get("recommendations", [])[:2]
        return {
            "base_product": pick_fields(body.get("base_product", {}), ["report_no", "product_name", "product_main_category", "primary_ingredients"]),
            "cache_used": body.get("cache_used"),
            "execution_seconds": body.get("execution_seconds"),
            "recommendation_count": len(body.get("recommendations", [])),
            "recommendations": [
                pick_fields(
                    item,
                    [
                        "rank",
                        "target_report_no",
                        "target_product_name",
                        "similarity_score",
                        "substitutability",
                        "shared_ingredients",
                        "shared_categories",
                        "reason",
                        "caution",
                    ],
                )
                for item in recs
            ],
        }

    if test_name in {"ocr_text_recommendation", "ingredients_structured_eye", "ingredients_raw_hongsam", "image_upload_product_001"}:
        ocr_payload = body.get("ocr", {}) if isinstance(body.get("ocr"), dict) else {}
        parsed = body.get("parsed", {}) if isinstance(body.get("parsed"), dict) else {}
        estimated = body.get("estimated_profile", {}) if isinstance(body.get("estimated_profile"), dict) else {}
        recs = body.get("recommendations", [])[:2]
        sample = {
            "input_type": body.get("input_type"),
            "needs_user_review": body.get("needs_user_review"),
            "review_message": body.get("review_message"),
            "product_name_category_hint": body.get("product_name_category_hint"),
            "estimated_profile": pick_fields(estimated, ["product_main_category", "primary_ingredients", "secondary_ingredients", "support_ingredients"]),
            "parsed": pick_fields(
                parsed,
                [
                    "product_name_candidate",
                    "normalized_ingredients",
                    "primary_ingredients",
                    "primary_ingredients_normalized",
                    "excluded_ingredients",
                ],
            ),
            "critical_warnings": body.get("critical_warnings", []),
            "notices": body.get("notices", []),
            "info_warnings": body.get("info_warnings", []),
            "recommendation_count": len(body.get("recommendations", [])),
            "recommendations": [
                pick_fields(
                    item,
                    [
                        "rank",
                        "target_report_no",
                        "target_product_name",
                        "similarity_score",
                        "substitutability",
                        "reason",
                        "caution",
                    ],
                )
                for item in recs
            ],
        }
        if ocr_payload:
            sample["ocr"] = {
                "confidence": ocr_payload.get("confidence"),
                "confidence_source": ocr_payload.get("confidence_source"),
                "raw_text_preview": truncate_text(ocr_payload.get("raw_text", ""), 500),
            }
        return sample

    return pick_fields(body, list(body.keys())[:12])


def expect(condition: bool, key_assertions: List[Dict[str, Any]], description: str) -> None:
    key_assertions.append({"description": description, "passed": bool(condition)})


def run_json_request(session: requests.Session, method: str, url: str, payload: Optional[dict] = None) -> requests.Response:
    if method == "GET":
        return session.get(url, timeout=120)
    return session.post(url, json=payload, timeout=180)


def run_image_request(session: requests.Session, url: str) -> requests.Response:
    with IMAGE_PATH.open("rb") as fh:
        files = {"file": (IMAGE_PATH.name, fh, "image/png")}
        data = {"top_k": "10", "candidate_limit": "1000"}
        return session.post(url, files=files, data=data, timeout=300)


def execute_test(session: requests.Session, base_url: str, name: str, method: str, path: str, payload: Optional[dict] = None) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    start = time.perf_counter()
    error = ""
    status_code = 0
    body: Any = {}
    try:
        if name == "image_upload_product_001":
            response = run_image_request(session, url)
        else:
            response = run_json_request(session, method, url, payload)
        elapsed = round(time.perf_counter() - start, 6)
        status_code = response.status_code
        content_type = response.headers.get("content-type", "")
        body = response.json() if "application/json" in content_type else {"text": truncate_text(response.text, 500)}
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.perf_counter() - start, 6)
        error = f"{type(exc).__name__}: {exc}"
        body = {}

    assertions: List[Dict[str, Any]] = []
    passed = False

    if not error:
        expect(status_code == 200, assertions, "status_code == 200")

        if name == "health":
            expect(body.get("status") == "ok", assertions, "status == ok")

        elif name == "search_joint":
            results = body.get("results", [])
            expect(isinstance(results, list), assertions, "results 배열 존재")
            expect(any(item.get("report_no") for item in results), assertions, "report_no 포함")
            expect(any(item.get("product_name") for item in results), assertions, "product_name 포함")
            expect(any(item.get("product_main_category") for item in results), assertions, "product_main_category 포함")
            expect(any(item.get("primary_ingredients") for item in results), assertions, "primary_ingredients 포함")

        elif name == "search_duplicate_name":
            results = body.get("results", [])
            expect(isinstance(results, list), assertions, "results 배열 존재")
            expect(all(item.get("report_no") for item in results), assertions, "각 결과에 report_no 포함")
            expect(body.get("count", 0) >= 1, assertions, "검색 결과 1건 이상")
            expect(body.get("count", 0) == len(results), assertions, "count와 실제 결과 수 일치")

        elif name == "profile_joint":
            primary = body.get("primary_ingredients", [])
            primary_text = " ".join(str(item) for item in primary)
            expect(body.get("product_main_category") == "관절/연골", assertions, "product_main_category == 관절/연골")
            expect("난각막" in primary_text or "NEM" in primary_text, assertions, "primary에 난각막/NEM 계열 포함")
            expect("secondary_ingredients" in body, assertions, "secondary_ingredients 존재")
            expect("support_ingredients" in body, assertions, "support_ingredients 존재")

        elif name in {"similar_joint_first", "similar_joint_second"}:
            recs = body.get("recommendations", [])
            first = recs[0] if recs else {}
            expect("base_product" in body, assertions, "base_product 존재")
            expect(len(recs) == 10, assertions, "recommendations 10개")
            expect("similarity_score" in first, assertions, "similarity_score 존재")
            expect("substitutability" in first, assertions, "substitutability 존재")
            expect("reason" in first, assertions, "reason 존재")
            expect("caution" in first, assertions, "caution 존재")
            expect("explanation" in first, assertions, "explanation 존재")
            expect("cache_used" in body, assertions, "cache_used 필드 존재")

        elif name == "ocr_text_recommendation":
            parsed = body.get("parsed", {})
            estimated = body.get("estimated_profile", {})
            recs = body.get("recommendations", [])
            category = estimated.get("product_main_category", "")
            expect(bool(parsed), assertions, "parsed 결과 존재")
            expect(bool(parsed.get("normalized_ingredients")), assertions, "normalized_ingredients 존재")
            expect(bool(estimated), assertions, "estimated_profile 존재")
            expect(len(recs) == 10, assertions, "recommendations 10개")
            expect(category in {"눈 건강", "영양보충"}, assertions, "estimated_main_category가 눈 건강 또는 관련 카테고리")
            expect(bool(recs and recs[0].get("caution")), assertions, "caution 포함")
            expect("needs_user_review" in body, assertions, "needs_user_review 필드 포함")
            expect("critical_warnings" in body and "notices" in body and "info_warnings" in body, assertions, "warning severity 필드 존재")

        elif name == "ingredients_structured_eye":
            estimated = body.get("estimated_profile", {})
            recs = body.get("recommendations", [])
            top_reason = recs[0].get("reason", "") if recs else ""
            critical_codes = {item.get("code") for item in body.get("critical_warnings", [])}
            notice_codes = {item.get("code") for item in body.get("notices", [])}
            info_codes = {item.get("code") for item in body.get("info_warnings", [])}
            expect(estimated.get("product_main_category") in {"눈 건강", "영양보충"}, assertions, "estimated_main_category가 눈 건강 또는 관련 카테고리")
            expect(len(recs) == 10, assertions, "recommendations 10개")
            expect(body.get("needs_user_review") is False, assertions, "needs_user_review == false")
            expect(len(body.get("critical_warnings", [])) == 0, assertions, "critical_warnings empty")
            expect(not (critical_codes | notice_codes | info_codes) & OCR_WARNING_CODES, assertions, "OCR warning code 없음")
            expect(any(token in top_reason for token in ["루테인", "지아잔틴", "눈 건강"]), assertions, "reason에 루테인/지아잔틴/눈 건강 관련 표현 포함")

        elif name == "ingredients_raw_hongsam":
            estimated = body.get("estimated_profile", {})
            recs = body.get("recommendations", [])
            top_reason = recs[0].get("reason", "") if recs else ""
            critical_codes = {item.get("code") for item in body.get("critical_warnings", [])}
            notice_codes = {item.get("code") for item in body.get("notices", [])}
            info_codes = {item.get("code") for item in body.get("info_warnings", [])}
            expect(estimated.get("product_main_category") in {"면역", "영양보충"}, assertions, "estimated_main_category가 면역 또는 홍삼 계열")
            expect(len(recs) == 10, assertions, "recommendations 10개")
            expect(body.get("needs_user_review") is False, assertions, "needs_user_review == false")
            expect(len(body.get("critical_warnings", [])) == 0, assertions, "critical_warnings empty")
            expect(not (critical_codes | notice_codes | info_codes) & OCR_WARNING_CODES, assertions, "OCR warning code 없음")
            expect(any(token in top_reason for token in ["홍삼", "면역"]), assertions, "reason에 홍삼 또는 면역 관련 표현 포함")

        elif name == "image_upload_product_001":
            ocr = body.get("ocr", {})
            parsed = body.get("parsed", {})
            estimated = body.get("estimated_profile", {})
            recs = body.get("recommendations", [])
            category = estimated.get("product_main_category", "")
            expect(bool(ocr.get("raw_text")), assertions, "OCR raw_text 존재")
            expect(bool(parsed), assertions, "parsed 결과 존재")
            expect(category in {"면역", "영양보충"}, assertions, "estimated_main_category가 면역 또는 홍삼 계열")
            expect(len(recs) == 10, assertions, "recommendations 10개")
            expect(bool(recs and recs[0].get("caution")), assertions, "caution 포함")
            expect("needs_user_review" in body, assertions, "needs_user_review 필드 포함")
            expect("critical_warnings" in body and "notices" in body and "info_warnings" in body, assertions, "warning severity 분리 확인")

        passed = all(item["passed"] for item in assertions)

    result = {
        "test_name": name,
        "method": method,
        "url": url,
        "status_code": status_code,
        "passed": passed,
        "execution_seconds": elapsed,
        "error": error,
        "key_assertions": assertions,
        "response_sample": summarize_response(name, body),
    }

    if name == "similar_joint_second" and not error:
        result["cache_used"] = body.get("cache_used")
        result["execution_seconds_server"] = body.get("execution_seconds")
    elif name == "similar_joint_first" and not error:
        result["cache_used"] = body.get("cache_used")
        result["execution_seconds_server"] = body.get("execution_seconds")

    return result


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "api_endpoint_test_summary.json"
    results_path = OUTPUT_DIR / "api_endpoint_test_results.jsonl"

    cases = [
        ("health", "GET", "/health", None),
        ("search_joint", "GET", "/api/products/search?q=관절연골케어엔NEM&limit=20", None),
        ("search_duplicate_name", "GET", "/api/products/search?q=인지력 건강엔 포스파티딜세린&limit=20", None),
        ("profile_joint", "GET", "/api/products/2018001205671/profile", None),
        ("similar_joint_first", "GET", "/api/products/2018001205671/similar?top_k=10&candidate_limit=1000&force_refresh=true", None),
        ("similar_joint_second", "GET", "/api/products/2018001205671/similar?top_k=10&candidate_limit=1000&force_refresh=false", None),
        (
            "ocr_text_recommendation",
            "POST",
            "/api/recommend/by-ocr-text",
            {
                "ocr_text": "제품명: 루테인 지아잔틴 플러스\n원재료명: 마리골드꽃추출물, 비타민A, 산화아연, 대두유, 밀납, 대두레시틴\n섭취량: 1일 1회, 1회 1캡슐",
                "top_k": 10,
                "candidate_limit": 1000,
            },
        ),
        (
            "ingredients_structured_eye",
            "POST",
            "/api/recommend/by-ingredients",
            {
                "ingredients": ["루테인", "지아잔틴", "비타민 A", "아연"],
                "top_k": 10,
                "candidate_limit": 1000,
            },
        ),
        (
            "ingredients_raw_hongsam",
            "POST",
            "/api/recommend/by-ingredients",
            {
                "raw_ingredients": "홍삼농축액, 진세노사이드, 프락토올리고당",
                "top_k": 10,
                "candidate_limit": 1000,
            },
        ),
        ("image_upload_product_001", "POST", "/api/recommend/by-image", None),
    ]

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    results = [execute_test(session, base_url, *case) for case in cases]

    with results_path.open("w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    passed = sum(1 for item in results if item["passed"])
    failed = len(results) - passed

    cache_first = next((item for item in results if item["test_name"] == "similar_joint_first"), {})
    cache_second = next((item for item in results if item["test_name"] == "similar_joint_second"), {})

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "tests": [
            {
                "name": item["test_name"],
                "passed": item["passed"],
                "status_code": item["status_code"],
                "execution_seconds": item["execution_seconds"],
            }
            for item in results
        ],
        "cache_check": {
            "first_call_cache_used": cache_first.get("cache_used"),
            "second_call_cache_used": cache_second.get("cache_used"),
            "first_call_server_execution_seconds": cache_first.get("execution_seconds_server"),
            "second_call_server_execution_seconds": cache_second.get("execution_seconds_server"),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
