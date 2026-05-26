from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_text(value: Any, limit: int = 240) -> str:
    text = str(value or "")
    return text[:limit]


def extract_frontend_markers(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    analyze_button = soup.find(id="analyzeButton")
    rerun_button_text = "수정한 원료로 다시 추천받기" in html
    api_base_input = soup.find(id="apiBaseInput")
    title = soup.title.get_text(strip=True) if soup.title else ""
    flow_steps = [node.get_text(" ", strip=True) for node in soup.select(".flow-step strong")]

    return {
        "title": title,
        "has_file_input": soup.find(id="fileInput") is not None,
        "has_api_base_input": api_base_input is not None,
        "has_analyze_button": analyze_button is not None,
        "analyze_button_disabled_in_html": bool(analyze_button and analyze_button.has_attr("disabled")),
        "has_rerun_button_text": rerun_button_text,
        "has_upload_heading": "이미지 업로드" in html,
        "has_result_heading": "OCR 결과 확인" in html,
        "has_editor_heading": "원료 수정 후 재추천" in html,
        "has_recommendation_heading": "추천 결과 TOP10" in html,
        "flow_steps": flow_steps,
    }


def run_route_test(base_url: str, path: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    started = time.time()
    response = requests.get(url, timeout=20)
    elapsed = round(time.time() - started, 3)
    response.raise_for_status()
    markers = extract_frontend_markers(response.text)
    passed = (
        response.status_code == 200
        and markers["has_file_input"]
        and markers["has_api_base_input"]
        and markers["has_analyze_button"]
        and markers["has_rerun_button_text"]
    )
    return {
        "test_name": path,
        "method": "GET",
        "url": url,
        "status_code": response.status_code,
        "passed": passed,
        "execution_seconds": elapsed,
        "response_sample": {
            "title": markers["title"],
            "flow_steps": markers["flow_steps"],
        },
        "key_assertions": markers,
        "error": "",
    }


def run_image_api_flow(base_url: str, image_path: Path) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/recommend/by-image"
    started = time.time()
    with image_path.open("rb") as handle:
        response = requests.post(
            url,
            files={"file": (image_path.name, handle, "image/png")},
            data={"top_k": "10", "candidate_limit": "1000"},
            timeout=120,
        )
    elapsed = round(time.time() - started, 3)
    response.raise_for_status()
    payload = response.json()
    recommendations = payload.get("recommendations") or []
    parsed = payload.get("parsed") or {}
    estimated = payload.get("estimated_profile") or {}
    return {
        "test_name": "by-image-flow",
        "method": "POST",
        "url": url,
        "status_code": response.status_code,
        "passed": response.status_code == 200 and len(recommendations) == 10,
        "execution_seconds": elapsed,
        "response_sample": {
            "input_type": payload.get("input_type"),
            "product_name_candidate": parsed.get("product_name_candidate"),
            "category": estimated.get("product_main_category"),
            "primary_ingredients_normalized": sample_text(parsed.get("primary_ingredients_normalized")),
            "recommendation_count": len(recommendations),
            "needs_user_review": payload.get("needs_user_review"),
        },
        "key_assertions": {
            "recommendations_10": len(recommendations) == 10,
            "has_ocr_text": bool((payload.get("ocr") or {}).get("raw_text")),
            "has_parsed": bool(parsed),
            "has_estimated_profile": bool(estimated),
        },
        "error": "",
        "payload": payload,
    }


def run_ingredient_rerun(base_url: str, ingredients: List[str]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/recommend/by-ingredients"
    started = time.time()
    response = requests.post(
        url,
        json={"ingredients": ingredients, "top_k": 10, "candidate_limit": 1000},
        timeout=60,
    )
    elapsed = round(time.time() - started, 3)
    response.raise_for_status()
    payload = response.json()
    recommendations = payload.get("recommendations") or []
    estimated = payload.get("estimated_profile") or {}
    return {
        "test_name": "by-ingredients-rerun",
        "method": "POST",
        "url": url,
        "status_code": response.status_code,
        "passed": response.status_code == 200 and len(recommendations) == 10 and payload.get("needs_user_review") is False,
        "execution_seconds": elapsed,
        "response_sample": {
            "input_type": payload.get("input_type"),
            "category": estimated.get("product_main_category"),
            "recommendation_count": len(recommendations),
            "needs_user_review": payload.get("needs_user_review"),
            "top_reason": sample_text((recommendations[0] or {}).get("reason") if recommendations else ""),
        },
        "key_assertions": {
            "recommendations_10": len(recommendations) == 10,
            "critical_warnings_empty": not (payload.get("critical_warnings") or []),
            "ocr_info_absent": not any(
                (item.get("code") if isinstance(item, dict) else item) in {
                    "ocr_confidence_unavailable",
                    "low_ocr_confidence",
                    "low_ocr_text_quality",
                    "ingredient_section_unclear",
                    "ocr_parse_confidence_low",
                    "missing_ocr_text",
                }
                for item in (payload.get("info_warnings") or [])
            ),
        },
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image-path", default=r"D:\health_project\input_images\product_001.png")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    route_a = run_route_test(args.base_url, "/recommend/image")
    route_b = run_route_test(args.base_url, "/ocr-recommend")
    results.extend([route_a, route_b])

    image_result = run_image_api_flow(args.base_url, Path(args.image_path))
    results.append({k: v for k, v in image_result.items() if k != "payload"})

    rerun_source = image_result.get("payload", {}).get("parsed", {}).get("primary_ingredients_normalized") or ["홍삼", "진세노사이드"]
    rerun_result = run_ingredient_rerun(args.base_url, list(rerun_source) + ["진세노사이드"])
    results.append(rerun_result)

    route_consistency = {
        "test_name": "route-consistency",
        "method": "GET",
        "url": f"{args.base_url.rstrip('/')}/recommend/image vs /ocr-recommend",
        "status_code": 200,
        "passed": route_a["response_sample"]["title"] == route_b["response_sample"]["title"]
        and route_a["key_assertions"]["flow_steps"] == route_b["key_assertions"]["flow_steps"],
        "execution_seconds": 0,
        "response_sample": {
            "title_a": route_a["response_sample"]["title"],
            "title_b": route_b["response_sample"]["title"],
        },
        "key_assertions": {
            "same_title": route_a["response_sample"]["title"] == route_b["response_sample"]["title"],
            "same_steps": route_a["key_assertions"]["flow_steps"] == route_b["key_assertions"]["flow_steps"],
        },
        "error": "",
    }
    results.append(route_consistency)

    passed = sum(1 for item in results if item["passed"])
    failed = len(results) - passed
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_url": args.base_url,
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
    }

    write_json(OUTPUT_DIR / "frontend_mvp_flow_test_summary.json", summary)
    write_jsonl(OUTPUT_DIR / "frontend_mvp_flow_test_results.jsonl", results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
