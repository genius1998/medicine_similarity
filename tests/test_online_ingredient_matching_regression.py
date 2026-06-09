from __future__ import annotations

import json
import re
import sys
from typing import List

from api.ingredient_parse_service import canonicalize_ingredient_for_matching
from api.recommendation_service import RecommendationService
from api.upload_recommendation_service import UploadRecommendationService


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", str(value or "").strip().lower())


def contains_any(value: str, keywords: List[str]) -> bool:
    normalized = normalize(value)
    return any(normalize(keyword) in normalized for keyword in keywords if str(keyword or "").strip())


def build_display_payload(match: dict | None) -> dict | None:
    if not match:
        return None
    return {
        "functional_ingredient_name": str(match.get("functional_ingredient_name", "") or ""),
        "relation_type": str(match.get("relation_type", "") or ""),
        "confidence": float(match.get("confidence", 0.0) or 0.0),
        "match_source": str(match.get("match_source", "") or ""),
        "category_main": str((match.get("category_row") or {}).get("category_main", "") or ""),
    }


CASES = [
    {
        "raw": "HCA",
        "allow_none": False,
        "must_allow": ["가르시니아", "garcinia", "hca"],
        "must_block": ["녹차", "카테킨", "egcg", "green tea", "green tea extract"],
    },
    {
        "raw": "hydroxycitric acid",
        "allow_none": False,
        "must_allow": ["가르시니아", "garcinia", "hca"],
        "must_block": ["녹차", "카테킨", "egcg", "green tea", "green tea extract"],
    },
    {
        "raw": "실리마린",
        "allow_none": False,
        "must_allow": ["밀크씨슬", "밀크시슬", "milk thistle"],
        "must_block": ["녹차", "홍삼", "가르시니아"],
    },
    {
        "raw": "silymarin",
        "allow_none": False,
        "must_allow": ["밀크씨슬", "밀크시슬", "milk thistle"],
        "must_block": ["녹차", "홍삼", "가르시니아"],
    },
    {
        "raw": "진세노사이드",
        "allow_none": False,
        "must_allow": ["홍삼", "인삼", "ginseng"],
        "must_block": ["녹차", "밀크씨슬", "가르시니아"],
    },
    {
        "raw": "ginsenoside",
        "allow_none": False,
        "must_allow": ["홍삼", "인삼", "ginseng"],
        "must_block": ["녹차", "밀크씨슬", "가르시니아"],
    },
    {
        "raw": "루테인",
        "allow_none": False,
        "must_allow": ["루테인", "지아잔틴", "마리골드", "lutein", "zeaxanthin"],
        "must_block": ["녹차", "홍삼", "가르시니아"],
    },
    {
        "raw": "지아잔틴",
        "allow_none": False,
        "must_allow": ["루테인", "지아잔틴", "마리골드", "lutein", "zeaxanthin"],
        "must_block": ["녹차", "홍삼", "가르시니아"],
    },
    {
        "raw": "EPA",
        "allow_none": False,
        "must_allow": ["epa", "dha", "오메가", "omega", "어유", "fish oil"],
        "must_block": ["녹차", "카테킨", "가르시니아"],
    },
    {
        "raw": "DHA",
        "allow_none": False,
        "must_allow": ["epa", "dha", "오메가", "omega", "어유", "fish oil"],
        "must_block": ["녹차", "카테킨", "가르시니아"],
    },
    {
        "raw": "오메가3",
        "allow_none": False,
        "must_allow": ["epa", "dha", "오메가", "omega", "어유", "fish oil"],
        "must_block": ["녹차", "카테킨", "가르시니아"],
    },
    {
        "raw": "비오틴",
        "allow_none": True,
        "must_allow": ["비오틴", "biotin"],
        "must_block": ["녹차", "홍삼", "가르시니아", "프로바이오틱스"],
    },
    {
        "raw": "건조비타민D3",
        "allow_none": False,
        "must_allow": ["비타민d", "vitamin d"],
        "must_block": ["녹차", "홍삼", "가르시니아"],
    },
    {
        "raw": "산화아연",
        "allow_none": False,
        "must_allow": ["아연", "zinc"],
        "must_block": ["녹차", "홍삼", "가르시니아"],
    },
    {
        "raw": "유산균",
        "allow_none": True,
        "must_allow": ["유산균", "프로바이오틱스", "lactobacillus", "bifidobacterium"],
        "must_block": ["녹차", "홍삼", "가르시니아", "밀크씨슬"],
    },
    {
        "raw": "밀크씨슬추출물분말",
        "allow_none": False,
        "must_allow": ["밀크씨슬", "밀크시슬", "milk thistle"],
        "must_block": ["녹차", "홍삼", "가르시니아"],
    },
    {
        "raw": "결정셀룰로스",
        "allow_none": True,
        "must_allow": ["결정셀룰로스"],
        "must_block": ["녹차", "홍삼", "가르시니아", "밀크씨슬"],
        "allow_relation_types": ["excipient"],
    },
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    service = RecommendationService()
    upload_service = UploadRecommendationService(service)
    service.ensure_loaded()
    upload_service._ensure_loaded()

    failures: List[str] = []

    for case in CASES:
        raw = case["raw"]
        normalized_input = canonicalize_ingredient_for_matching(raw) or raw
        match = upload_service._find_match_in_cache(normalized_input, raw_ingredient=raw, display_name=raw)
        payload = build_display_payload(match)

        print(
            json.dumps(
                {
                    "raw": raw,
                    "normalized_input": normalized_input,
                    "result": payload,
                },
                ensure_ascii=False,
            )
        )

        if payload is None:
            if not case.get("allow_none", False):
                failures.append(f"{raw}: expected a safe exact/alias match but got None")
            continue

        functional_name = str(payload.get("functional_ingredient_name", "") or "")
        relation_type = str(payload.get("relation_type", "") or "")

        if case.get("allow_relation_types") and relation_type not in set(case["allow_relation_types"]):
            failures.append(f"{raw}: relation_type {relation_type!r} is not allowed")

        if case.get("must_block") and contains_any(functional_name, case["must_block"]):
            failures.append(f"{raw}: blocked candidate matched -> {functional_name}")

        if case.get("must_allow") and not contains_any(functional_name, case["must_allow"]):
            failures.append(f"{raw}: expected keywords {case['must_allow']} not found in {functional_name}")

    passed = len(CASES) - len(failures)
    print(json.dumps({"passed": passed, "failed": len(failures)}, ensure_ascii=False))
    for failure in failures:
        print(f"FAIL: {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
