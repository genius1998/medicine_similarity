import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.recommendation_quality_judge_batch import (  # noqa: E402
    build_batch_requests,
    build_snapshot_item,
    compact_profile,
    extract_response_text,
    merge_parts,
    parse_model_json,
    response_from_batch_line,
    select_product_ids,
)


def test_select_product_ids_stratifies_by_main_category():
    profiles = {
        "p1": {"report_no": "1", "product_main_category": "면역"},
        "p2": {"report_no": "2", "product_main_category": "면역"},
        "p3": {"report_no": "3", "product_main_category": "장 건강"},
        "p4": {"report_no": "4", "product_main_category": "장 건강"},
    }
    product_vectors = {key: {"x": 1.0} for key in profiles}

    selected = select_product_ids(
        profiles,
        product_vectors,
        all_products=False,
        report_nos=[],
        sample_size=0,
        per_category=1,
        seed=1,
    )

    assert len(selected) == 2
    assert {profiles[product_id]["product_main_category"] for product_id in selected} == {"면역", "장 건강"}


def test_build_batch_requests_contains_json_response_config():
    snapshot = {
        "key": "recjudge_000001",
        "base_product": compact_profile(
            {
                "report_no": "B1",
                "product_name": "base",
                "product_main_category": "면역",
                "primary_ingredients": ["홍삼"],
            }
        ),
        "recommendations": [
            {
                "rank": 1,
                "target_profile": compact_profile(
                    {
                        "report_no": "T1",
                        "product_name": "target",
                        "product_main_category": "면역",
                        "primary_ingredients": ["홍삼"],
                    }
                ),
                "similarity_score": 0.9,
                "function_similarity_score": 1.0,
                "core_match_score": 1.0,
                "substitutability": "high",
                "shared_ingredients": ["홍삼"],
                "shared_categories": ["면역"],
                "semantic_core_overlap": ["홍삼"],
                "semantic_core_reason": "full_semantic_core_coverage",
                "score_adjustments": [],
                "local_risk_flags": [],
                "algorithm_reason": "홍삼 공통",
            }
        ],
    }

    requests = build_batch_requests([snapshot])

    assert requests[0]["key"] == "recjudge_000001"
    assert requests[0]["request"]["generation_config"]["response_mime_type"] == "application/json"
    prompt = requests[0]["request"]["contents"][0]["parts"][0]["text"]
    assert "reasonable|acceptable_adjacent|weak|bad" in prompt
    assert "홍삼" in prompt


def test_build_batch_requests_skips_empty_recommendations():
    assert build_batch_requests([{"key": "recjudge_000001", "base_product": {}, "recommendations": []}]) == []


def test_build_snapshot_item_filters_display_ineligible_rows():
    profiles = {
        "base": {
            "report_no": "B1",
            "product_name": "base",
            "product_main_category": "nutrition",
            "primary_ingredients": ["a"],
        },
        "low": {
            "report_no": "T1",
            "product_name": "low",
            "product_main_category": "nutrition",
            "primary_ingredients": ["a"],
        },
        "high": {
            "report_no": "T2",
            "product_name": "high",
            "product_main_category": "nutrition",
            "primary_ingredients": ["a"],
        },
    }
    rows = [
        {
            "target_product_id": "low",
            "similarity_score": 0.64,
            "function_similarity_score": 1.0,
            "core_match_score": 1.0,
            "substitutability": "low",
            "recommendation_display_eligible": False,
            "shared_ingredients_json": json.dumps(["a"]),
            "shared_categories_json": json.dumps(["nutrition"]),
            "base_only_ingredients_json": "[]",
            "target_only_ingredients_json": "[]",
            "explanation_json": "{}",
            "reason": "low",
        },
        {
            "target_product_id": "high",
            "similarity_score": 0.8,
            "function_similarity_score": 1.0,
            "core_match_score": 1.0,
            "substitutability": "high",
            "recommendation_quality": "strong_match",
            "recommendation_display_eligible": True,
            "shared_ingredients_json": json.dumps(["a"]),
            "shared_categories_json": json.dumps(["nutrition"]),
            "base_only_ingredients_json": "[]",
            "target_only_ingredients_json": "[]",
            "explanation_json": "{}",
            "reason": "high",
        },
    ]

    snapshot = build_snapshot_item("recjudge_000001", "base", rows, profiles)

    assert len(snapshot["recommendations"]) == 1
    assert snapshot["recommendations"][0]["target_profile"]["report_no"] == "T2"
    assert snapshot["recommendations"][0]["rank"] == 1


def test_parse_batch_inline_response_text_json():
    payload = {
        "key": "recjudge_000001",
        "inlineResponse": {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "base_report_no": "B1",
                                            "labels": [
                                                {
                                                    "rank": 1,
                                                    "target_report_no": "T1",
                                                    "judgment": "reasonable",
                                                    "confidence": 0.9,
                                                    "reason": "핵심 원료와 카테고리가 일치",
                                                    "risk_flags": [],
                                                    "suggested_rule": "",
                                                }
                                            ],
                                            "overall_notes": "",
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        },
    }

    response = response_from_batch_line(payload)
    parsed = parse_model_json(extract_response_text(response))

    assert parsed["labels"][0]["judgment"] == "reasonable"


def test_merge_parts_suppresses_retry_covered_errors(tmp_path, monkeypatch):
    output_dir = tmp_path / "merged"
    part_dir = tmp_path / "chunk_part_000"
    retry_dir = tmp_path / "chunk_part_retry_000"
    part_dir.mkdir()
    retry_dir.mkdir()

    (part_dir / "gemini_judge_results.csv").write_text(
        "key,base_main_category,judge_judgment,similarity_score\n"
        "recjudge_000001,장 건강,reasonable,0.8\n",
        encoding="utf-8-sig",
    )
    (part_dir / "gemini_judge_errors.csv").write_text(
        "line_number,key,error,raw\n"
        "1,recjudge_000002,timeout,{}\n",
        encoding="utf-8-sig",
    )
    (part_dir / "gemini_judge_summary.json").write_text(
        json.dumps({"error_count": 1}),
        encoding="utf-8",
    )
    (retry_dir / "gemini_judge_results.csv").write_text(
        "key,base_main_category,judge_judgment,similarity_score\n"
        "recjudge_000002,장 건강,acceptable_adjacent,0.7\n",
        encoding="utf-8-sig",
    )
    (retry_dir / "gemini_judge_errors.csv").write_text(
        "line_number,key,error,raw\n",
        encoding="utf-8-sig",
    )
    (retry_dir / "gemini_judge_summary.json").write_text(
        json.dumps({"error_count": 0}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    merge_parts(
        SimpleNamespace(
            output_dir=str(output_dir),
            parts_glob=str(tmp_path / "chunk_part*"),
            high_score_threshold=0.65,
        )
    )

    summary = json.loads((output_dir / "merged_gemini_judge_summary.json").read_text(encoding="utf-8"))

    assert summary["result_count"] == 2
    assert summary["error_count"] == 0
    assert summary["raw_error_count"] == 1
    assert summary["retry_covered_error_count"] == 1
    assert (output_dir / "merged_gemini_judge_errors.csv").read_text(encoding="utf-8-sig").strip() == "part_dir,line_number,key,error,raw"
    assert "recjudge_000002" in (output_dir / "merged_gemini_judge_errors_all.csv").read_text(encoding="utf-8-sig")
