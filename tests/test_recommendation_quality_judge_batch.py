import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.recommendation_quality_judge_batch as judge_batch  # noqa: E402
from scripts.recommendation_quality_judge_batch import (  # noqa: E402
    build_batch_requests,
    build_openai_batch_requests,
    build_snapshot_item,
    compact_profile,
    extract_response_text,
    merge_parts,
    parse_model_json,
    next_sample_plan,
    pattern_features_for_row,
    pattern_impact_rows,
    quality_gate_decision,
    response_from_batch_line,
    select_product_ids,
    summarize_chunks,
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
        main_categories=[],
        sample_size=0,
        per_category=1,
        seed=1,
    )

    assert len(selected) == 2
    assert {profiles[product_id]["product_main_category"] for product_id in selected} == {"면역", "장 건강"}


def test_select_product_ids_filters_by_main_category():
    profiles = {
        "p1": {"report_no": "1", "product_main_category": "A"},
        "p2": {"report_no": "2", "product_main_category": "A"},
        "p3": {"report_no": "3", "product_main_category": "B"},
    }
    product_vectors = {key: {"x": 1.0} for key in profiles}

    selected = select_product_ids(
        profiles,
        product_vectors,
        all_products=False,
        report_nos=[],
        main_categories=["B"],
        sample_size=0,
        per_category=10,
        seed=1,
    )

    assert selected == ["p3"]


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


def test_build_openai_batch_requests_uses_responses_json_schema():
    snapshot = {
        "key": "recjudge_000001",
        "base_product": compact_profile(
            {
                "report_no": "B1",
                "product_name": "base",
                "product_main_category": "硫댁뿭",
                "primary_ingredients": ["?띿궪"],
            }
        ),
        "recommendations": [
            {
                "rank": 1,
                "target_profile": compact_profile(
                    {
                        "report_no": "T1",
                        "product_name": "target",
                        "product_main_category": "硫댁뿭",
                        "primary_ingredients": ["?띿궪"],
                    }
                ),
                "similarity_score": 0.9,
                "function_similarity_score": 1.0,
                "core_match_score": 1.0,
                "substitutability": "high",
                "recommendation_quality": "strong_match",
                "recommendation_review_reason": "",
                "shared_ingredients": ["?띿궪"],
                "shared_categories": ["硫댁뿭"],
                "semantic_core_overlap": ["?띿궪"],
                "semantic_core_reason": "full_semantic_core_coverage",
                "score_adjustments": [],
                "local_risk_flags": [],
                "algorithm_reason": "?띿궪 怨듯넻",
            }
        ],
    }

    requests = build_openai_batch_requests([snapshot])

    assert requests[0]["custom_id"] == "recjudge_000001"
    assert requests[0]["method"] == "POST"
    assert requests[0]["url"] == "/v1/responses"
    assert requests[0]["body"]["model"] == "gpt-5-nano"
    assert requests[0]["body"]["text"]["format"]["type"] == "json_schema"
    assert requests[0]["body"]["text"]["format"]["strict"] is True
    assert requests[0]["body"]["reasoning"]["effort"] == "minimal"


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


def test_parse_openai_batch_response_text_json():
    payload = {
        "custom_id": "recjudge_000001",
        "response": {
            "status_code": 200,
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "base_report_no": "B1",
                                        "labels": [
                                            {
                                                "rank": 1,
                                                "target_report_no": "T1",
                                                "judgment": "reasonable",
                                                "confidence": 0.9,
                                                "reason": "same core ingredient",
                                                "risk_flags": [],
                                                "suggested_rule": "",
                                            }
                                        ],
                                        "overall_notes": "",
                                    }
                                ),
                            }
                        ],
                    }
                ]
            },
        },
    }

    response = response_from_batch_line(payload)
    parsed = parse_model_json(extract_response_text(response))

    assert parsed["labels"][0]["target_report_no"] == "T1"


def test_openai_list_filters_active_batches_and_writes_json(tmp_path, monkeypatch):
    class FakeBatches:
        def __init__(self):
            self.limit = None

        def list(self, **kwargs):
            self.limit = kwargs.get("limit")
            return [
                {
                    "id": "batch_active",
                    "status": "in_progress",
                    "created_at": 1,
                    "request_counts": {"completed": 3, "failed": 0, "total": 5},
                    "metadata": {"name": "active validation"},
                    "input_file_id": "file_input",
                },
                {
                    "id": "batch_done",
                    "status": "completed",
                    "created_at": 2,
                    "completed_at": 3,
                    "request_counts": {"completed": 7, "failed": 0, "total": 7},
                    "metadata": {"name": "done validation"},
                },
            ]

    fake_batches = FakeBatches()
    fake_client = SimpleNamespace(batches=fake_batches)
    monkeypatch.setattr(judge_batch, "load_openai_client", lambda env_path: fake_client)
    output_json = tmp_path / "openai_batches.json"

    judge_batch.openai_list(
        SimpleNamespace(
            env_path="unused.env",
            limit=10,
            active_only=True,
            output_json=str(output_json),
        )
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))

    assert fake_batches.limit == 10
    assert result["count"] == 1
    assert result["batches"][0]["id"] == "batch_active"
    assert result["batches"][0]["metadata_name"] == "active validation"
    assert result["batches"][0]["request_counts"]["total"] == 5
    assert "validating" in result["active_statuses"]


def test_openai_check_watch_downloads_when_completed(tmp_path, monkeypatch):
    class FakeBatches:
        def __init__(self):
            self.calls = 0

        def retrieve(self, job_id):
            self.calls += 1
            if self.calls == 1:
                return {
                    "id": job_id,
                    "status": "in_progress",
                    "request_counts": {"completed": 1, "failed": 0, "total": 2},
                }
            return {
                "id": job_id,
                "status": "completed",
                "request_counts": {"completed": 2, "failed": 0, "total": 2},
                "output_file_id": "file_output",
            }

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_output"
            return b'{"custom_id":"recjudge_000001"}\n'

    fake_batches = FakeBatches()
    fake_client = SimpleNamespace(batches=fake_batches, files=FakeFiles())
    monkeypatch.setattr(judge_batch, "load_openai_client", lambda env_path: fake_client)
    monkeypatch.setattr(judge_batch.time, "sleep", lambda seconds: None)

    judge_batch.openai_check(
        SimpleNamespace(
            output_dir=str(tmp_path),
            env_path="unused.env",
            job="batch_test",
            job_file="",
            download=True,
            download_output="",
            download_errors=True,
            error_output="",
            watch=True,
            poll_seconds=1,
            timeout_seconds=30,
        )
    )

    result_path = tmp_path / "openai_recommendation_judge_result.jsonl"

    assert fake_batches.calls == 2
    assert result_path.read_text(encoding="utf-8") == '{"custom_id":"recjudge_000001"}\n'


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


def test_summarize_chunks_prefers_impact_scores_and_applies_retry(tmp_path):
    output_dir = tmp_path / "merged"
    impact_part = tmp_path / "recommendation_quality_judge_v2_8_openai_chunk_000_002"
    retry_parent = tmp_path / "recommendation_quality_judge_v2_9_openai_chunk_002_003"
    retry_dir = tmp_path / "recommendation_quality_judge_v2_9_openai_chunk_002_003_retry_000001"
    impact_part.mkdir()
    retry_parent.mkdir()
    retry_dir.mkdir()

    snapshot_impact = [
        {
            "key": "recjudge_000001",
            "base_product": {"report_no": "B1"},
            "recommendations": [{"rank": 1}, {"rank": 2}],
        },
        {
            "key": "recjudge_000002",
            "base_product": {"report_no": "B2"},
            "recommendations": [{"rank": 1}],
        },
    ]
    (impact_part / "recommendation_snapshot.jsonl").write_text(
        "\n".join(json.dumps(row) for row in snapshot_impact) + "\n",
        encoding="utf-8",
    )
    (impact_part / "post_fix_score_impact.csv").write_text(
        "base_report_no,base_product_name,base_main_category,target_report_no,target_product_name,target_main_category,rank,"
        "judge_judgment,judge_confidence,judge_reason,old_similarity_score,new_similarity_score,score_delta,"
        "old_high_weak_or_bad,new_high_weak_or_bad,reduced_below_high_threshold,shared_after_json,"
        "score_adjustment_types_json,score_adjustments_json\n"
        "B1,base one,cat,T1,target one,cat,1,weak,0.7,too broad,0.8,0.5,-0.3,1,0,1,[],[],[]\n"
        "B1,base one,cat,T2,target two,cat,2,reasonable,0.9,good,0.7,0.7,0.0,0,0,0,[],[],[]\n"
        "B2,base two,cat,T3,target three,cat,1,weak,0.8,still high,0.8,0.8,0.0,1,1,0,[],[],[]\n",
        encoding="utf-8-sig",
    )

    (retry_parent / "recommendation_snapshot.jsonl").write_text(
        json.dumps(
            {
                "key": "recjudge_000001",
                "base_product": {"report_no": "B3"},
                "recommendations": [{"rank": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (retry_parent / "gemini_judge_results.csv").write_text(
        "key,base_report_no,base_product_name,base_main_category,rank,target_report_no,target_product_name,target_main_category,"
        "similarity_score,function_similarity_score,core_match_score,shared_ingredients_json,shared_categories_json,"
        "local_risk_flags_json,local_quality_bucket,judge_judgment,judge_confidence,judge_reason,judge_risk_flags_json,suggested_rule\n"
        "recjudge_000001,B3,base three,cat,1,T4,target four,cat,0.9,,,[],[],[],likely_reasonable,weak,0.6,extra label,[],\n",
        encoding="utf-8-sig",
    )

    (retry_dir / "recommendation_snapshot.jsonl").write_text(
        json.dumps(
            {
                "key": "recjudge_000001",
                "base_product": {"report_no": "B3"},
                "recommendations": [{"rank": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (retry_dir / "gemini_judge_results.csv").write_text(
        "key,base_report_no,base_product_name,base_main_category,rank,target_report_no,target_product_name,target_main_category,"
        "similarity_score,function_similarity_score,core_match_score,shared_ingredients_json,shared_categories_json,"
        "local_risk_flags_json,local_quality_bucket,judge_judgment,judge_confidence,judge_reason,judge_risk_flags_json,suggested_rule\n"
        "recjudge_000001,B3,base three,cat,1,T4,target four,cat,0.9,,,[],[],[],likely_reasonable,acceptable_adjacent,0.9,retry fixed,[],\n",
        encoding="utf-8-sig",
    )

    summarize_chunks(
        SimpleNamespace(
            output_dir=str(output_dir),
            parts_glob=[str(tmp_path / "recommendation_quality_judge_v2_*_openai_chunk_*_*")],
            retry_glob=[str(tmp_path / "recommendation_quality_judge_v2_*_openai_chunk_*_retry_*")],
            high_score_threshold=0.65,
        )
    )

    summary = json.loads((output_dir / "openai_chunk_judge_summary.json").read_text(encoding="utf-8"))
    merged_csv = (output_dir / "openai_chunk_judge_results.csv").read_text(encoding="utf-8-sig")

    assert summary["coverage_ok"] is True
    assert summary["expected_label_count"] == 4
    assert summary["actual_label_count"] == 4
    assert summary["judgment_counts"] == {"weak": 2, "reasonable": 1, "acceptable_adjacent": 1}
    assert summary["current_high_score_weak_or_bad_count"] == 1
    assert summary["retry_replacements"][0]["removed_label_count"] == 1
    assert "post_fix_score_impact" in merged_csv
    assert "retry fixed" in merged_csv
    assert "extra label" not in merged_csv


def test_pattern_features_and_impact_rows_capture_cross_role_primary_overlap():
    base = {
        "report_no": "B1",
        "product_name": "base",
        "product_main_category": "skin",
        "primary_ingredients": ["hyaluronic"],
        "secondary_ingredients": ["vitamin-c"],
        "category_scores": {"skin": 5.0},
        "role_by_ingredient": {"hyaluronic": "primary", "vitamin-c": "secondary"},
        "ingredient_scores": [
            {"ingredient": "hyaluronic", "weight": 0.95, "role": "primary", "category_main": "skin", "category_sub": ""},
            {"ingredient": "vitamin-c", "weight": 0.95, "role": "secondary", "category_main": "skin", "category_sub": ""},
        ],
    }
    target = {
        "report_no": "T1",
        "product_name": "target",
        "product_main_category": "skin",
        "primary_ingredients": ["vitamin-c"],
        "secondary_ingredients": ["hyaluronic"],
        "category_scores": {"skin": 5.0},
        "role_by_ingredient": {"vitamin-c": "primary", "hyaluronic": "secondary"},
        "ingredient_scores": [
            {"ingredient": "vitamin-c", "weight": 0.95, "role": "primary", "category_main": "skin", "category_sub": ""},
            {"ingredient": "hyaluronic", "weight": 0.95, "role": "secondary", "category_main": "skin", "category_sub": ""},
        ],
    }
    ingredient_profiles = {
        "hyaluronic": {"ingredient_main_category": "skin", "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
        "vitamin-c": {"ingredient_main_category": "skin", "ingredient_type": "functional", "vector_include": True, "is_excipient": False},
    }
    weak_row = {
        "base_report_no": "B1",
        "target_report_no": "T1",
        "judge_judgment": "weak",
        "similarity_score": "0.8",
        "function_similarity_score": "0.5",
    }
    reasonable_row = dict(weak_row, judge_judgment="reasonable")

    weak_features = pattern_features_for_row(weak_row, base, target, ingredient_profiles)
    reasonable_features = pattern_features_for_row(reasonable_row, base, target, ingredient_profiles)
    impact_rows = pattern_impact_rows([weak_features, reasonable_features], high_score_threshold=0.65)
    impact_by_name = {row["pattern"]: row for row in impact_rows}

    assert weak_features["no_primary_primary_overlap_cross_role"] == 1
    assert weak_features["primary_primary_overlap_count"] == 0
    assert weak_features["shared_count"] == 2
    assert impact_by_name["no_primary_primary_overlap_cross_role_shared_le2"]["matched_count"] == 2
    assert impact_by_name["no_primary_primary_overlap_cross_role_shared_le2"]["weak_or_bad_count"] == 1
    assert impact_by_name["no_primary_primary_overlap_cross_role_shared_le2"]["non_weak_affected_count"] == 1


def test_quality_gate_passes_when_rates_are_low_and_patterns_are_not_actionable():
    args = SimpleNamespace(
        max_weak_or_bad_rate=0.10,
        max_high_score_weak_or_bad_rate=0.02,
        min_actionable_pattern_weak_count=5,
        min_actionable_pattern_weak_rate=0.50,
        max_actionable_pattern_non_weak_affected=5,
    )
    summary = {
        "row_count": 2135,
        "weak_or_bad_rate": 0.0745,
        "high_score_weak_or_bad_count": 25,
        "pattern_impacts": [
            {
                "pattern": "cross_role",
                "weak_or_bad_count": 6,
                "weak_or_bad_rate": 0.1333,
                "non_weak_affected_count": 39,
            }
        ],
    }

    result = quality_gate_decision(summary, args)

    assert result["decision"] == "pass_continue_validation_without_algorithm_change"
    assert result["high_score_weak_or_bad_rate"] == 0.0117
    assert result["actionable_pattern_count"] == 0


def test_quality_gate_flags_low_blast_radius_algorithm_candidate():
    args = SimpleNamespace(
        max_weak_or_bad_rate=0.10,
        max_high_score_weak_or_bad_rate=0.02,
        min_actionable_pattern_weak_count=5,
        min_actionable_pattern_weak_rate=0.50,
        max_actionable_pattern_non_weak_affected=5,
    )
    summary = {
        "row_count": 1000,
        "weak_or_bad_rate": 0.03,
        "high_score_weak_or_bad_count": 10,
        "pattern_impacts": [
            {
                "pattern": "candidate",
                "weak_or_bad_count": 5,
                "weak_or_bad_rate": 0.7143,
                "non_weak_affected_count": 2,
            }
        ],
    }

    result = quality_gate_decision(summary, args)

    assert result["decision"] == "algorithm_change_candidate"
    assert result["actionable_pattern_count"] == 1


def test_quality_gate_requests_more_sampling_when_rates_exceed_gate_without_actionable_pattern():
    args = SimpleNamespace(
        max_weak_or_bad_rate=0.10,
        max_high_score_weak_or_bad_rate=0.02,
        min_actionable_pattern_weak_count=5,
        min_actionable_pattern_weak_rate=0.50,
        max_actionable_pattern_non_weak_affected=5,
    )
    summary = {
        "row_count": 1000,
        "weak_or_bad_rate": 0.12,
        "high_score_weak_or_bad_count": 35,
        "pattern_impacts": [
            {
                "pattern": "broad_pattern",
                "weak_or_bad_count": 12,
                "weak_or_bad_rate": 0.10,
                "non_weak_affected_count": 108,
            }
        ],
    }

    result = quality_gate_decision(summary, args)

    assert result["decision"] == "review_collect_more_targeted_samples"
    assert "overall_weak_rate_exceeds_gate" in result["reasons"]
    assert "high_score_weak_rate_exceeds_gate" in result["reasons"]


def test_validate_results_orchestrates_summary_analysis_and_gate(tmp_path, monkeypatch):
    calls = []

    def fake_summarize(args):
        calls.append(("summarize", args.output_dir, list(args.parts_glob), list(args.retry_glob)))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "openai_chunk_judge_results.csv").write_text("base_report_no,target_report_no\n", encoding="utf-8")
        (output_dir / "openai_chunk_judge_summary.json").write_text(json.dumps({"actual_label_count": 1}), encoding="utf-8")

    def fake_analyze(args):
        calls.append(("analyze", args.results_csv, args.output_dir))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "judge_pattern_summary.json").write_text(
            json.dumps(
                {
                    "row_count": 1,
                    "weak_or_bad_rate": 0.0,
                    "high_score_weak_or_bad_count": 0,
                    "pattern_impacts": [],
                }
            ),
            encoding="utf-8",
        )

    def fake_quality_gate_decision(summary, args):
        calls.append(("gate", summary["row_count"], args.max_weak_or_bad_rate))
        return {
            "decision": "pass_continue_validation_without_algorithm_change",
            "row_count": summary["row_count"],
            "weak_or_bad_rate": summary["weak_or_bad_rate"],
        }

    monkeypatch.setattr(judge_batch, "summarize_chunks", fake_summarize)
    monkeypatch.setattr(judge_batch, "analyze_patterns", fake_analyze)
    monkeypatch.setattr(judge_batch, "quality_gate_decision", fake_quality_gate_decision)

    judge_batch.validate_results(
        SimpleNamespace(
            output_dir=str(tmp_path / "validation"),
            parts_glob=["output/chunk_*"],
            retry_glob=["output/retry_*"],
            ingredient_category_profile="",
            high_score_threshold=0.65,
            max_weak_or_bad_rate=0.10,
            max_high_score_weak_or_bad_rate=0.02,
            min_actionable_pattern_weak_count=5,
            min_actionable_pattern_weak_rate=0.50,
            max_actionable_pattern_non_weak_affected=5,
            fail_on_review=False,
        )
    )

    validation_summary = json.loads((tmp_path / "validation" / "judge_validation_summary.json").read_text(encoding="utf-8"))

    assert [item[0] for item in calls] == ["summarize", "analyze", "gate"]
    assert validation_summary["decision"] == "pass_continue_validation_without_algorithm_change"
    assert (tmp_path / "validation" / "judge_quality_gate.json").exists()


def test_finalize_openai_result_orchestrates_apply_analysis_and_gate(tmp_path, monkeypatch):
    calls = []
    output_dir = tmp_path / "part"
    output_dir.mkdir()
    result_jsonl = output_dir / "openai_recommendation_judge_result.jsonl"
    result_jsonl.write_text("{}\n", encoding="utf-8")

    def fake_apply(args):
        calls.append(("apply", args.output_dir, args.result_jsonl, args.high_score_threshold))
        Path(args.output_dir, "gemini_judge_results.csv").write_text("base_report_no,target_report_no\n", encoding="utf-8")

    def fake_analyze(args):
        calls.append(("analyze", args.results_csv, args.output_dir, args.high_score_threshold))
        pattern_dir = Path(args.output_dir)
        pattern_dir.mkdir(parents=True, exist_ok=True)
        (pattern_dir / "judge_pattern_summary.json").write_text(
            json.dumps(
                {
                    "row_count": 10,
                    "weak_or_bad_rate": 0.1,
                    "high_score_weak_or_bad_count": 1,
                    "pattern_impacts": [],
                }
            ),
            encoding="utf-8",
        )

    def fake_quality_gate_decision(summary, args):
        calls.append(("gate", summary["row_count"], args.max_weak_or_bad_rate))
        return {
            "decision": "pass_continue_validation_without_algorithm_change",
            "row_count": summary["row_count"],
            "weak_or_bad_rate": summary["weak_or_bad_rate"],
        }

    monkeypatch.setattr(judge_batch, "apply_results", fake_apply)
    monkeypatch.setattr(judge_batch, "analyze_patterns", fake_analyze)
    monkeypatch.setattr(judge_batch, "quality_gate_decision", fake_quality_gate_decision)

    judge_batch.finalize_openai_result(
        SimpleNamespace(
            output_dir=str(output_dir),
            result_jsonl="",
            snapshot_jsonl="",
            pattern_output_dir="",
            ingredient_category_profile="",
            high_score_threshold=0.65,
            max_weak_or_bad_rate=0.10,
            max_high_score_weak_or_bad_rate=0.02,
            min_actionable_pattern_weak_count=5,
            min_actionable_pattern_weak_rate=0.50,
            max_actionable_pattern_non_weak_affected=5,
            fail_on_review=False,
        )
    )

    summary = json.loads((output_dir / "openai_finalize_summary.json").read_text(encoding="utf-8"))

    assert [item[0] for item in calls] == ["apply", "analyze", "gate"]
    assert summary["decision"] == "pass_continue_validation_without_algorithm_change"
    assert (output_dir / "judge_quality_gate.json").exists()
    pattern_summary_json = summary["outputs"]["pattern_summary_json"]
    assert pattern_summary_json.endswith("patterns\\judge_pattern_summary.json") or pattern_summary_json.endswith("patterns/judge_pattern_summary.json")


def test_write_validation_report_builds_markdown_from_validation_outputs(tmp_path):
    validation_dir = tmp_path / "validation"
    pattern_dir = validation_dir / "patterns"
    pattern_dir.mkdir(parents=True)
    (validation_dir / "openai_chunk_judge_summary.json").write_text(
        json.dumps(
            {
                "actual_label_count": 100,
                "expected_label_count": 100,
                "coverage_ok": True,
                "product_count": 20,
                "request_count": 10,
                "judgment_counts": {
                    "reasonable": 30,
                    "acceptable_adjacent": 60,
                    "weak": 10,
                },
                "weak_or_bad_rate": 0.10,
                "current_high_score_weak_or_bad_count": 1,
                "category_judgment_counts": {
                    "A|reasonable": 5,
                    "A|acceptable_adjacent": 5,
                    "A|weak": 5,
                    "B|reasonable": 20,
                    "B|acceptable_adjacent": 20,
                    "B|weak": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (pattern_dir / "judge_pattern_summary.json").write_text(
        json.dumps(
            {
                "pattern_impacts": [
                    {
                        "pattern": "candidate",
                        "matched_count": 12,
                        "weak_or_bad_count": 5,
                        "weak_or_bad_rate": 0.4167,
                        "non_weak_affected_count": 7,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (validation_dir / "judge_quality_gate.json").write_text(
        json.dumps(
            {
                "decision": "pass_continue_validation_without_algorithm_change",
                "reasons": ["candidate_patterns_are_not_actionable_without_overfiltering"],
                "high_score_weak_or_bad_count": 1,
                "high_score_weak_or_bad_rate": 0.01,
                "actionable_pattern_count": 0,
            }
        ),
        encoding="utf-8",
    )

    judge_batch.write_validation_report(
        SimpleNamespace(
            validation_dir=str(validation_dir),
            summary_json="",
            pattern_summary_json="",
            quality_gate_json="",
            output_md="",
            top_categories=1,
            top_patterns=1,
        )
    )

    report = (validation_dir / "judge_validation_report.md").read_text(encoding="utf-8")

    assert "pass_continue_validation_without_algorithm_change" in report
    assert "keep_current_algorithm_without_new_caps" in report
    assert "| Label coverage | 100 / 100 |" in report
    assert "| A | 15 | 5 | 33.33% |" in report
    assert "| candidate | 12 | 5 | 41.67% | 7 |" in report


def test_next_sample_plan_selects_high_weak_rate_categories_before_fallback():
    summary = {
        "category_judgment_counts": {
            "A|reasonable": 20,
            "A|acceptable_adjacent": 60,
            "A|weak": 20,
            "B|reasonable": 40,
            "B|acceptable_adjacent": 55,
            "B|weak": 5,
            "C|reasonable": 10,
            "C|acceptable_adjacent": 80,
            "C|weak": 10,
            "D|reasonable": 8,
            "D|acceptable_adjacent": 1,
            "D|weak": 1,
        }
    }
    args = SimpleNamespace(
        summary_json="summary.json",
        min_labels=50,
        min_weak_rate=0.10,
        max_categories=3,
        per_category=7,
        seed=123,
        sample_output_dir="output/next",
        top_k=10,
        workers=4,
        progress_every=10,
    )

    plan = next_sample_plan(summary, args)

    assert [row["category"] for row in plan["selected_categories"]] == ["A", "C", "B"]
    assert plan["selected_categories"][0]["selection_reason"] == "weak_rate_above_threshold"
    assert plan["selected_categories"][2]["selection_reason"] == "weak_count_fallback"
    assert "--main-category" in plan["prepare_command"]
    assert "A" in plan["prepare_command"]
    assert "D" not in [row["category"] for row in plan["selected_categories"]]
