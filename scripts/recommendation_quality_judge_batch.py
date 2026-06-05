from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.enhance_similarity_with_explanation import (  # noqa: E402
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_MAX_DF_FOR_SEED,
    DEFAULT_TOP_K,
    SIMILARITY_ALGORITHM_VERSION,
    build_vector_indexes,
    calculate_semantic_weighted_jaccard_v2,
    calculate_function_similarity,
    compute_topk_for_single_product,
    load_ingredient_category_profiles,
    load_product_function_profiles,
    load_vector_inputs,
    normalize_similarity_algorithm,
    recommendation_quality_metadata,
    resolve_runtime_paths,
    safe_json_loads,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "recommendation_quality_judge"
DEFAULT_HEALTH_BATCH_DIR = Path(r"D:\health_batch_project")
DEFAULT_BATCH_JSONL_NAME = "gemini_recommendation_judge_batch.jsonl"
DEFAULT_RESULT_JSONL_NAME = "gemini_recommendation_judge_result.jsonl"
DEFAULT_OPENAI_BATCH_JSONL_NAME = "openai_recommendation_judge_batch.jsonl"
DEFAULT_OPENAI_RESULT_JSONL_NAME = "openai_recommendation_judge_result.jsonl"
DEFAULT_OPENAI_JOB_FILE_NAME = "openai_recommendation_judge.job.txt"
JUDGMENTS = {"reasonable", "acceptable_adjacent", "weak", "bad"}
MODEL_NAME = "gemini-2.5-flash-lite"
OPENAI_MODEL_NAME = "gpt-5-nano"
QUALITY_GATE_MAX_WEAK_OR_BAD_RATE = 0.10
QUALITY_GATE_MAX_HIGH_SCORE_WEAK_OR_BAD_RATE = 0.02
QUALITY_GATE_MIN_ACTIONABLE_PATTERN_WEAK_COUNT = 5
QUALITY_GATE_MIN_ACTIONABLE_PATTERN_WEAK_RATE = 0.50
QUALITY_GATE_MAX_ACTIONABLE_PATTERN_NON_WEAK_AFFECTED = 5
_WORKER_CONTEXT: dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compact_list(values: Any, limit: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def compact_profile(profile: dict[str, Any], *, include_product_id: bool = False) -> dict[str, Any]:
    item = {
        "report_no": str(profile.get("report_no", "") or ""),
        "product_name": str(profile.get("product_name", "") or ""),
        "product_main_category": str(profile.get("product_main_category", "") or ""),
        "sub_categories": compact_list(
            profile.get("llm_sub_function_categories")
            or profile.get("product_sub_categories")
            or profile.get("legacy_product_sub_categories")
            or [],
            limit=4,
        ),
        "primary_ingredients": compact_list(profile.get("primary_ingredients", []), limit=8),
        "secondary_ingredients": compact_list(profile.get("secondary_ingredients", []), limit=8),
        "support_ingredients": compact_list(profile.get("support_ingredients", []), limit=8),
        "confidence": round(float(profile.get("confidence", 0.0) or 0.0), 4),
    }
    if include_product_id:
        item["product_id"] = str(profile.get("product_id", "") or "")
    return item


def profile_categories(profile: dict[str, Any]) -> set[str]:
    categories = {str(profile.get("product_main_category", "") or "").strip()}
    for key in ("sub_categories", "llm_sub_function_categories", "product_sub_categories", "legacy_product_sub_categories"):
        for value in profile.get(key, []) or []:
            text = str(value or "").strip()
            if text:
                categories.add(text)
    return {item for item in categories if item}


def local_risk_flags(base_profile: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    target_profile = candidate.get("target_profile", {})
    base_main = str(base_profile.get("product_main_category", "") or "").strip()
    target_main = str(target_profile.get("product_main_category", "") or "").strip()
    shared_ingredients = list(candidate.get("shared_ingredients", []) or [])
    shared_categories = list(candidate.get("shared_categories", []) or [])
    score = float(candidate.get("similarity_score", 0.0) or 0.0)
    semantic_core_reason = str(candidate.get("semantic_core_reason", "") or "")

    if base_main and target_main and base_main != target_main and not (profile_categories(base_profile) & profile_categories(target_profile)):
        flags.append("category_mismatch")
    if not shared_ingredients:
        flags.append("no_shared_ingredients")
    if semantic_core_reason == "no_semantic_core_overlap":
        flags.append("no_core_overlap")
    if shared_categories and not shared_ingredients:
        flags.append("category_only_match")
    if score >= 0.65 and any(flag in flags for flag in ("category_mismatch", "no_core_overlap", "category_only_match")):
        flags.append("high_score_with_weak_signal")
    if base_main == "영양보충" or target_main == "영양보충":
        flags.append("nutrition_generic_review")
    return flags


def local_quality_bucket(flags: list[str], similarity_score: float) -> str:
    high_risk = {"category_mismatch", "no_shared_ingredients", "no_core_overlap", "category_only_match"}
    if similarity_score >= 0.65 and high_risk.intersection(flags):
        return "high_risk"
    if high_risk.intersection(flags):
        return "needs_review"
    return "likely_reasonable"


def select_product_ids(
    profiles: dict[str, dict[str, Any]],
    product_vectors: dict[str, dict[str, float]],
    *,
    all_products: bool,
    report_nos: list[str],
    main_categories: list[str],
    sample_size: int,
    per_category: int,
    seed: int,
) -> list[str]:
    wanted_main_categories = {str(value or "").strip() for value in main_categories if str(value or "").strip()}
    eligible = [
        product_id
        for product_id, profile in profiles.items()
        if product_id in product_vectors and str(profile.get("report_no", "") or "").strip()
        and (
            not wanted_main_categories
            or str(profile.get("product_main_category", "") or "").strip() in wanted_main_categories
        )
    ]
    eligible.sort(key=lambda product_id: (str(profiles[product_id].get("report_no", "") or ""), product_id))

    if report_nos:
        wanted = {str(value or "").strip() for value in report_nos if str(value or "").strip()}
        selected = [product_id for product_id in eligible if str(profiles[product_id].get("report_no", "") or "") in wanted]
        missing = sorted(wanted - {str(profiles[product_id].get("report_no", "") or "") for product_id in selected})
        if missing:
            raise ValueError(f"report_no not found: {missing[:10]}")
        return selected

    if all_products:
        return eligible

    rng = random.Random(seed)
    by_category: dict[str, list[str]] = defaultdict(list)
    for product_id in eligible:
        category = str(profiles[product_id].get("product_main_category", "") or "기타").strip() or "기타"
        by_category[category].append(product_id)

    selected: list[str] = []
    for category in sorted(by_category):
        group = list(by_category[category])
        rng.shuffle(group)
        selected.extend(group[: max(0, per_category)])

    if sample_size > 0:
        selected_set = set(selected)
        remaining = [product_id for product_id in eligible if product_id not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, sample_size - len(selected))])
        if len(selected) > sample_size:
            rng.shuffle(selected)
            selected = selected[:sample_size]

    selected.sort(key=lambda product_id: (str(profiles[product_id].get("product_main_category", "") or ""), str(profiles[product_id].get("report_no", "") or ""), product_id))
    return selected


def build_snapshot_item(
    key: str,
    base_product_id: str,
    rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_profile = profiles[base_product_id]
    recommendations: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("recommendation_display_eligible", True)):
            continue
        rank = len(recommendations) + 1
        target_product_id = str(row.get("target_product_id", "") or "")
        target_profile = profiles.get(target_product_id, {})
        explanation = safe_json_loads(row.get("explanation_json", "{}"), {})
        semantic = dict(explanation.get("semantic_weighted_jaccard_v2", {}) or {})
        core_coverage = dict(semantic.get("core_coverage", {}) or {})
        candidate = {
            "rank": rank,
            "target_product_id": target_product_id,
            "target_profile": compact_profile(target_profile, include_product_id=True),
            "similarity_score": round(float(row.get("similarity_score", 0.0) or 0.0), 6),
            "function_similarity_score": round(float(row.get("function_similarity_score", 0.0) or 0.0), 6),
            "core_match_score": round(float(row.get("core_match_score", 0.0) or 0.0), 6),
            "substitutability": str(row.get("substitutability", "") or ""),
            "recommendation_quality": str(row.get("recommendation_quality", "") or ""),
            "recommendation_review_reason": str(row.get("recommendation_review_reason", "") or ""),
            "shared_ingredients": compact_list(safe_json_loads(row.get("shared_ingredients_json", "[]"), []), limit=10),
            "shared_categories": compact_list(safe_json_loads(row.get("shared_categories_json", "[]"), []), limit=8),
            "base_only_ingredients": compact_list(safe_json_loads(row.get("base_only_ingredients_json", "[]"), []), limit=8),
            "target_only_ingredients": compact_list(safe_json_loads(row.get("target_only_ingredients_json", "[]"), []), limit=8),
            "semantic_core_overlap": compact_list(core_coverage.get("shared_core_semantic_keys", []) or core_coverage.get("shared_primary_semantic_keys", []), limit=8),
            "semantic_core_reason": str(core_coverage.get("reason", "") or ""),
            "score_adjustments": list(semantic.get("score_adjustments", []) or []),
            "algorithm_reason": str(row.get("reason", "") or ""),
        }
        flags = local_risk_flags(base_profile, candidate)
        candidate["local_risk_flags"] = flags
        candidate["local_quality_bucket"] = local_quality_bucket(flags, float(candidate["similarity_score"]))
        recommendations.append(candidate)

    return {
        "key": key,
        "base_product_id": base_product_id,
        "base_product": compact_profile(base_profile, include_product_id=True),
        "recommendations": recommendations,
    }


def prepare_worker_init(
    similarity_algorithm: str,
    ingredient_profile_path_value: str,
) -> None:
    runtime = resolve_runtime_paths()
    ingredient_profile_path = Path(ingredient_profile_path_value) if ingredient_profile_path_value else runtime.get("ingredient_category_profile_path")
    vector_df = load_vector_inputs(runtime["vector_csv_path"])
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    ingredient_category_profiles = load_ingredient_category_profiles(ingredient_profile_path)
    product_vectors, product_names, _report_to_product_ids, ingredient_postings, ingredient_frequency = build_vector_indexes(vector_df)
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "profiles": profiles,
            "product_vectors": product_vectors,
            "product_names": product_names,
            "ingredient_postings": ingredient_postings,
            "ingredient_frequency": ingredient_frequency,
            "ingredient_category_profiles": ingredient_category_profiles,
            "similarity_algorithm": similarity_algorithm,
        }
    )


def prepare_product_worker(task: tuple[int, str, int, int, int]) -> dict[str, Any]:
    index, base_product_id, top_k, candidate_limit, max_df_for_seed = task
    profiles = _WORKER_CONTEXT["profiles"]
    base_profile = profiles.get(base_product_id, {})
    key = f"recjudge_{index:06d}"
    try:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rows, failed_candidates, _stats = compute_topk_for_single_product(
                base_product_id,
                top_k,
                candidate_limit,
                max_df_for_seed,
                _WORKER_CONTEXT["product_vectors"],
                _WORKER_CONTEXT["product_names"],
                profiles,
                _WORKER_CONTEXT["ingredient_postings"],
                _WORKER_CONTEXT["ingredient_frequency"],
                _WORKER_CONTEXT["similarity_algorithm"],
                _WORKER_CONTEXT["ingredient_category_profiles"],
            )
        failed_rows = [{"key": key, "stage": "candidate_scoring", **failed} for failed in failed_candidates]
        return {
            "index": index,
            "key": key,
            "base_product_id": base_product_id,
            "base_product_name": str(base_profile.get("product_name", "") or ""),
            "snapshot": build_snapshot_item(key, base_product_id, rows, profiles),
            "failed_rows": failed_rows,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "index": index,
            "key": key,
            "base_product_id": base_product_id,
            "base_product_name": str(base_profile.get("product_name", "") or ""),
            "snapshot": None,
            "failed_rows": [
                {
                    "key": key,
                    "stage": "topk",
                    "base_product_id": base_product_id,
                    "base_report_no": str(base_profile.get("report_no", "") or ""),
                    "base_product_name": str(base_profile.get("product_name", "") or ""),
                    "error": str(exc),
                }
            ],
            "error": str(exc),
        }


def flatten_snapshot(snapshot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for item in snapshot_rows:
        base = item["base_product"]
        for rec in item.get("recommendations", []) or []:
            target = rec.get("target_profile", {})
            flat.append(
                {
                    "key": item["key"],
                    "base_report_no": base.get("report_no", ""),
                    "base_product_name": base.get("product_name", ""),
                    "base_main_category": base.get("product_main_category", ""),
                    "base_sub_categories_json": json.dumps(base.get("sub_categories", []), ensure_ascii=False),
                    "rank": rec.get("rank", ""),
                    "target_report_no": target.get("report_no", ""),
                    "target_product_name": target.get("product_name", ""),
                    "target_main_category": target.get("product_main_category", ""),
                    "target_sub_categories_json": json.dumps(target.get("sub_categories", []), ensure_ascii=False),
                    "similarity_score": rec.get("similarity_score", 0.0),
                    "function_similarity_score": rec.get("function_similarity_score", 0.0),
                    "core_match_score": rec.get("core_match_score", 0.0),
                    "substitutability": rec.get("substitutability", ""),
                    "recommendation_quality": rec.get("recommendation_quality", ""),
                    "recommendation_review_reason": rec.get("recommendation_review_reason", ""),
                    "shared_ingredients_json": json.dumps(rec.get("shared_ingredients", []), ensure_ascii=False),
                    "shared_categories_json": json.dumps(rec.get("shared_categories", []), ensure_ascii=False),
                    "semantic_core_overlap_json": json.dumps(rec.get("semantic_core_overlap", []), ensure_ascii=False),
                    "semantic_core_reason": rec.get("semantic_core_reason", ""),
                    "local_risk_flags_json": json.dumps(rec.get("local_risk_flags", []), ensure_ascii=False),
                    "local_quality_bucket": rec.get("local_quality_bucket", ""),
                    "algorithm_reason": rec.get("algorithm_reason", ""),
                }
            )
    return flat


def judge_prompt(snapshot_item: dict[str, Any]) -> str:
    payload = {
        "base_product": snapshot_item["base_product"],
        "candidates": [
            {
                "rank": rec["rank"],
                "target_product": rec["target_profile"],
                "algorithm_scores": {
                    "similarity_score": rec["similarity_score"],
                    "function_similarity_score": rec["function_similarity_score"],
                    "core_match_score": rec["core_match_score"],
                    "substitutability": rec["substitutability"],
                    "recommendation_quality": rec.get("recommendation_quality", ""),
                    "recommendation_review_reason": rec.get("recommendation_review_reason", ""),
                },
                "match_evidence": {
                    "shared_ingredients": rec["shared_ingredients"],
                    "shared_categories": rec["shared_categories"],
                    "semantic_core_overlap": rec["semantic_core_overlap"],
                    "semantic_core_reason": rec["semantic_core_reason"],
                    "score_adjustments": rec["score_adjustments"],
                    "local_risk_flags": rec["local_risk_flags"],
                    "algorithm_reason": rec["algorithm_reason"],
                },
            }
            for rec in snapshot_item.get("recommendations", []) or []
        ],
    }
    return (
        "당신은 건강기능식품 제품 추천 품질을 평가하는 전문가입니다.\n"
        "주어진 base_product에 대해 candidates가 추천 결과로 합리적인지 평가하세요.\n\n"
        "판정 라벨:\n"
        "- reasonable: 같은 기능 목적군이며 핵심 기능성 원료 또는 강한 semantic 근거가 있다.\n"
        "- acceptable_adjacent: 완전 대체품은 아니지만 인접 기능군이라 비교/보조 추천으로 납득 가능하다.\n"
        "- weak: 일부 근거는 있으나 상위 추천으로는 약하다.\n"
        "- bad: 기능 목적, 핵심 원료, 카테고리가 어긋나거나 흔한 영양소/부원료만 근거다.\n\n"
        "평가 규칙:\n"
        "1. product_name만 비슷한 것은 충분한 근거가 아니다.\n"
        "2. 비타민/미네랄 같은 흔한 영양소만 겹치면 보수적으로 weak 또는 bad로 판단한다.\n"
        "3. main category가 다르더라도 sub category 또는 핵심 원료가 강하게 겹치면 acceptable_adjacent 이상이 가능하다.\n"
        "4. 치료, 질병 개선, 의약품 대체 같은 표현은 쓰지 않는다.\n"
        "5. candidates의 rank와 target_product.report_no는 반드시 그대로 반환한다.\n\n"
        "JSON만 반환하세요. 마크다운 코드블록은 쓰지 마세요.\n"
        "출력 형식:\n"
        "{\n"
        '  "base_report_no": "string",\n'
        '  "labels": [\n'
        '    {"rank": 1, "target_report_no": "string", "judgment": "reasonable|acceptable_adjacent|weak|bad", "confidence": 0.0, "reason": "80자 이내", "risk_flags": ["string"], "suggested_rule": "string"}\n'
        "  ],\n"
        '  "overall_notes": "string"\n'
        "}\n\n"
        "입력:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def openai_judge_prompt(snapshot_item: dict[str, Any]) -> str:
    candidates = [
        {
            "rank": rec["rank"],
            "target_product": rec["target_profile"],
            "algorithm_scores": {
                "similarity_score": rec["similarity_score"],
                "function_similarity_score": rec["function_similarity_score"],
                "core_match_score": rec["core_match_score"],
                "substitutability": rec["substitutability"],
                "recommendation_quality": rec.get("recommendation_quality", ""),
                "recommendation_review_reason": rec.get("recommendation_review_reason", ""),
            },
            "match_evidence": {
                "shared_ingredients": rec["shared_ingredients"],
                "shared_categories": rec["shared_categories"],
                "semantic_core_overlap": rec["semantic_core_overlap"],
                "semantic_core_reason": rec["semantic_core_reason"],
                "score_adjustments": rec["score_adjustments"],
                "local_risk_flags": rec["local_risk_flags"],
                "algorithm_reason": rec["algorithm_reason"],
            },
        }
        for rec in snapshot_item.get("recommendations", []) or []
    ]
    payload = {
        "base_product": snapshot_item["base_product"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    return (
        "You are auditing health supplement recommendation quality.\n"
        "Evaluate whether each candidate is a reasonable recommendation for the base product.\n\n"
        "Judgment labels:\n"
        "- reasonable: same functional goal and strong overlap in core functional ingredients or strong semantic evidence.\n"
        "- acceptable_adjacent: not a direct substitute, but close enough as an adjacent or supporting recommendation.\n"
        "- weak: some signal exists, but it is too weak for a top recommendation.\n"
        "- bad: functional goal, core ingredients, or category are clearly mismatched, or only generic nutrients overlap.\n\n"
        "Rules:\n"
        "1. Product name similarity alone is not sufficient evidence.\n"
        "2. Generic nutrients such as vitamins/minerals alone should be judged conservatively.\n"
        "3. If main categories differ, require strong sub-category or core ingredient overlap for acceptable_adjacent or better.\n"
        "4. Do not claim disease treatment, disease prevention, drug substitution, or medical efficacy.\n"
        f"5. Return exactly {len(candidates)} labels, one for every candidate. Do not omit any candidate.\n"
        "6. For each label, copy rank and target_product.report_no exactly from the input candidate.\n"
        "7. Return JSON only, matching the schema. No markdown.\n\n"
        "Input JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_batch_requests(snapshot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for item in snapshot_rows:
        if not item.get("recommendations"):
            continue
        requests.append(
            {
                "key": item["key"],
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": judge_prompt(item)}]}],
                    "generation_config": {
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "max_output_tokens": 2048,
                    },
                },
            }
        )
    return requests


def openai_judge_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "base_report_no": {"type": "string"},
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rank": {"type": "integer"},
                        "target_report_no": {"type": "string"},
                        "judgment": {
                            "type": "string",
                            "enum": ["reasonable", "acceptable_adjacent", "weak", "bad"],
                        },
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                        "suggested_rule": {"type": "string"},
                    },
                    "required": [
                        "rank",
                        "target_report_no",
                        "judgment",
                        "confidence",
                        "reason",
                        "risk_flags",
                        "suggested_rule",
                    ],
                },
            },
            "overall_notes": {"type": "string"},
        },
        "required": ["base_report_no", "labels", "overall_notes"],
    }


def build_openai_batch_requests(
    snapshot_rows: list[dict[str, Any]],
    *,
    model: str = OPENAI_MODEL_NAME,
    max_output_tokens: int = 4096,
) -> list[dict[str, Any]]:
    response_schema = openai_judge_response_schema()
    requests: list[dict[str, Any]] = []
    for item in snapshot_rows:
        if not item.get("recommendations"):
            continue
        requests.append(
            {
                "custom_id": item["key"],
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": openai_judge_prompt(item),
                    "reasoning": {"effort": "minimal"},
                    "max_output_tokens": max_output_tokens,
                    "text": {
                        "verbosity": "low",
                        "format": {
                            "type": "json_schema",
                            "name": "recommendation_quality_judge",
                            "schema": response_schema,
                            "strict": True,
                        },
                    },
                    "store": False,
                },
            }
        )
    return requests


def estimate_batch_cost(batch_requests: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    request_text = "\n".join(json.dumps(item, ensure_ascii=False) for item in batch_requests)
    input_tokens_estimate = math.ceil(len(request_text) / 2.7)
    output_tokens_low = len(batch_requests) * max(160, top_k * 45)
    output_tokens_high = len(batch_requests) * max(260, top_k * 75)
    input_cost = input_tokens_estimate / 1_000_000 * 0.05
    output_cost_low = output_tokens_low / 1_000_000 * 0.20
    output_cost_high = output_tokens_high / 1_000_000 * 0.20
    return {
        "model": MODEL_NAME,
        "pricing_mode": "Gemini Batch API estimate",
        "input_tokens_estimate": input_tokens_estimate,
        "output_tokens_low_estimate": output_tokens_low,
        "output_tokens_high_estimate": output_tokens_high,
        "estimated_usd_low": round(input_cost + output_cost_low, 4),
        "estimated_usd_high": round(input_cost + output_cost_high, 4),
        "pricing_assumption": "Batch input $0.05/1M tokens, output $0.20/1M tokens",
    }


def prepare(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_paths()
    ingredient_profile_path = (
        Path(args.ingredient_category_profile)
        if str(args.ingredient_category_profile or "").strip()
        else runtime.get("ingredient_category_profile_path")
    )
    similarity_algorithm = normalize_similarity_algorithm(args.similarity_algorithm)

    vector_df = load_vector_inputs(runtime["vector_csv_path"])
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    ingredient_category_profiles = load_ingredient_category_profiles(ingredient_profile_path)
    product_vectors, product_names, _report_to_product_ids, ingredient_postings, ingredient_frequency = build_vector_indexes(vector_df)

    selected_product_ids = select_product_ids(
        profiles,
        product_vectors,
        all_products=args.all,
        report_nos=args.report_no or [],
        main_categories=args.main_category or [],
        sample_size=max(0, int(args.sample_size or 0)),
        per_category=max(0, int(args.per_category or 0)),
        seed=int(args.seed),
    )
    if args.limit > 0:
        if args.offset > 0:
            selected_product_ids = selected_product_ids[args.offset :]
        selected_product_ids = selected_product_ids[: args.limit]
    elif args.offset > 0:
        selected_product_ids = selected_product_ids[args.offset :]

    snapshot_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    workers = max(1, int(args.workers or 1))
    if workers > 1 and selected_product_ids:
        print(f"[INFO] parallel_prepare workers={workers} products={len(selected_product_ids)}", flush=True)
        worker_results: list[dict[str, Any]] = []
        tasks = [
            (index, base_product_id, args.top_k, args.candidate_limit, args.max_df_for_seed)
            for index, base_product_id in enumerate(selected_product_ids, start=1)
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=prepare_worker_init,
            initargs=(similarity_algorithm, str(ingredient_profile_path or "")),
        ) as executor:
            future_to_index = {executor.submit(prepare_product_worker, task): task[0] for task in tasks}
            completed = 0
            for future in as_completed(future_to_index):
                result = future.result()
                completed += 1
                worker_results.append(result)
                failed_rows.extend(result.get("failed_rows", []) or [])
                if completed % max(1, args.progress_every) == 0 or completed == len(selected_product_ids):
                    elapsed = round(time.perf_counter() - started, 1)
                    print(
                        f"[PROGRESS] {completed}/{len(selected_product_ids)} elapsed={elapsed}s last={result.get('base_product_name', '')}",
                        flush=True,
                    )
        for result in sorted(worker_results, key=lambda item: int(item.get("index", 0) or 0)):
            if result.get("snapshot"):
                snapshot_rows.append(result["snapshot"])
    else:
        for index, base_product_id in enumerate(selected_product_ids, start=1):
            base_profile = profiles[base_product_id]
            key = f"recjudge_{index:06d}"
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    rows, failed_candidates, _stats = compute_topk_for_single_product(
                        base_product_id,
                        args.top_k,
                        args.candidate_limit,
                        args.max_df_for_seed,
                        product_vectors,
                        product_names,
                        profiles,
                        ingredient_postings,
                        ingredient_frequency,
                        similarity_algorithm,
                        ingredient_category_profiles,
                    )
                snapshot_rows.append(build_snapshot_item(key, base_product_id, rows, profiles))
                for failed in failed_candidates:
                    failed_rows.append({"key": key, "stage": "candidate_scoring", **failed})
                if index % max(1, args.progress_every) == 0 or index == len(selected_product_ids):
                    elapsed = round(time.perf_counter() - started, 1)
                    print(f"[PROGRESS] {index}/{len(selected_product_ids)} elapsed={elapsed}s last={base_profile.get('product_name', '')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed_rows.append(
                    {
                        "key": key,
                        "stage": "topk",
                        "base_product_id": base_product_id,
                        "base_report_no": str(base_profile.get("report_no", "") or ""),
                        "base_product_name": str(base_profile.get("product_name", "") or ""),
                        "error": str(exc),
                    }
                )

    batch_requests = build_batch_requests(snapshot_rows)
    openai_batch_requests = build_openai_batch_requests(snapshot_rows)
    snapshot_jsonl = output_dir / "recommendation_snapshot.jsonl"
    snapshot_csv = output_dir / "recommendation_snapshot.csv"
    batch_jsonl = output_dir / DEFAULT_BATCH_JSONL_NAME
    openai_batch_jsonl = output_dir / DEFAULT_OPENAI_BATCH_JSONL_NAME
    request_map = {
        item["key"]: {
            "base_product_id": item["base_product_id"],
            "base_report_no": item["base_product"].get("report_no", ""),
            "candidate_report_nos": [rec.get("target_profile", {}).get("report_no", "") for rec in item.get("recommendations", [])],
        }
        for item in snapshot_rows
    }

    write_jsonl(snapshot_jsonl, snapshot_rows)
    write_csv(
        snapshot_csv,
        [
            "key",
            "base_report_no",
            "base_product_name",
            "base_main_category",
            "base_sub_categories_json",
            "rank",
            "target_report_no",
            "target_product_name",
            "target_main_category",
            "target_sub_categories_json",
            "similarity_score",
            "function_similarity_score",
            "core_match_score",
            "substitutability",
            "recommendation_quality",
            "recommendation_review_reason",
            "shared_ingredients_json",
            "shared_categories_json",
            "semantic_core_overlap_json",
            "semantic_core_reason",
            "local_risk_flags_json",
            "local_quality_bucket",
            "algorithm_reason",
        ],
        flatten_snapshot(snapshot_rows),
    )
    write_jsonl(batch_jsonl, batch_requests)
    write_jsonl(openai_batch_jsonl, openai_batch_requests)
    (output_dir / "request_map.json").write_text(json.dumps(request_map, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        output_dir / "failed_products.csv",
        ["key", "stage", "base_product_id", "base_report_no", "base_product_name", "target_product_id", "target_product_name", "notes", "error"],
        failed_rows,
    )

    bucket_counts = Counter()
    risk_counts = Counter()
    category_counts = Counter()
    for item in snapshot_rows:
        category_counts[str(item["base_product"].get("product_main_category", "") or "")] += 1
        for rec in item.get("recommendations", []) or []:
            bucket_counts[str(rec.get("local_quality_bucket", "") or "")] += 1
            risk_counts.update(rec.get("local_risk_flags", []) or [])

    summary = {
        "created_at": now_iso(),
        "mode": "all" if args.all else "sample",
        "runtime": {key: str(value) for key, value in runtime.items()},
        "ingredient_category_profile_path": str(ingredient_profile_path or ""),
        "similarity_algorithm": similarity_algorithm,
        "selected_main_categories": list(args.main_category or []),
        "profile_count": len(profiles),
        "selected_product_count": len(selected_product_ids),
        "offset": args.offset,
        "limit": args.limit,
        "successful_product_count": len(snapshot_rows),
        "no_display_recommendation_product_count": sum(1 for item in snapshot_rows if not item.get("recommendations")),
        "failed_row_count": len(failed_rows),
        "top_k": args.top_k,
        "candidate_limit": args.candidate_limit,
        "max_df_for_seed": args.max_df_for_seed,
        "workers": workers,
        "base_category_counts": dict(category_counts),
        "local_quality_bucket_counts": dict(bucket_counts),
        "local_risk_flag_counts": dict(risk_counts),
        "batch_request_count": len(batch_requests),
        "batch_jsonl": str(batch_jsonl),
        "openai_batch_request_count": len(openai_batch_requests),
        "openai_batch_jsonl": str(openai_batch_jsonl),
        "openai_model": OPENAI_MODEL_NAME,
        "snapshot_jsonl": str(snapshot_jsonl),
        "snapshot_csv": str(snapshot_csv),
        "request_map": str(output_dir / "request_map.json"),
        "estimated_batch_cost": estimate_batch_cost(batch_requests, args.top_k),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def response_from_batch_line(line_obj: dict[str, Any]) -> dict[str, Any] | None:
    response = line_obj.get("response")
    if isinstance(response, dict):
        body = response.get("body")
        if isinstance(body, dict):
            return body
        return response
    inline_response = line_obj.get("inlineResponse") or line_obj.get("inline_response")
    if isinstance(inline_response, dict) and isinstance(inline_response.get("response"), dict):
        return inline_response["response"]
    if isinstance(line_obj.get("candidates"), list):
        return line_obj
    return None


def extract_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if output_text:
        return str(output_text).strip()
    output = response.get("output") or []
    output_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if text:
                output_parts.append(str(text))
    if output_parts:
        return "\n".join(output_parts).strip()
    choices = response.get("choices") or []
    choice_parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content:
            choice_parts.append(content)
    if choice_parts:
        return "\n".join(choice_parts).strip()
    candidates = response.get("candidates") or []
    parts: list[str] = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if text:
                parts.append(str(text))
    if parts:
        return "\n".join(parts).strip()
    text = response.get("text")
    return str(text or "").strip()


def parse_model_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model output root must be an object")
    return parsed


def normalize_label(label: dict[str, Any]) -> dict[str, Any]:
    judgment = str(label.get("judgment", "") or "").strip()
    if judgment not in JUDGMENTS:
        judgment = "weak"
    try:
        rank = int(label.get("rank", 0) or 0)
    except (TypeError, ValueError):
        rank = 0
    try:
        confidence = float(label.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    risk_flags = label.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = [str(risk_flags)]
    return {
        "rank": rank,
        "target_report_no": str(label.get("target_report_no", "") or "").strip(),
        "judgment": judgment,
        "confidence": round(confidence, 4),
        "reason": str(label.get("reason", "") or "").strip(),
        "risk_flags": [str(item or "").strip() for item in risk_flags if str(item or "").strip()],
        "suggested_rule": str(label.get("suggested_rule", "") or "").strip(),
    }


def snapshot_indexes(snapshot_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("key", "") or ""): item for item in snapshot_rows}


def find_recommendation(snapshot_item: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    target_report_no = str(label.get("target_report_no", "") or "")
    rank = int(label.get("rank", 0) or 0)
    recommendations = list(snapshot_item.get("recommendations", []) or [])
    if target_report_no:
        for rec in recommendations:
            if str(rec.get("target_profile", {}).get("report_no", "") or "") == target_report_no:
                return rec
    if rank:
        for rec in recommendations:
            if int(rec.get("rank", 0) or 0) == rank:
                return rec
    return {}


def apply_results(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    result_jsonl = resolve_path(args.result_jsonl) if args.result_jsonl else output_dir / DEFAULT_RESULT_JSONL_NAME
    snapshot_jsonl = resolve_path(args.snapshot_jsonl) if args.snapshot_jsonl else output_dir / "recommendation_snapshot.jsonl"
    if not result_jsonl.exists():
        raise FileNotFoundError(result_jsonl)
    if not snapshot_jsonl.exists():
        raise FileNotFoundError(snapshot_jsonl)

    snapshot_by_key = snapshot_indexes(read_jsonl(snapshot_jsonl))
    parsed_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    label_counts_by_key: Counter[str] = Counter()
    raw_rows = read_jsonl(result_jsonl)
    for line_number, line_obj in enumerate(raw_rows, start=1):
        key = str(line_obj.get("key", "") or line_obj.get("custom_id", "") or "")
        try:
            if line_obj.get("error"):
                raise ValueError(json.dumps(line_obj["error"], ensure_ascii=False))
            response = response_from_batch_line(line_obj)
            if not response:
                raise ValueError("missing response")
            text = extract_response_text(response)
            parsed = parse_model_json(text)
            labels = parsed.get("labels")
            if not isinstance(labels, list):
                raise ValueError("missing labels array")
            snapshot_item = snapshot_by_key.get(key)
            if not snapshot_item:
                raise ValueError(f"unknown key: {key}")
            label_counts_by_key[key] += len(labels)
            base = snapshot_item.get("base_product", {})
            for raw_label in labels:
                label = normalize_label(raw_label if isinstance(raw_label, dict) else {})
                rec = find_recommendation(snapshot_item, label)
                target = rec.get("target_profile", {}) if rec else {}
                parsed_rows.append(
                    {
                        "key": key,
                        "base_report_no": base.get("report_no", ""),
                        "base_product_name": base.get("product_name", ""),
                        "base_main_category": base.get("product_main_category", ""),
                        "rank": label["rank"] or rec.get("rank", ""),
                        "target_report_no": label["target_report_no"] or target.get("report_no", ""),
                        "target_product_name": target.get("product_name", ""),
                        "target_main_category": target.get("product_main_category", ""),
                        "similarity_score": rec.get("similarity_score", ""),
                        "function_similarity_score": rec.get("function_similarity_score", ""),
                        "core_match_score": rec.get("core_match_score", ""),
                        "shared_ingredients_json": json.dumps(rec.get("shared_ingredients", []), ensure_ascii=False),
                        "shared_categories_json": json.dumps(rec.get("shared_categories", []), ensure_ascii=False),
                        "local_risk_flags_json": json.dumps(rec.get("local_risk_flags", []), ensure_ascii=False),
                        "local_quality_bucket": rec.get("local_quality_bucket", ""),
                        "judge_judgment": label["judgment"],
                        "judge_confidence": label["confidence"],
                        "judge_reason": label["reason"],
                        "judge_risk_flags_json": json.dumps(label["risk_flags"], ensure_ascii=False),
                        "suggested_rule": label["suggested_rule"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            error_rows.append({"line_number": line_number, "key": key, "error": str(exc), "raw": json.dumps(line_obj, ensure_ascii=False)[:4000]})

    result_csv = output_dir / "gemini_judge_results.csv"
    result_rows_jsonl = output_dir / "gemini_judge_results.jsonl"
    errors_csv = output_dir / "gemini_judge_errors.csv"
    result_fields = [
        "key",
        "base_report_no",
        "base_product_name",
        "base_main_category",
        "rank",
        "target_report_no",
        "target_product_name",
        "target_main_category",
        "similarity_score",
        "function_similarity_score",
        "core_match_score",
        "shared_ingredients_json",
        "shared_categories_json",
        "local_risk_flags_json",
        "local_quality_bucket",
        "judge_judgment",
        "judge_confidence",
        "judge_reason",
        "judge_risk_flags_json",
        "suggested_rule",
    ]
    write_csv(result_csv, result_fields, parsed_rows)
    write_jsonl(result_rows_jsonl, parsed_rows)
    write_csv(errors_csv, ["line_number", "key", "error", "raw"], error_rows)

    judgment_counts = Counter(row["judge_judgment"] for row in parsed_rows)
    category_counts = Counter()
    suggested_rule_counts = Counter()
    high_score_bad_or_weak: list[dict[str, Any]] = []
    expected_label_counts = {
        key: len(item.get("recommendations", []) or [])
        for key, item in snapshot_by_key.items()
        if item.get("recommendations")
    }
    label_count_mismatches = [
        {
            "key": key,
            "expected": expected_count,
            "actual": int(label_counts_by_key.get(key, 0)),
        }
        for key, expected_count in sorted(expected_label_counts.items())
        if int(label_counts_by_key.get(key, 0)) != expected_count
    ]
    expected_label_count = sum(expected_label_counts.values())
    actual_label_count = sum(label_counts_by_key.values())
    missing_label_count = sum(max(0, item["expected"] - item["actual"]) for item in label_count_mismatches)
    extra_label_count = sum(max(0, item["actual"] - item["expected"]) for item in label_count_mismatches)
    for row in parsed_rows:
        category_counts[(row["base_main_category"], row["judge_judgment"])] += 1
        if row.get("suggested_rule"):
            suggested_rule_counts[row["suggested_rule"]] += 1
        try:
            score = float(row.get("similarity_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score >= args.high_score_threshold and row["judge_judgment"] in {"weak", "bad"}:
            high_score_bad_or_weak.append(row)

    summary = {
        "created_at": now_iso(),
        "result_jsonl": str(result_jsonl),
        "snapshot_jsonl": str(snapshot_jsonl),
        "result_count": len(parsed_rows),
        "error_count": len(error_rows),
        "judgment_counts": dict(judgment_counts),
        "weak_or_bad_rate": round(
            (judgment_counts.get("weak", 0) + judgment_counts.get("bad", 0)) / len(parsed_rows),
            4,
        )
        if parsed_rows
        else 0.0,
        "high_score_threshold": args.high_score_threshold,
        "high_score_weak_or_bad_count": len(high_score_bad_or_weak),
        "suggested_rule_counts": dict(suggested_rule_counts.most_common(30)),
        "category_judgment_counts": {f"{category}|{judgment}": count for (category, judgment), count in category_counts.items()},
        "outputs": {
            "results_csv": str(result_csv),
            "results_jsonl": str(result_rows_jsonl),
            "errors_csv": str(errors_csv),
        },
        "label_coverage": {
            "expected_label_count": expected_label_count,
            "actual_label_count": int(actual_label_count),
            "missing_label_count": int(missing_label_count),
            "extra_label_count": int(extra_label_count),
            "coverage_rate": round(float(actual_label_count / expected_label_count), 4) if expected_label_count else 0.0,
            "mismatch_count": len(label_count_mismatches),
            "mismatches": label_count_mismatches[:30],
        },
    }
    summary_path = output_dir / "gemini_judge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def refresh_snapshot(args: argparse.Namespace) -> None:
    source_snapshot = resolve_path(args.source_snapshot_jsonl)
    output_dir = resolve_path(args.output_dir)
    if not source_snapshot.exists():
        raise FileNotFoundError(source_snapshot)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_rows = read_jsonl(source_snapshot)
    if args.offset > 0:
        snapshot_rows = snapshot_rows[args.offset :]
    if args.limit > 0:
        snapshot_rows = snapshot_rows[: args.limit]
    bucket_counts = Counter()
    risk_counts = Counter()
    category_counts = Counter()
    for item in snapshot_rows:
        base = item.get("base_product", {})
        category_counts[str(base.get("product_main_category", "") or "")] += 1
        refreshed_recommendations: list[dict[str, Any]] = []
        for rec in item.get("recommendations", []) or []:
            quality_metadata = recommendation_quality_metadata(
                float(rec.get("similarity_score", 0.0) or 0.0),
                {"score_adjustments": rec.get("score_adjustments", []) or []},
                float(rec.get("core_match_score", 0.0) or 0.0),
                float(rec.get("function_similarity_score", 0.0) or 0.0),
            )
            rec.update(quality_metadata)
            if not bool(rec.get("recommendation_display_eligible", True)):
                continue
            rec["rank"] = len(refreshed_recommendations) + 1
            flags = local_risk_flags(base, rec)
            bucket = local_quality_bucket(flags, float(rec.get("similarity_score", 0.0) or 0.0))
            rec["local_risk_flags"] = flags
            rec["local_quality_bucket"] = bucket
            bucket_counts[bucket] += 1
            risk_counts.update(flags)
            refreshed_recommendations.append(rec)
        item["recommendations"] = refreshed_recommendations

    batch_requests = build_batch_requests(snapshot_rows)
    openai_batch_requests = build_openai_batch_requests(snapshot_rows)
    snapshot_jsonl = output_dir / "recommendation_snapshot.jsonl"
    snapshot_csv = output_dir / "recommendation_snapshot.csv"
    batch_jsonl = output_dir / DEFAULT_BATCH_JSONL_NAME
    openai_batch_jsonl = output_dir / DEFAULT_OPENAI_BATCH_JSONL_NAME
    request_map = {
        item["key"]: {
            "base_product_id": item.get("base_product_id", ""),
            "base_report_no": item.get("base_product", {}).get("report_no", ""),
            "candidate_report_nos": [
                rec.get("target_profile", {}).get("report_no", "")
                for rec in item.get("recommendations", []) or []
            ],
        }
        for item in snapshot_rows
    }

    write_jsonl(snapshot_jsonl, snapshot_rows)
    write_csv(
        snapshot_csv,
        [
            "key",
            "base_report_no",
            "base_product_name",
            "base_main_category",
            "base_sub_categories_json",
            "rank",
            "target_report_no",
            "target_product_name",
            "target_main_category",
            "target_sub_categories_json",
            "similarity_score",
            "function_similarity_score",
        "core_match_score",
        "substitutability",
        "recommendation_quality",
        "recommendation_review_reason",
        "shared_ingredients_json",
        "shared_categories_json",
            "semantic_core_overlap_json",
            "semantic_core_reason",
            "local_risk_flags_json",
            "local_quality_bucket",
            "algorithm_reason",
        ],
        flatten_snapshot(snapshot_rows),
    )
    write_jsonl(batch_jsonl, batch_requests)
    write_jsonl(openai_batch_jsonl, openai_batch_requests)
    (output_dir / "request_map.json").write_text(json.dumps(request_map, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "created_at": now_iso(),
        "source_snapshot_jsonl": str(source_snapshot),
        "successful_product_count": len(snapshot_rows),
        "no_display_recommendation_product_count": sum(1 for item in snapshot_rows if not item.get("recommendations")),
        "recommendation_pair_count": sum(len(item.get("recommendations", []) or []) for item in snapshot_rows),
        "base_category_counts": dict(category_counts),
        "local_quality_bucket_counts": dict(bucket_counts),
        "local_risk_flag_counts": dict(risk_counts),
        "batch_request_count": len(batch_requests),
        "batch_jsonl": str(batch_jsonl),
        "openai_batch_request_count": len(openai_batch_requests),
        "openai_batch_jsonl": str(openai_batch_jsonl),
        "openai_model": OPENAI_MODEL_NAME,
        "snapshot_jsonl": str(snapshot_jsonl),
        "snapshot_csv": str(snapshot_csv),
        "request_map": str(output_dir / "request_map.json"),
        "estimated_batch_cost": estimate_batch_cost(batch_requests, int(args.top_k)),
    }
    summary_path = output_dir / "refresh_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def score_impact(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    judge_csv = resolve_path(args.judge_csv) if args.judge_csv else output_dir / "gemini_judge_results.csv"
    if not judge_csv.exists():
        raise FileNotFoundError(judge_csv)

    runtime = resolve_runtime_paths()
    ingredient_profile_path = (
        Path(args.ingredient_category_profile)
        if str(args.ingredient_category_profile or "").strip()
        else runtime.get("ingredient_category_profile_path")
    )
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    ingredient_category_profiles = load_ingredient_category_profiles(ingredient_profile_path)
    profile_by_report_no = {
        str(profile.get("report_no", "") or ""): profile
        for profile in profiles.values()
        if str(profile.get("report_no", "") or "")
    }

    df = pd.read_csv(judge_csv, encoding="utf-8-sig", low_memory=False).fillna("")
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    adjustment_counts: Counter[str] = Counter()
    high_threshold = float(args.high_score_threshold)

    for source in df.to_dict(orient="records"):
        base_report_no = str(source.get("base_report_no", "") or "").strip()
        target_report_no = str(source.get("target_report_no", "") or "").strip()
        base_profile = profile_by_report_no.get(base_report_no)
        target_profile = profile_by_report_no.get(target_report_no)
        if not base_profile or not target_profile:
            missing_rows.append(
                {
                    "base_report_no": base_report_no,
                    "target_report_no": target_report_no,
                    "reason": "missing_profile",
                }
            )
            continue

        old_score = float(source.get("similarity_score", 0.0) or 0.0)
        judgment = str(source.get("judge_judgment", "") or "")
        new_score, shared, detail = calculate_semantic_weighted_jaccard_v2(
            base_profile,
            target_profile,
            ingredient_category_profiles,
        )
        adjustments = list(detail.get("score_adjustments", []) or [])
        adjustment_types = [str(item.get("type", "") or "") for item in adjustments if str(item.get("type", "") or "")]
        adjustment_counts.update(adjustment_types)
        old_high_weak_bad = judgment in {"weak", "bad"} and old_score >= high_threshold
        new_high_weak_bad = judgment in {"weak", "bad"} and float(new_score) >= high_threshold
        rows.append(
            {
                "base_report_no": base_report_no,
                "base_product_name": str(source.get("base_product_name", "") or ""),
                "base_main_category": str(source.get("base_main_category", "") or ""),
                "target_report_no": target_report_no,
                "target_product_name": str(source.get("target_product_name", "") or ""),
                "target_main_category": str(source.get("target_main_category", "") or ""),
                "rank": source.get("rank", ""),
                "judge_judgment": judgment,
                "judge_confidence": source.get("judge_confidence", ""),
                "judge_reason": str(source.get("judge_reason", "") or ""),
                "old_similarity_score": round(old_score, 6),
                "new_similarity_score": round(float(new_score), 6),
                "score_delta": round(float(new_score) - old_score, 6),
                "old_high_weak_or_bad": int(old_high_weak_bad),
                "new_high_weak_or_bad": int(new_high_weak_bad),
                "reduced_below_high_threshold": int(old_high_weak_bad and not new_high_weak_bad),
                "shared_after_json": json.dumps(shared, ensure_ascii=False),
                "score_adjustment_types_json": json.dumps(adjustment_types, ensure_ascii=False),
                "score_adjustments_json": json.dumps(adjustments, ensure_ascii=False),
            }
        )

    result_path = output_dir / "post_fix_score_impact.csv"
    missing_path = output_dir / "post_fix_score_impact_missing.csv"
    fieldnames = [
        "base_report_no",
        "base_product_name",
        "base_main_category",
        "target_report_no",
        "target_product_name",
        "target_main_category",
        "rank",
        "judge_judgment",
        "judge_confidence",
        "judge_reason",
        "old_similarity_score",
        "new_similarity_score",
        "score_delta",
        "old_high_weak_or_bad",
        "new_high_weak_or_bad",
        "reduced_below_high_threshold",
        "shared_after_json",
        "score_adjustment_types_json",
        "score_adjustments_json",
    ]
    write_csv(result_path, fieldnames, rows)
    write_csv(missing_path, ["base_report_no", "target_report_no", "reason"], missing_rows)

    judgment_counts = Counter(row["judge_judgment"] for row in rows)
    old_high_weak_bad = sum(int(row["old_high_weak_or_bad"]) for row in rows)
    new_high_weak_bad = sum(int(row["new_high_weak_or_bad"]) for row in rows)
    reduced_count = sum(int(row["reduced_below_high_threshold"]) for row in rows)
    score_changed_count = sum(1 for row in rows if abs(float(row["score_delta"])) > 0.000001)
    summary = {
        "created_at": now_iso(),
        "judge_csv": str(judge_csv),
        "ingredient_category_profile_path": str(ingredient_profile_path or ""),
        "row_count": len(rows),
        "missing_count": len(missing_rows),
        "judgment_counts": dict(judgment_counts),
        "high_score_threshold": high_threshold,
        "old_high_score_weak_or_bad_count": old_high_weak_bad,
        "new_high_score_weak_or_bad_count": new_high_weak_bad,
        "reduced_below_high_threshold_count": reduced_count,
        "score_changed_count": score_changed_count,
        "adjustment_counts": dict(adjustment_counts.most_common(30)),
        "outputs": {
            "impact_csv": str(result_path),
            "missing_csv": str(missing_path),
        },
    }
    summary_path = output_dir / "post_fix_score_impact_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def merge_parts(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    parts_glob = Path(args.parts_glob)
    if parts_glob.is_absolute():
        part_dirs = sorted(parts_glob.parent.glob(parts_glob.name))
    else:
        part_dirs = sorted(resolve_path(".").glob(args.parts_glob))
    if not part_dirs:
        raise FileNotFoundError(f"no part dirs matched: {args.parts_glob}")

    result_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for part_dir in part_dirs:
        result_csv = part_dir / "gemini_judge_results.csv"
        error_csv = part_dir / "gemini_judge_errors.csv"
        summary_json = part_dir / "gemini_judge_summary.json"
        if result_csv.exists():
            frame = pd.read_csv(result_csv, encoding="utf-8-sig", low_memory=False).fillna("")
            frame.insert(0, "part_dir", str(part_dir))
            result_frames.append(frame)
        if error_csv.exists():
            errors = pd.read_csv(error_csv, encoding="utf-8-sig", low_memory=False).fillna("")
            if not errors.empty:
                errors.insert(0, "part_dir", str(part_dir))
                error_frames.append(errors)
        if summary_json.exists():
            summaries.append(json.loads(summary_json.read_text(encoding="utf-8")))

    merged = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    all_errors_merged = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    errors_merged = all_errors_merged
    retry_covered_error_count = 0
    if not all_errors_merged.empty and "key" in all_errors_merged.columns and "key" in merged.columns:
        successful_keys = {str(value) for value in merged["key"].tolist() if str(value)}
        covered_mask = all_errors_merged["key"].astype(str).isin(successful_keys)
        retry_covered_error_count = int(covered_mask.sum())
        errors_merged = all_errors_merged.loc[~covered_mask].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = output_dir / "merged_gemini_judge_results.csv"
    errors_csv = output_dir / "merged_gemini_judge_errors.csv"
    all_errors_csv = output_dir / "merged_gemini_judge_errors_all.csv"
    merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    errors_merged.to_csv(errors_csv, index=False, encoding="utf-8-sig")
    all_errors_merged.to_csv(all_errors_csv, index=False, encoding="utf-8-sig")

    judgment_counts = Counter(str(value) for value in merged.get("judge_judgment", []) if str(value))
    category_counts = Counter()
    if not merged.empty and {"base_main_category", "judge_judgment"}.issubset(set(merged.columns)):
        for row in merged[["base_main_category", "judge_judgment"]].to_dict(orient="records"):
            category_counts[(str(row["base_main_category"]), str(row["judge_judgment"]))] += 1
    weak_or_bad_count = judgment_counts.get("weak", 0) + judgment_counts.get("bad", 0)
    summary = {
        "created_at": now_iso(),
        "parts_glob": args.parts_glob,
        "part_count": len(part_dirs),
        "parts": [str(path) for path in part_dirs],
        "result_count": int(len(merged)),
        "error_count": int(len(errors_merged)),
        "raw_error_count": int(len(all_errors_merged)),
        "retry_covered_error_count": retry_covered_error_count,
        "part_summaries_error_count": sum(int(item.get("error_count", 0) or 0) for item in summaries),
        "judgment_counts": dict(judgment_counts),
        "weak_or_bad_rate": round(float(weak_or_bad_count / len(merged)), 4) if len(merged) else 0.0,
        "high_score_weak_or_bad_count": int(
            len(
                merged[
                    merged.get("judge_judgment", pd.Series(dtype=str)).isin(["weak", "bad"])
                    & (pd.to_numeric(merged.get("similarity_score", pd.Series(dtype=float)), errors="coerce") >= float(args.high_score_threshold))
                ]
            )
        )
        if not merged.empty
        else 0,
        "category_judgment_counts": {f"{category}|{judgment}": count for (category, judgment), count in category_counts.items()},
        "outputs": {
            "merged_results_csv": str(merged_csv),
            "merged_errors_csv": str(errors_csv),
            "merged_all_errors_csv": str(all_errors_csv),
        },
    }
    summary_path = output_dir / "merged_gemini_judge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


OPENAI_CHUNK_RESULT_FIELDS = [
    "chunk",
    "part_dir",
    "source_csv",
    "current_score_source",
    "is_retry",
    "key",
    "base_report_no",
    "base_product_name",
    "base_main_category",
    "rank",
    "target_report_no",
    "target_product_name",
    "target_main_category",
    "similarity_score",
    "old_similarity_score",
    "score_delta",
    "function_similarity_score",
    "core_match_score",
    "shared_ingredients_json",
    "shared_categories_json",
    "shared_after_json",
    "local_risk_flags_json",
    "local_quality_bucket",
    "score_adjustment_types_json",
    "score_adjustments_json",
    "judge_judgment",
    "judge_confidence",
    "judge_reason",
    "judge_risk_flags_json",
    "suggested_rule",
]


def coerce_patterns(patterns: Any) -> list[str]:
    if not patterns:
        return []
    if isinstance(patterns, str):
        return [patterns]
    return [str(pattern) for pattern in patterns if str(pattern or "").strip()]


def glob_directories(patterns: Any) -> list[Path]:
    matched: list[Path] = []
    for pattern in coerce_patterns(patterns):
        path_pattern = Path(pattern)
        if path_pattern.is_absolute():
            candidates = path_pattern.parent.glob(path_pattern.name)
        else:
            candidates = ROOT_DIR.glob(pattern.replace("\\", "/"))
        matched.extend(path for path in candidates if path.is_dir())
    deduped = {str(path.resolve()): path for path in matched}
    return [deduped[key] for key in sorted(deduped)]


def chunk_label_from_dir(path: Path) -> str:
    name = path.name
    match = re.search(r"openai_chunk_(\d+_\d+)(?:_retry_.+)?$", name)
    if match:
        return match.group(1)
    if "_retry_" in name:
        return name.split("_retry_", 1)[0]
    return name


def snapshot_expected_counts(part_dir: Path) -> dict[str, Any]:
    snapshot_jsonl = part_dir / "recommendation_snapshot.jsonl"
    if not snapshot_jsonl.exists():
        raise FileNotFoundError(snapshot_jsonl)

    expected_by_base: dict[str, int] = {}
    request_count = 0
    expected_label_count = 0
    for item in read_jsonl(snapshot_jsonl):
        base = item.get("base_product", {}) if isinstance(item, dict) else {}
        base_report_no = str(base.get("report_no", "") or "").strip()
        recommendations = item.get("recommendations", []) if isinstance(item, dict) else []
        recommendation_count = len(recommendations or [])
        if recommendation_count:
            request_count += 1
            expected_label_count += recommendation_count
        if base_report_no:
            expected_by_base[base_report_no] = recommendation_count
    return {
        "product_count": len(expected_by_base),
        "request_count": request_count,
        "expected_label_count": expected_label_count,
        "expected_by_base": expected_by_base,
        "base_report_nos": set(expected_by_base),
    }


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False).fillna("")
    return [dict(row) for row in frame.to_dict(orient="records")]


def current_score_rows_for_part(part_dir: Path, *, chunk: str, is_retry: bool) -> list[dict[str, Any]]:
    impact_csv = part_dir / "post_fix_score_impact.csv"
    result_csv = part_dir / "gemini_judge_results.csv"
    if impact_csv.exists():
        source_csv = impact_csv
        source = "post_fix_score_impact"
        records = read_csv_records(impact_csv)
    elif result_csv.exists():
        source_csv = result_csv
        source = "gemini_judge_results"
        records = read_csv_records(result_csv)
    else:
        raise FileNotFoundError(f"missing post_fix_score_impact.csv or gemini_judge_results.csv in {part_dir}")

    rows: list[dict[str, Any]] = []
    for record in records:
        if source == "post_fix_score_impact":
            similarity_score = record.get("new_similarity_score", "")
            old_similarity_score = record.get("old_similarity_score", "")
            score_delta = record.get("score_delta", "")
        else:
            similarity_score = record.get("similarity_score", "")
            old_similarity_score = record.get("old_similarity_score", "")
            score_delta = record.get("score_delta", "")

        row = {
            "chunk": chunk,
            "part_dir": str(part_dir),
            "source_csv": str(source_csv),
            "current_score_source": source,
            "is_retry": int(is_retry),
            "key": record.get("key", ""),
            "base_report_no": str(record.get("base_report_no", "") or "").strip(),
            "base_product_name": record.get("base_product_name", ""),
            "base_main_category": record.get("base_main_category", ""),
            "rank": record.get("rank", ""),
            "target_report_no": str(record.get("target_report_no", "") or "").strip(),
            "target_product_name": record.get("target_product_name", ""),
            "target_main_category": record.get("target_main_category", ""),
            "similarity_score": similarity_score,
            "old_similarity_score": old_similarity_score,
            "score_delta": score_delta,
            "function_similarity_score": record.get("function_similarity_score", ""),
            "core_match_score": record.get("core_match_score", ""),
            "shared_ingredients_json": record.get("shared_ingredients_json", ""),
            "shared_categories_json": record.get("shared_categories_json", ""),
            "shared_after_json": record.get("shared_after_json", ""),
            "local_risk_flags_json": record.get("local_risk_flags_json", ""),
            "local_quality_bucket": record.get("local_quality_bucket", ""),
            "score_adjustment_types_json": record.get("score_adjustment_types_json", ""),
            "score_adjustments_json": record.get("score_adjustments_json", ""),
            "judge_judgment": record.get("judge_judgment", ""),
            "judge_confidence": record.get("judge_confidence", ""),
            "judge_reason": record.get("judge_reason", ""),
            "judge_risk_flags_json": record.get("judge_risk_flags_json", ""),
            "suggested_rule": record.get("suggested_rule", ""),
        }
        rows.append(row)
    return rows


def score_as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ordered_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    present = set()
    for row in rows:
        present.update(row.keys())
    return preferred + sorted(present - set(preferred))


def summarize_chunks(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    part_dirs = [path for path in glob_directories(args.parts_glob) if "_retry_" not in path.name]
    retry_dirs = glob_directories(getattr(args, "retry_glob", []))
    if not part_dirs:
        raise FileNotFoundError(f"no part dirs matched: {coerce_patterns(args.parts_glob)}")

    rows: list[dict[str, Any]] = []
    expected_by_chunk_base: dict[tuple[str, str], int] = {}
    part_summaries: list[dict[str, Any]] = []
    product_count = 0
    request_count = 0
    expected_label_count = 0
    for part_dir in part_dirs:
        chunk = chunk_label_from_dir(part_dir)
        expected = snapshot_expected_counts(part_dir)
        product_count += int(expected["product_count"])
        request_count += int(expected["request_count"])
        expected_label_count += int(expected["expected_label_count"])
        for base_report_no, label_count in expected["expected_by_base"].items():
            expected_by_chunk_base[(chunk, base_report_no)] = int(label_count)
        part_rows = current_score_rows_for_part(part_dir, chunk=chunk, is_retry=False)
        rows.extend(part_rows)
        part_summaries.append(
            {
                "chunk": chunk,
                "part_dir": str(part_dir),
                "product_count": int(expected["product_count"]),
                "request_count": int(expected["request_count"]),
                "expected_label_count": int(expected["expected_label_count"]),
                "actual_label_count": len(part_rows),
            }
        )

    retry_replacements: list[dict[str, Any]] = []
    for retry_dir in retry_dirs:
        parent_chunk = chunk_label_from_dir(retry_dir)
        retry_expected = snapshot_expected_counts(retry_dir)
        retry_base_report_nos = set(retry_expected["base_report_nos"])
        retry_rows = current_score_rows_for_part(retry_dir, chunk=parent_chunk, is_retry=True)
        if not retry_base_report_nos:
            retry_base_report_nos = {row["base_report_no"] for row in retry_rows if row.get("base_report_no")}

        before_count = len(rows)
        rows = [
            row
            for row in rows
            if not (row.get("chunk") == parent_chunk and str(row.get("base_report_no", "") or "") in retry_base_report_nos)
        ]
        removed_count = before_count - len(rows)
        rows.extend(retry_rows)
        retry_replacements.append(
            {
                "retry_dir": str(retry_dir),
                "parent_chunk": parent_chunk,
                "base_report_nos": sorted(retry_base_report_nos),
                "removed_label_count": removed_count,
                "retry_label_count": len(retry_rows),
            }
        )

    judgment_counts = Counter(str(row.get("judge_judgment", "") or "") for row in rows if str(row.get("judge_judgment", "") or ""))
    category_counts = Counter(
        (str(row.get("base_main_category", "") or ""), str(row.get("judge_judgment", "") or ""))
        for row in rows
        if str(row.get("judge_judgment", "") or "")
    )
    high_threshold = float(args.high_score_threshold)
    high_score_weak_or_bad = [
        row
        for row in rows
        if str(row.get("judge_judgment", "") or "") in {"weak", "bad"} and score_as_float(row.get("similarity_score")) >= high_threshold
    ]

    actual_by_chunk_base = Counter(
        (str(row.get("chunk", "") or ""), str(row.get("base_report_no", "") or ""))
        for row in rows
        if str(row.get("base_report_no", "") or "")
    )
    coverage_mismatches = [
        {
            "chunk": chunk,
            "base_report_no": base_report_no,
            "expected": expected_count,
            "actual": int(actual_by_chunk_base.get((chunk, base_report_no), 0)),
        }
        for (chunk, base_report_no), expected_count in sorted(expected_by_chunk_base.items())
        if int(actual_by_chunk_base.get((chunk, base_report_no), 0)) != expected_count
    ]
    actual_label_count = len(rows)
    missing_label_count = sum(max(0, int(item["expected"]) - int(item["actual"])) for item in coverage_mismatches)
    extra_label_count = sum(max(0, int(item["actual"]) - int(item["expected"])) for item in coverage_mismatches)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "openai_chunk_judge_results.csv"
    high_score_csv = output_dir / "openai_chunk_high_score_weak_or_bad.csv"
    coverage_csv = output_dir / "openai_chunk_label_coverage_mismatches.csv"
    result_fields = ordered_fields(rows, OPENAI_CHUNK_RESULT_FIELDS)
    write_csv(results_csv, result_fields, rows)
    write_csv(high_score_csv, result_fields, high_score_weak_or_bad)
    write_csv(coverage_csv, ["chunk", "base_report_no", "expected", "actual"], coverage_mismatches)

    weak_or_bad_count = judgment_counts.get("weak", 0) + judgment_counts.get("bad", 0)
    summary = {
        "created_at": now_iso(),
        "parts_glob": coerce_patterns(args.parts_glob),
        "retry_glob": coerce_patterns(getattr(args, "retry_glob", [])),
        "part_count": len(part_dirs),
        "retry_part_count": len(retry_dirs),
        "product_count": product_count,
        "request_count": request_count,
        "expected_label_count": expected_label_count,
        "actual_label_count": actual_label_count,
        "coverage_ok": expected_label_count == actual_label_count and not coverage_mismatches,
        "missing_label_count": int(missing_label_count),
        "extra_label_count": int(extra_label_count),
        "coverage_mismatch_count": len(coverage_mismatches),
        "coverage_mismatches": coverage_mismatches[:50],
        "judgment_counts": dict(judgment_counts),
        "weak_or_bad_rate": round(float(weak_or_bad_count / actual_label_count), 4) if actual_label_count else 0.0,
        "high_score_threshold": high_threshold,
        "current_high_score_weak_or_bad_count": len(high_score_weak_or_bad),
        "current_high_score_weak_or_bad_rate": round(float(len(high_score_weak_or_bad) / actual_label_count), 4)
        if actual_label_count
        else 0.0,
        "category_judgment_counts": {f"{category}|{judgment}": count for (category, judgment), count in category_counts.items()},
        "parts": part_summaries,
        "retry_replacements": retry_replacements,
        "outputs": {
            "results_csv": str(results_csv),
            "high_score_weak_or_bad_csv": str(high_score_csv),
            "coverage_mismatches_csv": str(coverage_csv),
        },
    }
    summary_path = output_dir / "openai_chunk_judge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


PATTERN_FEATURE_FIELDS = [
    "base_report_no",
    "base_product_name",
    "base_main_category",
    "target_report_no",
    "target_product_name",
    "target_main_category",
    "rank",
    "judge_judgment",
    "judge_confidence",
    "judge_reason",
    "current_similarity_score",
    "source_similarity_score",
    "function_similarity_score",
    "shared_count",
    "shared_core_count",
    "same_primary_set",
    "primary_primary_overlap_count",
    "base_primary_count",
    "target_primary_count",
    "base_primary_in_target_secondary_count",
    "target_primary_in_base_secondary_count",
    "no_primary_primary_overlap_cross_role",
    "base_only_semantic_count",
    "target_only_semantic_count",
    "shared_labels_json",
    "shared_core_keys_json",
    "score_adjustment_types_json",
]


def ingredient_name_set(values: Any) -> set[str]:
    return {str(value or "").strip() for value in values or [] if str(value or "").strip()}


def json_list(value: Any) -> str:
    return json.dumps(list(value or []), ensure_ascii=False)


def pattern_features_for_row(
    row: dict[str, Any],
    base_profile: dict[str, Any],
    target_profile: dict[str, Any],
    ingredient_category_profiles: dict[str, dict],
) -> dict[str, Any]:
    current_score, shared_labels, detail = calculate_semantic_weighted_jaccard_v2(
        base_profile,
        target_profile,
        ingredient_category_profiles,
    )
    core_coverage = detail.get("core_coverage", {}) or {}
    shared_core_keys = list(core_coverage.get("shared_core_semantic_keys", []) or [])
    adjustment_types = [
        str(item.get("type", "") or "")
        for item in detail.get("score_adjustments", []) or []
        if str(item.get("type", "") or "")
    ]
    base_primary = ingredient_name_set(base_profile.get("primary_ingredients"))
    base_secondary = ingredient_name_set(base_profile.get("secondary_ingredients"))
    target_primary = ingredient_name_set(target_profile.get("primary_ingredients"))
    target_secondary = ingredient_name_set(target_profile.get("secondary_ingredients"))
    primary_primary_overlap = sorted(base_primary & target_primary)
    base_primary_in_target_secondary = sorted(base_primary & target_secondary)
    target_primary_in_base_secondary = sorted(target_primary & base_secondary)
    function_similarity = score_as_float(row.get("function_similarity_score"))
    if function_similarity <= 0.0:
        function_similarity = calculate_function_similarity(base_profile, target_profile)

    return {
        "base_report_no": str(row.get("base_report_no", "") or "").strip(),
        "base_product_name": row.get("base_product_name", base_profile.get("product_name", "")),
        "base_main_category": row.get("base_main_category", base_profile.get("product_main_category", "")),
        "target_report_no": str(row.get("target_report_no", "") or "").strip(),
        "target_product_name": row.get("target_product_name", target_profile.get("product_name", "")),
        "target_main_category": row.get("target_main_category", target_profile.get("product_main_category", "")),
        "rank": row.get("rank", ""),
        "judge_judgment": row.get("judge_judgment", ""),
        "judge_confidence": row.get("judge_confidence", ""),
        "judge_reason": row.get("judge_reason", ""),
        "current_similarity_score": round(float(current_score), 6),
        "source_similarity_score": row.get("similarity_score", ""),
        "function_similarity_score": round(float(function_similarity), 6),
        "shared_count": len(shared_labels),
        "shared_core_count": len(shared_core_keys),
        "same_primary_set": int(bool(base_primary and target_primary and base_primary == target_primary)),
        "primary_primary_overlap_count": len(primary_primary_overlap),
        "base_primary_count": len(base_primary),
        "target_primary_count": len(target_primary),
        "base_primary_in_target_secondary_count": len(base_primary_in_target_secondary),
        "target_primary_in_base_secondary_count": len(target_primary_in_base_secondary),
        "no_primary_primary_overlap_cross_role": int(
            bool(
                base_primary
                and target_primary
                and not primary_primary_overlap
                and (base_primary_in_target_secondary or target_primary_in_base_secondary)
            )
        ),
        "base_only_semantic_count": max(0, int(detail.get("base_semantic_ingredient_count", 0) or 0) - len(shared_labels)),
        "target_only_semantic_count": max(0, int(detail.get("target_semantic_ingredient_count", 0) or 0) - len(shared_labels)),
        "shared_labels_json": json_list(shared_labels),
        "shared_core_keys_json": json_list(shared_core_keys),
        "score_adjustment_types_json": json_list(adjustment_types),
    }


def bool_feature(row: dict[str, Any], key: str) -> bool:
    return bool(int(row.get(key, 0) or 0))


def pattern_rule_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "no_primary_primary_overlap_cross_role_shared_le2",
            "description": "No primary-primary overlap, but a primary ingredient appears only as the other product's secondary ingredient; shared ingredients <= 2.",
            "predicate": lambda row: bool_feature(row, "no_primary_primary_overlap_cross_role") and int(row.get("shared_count", 0) or 0) <= 2,
        },
        {
            "name": "no_primary_primary_overlap_cross_role_shared_le1",
            "description": "Stricter cross-role primary signal with shared ingredients <= 1.",
            "predicate": lambda row: bool_feature(row, "no_primary_primary_overlap_cross_role") and int(row.get("shared_count", 0) or 0) <= 1,
        },
        {
            "name": "single_core_shared_le2",
            "description": "Only one shared semantic core and shared ingredients <= 2.",
            "predicate": lambda row: int(row.get("shared_core_count", 0) or 0) == 1 and int(row.get("shared_count", 0) or 0) <= 2,
        },
        {
            "name": "single_core_shared_le3_low_function",
            "description": "Only one shared semantic core, shared ingredients <= 3, and function similarity < 0.40.",
            "predicate": lambda row: int(row.get("shared_core_count", 0) or 0) == 1
            and int(row.get("shared_count", 0) or 0) <= 3
            and score_as_float(row.get("function_similarity_score")) < 0.40,
        },
        {
            "name": "single_core_base_only_ge3",
            "description": "Only one shared semantic core and base has at least 3 unshared semantic ingredients.",
            "predicate": lambda row: int(row.get("shared_core_count", 0) or 0) == 1
            and int(row.get("base_only_semantic_count", 0) or 0) >= 3,
        },
        {
            "name": "high_score_low_function",
            "description": "High current score but function similarity < 0.40.",
            "predicate": lambda row: score_as_float(row.get("function_similarity_score")) < 0.40,
        },
        {
            "name": "same_primary_set",
            "description": "Base and target have the same primary ingredient set; useful as a control group.",
            "predicate": lambda row: bool_feature(row, "same_primary_set"),
        },
    ]


def pattern_impact_rows(feature_rows: list[dict[str, Any]], high_score_threshold: float) -> list[dict[str, Any]]:
    high_rows = [row for row in feature_rows if score_as_float(row.get("current_similarity_score")) >= high_score_threshold]
    rows: list[dict[str, Any]] = []
    for rule in pattern_rule_definitions():
        matched = [row for row in high_rows if rule["predicate"](row)]
        judgment_counts = Counter(str(row.get("judge_judgment", "") or "") for row in matched)
        weak_or_bad_count = judgment_counts.get("weak", 0) + judgment_counts.get("bad", 0)
        rows.append(
            {
                "pattern": rule["name"],
                "description": rule["description"],
                "matched_count": len(matched),
                "reasonable_count": judgment_counts.get("reasonable", 0),
                "acceptable_adjacent_count": judgment_counts.get("acceptable_adjacent", 0),
                "weak_count": judgment_counts.get("weak", 0),
                "bad_count": judgment_counts.get("bad", 0),
                "weak_or_bad_count": weak_or_bad_count,
                "weak_or_bad_rate": round(float(weak_or_bad_count / len(matched)), 4) if matched else 0.0,
                "non_weak_affected_count": len(matched) - weak_or_bad_count,
            }
        )
    return rows


def analyze_patterns(args: argparse.Namespace) -> None:
    results_csv = resolve_path(args.results_csv)
    output_dir = resolve_path(args.output_dir)
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    runtime = resolve_runtime_paths()
    ingredient_profile_path = (
        Path(args.ingredient_category_profile)
        if str(args.ingredient_category_profile or "").strip()
        else runtime.get("ingredient_category_profile_path")
    )
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    ingredient_category_profiles = load_ingredient_category_profiles(ingredient_profile_path)
    profile_by_report_no = {
        str(profile.get("report_no", "") or ""): profile
        for profile in profiles.values()
        if str(profile.get("report_no", "") or "")
    }

    feature_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in read_csv_records(results_csv):
        base_report_no = str(row.get("base_report_no", "") or "").strip()
        target_report_no = str(row.get("target_report_no", "") or "").strip()
        base_profile = profile_by_report_no.get(base_report_no)
        target_profile = profile_by_report_no.get(target_report_no)
        if not base_profile or not target_profile:
            missing_rows.append(
                {
                    "base_report_no": base_report_no,
                    "target_report_no": target_report_no,
                    "reason": "missing_profile",
                }
            )
            continue
        feature_rows.append(pattern_features_for_row(row, base_profile, target_profile, ingredient_category_profiles))

    high_score_threshold = float(args.high_score_threshold)
    high_score_weak_rows = [
        row
        for row in feature_rows
        if str(row.get("judge_judgment", "") or "") in {"weak", "bad"}
        and score_as_float(row.get("current_similarity_score")) >= high_score_threshold
    ]
    impact_rows = pattern_impact_rows(feature_rows, high_score_threshold)
    judgment_counts = Counter(str(row.get("judge_judgment", "") or "") for row in feature_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    features_csv = output_dir / "judge_pattern_features.csv"
    high_score_csv = output_dir / "judge_high_score_weak_patterns.csv"
    impact_csv = output_dir / "judge_pattern_impact.csv"
    missing_csv = output_dir / "judge_pattern_missing_profiles.csv"
    write_csv(features_csv, ordered_fields(feature_rows, PATTERN_FEATURE_FIELDS), feature_rows)
    write_csv(high_score_csv, ordered_fields(high_score_weak_rows, PATTERN_FEATURE_FIELDS), high_score_weak_rows)
    write_csv(
        impact_csv,
        [
            "pattern",
            "description",
            "matched_count",
            "reasonable_count",
            "acceptable_adjacent_count",
            "weak_count",
            "bad_count",
            "weak_or_bad_count",
            "weak_or_bad_rate",
            "non_weak_affected_count",
        ],
        impact_rows,
    )
    write_csv(missing_csv, ["base_report_no", "target_report_no", "reason"], missing_rows)

    weak_or_bad_count = judgment_counts.get("weak", 0) + judgment_counts.get("bad", 0)
    summary = {
        "created_at": now_iso(),
        "results_csv": str(results_csv),
        "ingredient_category_profile_path": str(ingredient_profile_path or ""),
        "row_count": len(feature_rows),
        "missing_count": len(missing_rows),
        "judgment_counts": dict(judgment_counts),
        "weak_or_bad_rate": round(float(weak_or_bad_count / len(feature_rows)), 4) if feature_rows else 0.0,
        "high_score_threshold": high_score_threshold,
        "high_score_weak_or_bad_count": len(high_score_weak_rows),
        "pattern_impacts": impact_rows,
        "outputs": {
            "features_csv": str(features_csv),
            "high_score_weak_patterns_csv": str(high_score_csv),
            "pattern_impact_csv": str(impact_csv),
            "missing_profiles_csv": str(missing_csv),
        },
    }
    summary_path = output_dir / "judge_pattern_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def quality_gate_decision(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    row_count = int(summary.get("row_count", 0) or 0)
    weak_or_bad_rate = float(summary.get("weak_or_bad_rate", 0.0) or 0.0)
    high_score_weak_or_bad_count = int(summary.get("high_score_weak_or_bad_count", 0) or 0)
    high_score_weak_or_bad_rate = (
        float(high_score_weak_or_bad_count / row_count)
        if row_count
        else 0.0
    )
    actionable_patterns = []
    for pattern in summary.get("pattern_impacts", []) or []:
        if not isinstance(pattern, dict):
            continue
        weak_count = int(pattern.get("weak_or_bad_count", 0) or 0)
        weak_rate = float(pattern.get("weak_or_bad_rate", 0.0) or 0.0)
        non_weak_affected = int(pattern.get("non_weak_affected_count", 0) or 0)
        if (
            weak_count >= int(args.min_actionable_pattern_weak_count)
            and weak_rate >= float(args.min_actionable_pattern_weak_rate)
            and non_weak_affected <= int(args.max_actionable_pattern_non_weak_affected)
        ):
            actionable_patterns.append(pattern)

    reasons: list[str] = []
    if actionable_patterns:
        decision = "algorithm_change_candidate"
        reasons.append("at_least_one_pattern_has_high_weak_rate_with_low_non_weak_impact")
    elif (
        weak_or_bad_rate <= float(args.max_weak_or_bad_rate)
        and high_score_weak_or_bad_rate <= float(args.max_high_score_weak_or_bad_rate)
    ):
        decision = "pass_continue_validation_without_algorithm_change"
        reasons.append("overall_weak_rate_and_high_score_weak_rate_are_within_gate")
        reasons.append("candidate_patterns_are_not_actionable_without_overfiltering")
    else:
        decision = "review_collect_more_targeted_samples"
        if weak_or_bad_rate > float(args.max_weak_or_bad_rate):
            reasons.append("overall_weak_rate_exceeds_gate")
        if high_score_weak_or_bad_rate > float(args.max_high_score_weak_or_bad_rate):
            reasons.append("high_score_weak_rate_exceeds_gate")
        reasons.append("no_low_blast_radius_pattern_was_found")

    return {
        "decision": decision,
        "reasons": reasons,
        "row_count": row_count,
        "weak_or_bad_rate": round(float(weak_or_bad_rate), 4),
        "max_weak_or_bad_rate": float(args.max_weak_or_bad_rate),
        "high_score_weak_or_bad_count": high_score_weak_or_bad_count,
        "high_score_weak_or_bad_rate": round(float(high_score_weak_or_bad_rate), 4),
        "max_high_score_weak_or_bad_rate": float(args.max_high_score_weak_or_bad_rate),
        "actionable_pattern_count": len(actionable_patterns),
        "actionable_patterns": actionable_patterns,
        "gate": {
            "min_actionable_pattern_weak_count": int(args.min_actionable_pattern_weak_count),
            "min_actionable_pattern_weak_rate": float(args.min_actionable_pattern_weak_rate),
            "max_actionable_pattern_non_weak_affected": int(args.max_actionable_pattern_non_weak_affected),
        },
    }


def quality_gate(args: argparse.Namespace) -> None:
    summary_path = resolve_path(args.pattern_summary_json)
    output_path = resolve_path(args.output_json) if args.output_json else summary_path.with_name("judge_quality_gate.json")
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = quality_gate_decision(summary, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_review and result["decision"] != "pass_continue_validation_without_algorithm_change":
        raise SystemExit(2)


def validate_results(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summarize_args = argparse.Namespace(
        parts_glob=args.parts_glob,
        retry_glob=args.retry_glob,
        output_dir=str(output_dir),
        high_score_threshold=float(args.high_score_threshold),
    )
    summarize_chunks(summarize_args)

    results_csv = output_dir / "openai_chunk_judge_results.csv"
    pattern_dir = output_dir / "patterns"
    analyze_args = argparse.Namespace(
        results_csv=str(results_csv),
        output_dir=str(pattern_dir),
        ingredient_category_profile=args.ingredient_category_profile,
        high_score_threshold=float(args.high_score_threshold),
    )
    analyze_patterns(analyze_args)

    pattern_summary_path = pattern_dir / "judge_pattern_summary.json"
    pattern_summary = json.loads(pattern_summary_path.read_text(encoding="utf-8"))
    gate_args = argparse.Namespace(
        max_weak_or_bad_rate=float(args.max_weak_or_bad_rate),
        max_high_score_weak_or_bad_rate=float(args.max_high_score_weak_or_bad_rate),
        min_actionable_pattern_weak_count=int(args.min_actionable_pattern_weak_count),
        min_actionable_pattern_weak_rate=float(args.min_actionable_pattern_weak_rate),
        max_actionable_pattern_non_weak_affected=int(args.max_actionable_pattern_non_weak_affected),
    )
    gate_result = quality_gate_decision(pattern_summary, gate_args)
    gate_json = output_dir / "judge_quality_gate.json"
    gate_json.write_text(json.dumps(gate_result, ensure_ascii=False, indent=2), encoding="utf-8")

    validation_summary = {
        "created_at": now_iso(),
        "decision": gate_result["decision"],
        "parts_glob": coerce_patterns(args.parts_glob),
        "retry_glob": coerce_patterns(args.retry_glob),
        "high_score_threshold": float(args.high_score_threshold),
        "quality_gate": gate_result,
        "outputs": {
            "merged_results_csv": str(results_csv),
            "merged_summary_json": str(output_dir / "openai_chunk_judge_summary.json"),
            "pattern_summary_json": str(pattern_summary_path),
            "quality_gate_json": str(gate_json),
        },
    }
    validation_summary_path = output_dir / "judge_validation_summary.json"
    validation_summary_path.write_text(json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation_summary, ensure_ascii=False, indent=2))

    if args.fail_on_review and gate_result["decision"] != "pass_continue_validation_without_algorithm_change":
        raise SystemExit(2)


def format_rate(value: Any) -> str:
    return f"{float(value or 0.0) * 100:.2f}%"


def build_validation_report(
    merged_summary: dict[str, Any],
    pattern_summary: dict[str, Any],
    quality_gate_result: dict[str, Any],
    *,
    validation_dir: str,
    top_categories: int,
    top_patterns: int,
) -> str:
    judgment_counts = Counter(merged_summary.get("judgment_counts", {}) or {})
    weak_or_bad_count = int(judgment_counts.get("weak", 0)) + int(judgment_counts.get("bad", 0))
    label_count = int(merged_summary.get("actual_label_count") or merged_summary.get("row_count") or 0)
    expected_label_count = int(merged_summary.get("expected_label_count") or label_count)
    coverage_ok = bool(merged_summary.get("coverage_ok", expected_label_count == label_count))
    high_score_weak_count = int(
        merged_summary.get("current_high_score_weak_or_bad_count")
        or merged_summary.get("high_score_weak_or_bad_count")
        or quality_gate_result.get("high_score_weak_or_bad_count")
        or 0
    )
    high_score_weak_rate = (
        float(quality_gate_result.get("high_score_weak_or_bad_rate", 0.0) or 0.0)
        if quality_gate_result
        else float(high_score_weak_count / label_count) if label_count else 0.0
    )

    category_rows = category_judgment_rows(merged_summary)
    category_rows.sort(
        key=lambda row: (
            -float(row["weak_or_bad_rate"]),
            -int(row["weak_or_bad_count"]),
            str(row["category"]),
        )
    )
    pattern_rows = [
        row for row in (pattern_summary.get("pattern_impacts", []) or [])
        if isinstance(row, dict)
    ]
    pattern_rows.sort(
        key=lambda row: (
            -int(row.get("weak_or_bad_count", 0) or 0),
            -float(row.get("weak_or_bad_rate", 0.0) or 0.0),
            int(row.get("non_weak_affected_count", 0) or 0),
            str(row.get("pattern", "")),
        )
    )

    lines = [
        "# Recommendation Quality Validation Report",
        "",
        f"- Created at: {now_iso()}",
        f"- Validation dir: `{validation_dir}`",
        f"- Decision: `{quality_gate_result.get('decision', '')}`",
        f"- Algorithm recommendation: `keep_current_algorithm_without_new_caps`"
        if quality_gate_result.get("decision") == "pass_continue_validation_without_algorithm_change"
        else "- Algorithm recommendation: `review_before_algorithm_change`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Label coverage | {label_count} / {expected_label_count} |",
        f"| Coverage OK | {str(coverage_ok).lower()} |",
        f"| Product count | {int(merged_summary.get('product_count', 0) or 0)} |",
        f"| Request count | {int(merged_summary.get('request_count', 0) or 0)} |",
        f"| Weak/bad count | {weak_or_bad_count} |",
        f"| Weak/bad rate | {format_rate(merged_summary.get('weak_or_bad_rate', 0.0))} |",
        f"| High-score weak/bad count | {high_score_weak_count} |",
        f"| High-score weak/bad rate | {format_rate(high_score_weak_rate)} |",
        f"| Actionable pattern count | {int(quality_gate_result.get('actionable_pattern_count', 0) or 0)} |",
        "",
        "## Judgment Counts",
        "",
        "| Judgment | Count |",
        "| --- | ---: |",
    ]
    for judgment in ("reasonable", "acceptable_adjacent", "weak", "bad"):
        lines.append(f"| {judgment} | {int(judgment_counts.get(judgment, 0))} |")

    lines.extend(
        [
            "",
            "## Gate Reasons",
            "",
        ]
    )
    for reason in quality_gate_result.get("reasons", []) or []:
        lines.append(f"- `{reason}`")
    if not quality_gate_result.get("reasons"):
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Top Category Weak Rates",
            "",
            "| Category | Labels | Weak/Bad | Weak/Bad Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in category_rows[: max(0, int(top_categories))]:
        lines.append(
            f"| {row['category']} | {int(row['total_count'])} | "
            f"{int(row['weak_or_bad_count'])} | {format_rate(row['weak_or_bad_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Pattern Impact",
            "",
            "| Pattern | Matched | Weak/Bad | Weak/Bad Rate | Non-Weak Affected |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pattern_rows[: max(0, int(top_patterns))]:
        lines.append(
            f"| {row.get('pattern', '')} | {int(row.get('matched_count', 0) or 0)} | "
            f"{int(row.get('weak_or_bad_count', 0) or 0)} | "
            f"{format_rate(row.get('weak_or_bad_rate', 0.0))} | "
            f"{int(row.get('non_weak_affected_count', 0) or 0)} |"
        )

    lines.extend(
        [
            "",
            "## Source Files",
            "",
            f"- Merged summary: `{merged_summary.get('outputs', {}).get('merged_summary_json', 'openai_chunk_judge_summary.json')}`",
            f"- Pattern summary: `{pattern_summary.get('outputs', {}).get('pattern_summary_json', 'patterns/judge_pattern_summary.json')}`",
            "- Quality gate: `judge_quality_gate.json`",
            "",
        ]
    )
    return "\n".join(lines)


def write_validation_report(args: argparse.Namespace) -> None:
    validation_dir = resolve_path(args.validation_dir)
    merged_summary_path = resolve_path(args.summary_json) if args.summary_json else validation_dir / "openai_chunk_judge_summary.json"
    pattern_summary_path = (
        resolve_path(args.pattern_summary_json)
        if args.pattern_summary_json
        else validation_dir / "patterns" / "judge_pattern_summary.json"
    )
    quality_gate_path = resolve_path(args.quality_gate_json) if args.quality_gate_json else validation_dir / "judge_quality_gate.json"
    output_md = resolve_path(args.output_md) if args.output_md else validation_dir / "judge_validation_report.md"
    for path in (merged_summary_path, pattern_summary_path, quality_gate_path):
        if not path.exists():
            raise FileNotFoundError(path)

    merged_summary = json.loads(merged_summary_path.read_text(encoding="utf-8"))
    pattern_summary = json.loads(pattern_summary_path.read_text(encoding="utf-8"))
    quality_gate_result = json.loads(quality_gate_path.read_text(encoding="utf-8"))
    report = build_validation_report(
        merged_summary,
        pattern_summary,
        quality_gate_result,
        validation_dir=str(validation_dir),
        top_categories=int(args.top_categories),
        top_patterns=int(args.top_patterns),
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_md.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(report)
    print(json.dumps({"output_md": str(output_md), "decision": quality_gate_result.get("decision", "")}, ensure_ascii=False, indent=2))


def category_judgment_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, count in (summary.get("category_judgment_counts", {}) or {}).items():
        text = str(key or "")
        if "|" not in text:
            continue
        category, judgment = text.rsplit("|", 1)
        counts[category][judgment] += int(count or 0)

    rows: list[dict[str, Any]] = []
    for category, judgment_counts in sorted(counts.items()):
        reasonable = int(judgment_counts.get("reasonable", 0))
        acceptable_adjacent = int(judgment_counts.get("acceptable_adjacent", 0))
        weak = int(judgment_counts.get("weak", 0))
        bad = int(judgment_counts.get("bad", 0))
        total = reasonable + acceptable_adjacent + weak + bad
        weak_or_bad = weak + bad
        rows.append(
            {
                "category": category,
                "total_count": total,
                "reasonable_count": reasonable,
                "acceptable_adjacent_count": acceptable_adjacent,
                "weak_count": weak,
                "bad_count": bad,
                "weak_or_bad_count": weak_or_bad,
                "weak_or_bad_rate": round(float(weak_or_bad / total), 4) if total else 0.0,
            }
        )
    return rows


def shell_quote_arg(value: str) -> str:
    text = str(value)
    return '"' + text.replace('"', '\\"') + '"'


def next_sample_plan(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rows = category_judgment_rows(summary)
    min_labels = int(args.min_labels)
    min_weak_rate = float(args.min_weak_rate)
    max_categories = int(args.max_categories)
    eligible = [
        row
        for row in rows
        if int(row["total_count"]) >= min_labels and float(row["weak_or_bad_rate"]) >= min_weak_rate
    ]
    eligible.sort(key=lambda row: (-float(row["weak_or_bad_rate"]), -int(row["weak_or_bad_count"]), str(row["category"])))

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in eligible:
        item = dict(row)
        item["selection_reason"] = "weak_rate_above_threshold"
        selected.append(item)
        seen.add(str(row["category"]))
        if len(selected) >= max_categories:
            break

    if len(selected) < max_categories:
        fallback = [
            row
            for row in rows
            if int(row["total_count"]) >= min_labels
            and int(row["weak_or_bad_count"]) > 0
            and str(row["category"]) not in seen
        ]
        fallback.sort(key=lambda row: (-int(row["weak_or_bad_count"]), -float(row["weak_or_bad_rate"]), str(row["category"])))
        for row in fallback:
            item = dict(row)
            item["selection_reason"] = "weak_count_fallback"
            selected.append(item)
            seen.add(str(row["category"]))
            if len(selected) >= max_categories:
                break

    seed = int(args.seed or 0)
    if seed <= 0:
        seed = int(datetime.now().strftime("%Y%m%d"))
    sample_output_dir = str(args.sample_output_dir or "").strip()
    if not sample_output_dir:
        sample_output_dir = str(ROOT_DIR / "output" / f"recommendation_quality_judge_v2_9_openai_targeted_next_seed{seed}")
    command = [
        "python",
        "scripts\\recommendation_quality_judge_batch.py",
        "prepare",
        "--output-dir",
        sample_output_dir,
        "--per-category",
        str(int(args.per_category)),
        "--seed",
        str(seed),
        "--top-k",
        str(int(args.top_k)),
        "--workers",
        str(int(args.workers)),
        "--progress-every",
        str(int(args.progress_every)),
    ]
    for row in selected:
        command.extend(["--main-category", str(row["category"])])

    return {
        "created_at": now_iso(),
        "source_summary": str(args.summary_json),
        "min_labels": min_labels,
        "min_weak_rate": min_weak_rate,
        "max_categories": max_categories,
        "selected_category_count": len(selected),
        "selected_categories": selected,
        "prepare_command": command,
        "prepare_command_powershell": " ".join(shell_quote_arg(item) if " " in item or "\\" in item else item for item in command),
        "all_category_rows": rows,
    }


def plan_next_sample(args: argparse.Namespace) -> None:
    summary_path = resolve_path(args.summary_json)
    output_dir = resolve_path(args.output_dir)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plan = next_sample_plan(summary, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_json = output_dir / "judge_next_sample_plan.json"
    selected_csv = output_dir / "judge_next_sample_categories.csv"
    all_csv = output_dir / "judge_category_judgment_rates.csv"
    plan_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        selected_csv,
        [
            "category",
            "total_count",
            "reasonable_count",
            "acceptable_adjacent_count",
            "weak_count",
            "bad_count",
            "weak_or_bad_count",
            "weak_or_bad_rate",
            "selection_reason",
        ],
        list(plan["selected_categories"]),
    )
    write_csv(
        all_csv,
        [
            "category",
            "total_count",
            "reasonable_count",
            "acceptable_adjacent_count",
            "weak_count",
            "bad_count",
            "weak_or_bad_count",
            "weak_or_bad_rate",
        ],
        list(plan["all_category_rows"]),
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def run_command(command: list[str], cwd: Path, *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")
    return proc


def extract_job_name(output: str) -> str:
    match = re.search(r"Batch job name:\s*(\S+)", output)
    if not match:
        raise RuntimeError("Batch job name not found in submit output")
    return match.group(1)


def extract_state(output: str) -> str:
    match = re.search(r"(JOB_STATE_[A-Z_]+)", output)
    return match.group(1) if match else "UNKNOWN"


def submit(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    health_batch_dir = resolve_path(args.health_batch_dir)
    jsonl_path = resolve_path(args.jsonl) if args.jsonl else output_dir / DEFAULT_BATCH_JSONL_NAME
    if not health_batch_dir.exists():
        raise FileNotFoundError(health_batch_dir)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    job_file = resolve_path(args.job_file) if args.job_file else output_dir / "gemini_recommendation_judge.job.txt"
    if job_file.exists() and not args.force:
        job_name = job_file.read_text(encoding="utf-8").strip()
        print(json.dumps({"job_name": job_name, "job_file": str(job_file), "reused": True}, ensure_ascii=False, indent=2))
        return

    display_name = args.name or f"recommendation-quality-judge-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    proc = run_command(
        ["py", "-3.12", "submit_gemini_batch.py", "--jsonl", str(jsonl_path), "--name", display_name],
        cwd=health_batch_dir,
    )
    job_name = extract_job_name(proc.stdout or "")
    job_file.parent.mkdir(parents=True, exist_ok=True)
    job_file.write_text(job_name + "\n", encoding="utf-8")
    print(json.dumps({"job_name": job_name, "job_file": str(job_file), "jsonl": str(jsonl_path)}, ensure_ascii=False, indent=2))


def check(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    health_batch_dir = resolve_path(args.health_batch_dir)
    job_name = args.job
    if not job_name:
        job_file = resolve_path(args.job_file) if args.job_file else output_dir / "gemini_recommendation_judge.job.txt"
        if not job_file.exists():
            raise FileNotFoundError(job_file)
        job_name = job_file.read_text(encoding="utf-8").strip()

    command = ["py", "-3.12", "check_gemini_batch.py", "--job", job_name]
    if args.watch:
        command.append("--watch")
    if args.download:
        result_path = resolve_path(args.download_output) if args.download_output else output_dir / DEFAULT_RESULT_JSONL_NAME
        command.extend(["--download-output", str(result_path)])
    proc = run_command(command, cwd=health_batch_dir, allow_failure=args.allow_failure)
    print(json.dumps({"job_name": job_name, "state": extract_state(proc.stdout or "")}, ensure_ascii=False, indent=2))


def load_openai_client(env_path: str | Path):
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai and python-dotenv packages are required for OpenAI Batch commands") from exc

    resolved_env = Path(env_path)
    if not resolved_env.is_absolute():
        resolved_env = ROOT_DIR / resolved_env
    load_dotenv(dotenv_path=resolved_env, override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY is not set in {resolved_env}")
    return OpenAI(api_key=api_key)


def model_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))


def openai_submit(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    jsonl_path = resolve_path(args.jsonl) if args.jsonl else output_dir / DEFAULT_OPENAI_BATCH_JSONL_NAME
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    job_file = resolve_path(args.job_file) if args.job_file else output_dir / DEFAULT_OPENAI_JOB_FILE_NAME
    if job_file.exists() and not args.force:
        job_id = job_file.read_text(encoding="utf-8").strip()
        print(json.dumps({"job_id": job_id, "job_file": str(job_file), "reused": True}, ensure_ascii=False, indent=2))
        return

    client = load_openai_client(args.env_path)
    display_name = args.name or f"recommendation-quality-judge-openai-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with jsonl_path.open("rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "name": display_name[:512],
            "script": "recommendation_quality_judge_batch",
            "model": args.model,
        },
    )
    job_file.parent.mkdir(parents=True, exist_ok=True)
    job_file.write_text(batch.id + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "job_id": batch.id,
                "status": batch.status,
                "input_file_id": input_file.id,
                "job_file": str(job_file),
                "jsonl": str(jsonl_path),
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_file_content_bytes(content: Any) -> bytes:
    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data
        return str(data).encode("utf-8")
    data = getattr(content, "content", None)
    if isinstance(data, bytes):
        return data
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    if isinstance(content, bytes):
        return content
    return str(content).encode("utf-8")


def openai_check(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    job_id = args.job
    if not job_id:
        job_file = resolve_path(args.job_file) if args.job_file else output_dir / DEFAULT_OPENAI_JOB_FILE_NAME
        if not job_file.exists():
            raise FileNotFoundError(job_file)
        job_id = job_file.read_text(encoding="utf-8").strip()

    client = load_openai_client(args.env_path)
    batch = client.batches.retrieve(job_id)
    batch_data = model_to_dict(batch)
    downloaded: dict[str, str] = {}
    if args.download and batch.output_file_id:
        result_path = resolve_path(args.download_output) if args.download_output else output_dir / DEFAULT_OPENAI_RESULT_JSONL_NAME
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(read_file_content_bytes(client.files.content(batch.output_file_id)))
        downloaded["output"] = str(result_path)
    if args.download_errors and batch.error_file_id:
        error_path = resolve_path(args.error_output) if args.error_output else output_dir / "openai_recommendation_judge_errors_raw.jsonl"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_bytes(read_file_content_bytes(client.files.content(batch.error_file_id)))
        downloaded["errors"] = str(error_path)

    print(
        json.dumps(
            {
                "job_id": batch.id,
                "status": batch.status,
                "request_counts": batch_data.get("request_counts"),
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "downloaded": downloaded,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def openai_cancel(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    job_id = args.job
    if not job_id:
        job_file = resolve_path(args.job_file) if args.job_file else output_dir / DEFAULT_OPENAI_JOB_FILE_NAME
        if not job_file.exists():
            raise FileNotFoundError(job_file)
        job_id = job_file.read_text(encoding="utf-8").strip()

    client = load_openai_client(args.env_path)
    batch = client.batches.cancel(job_id)
    print(json.dumps({"job_id": batch.id, "status": batch.status}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and parse Gemini Batch recommendation quality judge jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create top-k recommendation snapshot and Gemini Batch JSONL.")
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument("--all", action="store_true", help="Evaluate every product instead of a stratified sample.")
    prepare_parser.add_argument("--sample-size", type=int, default=0, help="Optional total sample cap after stratified selection.")
    prepare_parser.add_argument("--per-category", type=int, default=10, help="Sample count per main category when --all is not set.")
    prepare_parser.add_argument("--report-no", action="append", default=[], help="Specific report_no to include. Can be repeated.")
    prepare_parser.add_argument("--main-category", action="append", default=[], help="Restrict sampling to a main category. Can be repeated.")
    prepare_parser.add_argument("--offset", type=int, default=0, help="Skip this many selected products before applying --limit.")
    prepare_parser.add_argument("--limit", type=int, default=0, help="Final cap for quick smoke runs.")
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    prepare_parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    prepare_parser.add_argument("--max-df-for-seed", type=int, default=DEFAULT_MAX_DF_FOR_SEED)
    prepare_parser.add_argument("--similarity-algorithm", default=SIMILARITY_ALGORITHM_VERSION)
    prepare_parser.add_argument("--ingredient-category-profile", default="")
    prepare_parser.add_argument("--progress-every", type=int, default=10)
    prepare_parser.add_argument("--workers", type=int, default=1, help="Parallel worker process count for top-k generation.")
    prepare_parser.set_defaults(func=prepare)

    apply_parser = subparsers.add_parser("apply", help="Parse Gemini Batch result JSONL into CSV and summary.")
    apply_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    apply_parser.add_argument("--result-jsonl", default="")
    apply_parser.add_argument("--snapshot-jsonl", default="")
    apply_parser.add_argument("--high-score-threshold", type=float, default=0.65)
    apply_parser.set_defaults(func=apply_results)

    refresh_parser = subparsers.add_parser("refresh", help="Recompute local risk flags and Batch JSONL from an existing snapshot.")
    refresh_parser.add_argument("--source-snapshot-jsonl", required=True)
    refresh_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    refresh_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    refresh_parser.add_argument("--offset", type=int, default=0)
    refresh_parser.add_argument("--limit", type=int, default=0)
    refresh_parser.set_defaults(func=refresh_snapshot)

    impact_parser = subparsers.add_parser("impact", help="Recompute current scores for Gemini-judged pairs and summarize score changes.")
    impact_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    impact_parser.add_argument("--judge-csv", default="")
    impact_parser.add_argument("--ingredient-category-profile", default="")
    impact_parser.add_argument("--high-score-threshold", type=float, default=0.65)
    impact_parser.set_defaults(func=score_impact)

    merge_parser = subparsers.add_parser("merge-parts", help="Merge Gemini judge CSV results from multiple part output directories.")
    merge_parser.add_argument("--parts-glob", required=True)
    merge_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    merge_parser.add_argument("--high-score-threshold", type=float, default=0.65)
    merge_parser.set_defaults(func=merge_parts)

    summarize_parser = subparsers.add_parser(
        "summarize-chunks",
        help="Summarize OpenAI chunk judge outputs, preferring post-fix impact scores and applying retry replacements.",
    )
    summarize_parser.add_argument("--parts-glob", action="append", required=True)
    summarize_parser.add_argument("--retry-glob", action="append", default=[])
    summarize_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    summarize_parser.add_argument("--high-score-threshold", type=float, default=0.65)
    summarize_parser.set_defaults(func=summarize_chunks)

    analyze_patterns_parser = subparsers.add_parser(
        "analyze-patterns",
        help="Analyze high-score weak judge results and estimate candidate cap pattern impact.",
    )
    analyze_patterns_parser.add_argument("--results-csv", required=True)
    analyze_patterns_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    analyze_patterns_parser.add_argument("--ingredient-category-profile", default="")
    analyze_patterns_parser.add_argument("--high-score-threshold", type=float, default=0.65)
    analyze_patterns_parser.set_defaults(func=analyze_patterns)

    quality_gate_parser = subparsers.add_parser(
        "quality-gate",
        help="Decide whether judge results support continuing, more sampling, or an algorithm-change candidate.",
    )
    quality_gate_parser.add_argument("--pattern-summary-json", required=True)
    quality_gate_parser.add_argument("--output-json", default="")
    quality_gate_parser.add_argument("--max-weak-or-bad-rate", type=float, default=QUALITY_GATE_MAX_WEAK_OR_BAD_RATE)
    quality_gate_parser.add_argument(
        "--max-high-score-weak-or-bad-rate",
        type=float,
        default=QUALITY_GATE_MAX_HIGH_SCORE_WEAK_OR_BAD_RATE,
    )
    quality_gate_parser.add_argument(
        "--min-actionable-pattern-weak-count",
        type=int,
        default=QUALITY_GATE_MIN_ACTIONABLE_PATTERN_WEAK_COUNT,
    )
    quality_gate_parser.add_argument(
        "--min-actionable-pattern-weak-rate",
        type=float,
        default=QUALITY_GATE_MIN_ACTIONABLE_PATTERN_WEAK_RATE,
    )
    quality_gate_parser.add_argument(
        "--max-actionable-pattern-non-weak-affected",
        type=int,
        default=QUALITY_GATE_MAX_ACTIONABLE_PATTERN_NON_WEAK_AFFECTED,
    )
    quality_gate_parser.add_argument("--fail-on-review", action="store_true")
    quality_gate_parser.set_defaults(func=quality_gate)

    validate_parser = subparsers.add_parser(
        "validate-results",
        help="Run summarize-chunks, analyze-patterns, and quality-gate as one validation workflow.",
    )
    validate_parser.add_argument("--parts-glob", action="append", required=True)
    validate_parser.add_argument("--retry-glob", action="append", default=[])
    validate_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    validate_parser.add_argument("--ingredient-category-profile", default="")
    validate_parser.add_argument("--high-score-threshold", type=float, default=0.65)
    validate_parser.add_argument("--max-weak-or-bad-rate", type=float, default=QUALITY_GATE_MAX_WEAK_OR_BAD_RATE)
    validate_parser.add_argument(
        "--max-high-score-weak-or-bad-rate",
        type=float,
        default=QUALITY_GATE_MAX_HIGH_SCORE_WEAK_OR_BAD_RATE,
    )
    validate_parser.add_argument(
        "--min-actionable-pattern-weak-count",
        type=int,
        default=QUALITY_GATE_MIN_ACTIONABLE_PATTERN_WEAK_COUNT,
    )
    validate_parser.add_argument(
        "--min-actionable-pattern-weak-rate",
        type=float,
        default=QUALITY_GATE_MIN_ACTIONABLE_PATTERN_WEAK_RATE,
    )
    validate_parser.add_argument(
        "--max-actionable-pattern-non-weak-affected",
        type=int,
        default=QUALITY_GATE_MAX_ACTIONABLE_PATTERN_NON_WEAK_AFFECTED,
    )
    validate_parser.add_argument("--fail-on-review", action="store_true")
    validate_parser.set_defaults(func=validate_results)

    report_parser = subparsers.add_parser(
        "validation-report",
        help="Write a Markdown report from validate-results JSON outputs.",
    )
    report_parser.add_argument("--validation-dir", required=True)
    report_parser.add_argument("--summary-json", default="")
    report_parser.add_argument("--pattern-summary-json", default="")
    report_parser.add_argument("--quality-gate-json", default="")
    report_parser.add_argument("--output-md", default="")
    report_parser.add_argument("--top-categories", type=int, default=10)
    report_parser.add_argument("--top-patterns", type=int, default=10)
    report_parser.set_defaults(func=write_validation_report)

    next_sample_parser = subparsers.add_parser(
        "plan-next-sample",
        help="Recommend main categories for the next targeted judge sample from a merged judge summary.",
    )
    next_sample_parser.add_argument("--summary-json", required=True)
    next_sample_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    next_sample_parser.add_argument("--min-labels", type=int, default=50)
    next_sample_parser.add_argument("--min-weak-rate", type=float, default=0.08)
    next_sample_parser.add_argument("--max-categories", type=int, default=8)
    next_sample_parser.add_argument("--per-category", type=int, default=10)
    next_sample_parser.add_argument("--seed", type=int, default=0)
    next_sample_parser.add_argument("--sample-output-dir", default="")
    next_sample_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    next_sample_parser.add_argument("--workers", type=int, default=4)
    next_sample_parser.add_argument("--progress-every", type=int, default=10)
    next_sample_parser.set_defaults(func=plan_next_sample)

    submit_parser = subparsers.add_parser("submit", help="Submit the prepared JSONL through D:\\health_batch_project helpers.")
    submit_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    submit_parser.add_argument("--health-batch-dir", default=str(DEFAULT_HEALTH_BATCH_DIR))
    submit_parser.add_argument("--jsonl", default="")
    submit_parser.add_argument("--name", default="")
    submit_parser.add_argument("--job-file", default="")
    submit_parser.add_argument("--force", action="store_true")
    submit_parser.set_defaults(func=submit)

    check_parser = subparsers.add_parser("check", help="Check or download a submitted Gemini Batch job.")
    check_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    check_parser.add_argument("--health-batch-dir", default=str(DEFAULT_HEALTH_BATCH_DIR))
    check_parser.add_argument("--job", default="")
    check_parser.add_argument("--job-file", default="")
    check_parser.add_argument("--watch", action="store_true")
    check_parser.add_argument("--download", action="store_true")
    check_parser.add_argument("--download-output", default="")
    check_parser.add_argument("--allow-failure", action="store_true")
    check_parser.set_defaults(func=check)

    openai_submit_parser = subparsers.add_parser("openai-submit", help="Submit the prepared OpenAI Batch JSONL.")
    openai_submit_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    openai_submit_parser.add_argument("--env-path", default=str(DEFAULT_HEALTH_BATCH_DIR / ".env"))
    openai_submit_parser.add_argument("--jsonl", default="")
    openai_submit_parser.add_argument("--name", default="")
    openai_submit_parser.add_argument("--job-file", default="")
    openai_submit_parser.add_argument("--model", default=OPENAI_MODEL_NAME)
    openai_submit_parser.add_argument("--force", action="store_true")
    openai_submit_parser.set_defaults(func=openai_submit)

    openai_check_parser = subparsers.add_parser("openai-check", help="Check or download a submitted OpenAI Batch job.")
    openai_check_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    openai_check_parser.add_argument("--env-path", default=str(DEFAULT_HEALTH_BATCH_DIR / ".env"))
    openai_check_parser.add_argument("--job", default="")
    openai_check_parser.add_argument("--job-file", default="")
    openai_check_parser.add_argument("--download", action="store_true")
    openai_check_parser.add_argument("--download-output", default="")
    openai_check_parser.add_argument("--download-errors", action="store_true")
    openai_check_parser.add_argument("--error-output", default="")
    openai_check_parser.set_defaults(func=openai_check)

    openai_cancel_parser = subparsers.add_parser("openai-cancel", help="Cancel a submitted OpenAI Batch job.")
    openai_cancel_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    openai_cancel_parser.add_argument("--env-path", default=str(DEFAULT_HEALTH_BATCH_DIR / ".env"))
    openai_cancel_parser.add_argument("--job", default="")
    openai_cancel_parser.add_argument("--job-file", default="")
    openai_cancel_parser.set_defaults(func=openai_cancel)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
