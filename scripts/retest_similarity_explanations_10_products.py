from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.enhance_similarity_with_explanation import (  # noqa: E402
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_MAX_DF_FOR_SEED,
    DEFAULT_TOP_K,
    TABLE_NAME,
    build_vector_indexes,
    compute_topk_for_single_product,
    ensure_cache_table,
    ingredient_idf,
    load_product_function_profiles,
    load_vector_inputs,
    refresh_cache_rows,
    resolve_runtime_paths,
    safe_json_loads,
    safe_product_key,
    write_outputs,
)


TEST_PRODUCT_NAMES = [
    "고려홍삼정(錠)",
    "제주알로에정",
    "하모네 메가플러스 비타민C 3000",
    "뼈건강 K2 칼마디",
    "한 입에 톡 비타민D 400IU",
    "관절연골케어엔NEM",
    "광동녹용전립선",
    "인지력 건강엔 포스파티딜세린",
    "에버콜라겐 레티놀A",
    "뼈건강 지킴이",
]

BANNED_REASON_TERMS = [
    "치료",
    "완치",
    "의약품 대체",
    "질병 개선",
    "효능이 같다",
    "효과가 같다",
]

AWKWARD_REASON_PATTERNS = [
    r"와 .*가 공통",
    r"과 .*가 공통",
    r"칼슘가",
    r"마그네슘와",
    r"아연가",
]

EXCIPIENT_NAMES = [
    "히드록시프로필메틸셀룰로오스",
    "결정셀룰로스",
    "스테아린산마그네슘",
    "이산화규소",
    "카복시메틸셀룰로오스",
    "글리세린지방산에스테르",
    "HPMC",
]

EXPECTED_RULES = {
    "고려홍삼정(錠)": {
        "expected_categories": {"면역", "피로개선", "혈행"},
        "reason_keywords": ["홍삼", "진세노사이드"],
    },
    "제주알로에정": {
        "expected_categories": {"장 건강", "배변활동"},
        "reason_keywords": ["장", "유산균", "프로바이오틱스", "이눌린", "치커리"],
    },
    "하모네 메가플러스 비타민C 3000": {
        "expected_categories": {"항산화", "영양보충"},
        "reason_keywords": ["비타민 C", "비타민C"],
    },
    "뼈건강 K2 칼마디": {
        "expected_categories": {"뼈 건강"},
        "reason_keywords": ["칼슘", "비타민 D", "비타민 K", "비타민D", "비타민K"],
    },
    "한 입에 톡 비타민D 400IU": {
        "expected_categories": {"뼈 건강", "영양보충"},
        "reason_keywords": ["비타민 D", "비타민D"],
    },
    "관절연골케어엔NEM": {
        "expected_categories": {"관절/연골"},
        "reason_keywords": ["난각막가수분해물", "NEM", "난각막"],
    },
    "광동녹용전립선": {
        "expected_categories": {"남성 건강", "영양보충", "기타"},
        "reason_keywords": ["전립선", "남성", "쏘팔메토", "녹용"],
    },
    "인지력 건강엔 포스파티딜세린": {
        "expected_categories": {"기억력", "인지력", "기억력/인지력"},
        "reason_keywords": ["포스파티딜세린"],
    },
    "에버콜라겐 레티놀A": {
        "expected_categories": {"피부 건강"},
        "reason_keywords": ["콜라겐", "피부", "레티놀"],
    },
    "뼈건강 지킴이": {
        "expected_categories": {"뼈 건강"},
        "reason_keywords": ["칼슘", "비타민 D", "비타민D"],
    },
}


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("retest_10_products")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_profile_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["product_name"] = df["product_name"].fillna("").astype(str)
    df["report_no"] = df["report_no"].fillna("").astype(str)
    df["normalized_product_name"] = df["product_name"].map(normalize_text)
    return df


def select_product_row(query_name: str, profile_df: pd.DataFrame, logger: logging.Logger) -> tuple[dict | None, str]:
    exact = profile_df[profile_df["product_name"] == query_name].copy()
    if not exact.empty:
        exact = exact.sort_values(
            by=["confidence", "report_no", "product_name"],
            ascending=[False, True, True],
            na_position="last",
        )
        selected = exact.iloc[0].to_dict()
        logger.info("product_match query=%s mode=exact count=%s selected=%s (%s)", query_name, len(exact), selected["product_name"], selected["report_no"])
        return selected, f"exact:{selected['product_name']}:{selected['report_no']}"

    normalized_query = normalize_text(query_name)
    partial = profile_df[
        profile_df["product_name"].str.contains(query_name, na=False, regex=False)
        | profile_df["normalized_product_name"].str.contains(normalized_query, na=False, regex=False)
        | profile_df["normalized_product_name"].map(lambda value: normalized_query in value)
    ].copy()
    if partial.empty:
        logger.warning("product_match query=%s mode=none count=0", query_name)
        return None, "not_found"

    partial["match_score"] = partial["normalized_product_name"].map(lambda value: SequenceMatcher(None, normalized_query, value).ratio())
    partial["length_gap"] = partial["normalized_product_name"].map(lambda value: abs(len(value) - len(normalized_query)))
    partial = partial.sort_values(
        by=["match_score", "confidence", "length_gap", "report_no", "product_name"],
        ascending=[False, False, True, True, True],
        na_position="last",
    )
    selected = partial.iloc[0].to_dict()
    candidates = partial[["product_name", "report_no", "match_score", "confidence"]].head(5).to_dict(orient="records")
    logger.info(
        "product_match query=%s mode=partial count=%s selected=%s (%s) candidates=%s",
        query_name,
        len(partial),
        selected["product_name"],
        selected["report_no"],
        json.dumps(candidates, ensure_ascii=False),
    )
    return selected, f"partial:{selected['product_name']}:{selected['report_no']}"


def compute_candidate_pool_stats(
    base_product_id: str,
    base_profile: dict,
    profiles: dict[str, dict],
    product_vectors: dict[str, dict[str, float]],
    ingredient_postings: dict[str, list[str]],
    ingredient_frequency: dict[str, int],
    candidate_limit: int,
    max_df_for_seed: int,
) -> dict:
    candidate_stats: dict[str, dict] = {}
    base_categories = {str(base_profile.get("product_main_category", "기타"))}
    base_categories.update([str(item) for item in base_profile.get("llm_sub_function_categories", []) if str(item).strip()])
    total_product_count = len(product_vectors)

    def ensure_candidate(candidate_id: str) -> dict:
        if candidate_id not in candidate_stats:
            candidate_profile = profiles[candidate_id]
            candidate_categories = {str(candidate_profile.get("product_main_category", "기타"))}
            candidate_categories.update([str(item) for item in candidate_profile.get("llm_sub_function_categories", []) if str(item).strip()])
            candidate_stats[candidate_id] = {
                "target_product_id": candidate_id,
                "same_main_category": int(candidate_profile.get("product_main_category") == base_profile.get("product_main_category")),
                "shared_primary_count": 0,
                "shared_secondary_count": 0,
                "shared_support_count": 0,
                "shared_category_count": len(base_categories & candidate_categories),
                "preliminary_score": 0.0,
            }
        return candidate_stats[candidate_id]

    for candidate_id, profile in profiles.items():
        if candidate_id == base_product_id:
            continue
        if profile.get("product_main_category") == base_profile.get("product_main_category"):
            ensure_candidate(candidate_id)

    seed_groups = [
        ("primary", list(base_profile.get("primary_ingredients", [])), 4.0),
        ("secondary", list(base_profile.get("secondary_ingredients", [])), 1.2),
        ("support", list(base_profile.get("support_ingredients", [])), 0.25),
    ]
    for role_name, ingredients, score_weight in seed_groups:
        for ingredient in ingredients:
            ingredient_df = ingredient_frequency.get(ingredient, 0)
            if role_name != "support" and ingredient_df > max_df_for_seed:
                continue
            for candidate_id in ingredient_postings.get(ingredient, []):
                if candidate_id == base_product_id or candidate_id not in profiles:
                    continue
                stat = ensure_candidate(candidate_id)
                if role_name == "primary":
                    stat["shared_primary_count"] += 1
                elif role_name == "secondary":
                    stat["shared_secondary_count"] += 1
                else:
                    stat["shared_support_count"] += 1
                stat["preliminary_score"] += score_weight * ingredient_idf(total_product_count, max(1, ingredient_df))

    rows = list(candidate_stats.values())
    for row in rows:
        if row["shared_support_count"] > 0 and row["shared_primary_count"] == 0 and row["shared_secondary_count"] == 0:
            row["preliminary_score"] -= 2.0
    rows = sorted(
        rows,
        key=lambda item: (
            -item["same_main_category"],
            -item["shared_primary_count"],
            -item["shared_category_count"],
            -item["preliminary_score"],
            item["target_product_id"],
        ),
    )
    return {
        "candidate_count_before_limit": len(rows),
        "candidate_count_after_limit": min(len(rows), candidate_limit),
    }


def build_result_row(base_profile: dict, row: dict, rank: int) -> dict:
    target_report_no = ""
    target_product_id = str(row.get("target_product_id", ""))
    if "::" in target_product_id:
        target_report_no = target_product_id.split("::", 1)[1]
    explanation_text = str(row.get("explanation_json", "") or "").strip()
    caution_text = str(row.get("caution", "") or "").strip()
    return {
        "base_report_no": str(base_profile.get("report_no", "")),
        "base_product_name": str(base_profile.get("product_name", "")),
        "base_product_main_category": str(base_profile.get("product_main_category", "")),
        "rank": rank,
        "target_report_no": target_report_no,
        "target_product_id": target_product_id,
        "target_product_name": str(row.get("target_product_name", "")),
        "similarity_score": row.get("similarity_score", 0.0),
        "function_similarity_score": row.get("function_similarity_score", 0.0),
        "core_match_score": row.get("core_match_score", 0.0),
        "substitutability": str(row.get("substitutability", "")),
        "shared_ingredients_json": str(row.get("shared_ingredients_json", "")),
        "shared_categories_json": str(row.get("shared_categories_json", "")),
        "reason": str(row.get("reason", "")),
        "has_caution": bool(caution_text),
        "has_explanation_json": bool(explanation_text),
        "caution": caution_text,
        "explanation_json": explanation_text,
    }


def run_quality_checks(base_query_name: str, base_profile: dict, result_rows: list[dict]) -> list[dict]:
    warnings: list[dict] = []

    def add_warning(code: str, message: str, rank: int | None = None, target_product_name: str = "") -> None:
        warnings.append(
            {
                "base_report_no": str(base_profile.get("report_no", "")),
                "base_product_name": str(base_profile.get("product_name", "")),
                "warning_code": code,
                "warning_message": message,
                "rank": rank if rank is not None else "",
                "target_product_name": target_product_name,
            }
        )

    if not str(base_profile.get("product_main_category", "")).strip():
        add_warning("empty_main_category", "product_main_category is empty")
    if not list(base_profile.get("primary_ingredients", [])):
        add_warning("empty_primary_ingredients", "primary_ingredients_json is empty")
    if len(result_rows) < 10:
        add_warning("topk_short", f"TOP10 results fewer than 10: {len(result_rows)}")

    expected = EXPECTED_RULES.get(base_query_name)
    if expected:
        actual_category = str(base_profile.get("product_main_category", "")).strip()
        if expected["expected_categories"] and actual_category not in expected["expected_categories"]:
            add_warning(
                "unexpected_main_category",
                f"expected one of {sorted(expected['expected_categories'])}, got {actual_category or '(empty)'}",
            )

    for row in result_rows:
        rank = int(row["rank"])
        target_name = row["target_product_name"]
        reason = str(row.get("reason", "") or "")
        explanation = str(row.get("explanation_json", "") or "")
        caution = str(row.get("caution", "") or "")
        substitutability = str(row.get("substitutability", "") or "")

        if not reason.strip():
            add_warning("empty_reason", "reason is empty", rank, target_name)
        if not explanation.strip():
            add_warning("empty_explanation_json", "explanation_json is empty", rank, target_name)
        if not caution.strip():
            add_warning("empty_caution", "caution is empty", rank, target_name)

        for banned in BANNED_REASON_TERMS:
            if banned in reason:
                add_warning("banned_reason_term", f"reason contains banned term: {banned}", rank, target_name)

        for pattern in AWKWARD_REASON_PATTERNS:
            if re.search(pattern, reason):
                add_warning("awkward_reason_pattern", f"reason matched awkward pattern: {pattern}", rank, target_name)

        for excipient in EXCIPIENT_NAMES:
            if excipient.lower() in reason.lower():
                add_warning("excipient_in_reason", f"excipient used in reason: {excipient}", rank, target_name)

        if substitutability not in {"높음", "보통", "낮음"}:
            add_warning("invalid_substitutability", f"unexpected substitutability: {substitutability}", rank, target_name)

    if expected and result_rows:
        top_reason = str(result_rows[0].get("reason", "") or "")
        if expected["reason_keywords"] and not any(keyword in top_reason for keyword in expected["reason_keywords"]):
            add_warning(
                "reason_keyword_mismatch",
                f"top reason missing expected keywords: {expected['reason_keywords']}",
                int(result_rows[0]["rank"]),
                str(result_rows[0]["target_product_name"]),
            )

    return warnings


def main() -> None:
    start_time = time.perf_counter()
    runtime = resolve_runtime_paths()
    output_dir = ROOT_DIR / "output"
    log_path = ROOT_DIR / "logs" / "retest_10_products.log"
    logger = setup_logger(log_path)

    logger.info("runtime vector_csv_path=%s", runtime["vector_csv_path"])
    logger.info("runtime product_profile_csv_path=%s", runtime["product_profile_csv_path"])
    logger.info("runtime sqlite_path=%s", runtime["sqlite_path"])

    vector_df = load_vector_inputs(runtime["vector_csv_path"])
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    profile_df = load_profile_dataframe(runtime["product_profile_csv_path"])
    product_vectors, product_names, report_to_product_ids, ingredient_postings, ingredient_frequency = build_vector_indexes(vector_df)

    summary_results: list[dict] = []
    flat_rows: list[dict] = []
    warnings_rows: list[dict] = []
    failed_rows: list[dict] = []
    base_product_cache_counts: dict[str, int] = {}

    with sqlite3.connect(runtime["sqlite_path"]) as conn:
        ensure_cache_table(conn)
        total_cache_rows_before = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        logger.info("sqlite total_cache_rows_before=%s", total_cache_rows_before)

        for query_name in TEST_PRODUCT_NAMES:
            item_start = time.perf_counter()
            logger.info("retest_start query=%s", query_name)
            try:
                selected_row, selection_reason = select_product_row(query_name, profile_df, logger)
                if not selected_row:
                    failed_rows.append(
                        {
                            "query_name": query_name,
                            "stage": "product_lookup",
                            "selection_reason": selection_reason,
                            "notes": "product not found",
                        }
                    )
                    continue

                base_product_id = str(selected_row["product_id"])
                base_profile = profiles.get(base_product_id)
                if not base_profile:
                    failed_rows.append(
                        {
                            "query_name": query_name,
                            "stage": "profile_lookup",
                            "selection_reason": selection_reason,
                            "notes": f"profile missing for product_id={base_product_id}",
                        }
                    )
                    continue

                candidate_stats = compute_candidate_pool_stats(
                    base_product_id,
                    base_profile,
                    profiles,
                    product_vectors,
                    ingredient_postings,
                    ingredient_frequency,
                    DEFAULT_CANDIDATE_LIMIT,
                    DEFAULT_MAX_DF_FOR_SEED,
                )
                rows, failed_candidates, compute_stats = compute_topk_for_single_product(
                    base_product_id,
                    DEFAULT_TOP_K,
                    DEFAULT_CANDIDATE_LIMIT,
                    DEFAULT_MAX_DF_FOR_SEED,
                    product_vectors,
                    product_names,
                    profiles,
                    ingredient_postings,
                    ingredient_frequency,
                )

                refresh_cache_rows(conn, base_product_id, rows)
                cache_rows_written = conn.execute(
                    f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE base_product_id = ?",
                    (base_product_id,),
                ).fetchone()[0]
                base_product_cache_counts[str(base_profile.get("report_no", ""))] = int(cache_rows_written)

                safe_key = safe_product_key(base_profile["report_no"] or base_profile["product_name"] or base_product_id)
                elapsed_seconds = round(time.perf_counter() - item_start, 3)
                summary_payload = {
                    "generated_at": datetime.now().isoformat(),
                    "base_product_id": base_product_id,
                    "base_product_name": base_profile["product_name"],
                    "report_no": base_profile["report_no"],
                    "product_main_category": base_profile["product_main_category"],
                    "candidate_count_before_limit": candidate_stats["candidate_count_before_limit"],
                    "candidate_count_after_limit": candidate_stats["candidate_count_after_limit"],
                    "candidate_count": compute_stats["candidate_count"],
                    "topk_count": len(rows),
                    "cache_used": False,
                    "cache_row_count_for_base_product": int(cache_rows_written),
                    "candidate_stage_seconds": compute_stats["candidate_stage_seconds"],
                    "execution_seconds": elapsed_seconds,
                    "top_k": DEFAULT_TOP_K,
                    "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
                    "max_df_for_seed": DEFAULT_MAX_DF_FOR_SEED,
                    "selection_reason": selection_reason,
                }
                write_outputs(output_dir / "similarity_top10_with_explanation", safe_key, rows, failed_candidates, summary_payload)

                result_rows = [build_result_row(base_profile, row, rank) for rank, row in enumerate(rows, start=1)]
                warnings_for_product = run_quality_checks(query_name, base_profile, result_rows)
                warnings_rows.extend(warnings_for_product)
                flat_rows.extend(result_rows)

                summary_results.append(
                    {
                        "query_name": query_name,
                        "selection_reason": selection_reason,
                        "base_product": {
                            "report_no": str(base_profile.get("report_no", "")),
                            "product_id": base_product_id,
                            "product_name": str(base_profile.get("product_name", "")),
                            "product_main_category": str(base_profile.get("product_main_category", "")),
                            "product_sub_categories_json": list(base_profile.get("product_sub_categories", [])),
                            "primary_ingredients_json": list(base_profile.get("primary_ingredients", [])),
                            "secondary_ingredients_json": list(base_profile.get("secondary_ingredients", [])),
                            "support_ingredients_json": list(base_profile.get("support_ingredients", [])),
                        },
                        "execution": {
                            "candidate_count_before_limit": candidate_stats["candidate_count_before_limit"],
                            "candidate_count_after_limit": candidate_stats["candidate_count_after_limit"],
                            "top_k_generated": len(rows),
                            "cache_rows_written": int(cache_rows_written),
                            "execution_time_sec": elapsed_seconds,
                            "failed_count": len(failed_candidates),
                        },
                        "recommendations_top10": result_rows,
                        "warnings": warnings_for_product,
                    }
                )
                logger.info(
                    "retest_done query=%s report_no=%s category=%s candidates=%s/%s topk=%s warnings=%s elapsed=%.3fs",
                    query_name,
                    base_profile.get("report_no", ""),
                    base_profile.get("product_main_category", ""),
                    candidate_stats["candidate_count_before_limit"],
                    candidate_stats["candidate_count_after_limit"],
                    len(rows),
                    len(warnings_for_product),
                    elapsed_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed_seconds = round(time.perf_counter() - item_start, 3)
                logger.exception("retest_failed query=%s elapsed=%.3fs", query_name, elapsed_seconds)
                failed_rows.append(
                    {
                        "query_name": query_name,
                        "stage": "execution",
                        "selection_reason": "",
                        "notes": str(exc),
                    }
                )

        total_cache_rows_after = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    results_jsonl_path = output_dir / "retest_10_products_results.jsonl"
    results_csv_path = output_dir / "retest_10_products_results.csv"
    warnings_csv_path = output_dir / "retest_10_products_warnings.csv"
    failed_csv_path = output_dir / "failed_retest_products.csv"
    summary_json_path = output_dir / "retest_10_products_summary.json"

    with results_jsonl_path.open("w", encoding="utf-8") as file:
        for item in summary_results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    pd.DataFrame(flat_rows).to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(warnings_rows).to_csv(warnings_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(failed_csv_path, index=False, encoding="utf-8-sig")

    total_elapsed = round(time.perf_counter() - start_time, 3)
    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "success_count": len(summary_results),
        "failure_count": len(failed_rows),
        "total_execution_time_sec": total_elapsed,
        "products": [
            {
                "query_name": item["query_name"],
                "report_no": item["base_product"]["report_no"],
                "product_name": item["base_product"]["product_name"],
                "product_main_category": item["base_product"]["product_main_category"],
                "execution_time_sec": item["execution"]["execution_time_sec"],
                "candidate_count_before_limit": item["execution"]["candidate_count_before_limit"],
                "candidate_count_after_limit": item["execution"]["candidate_count_after_limit"],
                "top_k_generated": item["execution"]["top_k_generated"],
                "warning_count": len(item["warnings"]),
            }
            for item in summary_results
        ],
        "total_cache_rows": int(total_cache_rows_after),
        "tested_base_product_count": len(base_product_cache_counts),
        "base_product_cache_counts": base_product_cache_counts,
    }
    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("summary success=%s failure=%s total_elapsed=%.3fs total_cache_rows=%s", len(summary_results), len(failed_rows), total_elapsed, total_cache_rows_after)
    logger.info("summary_json=%s", summary_json_path)
    logger.info("results_csv=%s", results_csv_path)
    logger.info("warnings_csv=%s", warnings_csv_path)
    logger.info("failed_csv=%s", failed_csv_path)


if __name__ == "__main__":
    main()
