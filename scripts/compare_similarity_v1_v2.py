from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.enhance_similarity_with_explanation import (
    SIMILARITY_ALGORITHM_V1,
    SIMILARITY_ALGORITHM_V2,
    calculate_function_similarity,
    calculate_semantic_weighted_jaccard_v2,
    calculate_weighted_jaccard_with_idf,
    compute_topk_for_single_product,
    load_ingredient_category_profiles,
    load_product_function_profiles,
    load_vector_inputs,
    normalize_token,
    resolve_runtime_paths,
    build_vector_indexes,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "semantic_jaccard_v2"
DEFAULT_PAIR_QUERIES = [
    ("(구) 무르핀", "관절사랑1000"),
    ("100세팔팔관절엔환", "관절사랑1000"),
]
DEFAULT_BASE_QUERIES = [
    "(구) 무르핀",
    "관절사랑1000",
    "100세팔팔관절엔환",
    "오메가3",
    "홍삼",
    "루테인",
    "밀크씨슬",
    "프로바이오틱스",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="role_component_v2와 semantic_weighted_jaccard_v2 비교")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=300)
    parser.add_argument("--max-df-for-seed", type=int, default=3000)
    parser.add_argument("--ingredient-category-profile", default="")
    return parser.parse_args()


def find_product_id(query: str, product_names: dict[str, str]) -> str:
    normalized_query = normalize_token(query)
    exact_matches = [product_id for product_id, name in product_names.items() if normalize_token(name) == normalized_query]
    if exact_matches:
        return exact_matches[0]
    partial_matches = [product_id for product_id, name in product_names.items() if normalized_query and normalized_query in normalize_token(name)]
    if partial_matches:
        return partial_matches[0]
    raise KeyError(f"제품명을 찾을 수 없습니다: {query}")


def pair_score_row(
    base_query: str,
    target_query: str,
    product_vectors: dict[str, dict[str, float]],
    product_names: dict[str, str],
    profiles: dict[str, dict],
    ingredient_frequency: dict[str, int],
    ingredient_category_profiles: dict[str, dict],
) -> dict:
    base_product_id = find_product_id(base_query, product_names)
    target_product_id = find_product_id(target_query, product_names)
    base_profile = profiles[base_product_id]
    target_profile = profiles[target_product_id]
    v1_score, v1_shared = calculate_weighted_jaccard_with_idf(
        product_vectors[base_product_id],
        product_vectors[target_product_id],
        ingredient_frequency,
        len(product_vectors),
        base_profile,
        target_profile,
    )
    v2_score, v2_shared, v2_detail = calculate_semantic_weighted_jaccard_v2(
        base_profile,
        target_profile,
        ingredient_category_profiles,
    )
    return {
        "base_query": base_query,
        "target_query": target_query,
        "base_product_id": base_product_id,
        "target_product_id": target_product_id,
        "base_product_name": base_profile.get("product_name", ""),
        "target_product_name": target_profile.get("product_name", ""),
        "base_main_category": base_profile.get("product_main_category", ""),
        "target_main_category": target_profile.get("product_main_category", ""),
        "v1_score": v1_score,
        "v2_score": v2_score,
        "function_similarity_score": calculate_function_similarity(base_profile, target_profile),
        "v1_shared_ingredients_json": json.dumps(v1_shared, ensure_ascii=False),
        "v2_shared_semantic_ingredients_json": json.dumps(v2_shared, ensure_ascii=False),
        "v2_semantic_detail_json": json.dumps(v2_detail, ensure_ascii=False),
    }


def topk_rows_for_base(
    base_query: str,
    product_vectors: dict[str, dict[str, float]],
    product_names: dict[str, str],
    profiles: dict[str, dict],
    ingredient_postings: dict[str, list[str]],
    ingredient_frequency: dict[str, int],
    ingredient_category_profiles: dict[str, dict],
    top_k: int,
    candidate_limit: int,
    max_df_for_seed: int,
) -> list[dict]:
    base_product_id = find_product_id(base_query, product_names)
    output_rows = []
    for algorithm in (SIMILARITY_ALGORITHM_V1, SIMILARITY_ALGORITHM_V2):
        rows, failed_rows, stats = compute_topk_for_single_product(
            base_product_id,
            top_k,
            candidate_limit,
            max_df_for_seed,
            product_vectors,
            product_names,
            profiles,
            ingredient_postings,
            ingredient_frequency,
            algorithm,
            ingredient_category_profiles,
        )
        for rank, row in enumerate(rows, start=1):
            output_rows.append(
                {
                    "base_query": base_query,
                    "similarity_algorithm": algorithm,
                    "rank": rank,
                    "base_product_id": row.get("base_product_id", ""),
                    "target_product_id": row.get("target_product_id", ""),
                    "base_product_name": row.get("base_product_name", ""),
                    "target_product_name": row.get("target_product_name", ""),
                    "similarity_score": row.get("similarity_score", 0.0),
                    "function_similarity_score": row.get("function_similarity_score", 0.0),
                    "core_match_score": row.get("core_match_score", 0.0),
                    "shared_ingredients_json": row.get("shared_ingredients_json", "[]"),
                    "shared_categories_json": row.get("shared_categories_json", "[]"),
                    "candidate_count": stats.get("candidate_count", 0),
                    "failed_count": len(failed_rows),
                }
            )
    return output_rows


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime_paths()
    ingredient_profile_path = Path(args.ingredient_category_profile) if str(args.ingredient_category_profile or "").strip() else runtime.get("ingredient_category_profile_path")
    vector_df = load_vector_inputs(runtime["vector_csv_path"])
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    product_vectors, product_names, _report_to_product_ids, ingredient_postings, ingredient_frequency = build_vector_indexes(vector_df)
    ingredient_category_profiles = load_ingredient_category_profiles(ingredient_profile_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    for base_query, target_query in DEFAULT_PAIR_QUERIES:
        try:
            pair_rows.append(
                pair_score_row(
                    base_query,
                    target_query,
                    product_vectors,
                    product_names,
                    profiles,
                    ingredient_frequency,
                    ingredient_category_profiles,
                )
            )
        except Exception as exc:  # noqa: BLE001
            pair_rows.append({"base_query": base_query, "target_query": target_query, "error": str(exc)})

    topk_rows = []
    for base_query in DEFAULT_BASE_QUERIES:
        try:
            topk_rows.extend(
                topk_rows_for_base(
                    base_query,
                    product_vectors,
                    product_names,
                    profiles,
                    ingredient_postings,
                    ingredient_frequency,
                    ingredient_category_profiles,
                    args.top_k,
                    args.candidate_limit,
                    args.max_df_for_seed,
                )
            )
        except Exception as exc:  # noqa: BLE001
            topk_rows.append({"base_query": base_query, "error": str(exc)})

    pair_output = output_dir / "v1_v2_pair_comparison.csv"
    topk_output = output_dir / "v1_v2_topk_comparison.csv"
    pd.DataFrame(pair_rows).to_csv(pair_output, index=False, encoding="utf-8-sig")
    pd.DataFrame(topk_rows).to_csv(topk_output, index=False, encoding="utf-8-sig")
    print(f"[DONE] ingredient_category_profile={ingredient_profile_path or '(fallback only)'}")
    print(f"[DONE] pair_output={pair_output}")
    print(pd.DataFrame(pair_rows).to_string(index=False))
    print(f"[DONE] topk_output={topk_output}")
    print(pd.DataFrame(topk_rows).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
