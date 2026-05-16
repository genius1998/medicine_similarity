from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import get_config_value, load_config


TABLE_NAME = "product_similarity_explanation_cache"
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_LIMIT = 1000
DEFAULT_MAX_DF_FOR_SEED = 3000
CAUTION_TEXT = (
    "본 추천은 건강기능식품의 기능성원료, 표시 기능, 원재료 정보를 기준으로 한 비교 결과입니다. "
    "질병의 예방·치료 효과를 의미하지 않으며, 실제 섭취 전 제품 라벨의 함량, 1일 섭취량, 주의사항을 확인해야 합니다."
)

LOW_PRIORITY_PATTERNS = [
    r"^비타민\s*[a-z]?\d*$",
    r"비타민 c",
    r"비타민 d",
    r"비타민 k",
    r"비타민 b",
    r"아연",
    r"셀렌",
    r"셀레늄",
    r"마그네슘",
    r"칼슘",
    r"철",
    r"엽산",
    r"히드록시프로필메틸셀룰로오스",
    r"\bhpmc\b",
    r"난소화성말토덱스트린",
]

EXCIPIENT_PATTERNS = [
    r"히드록시프로필메틸셀룰로오스",
    r"\bhpmc\b",
    r"결정셀룰로스",
    r"스테아린산마그네슘",
    r"이산화규소",
    r"카복시메틸셀룰로오스",
    r"글리세린지방산에스테르",
]

JOINT_KEYWORDS = [r"\bnem\b", r"난각막", r"난각막가수분해물", r"\bmsm\b", r"엠에스엠", r"보스웰리아", r"초록입홍합", r"\bnag\b", r"글루코사민", r"콘드로이친", r"관절", r"연골"]
BONE_KEYWORDS = [r"칼슘", r"비타민 d", r"비타민 k", r"\bk2\b", r"마그네슘", r"뼈", r"칼마디"]
GUT_KEYWORDS = [r"프로바이오틱스", r"유산균", r"이눌린", r"치커리", r"프락토올리고당", r"난소화성말토덱스트린", r"장"]
MEMORY_KEYWORDS = [r"포스파티딜세린", r"phosphatidylserine", r"\bps\b", r"은행잎추출물", r"은행잎", r"테아닌", r"\bdha\b", r"기억", r"인지", r"두뇌", r"브레인"]
SKIN_KEYWORDS = [r"콜라겐", r"저분자콜라겐", r"저분자콜라겐펩타이드", r"콜라겐펩타이드", r"히알루론산", r"세라마이드", r"엘라스틴", r"비오틴", r"비타민 a", r"레티놀", r"셀렌", r"셀레늄", r"피부"]
MALE_KEYWORDS = [r"쏘팔메토", r"쏘팔메토열매추출물", r"saw palmetto", r"로르산", r"옥타코사놀", r"아연", r"녹용", r"마카", r"아르기닌", r"l-아르지닌", r"남성", r"전립선"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="단일 제품 기준 TOP-K 추천 + explanation 생성")
    parser.add_argument("--report-no", default="", help="품목제조번호 또는 보고번호")
    parser.add_argument("--product-id", default="", help="내부 product_id")
    parser.add_argument("--product-name", default="", help="제품명")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--max-df-for-seed", type=int, default=DEFAULT_MAX_DF_FOR_SEED)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output-prefix", default="output/similarity_top10_with_explanation")
    args = parser.parse_args()
    if not any([args.report_no.strip(), args.product_id.strip(), args.product_name.strip()]):
        parser.error("`--report-no`, `--product-id`, `--product-name` 중 하나는 필수입니다.")
    return args


def first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {[str(path) for path in candidates]}")


def resolve_output_dir(config: dict) -> Path:
    configured = Path(str(get_config_value(config, "output_dir", ROOT_DIR / "output")))
    if configured.exists():
        return configured
    return ROOT_DIR / "output"


def resolve_runtime_paths() -> dict[str, Path]:
    config = load_config()
    output_dir = resolve_output_dir(config)
    vector_candidates = [
        output_dir / "c003_product_functional_vectors_final_rebuilt.csv",
        Path(r"D:\ec2_cache_snapshot\c003_product_functional_vectors_final_rebuilt.csv"),
        Path(r"C:\Users\com\Downloads\c003_product_vector_output_final_rebuilt\c003_product_functional_vectors_final_rebuilt.csv"),
    ]
    product_profile_candidates = [
        output_dir / "product_function_profile.csv",
        ROOT_DIR / "output" / "product_function_profile.csv",
    ]
    sqlite_candidates = [
        Path(str(get_config_value(config, "sqlite_path", ""))),
        Path(r"D:\ec2_cache_snapshot\ingredient_match_cache_rebuilt_item_class_i0050_final.sqlite"),
    ]
    return {
        "output_dir": output_dir,
        "vector_csv_path": first_existing_path(vector_candidates),
        "product_profile_csv_path": first_existing_path(product_profile_candidates),
        "sqlite_path": first_existing_path([path for path in sqlite_candidates if str(path)]),
    }


def normalize_token(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def safe_product_key(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w가-힣.-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._")
    return text[:80] or "unknown_product"


def ingredient_matches_patterns(name: str, patterns: list[str]) -> bool:
    lowered = normalize_text(name)
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def is_excipient(name: str) -> bool:
    return ingredient_matches_patterns(name, EXCIPIENT_PATTERNS)


def is_low_priority_reason_ingredient(name: str, category: str) -> bool:
    if category == "뼈 건강" and ingredient_matches_patterns(name, [r"칼슘", r"비타민 d", r"비타민 k"]):
        return False
    if category in {"장 건강", "혈당"} and ingredient_matches_patterns(name, [r"난소화성말토덱스트린"]):
        return False
    return ingredient_matches_patterns(name, LOW_PRIORITY_PATTERNS)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_vector_inputs(vector_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(vector_csv_path, encoding="utf-8-sig", low_memory=False)
    required = ["product_id", "product_name", "matched_standard_name", "weight"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"벡터 CSV 필수 컬럼이 없습니다: {missing}")
    df["product_id"] = df["product_id"].fillna("").astype(str).str.strip()
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()
    df["matched_standard_name"] = df["matched_standard_name"].fillna("").astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    for column in ["품목제조번호", "품목보고번호"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
    df = df[(df["product_id"] != "") & (df["matched_standard_name"] != "") & (df["weight"] > 0)].copy()
    return df


def safe_json_loads(value: object, default):
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return default


def load_product_function_profiles(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    profiles = {}
    for row in df.itertuples(index=False):
        ingredient_scores = safe_json_loads(getattr(row, "ingredient_scores_json", "[]"), [])
        role_by_ingredient = {}
        category_by_ingredient = {}
        for item in ingredient_scores:
            name = str(item.get("ingredient", "")).strip()
            if not name:
                continue
            role_by_ingredient[name] = str(item.get("role", "")).strip()
            category_by_ingredient[name] = str(item.get("category_main", "")).strip()
        profile = {
            "product_id": str(row.product_id or ""),
            "report_no": str(row.report_no or ""),
            "product_name": str(row.product_name or ""),
            "product_main_category": str(row.product_main_category or "기타"),
            "product_sub_categories": safe_json_loads(getattr(row, "product_sub_categories_json", "[]"), []),
            "primary_ingredients": safe_json_loads(getattr(row, "primary_ingredients_json", "[]"), []),
            "secondary_ingredients": safe_json_loads(getattr(row, "secondary_ingredients_json", "[]"), []),
            "support_ingredients": safe_json_loads(getattr(row, "support_ingredients_json", "[]"), []),
            "category_scores": safe_json_loads(getattr(row, "category_scores_json", "{}"), {}),
            "ingredient_scores": ingredient_scores,
            "role_by_ingredient": role_by_ingredient,
            "category_by_ingredient": category_by_ingredient,
            "confidence": float(getattr(row, "confidence", 0.0) or 0.0),
            "notes": str(getattr(row, "notes", "") or ""),
        }
        profiles[profile["product_id"]] = profile
    return profiles


def build_vector_indexes(vector_df: pd.DataFrame):
    product_vectors: dict[str, dict[str, float]] = {}
    product_names: dict[str, str] = {}
    report_to_product_ids: dict[str, list[str]] = defaultdict(list)
    ingredient_postings: dict[str, list[str]] = defaultdict(list)
    ingredient_frequency: dict[str, int] = defaultdict(int)

    for row in vector_df.itertuples(index=False):
        product_id = str(row.product_id)
        ingredient = str(row.matched_standard_name)
        product_vectors.setdefault(product_id, {})
        product_vectors[product_id][ingredient] = max(
            float(row.weight),
            product_vectors[product_id].get(ingredient, 0.0),
        )
        product_names[product_id] = str(row.product_name)

    for row in vector_df.drop_duplicates(subset=["product_id"]).itertuples(index=False):
        product_id = str(row.product_id)
        for column in ["품목제조번호", "품목보고번호"]:
            if hasattr(row, column):
                value = str(getattr(row, column) or "").strip()
                if value and value.lower() != "nan" and product_id not in report_to_product_ids[value]:
                    report_to_product_ids[value].append(product_id)

    for product_id, vector in product_vectors.items():
        for ingredient in vector:
            ingredient_postings[ingredient].append(product_id)
            ingredient_frequency[ingredient] += 1

    return product_vectors, product_names, report_to_product_ids, ingredient_postings, ingredient_frequency


def resolve_base_product_id(args: argparse.Namespace, profiles: dict[str, dict], product_names: dict[str, str], report_to_product_ids: dict[str, list[str]]) -> str:
    if args.product_id.strip():
        product_id = args.product_id.strip()
        if product_id in profiles:
            return product_id
        raise ValueError(f"product_id를 찾지 못했습니다: {product_id}")

    if args.report_no.strip():
        report_no = args.report_no.strip()
        matches = report_to_product_ids.get(report_no, [])
        if matches:
            return matches[0]
        raise ValueError(f"report_no를 찾지 못했습니다: {report_no}")

    normalized_query = normalize_token(args.product_name)
    exact_matches = [product_id for product_id, name in product_names.items() if normalize_token(name) == normalized_query]
    if exact_matches:
        return exact_matches[0]
    partial_matches = [product_id for product_id, name in product_names.items() if normalized_query and normalized_query in normalize_token(name)]
    if partial_matches:
        return partial_matches[0]
    raise ValueError(f"product_name을 찾지 못했습니다: {args.product_name}")


def role_factor(role: str) -> float:
    if role == "primary":
        return 1.15
    if role == "secondary":
        return 1.0
    if role == "support":
        return 0.35
    return 0.8


def ingredient_idf(total_product_count: int, ingredient_df: int) -> float:
    return math.log((1.0 + total_product_count) / (1.0 + ingredient_df)) + 1.0


def calculate_weighted_jaccard_with_idf(
    base_vector: dict[str, float],
    target_vector: dict[str, float],
    ingredient_frequency: dict[str, int],
    total_product_count: int,
    base_profile: dict,
    target_profile: dict,
) -> tuple[float, list[str]]:
    union_ingredients = set(base_vector) | set(target_vector)
    shared_ingredients = sorted(set(base_vector) & set(target_vector))
    numerator = 0.0
    denominator = 0.0
    for ingredient in union_ingredients:
        base_weight = float(base_vector.get(ingredient, 0.0))
        target_weight = float(target_vector.get(ingredient, 0.0))
        idf = ingredient_idf(total_product_count, ingredient_frequency.get(ingredient, 0))
        base_role = base_profile["role_by_ingredient"].get(ingredient, "")
        target_role = target_profile["role_by_ingredient"].get(ingredient, "")
        effective_base = base_weight * idf * role_factor(base_role)
        effective_target = target_weight * idf * role_factor(target_role)
        numerator += min(effective_base, effective_target)
        denominator += max(effective_base, effective_target)
    similarity = 0.0 if denominator <= 0 else numerator / denominator
    return round(float(similarity), 6), shared_ingredients


def calculate_function_similarity(base_profile: dict, target_profile: dict) -> float:
    scores_a = {str(k): float(v) for k, v in dict(base_profile.get("category_scores", {})).items()}
    scores_b = {str(k): float(v) for k, v in dict(target_profile.get("category_scores", {})).items()}
    keys = set(scores_a) | set(scores_b)
    if not keys:
        return 0.0
    numerator = sum(min(scores_a.get(key, 0.0), scores_b.get(key, 0.0)) for key in keys)
    denominator = sum(max(scores_a.get(key, 0.0), scores_b.get(key, 0.0)) for key in keys)
    return 0.0 if denominator <= 0 else round(float(numerator / denominator), 6)


def calculate_core_match_score(base_profile: dict, target_profile: dict) -> float:
    base_primary = set(base_profile.get("primary_ingredients", []))
    target_primary = set(target_profile.get("primary_ingredients", []))
    union = base_primary | target_primary
    if not union:
        return 0.0
    return round(float(len(base_primary & target_primary) / len(union)), 6)


def compare_product_profiles(base_profile: dict, target_profile: dict, base_vector: dict[str, float], target_vector: dict[str, float]) -> dict:
    shared_ingredients_raw = sorted(set(base_vector) & set(target_vector))
    shared_ingredients_for_display = [name for name in shared_ingredients_raw if not is_excipient(name)]
    base_only_ingredients = sorted(set(base_vector) - set(target_vector))
    target_only_ingredients = sorted(set(target_vector) - set(base_vector))

    def profile_categories(profile: dict) -> set[str]:
        categories = {str(profile.get("product_main_category", "기타"))}
        categories.update([str(item) for item in profile.get("product_sub_categories", []) if str(item).strip()])
        for item in profile.get("ingredient_scores", []):
            if str(item.get("role", "")) != "support" and not is_excipient(str(item.get("ingredient", ""))):
                category = str(item.get("category_main", "")).strip()
                if category:
                    categories.add(category)
        return {item for item in categories if item and item != "기타"} | ({str(profile.get("product_main_category", "기타"))} if profile.get("product_main_category") else set())

    base_categories = profile_categories(base_profile)
    target_categories = profile_categories(target_profile)
    return {
        "shared_ingredients_raw": shared_ingredients_raw,
        "shared_ingredients": shared_ingredients_for_display,
        "base_only_ingredients": base_only_ingredients,
        "target_only_ingredients": target_only_ingredients,
        "shared_categories": sorted(base_categories & target_categories),
        "different_categories": sorted((base_categories | target_categories) - (base_categories & target_categories)),
    }


def classify_substitutability(similarity_score: float, base_profile: dict, target_profile: dict) -> str:
    same_main = base_profile.get("product_main_category") == target_profile.get("product_main_category")
    shared_primary = set(base_profile.get("primary_ingredients", [])) & set(target_profile.get("primary_ingredients", []))
    if similarity_score >= 0.75 and same_main and shared_primary:
        return "높음"
    if similarity_score >= 0.45 or same_main:
        return "보통"
    return "낮음"


def product_is_joint_related(profile: dict) -> bool:
    text = normalize_text(profile.get("product_name", ""))
    if profile.get("product_main_category") == "관절/연골":
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in JOINT_KEYWORDS)


def product_is_bone_related(profile: dict) -> bool:
    text = normalize_text(profile.get("product_name", ""))
    if profile.get("product_main_category") == "뼈 건강":
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in BONE_KEYWORDS)


def product_is_memory_related(profile: dict) -> bool:
    text = normalize_text(profile.get("product_name", ""))
    if profile.get("product_main_category") in {"기억력", "인지력", "기억력/인지력"}:
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in MEMORY_KEYWORDS)


def product_is_skin_related(profile: dict) -> bool:
    text = normalize_text(profile.get("product_name", ""))
    if profile.get("product_main_category") == "피부 건강":
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SKIN_KEYWORDS)


def product_is_male_related(profile: dict) -> bool:
    text = normalize_text(profile.get("product_name", ""))
    if profile.get("product_main_category") == "남성 건강":
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in MALE_KEYWORDS)


def choose_reason_ingredients(base_profile: dict, target_profile: dict, comparison: dict, ingredient_frequency: dict[str, int]) -> tuple[list[str], list[str]]:
    shared = [name for name in comparison["shared_ingredients"] if not is_excipient(name)]
    if not shared:
        return [], []

    base_primary = set(base_profile.get("primary_ingredients", []))
    target_primary = set(target_profile.get("primary_ingredients", []))
    base_secondary = set(base_profile.get("secondary_ingredients", []))
    target_secondary = set(target_profile.get("secondary_ingredients", []))
    base_support = set(base_profile.get("support_ingredients", []))
    target_support = set(target_profile.get("support_ingredients", []))
    category = str(base_profile.get("product_main_category") or "기타")

    joint_related = product_is_joint_related(base_profile) or product_is_joint_related(target_profile)
    bone_related = product_is_bone_related(base_profile) or product_is_bone_related(target_profile)
    memory_related = product_is_memory_related(base_profile) or product_is_memory_related(target_profile)
    skin_related = product_is_skin_related(base_profile) or product_is_skin_related(target_profile)
    male_related = product_is_male_related(base_profile) or product_is_male_related(target_profile)

    if category == "관절/연골":
        category_patterns = JOINT_KEYWORDS
    elif category == "뼈 건강":
        category_patterns = BONE_KEYWORDS
    elif category == "장 건강":
        category_patterns = GUT_KEYWORDS
    elif category in {"기억력", "인지력", "기억력/인지력"}:
        category_patterns = MEMORY_KEYWORDS
    elif category == "피부 건강":
        category_patterns = SKIN_KEYWORDS
    elif category == "남성 건강":
        category_patterns = MALE_KEYWORDS
    else:
        category_patterns = []

    def ingredient_priority(name: str) -> tuple:
        shared_primary = int(name in base_primary and name in target_primary)
        direct_category = int(ingredient_matches_patterns(name, category_patterns))
        joint_core = int(joint_related and ingredient_matches_patterns(name, JOINT_KEYWORDS))
        memory_core = int(memory_related and ingredient_matches_patterns(name, MEMORY_KEYWORDS))
        skin_core = int(skin_related and ingredient_matches_patterns(name, SKIN_KEYWORDS))
        male_core = int(male_related and ingredient_matches_patterns(name, MALE_KEYWORDS))
        secondary_shared = int(name in base_secondary and name in target_secondary)
        support_shared = int(name in base_support and name in target_support)
        low_priority = int(is_low_priority_reason_ingredient(name, category))
        rarity = ingredient_frequency.get(name, 999999)
        return (
            joint_core,
            memory_core,
            skin_core,
            male_core,
            shared_primary,
            direct_category,
            -low_priority,
            secondary_shared,
            -support_shared,
            -min(rarity, 999999),
            len(name),
        )

    ranked = sorted(shared, key=ingredient_priority, reverse=True)
    top = ranked[:3]

    joint_ingredients = [name for name in ranked if ingredient_matches_patterns(name, JOINT_KEYWORDS)]
    bone_ingredients = [name for name in ranked if ingredient_matches_patterns(name, BONE_KEYWORDS) and not is_excipient(name)]
    memory_ingredients = [name for name in ranked if ingredient_matches_patterns(name, MEMORY_KEYWORDS)]
    skin_ingredients = [name for name in ranked if ingredient_matches_patterns(name, SKIN_KEYWORDS)]
    male_ingredients = [name for name in ranked if ingredient_matches_patterns(name, MALE_KEYWORDS)]
    return top, list(dict.fromkeys(joint_ingredients[:3] + bone_ingredients[:3] + memory_ingredients[:3] + skin_ingredients[:3] + male_ingredients[:3]))


def render_reason_ingredient_phrase(ingredients: list[str]) -> str:
    if not ingredients:
        return "공통 원료"
    if len(ingredients) == 1:
        value = str(ingredients[0]).strip()
        return f"{value} 성분이"
    if len(ingredients) == 2:
        return f"{ingredients[0]}, {ingredients[1]} 등이"
    return f"{ingredients[0]}, {ingredients[1]}, {ingredients[2]} 등이"


def generate_recommendation_reason(base_profile: dict, target_profile: dict, comparison: dict, ingredient_frequency: dict[str, int]) -> str:
    main_category = str(base_profile.get("product_main_category") or "기타")
    top_reason_ingredients, highlighted = choose_reason_ingredients(base_profile, target_profile, comparison, ingredient_frequency)
    phrase = render_reason_ingredient_phrase(top_reason_ingredients)
    shared_primary = set(base_profile.get("primary_ingredients", [])) & set(target_profile.get("primary_ingredients", []))
    same_main = base_profile.get("product_main_category") == target_profile.get("product_main_category")
    joint_related = product_is_joint_related(base_profile) or product_is_joint_related(target_profile)
    bone_related = product_is_bone_related(base_profile) or product_is_bone_related(target_profile)
    memory_related = product_is_memory_related(base_profile) or product_is_memory_related(target_profile)
    skin_related = product_is_skin_related(base_profile) or product_is_skin_related(target_profile)
    male_related = product_is_male_related(base_profile) or product_is_male_related(target_profile)
    joint_shared = [name for name in highlighted if ingredient_matches_patterns(name, JOINT_KEYWORDS)]
    bone_shared = [name for name in highlighted if ingredient_matches_patterns(name, BONE_KEYWORDS)]
    memory_shared = [name for name in highlighted if ingredient_matches_patterns(name, MEMORY_KEYWORDS)]
    skin_shared = [name for name in highlighted if ingredient_matches_patterns(name, SKIN_KEYWORDS)]
    male_shared = [name for name in highlighted if ingredient_matches_patterns(name, MALE_KEYWORDS)]

    if joint_related and joint_shared:
        first_sentence = f"{render_reason_ingredient_phrase(joint_shared[:3])} 공통으로 포함되어 관절/연골 건강 기능성 측면에서 유사합니다."
        if bone_related and bone_shared:
            second_sentence = f"칼슘, 비타민 D, 비타민 K 등이 공통으로 포함되어 뼈 건강 관련 구성도 함께 확인됩니다."
            return f"{first_sentence} {second_sentence}"
        return first_sentence

    if main_category in {"기억력", "인지력", "기억력/인지력"} and memory_related and memory_shared:
        ps_shared = [name for name in memory_shared if ingredient_matches_patterns(name, [r"포스파티딜세린", r"phosphatidylserine", r"\bps\b"])]
        if ps_shared:
            return f"{render_reason_ingredient_phrase(ps_shared[:1])} 공통으로 포함되어 기억력/인지력 기능성 측면에서 유사합니다."
        return f"{render_reason_ingredient_phrase(memory_shared[:3])} 공통으로 포함되어 기억력/인지력 기능성 측면에서 유사합니다."

    if main_category == "피부 건강" and skin_related and skin_shared:
        return f"{render_reason_ingredient_phrase(skin_shared[:3])} 공통으로 포함되어 피부 건강 기능성 측면에서 유사합니다."

    if main_category == "남성 건강" and male_related and male_shared:
        return f"{render_reason_ingredient_phrase(male_shared[:3])} 공통으로 포함되어 남성 건강 관련 기능성 측면에서 유사합니다."

    if main_category == "뼈 건강" and bone_shared:
        first_sentence = f"{render_reason_ingredient_phrase(bone_shared[:3])} 공통으로 포함되어 뼈 건강 기능성 측면에서 유사합니다."
        joint_extra = [name for name in highlighted if ingredient_matches_patterns(name, JOINT_KEYWORDS)]
        if joint_extra:
            second_sentence = f"또한 {', '.join(joint_extra[:3])} 등 관절/연골 관련 원료도 함께 확인됩니다."
            return f"{first_sentence} {second_sentence}"
        return first_sentence

    if shared_primary and main_category != "기타":
        return f"{phrase} 공통으로 포함되어 {main_category} 기능성 측면에서 유사합니다."
    if same_main and main_category != "기타":
        return f"두 제품 모두 {main_category} 계열이지만, 핵심 기능성원료 구성에는 차이가 있습니다."
    if top_reason_ingredients and main_category != "기타":
        return f"{phrase} 공통으로 포함되어 {main_category} 기능성 측면에서 유사합니다."
    return "일부 기능성원료는 공통으로 확인되지만, 대표 기능 또는 핵심 원료가 달라 대체 후보로 비교할 때 제한이 있습니다."


def generate_caution() -> str:
    return CAUTION_TEXT


def build_explanation_json(reason: str, comparison: dict, substitutability: str) -> dict:
    return {
        "reason": reason,
        "shared_ingredients": comparison["shared_ingredients"],
        "base_only_ingredients": comparison["base_only_ingredients"],
        "target_only_ingredients": comparison["target_only_ingredients"],
        "shared_categories": comparison["shared_categories"],
        "different_categories": comparison["different_categories"],
        "substitutability": substitutability,
        "caution": generate_caution(),
    }


def build_candidate_pool(
    base_product_id: str,
    base_profile: dict,
    product_vectors: dict[str, dict[str, float]],
    profiles: dict[str, dict],
    ingredient_postings: dict[str, list[str]],
    ingredient_frequency: dict[str, int],
    candidate_limit: int,
    max_df_for_seed: int,
) -> list[dict]:
    candidate_stats: dict[str, dict] = {}
    base_categories = {str(base_profile.get("product_main_category", "기타"))}
    base_categories.update([str(item) for item in base_profile.get("product_sub_categories", []) if str(item).strip()])

    def ensure_candidate(candidate_id: str) -> dict:
        if candidate_id not in candidate_stats:
            candidate_profile = profiles[candidate_id]
            candidate_categories = {str(candidate_profile.get("product_main_category", "기타"))}
            candidate_categories.update([str(item) for item in candidate_profile.get("product_sub_categories", []) if str(item).strip()])
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
                stat["preliminary_score"] += score_weight * ingredient_idf(len(product_vectors), max(1, ingredient_df))

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
    return rows[:candidate_limit]


def ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            base_product_id TEXT NOT NULL,
            target_product_id TEXT NOT NULL,
            base_product_name TEXT,
            target_product_name TEXT,
            similarity_score REAL,
            function_similarity_score REAL,
            core_match_score REAL,
            shared_ingredients_json TEXT,
            base_only_ingredients_json TEXT,
            target_only_ingredients_json TEXT,
            shared_categories_json TEXT,
            different_categories_json TEXT,
            substitutability TEXT,
            reason TEXT,
            caution TEXT,
            explanation_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (base_product_id, target_product_id)
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_base_product_id ON {TABLE_NAME} (base_product_id)")
    conn.commit()


def load_cached_rows(conn: sqlite3.Connection, base_product_id: str, top_k: int) -> list[dict]:
    query = f"""
    SELECT base_product_id, target_product_id, base_product_name, target_product_name,
           similarity_score, function_similarity_score, core_match_score,
           shared_ingredients_json, base_only_ingredients_json, target_only_ingredients_json,
           shared_categories_json, different_categories_json, substitutability,
           reason, caution, explanation_json
    FROM {TABLE_NAME}
    WHERE base_product_id = ?
    ORDER BY similarity_score DESC, target_product_id ASC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(base_product_id, top_k))
    return df.to_dict(orient="records")


def refresh_cache_rows(conn: sqlite3.Connection, base_product_id: str, rows: list[dict]) -> None:
    now_text = datetime.now().isoformat()
    conn.execute(f"DELETE FROM {TABLE_NAME} WHERE base_product_id = ?", (base_product_id,))
    conn.executemany(
        f"""
        INSERT INTO {TABLE_NAME} (
            base_product_id, target_product_id, base_product_name, target_product_name,
            similarity_score, function_similarity_score, core_match_score,
            shared_ingredients_json, base_only_ingredients_json, target_only_ingredients_json,
            shared_categories_json, different_categories_json, substitutability,
            reason, caution, explanation_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["base_product_id"],
                row["target_product_id"],
                row["base_product_name"],
                row["target_product_name"],
                row["similarity_score"],
                row["function_similarity_score"],
                row["core_match_score"],
                row["shared_ingredients_json"],
                row["base_only_ingredients_json"],
                row["target_only_ingredients_json"],
                row["shared_categories_json"],
                row["different_categories_json"],
                row["substitutability"],
                row["reason"],
                row["caution"],
                row["explanation_json"],
                now_text,
            )
            for row in rows
        ],
    )
    conn.commit()


def write_outputs(output_prefix: Path, safe_key: str, rows: list[dict], failed_rows: list[dict], summary: dict) -> dict[str, Path]:
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_name = output_prefix.name
    output_csv_path = output_dir / f"{prefix_name}_{safe_key}.csv"
    output_jsonl_path = output_dir / f"{prefix_name}_{safe_key}.jsonl"
    summary_json_path = output_dir / f"explanation_summary_{safe_key}.json"
    failed_csv_path = output_dir / f"failed_explanation_{safe_key}.csv"

    pd.DataFrame(rows).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    save_jsonl(output_jsonl_path, rows)
    pd.DataFrame(failed_rows).to_csv(failed_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": output_csv_path, "jsonl": output_jsonl_path, "summary": summary_json_path, "failed": failed_csv_path}


def compute_topk_for_single_product(
    base_product_id: str,
    top_k: int,
    candidate_limit: int,
    max_df_for_seed: int,
    product_vectors: dict[str, dict[str, float]],
    product_names: dict[str, str],
    profiles: dict[str, dict],
    ingredient_postings: dict[str, list[str]],
    ingredient_frequency: dict[str, int],
) -> tuple[list[dict], list[dict], dict]:
    base_profile = profiles[base_product_id]
    base_vector = product_vectors[base_product_id]
    total_product_count = len(product_vectors)

    t0 = time.perf_counter()
    candidate_pool = build_candidate_pool(
        base_product_id,
        base_profile,
        product_vectors,
        profiles,
        ingredient_postings,
        ingredient_frequency,
        candidate_limit,
        max_df_for_seed,
    )
    candidate_stage_seconds = round(time.perf_counter() - t0, 3)
    print(f"[INFO] base_product={base_profile['product_name']} ({base_product_id})")
    print(f"[INFO] candidate_count_before_topk={len(candidate_pool)}")

    rows = []
    failed_rows = []
    for candidate in candidate_pool:
        target_product_id = candidate["target_product_id"]
        try:
            target_profile = profiles[target_product_id]
            target_vector = product_vectors[target_product_id]
            similarity_score, _ = calculate_weighted_jaccard_with_idf(
                base_vector,
                target_vector,
                ingredient_frequency,
                total_product_count,
                base_profile,
                target_profile,
            )
            if similarity_score <= 0:
                continue
            comparison = compare_product_profiles(base_profile, target_profile, base_vector, target_vector)
            function_similarity_score = calculate_function_similarity(base_profile, target_profile)
            core_match_score = calculate_core_match_score(base_profile, target_profile)
            substitutability = classify_substitutability(similarity_score, base_profile, target_profile)
            reason = generate_recommendation_reason(base_profile, target_profile, comparison, ingredient_frequency)
            explanation_json = build_explanation_json(reason, comparison, substitutability)
            rows.append(
                {
                    "base_product_id": base_product_id,
                    "target_product_id": target_product_id,
                    "base_product_name": base_profile["product_name"],
                    "target_product_name": target_profile["product_name"],
                    "similarity_score": similarity_score,
                    "function_similarity_score": function_similarity_score,
                    "core_match_score": core_match_score,
                    "shared_ingredients_json": json.dumps(comparison["shared_ingredients"], ensure_ascii=False),
                    "shared_ingredients_raw_json": json.dumps(comparison["shared_ingredients_raw"], ensure_ascii=False),
                    "base_only_ingredients_json": json.dumps(comparison["base_only_ingredients"], ensure_ascii=False),
                    "target_only_ingredients_json": json.dumps(comparison["target_only_ingredients"], ensure_ascii=False),
                    "shared_categories_json": json.dumps(comparison["shared_categories"], ensure_ascii=False),
                    "different_categories_json": json.dumps(comparison["different_categories"], ensure_ascii=False),
                    "substitutability": substitutability,
                    "reason": reason,
                    "caution": generate_caution(),
                    "explanation_json": json.dumps(explanation_json, ensure_ascii=False),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed_rows.append(
                {
                    "base_product_id": base_product_id,
                    "target_product_id": target_product_id,
                    "base_product_name": base_profile["product_name"],
                    "target_product_name": product_names.get(target_product_id, ""),
                    "notes": f"error: {exc}",
                }
            )

    rows = sorted(
        rows,
        key=lambda item: (
            -item["similarity_score"],
            -item["core_match_score"],
            -item["function_similarity_score"],
            item["target_product_id"],
        ),
    )[:top_k]
    print(f"[INFO] candidate_count_after_scoring={len(rows)}")
    print("[SAMPLE] top results")
    print(pd.DataFrame(rows).head(10).to_string(index=False) if rows else "No result")
    return rows, failed_rows, {"candidate_count": len(candidate_pool), "topk_count": len(rows), "candidate_stage_seconds": candidate_stage_seconds}


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    runtime = resolve_runtime_paths()
    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = ROOT_DIR / output_prefix

    print(f"[INFO] vector_csv_path={runtime['vector_csv_path']}")
    print(f"[INFO] product_profile_csv_path={runtime['product_profile_csv_path']}")
    print(f"[INFO] sqlite_path={runtime['sqlite_path']}")

    vector_df = load_vector_inputs(runtime["vector_csv_path"])
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    product_vectors, product_names, report_to_product_ids, ingredient_postings, ingredient_frequency = build_vector_indexes(vector_df)

    base_product_id = resolve_base_product_id(args, profiles, product_names, report_to_product_ids)
    base_profile = profiles[base_product_id]
    safe_key = safe_product_key(base_profile["report_no"] or base_profile["product_name"] or base_product_id)

    with sqlite3.connect(runtime["sqlite_path"]) as conn:
        ensure_cache_table(conn)
        cached_rows = []
        if not args.force_refresh:
            cached_rows = load_cached_rows(conn, base_product_id, args.top_k)
        if len(cached_rows) >= args.top_k:
            elapsed_seconds = round(time.perf_counter() - start_time, 3)
            summary = {
                "generated_at": datetime.now().isoformat(),
                "base_product_id": base_product_id,
                "base_product_name": base_profile["product_name"],
                "report_no": base_profile["report_no"],
                "candidate_count": 0,
                "topk_count": len(cached_rows),
                "cache_used": True,
                "execution_seconds": elapsed_seconds,
            }
            output_paths = write_outputs(output_prefix, safe_key, cached_rows, [], summary)
            print(f"[INFO] cache_hit base_product_id={base_product_id} rows={len(cached_rows)}")
            print(pd.DataFrame(cached_rows).head(10).to_string(index=False))
            print(f"[DONE] csv={output_paths['csv']}")
            print(f"[DONE] execution_seconds={elapsed_seconds}")
            return

        rows, failed_rows, stats = compute_topk_for_single_product(
            base_product_id,
            args.top_k,
            args.candidate_limit,
            args.max_df_for_seed,
            product_vectors,
            product_names,
            profiles,
            ingredient_postings,
            ingredient_frequency,
        )
        refresh_cache_rows(conn, base_product_id, rows)
        cache_row_count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE base_product_id = ?", (base_product_id,)).fetchone()[0]

    elapsed_seconds = round(time.perf_counter() - start_time, 3)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "base_product_id": base_product_id,
        "base_product_name": base_profile["product_name"],
        "report_no": base_profile["report_no"],
        "product_main_category": base_profile["product_main_category"],
        "candidate_count": stats["candidate_count"],
        "topk_count": len(rows),
        "cache_used": False,
        "cache_row_count_for_base_product": int(cache_row_count),
        "candidate_stage_seconds": stats["candidate_stage_seconds"],
        "execution_seconds": elapsed_seconds,
        "top_k": args.top_k,
        "candidate_limit": args.candidate_limit,
        "max_df_for_seed": args.max_df_for_seed,
    }
    output_paths = write_outputs(output_prefix, safe_key, rows, failed_rows, summary)
    print(f"[DONE] csv={output_paths['csv']}")
    print(f"[DONE] jsonl={output_paths['jsonl']}")
    print(f"[DONE] summary={output_paths['summary']}")
    print(f"[DONE] failed={output_paths['failed']}")
    print(f"[DONE] cache_row_count_for_base_product={cache_row_count}")
    print(f"[DONE] execution_seconds={elapsed_seconds}")


if __name__ == "__main__":
    main()
