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

from api.ingredient_family import collect_joint_ingredient_families, infer_joint_ingredient_family
from config_loader import get_config_value, load_config


TABLE_NAME = "product_similarity_explanation_cache"
SIMILARITY_ALGORITHM_V1 = "role_component_v2"
SIMILARITY_ALGORITHM_V2 = "semantic_weighted_jaccard_v2"
SIMILARITY_ALGORITHM_VERSION = SIMILARITY_ALGORITHM_V1
SIMILARITY_ALGORITHM_ALIASES = {
    "v1": SIMILARITY_ALGORITHM_V1,
    SIMILARITY_ALGORITHM_V1: SIMILARITY_ALGORITHM_V1,
    "role_component": SIMILARITY_ALGORITHM_V1,
    "v2": SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2: SIMILARITY_ALGORITHM_V2,
    "semantic": SIMILARITY_ALGORITHM_V2,
    "semantic_jaccard": SIMILARITY_ALGORITHM_V2,
}
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_LIMIT = 1000
DEFAULT_MAX_DF_FOR_SEED = 3000
SEMANTIC_MATCH_CONFIDENCE_MIN = 0.45
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

PRODUCT_CATEGORY_LABELS = {
    "영양보충",
    "장 건강",
    "뼈 건강",
    "면역",
    "기타",
    "관절/연골",
    "혈중지질",
    "눈 건강",
    "혈당",
    "남성 건강",
    "체지방",
    "구강 건강",
    "피부 건강",
    "기억력",
    "여성 건강",
    "피로개선",
    "운동/근력",
    "수면/긴장완화",
    "간 건강",
    "항산화",
    "혈행",
    "인지력",
    "혈압",
}

NON_SPECIFIC_CATEGORY_LABELS = {"", "기타"}

SEMANTIC_EXCIPIENT_PATTERNS = [
    r"^히드록시프로필메틸셀룰로오스(\(|$)",
    r"^hpmc$",
    r"^결정셀룰로오스(\(|$)",
    r"^결정셀룰로스(\(|$)",
    r"^미결정셀룰로오스(\(|$)",
    r"^스테아린산마그네슘(\(|$)",
    r"^스테아린산칼슘(\(|$)",
    r"^이산화규소(\(|$)",
    r"^카복시메틸셀룰로오스(\(|$)",
    r"^카르복시메틸셀룰로오스(\(|$)",
    r"^글리세린(\(|$)",
    r"^캡슐기제(\(|$)",
    r"^착색료(\(|$)",
    r"^향료(\(|$)",
    r"^감미료(\(|$)",
    r"^정제수$",
    r"^정제소금$",
]

SEMANTIC_EXCIPIENT_EXACT_NAMES = {
    "말토덱스트린",
    "덱스트린",
    "유당",
    "옥수수전분",
    "감자전분",
    "돼지젤라틴",
    "젤라틴",
}

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
    parser.add_argument(
        "--similarity-algorithm",
        default="v1",
        choices=sorted(SIMILARITY_ALGORITHM_ALIASES),
        help="유사도 알고리즘. v1은 기존 role component, v2는 semantic weighted Jaccard입니다.",
    )
    parser.add_argument(
        "--ingredient-category-profile",
        default="",
        help="semantic weighted Jaccard v2용 ingredient_category_profile.csv 경로",
    )
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


def first_existing_optional_path(candidates: list[Path]):
    for path in candidates:
        if path and path.exists():
            return path
    return None


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
    sqlite_candidates = []
    configured_sqlite = str(get_config_value(config, "sqlite_path", "") or "").strip()
    if configured_sqlite:
        sqlite_candidates.append(Path(configured_sqlite))
    sqlite_candidates.append(Path(r"D:\ec2_cache_snapshot\ingredient_match_cache_rebuilt_item_class_i0050_final.sqlite"))
    ingredient_profile_candidates = [
        output_dir / "ingredient_category_profile.csv",
        ROOT_DIR / "output" / "ingredient_category_profile.csv",
        output_dir / "semantic_jaccard_v2" / "ingredient_category_profile.csv",
    ]
    return {
        "output_dir": output_dir,
        "vector_csv_path": first_existing_path(vector_candidates),
        "product_profile_csv_path": first_existing_path(product_profile_candidates),
        "sqlite_path": first_existing_path(sqlite_candidates),
        "ingredient_category_profile_path": first_existing_optional_path(ingredient_profile_candidates),
    }


def normalize_token(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_similarity_algorithm(value: object) -> str:
    key = str(value or "").strip().lower()
    if not key:
        return SIMILARITY_ALGORITHM_VERSION
    if key in SIMILARITY_ALGORITHM_ALIASES:
        return SIMILARITY_ALGORITHM_ALIASES[key]
    raise ValueError(f"지원하지 않는 similarity_algorithm입니다: {value}")


def safe_product_key(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w가-힣.-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._")
    return text[:80] or "unknown_product"


def ingredient_matches_patterns(name: str, patterns: list[str]) -> bool:
    lowered = normalize_text(name)
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def is_semantic_excipient_name(name: str) -> bool:
    if normalize_token(name) in {normalize_token(item) for item in SEMANTIC_EXCIPIENT_EXACT_NAMES}:
        return True
    return ingredient_matches_patterns(name, SEMANTIC_EXCIPIENT_PATTERNS)


def is_excipient(name: str) -> bool:
    return ingredient_matches_patterns(name, EXCIPIENT_PATTERNS) or is_semantic_excipient_name(name)


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
            "llm_sub_function_categories": safe_json_loads(getattr(row, "llm_sub_function_categories_json", "[]"), []),
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


def coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_category_list(value: object) -> list[str]:
    parsed = safe_json_loads(value, [])
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def load_ingredient_category_profiles(path=None) -> dict[str, dict]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "functional_ingredient_name" not in df.columns:
        raise ValueError(f"ingredient category profile에 functional_ingredient_name 컬럼이 없습니다: {path}")
    profiles: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        name = str(row.get("functional_ingredient_name", "") or "").strip()
        if not name:
            continue
        sub_categories = parse_category_list(row.get("ingredient_sub_function_categories_json", "[]"))
        main_category = str(row.get("ingredient_main_category", "") or "").strip()
        ingredient_type = str(row.get("ingredient_type", "functional") or "functional").strip()
        profiles[name] = {
            "functional_ingredient_name": name,
            "ingredient_main_category": main_category,
            "ingredient_sub_function_categories": sub_categories,
            "ingredient_type": ingredient_type,
            "vector_include": coerce_bool(row.get("vector_include", True), True),
            "is_excipient": coerce_bool(row.get("is_excipient", False), False),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "reason": str(row.get("reason", "") or "").strip(),
            "source": str(row.get("source", "") or "").strip(),
        }
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


ROLE_FACTORS = {
    "primary": 1.2,
    "secondary": 0.55,
    "support": 0.2,
}

ROLE_IDF_CAPS = {
    "primary": 6.0,
    "secondary": 3.5,
    "support": 2.0,
}

OFF_CATEGORY_FACTORS = {
    "primary": 0.75,
    "secondary": 0.35,
    "support": 0.5,
}

FINAL_SIMILARITY_COMPONENT_WEIGHTS = {
    "primary": 0.65,
    "secondary": 0.20,
    "function": 0.10,
    "support": 0.05,
}

NO_PRIMARY_OVERLAP_SCORE_CAP = 0.35


def normalize_role(role: str) -> str:
    role = str(role or "").strip()
    return role if role in ROLE_FACTORS else "secondary"


def role_factor(role: str) -> float:
    return ROLE_FACTORS.get(normalize_role(role), 0.45)


def ingredient_idf(total_product_count: int, ingredient_df: int) -> float:
    return math.log((1.0 + total_product_count) / (1.0 + ingredient_df)) + 1.0


def capped_ingredient_idf(total_product_count: int, ingredient_df: int, role: str) -> float:
    normalized_role = normalize_role(role)
    return min(
        ingredient_idf(total_product_count, ingredient_df),
        ROLE_IDF_CAPS.get(normalized_role, ROLE_IDF_CAPS["secondary"]),
    )


def product_ingredient_category(profile: dict, ingredient: str) -> str:
    return str(dict(profile.get("category_by_ingredient", {}) or {}).get(ingredient, "") or "").strip()


def category_alignment_factor(profile: dict, ingredient: str, role: str) -> float:
    product_main_category = str(profile.get("product_main_category", "") or "").strip()
    ingredient_category = product_ingredient_category(profile, ingredient)
    if not product_main_category or not ingredient_category or product_main_category == ingredient_category:
        return 1.0
    return OFF_CATEGORY_FACTORS.get(normalize_role(role), OFF_CATEGORY_FACTORS["secondary"])


def effective_ingredient_weight(
    vector_weight: float,
    ingredient: str,
    profile: dict,
    ingredient_frequency: dict[str, int],
    total_product_count: int,
) -> float:
    if vector_weight <= 0:
        return 0.0
    role = normalize_role(dict(profile.get("role_by_ingredient", {}) or {}).get(ingredient, ""))
    idf = capped_ingredient_idf(total_product_count, ingredient_frequency.get(ingredient, 0), role)
    return float(vector_weight) * idf * role_factor(role) * category_alignment_factor(profile, ingredient, role)


def strongest_component_role(*roles: str) -> str:
    normalized_roles = [normalize_role(role) for role in roles if str(role or "").strip()]
    if "primary" in normalized_roles:
        return "primary"
    if "secondary" in normalized_roles:
        return "secondary"
    if "support" in normalized_roles:
        return "support"
    return "secondary"


def all_profile_ingredients(profile: dict) -> set[str]:
    return {
        str(item or "")
        for item in (
            list(profile.get("primary_ingredients", []) or [])
            + list(profile.get("secondary_ingredients", []) or [])
            + list(profile.get("support_ingredients", []) or [])
        )
        if str(item or "")
    }


def has_primary_ingredient_overlap(base_profile: dict, target_profile: dict) -> bool:
    base_primary = {str(item or "") for item in base_profile.get("primary_ingredients", []) if str(item or "")}
    target_primary = {str(item or "") for item in target_profile.get("primary_ingredients", []) if str(item or "")}
    if not base_primary or not target_primary:
        return True
    base_all = all_profile_ingredients(base_profile)
    target_all = all_profile_ingredients(target_profile)
    if base_primary & target_all or target_primary & base_all:
        return True
    base_primary_families = collect_joint_ingredient_families(base_primary)
    target_all_families = collect_joint_ingredient_families(target_all)
    target_primary_families = collect_joint_ingredient_families(target_primary)
    base_all_families = collect_joint_ingredient_families(base_all)
    return bool((base_primary_families & target_all_families) or (target_primary_families & base_all_families))


def weighted_jaccard_for_component(
    component: str,
    union_ingredients: set[str],
    base_vector: dict[str, float],
    target_vector: dict[str, float],
    ingredient_frequency: dict[str, int],
    total_product_count: int,
    base_profile: dict,
    target_profile: dict,
) -> float:
    base_roles = dict(base_profile.get("role_by_ingredient", {}) or {})
    target_roles = dict(target_profile.get("role_by_ingredient", {}) or {})
    numerator = 0.0
    denominator = 0.0
    for ingredient in union_ingredients:
        ingredient_component = strongest_component_role(
            base_roles.get(ingredient, "") if ingredient in base_vector else "",
            target_roles.get(ingredient, "") if ingredient in target_vector else "",
        )
        if ingredient_component != component:
            continue
        effective_base = effective_ingredient_weight(
            float(base_vector.get(ingredient, 0.0)),
            ingredient,
            base_profile,
            ingredient_frequency,
            total_product_count,
        )
        effective_target = effective_ingredient_weight(
            float(target_vector.get(ingredient, 0.0)),
            ingredient,
            target_profile,
            ingredient_frequency,
            total_product_count,
        )
        numerator += min(effective_base, effective_target)
        denominator += max(effective_base, effective_target)
    return 0.0 if denominator <= 0 else float(numerator / denominator)


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
    if not shared_ingredients:
        return 0.0, shared_ingredients
    primary_similarity = weighted_jaccard_for_component(
        "primary",
        union_ingredients,
        base_vector,
        target_vector,
        ingredient_frequency,
        total_product_count,
        base_profile,
        target_profile,
    )
    secondary_similarity = weighted_jaccard_for_component(
        "secondary",
        union_ingredients,
        base_vector,
        target_vector,
        ingredient_frequency,
        total_product_count,
        base_profile,
        target_profile,
    )
    support_similarity = weighted_jaccard_for_component(
        "support",
        union_ingredients,
        base_vector,
        target_vector,
        ingredient_frequency,
        total_product_count,
        base_profile,
        target_profile,
    )
    function_similarity = calculate_function_similarity(base_profile, target_profile)
    similarity = (
        FINAL_SIMILARITY_COMPONENT_WEIGHTS["primary"] * primary_similarity
        + FINAL_SIMILARITY_COMPONENT_WEIGHTS["secondary"] * secondary_similarity
        + FINAL_SIMILARITY_COMPONENT_WEIGHTS["function"] * function_similarity
        + FINAL_SIMILARITY_COMPONENT_WEIGHTS["support"] * support_similarity
    )
    if not has_primary_ingredient_overlap(base_profile, target_profile):
        similarity = min(similarity, NO_PRIMARY_OVERLAP_SCORE_CAP)
    return round(max(0.0, min(float(similarity), 1.0)), 6), shared_ingredients


def ingredient_score_items(profile: dict) -> list[dict]:
    items = profile.get("ingredient_scores", []) or []
    if items:
        return [dict(item) for item in items if str(item.get("ingredient", "") or "").strip()]
    fallback_items = []
    for role_name in ("primary", "secondary", "support"):
        for ingredient in profile.get(f"{role_name}_ingredients", []) or []:
            name = str(ingredient or "").strip()
            if name:
                fallback_items.append(
                    {
                        "ingredient": name,
                        "weight": 1.0,
                        "role": role_name,
                        "category_main": product_ingredient_category(profile, name),
                        "category_sub": "",
                    }
                )
    return fallback_items


def fallback_ingredient_category_profile(ingredient: str, score_item: dict) -> dict:
    category_main = str(score_item.get("category_main", "") or "").strip()
    if category_main not in PRODUCT_CATEGORY_LABELS:
        category_main = "기타"
    is_excipient_value = is_semantic_excipient_name(ingredient)
    ingredient_type = "excipient" if is_excipient_value else "functional"
    return {
        "functional_ingredient_name": ingredient,
        "ingredient_main_category": category_main,
        "ingredient_sub_function_categories": [],
        "ingredient_type": ingredient_type,
        "vector_include": not is_excipient_value,
        "is_excipient": is_excipient_value,
        "confidence": float(score_item.get("weight", 0.0) or 0.0),
        "reason": "ingredient_category_profile.csv가 없어 functional_category_map의 category_main만 fallback으로 사용",
        "source": "functional_category_map_fallback",
    }


def ingredient_category_profile_for_item(ingredient: str, score_item: dict, ingredient_category_profiles: dict[str, dict] | None) -> dict:
    profile = dict((ingredient_category_profiles or {}).get(ingredient, {}) or {})
    if not profile:
        return fallback_ingredient_category_profile(ingredient, score_item)
    profile.setdefault("functional_ingredient_name", ingredient)
    profile.setdefault("ingredient_sub_function_categories", [])
    profile.setdefault("ingredient_type", "functional")
    profile.setdefault("vector_include", True)
    profile.setdefault("is_excipient", False)
    if is_semantic_excipient_name(ingredient):
        profile["ingredient_type"] = "excipient"
        profile["vector_include"] = False
        profile["is_excipient"] = True
    return profile


def product_semantic_category_set(profile: dict) -> set[str]:
    categories = {str(profile.get("product_main_category", "") or "").strip()}
    categories.update(sub_function_categories(profile))
    return {category for category in categories if category}


def semantic_weight_for_product_ingredient(
    product_profile: dict,
    score_item: dict,
    ingredient_category_profiles: dict[str, dict] | None = None,
    min_match_confidence: float = SEMANTIC_MATCH_CONFIDENCE_MIN,
) -> tuple[float, str, dict]:
    ingredient = str(score_item.get("ingredient", "") or "").strip()
    if not ingredient:
        return 0.0, "empty_ingredient", {}
    match_confidence = float(score_item.get("weight", 0.0) or 0.0)
    if match_confidence > 0 and match_confidence < min_match_confidence:
        return 0.0, "low_match_confidence", {}

    ingredient_profile = ingredient_category_profile_for_item(ingredient, score_item, ingredient_category_profiles)
    ingredient_type = str(ingredient_profile.get("ingredient_type", "functional") or "functional").strip()
    vector_include = coerce_bool(ingredient_profile.get("vector_include", True), True)
    is_excipient_value = coerce_bool(ingredient_profile.get("is_excipient", False), False)
    if is_excipient_value or ingredient_type in {"excipient", "additive", "formulation_aid"} or not vector_include:
        return 0.0, "excluded_non_functional", ingredient_profile

    product_main = str(product_profile.get("product_main_category", "") or "").strip()
    product_subs = set(sub_function_categories(product_profile))
    ingredient_main = str(ingredient_profile.get("ingredient_main_category", "") or "").strip()
    ingredient_subs = {
        str(item).strip()
        for item in ingredient_profile.get("ingredient_sub_function_categories", []) or []
        if str(item).strip()
    }
    ingredient_categories = ({ingredient_main} if ingredient_main else set()) | ingredient_subs
    product_category_is_specific = product_main not in NON_SPECIFIC_CATEGORY_LABELS

    if ingredient_type in {"nutrient", "generic_nutrient"}:
        if product_category_is_specific and product_main in ingredient_categories:
            return 0.4, "generic_nutrient_category_aligned", ingredient_profile
        if product_subs & ingredient_categories:
            return 0.3, "generic_nutrient_sub_category_overlap", ingredient_profile
        return 0.2, "generic_nutrient_weak_signal", ingredient_profile

    if product_category_is_specific and product_main == ingredient_main:
        return 1.0, "main_category_match", ingredient_profile
    if product_category_is_specific and product_main in ingredient_subs:
        return 0.7, "product_main_in_ingredient_sub_functions", ingredient_profile
    if product_subs & ingredient_categories:
        return 0.7, "product_sub_function_overlap", ingredient_profile
    if ingredient_type == "functional":
        return 0.3, "functional_weak_signal", ingredient_profile
    return 0.0, "unsupported_ingredient_type", ingredient_profile


def semantic_ingredient_key(ingredient: str) -> str:
    family = infer_joint_ingredient_family(ingredient)
    if family:
        return f"family::{family}"
    return ingredient


def build_semantic_weight_vector(
    product_profile: dict,
    ingredient_category_profiles: dict[str, dict] | None = None,
    include_details: bool = False,
):
    vector: dict[str, float] = {}
    details: dict[str, dict] = {}
    for score_item in ingredient_score_items(product_profile):
        ingredient = str(score_item.get("ingredient", "") or "").strip()
        weight, reason, ingredient_profile = semantic_weight_for_product_ingredient(
            product_profile,
            score_item,
            ingredient_category_profiles,
        )
        key = semantic_ingredient_key(ingredient)
        if include_details:
            details.setdefault(
                key,
                {
                    "semantic_key": key,
                    "ingredients": [],
                    "weight": 0.0,
                    "reason": reason,
                    "ingredient_main_category": str(ingredient_profile.get("ingredient_main_category", "") or ""),
                    "ingredient_sub_function_categories": list(ingredient_profile.get("ingredient_sub_function_categories", []) or []),
                    "ingredient_type": str(ingredient_profile.get("ingredient_type", "") or ""),
                    "vector_include": coerce_bool(ingredient_profile.get("vector_include", True), True),
                    "is_excipient": coerce_bool(ingredient_profile.get("is_excipient", False), False),
                },
            )
            details[key]["ingredients"].append(ingredient)
            if weight >= float(details[key].get("weight", 0.0) or 0.0):
                details[key]["weight"] = weight
                details[key]["reason"] = reason
        if weight <= 0:
            continue
        vector[key] = max(vector.get(key, 0.0), float(weight))
    if include_details:
        return vector, details
    return vector


def calculate_semantic_weighted_jaccard_v2(
    base_profile: dict,
    target_profile: dict,
    ingredient_category_profiles: dict[str, dict] | None = None,
) -> tuple[float, list[str], dict]:
    base_vector, base_details = build_semantic_weight_vector(base_profile, ingredient_category_profiles, include_details=True)
    target_vector, target_details = build_semantic_weight_vector(target_profile, ingredient_category_profiles, include_details=True)
    union_keys = set(base_vector) | set(target_vector)
    shared_keys = sorted(set(base_vector) & set(target_vector))
    if not shared_keys:
        return 0.0, [], {
            "algorithm": SIMILARITY_ALGORITHM_V2,
            "base_semantic_ingredient_count": len(base_vector),
            "target_semantic_ingredient_count": len(target_vector),
            "shared_semantic_keys": [],
            "excluded_base_ingredients": [
                name
                for detail in base_details.values()
                if float(detail.get("weight", 0.0) or 0.0) <= 0
                for name in detail.get("ingredients", [])
            ],
            "excluded_target_ingredients": [
                name
                for detail in target_details.values()
                if float(detail.get("weight", 0.0) or 0.0) <= 0
                for name in detail.get("ingredients", [])
            ],
        }
    numerator = sum(min(base_vector.get(key, 0.0), target_vector.get(key, 0.0)) for key in union_keys)
    denominator = sum(max(base_vector.get(key, 0.0), target_vector.get(key, 0.0)) for key in union_keys)
    score = 0.0 if denominator <= 0 else float(numerator / denominator)

    shared_labels = []
    shared_details = []
    for key in shared_keys:
        base_names = list(base_details.get(key, {}).get("ingredients", []) or [])
        target_names = list(target_details.get(key, {}).get("ingredients", []) or [])
        if key.startswith("family::") and base_names and target_names and base_names[0] != target_names[0]:
            label = f"{base_names[0]} ~ {target_names[0]}"
        elif base_names and target_names and base_names[0] == target_names[0]:
            label = base_names[0]
        else:
            label = key
        shared_labels.append(label)
        shared_details.append(
            {
                "semantic_key": key,
                "label": label,
                "base_weight": base_vector.get(key, 0.0),
                "target_weight": target_vector.get(key, 0.0),
                "base_ingredients": base_names,
                "target_ingredients": target_names,
            }
        )

    return round(max(0.0, min(score, 1.0)), 6), shared_labels, {
        "algorithm": SIMILARITY_ALGORITHM_V2,
        "numerator": round(float(numerator), 6),
        "denominator": round(float(denominator), 6),
        "base_semantic_ingredient_count": len(base_vector),
        "target_semantic_ingredient_count": len(target_vector),
        "shared_semantic_keys": shared_details,
    }


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
    exact_score = 0.0 if not union else float(len(base_primary & target_primary) / len(union))
    base_primary_families = collect_joint_ingredient_families(base_primary)
    target_primary_families = collect_joint_ingredient_families(target_primary)
    family_union = base_primary_families | target_primary_families
    family_score = 0.0 if not family_union else float(len(base_primary_families & target_primary_families) / len(family_union))
    return round(max(exact_score, family_score), 6)


def sub_function_categories(profile: dict) -> list[str]:
    """Batch API sub-function labels used for category-level recommendation signals."""
    return [
        str(item).strip()
        for item in profile.get("llm_sub_function_categories", [])
        if str(item).strip()
    ]


def focus_ingredients(profile: dict) -> list[str]:
    primary = [str(value or "") for value in profile.get("primary_ingredients", []) if str(value or "")]
    if primary:
        return primary
    return [str(value or "") for value in profile.get("secondary_ingredients", []) if str(value or "")][:3]


def focus_joint_families(profile: dict) -> set[str]:
    return collect_joint_ingredient_families(focus_ingredients(profile))


def shared_joint_families(base_profile: dict, target_profile: dict) -> set[str]:
    return focus_joint_families(base_profile) & focus_joint_families(target_profile)


def compare_product_profiles(base_profile: dict, target_profile: dict, base_vector: dict[str, float], target_vector: dict[str, float]) -> dict:
    shared_ingredients_raw = sorted(set(base_vector) & set(target_vector))
    shared_ingredients_for_display = [name for name in shared_ingredients_raw if not is_excipient(name)]
    base_only_ingredients = sorted(set(base_vector) - set(target_vector))
    target_only_ingredients = sorted(set(target_vector) - set(base_vector))

    def profile_categories(profile: dict) -> set[str]:
        categories = {str(profile.get("product_main_category", "기타"))}
        categories.update(sub_function_categories(profile))
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
    shared_primary_families = collect_joint_ingredient_families(base_profile.get("primary_ingredients", [])) & collect_joint_ingredient_families(
        target_profile.get("primary_ingredients", [])
    )
    shared_focus_family_values = shared_joint_families(base_profile, target_profile)
    has_core_overlap = bool(shared_primary or shared_primary_families or shared_focus_family_values)
    if similarity_score >= 0.75 and same_main and has_core_overlap:
        return "높음"
    if similarity_score >= 0.45 and has_core_overlap:
        return "보통"
    if similarity_score >= 0.35 and same_main and has_core_overlap:
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
    joint_family_overlap = shared_joint_families(base_profile, target_profile)

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

    joint_ingredients = [
        name
        for name in ranked
        if ingredient_matches_patterns(name, JOINT_KEYWORDS)
        and infer_joint_ingredient_family(name) in joint_family_overlap
    ]
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
    base_categories.update(sub_function_categories(base_profile))
    base_focus_families = focus_joint_families(base_profile)
    base_joint_related = product_is_joint_related(base_profile)

    def ensure_candidate(candidate_id: str) -> dict:
        if candidate_id not in candidate_stats:
            candidate_profile = profiles[candidate_id]
            candidate_categories = {str(candidate_profile.get("product_main_category", "기타"))}
            candidate_categories.update(sub_function_categories(candidate_profile))
            candidate_focus_families = focus_joint_families(candidate_profile)
            shared_focus_family_count = len(base_focus_families & candidate_focus_families)
            candidate_joint_related = product_is_joint_related(candidate_profile)
            candidate_stats[candidate_id] = {
                "target_product_id": candidate_id,
                "same_main_category": int(candidate_profile.get("product_main_category") == base_profile.get("product_main_category")),
                "shared_primary_count": 0,
                "shared_secondary_count": 0,
                "shared_support_count": 0,
                "shared_focus_family_count": shared_focus_family_count,
                "joint_family_conflict": int(base_joint_related and candidate_joint_related and shared_focus_family_count == 0),
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
        if row["joint_family_conflict"]:
            row["preliminary_score"] -= 1.5
    rows = sorted(
        rows,
        key=lambda item: (
            -item["same_main_category"],
            -item["shared_primary_count"],
            -item["shared_focus_family_count"],
            item["joint_family_conflict"],
            -item["shared_category_count"],
            -item["preliminary_score"],
            item["target_product_id"],
        ),
    )
    return rows[:candidate_limit]


def build_candidate_pool_v2(
    base_product_id: str,
    base_profile: dict,
    product_vectors: dict[str, dict[str, float]],
    profiles: dict[str, dict],
    ingredient_postings: dict[str, list[str]],
    ingredient_frequency: dict[str, int],
    candidate_limit: int,
    max_df_for_seed: int,
    ingredient_category_profiles: dict[str, dict] | None = None,
) -> list[dict]:
    candidate_stats: dict[str, dict] = {}
    base_categories = product_semantic_category_set(base_profile)
    base_sub_categories = set(sub_function_categories(base_profile))
    base_focus_families = focus_joint_families(base_profile)
    base_semantic_vector, base_semantic_details = build_semantic_weight_vector(
        base_profile,
        ingredient_category_profiles,
        include_details=True,
    )

    def ensure_candidate(candidate_id: str) -> dict:
        if candidate_id not in candidate_stats:
            candidate_profile = profiles[candidate_id]
            candidate_categories = product_semantic_category_set(candidate_profile)
            candidate_focus_families = focus_joint_families(candidate_profile)
            candidate_stats[candidate_id] = {
                "target_product_id": candidate_id,
                "same_main_category": int(candidate_profile.get("product_main_category") == base_profile.get("product_main_category")),
                "shared_primary_count": 0,
                "shared_secondary_count": 0,
                "shared_support_count": 0,
                "shared_semantic_ingredient_count": 0,
                "shared_focus_family_count": len(base_focus_families & candidate_focus_families),
                "joint_family_conflict": 0,
                "shared_category_count": len(base_categories & candidate_categories),
                "preliminary_score": 0.0,
            }
        return candidate_stats[candidate_id]

    for candidate_id, profile in profiles.items():
        if candidate_id == base_product_id:
            continue
        candidate_categories = product_semantic_category_set(profile)
        same_main = profile.get("product_main_category") == base_profile.get("product_main_category")
        sub_overlap = bool(base_sub_categories & candidate_categories)
        family_overlap = bool(base_focus_families & focus_joint_families(profile))
        if same_main or sub_overlap or family_overlap:
            ensure_candidate(candidate_id)

    for semantic_key, semantic_weight in base_semantic_vector.items():
        exact_ingredient_names = list(base_semantic_details.get(semantic_key, {}).get("ingredients", []) or [])
        for ingredient in exact_ingredient_names:
            ingredient_df = ingredient_frequency.get(ingredient, 0)
            if ingredient_df <= 0:
                continue
            if ingredient_df > max_df_for_seed and semantic_weight < 0.7:
                continue
            for candidate_id in ingredient_postings.get(ingredient, []):
                if candidate_id == base_product_id or candidate_id not in profiles:
                    continue
                stat = ensure_candidate(candidate_id)
                stat["shared_semantic_ingredient_count"] += 1
                stat["preliminary_score"] += float(semantic_weight) * ingredient_idf(len(product_vectors), max(1, ingredient_df))

    rows = list(candidate_stats.values())
    rows = sorted(
        rows,
        key=lambda item: (
            -item["same_main_category"],
            -item["shared_semantic_ingredient_count"],
            -item["shared_focus_family_count"],
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
            algorithm_version TEXT NOT NULL DEFAULT 'legacy',
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
    existing_columns = {
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{TABLE_NAME}")').fetchall()
    }
    if "algorithm_version" not in existing_columns:
        conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN algorithm_version TEXT NOT NULL DEFAULT 'legacy'")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_base_product_id ON {TABLE_NAME} (base_product_id)")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_base_product_algorithm ON {TABLE_NAME} (base_product_id, algorithm_version)"
    )
    conn.commit()


def load_cached_rows(
    conn: sqlite3.Connection,
    base_product_id: str,
    top_k: int,
    algorithm_version: str = SIMILARITY_ALGORITHM_VERSION,
) -> list[dict]:
    query = f"""
    SELECT base_product_id, target_product_id, base_product_name, target_product_name,
           similarity_score, function_similarity_score, core_match_score,
           shared_ingredients_json, base_only_ingredients_json, target_only_ingredients_json,
           shared_categories_json, different_categories_json, substitutability,
           reason, caution, explanation_json
    FROM {TABLE_NAME}
    WHERE base_product_id = ?
      AND algorithm_version = ?
    ORDER BY similarity_score DESC, target_product_id ASC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(base_product_id, algorithm_version, top_k))
    return df.to_dict(orient="records")


def refresh_cache_rows(
    conn: sqlite3.Connection,
    base_product_id: str,
    rows: list[dict],
    algorithm_version: str = SIMILARITY_ALGORITHM_VERSION,
) -> None:
    if algorithm_version != SIMILARITY_ALGORITHM_VERSION:
        raise ValueError("현재 캐시 테이블 기본키는 v1 전용입니다. v2 비교 결과는 캐시에 저장하지 않습니다.")
    now_text = datetime.now().isoformat()
    conn.execute(f"DELETE FROM {TABLE_NAME} WHERE base_product_id = ?", (base_product_id,))
    conn.executemany(
        f"""
        INSERT INTO {TABLE_NAME} (
            base_product_id, target_product_id, algorithm_version, base_product_name, target_product_name,
            similarity_score, function_similarity_score, core_match_score,
            shared_ingredients_json, base_only_ingredients_json, target_only_ingredients_json,
            shared_categories_json, different_categories_json, substitutability,
            reason, caution, explanation_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["base_product_id"],
                row["target_product_id"],
                algorithm_version,
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
    similarity_algorithm: str = SIMILARITY_ALGORITHM_VERSION,
    ingredient_category_profiles: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    base_profile = profiles[base_product_id]
    base_vector = product_vectors[base_product_id]
    total_product_count = len(product_vectors)
    similarity_algorithm = normalize_similarity_algorithm(similarity_algorithm)

    t0 = time.perf_counter()
    if similarity_algorithm == SIMILARITY_ALGORITHM_V2:
        candidate_pool = build_candidate_pool_v2(
            base_product_id,
            base_profile,
            product_vectors,
            profiles,
            ingredient_postings,
            ingredient_frequency,
            candidate_limit,
            max_df_for_seed,
            ingredient_category_profiles,
        )
    else:
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
    print(f"[INFO] similarity_algorithm={similarity_algorithm}")
    print(f"[INFO] candidate_count_before_topk={len(candidate_pool)}")

    rows = []
    failed_rows = []
    for candidate in candidate_pool:
        target_product_id = candidate["target_product_id"]
        try:
            target_profile = profiles[target_product_id]
            target_vector = product_vectors[target_product_id]
            semantic_explanation = {}
            shared_semantic_ingredients = []
            if similarity_algorithm == SIMILARITY_ALGORITHM_V2:
                similarity_score, shared_semantic_ingredients, semantic_explanation = calculate_semantic_weighted_jaccard_v2(
                    base_profile,
                    target_profile,
                    ingredient_category_profiles,
                )
            else:
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
            if similarity_algorithm == SIMILARITY_ALGORITHM_V2 and shared_semantic_ingredients:
                comparison["shared_ingredients"] = shared_semantic_ingredients
                comparison["shared_ingredients_raw"] = shared_semantic_ingredients
            function_similarity_score = calculate_function_similarity(base_profile, target_profile)
            core_match_score = calculate_core_match_score(base_profile, target_profile)
            substitutability = classify_substitutability(similarity_score, base_profile, target_profile)
            reason = generate_recommendation_reason(base_profile, target_profile, comparison, ingredient_frequency)
            explanation_json = build_explanation_json(reason, comparison, substitutability)
            explanation_json["similarity_algorithm"] = similarity_algorithm
            if semantic_explanation:
                explanation_json["semantic_weighted_jaccard_v2"] = semantic_explanation
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
                    "similarity_algorithm": similarity_algorithm,
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
    return rows, failed_rows, {
        "candidate_count": len(candidate_pool),
        "topk_count": len(rows),
        "candidate_stage_seconds": candidate_stage_seconds,
        "similarity_algorithm": similarity_algorithm,
    }


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    runtime = resolve_runtime_paths()
    similarity_algorithm = normalize_similarity_algorithm(args.similarity_algorithm)
    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = ROOT_DIR / output_prefix
    ingredient_profile_path = Path(args.ingredient_category_profile) if str(args.ingredient_category_profile or "").strip() else runtime.get("ingredient_category_profile_path")

    print(f"[INFO] vector_csv_path={runtime['vector_csv_path']}")
    print(f"[INFO] product_profile_csv_path={runtime['product_profile_csv_path']}")
    print(f"[INFO] sqlite_path={runtime['sqlite_path']}")
    print(f"[INFO] similarity_algorithm={similarity_algorithm}")
    if similarity_algorithm == SIMILARITY_ALGORITHM_V2:
        print(f"[INFO] ingredient_category_profile_path={ingredient_profile_path or '(fallback only)'}")

    vector_df = load_vector_inputs(runtime["vector_csv_path"])
    profiles = load_product_function_profiles(runtime["product_profile_csv_path"])
    ingredient_category_profiles = load_ingredient_category_profiles(ingredient_profile_path)
    product_vectors, product_names, report_to_product_ids, ingredient_postings, ingredient_frequency = build_vector_indexes(vector_df)

    base_product_id = resolve_base_product_id(args, profiles, product_names, report_to_product_ids)
    base_profile = profiles[base_product_id]
    safe_key = safe_product_key(base_profile["report_no"] or base_profile["product_name"] or base_product_id)

    with sqlite3.connect(runtime["sqlite_path"]) as conn:
        ensure_cache_table(conn)
        cached_rows = []
        cache_enabled = similarity_algorithm == SIMILARITY_ALGORITHM_VERSION
        if cache_enabled and not args.force_refresh:
            cached_rows = load_cached_rows(conn, base_product_id, args.top_k, similarity_algorithm)
        if len(cached_rows) >= args.top_k:
            elapsed_seconds = round(time.perf_counter() - start_time, 3)
            summary = {
                "generated_at": datetime.now().isoformat(),
                "base_product_id": base_product_id,
                "base_product_name": base_profile["product_name"],
                "report_no": base_profile["report_no"],
                "similarity_algorithm": similarity_algorithm,
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
            similarity_algorithm,
            ingredient_category_profiles,
        )
        if cache_enabled:
            refresh_cache_rows(conn, base_product_id, rows, similarity_algorithm)
            cache_row_count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE base_product_id = ?", (base_product_id,)).fetchone()[0]
        else:
            cache_row_count = 0

    elapsed_seconds = round(time.perf_counter() - start_time, 3)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "base_product_id": base_product_id,
        "base_product_name": base_profile["product_name"],
        "report_no": base_profile["report_no"],
        "product_main_category": base_profile["product_main_category"],
        "similarity_algorithm": similarity_algorithm,
        "ingredient_category_profile_path": str(ingredient_profile_path or ""),
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
