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
SIMILARITY_ALGORITHM_V2_LEGACY = "semantic_weighted_jaccard_v2"
SIMILARITY_ALGORITHM_V2_1 = "semantic_weighted_jaccard_v2_1"
SIMILARITY_ALGORITHM_V2_2 = "semantic_weighted_jaccard_v2_2"
SIMILARITY_ALGORITHM_V2_3 = "semantic_weighted_jaccard_v2_3"
SIMILARITY_ALGORITHM_V2_4 = "semantic_weighted_jaccard_v2_4"
SIMILARITY_ALGORITHM_V2_5 = "semantic_weighted_jaccard_v2_5"
SIMILARITY_ALGORITHM_V2_6 = "semantic_weighted_jaccard_v2_6"
SIMILARITY_ALGORITHM_V2_7 = "semantic_weighted_jaccard_v2_7"
SIMILARITY_ALGORITHM_V2_8 = "semantic_weighted_jaccard_v2_8"
SIMILARITY_ALGORITHM_V2 = "semantic_weighted_jaccard_v2_9"
SIMILARITY_ALGORITHM_VERSION = SIMILARITY_ALGORITHM_V2
SIMILARITY_ALGORITHM_ALIASES = {
    "v1": SIMILARITY_ALGORITHM_V1,
    SIMILARITY_ALGORITHM_V1: SIMILARITY_ALGORITHM_V1,
    "role_component": SIMILARITY_ALGORITHM_V1,
    "v2": SIMILARITY_ALGORITHM_V2,
    "v2.1": SIMILARITY_ALGORITHM_V2,
    "v2_1": SIMILARITY_ALGORITHM_V2,
    "v2.2": SIMILARITY_ALGORITHM_V2,
    "v2_2": SIMILARITY_ALGORITHM_V2,
    "v2.3": SIMILARITY_ALGORITHM_V2,
    "v2_3": SIMILARITY_ALGORITHM_V2,
    "v2.4": SIMILARITY_ALGORITHM_V2,
    "v2_4": SIMILARITY_ALGORITHM_V2,
    "v2.5": SIMILARITY_ALGORITHM_V2,
    "v2_5": SIMILARITY_ALGORITHM_V2,
    "v2.6": SIMILARITY_ALGORITHM_V2,
    "v2_6": SIMILARITY_ALGORITHM_V2,
    "v2.7": SIMILARITY_ALGORITHM_V2,
    "v2_7": SIMILARITY_ALGORITHM_V2,
    "v2.8": SIMILARITY_ALGORITHM_V2,
    "v2_8": SIMILARITY_ALGORITHM_V2,
    "v2.9": SIMILARITY_ALGORITHM_V2,
    "v2_9": SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_LEGACY: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_1: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_2: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_3: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_4: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_5: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_6: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_7: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2_8: SIMILARITY_ALGORITHM_V2,
    SIMILARITY_ALGORITHM_V2: SIMILARITY_ALGORITHM_V2,
    "semantic": SIMILARITY_ALGORITHM_V2,
    "semantic_jaccard": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_1": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_2": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_3": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_4": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_5": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_6": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_7": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_8": SIMILARITY_ALGORITHM_V2,
    "semantic_v2_9": SIMILARITY_ALGORITHM_V2,
}
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_LIMIT = 1000
DEFAULT_MAX_DF_FOR_SEED = 3000
SEMANTIC_MATCH_CONFIDENCE_MIN = 0.45
NUTRITION_MAIN_CATEGORY = "\uc601\uc591\ubcf4\ucda9"
BONE_HEALTH_MAIN_CATEGORY = "\ubf08 \uac74\uac15"
ORAL_HEALTH_MAIN_CATEGORY = "\uad6c\uac15 \uac74\uac15"
LIPID_MAIN_CATEGORY = "\ud608\uc911\uc9c0\uc9c8"
LECITHIN_SEMANTIC_KEY = "\ub808\uc2dc\ud2f4"
GENERIC_NUTRIENT_VECTOR_SHARE_CAP = 0.20
GENERIC_NUTRIENT_ONLY_CROSS_MAIN_SCORE_CAP = 0.45
GENERIC_NUTRIENT_ONLY_NUTRITION_MAIN_SCORE_CAP = 0.64
GENERIC_NUTRIENT_ONLY_SAME_MAIN_SCORE_CAP = 0.55
GENERIC_NUTRIENT_DOMINANT_SHARED_RATIO = 0.75
GENERIC_NUTRIENT_DOMINANT_MIN_GENERIC_KEYS = 3
GENERIC_NUTRIENT_DOMINANT_MAX_NON_GENERIC_KEYS = 2
GENERIC_NUTRIENT_DOMINANT_MAX_WEAK_ADJUNCT_KEYS = 3
WEAK_SIGNAL_ONLY_CROSS_MAIN_SCORE_CAP = 0.45
WEAK_SIGNAL_ONLY_SCORE_CAP = 0.64
SINGLE_CORE_LOW_SHARED_SCORE_CAP = 0.64
SINGLE_CORE_LOW_SHARED_MAX_SHARED_KEYS = 2
SINGLE_CORE_LOW_SHARED_MIN_BASE_ONLY_KEYS = 3
NO_CORE_LOW_SHARED_SCORE_CAP = 0.64
NO_CORE_LOW_SHARED_MAX_SHARED_KEYS = 2
NO_CORE_LOW_SHARED_MAX_WEAK_SHARED_KEYS = 3
NO_CORE_LOW_SHARED_MIN_BASE_ONLY_KEYS = 3
NO_CORE_WEAK_SHARED_WITH_EXTRA_SCORE_CAP = 0.64
NO_CORE_WEAK_SHARED_WITH_EXTRA_MAX_SHARED_KEYS = 2
ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_CAP = 0.64
ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_MAX = 0.75
ORAL_SINGLE_CORE_BROAD_TARGET_MIN_TARGET_ONLY_KEYS = 2
LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_CAP = 0.64
LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_MAX = 0.90
LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_TARGET_ONLY_KEYS = 1
CROSS_MAIN_SHARED_SUBCATEGORY_ONLY_SCORE_CAP = 0.64
NUTRITION_SUBTYPE_MISMATCH_SCORE_CAP = 0.52
NUTRITION_GENERIC_LOW_CORE_COVERAGE_SCORE_CAP = 0.64
NUTRITION_GENERIC_LOW_CORE_COVERAGE_MAX_BASE_CORE_COVERAGE = 0.35
NUTRITION_GENERIC_LOW_CORE_COVERAGE_MIN_GENERIC_RATIO = 0.60
RECOMMENDATION_DISPLAY_SCORE_MIN = 0.65
RECOMMENDATION_REVIEW_SCORE_MIN = 0.45
RECOMMENDATION_REVIEW_FUNCTION_MIN = 0.40
RECOMMENDATION_LOW_SCORE_CORE_MIN = 0.50
RECOMMENDATION_LOW_SCORE_CORE_MATCH_MIN = 0.60
RECOMMENDATION_RISKY_ADJUSTMENT_MARKERS = (
    "generic_nutrient",
    "nutrition_subtype_mismatch",
    "weak_signal_only",
)
REVIEW_SIGNAL_MARKERS = (
    "claim_text_category_fallback",
    "\uac80\ud1a0\ud544\uc694",
)
GENERIC_NUTRIENT_ONLY_MAIN_CATEGORY_ALLOWLIST = {
    "\uae30\ud0c0",
    "\ud608\uc555",
}
FAMILY_SIGNAL_MAIN_CATEGORY_WEIGHT = 0.35
FAMILY_SIGNAL_SUB_CATEGORY_WEIGHT = 0.30
FAMILY_SIGNAL_WEAK_WEIGHT = 0.20
REVIEW_FALLBACK_SIGNAL_WEIGHT = 0.30
PARTIAL_CORE_COVERAGE_FLOOR = 0.85
NO_CORE_COVERAGE_MULTIPLIER = 0.68
SEMANTIC_CORE_WEIGHT_MIN = 0.7
SPARSE_EXACT_SCORE_CAP = 0.97
SPARSE_EXACT_EFFECTIVE_WEIGHT_MAX = 1.5
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

GENERIC_NUTRIENT_NAME_PATTERNS = [
    r"^비타민\s*[a-z]?\d*",
    r"비타민\s*[a-z]?\d*",
    r"나이아신",
    r"니아신",
    r"비오틴",
    r"판토텐산",
    r"엽산",
    r"아연",
    r"셀렌",
    r"셀레늄",
    r"마그네슘",
    r"칼슘",
    r"철",
    r"망간",
    r"구리",
    r"크롬",
    r"몰리브덴",
    r"요오드",
    r"칼륨",
    r"베타카로틴",
]

BONE_CORE_NUTRIENT_NAME_PATTERNS = [
    r"칼슘",
    r"마그네슘",
    r"비타민\s*d",
    r"비타민\s*k",
    r"\bk2\b",
]

NUTRITION_ADJUNCT_NAME_PATTERNS = [
    r"농축분말",
    r"과즙분말",
    r"향분말",
    r"맛분말",
    r"자일리톨",
]

NUTRITION_SUBTYPE_NAME_RULES = [
    ("folate", [r"엽산", r"folate", r"테트라히드로엽산"]),
    ("iron", [r"철분", r"\bfe\b", r"\biron\b", r"제일철"]),
    ("zinc", [r"아연", r"\bzinc\b"]),
    ("selenium", [r"셀렌", r"셀레늄", r"selen"]),
    ("calcium_magnesium_d", [r"칼슘", r"마그네슘", r"비타민\s*d", r"\bcalcium\b", r"\bmagnesium\b"]),
    ("vitamin_c", [r"비타민\s*c", r"\bvitamin\s*c\b", r"\bvita\s*c\b"]),
    ("vitamin_d", [r"비타민\s*d", r"\bvitamin\s*d\b"]),
    ("b_complex", [r"비타민\s*b", r"비타민b", r"\bb\s*complex\b", r"b컴플렉스"]),
    ("biotin", [r"비오틴", r"biotin"]),
    ("multivitamin", [r"멀티비타민", r"종합비타민", r"multi\s*vitamin", r"multivitamin"]),
]

NUTRITION_SUBTYPE_INGREDIENT_RULES = [
    ("folate", [r"엽산", r"테트라히드로엽산"]),
    ("iron", [r"철", r"제일철"]),
    ("zinc", [r"아연"]),
    ("selenium", [r"셀렌", r"셀레늄"]),
    ("calcium_magnesium_d", [r"칼슘", r"마그네슘", r"비타민\s*d", r"비타민\s*k"]),
    ("vitamin_c", [r"비타민\s*c"]),
    ("vitamin_d", [r"비타민\s*d"]),
    ("b_complex", [r"비타민\s*b", r"나이아신", r"판토텐산", r"비오틴"]),
    ("biotin", [r"비오틴"]),
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


def is_generic_nutrient_name(name: str) -> bool:
    return ingredient_matches_patterns(name, GENERIC_NUTRIENT_NAME_PATTERNS)


def is_bone_core_nutrient_name(name: str) -> bool:
    return ingredient_matches_patterns(name, BONE_CORE_NUTRIENT_NAME_PATTERNS)


def is_nutrition_adjunct_name(name: str) -> bool:
    return ingredient_matches_patterns(name, NUTRITION_ADJUNCT_NAME_PATTERNS)


def nutrition_subtype_from_profile(profile: dict, details: dict[str, dict] | None = None) -> str:
    if str(profile.get("product_main_category", "") or "").strip() != NUTRITION_MAIN_CATEGORY:
        return ""

    product_name = str(profile.get("product_name", "") or "")
    for subtype, patterns in NUTRITION_SUBTYPE_NAME_RULES:
        if ingredient_matches_patterns(product_name, patterns):
            return subtype

    ingredient_names: list[str] = []
    for detail in (details or {}).values():
        if str(detail.get("semantic_key", "") or "").startswith("__"):
            continue
        ingredient_names.extend(
            str(name or "").strip()
            for name in detail.get("ingredients", []) or []
            if str(name or "").strip()
        )
    if not ingredient_names:
        for item in ingredient_score_items(profile):
            name = str(item.get("ingredient", "") or "").strip()
            if name:
                ingredient_names.append(name)

    subtype_counts: dict[str, int] = defaultdict(int)
    for ingredient in ingredient_names:
        for subtype, patterns in NUTRITION_SUBTYPE_INGREDIENT_RULES:
            if ingredient_matches_patterns(ingredient, patterns):
                subtype_counts[subtype] += 1

    if len(ingredient_names) >= 6 and sum(subtype_counts.values()) >= 4:
        return "multivitamin"
    if not subtype_counts:
        return ""

    ranked = sorted(subtype_counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return "multivitamin" if sum(subtype_counts.values()) >= 4 else ranked[0][0]


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
            "legacy_category_main": str(row.get("legacy_category_main", "") or "").strip(),
            "legacy_category_sub": str(row.get("legacy_category_sub", "") or "").strip(),
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


def ingredient_profile_has_review_marker(ingredient_profile: dict) -> bool:
    values = [
        str(ingredient_profile.get("reason", "") or ""),
        str(ingredient_profile.get("source", "") or ""),
        str(ingredient_profile.get("legacy_category_main", "") or ""),
        str(ingredient_profile.get("legacy_category_sub", "") or ""),
    ]
    values.extend(str(item or "") for item in ingredient_profile.get("ingredient_sub_function_categories", []) or [])
    text = " ".join(values)
    return any(marker in text for marker in REVIEW_SIGNAL_MARKERS)


def semantic_weight_for_product_ingredient(
    product_profile: dict,
    score_item: dict,
    ingredient_category_profiles: dict[str, dict] | None = None,
    min_match_confidence: float = SEMANTIC_MATCH_CONFIDENCE_MIN,
) -> tuple[float, str, dict]:
    ingredient = str(score_item.get("ingredient", "") or "").strip()
    if not ingredient:
        return 0.0, "empty_ingredient", {}
    match_confidence = float(
        score_item.get(
            "match_confidence",
            score_item.get("confidence", score_item.get("weight", 0.0)),
        )
        or 0.0
    )
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
    review_fallback_signal = ingredient_profile_has_review_marker(ingredient_profile)

    if product_main == NUTRITION_MAIN_CATEGORY and is_nutrition_adjunct_name(ingredient):
        return 0.2, "nutrition_adjunct_weak_signal", ingredient_profile

    if ingredient_type in {"nutrient", "generic_nutrient"}:
        if product_category_is_specific and product_main in ingredient_categories:
            return 0.4, "generic_nutrient_category_aligned", ingredient_profile
        if product_subs & ingredient_categories:
            return 0.3, "generic_nutrient_sub_category_overlap", ingredient_profile
        return 0.2, "generic_nutrient_weak_signal", ingredient_profile

    relation_type = str(score_item.get("relation_type", "") or score_item.get("best_relation_type", "") or "").strip()
    v2_decision = str(score_item.get("v2_decision", "") or "").strip()
    if relation_type == "family_signal" or v2_decision == "family_signal":
        if product_category_is_specific and product_main == ingredient_main:
            return FAMILY_SIGNAL_MAIN_CATEGORY_WEIGHT, "family_signal_main_category_match", ingredient_profile
        if product_category_is_specific and product_main in ingredient_subs:
            return FAMILY_SIGNAL_SUB_CATEGORY_WEIGHT, "family_signal_product_main_in_sub_functions", ingredient_profile
        if product_subs & ingredient_categories:
            return FAMILY_SIGNAL_SUB_CATEGORY_WEIGHT, "family_signal_product_sub_function_overlap", ingredient_profile
        return FAMILY_SIGNAL_WEAK_WEIGHT, "family_signal_weak", ingredient_profile

    if product_category_is_specific and product_main == ingredient_main:
        if review_fallback_signal:
            return REVIEW_FALLBACK_SIGNAL_WEIGHT, "review_fallback_main_category_weak_signal", ingredient_profile
        return 1.0, "main_category_match", ingredient_profile
    if product_category_is_specific and product_main in ingredient_subs:
        if review_fallback_signal:
            return REVIEW_FALLBACK_SIGNAL_WEIGHT, "review_fallback_sub_category_weak_signal", ingredient_profile
        return 0.7, "product_main_in_ingredient_sub_functions", ingredient_profile
    if product_subs & ingredient_categories:
        if review_fallback_signal:
            return REVIEW_FALLBACK_SIGNAL_WEIGHT, "review_fallback_sub_category_weak_signal", ingredient_profile
        return 0.7, "product_sub_function_overlap", ingredient_profile
    if ingredient_type == "functional":
        return 0.3, "functional_weak_signal", ingredient_profile
    return 0.0, "unsupported_ingredient_type", ingredient_profile


def semantic_ingredient_key(ingredient: str) -> str:
    family = infer_joint_ingredient_family(ingredient)
    if family:
        return f"family::{family}"
    return ingredient


def apply_generic_nutrient_total_cap(
    product_profile: dict,
    vector: dict[str, float],
    generic_nutrient_keys: set[str],
    details: dict[str, dict] | None = None,
) -> dict:
    product_main = str(product_profile.get("product_main_category", "") or "").strip()
    if product_main == NUTRITION_MAIN_CATEGORY or not generic_nutrient_keys:
        return {}
    active_nutrient_keys = {key for key in generic_nutrient_keys if vector.get(key, 0.0) > 0}
    if not active_nutrient_keys:
        return {}
    nutrient_total = sum(float(vector.get(key, 0.0) or 0.0) for key in active_nutrient_keys)
    non_nutrient_total = sum(
        float(weight or 0.0)
        for key, weight in vector.items()
        if key not in active_nutrient_keys
    )
    if nutrient_total <= 0 or non_nutrient_total <= 0:
        return {}
    max_nutrient_total = non_nutrient_total * GENERIC_NUTRIENT_VECTOR_SHARE_CAP / (1.0 - GENERIC_NUTRIENT_VECTOR_SHARE_CAP)
    if nutrient_total <= max_nutrient_total:
        return {}

    scale = max_nutrient_total / nutrient_total
    for key in active_nutrient_keys:
        vector[key] = round(float(vector.get(key, 0.0) or 0.0) * scale, 6)
        if details is not None and key in details:
            details[key]["weight"] = vector[key]
            details[key]["reason"] = f"{details[key].get('reason', '')}|generic_nutrient_total_cap"
    return {
        "cap": GENERIC_NUTRIENT_VECTOR_SHARE_CAP,
        "original_nutrient_total": round(float(nutrient_total), 6),
        "capped_nutrient_total": round(float(max_nutrient_total), 6),
        "non_nutrient_total": round(float(non_nutrient_total), 6),
        "scale": round(float(scale), 6),
        "keys": sorted(active_nutrient_keys),
    }


def build_semantic_weight_vector(
    product_profile: dict,
    ingredient_category_profiles: dict[str, dict] | None = None,
    include_details: bool = False,
):
    vector: dict[str, float] = {}
    details: dict[str, dict] = {}
    generic_nutrient_keys: set[str] = set()
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
                    "ingredient_confidence": ingredient_profile.get("confidence", ""),
                    "ingredient_category_reason": str(ingredient_profile.get("reason", "") or ""),
                    "ingredient_source": str(ingredient_profile.get("source", "") or ""),
                    "legacy_category_main": str(ingredient_profile.get("legacy_category_main", "") or ""),
                    "legacy_category_sub": str(ingredient_profile.get("legacy_category_sub", "") or ""),
                    "vector_include": coerce_bool(ingredient_profile.get("vector_include", True), True),
                    "is_excipient": coerce_bool(ingredient_profile.get("is_excipient", False), False),
                    "relation_type": str(score_item.get("relation_type", "") or score_item.get("best_relation_type", "") or ""),
                    "v2_decision": str(score_item.get("v2_decision", "") or ""),
                    "source_raw_ingredients": str(score_item.get("source_raw_ingredients", "") or ""),
                    "family_signal_source_ingredient": str(score_item.get("family_signal_source_ingredient", "") or ""),
                    "family_signal_matched_standard_name": str(score_item.get("family_signal_matched_standard_name", "") or ""),
                },
            )
            details[key]["ingredients"].append(ingredient)
            if weight >= float(details[key].get("weight", 0.0) or 0.0):
                details[key]["weight"] = weight
                details[key]["reason"] = reason
        if weight <= 0:
            continue
        if str(ingredient_profile.get("ingredient_type", "") or "").strip() in {"nutrient", "generic_nutrient"}:
            generic_nutrient_keys.add(key)
        vector[key] = max(vector.get(key, 0.0), float(weight))
    cap_detail = apply_generic_nutrient_total_cap(
        product_profile,
        vector,
        generic_nutrient_keys,
        details if include_details else None,
    )
    if include_details and cap_detail:
        details["__generic_nutrient_total_cap__"] = {
            "semantic_key": "__generic_nutrient_total_cap__",
            "ingredients": [],
            "weight": 0.0,
            "reason": "generic_nutrient_total_cap_summary",
            "cap_detail": cap_detail,
        }
    if include_details:
        return vector, details
    return vector


def semantic_core_keys_from_details(active_vector: dict[str, float], details: dict[str, dict] | None = None) -> set[str]:
    keys = set()
    for key, weight in active_vector.items():
        if key.startswith("__") or float(weight or 0.0) < SEMANTIC_CORE_WEIGHT_MIN:
            continue
        detail = dict((details or {}).get(key, {}) or {})
        ingredient_type = str(detail.get("ingredient_type", "") or "").strip()
        vector_include = coerce_bool(detail.get("vector_include", True), True)
        is_excipient_value = coerce_bool(detail.get("is_excipient", False), False)
        if is_excipient_value or ingredient_type in {"excipient", "additive", "formulation_aid"} or not vector_include:
            continue
        keys.add(key)
    return keys


def is_generic_nutrient_signal_detail(detail: dict) -> bool:
    ingredient_type = str(detail.get("ingredient_type", "") or "").strip()
    if ingredient_type in {"nutrient", "generic_nutrient"}:
        return True
    ingredient_names = [str(name or "").strip() for name in detail.get("ingredients", []) or [] if str(name or "").strip()]
    if ingredient_names and all(is_generic_nutrient_name(name) for name in ingredient_names):
        return True
    ingredient_main = str(detail.get("ingredient_main_category", "") or "").strip()
    weight = float(detail.get("weight", 0.0) or 0.0)
    return ingredient_main == NUTRITION_MAIN_CATEGORY and weight <= 0.4


def is_bone_core_nutrient_signal_detail(detail: dict) -> bool:
    ingredient_names = [str(name or "").strip() for name in detail.get("ingredients", []) or [] if str(name or "").strip()]
    return bool(ingredient_names) and any(is_bone_core_nutrient_name(name) for name in ingredient_names)


def semantic_core_coverage_detail(
    base_profile: dict,
    target_profile: dict,
    base_vector: dict[str, float],
    target_vector: dict[str, float],
    base_details: dict[str, dict] | None = None,
    target_details: dict[str, dict] | None = None,
) -> dict:
    base_core_keys = semantic_core_keys_from_details(base_vector, base_details)
    target_core_keys = semantic_core_keys_from_details(target_vector, target_details)
    shared_core_keys = sorted(base_core_keys & target_core_keys)
    base_coverage = 1.0 if not base_core_keys else len(shared_core_keys) / len(base_core_keys)
    target_coverage = 1.0 if not target_core_keys else len(shared_core_keys) / len(target_core_keys)
    coverage = min(base_coverage, target_coverage)
    if not base_core_keys and not target_core_keys:
        coverage = 1.0
    multiplier = 1.0
    reason = "full_semantic_core_coverage"
    if base_core_keys or target_core_keys:
        if coverage <= 0:
            multiplier = NO_CORE_COVERAGE_MULTIPLIER
            reason = "no_semantic_core_overlap"
        elif coverage < 1.0:
            multiplier = PARTIAL_CORE_COVERAGE_FLOOR + ((1.0 - PARTIAL_CORE_COVERAGE_FLOOR) * coverage)
            reason = "partial_semantic_core_overlap"
    return {
        "base_core_semantic_keys": sorted(base_core_keys),
        "target_core_semantic_keys": sorted(target_core_keys),
        "shared_core_semantic_keys": shared_core_keys,
        "base_core_coverage": round(float(base_coverage), 6),
        "target_core_coverage": round(float(target_coverage), 6),
        "base_primary_semantic_keys": sorted(base_core_keys),
        "target_primary_semantic_keys": sorted(target_core_keys),
        "shared_primary_semantic_keys": shared_core_keys,
        "base_primary_coverage": round(float(base_coverage), 6),
        "target_primary_coverage": round(float(target_coverage), 6),
        "core_coverage": round(float(coverage), 6),
        "multiplier": round(float(multiplier), 6),
        "reason": reason,
        "basis": f"semantic_weight>={SEMANTIC_CORE_WEIGHT_MIN}",
    }


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
        core_coverage = semantic_core_coverage_detail(base_profile, target_profile, base_vector, target_vector, base_details, target_details)
        return 0.0, [], {
            "algorithm": SIMILARITY_ALGORITHM_V2,
            "base_semantic_ingredient_count": len(base_vector),
            "target_semantic_ingredient_count": len(target_vector),
            "shared_semantic_keys": [],
            "core_coverage": core_coverage,
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
    raw_score = 0.0 if denominator <= 0 else float(numerator / denominator)
    core_coverage = semantic_core_coverage_detail(base_profile, target_profile, base_vector, target_vector, base_details, target_details)
    score = raw_score * float(core_coverage.get("multiplier", 1.0) or 1.0)
    score_adjustments = []
    if float(core_coverage.get("multiplier", 1.0) or 1.0) < 1.0:
        score_adjustments.append(
            {
                "type": "semantic_core_coverage_multiplier",
                "multiplier": core_coverage.get("multiplier"),
                "reason": core_coverage.get("reason"),
            }
        )
    base_main_category = str(base_profile.get("product_main_category", "") or "").strip()
    target_main_category = str(target_profile.get("product_main_category", "") or "").strip()
    base_nutrition_subtype = nutrition_subtype_from_profile(base_profile, base_details)
    target_nutrition_subtype = nutrition_subtype_from_profile(target_profile, target_details)
    generic_nutrient_shared_keys = [
        key
        for key in shared_keys
        if (
            is_generic_nutrient_signal_detail(base_details.get(key, {}) or {})
            and is_generic_nutrient_signal_detail(target_details.get(key, {}) or {})
        )
    ]
    non_generic_nutrient_shared_keys = [key for key in shared_keys if key not in set(generic_nutrient_shared_keys)]
    non_generic_nutrient_weak_adjunct_shared = (
        len(non_generic_nutrient_shared_keys) <= GENERIC_NUTRIENT_DOMINANT_MAX_WEAK_ADJUNCT_KEYS
        and all(
            max(float(base_vector.get(key, 0.0) or 0.0), float(target_vector.get(key, 0.0) or 0.0)) < SEMANTIC_CORE_WEIGHT_MIN
            for key in non_generic_nutrient_shared_keys
        )
    )
    generic_nutrient_only_shared = bool(shared_keys) and len(generic_nutrient_shared_keys) == len(shared_keys)
    generic_nutrient_dominant_shared = (
        bool(shared_keys)
        and len(generic_nutrient_shared_keys) >= GENERIC_NUTRIENT_DOMINANT_MIN_GENERIC_KEYS
        and (
            len(non_generic_nutrient_shared_keys) <= GENERIC_NUTRIENT_DOMINANT_MAX_NON_GENERIC_KEYS
            or non_generic_nutrient_weak_adjunct_shared
        )
        and (len(generic_nutrient_shared_keys) / len(shared_keys)) >= GENERIC_NUTRIENT_DOMINANT_SHARED_RATIO
    )
    bone_core_generic_shared = (
        base_main_category == BONE_HEALTH_MAIN_CATEGORY
        and target_main_category == BONE_HEALTH_MAIN_CATEGORY
        and any(
            is_bone_core_nutrient_signal_detail(base_details.get(key, {}) or {})
            or is_bone_core_nutrient_signal_detail(target_details.get(key, {}) or {})
            for key in generic_nutrient_shared_keys
        )
    )
    if generic_nutrient_only_shared and base_main_category != target_main_category:
        original_score = score
        score = min(score, GENERIC_NUTRIENT_ONLY_CROSS_MAIN_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "generic_nutrient_only_cross_main_score_cap",
                "cap": GENERIC_NUTRIENT_ONLY_CROSS_MAIN_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "shared_keys": shared_keys,
            }
        )
    elif (
        (generic_nutrient_only_shared or generic_nutrient_dominant_shared)
        and base_main_category == NUTRITION_MAIN_CATEGORY
        and target_main_category == NUTRITION_MAIN_CATEGORY
    ):
        original_score = score
        score = min(score, GENERIC_NUTRIENT_ONLY_NUTRITION_MAIN_SCORE_CAP)
        score_adjustments.append(
            {
                "type": (
                    "generic_nutrient_only_nutrition_main_score_cap"
                    if generic_nutrient_only_shared
                    else "generic_nutrient_dominant_nutrition_main_score_cap"
                ),
                "cap": GENERIC_NUTRIENT_ONLY_NUTRITION_MAIN_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "main_category": base_main_category,
                "shared_keys": shared_keys,
                "generic_nutrient_shared_keys": generic_nutrient_shared_keys,
                "non_generic_nutrient_shared_keys": non_generic_nutrient_shared_keys,
            }
        )
    elif (
        generic_nutrient_only_shared
        and base_main_category
        and base_main_category == target_main_category
        and base_main_category not in GENERIC_NUTRIENT_ONLY_MAIN_CATEGORY_ALLOWLIST
        and not bone_core_generic_shared
    ):
        original_score = score
        score = min(score, GENERIC_NUTRIENT_ONLY_SAME_MAIN_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "generic_nutrient_only_same_main_score_cap",
                "cap": GENERIC_NUTRIENT_ONLY_SAME_MAIN_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "main_category": base_main_category,
                "shared_keys": shared_keys,
            }
        )
    if (
        score > NUTRITION_SUBTYPE_MISMATCH_SCORE_CAP
        and base_main_category == NUTRITION_MAIN_CATEGORY
        and target_main_category == NUTRITION_MAIN_CATEGORY
        and base_nutrition_subtype
        and target_nutrition_subtype
        and base_nutrition_subtype != target_nutrition_subtype
    ):
        original_score = score
        score = min(score, NUTRITION_SUBTYPE_MISMATCH_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "nutrition_subtype_mismatch_score_cap",
                "cap": NUTRITION_SUBTYPE_MISMATCH_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "base_nutrition_subtype": base_nutrition_subtype,
                "target_nutrition_subtype": target_nutrition_subtype,
                "shared_keys": shared_keys,
            }
        )
    generic_nutrient_shared_ratio = (
        (len(generic_nutrient_shared_keys) / len(shared_keys))
        if shared_keys
        else 0.0
    )
    if (
        score > NUTRITION_GENERIC_LOW_CORE_COVERAGE_SCORE_CAP
        and base_main_category == NUTRITION_MAIN_CATEGORY
        and target_main_category == NUTRITION_MAIN_CATEGORY
        and len(core_coverage.get("shared_core_semantic_keys", []) or []) <= 1
        and len(core_coverage.get("base_core_semantic_keys", []) or []) >= 3
        and float(core_coverage.get("base_core_coverage", 1.0) or 1.0)
        <= NUTRITION_GENERIC_LOW_CORE_COVERAGE_MAX_BASE_CORE_COVERAGE
        and generic_nutrient_shared_ratio >= NUTRITION_GENERIC_LOW_CORE_COVERAGE_MIN_GENERIC_RATIO
    ):
        original_score = score
        score = min(score, NUTRITION_GENERIC_LOW_CORE_COVERAGE_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "nutrition_generic_low_core_coverage_score_cap",
                "cap": NUTRITION_GENERIC_LOW_CORE_COVERAGE_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "base_core_coverage": round(float(core_coverage.get("base_core_coverage", 0.0) or 0.0), 6),
                "shared_core_semantic_keys": list(core_coverage.get("shared_core_semantic_keys", []) or []),
                "base_core_semantic_keys": list(core_coverage.get("base_core_semantic_keys", []) or []),
                "generic_nutrient_shared_ratio": round(float(generic_nutrient_shared_ratio), 6),
                "generic_nutrient_shared_keys": generic_nutrient_shared_keys,
                "shared_keys": shared_keys,
            }
        )
    weak_signal_only_shared = bool(shared_keys) and all(
        max(float(base_vector.get(key, 0.0) or 0.0), float(target_vector.get(key, 0.0) or 0.0)) < SEMANTIC_CORE_WEIGHT_MIN
        for key in shared_keys
    )
    no_semantic_core_keys = not core_coverage.get("base_core_semantic_keys") and not core_coverage.get("target_core_semantic_keys")
    if (
        score > WEAK_SIGNAL_ONLY_CROSS_MAIN_SCORE_CAP
        and weak_signal_only_shared
        and no_semantic_core_keys
        and base_main_category
        and target_main_category
        and base_main_category != target_main_category
    ):
        original_score = score
        score = min(score, WEAK_SIGNAL_ONLY_CROSS_MAIN_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "weak_signal_only_cross_main_score_cap",
                "cap": WEAK_SIGNAL_ONLY_CROSS_MAIN_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "base_main_category": base_main_category,
                "target_main_category": target_main_category,
                "shared_keys": shared_keys,
            }
        )
    elif (
        score > WEAK_SIGNAL_ONLY_SCORE_CAP
        and weak_signal_only_shared
        and no_semantic_core_keys
        and base_main_category
        and base_main_category == target_main_category
        and base_main_category not in GENERIC_NUTRIENT_ONLY_MAIN_CATEGORY_ALLOWLIST
        and not bone_core_generic_shared
    ):
        original_score = score
        score = min(score, WEAK_SIGNAL_ONLY_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "weak_signal_only_score_cap",
                "cap": WEAK_SIGNAL_ONLY_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "main_category": base_main_category,
                "shared_keys": shared_keys,
            }
        )
    base_only_semantic_keys = [
        key
        for key in base_vector
        if not key.startswith("__") and float(base_vector.get(key, 0.0) or 0.0) > 0.0 and key not in shared_keys
    ]
    target_only_semantic_keys = [
        key
        for key in target_vector
        if not key.startswith("__") and float(target_vector.get(key, 0.0) or 0.0) > 0.0 and key not in shared_keys
    ]
    if (
        score > ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_CAP
        and score < ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_MAX
        and base_main_category == ORAL_HEALTH_MAIN_CATEGORY
        and target_main_category == ORAL_HEALTH_MAIN_CATEGORY
        and len(shared_keys) == 1
        and len(core_coverage.get("shared_core_semantic_keys", []) or []) == 1
        and not base_only_semantic_keys
        and len(target_only_semantic_keys) >= ORAL_SINGLE_CORE_BROAD_TARGET_MIN_TARGET_ONLY_KEYS
    ):
        original_score = score
        score = min(score, ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "oral_single_core_broad_target_score_cap",
                "cap": ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "main_category": base_main_category,
                "shared_core_semantic_keys": list(core_coverage.get("shared_core_semantic_keys", []) or []),
                "shared_keys": shared_keys,
                "target_only_semantic_keys": target_only_semantic_keys,
            }
        )
    shared_core_semantic_keys = list(core_coverage.get("shared_core_semantic_keys", []) or [])
    if (
        score > LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_CAP
        and score < LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_MAX
        and base_main_category == LIPID_MAIN_CATEGORY
        and target_main_category == LIPID_MAIN_CATEGORY
        and len(shared_keys) == 1
        and len(shared_core_semantic_keys) == 1
        and shared_core_semantic_keys[0] == LECITHIN_SEMANTIC_KEY
        and not base_only_semantic_keys
        and len(target_only_semantic_keys) == LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_TARGET_ONLY_KEYS
    ):
        original_score = score
        score = min(score, LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "lipid_lecithin_single_core_broad_target_score_cap",
                "cap": LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "main_category": base_main_category,
                "shared_core_semantic_keys": shared_core_semantic_keys,
                "shared_keys": shared_keys,
                "target_only_semantic_keys": target_only_semantic_keys,
            }
        )
    if (
        score > NO_CORE_WEAK_SHARED_WITH_EXTRA_SCORE_CAP
        and base_main_category == target_main_category
        and not (core_coverage.get("shared_core_semantic_keys", []) or [])
        and weak_signal_only_shared
        and len(shared_keys) <= NO_CORE_WEAK_SHARED_WITH_EXTRA_MAX_SHARED_KEYS
        and len(base_only_semantic_keys) < NO_CORE_LOW_SHARED_MIN_BASE_ONLY_KEYS
        and (base_only_semantic_keys or target_only_semantic_keys)
    ):
        original_score = score
        score = min(score, NO_CORE_WEAK_SHARED_WITH_EXTRA_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "no_core_weak_shared_with_extra_score_cap",
                "cap": NO_CORE_WEAK_SHARED_WITH_EXTRA_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "main_category": base_main_category,
                "shared_keys": shared_keys,
                "base_only_semantic_keys": base_only_semantic_keys,
                "target_only_semantic_keys": target_only_semantic_keys,
            }
        )
    if (
        score > SINGLE_CORE_LOW_SHARED_SCORE_CAP
        and base_main_category
        and base_main_category == target_main_category
        and len(core_coverage.get("shared_core_semantic_keys", []) or []) == 1
        and len(shared_keys) <= SINGLE_CORE_LOW_SHARED_MAX_SHARED_KEYS
        and len(base_only_semantic_keys) >= SINGLE_CORE_LOW_SHARED_MIN_BASE_ONLY_KEYS
    ):
        original_score = score
        score = min(score, SINGLE_CORE_LOW_SHARED_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "single_core_low_shared_coverage_score_cap",
                "cap": SINGLE_CORE_LOW_SHARED_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "shared_core_semantic_keys": list(core_coverage.get("shared_core_semantic_keys", []) or []),
                "shared_keys": shared_keys,
                "base_only_semantic_keys": base_only_semantic_keys,
            }
        )
    if (
        score > NO_CORE_LOW_SHARED_SCORE_CAP
        and base_main_category
        and base_main_category == target_main_category
        and not (core_coverage.get("shared_core_semantic_keys", []) or [])
        and (
            len(shared_keys) <= NO_CORE_LOW_SHARED_MAX_SHARED_KEYS
            or (weak_signal_only_shared and len(shared_keys) <= NO_CORE_LOW_SHARED_MAX_WEAK_SHARED_KEYS)
        )
        and len(base_only_semantic_keys) >= NO_CORE_LOW_SHARED_MIN_BASE_ONLY_KEYS
    ):
        original_score = score
        score = min(score, NO_CORE_LOW_SHARED_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "no_core_low_shared_coverage_score_cap",
                "cap": NO_CORE_LOW_SHARED_SCORE_CAP,
                "original_score": round(float(original_score), 6),
                "shared_keys": shared_keys,
                "base_only_semantic_keys": base_only_semantic_keys,
            }
        )
    if (
        score > CROSS_MAIN_SHARED_SUBCATEGORY_ONLY_SCORE_CAP
        and base_main_category not in NON_SPECIFIC_CATEGORY_LABELS
        and target_main_category not in NON_SPECIFIC_CATEGORY_LABELS
        and base_main_category != target_main_category
    ):
        base_categories = product_semantic_category_set(base_profile)
        target_categories = product_semantic_category_set(target_profile)
        shared_categories = sorted((base_categories & target_categories) - NON_SPECIFIC_CATEGORY_LABELS)
        if (
            shared_categories
            and base_main_category not in target_categories
            and target_main_category not in base_categories
        ):
            original_score = score
            score = min(score, CROSS_MAIN_SHARED_SUBCATEGORY_ONLY_SCORE_CAP)
            score_adjustments.append(
                {
                    "type": "cross_main_shared_subcategory_only_score_cap",
                    "cap": CROSS_MAIN_SHARED_SUBCATEGORY_ONLY_SCORE_CAP,
                    "original_score": round(float(original_score), 6),
                    "base_main_category": base_main_category,
                    "target_main_category": target_main_category,
                    "shared_categories": shared_categories,
                    "shared_keys": shared_keys,
                }
            )
    effective_base_total = sum(float(value or 0.0) for value in base_vector.values())
    effective_target_total = sum(float(value or 0.0) for value in target_vector.values())
    base_name_key = normalize_token(base_profile.get("product_name", ""))
    target_name_key = normalize_token(target_profile.get("product_name", ""))
    if (
        score >= 0.999999
        and min(effective_base_total, effective_target_total) <= SPARSE_EXACT_EFFECTIVE_WEIGHT_MAX
        and base_name_key != target_name_key
    ):
        score = min(score, SPARSE_EXACT_SCORE_CAP)
        score_adjustments.append(
            {
                "type": "sparse_exact_score_cap",
                "cap": SPARSE_EXACT_SCORE_CAP,
                "effective_base_total": round(float(effective_base_total), 6),
                "effective_target_total": round(float(effective_target_total), 6),
            }
        )

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
                "base_relation_type": str(base_details.get(key, {}).get("relation_type", "") or ""),
                "target_relation_type": str(target_details.get(key, {}).get("relation_type", "") or ""),
                "base_ingredient_main_category": str(base_details.get(key, {}).get("ingredient_main_category", "") or ""),
                "target_ingredient_main_category": str(target_details.get(key, {}).get("ingredient_main_category", "") or ""),
                "base_ingredient_sub_function_categories": list(
                    base_details.get(key, {}).get("ingredient_sub_function_categories", []) or []
                ),
                "target_ingredient_sub_function_categories": list(
                    target_details.get(key, {}).get("ingredient_sub_function_categories", []) or []
                ),
                "base_ingredient_type": str(base_details.get(key, {}).get("ingredient_type", "") or ""),
                "target_ingredient_type": str(target_details.get(key, {}).get("ingredient_type", "") or ""),
                "base_ingredient_confidence": base_details.get(key, {}).get("ingredient_confidence", ""),
                "target_ingredient_confidence": target_details.get(key, {}).get("ingredient_confidence", ""),
                "base_ingredient_category_reason": str(base_details.get(key, {}).get("ingredient_category_reason", "") or ""),
                "target_ingredient_category_reason": str(target_details.get(key, {}).get("ingredient_category_reason", "") or ""),
                "base_ingredient_source": str(base_details.get(key, {}).get("ingredient_source", "") or ""),
                "target_ingredient_source": str(target_details.get(key, {}).get("ingredient_source", "") or ""),
                "base_legacy_category_main": str(base_details.get(key, {}).get("legacy_category_main", "") or ""),
                "target_legacy_category_main": str(target_details.get(key, {}).get("legacy_category_main", "") or ""),
                "base_legacy_category_sub": str(base_details.get(key, {}).get("legacy_category_sub", "") or ""),
                "target_legacy_category_sub": str(target_details.get(key, {}).get("legacy_category_sub", "") or ""),
                "base_v2_decision": str(base_details.get(key, {}).get("v2_decision", "") or ""),
                "target_v2_decision": str(target_details.get(key, {}).get("v2_decision", "") or ""),
                "base_source_raw_ingredients": str(base_details.get(key, {}).get("source_raw_ingredients", "") or ""),
                "target_source_raw_ingredients": str(target_details.get(key, {}).get("source_raw_ingredients", "") or ""),
                "base_family_signal_source_ingredient": str(base_details.get(key, {}).get("family_signal_source_ingredient", "") or ""),
                "target_family_signal_source_ingredient": str(target_details.get(key, {}).get("family_signal_source_ingredient", "") or ""),
            }
        )

    return round(max(0.0, min(score, 1.0)), 6), shared_labels, {
        "algorithm": SIMILARITY_ALGORITHM_V2,
        "numerator": round(float(numerator), 6),
        "denominator": round(float(denominator), 6),
        "raw_jaccard_score": round(float(raw_score), 6),
        "score_adjustments": score_adjustments,
        "core_coverage": core_coverage,
        "effective_base_total": round(float(effective_base_total), 6),
        "effective_target_total": round(float(effective_target_total), 6),
        "base_semantic_ingredient_count": len(base_vector),
        "target_semantic_ingredient_count": len(target_vector),
        "base_nutrition_subtype": base_nutrition_subtype,
        "target_nutrition_subtype": target_nutrition_subtype,
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


def semantic_core_overlap_score(semantic_explanation: dict | None) -> float | None:
    if not semantic_explanation:
        return None
    core_coverage = dict(semantic_explanation.get("core_coverage", {}) or {})
    base_core = set(core_coverage.get("base_core_semantic_keys", []) or core_coverage.get("base_primary_semantic_keys", []) or [])
    target_core = set(core_coverage.get("target_core_semantic_keys", []) or core_coverage.get("target_primary_semantic_keys", []) or [])
    if not base_core and not target_core:
        return 0.0
    union = base_core | target_core
    if not union:
        return 0.0
    shared_core = set(core_coverage.get("shared_core_semantic_keys", []) or core_coverage.get("shared_primary_semantic_keys", []) or [])
    return round(float(len(shared_core) / len(union)), 6)


def calculate_core_match_score(base_profile: dict, target_profile: dict, semantic_explanation: dict | None = None) -> float:
    semantic_score = semantic_core_overlap_score(semantic_explanation)
    if semantic_score is not None:
        return semantic_score
    base_primary = set(base_profile.get("primary_ingredients", []))
    target_primary = set(target_profile.get("primary_ingredients", []))
    union = base_primary | target_primary
    exact_score = 0.0 if not union else float(len(base_primary & target_primary) / len(union))
    base_primary_families = collect_joint_ingredient_families(base_primary)
    target_primary_families = collect_joint_ingredient_families(target_primary)
    family_union = base_primary_families | target_primary_families
    family_score = 0.0 if not family_union else float(len(base_primary_families & target_primary_families) / len(family_union))
    return round(max(exact_score, family_score), 6)


def semantic_shared_signal_is_weak_only(semantic_explanation: dict | None) -> bool:
    shared_details = list((semantic_explanation or {}).get("shared_semantic_keys", []) or [])
    if not shared_details:
        return False
    for detail in shared_details:
        base_weight = float((detail or {}).get("base_weight", 0.0) or 0.0)
        target_weight = float((detail or {}).get("target_weight", 0.0) or 0.0)
        if max(base_weight, target_weight) >= SEMANTIC_CORE_WEIGHT_MIN:
            return False
    return True


def _semantic_detail_has_review_marker(detail: dict, prefix: str) -> bool:
    values: list[str] = []
    for key in (
        f"{prefix}_ingredient_source",
        f"{prefix}_ingredient_category_reason",
        f"{prefix}_legacy_category_main",
        f"{prefix}_legacy_category_sub",
        f"{prefix}_ingredient_main_category",
    ):
        values.append(str(detail.get(key, "") or ""))
    for item in detail.get(f"{prefix}_ingredient_sub_function_categories", []) or []:
        values.append(str(item or ""))
    text = " ".join(values)
    return any(marker in text for marker in REVIEW_SIGNAL_MARKERS)


def semantic_sparse_review_signal_only(semantic_explanation: dict | None) -> bool:
    explanation = semantic_explanation or {}
    shared_details = list(explanation.get("shared_semantic_keys", []) or [])
    if not shared_details:
        return False
    base_count = int(explanation.get("base_semantic_ingredient_count", 0) or 0)
    target_count = int(explanation.get("target_semantic_ingredient_count", 0) or 0)
    if min(base_count, target_count) > 1:
        return False
    return all(
        _semantic_detail_has_review_marker(detail or {}, "base")
        or _semantic_detail_has_review_marker(detail or {}, "target")
        for detail in shared_details
    )


def semantic_single_core_low_shared_coverage(semantic_explanation: dict | None) -> bool:
    explanation = semantic_explanation or {}
    shared_details = list(explanation.get("shared_semantic_keys", []) or [])
    if not shared_details:
        return False
    core_coverage = explanation.get("core_coverage", {}) or {}
    shared_core_keys = set(core_coverage.get("shared_core_semantic_keys", []) or [])
    if len(shared_core_keys) != 1:
        return False
    if len(shared_details) > SINGLE_CORE_LOW_SHARED_MAX_SHARED_KEYS:
        non_core_shared = [
            detail
            for detail in shared_details
            if str(detail.get("semantic_key", "") or "") not in shared_core_keys
        ]
        if len(non_core_shared) > GENERIC_NUTRIENT_DOMINANT_MAX_WEAK_ADJUNCT_KEYS:
            return False
        if any(
            max(
                float(detail.get("base_weight", 0.0) or 0.0),
                float(detail.get("target_weight", 0.0) or 0.0),
            )
            >= SEMANTIC_CORE_WEIGHT_MIN
            for detail in non_core_shared
        ):
            return False
    base_count = int(explanation.get("base_semantic_ingredient_count", 0) or 0)
    return (base_count - len(shared_details)) >= SINGLE_CORE_LOW_SHARED_MIN_BASE_ONLY_KEYS


def semantic_oral_single_core_broad_target(
    base_main_category: str,
    target_main_category: str,
    semantic_explanation: dict | None,
    similarity_score: float,
) -> bool:
    score = float(similarity_score or 0.0)
    if not (
        ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_CAP < score < ORAL_SINGLE_CORE_BROAD_TARGET_SCORE_MAX
        and str(base_main_category or "").strip() == ORAL_HEALTH_MAIN_CATEGORY
        and str(target_main_category or "").strip() == ORAL_HEALTH_MAIN_CATEGORY
    ):
        return False
    explanation = semantic_explanation or {}
    shared_details = list(explanation.get("shared_semantic_keys", []) or [])
    core_coverage = explanation.get("core_coverage", {}) or {}
    shared_core_keys = set(
        core_coverage.get("shared_core_semantic_keys", [])
        or core_coverage.get("shared_primary_semantic_keys", [])
        or []
    )
    base_count = int(explanation.get("base_semantic_ingredient_count", 0) or 0)
    target_count = int(explanation.get("target_semantic_ingredient_count", 0) or 0)
    base_only_count = max(0, base_count - len(shared_details))
    target_only_count = max(0, target_count - len(shared_details))
    return (
        len(shared_details) == 1
        and len(shared_core_keys) == 1
        and base_only_count == 0
        and target_only_count >= ORAL_SINGLE_CORE_BROAD_TARGET_MIN_TARGET_ONLY_KEYS
    )


def semantic_lipid_lecithin_single_core_broad_target(
    base_main_category: str,
    target_main_category: str,
    semantic_explanation: dict | None,
    similarity_score: float,
) -> bool:
    score = float(similarity_score or 0.0)
    if not (
        LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_CAP
        < score
        < LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_SCORE_MAX
        and str(base_main_category or "").strip() == LIPID_MAIN_CATEGORY
        and str(target_main_category or "").strip() == LIPID_MAIN_CATEGORY
    ):
        return False
    explanation = semantic_explanation or {}
    shared_details = list(explanation.get("shared_semantic_keys", []) or [])
    core_coverage = explanation.get("core_coverage", {}) or {}
    shared_core_keys = list(
        core_coverage.get("shared_core_semantic_keys", [])
        or core_coverage.get("shared_primary_semantic_keys", [])
        or []
    )
    base_count = int(explanation.get("base_semantic_ingredient_count", 0) or 0)
    target_count = int(explanation.get("target_semantic_ingredient_count", 0) or 0)
    base_only_count = max(0, base_count - len(shared_details))
    target_only_count = max(0, target_count - len(shared_details))
    return (
        len(shared_details) == 1
        and len(shared_core_keys) == 1
        and shared_core_keys[0] == LECITHIN_SEMANTIC_KEY
        and base_only_count == 0
        and target_only_count == LIPID_LECITHIN_SINGLE_CORE_BROAD_TARGET_TARGET_ONLY_KEYS
    )


def recommendation_quality_metadata(
    similarity_score: float,
    semantic_explanation: dict | None = None,
    core_match_score: float | None = None,
    function_similarity_score: float | None = None,
    exact_match: bool = False,
) -> dict:
    if exact_match:
        return {
            "recommendation_quality": "exact_match",
            "recommendation_display_eligible": True,
            "recommendation_review_reason": "",
        }
    score = float(similarity_score or 0.0)
    if score >= RECOMMENDATION_DISPLAY_SCORE_MIN:
        return {
            "recommendation_quality": "strong_match",
            "recommendation_display_eligible": True,
            "recommendation_review_reason": "",
        }
    quality = "low_confidence_match" if score >= RECOMMENDATION_REVIEW_SCORE_MIN else "very_low_confidence_match"
    adjustment_types = [
        str(item.get("type", "") or "")
        for item in (semantic_explanation or {}).get("score_adjustments", []) or []
        if str(item.get("type", "") or "")
    ]
    weak_signal_only = any("weak_signal_only" in item for item in adjustment_types) or semantic_shared_signal_is_weak_only(
        semantic_explanation
    )
    sparse_review_signal_only = semantic_sparse_review_signal_only(semantic_explanation)
    single_core_low_shared_coverage = semantic_single_core_low_shared_coverage(semantic_explanation)
    if score < RECOMMENDATION_REVIEW_SCORE_MIN:
        reason = "below_review_score_min"
    elif any("nutrition_subtype_mismatch" in item for item in adjustment_types):
        reason = "nutrition_subtype_mismatch"
    elif any("nutrition_generic_low_core_coverage" in item for item in adjustment_types):
        reason = "nutrition_generic_low_core_coverage"
    elif any("generic_nutrient" in item for item in adjustment_types):
        reason = "generic_nutrient_only_or_dominant"
    elif any("cross_main_shared_subcategory_only" in item for item in adjustment_types):
        reason = "cross_main_shared_subcategory_only"
    elif any("oral_single_core_broad_target" in item for item in adjustment_types):
        reason = "oral_single_core_broad_target"
    elif any("lipid_lecithin_single_core_broad_target" in item for item in adjustment_types):
        reason = "lipid_lecithin_single_core_broad_target"
    elif any("low_shared_coverage" in item for item in adjustment_types):
        reason = "low_shared_coverage"
    elif single_core_low_shared_coverage:
        reason = "single_core_low_shared_coverage"
    elif sparse_review_signal_only:
        reason = "sparse_review_signal_only"
    elif weak_signal_only:
        reason = "weak_signal_only"
    elif core_match_score is not None and float(core_match_score or 0.0) <= 0.0:
        reason = "no_core_overlap"
    elif (
        score < RECOMMENDATION_LOW_SCORE_CORE_MIN
        and core_match_score is not None
        and float(core_match_score or 0.0) < RECOMMENDATION_LOW_SCORE_CORE_MATCH_MIN
    ):
        reason = "low_score_low_core_overlap"
    elif function_similarity_score is not None and float(function_similarity_score or 0.0) < RECOMMENDATION_REVIEW_FUNCTION_MIN:
        reason = "low_function_similarity"
    else:
        return {
            "recommendation_quality": quality,
            "recommendation_display_eligible": True,
            "recommendation_review_reason": "",
        }
    return {
        "recommendation_quality": quality,
        "recommendation_display_eligible": False,
        "recommendation_review_reason": reason,
    }


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


def semantic_shared_name_priority(semantic_explanation: dict | None) -> dict[str, dict]:
    priority: dict[str, dict] = {}
    if not semantic_explanation:
        return priority
    core_coverage = dict(semantic_explanation.get("core_coverage", {}) or {})
    core_keys = set(core_coverage.get("shared_core_semantic_keys", []) or core_coverage.get("shared_primary_semantic_keys", []) or [])
    for detail in semantic_explanation.get("shared_semantic_keys", []) or []:
        key = str(detail.get("semantic_key", "") or "")
        weight = min(float(detail.get("base_weight", 0.0) or 0.0), float(detail.get("target_weight", 0.0) or 0.0))
        is_core = int(key in core_keys)
        names = [str(detail.get("label", "") or "")]
        names.extend(str(name) for name in detail.get("base_ingredients", []) or [])
        names.extend(str(name) for name in detail.get("target_ingredients", []) or [])
        for name in names:
            name = name.strip()
            if not name:
                continue
            existing = priority.get(name, {})
            if is_core > int(existing.get("is_core", 0) or 0) or weight > float(existing.get("weight", 0.0) or 0.0):
                priority[name] = {"is_core": is_core, "weight": weight, "semantic_key": key}
    return priority


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


def has_semantic_core_overlap(semantic_explanation: dict | None) -> bool | None:
    if not semantic_explanation:
        return None
    core_coverage = dict(semantic_explanation.get("core_coverage", {}) or {})
    return bool(core_coverage.get("shared_core_semantic_keys") or core_coverage.get("shared_primary_semantic_keys"))


def classify_substitutability(
    similarity_score: float,
    base_profile: dict,
    target_profile: dict,
    semantic_explanation: dict | None = None,
) -> str:
    same_main = base_profile.get("product_main_category") == target_profile.get("product_main_category")
    has_core_overlap = has_semantic_core_overlap(semantic_explanation)
    if has_core_overlap is None:
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


def choose_reason_ingredients(
    base_profile: dict,
    target_profile: dict,
    comparison: dict,
    ingredient_frequency: dict[str, int],
    semantic_explanation: dict | None = None,
) -> tuple[list[str], list[str]]:
    shared = [name for name in comparison["shared_ingredients"] if not is_excipient(name)]
    if not shared:
        return [], []

    category = str(base_profile.get("product_main_category") or "기타")

    semantic_priority = semantic_shared_name_priority(semantic_explanation)

    joint_related = product_is_joint_related(base_profile) or product_is_joint_related(target_profile)
    bone_related = product_is_bone_related(base_profile) or product_is_bone_related(target_profile)
    memory_related = product_is_memory_related(base_profile) or product_is_memory_related(target_profile)
    skin_related = product_is_skin_related(base_profile) or product_is_skin_related(target_profile)
    male_related = product_is_male_related(base_profile) or product_is_male_related(target_profile)
    if semantic_explanation:
        joint_family_overlap = collect_joint_ingredient_families(
            [name for name, info in semantic_priority.items() if int(info.get("is_core", 0) or 0) > 0]
        )
    else:
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
        semantic_info = semantic_priority.get(name, {})
        semantic_core = int(semantic_info.get("is_core", 0) or 0)
        semantic_weight = float(semantic_info.get("weight", 0.0) or 0.0)
        direct_category = int(ingredient_matches_patterns(name, category_patterns))
        joint_core = int(joint_related and ingredient_matches_patterns(name, JOINT_KEYWORDS))
        memory_core = int(memory_related and ingredient_matches_patterns(name, MEMORY_KEYWORDS))
        skin_core = int(skin_related and ingredient_matches_patterns(name, SKIN_KEYWORDS))
        male_core = int(male_related and ingredient_matches_patterns(name, MALE_KEYWORDS))
        low_priority = int(is_low_priority_reason_ingredient(name, category))
        rarity = ingredient_frequency.get(name, 999999)
        return (
            semantic_core,
            semantic_weight,
            joint_core,
            memory_core,
            skin_core,
            male_core,
            direct_category,
            -low_priority,
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


def generate_recommendation_reason(
    base_profile: dict,
    target_profile: dict,
    comparison: dict,
    ingredient_frequency: dict[str, int],
    semantic_explanation: dict | None = None,
) -> str:
    main_category = str(base_profile.get("product_main_category") or "기타")
    top_reason_ingredients, highlighted = choose_reason_ingredients(
        base_profile,
        target_profile,
        comparison,
        ingredient_frequency,
        semantic_explanation,
    )
    phrase = render_reason_ingredient_phrase(top_reason_ingredients)
    semantic_core_overlap = has_semantic_core_overlap(semantic_explanation)
    if semantic_core_overlap is None:
        semantic_core_overlap = bool(set(base_profile.get("primary_ingredients", [])) & set(target_profile.get("primary_ingredients", [])))
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

    if semantic_core_overlap and main_category != "기타":
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
    base_main_category = base_profile.get("product_main_category")
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
        if profile.get("product_main_category") == base_main_category:
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
                if profiles[candidate_id].get("product_main_category") != base_main_category:
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
    base_main_category = base_profile.get("product_main_category")
    base_categories = product_semantic_category_set(base_profile)
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
        same_main = profile.get("product_main_category") == base_main_category
        if same_main:
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
                if profiles[candidate_id].get("product_main_category") != base_main_category:
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
        raise ValueError("Only the current default similarity algorithm can be cached.")
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
                base_semantic_shared = set()
                target_semantic_shared = set()
                for detail in semantic_explanation.get("shared_semantic_keys", []) or []:
                    base_semantic_shared.update(str(name) for name in detail.get("base_ingredients", []) or [])
                    target_semantic_shared.update(str(name) for name in detail.get("target_ingredients", []) or [])
                comparison["base_only_ingredients"] = [
                    name for name in comparison["base_only_ingredients"] if name not in base_semantic_shared
                ]
                comparison["target_only_ingredients"] = [
                    name for name in comparison["target_only_ingredients"] if name not in target_semantic_shared
                ]
                comparison["base_only_ingredients"] = [
                    name for name in comparison["base_only_ingredients"] if not is_semantic_excipient_name(name)
                ]
                comparison["target_only_ingredients"] = [
                    name for name in comparison["target_only_ingredients"] if not is_semantic_excipient_name(name)
                ]
            function_similarity_score = calculate_function_similarity(base_profile, target_profile)
            core_match_score = calculate_core_match_score(base_profile, target_profile, semantic_explanation)
            substitutability = classify_substitutability(similarity_score, base_profile, target_profile, semantic_explanation)
            reason = generate_recommendation_reason(base_profile, target_profile, comparison, ingredient_frequency, semantic_explanation)
            explanation_json = build_explanation_json(reason, comparison, substitutability)
            explanation_json["similarity_algorithm"] = similarity_algorithm
            if semantic_explanation:
                explanation_json["semantic_weighted_jaccard_v2"] = semantic_explanation
            quality_metadata = recommendation_quality_metadata(
                similarity_score,
                semantic_explanation,
                core_match_score,
                function_similarity_score,
            )
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
                    **quality_metadata,
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
