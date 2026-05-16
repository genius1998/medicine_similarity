from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import get_config_value, load_config


TABLE_NAME = "product_function_profile"


SUPPORT_PATTERNS = [
    r"^비타민",
    r"칼슘",
    r"마그네슘",
    r"아연",
    r"셀레늄",
    r"셀렌",
    r"망간",
    r"구리",
    r"칼륨",
    r"철",
    r"엽산",
    r"크롬",
    r"나이아신",
    r"판토텐산",
    r"비오틴",
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

CATEGORY_KEYWORDS = {
    "눈 건강": [r"루테인", r"지아잔틴", r"마리골드", r"아스타잔틴", r"눈"],
    "장 건강": [r"프로바이오틱스", r"유산균", r"이눌린", r"치커리", r"프락토올리고당", r"갈락토올리고당", r"장"],
    "면역": [r"홍삼", r"인삼", r"베타글루칸", r"면역"],
    "피로개선": [r"홍삼", r"인삼", r"옥타코사놀", r"피로", r"활력"],
    "혈행": [r"epa", r"dha", r"오메가", r"혈행", r"은행잎"],
    "혈중지질": [r"epa", r"dha", r"오메가", r"홍국", r"폴리코사놀", r"콜레스테롤"],
    "뼈 건강": [r"칼슘", r"비타민 d", r"비타민 k", r"k2", r"칼마디", r"뼈", r"마그네슘"],
    "관절/연골": [r"\bmsm\b", r"엠에스엠", r"\bnem\b", r"난각막", r"난각막가수분해물", r"보스웰리아", r"초록입홍합", r"nag", r"글루코사민", r"콘드로이친", r"관절", r"연골"],
    "간 건강": [r"밀크씨슬", r"실리마린", r"헛개", r"간"],
    "피부 건강": [r"콜라겐", r"히알루론산", r"세라마이드", r"레티놀", r"피부"],
    "항산화": [r"비타민 c", r"비타민 e", r"셀레늄", r"q10", r"항산화"],
    "기억력": [r"포스파티딜세린", r"은행잎", r"기억"],
    "인지력": [r"포스파티딜세린", r"인지"],
    "체지방": [r"가르시니아", r"\bcla\b", r"녹차추출물", r"체지방"],
    "혈당": [r"난소화성말토덱스트린", r"바나바", r"혈당"],
    "혈압": [r"q10", r"혈압"],
    "여성 건강": [r"감마리놀렌산", r"이소플라본", r"백수오", r"크랜베리", r"여성"],
    "남성 건강": [r"쏘팔메토", r"전립선", r"녹용", r"남성"],
    "구강 건강": [r"자일리톨", r"구강"],
    "수면/긴장완화": [r"테아닌", r"\bgaba\b", r"수면", r"긴장", r"스트레스"],
    "운동/근력": [r"\bhmb\b", r"크레아틴", r"근력", r"운동", r"근육"],
    "영양보충": SUPPORT_PATTERNS,
}

MEMORY_NAME_PATTERNS = [r"인지", r"기억", r"두뇌", r"브레인", r"포스파티딜세린", r"\bps\b"]
MEMORY_INGREDIENT_PATTERNS = [r"포스파티딜세린", r"phosphatidylserine", r"\bps\b", r"은행잎", r"은행잎추출물", r"테아닌", r"\bdha\b"]
SKIN_NAME_PATTERNS = [r"콜라겐", r"피부", r"레티놀", r"레티놀a", r"히알루론", r"세라마이드", r"엘라스틴", r"비오틴"]
SKIN_INGREDIENT_PATTERNS = [r"콜라겐", r"저분자콜라겐", r"저분자콜라겐펩타이드", r"콜라겐펩타이드", r"히알루론산", r"세라마이드", r"엘라스틴", r"비오틴", r"비타민 a", r"레티놀", r"셀렌", r"셀레늄"]
SKIN_PRIMARY_PATTERNS = [r"콜라겐", r"저분자콜라겐", r"저분자콜라겐펩타이드", r"콜라겐펩타이드", r"비오틴", r"비타민 a", r"레티놀"]
MALE_NAME_PATTERNS = [r"전립선", r"남성", r"쏘팔메토", r"옥타코사놀", r"로르산", r"지구력", r"활력"]
MALE_INGREDIENT_PATTERNS = [r"쏘팔메토", r"쏘팔메토열매추출물", r"saw palmetto", r"로르산", r"옥타코사놀", r"아연", r"녹용", r"마카", r"l-아르지닌", r"아르기닌"]
MALE_PRIMARY_PATTERNS = [r"쏘팔메토", r"쏘팔메토열매추출물", r"saw palmetto", r"로르산", r"옥타코사놀", r"녹용", r"마카", r"l-아르지닌", r"아르기닌", r"아연"]


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
    category_map_candidates = [
        output_dir / "functional_category_map.csv",
        ROOT_DIR / "output" / "functional_category_map.csv",
    ]
    sqlite_candidates = [
        Path(str(get_config_value(config, "sqlite_path", ""))),
        Path(r"D:\ec2_cache_snapshot\ingredient_match_cache_rebuilt_item_class_i0050_final.sqlite"),
    ]
    return {
        "output_dir": output_dir,
        "vector_csv_path": first_existing_path(vector_candidates),
        "category_map_csv_path": first_existing_path(category_map_candidates),
        "sqlite_path": first_existing_path([path for path in sqlite_candidates if str(path)]),
    }


def backup_table_if_exists(conn: sqlite3.Connection, table_name: str) -> str | None:
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if not existing:
        return None
    backup_name = f"{table_name}__backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conn.execute(f'CREATE TABLE "{backup_name}" AS SELECT * FROM "{table_name}"')
    conn.commit()
    return backup_name


def to_json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def ingredient_matches_patterns(name: str, patterns: list[str]) -> bool:
    lowered = normalize_text(name)
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def is_support_ingredient(name: str) -> bool:
    return ingredient_matches_patterns(name, SUPPORT_PATTERNS)


def is_excipient(name: str) -> bool:
    return ingredient_matches_patterns(name, EXCIPIENT_PATTERNS)


def product_matches_patterns(product_name: str, patterns: list[str]) -> bool:
    return ingredient_matches_patterns(product_name, patterns)


def get_name_keyword_boosts(product_name: str) -> dict[str, float]:
    name = normalize_text(product_name)
    boosts: dict[str, float] = defaultdict(float)
    rules = [
        ("관절/연골", [r"관절", r"연골", r"\bnem\b", r"\bmsm\b", r"난각막"], 2.4),
        ("뼈 건강", [r"뼈", r"칼마디", r"칼슘", r"비타민d", r"k2"], 2.1),
        ("장 건강", [r"장", r"프로바이오틱스", r"유산균"], 1.8),
        ("눈 건강", [r"눈", r"루테인", r"지아잔틴"], 1.8),
        ("인지력", [r"인지", r"포스파티딜세린"], 1.8),
        ("기억력", [r"기억", r"포스파티딜세린"], 1.5),
        ("기억력/인지력", MEMORY_NAME_PATTERNS, 3.2),
        ("피부 건강", SKIN_NAME_PATTERNS, 2.9),
        ("남성 건강", MALE_NAME_PATTERNS, 2.7),
    ]
    for category, patterns, score in rules:
        if any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in patterns):
            boosts[category] += score
    return boosts


def load_inputs(vector_csv_path: Path, category_map_csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    vector_df = pd.read_csv(vector_csv_path, encoding="utf-8-sig", low_memory=False)
    required_vector_columns = ["product_id", "product_name", "matched_standard_name", "weight"]
    missing = [column for column in required_vector_columns if column not in vector_df.columns]
    if missing:
        raise ValueError(f"벡터 CSV 필수 컬럼이 없습니다: {missing}")

    vector_df["product_id"] = vector_df["product_id"].fillna("").astype(str).str.strip()
    vector_df["product_name"] = vector_df["product_name"].fillna("").astype(str).str.strip()
    vector_df["matched_standard_name"] = vector_df["matched_standard_name"].fillna("").astype(str).str.strip()
    vector_df["weight"] = pd.to_numeric(vector_df["weight"], errors="coerce").fillna(0.0)
    vector_df = vector_df[(vector_df["product_id"] != "") & (vector_df["matched_standard_name"] != "") & (vector_df["weight"] > 0)].copy()

    category_df = pd.read_csv(category_map_csv_path, encoding="utf-8-sig", low_memory=False)
    category_df["functional_ingredient_name"] = category_df["functional_ingredient_name"].fillna("").astype(str).str.strip()
    category_df["category_main"] = category_df["category_main"].fillna("기타").astype(str).str.strip()
    category_df["category_sub"] = category_df["category_sub"].fillna("").astype(str).str.strip()
    category_df["claim_text"] = category_df["claim_text"].fillna("").astype(str).str.strip()
    category_df["confidence"] = pd.to_numeric(category_df["confidence"], errors="coerce").fillna(0.0)
    if "categories_json" not in category_df.columns:
        category_df["categories_json"] = "[]"
    return vector_df, category_df


def extract_report_no(row: pd.Series) -> str:
    for column in ["품목제조번호", "품목보고번호"]:
        value = str(row.get(column, "") or "").strip()
        if value and value.lower() != "nan":
            return value
    product_id = str(row.get("product_id", "") or "").strip()
    if "::" in product_id:
        return product_id.split("::", 1)[1]
    return product_id


def compute_profile_for_product(product_id: str, group: pd.DataFrame) -> dict:
    group = group.sort_values(by=["weight", "matched_standard_name"], ascending=[False, True]).reset_index(drop=True)
    product_name = str(group.iloc[0]["product_name"] or "").strip()
    report_no = extract_report_no(group.iloc[0])
    name_boosts = get_name_keyword_boosts(product_name)
    product_name_is_memory = product_matches_patterns(product_name, MEMORY_NAME_PATTERNS)
    product_name_is_skin = product_matches_patterns(product_name, SKIN_NAME_PATTERNS)
    product_name_is_male = product_matches_patterns(product_name, MALE_NAME_PATTERNS)

    category_scores: dict[str, float] = defaultdict(float)
    ingredient_rows = []
    sub_categories = []
    unknown_weight = 0.0
    total_weight = float(group["weight"].sum())
    top_weight = float(group["weight"].max()) if not group.empty else 0.0

    for row in group.itertuples(index=False):
        ingredient = str(row.matched_standard_name or "").strip()
        category_main = str(row.category_main or "기타").strip() or "기타"
        category_sub = str(row.category_sub or "").strip()
        claim_text = str(row.claim_text or "").strip()
        map_confidence = float(row.map_confidence or 0.0)
        weight = float(row.weight)
        support_flag = is_support_ingredient(ingredient)
        excipient_flag = is_excipient(ingredient)
        core_pattern_match = ingredient_matches_patterns(ingredient, CATEGORY_KEYWORDS.get(category_main, []))

        category_score = weight * max(0.45, map_confidence)
        if support_flag and category_main == "영양보충":
            category_score *= 0.65
        if support_flag and category_main in {"뼈 건강", "장 건강", "관절/연골"} and not core_pattern_match:
            category_score *= 0.75
        if excipient_flag:
            category_score *= 0.12
        if core_pattern_match:
            category_score *= 1.15

        category_scores[category_main] += category_score
        if category_main == "기타":
            unknown_weight += weight

        if category_sub:
            for item in [part.strip() for part in category_sub.split("|") if part.strip()]:
                if item not in sub_categories:
                    sub_categories.append(item)

        ingredient_rows.append(
            {
                "ingredient": ingredient,
                "weight": round(weight, 6),
                "category_main": category_main,
                "category_sub": category_sub,
                "claim_text": claim_text,
                "map_confidence": round(map_confidence, 4),
                "support_flag": support_flag,
                "excipient_flag": excipient_flag,
                "core_pattern_match": core_pattern_match,
            }
        )

    memory_ingredients = [item["ingredient"] for item in ingredient_rows if ingredient_matches_patterns(item["ingredient"], MEMORY_INGREDIENT_PATTERNS)]
    skin_ingredients = [item["ingredient"] for item in ingredient_rows if ingredient_matches_patterns(item["ingredient"], SKIN_INGREDIENT_PATTERNS)]
    skin_primary_ingredients = [item["ingredient"] for item in ingredient_rows if ingredient_matches_patterns(item["ingredient"], SKIN_PRIMARY_PATTERNS)]
    male_ingredients = [item["ingredient"] for item in ingredient_rows if ingredient_matches_patterns(item["ingredient"], MALE_INGREDIENT_PATTERNS)]
    male_primary_ingredients = [item["ingredient"] for item in ingredient_rows if ingredient_matches_patterns(item["ingredient"], MALE_PRIMARY_PATTERNS)]
    has_phosphatidylserine = any(ingredient_matches_patterns(name, [r"포스파티딜세린", r"phosphatidylserine", r"\bps\b"]) for name in memory_ingredients)
    has_collagen = any(ingredient_matches_patterns(name, [r"콜라겐", r"저분자콜라겐", r"콜라겐펩타이드"]) for name in skin_primary_ingredients)
    has_skin_support_pair = any(ingredient_matches_patterns(name, [r"비오틴", r"비타민 a", r"레티놀"]) for name in skin_ingredients)
    has_male_core = any(ingredient_matches_patterns(name, [r"쏘팔메토", r"saw palmetto", r"로르산", r"옥타코사놀"]) for name in male_primary_ingredients)
    has_male_zinc_only = product_name_is_male and any(ingredient_matches_patterns(name, [r"아연"]) for name in male_ingredients)

    for category, boost in name_boosts.items():
        category_scores[category] += boost

    if product_name_is_memory and has_phosphatidylserine:
        category_scores["기억력/인지력"] += 7.5
        category_scores["기억력"] += 2.0
        category_scores["인지력"] += 2.0

    if product_name_is_skin and (has_collagen or has_skin_support_pair):
        category_scores["피부 건강"] += 6.5 if has_collagen else 5.2

    if product_name_is_male and (has_male_core or has_male_zinc_only or male_ingredients):
        category_scores["남성 건강"] += 6.0 if has_male_core else 4.0

    sorted_categories = sorted(category_scores.items(), key=lambda item: (-item[1], item[0]))
    product_main_category = sorted_categories[0][0] if sorted_categories else "기타"
    if product_name_is_memory and has_phosphatidylserine:
        product_main_category = "기억력/인지력"
    elif product_name_is_skin and (has_collagen or has_skin_support_pair):
        product_main_category = "피부 건강"
    elif has_male_core:
        product_main_category = "남성 건강"
    elif product_name_is_male and has_male_zinc_only:
        product_main_category = "남성 건강"
    top_score = sorted_categories[0][1] if sorted_categories else 0.0
    second_score = sorted_categories[1][1] if len(sorted_categories) > 1 else 0.0

    category_rank = {category: idx for idx, (category, _) in enumerate(sorted_categories)}
    primary_ingredients: list[str] = []
    secondary_ingredients: list[str] = []
    support_ingredients: list[str] = []
    ingredient_scores = []

    only_support_style = all(item["support_flag"] or item["excipient_flag"] for item in ingredient_rows) if ingredient_rows else False

    for item in ingredient_rows:
        ingredient = item["ingredient"]
        weight = float(item["weight"])
        category_main = item["category_main"]
        support_flag = bool(item["support_flag"])
        excipient_flag = bool(item["excipient_flag"])
        core_pattern_match = bool(item["core_pattern_match"])
        direct_main_match = category_main == product_main_category
        product_name_match = ingredient_matches_patterns(product_name, [re.escape(ingredient[:20])]) if len(ingredient) >= 2 else False

        role = "secondary"
        if excipient_flag:
            role = "support"
        elif only_support_style and product_main_category in {"영양보충", "뼈 건강"}:
            role = "primary" if weight >= top_weight * 0.6 else "secondary"
        elif direct_main_match and (core_pattern_match or weight >= top_weight * 0.72 or product_name_match):
            role = "primary"
        elif category_rank.get(category_main, 99) <= 1 and (core_pattern_match or weight >= top_weight * 0.58):
            role = "secondary"
        elif support_flag:
            role = "support"

        if product_main_category == "관절/연골" and ingredient_matches_patterns(
            ingredient,
            [r"\bnem\b", r"난각막", r"난각막가수분해물", r"\bmsm\b", r"엠에스엠", r"보스웰리아", r"초록입홍합", r"\bnag\b", r"글루코사민", r"콘드로이친"],
        ):
            role = "primary"
        elif product_main_category == "뼈 건강" and ingredient_matches_patterns(
            ingredient,
            [r"칼슘", r"비타민 d", r"비타민 k", r"\bk2\b"],
        ):
            role = "primary"
        elif product_main_category == "뼈 건강" and ingredient_matches_patterns(
            ingredient,
            [r"\bmsm\b", r"엠에스엠", r"보스웰리아", r"초록입홍합", r"\bnem\b", r"난각막"],
        ):
            role = "secondary"
            if ingredient in support_ingredients:
                support_ingredients = [name for name in support_ingredients if name != ingredient]
        elif product_main_category == "기억력/인지력" and ingredient_matches_patterns(
            ingredient,
            [r"포스파티딜세린", r"phosphatidylserine", r"\bps\b"],
        ):
            role = "primary"
        elif product_main_category == "기억력/인지력" and ingredient_matches_patterns(
            ingredient,
            [r"은행잎", r"은행잎추출물", r"테아닌", r"\bdha\b"],
        ):
            role = "secondary"
        elif product_main_category == "기억력/인지력" and ingredient_matches_patterns(
            ingredient,
            [r"비타민 a", r"비타민 b", r"비타민 e", r"아연", r"비타민 d", r"비타민 k"],
        ):
            role = "support"
        elif product_main_category == "피부 건강" and ingredient_matches_patterns(
            ingredient,
            [r"콜라겐", r"저분자콜라겐", r"저분자콜라겐펩타이드", r"콜라겐펩타이드"],
        ):
            role = "primary"
        elif product_main_category == "피부 건강" and ingredient_matches_patterns(
            ingredient,
            [r"비오틴", r"비타민 a", r"레티놀", r"히알루론산", r"세라마이드", r"엘라스틴"],
        ):
            role = "primary" if product_name_is_skin and weight >= top_weight * 0.45 else "secondary"
        elif product_main_category == "피부 건강" and ingredient_matches_patterns(
            ingredient,
            [r"셀렌", r"셀레늄", r"아연"],
        ):
            role = "secondary"
        elif product_main_category == "피부 건강" and ingredient_matches_patterns(
            ingredient,
            [r"마그네슘", r"비타민 d", r"비타민 k"],
        ):
            role = "support"
        elif product_main_category == "남성 건강" and ingredient_matches_patterns(
            ingredient,
            [r"쏘팔메토", r"쏘팔메토열매추출물", r"saw palmetto", r"로르산", r"옥타코사놀", r"녹용", r"마카", r"l-아르지닌", r"아르기닌", r"아연"],
        ):
            role = "primary"
        elif product_main_category == "남성 건강" and ingredient_matches_patterns(
            ingredient,
            [r"비타민 b", r"망간", r"판토텐산", r"마그네슘"],
        ):
            role = "support"
        elif product_main_category == "남성 건강" and ingredient_matches_patterns(
            ingredient,
            [r"난소화성말토덱스트린"],
        ):
            role = "secondary"

        payload = {
            "ingredient": ingredient,
            "weight": round(weight, 6),
            "role": role,
            "category_main": category_main,
            "category_sub": item["category_sub"],
        }
        ingredient_scores.append(payload)
        if role == "primary":
            if ingredient not in primary_ingredients:
                primary_ingredients.append(ingredient)
        elif role == "support":
            if ingredient not in support_ingredients:
                support_ingredients.append(ingredient)
        else:
            if ingredient not in secondary_ingredients:
                secondary_ingredients.append(ingredient)

    if not primary_ingredients and ingredient_rows:
        primary_ingredients.append(ingredient_rows[0]["ingredient"])
        secondary_ingredients = [name for name in secondary_ingredients if name != ingredient_rows[0]["ingredient"]]

    if product_main_category == "기억력/인지력":
        primary_ingredients = sorted(primary_ingredients, key=lambda name: (0 if ingredient_matches_patterns(name, [r"포스파티딜세린", r"phosphatidylserine", r"\bps\b"]) else 1, primary_ingredients.index(name)))
    elif product_main_category == "피부 건강":
        primary_ingredients = sorted(primary_ingredients, key=lambda name: (0 if ingredient_matches_patterns(name, [r"콜라겐", r"저분자콜라겐", r"콜라겐펩타이드"]) else 1 if ingredient_matches_patterns(name, [r"비오틴", r"비타민 a", r"레티놀"]) else 2, primary_ingredients.index(name)))
    elif product_main_category == "남성 건강":
        primary_ingredients = sorted(primary_ingredients, key=lambda name: (0 if ingredient_matches_patterns(name, [r"쏘팔메토", r"쏘팔메토열매추출물", r"saw palmetto", r"로르산", r"옥타코사놀"]) else 1 if ingredient_matches_patterns(name, [r"녹용", r"마카", r"l-아르지닌", r"아르기닌"]) else 2, primary_ingredients.index(name)))

    ratio = top_score / second_score if second_score > 0 else 9.99
    confidence = 0.45
    if ratio >= 1.8:
        confidence += 0.28
    elif ratio >= 1.5:
        confidence += 0.2
    elif ratio >= 1.2:
        confidence += 0.1
    else:
        confidence -= 0.05

    if len(primary_ingredients) >= 2:
        confidence += 0.08
    elif len(primary_ingredients) == 1:
        confidence += 0.04

    if total_weight > 0:
        unknown_ratio = unknown_weight / total_weight
        confidence -= min(0.18, unknown_ratio * 0.35)

    notes = []
    if ratio < 1.2 and len(sorted_categories) > 1:
        notes.append("대표 카테고리와 차순위 카테고리 점수가 유사")
    if unknown_weight > 0 and total_weight > 0 and (unknown_weight / total_weight) >= 0.3:
        notes.append("기타/저신뢰 원료 비중이 높음")
    if only_support_style:
        notes.append("비타민/미네랄 중심 제품으로 판단")
    if name_boosts:
        notes.append(f"제품명 기반 가중치 반영: {dict(name_boosts)}")
    if product_name_is_male and has_male_zinc_only and not has_male_core:
        confidence = min(confidence, 0.72)
        notes.append("전립선/남성 제품명과 아연 기반으로 남성 건강 보정")

    confidence = max(0.1, min(0.99, round(confidence, 4)))
    category_scores_json = {key: round(float(value), 6) for key, value in sorted_categories}

    return {
        "product_id": product_id,
        "report_no": report_no,
        "product_name": product_name,
        "product_main_category": product_main_category,
        "product_sub_categories_json": to_json(sub_categories),
        "primary_ingredients_json": to_json(primary_ingredients),
        "secondary_ingredients_json": to_json(secondary_ingredients),
        "support_ingredients_json": to_json(support_ingredients),
        "category_scores_json": to_json(category_scores_json),
        "ingredient_scores_json": to_json(ingredient_scores),
        "confidence": confidence,
        "notes": " | ".join(notes),
    }


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    runtime = resolve_runtime_paths()
    output_dir = runtime["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_dir / "product_function_profile.csv"
    output_jsonl_path = output_dir / "product_function_profile.jsonl"
    failed_csv_path = output_dir / "failed_product_profile.csv"
    summary_json_path = output_dir / "product_profile_summary.json"

    print(f"[INFO] vector_csv_path={runtime['vector_csv_path']}")
    print(f"[INFO] category_map_csv_path={runtime['category_map_csv_path']}")
    print(f"[INFO] sqlite_path={runtime['sqlite_path']}")

    vector_df, category_df = load_inputs(runtime["vector_csv_path"], runtime["category_map_csv_path"])
    category_df = category_df.rename(
        columns={
            "functional_ingredient_name": "matched_standard_name",
            "confidence": "map_confidence",
        }
    )
    merged_df = vector_df.merge(category_df, on="matched_standard_name", how="left")
    merged_df["category_main"] = merged_df["category_main"].fillna("기타")
    merged_df["category_sub"] = merged_df["category_sub"].fillna("")
    merged_df["claim_text"] = merged_df["claim_text"].fillna("")
    merged_df["map_confidence"] = pd.to_numeric(merged_df["map_confidence"], errors="coerce").fillna(0.25)

    rows = []
    failed_rows = []
    grouped = merged_df.groupby("product_id", sort=True)
    for idx, (product_id, group) in enumerate(grouped, start=1):
        try:
            row = compute_profile_for_product(product_id, group.copy())
            rows.append(row)
            if not row["product_main_category"] or float(row["confidence"]) < 0.45:
                failed_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failed_rows.append(
                {
                    "product_id": product_id,
                    "report_no": "",
                    "product_name": str(group.iloc[0]["product_name"] or "") if not group.empty else "",
                    "product_main_category": "",
                    "product_sub_categories_json": "[]",
                    "primary_ingredients_json": "[]",
                    "secondary_ingredients_json": "[]",
                    "support_ingredients_json": "[]",
                    "category_scores_json": "{}",
                    "ingredient_scores_json": "[]",
                    "confidence": 0.0,
                    "notes": f"error: {exc}",
                }
            )
        if idx % 1000 == 0:
            print(f"[PROGRESS] profiled={idx}/{len(grouped)}")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        by=["product_main_category", "product_name", "product_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    failed_df = pd.DataFrame(failed_rows).reset_index(drop=True)

    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    save_jsonl(output_jsonl_path, result_df.to_dict(orient="records"))
    failed_df.to_csv(failed_csv_path, index=False, encoding="utf-8-sig")

    with sqlite3.connect(runtime["sqlite_path"]) as conn:
        backup_name = backup_table_if_exists(conn, TABLE_NAME)
        result_df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        if backup_name:
            print(f"[INFO] backup_table={backup_name}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "row_count": int(len(result_df)),
        "failed_row_count": int(len(failed_df)),
        "main_category_counts": dict(Counter(result_df["product_main_category"].astype(str))),
        "empty_main_category_count": int((result_df["product_main_category"].fillna("").astype(str).str.strip() == "").sum()),
        "input_vector_csv": str(runtime["vector_csv_path"]),
        "input_category_map_csv": str(runtime["category_map_csv_path"]),
        "sqlite_path": str(runtime["sqlite_path"]),
        "output_csv": str(output_csv_path),
        "output_jsonl": str(output_jsonl_path),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAMPLE] product_function_profile top 10")
    print(result_df.head(10).to_string(index=False))
    if not failed_df.empty:
        print("[SAMPLE] failed_product_profile top 10")
        print(failed_df.head(10).to_string(index=False))

    print(f"[DONE] product_function_profile rows={len(result_df)}")
    print(f"[DONE] failed_product_profile rows={len(failed_df)}")
    print(f"[DONE] csv={output_csv_path}")
    print(f"[DONE] jsonl={output_jsonl_path}")
    print(f"[DONE] summary={summary_json_path}")
    print(f"[DONE] sqlite_table={TABLE_NAME}")


if __name__ == "__main__":
    main()
