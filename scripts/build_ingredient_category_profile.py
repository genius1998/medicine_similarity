from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.enhance_similarity_with_explanation import (
    PRODUCT_CATEGORY_LABELS,
    ingredient_matches_patterns,
    is_semantic_excipient_name,
    normalize_text,
)


DEFAULT_INPUT = ROOT_DIR / "output" / "functional_category_map.csv"
DEFAULT_OUTPUT = ROOT_DIR / "output" / "ingredient_category_profile.csv"
DEFAULT_SAMPLE_OUTPUT = ROOT_DIR / "output" / "semantic_jaccard_v2" / "ingredient_category_profile_sample.csv"
PROFILE_VERSION = "ingredient_category_profile_rule_v1"

CATEGORY_ORDER = [
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
]

CATEGORY_PATTERNS = {
    "간 건강": [r"간 건강", r"간 기능", r"liver", r"밀크씨슬", r"실리마린", r"헛개"],
    "장 건강": [r"장 건강", r"배변", r"프로바이오틱스", r"유산균", r"prebiotic", r"probiotic", r"프락토올리고당", r"이눌린"],
    "면역": [r"면역"],
    "피부 건강": [r"피부", r"콜라겐", r"히알루론산", r"세라마이드", r"엘라스틴"],
    "관절/연골": [r"관절", r"연골", r"엠에스엠", r"\bmsm\b", r"보스웰리아", r"글루코사민", r"콘드로이친", r"뮤코다당", r"우슬", r"상어연골"],
    "뼈 건강": [r"뼈", r"골다공증", r"칼슘", r"비타민 d", r"비타민 k"],
    "혈행": [r"혈행", r"혈액.?흐름"],
    "혈중지질": [r"혈중지질", r"혈중 지질", r"중성지질", r"콜레스테롤", r"식물스테롤", r"폴리코사놀", r"홍국"],
    "눈 건강": [r"눈 건강", r"건조한 눈", r"루테인", r"지아잔틴", r"망막"],
    "기억력": [r"기억력", r"은행잎", r"\bdha\b"],
    "인지력": [r"인지기능", r"인지력", r"포스파티딜세린", r"phosphatidylserine"],
    "혈당": [r"혈당", r"바나바", r"난소화성말토덱스트린"],
    "혈압": [r"혈압", r"코엔자임", r"coq10"],
    "체지방": [r"체지방", r"가르시니아", r"공액리놀레산", r"\bcla\b", r"카테킨"],
    "피로개선": [r"피로", r"홍삼", r"옥타코사놀", r"매실"],
    "항산화": [r"항산화", r"코엔자임", r"비타민 c", r"셀렌", r"셀레늄", r"폴리페놀"],
    "남성 건강": [r"전립선", r"남성", r"쏘팔메토", r"saw palmetto"],
    "여성 건강": [r"갱년기", r"여성", r"회화나무", r"석류"],
    "수면/긴장완화": [r"수면", r"긴장", r"스트레스", r"테아닌", r"\bgaba\b"],
    "운동/근력": [r"근력", r"운동", r"단백질", r"\bhmb\b", r"크레아틴"],
    "구강 건강": [r"구강", r"충치", r"잇몸", r"프로폴리스"],
    "영양보충": [r"비타민", r"미네랄", r"아연", r"칼슘", r"마그네슘", r"철", r"엽산", r"셀렌", r"셀레늄"],
}

NUTRIENT_PATTERNS = [
    r"비타민",
    r"칼슘",
    r"마그네슘",
    r"아연",
    r"철",
    r"엽산",
    r"셀렌",
    r"셀레늄",
    r"크롬",
    r"망간",
    r"구리",
    r"요오드",
    r"나이아신",
    r"판토텐산",
    r"비오틴",
]

FORMULATION_PATTERNS = [
    r"캡슐",
    r"피막",
    r"코팅",
    r"부형",
    r"착색",
    r"감미",
    r"향료",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원료 카테고리 프로필(rule draft) 생성")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="functional_category_map.csv 경로")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="ingredient_category_profile.csv 출력 경로")
    parser.add_argument("--sample-output", default=str(DEFAULT_SAMPLE_OUTPUT), help="샘플 검증 CSV 출력 경로")
    parser.add_argument("--sample-size", type=int, default=80)
    return parser.parse_args()


def categories_from_text(text: str) -> list[str]:
    matches = []
    for category in CATEGORY_ORDER:
        patterns = CATEGORY_PATTERNS.get(category, [])
        if ingredient_matches_patterns(text, patterns):
            matches.append(category)
    return matches


def normalize_category(value: object) -> str:
    category = str(value or "").strip()
    return category if category in PRODUCT_CATEGORY_LABELS else "기타"


def source_priority(source: object) -> int:
    text = str(source or "").lower()
    if "item_class" in text:
        return 50
    if "curated" in text:
        return 40
    if "official" in text:
        return 30
    if "llm_rag" in text:
        return 20
    return 10


def deduplicate_category_map(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["functional_ingredient_name"] = work["functional_ingredient_name"].fillna("").astype(str).str.strip()
    work = work[work["functional_ingredient_name"] != ""].copy()
    work["_source_priority"] = work["source"].map(source_priority) if "source" in work.columns else 0
    work["_confidence"] = pd.to_numeric(work.get("confidence", 0.0), errors="coerce").fillna(0.0)
    work["_claim_len"] = work.get("claim_text", "").fillna("").astype(str).str.len()
    work = work.sort_values(
        ["functional_ingredient_name", "_source_priority", "_confidence", "_claim_len"],
        ascending=[True, False, False, False],
    )
    return work.drop_duplicates(subset=["functional_ingredient_name"], keep="first").drop(
        columns=["_source_priority", "_confidence", "_claim_len"],
        errors="ignore",
    )


def preferred_main_category(name: str, text: str, current_main: str, mentioned_categories: list[str]) -> tuple[str, list[str], str]:
    lowered = normalize_text(f"{name} {text}")
    if re.search(r"epa|dha|오메가", lowered, flags=re.IGNORECASE):
        main = current_main if current_main in {"혈행", "혈중지질"} else "혈중지질"
        subs = ["혈중지질", "혈행", "눈 건강", "기억력"]
        return main, subs, "omega_multi_function_rule"
    if re.search(r"홍삼|인삼|진세노사이드", lowered, flags=re.IGNORECASE):
        return "면역", ["피로개선", "혈행", "기억력", "항산화"], "red_ginseng_multi_function_rule"
    if re.search(r"프로바이오틱스|유산균|lactobacillus|lactiplantibacillus|bifidobacterium|rhamnosus|breve|plantarum", lowered, flags=re.IGNORECASE):
        if "인지력" in mentioned_categories:
            return "인지력", ["장 건강"], "probiotic_claim_overrides_microbe_family"
        if "혈중지질" in mentioned_categories:
            return "혈중지질", ["장 건강"], "probiotic_claim_overrides_microbe_family"
        if "피부 건강" in mentioned_categories:
            return "피부 건강", ["면역", "장 건강"], "probiotic_skin_immune_claim_rule"
        if "면역" in mentioned_categories:
            return "면역", ["장 건강"], "probiotic_immune_claim_rule"
        for claim_category in mentioned_categories:
            if claim_category not in {"장 건강", "영양보충", "기타"}:
                return claim_category, ["장 건강"], "probiotic_claim_overrides_microbe_family"
        return "장 건강", [], "probiotic_family_rule"
    if re.search(r"포스파티딜세린|phosphatidylserine", lowered, flags=re.IGNORECASE):
        return "인지력", ["기억력"], "phosphatidylserine_rule"
    if re.search(r"은행잎|ginkgo", lowered, flags=re.IGNORECASE):
        return "기억력", ["혈행", "인지력"], "ginkgo_rule"
    if re.search(r"루테인|지아잔틴|아스타잔틴", lowered, flags=re.IGNORECASE):
        return "눈 건강", ["항산화"], "eye_function_rule"
    if re.search(r"보스웰리아|엠에스엠|\bmsm\b|글루코사민|콘드로이친|뮤코다당|초록입홍합|우슬|상어연골", lowered, flags=re.IGNORECASE):
        return "관절/연골", [], "joint_cartilage_rule"
    if re.search(r"밀크씨슬|실리마린|헛개", lowered, flags=re.IGNORECASE):
        return "간 건강", [], "liver_rule"
    if re.search(r"가르시니아|공액리놀레산|\bcla\b|카테킨|녹차", lowered, flags=re.IGNORECASE):
        return "체지방", ["항산화"], "body_fat_rule"
    if re.search(r"바나바|난소화성말토덱스트린|구아바잎", lowered, flags=re.IGNORECASE):
        return "혈당", ["장 건강"], "blood_glucose_rule"
    if re.search(r"폴리코사놀|식물스테롤|홍국", lowered, flags=re.IGNORECASE):
        return "혈중지질", ["혈행"], "blood_lipid_rule"
    if re.search(r"코엔자임|coq10", lowered, flags=re.IGNORECASE):
        return "항산화", ["혈압"], "coq10_rule"
    if re.search(r"콜라겐|히알루론산|세라마이드|엘라스틴", lowered, flags=re.IGNORECASE):
        return "피부 건강", ["항산화"], "skin_rule"
    if re.search(r"쏘팔메토|saw palmetto|로르산", lowered, flags=re.IGNORECASE):
        return "남성 건강", [], "saw_palmetto_rule"
    if re.search(r"테아닌|\bgaba\b", lowered, flags=re.IGNORECASE):
        return "수면/긴장완화", ["피로개선"], "relaxation_rule"
    if current_main != "기타":
        return current_main, mentioned_categories, "existing_main_category_fallback"
    if mentioned_categories:
        return mentioned_categories[0], mentioned_categories, "claim_text_category_fallback"
    return "기타", [], "uncertain_fallback"


def classify_row(row: dict) -> dict:
    name = str(row.get("functional_ingredient_name", "") or "").strip()
    current_main = normalize_category(row.get("category_main", ""))
    claim_text = str(row.get("claim_text", "") or "")
    function_text = str(row.get("function_text", "") or "")
    category_sub = str(row.get("category_sub", "") or "")
    text = f"{name} {claim_text} {function_text} {current_main}"
    base_confidence = float(row.get("confidence", 0.0) or 0.0)

    is_excipient = is_semantic_excipient_name(name)
    if is_excipient:
        return {
            "functional_ingredient_name": name,
            "ingredient_main_category": "기타",
            "ingredient_sub_function_categories_json": "[]",
            "ingredient_type": "excipient",
            "vector_include": False,
            "is_excipient": True,
            "confidence": max(base_confidence, 0.9),
            "reason": "부형제/제형보조 패턴에 해당하여 semantic vector에서 제외",
            "source": PROFILE_VERSION,
            "legacy_category_main": current_main,
            "legacy_category_sub": category_sub,
        }

    formulation_aid = ingredient_matches_patterns(name, FORMULATION_PATTERNS)
    if formulation_aid:
        return {
            "functional_ingredient_name": name,
            "ingredient_main_category": "기타",
            "ingredient_sub_function_categories_json": "[]",
            "ingredient_type": "formulation_aid",
            "vector_include": False,
            "is_excipient": True,
            "confidence": max(base_confidence, 0.8),
            "reason": "제형보조/첨가물 패턴에 해당하여 semantic vector에서 제외",
            "source": PROFILE_VERSION,
            "legacy_category_main": current_main,
            "legacy_category_sub": category_sub,
        }

    mentioned_categories = categories_from_text(text)
    main, sub_categories, reason = preferred_main_category(name, text, current_main, mentioned_categories)
    sub_categories = [category for category in dict.fromkeys(sub_categories + mentioned_categories) if category != main and category in PRODUCT_CATEGORY_LABELS]
    nutrient = ingredient_matches_patterns(name, NUTRIENT_PATTERNS)
    ingredient_type = "nutrient" if nutrient else "functional"
    if nutrient and main == "기타":
        main = "영양보충"
        reason = "generic_nutrient_rule"

    confidence = min(0.99, max(base_confidence, 0.65 if reason.endswith("_fallback") else 0.82))
    if reason == "uncertain_fallback":
        confidence = min(confidence, 0.55)

    return {
        "functional_ingredient_name": name,
        "ingredient_main_category": main,
        "ingredient_sub_function_categories_json": json.dumps(sub_categories, ensure_ascii=False),
        "ingredient_type": ingredient_type,
        "vector_include": True,
        "is_excipient": False,
        "confidence": round(float(confidence), 3),
        "reason": reason,
        "source": PROFILE_VERSION,
        "legacy_category_main": current_main,
        "legacy_category_sub": category_sub,
    }


def build_profile(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    if "functional_ingredient_name" not in df.columns:
        raise ValueError(f"functional_ingredient_name 컬럼이 없습니다: {input_path}")
    deduped = deduplicate_category_map(df)
    rows = [classify_row(row) for row in deduped.to_dict(orient="records")]
    return pd.DataFrame(rows)


def build_sample(profile_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    priority_patterns = [
        r"EPA|DHA|오메가",
        r"홍삼|인삼",
        r"히드록시프로필메틸셀룰로오스|HPMC|결정셀룰로|스테아린산마그네슘",
        r"rhamnosus|breve|plantarum|프로바이오틱스|유산균",
        r"보스웰리아|엠에스엠|MSM|뮤코다당|우슬|상어연골",
        r"비타민|칼슘|마그네슘|아연",
        r"루테인|지아잔틴",
        r"밀크씨슬|실리마린",
        r"쏘팔메토",
        r"테아닌|GABA",
    ]
    sample_parts = []
    name_series = profile_df["functional_ingredient_name"].fillna("").astype(str)
    for pattern in priority_patterns:
        sample_parts.append(profile_df[name_series.str.contains(pattern, regex=True, case=False)].head(12))
    category_sample = profile_df.groupby("ingredient_main_category", group_keys=False).head(3)
    sample = pd.concat(sample_parts + [category_sample], ignore_index=True)
    sample = sample.drop_duplicates(subset=["functional_ingredient_name"], keep="first")
    return sample.head(max(1, int(sample_size))).copy()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    sample_output_path = Path(args.sample_output)
    profile_df = build_profile(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    sample_df = build_sample(profile_df, args.sample_size)
    sample_df.to_csv(sample_output_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] input_rows={len(pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False))}")
    print(f"[DONE] profile_rows={len(profile_df)}")
    print(f"[DONE] output={output_path}")
    print(f"[DONE] sample_rows={len(sample_df)} sample_output={sample_output_path}")
    print(profile_df["ingredient_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
