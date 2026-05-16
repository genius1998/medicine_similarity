from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import get_config_value, load_config


TABLE_NAME = "functional_category_map"


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^0-9a-z가-힣]+", "", text)


def simplify_ingredient_name(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"제\d{4}-\d+호", " ", text)
    text = re.sub(r"기능성원료인정제\d{4}-\d+호", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_existing_path(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {[str(path) for path in candidates]}")


def resolve_output_dir(config: dict) -> Path:
    configured = Path(str(get_config_value(config, "output_dir", ROOT_DIR / "output")))
    if configured.drive or str(configured).startswith("/"):
        if configured.exists():
            return configured
    return ROOT_DIR / "output"


def resolve_runtime_paths() -> dict[str, Path]:
    config = load_config()
    output_dir = resolve_output_dir(config)
    vector_csv_candidates = [
        output_dir / "c003_product_functional_vectors_final_rebuilt.csv",
        Path(r"D:\ec2_cache_snapshot\c003_product_functional_vectors_final_rebuilt.csv"),
        Path(r"C:\Users\com\Downloads\c003_product_vector_output_final_rebuilt\c003_product_functional_vectors_final_rebuilt.csv"),
    ]
    rag_csv_candidates = [
        output_dir / "functional_ingredient_rag_documents_merged_item_class_boosted.csv",
        ROOT_DIR / "deploy_ec2" / "data" / "functional_ingredient_rag_documents_merged_item_class_boosted.csv",
    ]
    synonym_csv_candidates = [
        output_dir / "functional_ingredient_synonym_dictionary_merged_item_class_boosted.csv",
        ROOT_DIR / "deploy_ec2" / "data" / "functional_ingredient_synonym_dictionary_merged_item_class_boosted.csv",
    ]
    sqlite_candidates = [
        Path(str(get_config_value(config, "sqlite_path", ""))),
        Path(r"D:\ec2_cache_snapshot\ingredient_match_cache_rebuilt_item_class_i0050_final.sqlite"),
    ]
    sqlite_path = first_existing_path([path for path in sqlite_candidates if str(path)])
    return {
        "output_dir": output_dir,
        "vector_csv_path": first_existing_path(vector_csv_candidates),
        "rag_csv_path": first_existing_path(rag_csv_candidates),
        "synonym_csv_path": first_existing_path(synonym_csv_candidates),
        "sqlite_path": sqlite_path,
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


SUPPORT_PATTERNS = [
    r"비타민\s*[abcdek]?\d*",
    r"엽산",
    r"철",
    r"칼슘",
    r"마그네슘",
    r"아연",
    r"셀레늄",
    r"셀렌",
    r"망간",
    r"구리",
    r"요오드",
    r"크롬",
    r"몰리브덴",
    r"칼륨",
]


CATEGORY_RULES = [
    {
        "category_main": "눈 건강",
        "category_sub": "루테인/지아잔틴 계열",
        "name_patterns": [r"루테인", r"지아잔틴", r"마리골드", r"헤마토코쿠스", r"아스타잔틴", r"베타카로틴"],
        "claim_patterns": [r"눈\s*건강", r"황반", r"시각", r"어두운 곳", r"눈의 피로"],
    },
    {
        "category_main": "장 건강",
        "category_sub": "프로바이오틱스/식이섬유 계열",
        "name_patterns": [r"프로바이오틱스", r"유산균", r"락토", r"비피도", r"프락토올리고당", r"갈락토올리고당", r"이눌린", r"치커리", r"식이섬유", r"차전자피"],
        "claim_patterns": [r"장\s*건강", r"배변활동", r"유익균", r"유해균", r"원활한 배변", r"배변"],
    },
    {
        "category_main": "면역",
        "category_sub": "홍삼/면역 계열",
        "name_patterns": [r"홍삼", r"인삼", r"진세노사이드", r"알로에겔", r"베타글루칸", r"프로폴리스", r"클로렐라"],
        "claim_patterns": [r"면역", r"면역기능", r"면역력"],
    },
    {
        "category_main": "피로개선",
        "category_sub": "홍삼/에너지 계열",
        "name_patterns": [r"홍삼", r"인삼", r"옥타코사놀", r"발효굴", r"비타민\s*b", r"판토텐산", r"나이아신"],
        "claim_patterns": [r"피로", r"에너지\s*생성", r"에너지\s*대사", r"활력"],
    },
    {
        "category_main": "혈행",
        "category_sub": "오메가3/혈행 계열",
        "name_patterns": [r"epa", r"dha", r"오메가\s*3", r"은행잎", r"나토", r"홍삼"],
        "claim_patterns": [r"혈행", r"혈액순환"],
    },
    {
        "category_main": "혈중지질",
        "category_sub": "오메가3/콜레스테롤 계열",
        "name_patterns": [r"epa", r"dha", r"오메가\s*3", r"홍국", r"폴리코사놀", r"베타글루칸", r"보리 베타글루칸"],
        "claim_patterns": [r"혈중.*지질", r"중성지방", r"콜레스테롤", r"혈중 콜레스테롤"],
    },
    {
        "category_main": "뼈 건강",
        "category_sub": "칼슘/비타민D 계열",
        "name_patterns": [r"칼슘", r"비타민\s*d", r"비타민\s*k", r"마그네슘", r"mbp", r"해조칼슘"],
        "claim_patterns": [r"뼈", r"치아", r"골다공증", r"골밀도"],
    },
    {
        "category_main": "관절/연골",
        "category_sub": "MSM/NAG/보스웰리아 계열",
        "name_patterns": [r"\bmsm\b", r"엠에스엠", r"nag", r"엔에이지", r"보스웰리아", r"초록입홍합", r"\bnem\b", r"뮤코다당", r"글루코사민", r"콘드로이친"],
        "claim_patterns": [r"관절", r"연골"],
    },
    {
        "category_main": "간 건강",
        "category_sub": "밀크씨슬/실리마린 계열",
        "name_patterns": [r"밀크씨슬", r"실리마린", r"헛개", r"아티초크"],
        "claim_patterns": [r"간\s*건강", r"간세포", r"간 기능"],
    },
    {
        "category_main": "피부 건강",
        "category_sub": "콜라겐/히알루론산 계열",
        "name_patterns": [r"콜라겐", r"히알루론산", r"세라마이드", r"엘라스틴", r"레티놀", r"알로에겔", r"석류", r"엘라그산"],
        "claim_patterns": [r"피부", r"보습", r"자외선", r"피부손상", r"탄력"],
    },
    {
        "category_main": "항산화",
        "category_sub": "비타민C/E/코엔자임Q10 계열",
        "name_patterns": [r"비타민\s*c", r"비타민\s*e", r"셀레늄", r"셀렌", r"코엔자임\s*q10", r"코엔자임q10", r"q10", r"폴리페놀", r"토마토추출물", r"리코펜"],
        "claim_patterns": [r"항산화", r"유해산소", r"산화"],
    },
    {
        "category_main": "기억력",
        "category_sub": "포스파티딜세린/은행잎 계열",
        "name_patterns": [r"포스파티딜세린", r"은행잎"],
        "claim_patterns": [r"기억력"],
    },
    {
        "category_main": "인지력",
        "category_sub": "포스파티딜세린 계열",
        "name_patterns": [r"포스파티딜세린"],
        "claim_patterns": [r"인지", r"인지능력"],
    },
    {
        "category_main": "체지방",
        "category_sub": "가르시니아/CLA/녹차 계열",
        "name_patterns": [r"가르시니아", r"\bcla\b", r"공액리놀레산", r"녹차추출물", r"카테킨", r"l-카르니틴", r"l카르니틴"],
        "claim_patterns": [r"체지방", r"체지방 감소", r"지방대사"],
    },
    {
        "category_main": "혈당",
        "category_sub": "바나바/난소화성말토덱스트린 계열",
        "name_patterns": [r"바나바", r"난소화성말토덱스트린", r"크롬", r"여주", r"달맞이", r"guava", r"구아바"],
        "claim_patterns": [r"혈당", r"식후 혈당"],
    },
    {
        "category_main": "혈압",
        "category_sub": "코엔자임Q10/혈압 계열",
        "name_patterns": [r"코엔자임\s*q10", r"코엔자임q10", r"q10", r"카제인가수분해물", r"올리브잎"],
        "claim_patterns": [r"혈압"],
    },
    {
        "category_main": "여성 건강",
        "category_sub": "감마리놀렌산/이소플라본 계열",
        "name_patterns": [r"감마리놀렌산", r"이소플라본", r"백수오", r"크랜베리", r"석류", r"회화나무열매"],
        "claim_patterns": [r"여성", r"갱년기", r"월경", r"요로", r"질 건강"],
    },
    {
        "category_main": "남성 건강",
        "category_sub": "전립선/남성 계열",
        "name_patterns": [r"쏘팔메토", r"전립선", r"옥타코사놀", r"녹용"],
        "claim_patterns": [r"전립선", r"남성"],
    },
    {
        "category_main": "구강 건강",
        "category_sub": "자일리톨/구강 계열",
        "name_patterns": [r"자일리톨", r"프로폴리스"],
        "claim_patterns": [r"구강", r"치아우식", r"충치", r"잇몸"],
    },
    {
        "category_main": "수면/긴장완화",
        "category_sub": "테아닌/GABA 계열",
        "name_patterns": [r"테아닌", r"\bgaba\b", r"감태", r"락티움", r"멜라토닌"],
        "claim_patterns": [r"스트레스", r"긴장", r"수면", r"이완"],
    },
    {
        "category_main": "운동/근력",
        "category_sub": "단백질/HMB/L-카르니틴 계열",
        "name_patterns": [r"\bhmb\b", r"크레아틴", r"l-카르니틴", r"l카르니틴", r"단백질", r"아미노산", r"bcaa"],
        "claim_patterns": [r"근력", r"운동", r"근육", r"지구력"],
    },
    {
        "category_main": "영양보충",
        "category_sub": "비타민/미네랄 계열",
        "name_patterns": SUPPORT_PATTERNS,
        "claim_patterns": [r"영양", r"필요", r"에너지 대사", r"정상적인", r"체내 에너지 생성"],
    },
]


CLAIM_TO_CATEGORY_HINTS = {
    "눈 건강": [r"눈\s*건강", r"황반색소밀도", r"시각적응"],
    "장 건강": [r"장\s*건강", r"배변활동", r"유산균 증식", r"유해균 억제"],
    "면역": [r"면역기능", r"면역"],
    "피로개선": [r"피로개선", r"피로"],
    "혈행": [r"혈행", r"혈액순환"],
    "혈중지질": [r"혈중 콜레스테롤", r"중성지방", r"혈중 지질"],
    "뼈 건강": [r"뼈", r"치아", r"골다공증", r"골밀도"],
    "관절/연골": [r"관절", r"연골"],
    "간 건강": [r"간 건강", r"간 기능"],
    "피부 건강": [r"피부", r"보습", r"자외선"],
    "항산화": [r"유해산소", r"항산화"],
    "기억력": [r"기억력"],
    "인지력": [r"인지능력", r"인지"],
    "체지방": [r"체지방"],
    "혈당": [r"혈당"],
    "혈압": [r"혈압"],
    "여성 건강": [r"갱년기", r"여성", r"요로"],
    "남성 건강": [r"전립선", r"남성"],
    "구강 건강": [r"구강", r"치아우식", r"충치"],
    "수면/긴장완화": [r"긴장", r"스트레스", r"수면"],
    "운동/근력": [r"근력", r"근육", r"운동"],
    "영양보충": [r"필요", r"에너지 생성", r"정상적인"],
}


def load_vector_ingredients(vector_csv_path: Path) -> list[str]:
    df = pd.read_csv(vector_csv_path, encoding="utf-8-sig", usecols=["matched_standard_name"])
    names = (
        df["matched_standard_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    names = names[names != ""]
    return sorted(names.drop_duplicates().tolist())


def aggregate_text_by_standard_name(df: pd.DataFrame) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for standard_name, group in df.groupby("standard_name", dropna=False):
        name = str(standard_name or "").strip()
        if not name:
            continue
        functions = []
        sources = []
        notes = []
        for row in group.itertuples(index=False):
            function_text = str(getattr(row, "function_text", "") or "").strip()
            caution_text = str(getattr(row, "caution_text", "") or "").strip()
            cleaned_function_text = "\n".join(
                [line.strip() for line in function_text.splitlines() if line.strip() and line.strip().lower() != "nan"]
            ).strip()
            cleaned_caution_text = "\n".join(
                [line.strip() for line in caution_text.splitlines() if line.strip() and line.strip().lower() != "nan"]
            ).strip()
            if cleaned_function_text:
                functions.append(cleaned_function_text)
            if cleaned_caution_text:
                notes.append(cleaned_caution_text)
            source_value = ""
            if hasattr(row, "sources_joined"):
                source_value = str(getattr(row, "sources_joined") or "").strip()
            elif hasattr(row, "source"):
                source_value = str(getattr(row, "source") or "").strip()
            if source_value and source_value.lower() != "nan":
                sources.extend([part.strip() for part in source_value.split(",") if part.strip()])
        grouped[name] = {
            "function_text": "\n".join(dict.fromkeys(functions)),
            "source": ", ".join(dict.fromkeys(sources)),
            "notes": "\n".join(dict.fromkeys(notes)),
        }
    return grouped


def build_lookup_maps(rag_csv_path: Path, synonym_csv_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    rag_df = pd.read_csv(
        rag_csv_path,
        encoding="utf-8-sig",
        usecols=["standard_name", "sources_joined", "function_text", "caution_text"],
        low_memory=False,
    )
    synonym_df = pd.read_csv(
        synonym_csv_path,
        encoding="utf-8-sig",
        usecols=["standard_name", "source", "function_text", "caution_text"],
        low_memory=False,
    )
    rag_map = aggregate_text_by_standard_name(rag_df)
    synonym_map = aggregate_text_by_standard_name(synonym_df)

    combined: dict[str, dict] = {}
    normalized_lookup: dict[str, dict] = {}
    for source_name in sorted(set(rag_map) | set(synonym_map)):
        rag_entry = rag_map.get(source_name, {})
        synonym_entry = synonym_map.get(source_name, {})
        function_texts = [text for text in [rag_entry.get("function_text", ""), synonym_entry.get("function_text", "")] if text]
        note_texts = [text for text in [rag_entry.get("notes", ""), synonym_entry.get("notes", "")] if text]
        source_tokens = []
        for source_text in [rag_entry.get("source", ""), synonym_entry.get("source", "")]:
            if source_text:
                source_tokens.extend([part.strip() for part in source_text.split(",") if part.strip()])
        entry = {
            "function_text": "\n".join(dict.fromkeys(function_texts)),
            "source": ", ".join(dict.fromkeys(source_tokens)),
            "notes": "\n".join(dict.fromkeys(note_texts)),
        }
        combined[source_name] = entry
        for alt_name in {source_name, simplify_ingredient_name(source_name)}:
            normalized_key = normalize_text(alt_name)
            if normalized_key and normalized_key not in normalized_lookup:
                normalized_lookup[normalized_key] = entry
    return combined, normalized_lookup


def collect_rule_hits(name: str, claim_text: str) -> list[dict]:
    hits = []
    simplified = simplify_ingredient_name(name)
    for rule in CATEGORY_RULES:
        score = 0.0
        name_match = any(re.search(pattern, simplified, flags=re.IGNORECASE) for pattern in rule["name_patterns"])
        claim_match = any(re.search(pattern, claim_text, flags=re.IGNORECASE) for pattern in rule["claim_patterns"])
        if name_match:
            score += 1.4
        if claim_match:
            score += 1.1
        if score > 0:
            hits.append(
                {
                    "category_main": rule["category_main"],
                    "category_sub": rule["category_sub"],
                    "score": score,
                    "matched_by_name": name_match,
                    "matched_by_claim": claim_match,
                }
            )
    return hits


def infer_claim_from_category(category_main: str) -> str:
    mapping = {
        "눈 건강": "눈 건강 관련 기능성원료로 분류",
        "장 건강": "장 건강 관련 기능성원료로 분류",
        "면역": "면역 관련 기능성원료로 분류",
        "피로개선": "피로 개선 관련 기능성원료로 분류",
        "혈행": "혈행 관련 기능성원료로 분류",
        "혈중지질": "혈중지질 관련 기능성원료로 분류",
        "뼈 건강": "뼈 건강 관련 기능성원료로 분류",
        "관절/연골": "관절 및 연골 관련 기능성원료로 분류",
        "간 건강": "간 건강 관련 기능성원료로 분류",
        "피부 건강": "피부 건강 관련 기능성원료로 분류",
        "항산화": "항산화 관련 기능성원료로 분류",
        "기억력": "기억력 관련 기능성원료로 분류",
        "인지력": "인지력 관련 기능성원료로 분류",
        "체지방": "체지방 관련 기능성원료로 분류",
        "혈당": "혈당 관련 기능성원료로 분류",
        "혈압": "혈압 관련 기능성원료로 분류",
        "여성 건강": "여성 건강 관련 기능성원료로 분류",
        "남성 건강": "남성 건강 관련 기능성원료로 분류",
        "구강 건강": "구강 건강 관련 기능성원료로 분류",
        "수면/긴장완화": "수면 또는 긴장완화 관련 기능성원료로 분류",
        "운동/근력": "운동 또는 근력 관련 기능성원료로 분류",
        "영양보충": "영양보충 성격의 기능성원료로 분류",
        "기타": "",
    }
    return mapping.get(category_main, "")


def choose_source_text(source_text: str, has_claim: bool, has_name_rule: bool) -> str:
    text = str(source_text or "").strip()
    lowered = text.lower()
    if "item_class" in lowered:
        return "item_class"
    if "i0050" in lowered:
        return "I0050"
    if text:
        return "synonym_master" if has_claim else text
    if has_name_rule:
        return "rule_based"
    return "unknown"


def map_single_ingredient(name: str, exact_lookup: dict[str, dict], normalized_lookup: dict[str, dict]) -> dict:
    evidence = exact_lookup.get(name)
    if evidence is None:
        evidence = normalized_lookup.get(normalize_text(name))
    if evidence is None:
        evidence = normalized_lookup.get(normalize_text(simplify_ingredient_name(name)), {})

    claim_text = str(evidence.get("function_text", "") if evidence else "").strip()
    source_text = str(evidence.get("source", "") if evidence else "").strip()
    evidence_notes = str(evidence.get("notes", "") if evidence else "").strip()
    rule_hits = collect_rule_hits(name, claim_text)

    score_by_category: dict[str, float] = defaultdict(float)
    sub_by_category: dict[str, list[str]] = defaultdict(list)
    categories_all = []
    for hit in rule_hits:
        score_by_category[hit["category_main"]] += hit["score"]
        if hit["category_sub"] not in sub_by_category[hit["category_main"]]:
            sub_by_category[hit["category_main"]].append(hit["category_sub"])
        categories_all.append(hit["category_main"])

    # Additional claim-only hints if the main rules did not catch a phrase.
    for category_main, patterns in CLAIM_TO_CATEGORY_HINTS.items():
        if any(re.search(pattern, claim_text, flags=re.IGNORECASE) for pattern in patterns):
            score_by_category[category_main] += 0.5
            categories_all.append(category_main)

    if score_by_category:
        ranked = sorted(score_by_category.items(), key=lambda item: (-item[1], item[0]))
        category_main = ranked[0][0]
        category_sub = " | ".join(sub_by_category.get(category_main, []))
        other_categories = [name for name, _ in ranked]
    else:
        category_main = "기타"
        category_sub = ""
        other_categories = []

    has_claim = bool(claim_text)
    has_name_rule = bool(rule_hits)
    source = choose_source_text(source_text, has_claim=has_claim, has_name_rule=has_name_rule)

    if not claim_text and category_main != "기타":
        claim_text = infer_claim_from_category(category_main)

    confidence = 0.35
    notes = []
    if has_claim and category_main != "기타":
        confidence = 0.94 if source in {"item_class", "I0050"} else 0.88
    elif has_name_rule and category_main != "기타":
        confidence = 0.72
    elif category_main == "기타":
        confidence = 0.25
        notes.append("명확한 기능성 문구를 찾지 못해 기타로 분류")

    if len(other_categories) > 1:
        confidence -= 0.08
        notes.append(f"복수 기능 후보: {', '.join(other_categories[:5])}")
    if evidence_notes:
        notes.append(evidence_notes.split("\n")[0])

    confidence = max(0.05, min(0.99, round(confidence, 4)))
    return {
        "functional_ingredient_name": name,
        "category_main": category_main,
        "category_sub": category_sub,
        "claim_text": claim_text,
        "source": source,
        "confidence": confidence,
        "notes": " | ".join(dict.fromkeys([note for note in notes if note])),
        "categories_json": json.dumps(other_categories, ensure_ascii=False),
    }


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    runtime = resolve_runtime_paths()
    output_dir = runtime["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_dir / "functional_category_map.csv"
    output_jsonl_path = output_dir / "functional_category_map.jsonl"
    summary_json_path = output_dir / "category_mapping_summary.json"
    failed_csv_path = output_dir / "failed_category_mapping.csv"

    print(f"[INFO] vector_csv_path={runtime['vector_csv_path']}")
    print(f"[INFO] rag_csv_path={runtime['rag_csv_path']}")
    print(f"[INFO] synonym_csv_path={runtime['synonym_csv_path']}")
    print(f"[INFO] sqlite_path={runtime['sqlite_path']}")

    ingredient_names = load_vector_ingredients(runtime["vector_csv_path"])
    exact_lookup, normalized_lookup = build_lookup_maps(runtime["rag_csv_path"], runtime["synonym_csv_path"])

    rows = []
    failed_rows = []
    for idx, name in enumerate(ingredient_names, start=1):
        row = map_single_ingredient(name, exact_lookup, normalized_lookup)
        rows.append(row)
        if row["category_main"] == "기타" or float(row["confidence"]) < 0.55:
            failed_rows.append(row)
        if idx % 100 == 0:
            print(f"[PROGRESS] mapped={idx}/{len(ingredient_names)}")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        by=["category_main", "confidence", "functional_ingredient_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    failed_df = pd.DataFrame(failed_rows)
    failed_df = failed_df.sort_values(
        by=["confidence", "functional_ingredient_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

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
        "unique_categories": sorted(result_df["category_main"].dropna().astype(str).unique().tolist()),
        "category_counts": dict(Counter(result_df["category_main"].astype(str))),
        "low_confidence_count": int((pd.to_numeric(result_df["confidence"], errors="coerce").fillna(0) < 0.55).sum()),
        "input_vector_csv": str(runtime["vector_csv_path"]),
        "input_rag_csv": str(runtime["rag_csv_path"]),
        "input_synonym_csv": str(runtime["synonym_csv_path"]),
        "sqlite_path": str(runtime["sqlite_path"]),
        "output_csv": str(output_csv_path),
        "output_jsonl": str(output_jsonl_path),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAMPLE] functional_category_map top 10")
    print(result_df.head(10).to_string(index=False))
    if not failed_df.empty:
        print("[SAMPLE] failed_category_mapping top 10")
        print(failed_df.head(10).to_string(index=False))

    print(f"[DONE] functional_category_map rows={len(result_df)}")
    print(f"[DONE] failed_category_mapping rows={len(failed_df)}")
    print(f"[DONE] csv={output_csv_path}")
    print(f"[DONE] jsonl={output_jsonl_path}")
    print(f"[DONE] summary={summary_json_path}")
    print(f"[DONE] sqlite_table={TABLE_NAME}")


if __name__ == "__main__":
    main()
