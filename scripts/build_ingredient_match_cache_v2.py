from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_UNIQUE_CSV = ROOT_DIR / "output" / "ingredient_normalization" / "unique_normalized_ingredients.csv"
DEFAULT_CATEGORY_MAP = ROOT_DIR / "output" / "cache_db_excel_export" / "temp_csv" / "functional_category_map.csv"
DEFAULT_OLD_CACHE = ROOT_DIR / "output" / "cache_db_excel_export" / "temp_csv" / "ingredient_match_cache.csv"
DEFAULT_METADATA = Path(r"D:\db\deploy_ec2\data\functional_ingredient_metadata_item_class_final_boosted.pkl")
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "ingredient_match_v2"
DEFAULT_EMBEDDING_CACHE_DIR = ROOT_DIR / "tmp" / "ingredient_embedding_cache"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CACHE_VERSION = "ingredient_match_v2_20260601"
POSITIVE_RELATIONS = {"same_ingredient", "ingredient_group", "marker_compound", "nutrient_form"}
ALLOWED_DECISIONS = {
    "existing_match",
    "family_signal",
    "new_standard_candidate",
    "non_functional",
    "no_match",
    "uncertain_review",
}
ALLOWED_RELATIONS = {
    "same_ingredient",
    "ingredient_group",
    "marker_compound",
    "nutrient_form",
    "same_function_only",
    "family_signal",
    "excipient",
    "unrelated",
    "unknown",
}

EXPLICIT_NON_FUNCTIONAL_KEYS = {
    "이산화규소",
    "스테아린산마그네슘",
    "스테아린산마그네슘고시형",
    "히드록시프로필메틸셀룰로스",
    "결정셀룰로스",
    "결정셀룰로오스",
    "글리세린",
    "정제수",
    "젤라틴",
    "덱스트린",
    "이산화티타늄",
    "글리세린지방산에스테르",
    "카복시메틸셀룰로스칼슘",
    "카복시메틸셀룰로오스칼슘",
    "밀납",
    "카라기난",
    "구연산",
    "구연산무수",
    "dl사과산",
    "d소비톨액",
    "포도당",
    "분말결정포도당",
    "옥수수전분",
    "변성전분",
    "수크랄로스",
    "효소처리스테비아",
    "에리스리톨",
    "이소말트",
    "에틸바닐린",
    "자당지방산에스테르",
    "카카오색소",
    "콩기름대두유",
    "혼합제제",
    "기타가공품",
    "기타농산가공품",
    "당류가공품",
    "올리고당가공품",
}

CURATED_ALIAS_TO_STANDARD = {
    "니코틴산아미드": "나이아신",
    "니코틴산아미드고시형": "나이아신",
    "니코틴산": "나이아신",
    "판토텐산칼슘": "판토텐산",
    "판토텐산칼슘고시형": "판토텐산",
    "d토코페롤": "비타민 E",
    "dl토코페롤": "비타민 E",
    "dα토코페롤": "비타민 E",
    "dlα토코페롤": "비타민 E",
    "d알파토코페롤": "비타민 E",
    "dl알파토코페롤": "비타민 E",
    "토코페롤": "비타민 E",
    "비타민e혼합제제": "비타민 E",
    "비타민b1염산염": "비타민 B1",
    "비타민b1염산염고시형": "비타민 B1",
    "비타민b2고시형": "비타민 B2",
    "비타민b6염산염": "비타민 B6",
    "비타민b6염산염고시형": "비타민 B6",
    "비타민c고시형": "비타민 C",
    "비타민c함량100": "비타민 C",
    "비타민d3혼합제제": "비타민 D",
    "비타민d3": "비타민 D",
    "해조칼슘": "칼슘",
    "산화아연": "아연",
    "산화아연고시형": "아연",
    "황산망간": "망간",
    "황산망간고시형": "망간",
    "산화마그네슘": "마그네슘",
    "산화마그네슘고시형": "마그네슘",
    "엽산고시형": "엽산",
    "비오틴고시형": "비오틴",
    "프로바이오틱스고시형": "프로바이오틱스",
}

GENERIC_RAW_KEYS_REQUIRING_LLM = {
    "콜라겐",
    "콜라겐분말",
    "생선콜라겐",
    "생선콜라겐분말",
    "상어연골",
    "상어연골추출물분말",
    "쇠무릅",
    "쇠무릅분말",
    "우슬",
    "우슬분말",
}


def normalize_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", str(value or "").strip().lower())


def readable(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def strip_approval_numbers(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"\(\s*(?:기능성원료인정)?제?\d{4}-\d+호\s*\)", "", text)
    text = re.sub(r"(?:기능성원료인정)?제?\d{4}-\d+호", "", text)
    return readable(text)


def strip_parenthetical(value: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value or ""))
    text = re.sub(r"\[[^\]]*\]", "", text)
    return readable(text)


def simplified_key(value: Any) -> str:
    key = normalize_key(strip_parenthetical(strip_approval_numbers(str(value or ""))))
    replacements = [
        "고시형원료",
        "고시형",
        "기능성원료",
        "추출물분말",
        "추출분말",
        "농축액분말",
        "농축분말",
        "가루과립",
        "분말",
        "추출물",
        "추출액",
        "농축액",
        "농축물",
        "유지",
        "오일",
        "액",
    ]
    previous = None
    while previous != key:
        previous = key
        for suffix in replacements:
            if key.endswith(suffix) and len(key) > len(suffix) + 1:
                key = key[: -len(suffix)]
    return key


def is_specificity_conflict(raw_name: str, matched_name: str) -> bool:
    raw_key = normalize_key(raw_name)
    matched_key = normalize_key(matched_name)
    if raw_key in {"콜라겐", "콜라겐분말"} and "콜라겐" in matched_key:
        specific_tokens = ["홍어껍질", "ap", "저분자", "피쉬", "어류", "돈피", "효소분해", "펩타이드"]
        if any(token in matched_key for token in specific_tokens) and not any(token in raw_key for token in specific_tokens):
            return True
    if "상어연골" in raw_key and "참깨박" in matched_key:
        return True
    if "쇠무릅" in raw_key and "참깨박" in matched_key:
        return True
    return False


def is_non_functional_rule(raw_name: str, normalized_key: str) -> bool:
    key = normalized_key or normalize_key(raw_name)
    if key in GENERIC_RAW_KEYS_REQUIRING_LLM:
        return False
    if key in EXPLICIT_NON_FUNCTIONAL_KEYS:
        return True
    if re.search(r"(향분말|향료|착색료|색소|캡슐기제|피막제|유화제)$", key):
        return True
    if key.endswith("가공품") and not any(token in key for token in ["홍삼", "인삼", "과채"]):
        return True
    return False


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def truncate(value: Any, limit: int) -> str:
    text = readable(value)
    return text if len(text) <= limit else text[:limit].rstrip()


def category_row_payload(row: dict[str, str]) -> dict[str, Any]:
    return {
        "functional_ingredient_name": readable(row.get("functional_ingredient_name")),
        "category_main": readable(row.get("category_main")),
        "category_sub": readable(row.get("category_sub")),
        "claim_text": readable(row.get("claim_text")),
        "source": readable(row.get("source")),
        "confidence": row.get("confidence", ""),
    }


def build_category_indexes(category_rows: list[dict[str, str]]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_name: dict[str, dict] = {}
    by_key: dict[str, dict] = {}
    for row in category_rows:
        name = readable(row.get("functional_ingredient_name"))
        if not name:
            continue
        payload = category_row_payload(row)
        by_name.setdefault(name, payload)
        by_key.setdefault(normalize_key(name), payload)
        stripped = strip_parenthetical(strip_approval_numbers(name))
        if stripped:
            by_key.setdefault(normalize_key(stripped), payload)
    return by_name, by_key


def resolve_category_row(name: str, by_name: dict[str, dict], by_key: dict[str, dict]) -> dict | None:
    if not name:
        return None
    if name in by_name:
        return by_name[name]
    key = normalize_key(name)
    if key in by_key:
        return by_key[key]
    stripped_key = normalize_key(strip_parenthetical(strip_approval_numbers(name)))
    return by_key.get(stripped_key)


def build_alias_terms(aliases: list[str]) -> list[dict[str, str]]:
    terms: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias in aliases:
        alias_key = normalize_key(alias)
        alias_simple = simplified_key(alias)
        if not alias_key:
            continue
        pair = (alias_key, alias_simple)
        if pair in seen:
            continue
        seen.add(pair)
        terms.append({"key": alias_key, "simple": alias_simple})
    return terms


def build_standard_records(
    category_rows: list[dict[str, str]],
    metadata_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, dict], dict[str, dict]]:
    category_by_name, category_by_key = build_category_indexes(category_rows)
    metadata = pickle.load(metadata_path.open("rb"))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for item in metadata:
        standard_name = readable(item.get("standard_name"))
        category = resolve_category_row(standard_name, category_by_name, category_by_key)
        if not category:
            continue
        synonyms = [
            readable(part)
            for part in str(item.get("synonyms_joined", "") or "").split(",")
            if readable(part)
        ]
        aliases = set([standard_name, category["functional_ingredient_name"], *synonyms])
        for value in list(aliases):
            stripped = strip_parenthetical(strip_approval_numbers(value))
            if stripped:
                aliases.add(stripped)
            simple = simplified_key(value)
            if simple and len(simple) >= 2:
                aliases.add(simple)
        key = (standard_name, category["functional_ingredient_name"])
        if key in seen:
            continue
        seen.add(key)
        alias_list = sorted(alias for alias in aliases if alias)
        records.append(
            {
                "standard_name": standard_name,
                "canonical_name": category["functional_ingredient_name"],
                "category_main": category.get("category_main", ""),
                "category_sub": category.get("category_sub", ""),
                "claim_text": category.get("claim_text", ""),
                "source": category.get("source", ""),
                "synonyms_joined": str(item.get("synonyms_joined", "") or ""),
                "embedding_text": str(item.get("embedding_text", "") or item.get("search_text", "") or standard_name),
                "search_text": str(item.get("search_text", "") or ""),
                "aliases": alias_list,
                "alias_terms": build_alias_terms(alias_list),
            }
        )

    category_names_in_records = {record["canonical_name"] for record in records}
    for row in category_rows:
        name = readable(row.get("functional_ingredient_name"))
        if not name or name in category_names_in_records:
            continue
        aliases = {name, strip_parenthetical(strip_approval_numbers(name)), simplified_key(name)}
        alias_list = sorted(alias for alias in aliases if alias)
        records.append(
            {
                "standard_name": name,
                "canonical_name": name,
                "category_main": readable(row.get("category_main")),
                "category_sub": readable(row.get("category_sub")),
                "claim_text": readable(row.get("claim_text")),
                "source": readable(row.get("source")),
                "synonyms_joined": name,
                "embedding_text": f"공식명: {name}\n기능성: {readable(row.get('claim_text'))}",
                "search_text": "",
                "aliases": alias_list,
                "alias_terms": build_alias_terms(alias_list),
            }
        )

    alias_map: dict[str, set[str]] = defaultdict(set)
    exact_name_map: dict[str, dict] = {}
    for record in records:
        exact_name_map.setdefault(normalize_key(record["canonical_name"]), record)
        exact_name_map.setdefault(normalize_key(record["standard_name"]), record)
        for alias in record["aliases"]:
            key = normalize_key(alias)
            if len(key) >= 2:
                alias_map[key].add(record["canonical_name"])
    by_canonical = {record["canonical_name"]: record for record in records}
    return records, alias_map, exact_name_map, by_canonical


def resolve_alias_match(
    raw_name: str,
    normalized_key: str,
    alias_map: dict[str, set[str]],
    exact_name_map: dict[str, dict],
    by_canonical: dict[str, dict],
) -> dict | None:
    candidate_keys = {
        normalized_key,
        normalize_key(raw_name),
        normalize_key(strip_parenthetical(strip_approval_numbers(raw_name))),
        simplified_key(raw_name),
        normalize_key(str(raw_name).replace("(고시형)", "").replace("고시형", "")),
    }
    candidate_keys = {key for key in candidate_keys if key}
    for key in candidate_keys:
        record = exact_name_map.get(key)
        if record and not is_specificity_conflict(raw_name, record["canonical_name"]):
            return record

    for key in candidate_keys:
        names = sorted(alias_map.get(key, set()))
        if len(names) != 1:
            continue
        record = by_canonical.get(names[0])
        if record and not is_specificity_conflict(raw_name, record["canonical_name"]):
            return record
    return None


def resolve_curated_alias(raw_name: str, normalized_key: str, category_by_name: dict[str, dict], category_by_key: dict[str, dict]) -> dict | None:
    candidate_keys = {
        normalized_key,
        normalize_key(raw_name),
        normalize_key(strip_parenthetical(strip_approval_numbers(raw_name))),
        simplified_key(raw_name),
    }
    for key in candidate_keys:
        standard_name = CURATED_ALIAS_TO_STANDARD.get(key)
        if not standard_name:
            continue
        category = resolve_category_row(standard_name, category_by_name, category_by_key)
        if category:
            return category
    return None


def load_old_positive_cache(old_cache_path: Path, category_by_name: dict[str, dict], category_by_key: dict[str, dict]) -> dict[str, dict]:
    rows = load_csv(old_cache_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = normalize_key(row.get("normalized_raw") or row.get("raw_ingredient"))
        if key:
            grouped[key].append(row)

    trusted: dict[str, dict] = {}
    for key, items in grouped.items():
        candidates: list[dict] = []
        for row in items:
            relation = readable(row.get("relation_type"))
            if relation not in POSITIVE_RELATIONS:
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except ValueError:
                confidence = 0.0
            if confidence < 0.9:
                continue
            matched = readable(row.get("matched_standard_name"))
            category = resolve_category_row(matched, category_by_name, category_by_key)
            if not category:
                continue
            raw = readable(row.get("raw_ingredient"))
            canonical = category["functional_ingredient_name"]
            if is_specificity_conflict(raw, canonical):
                continue
            if not old_positive_relation_is_safe(raw, canonical, relation):
                continue
            candidates.append(
                {
                    "raw_ingredient": raw,
                    "matched_standard_name": canonical,
                    "relation_type": relation,
                    "confidence": confidence,
                    "reason": readable(row.get("reason")),
                    "category_main": category.get("category_main", ""),
                    "category_sub": category.get("category_sub", ""),
                    "source_cache_method": readable(row.get("match_method")),
                    "old_cache_row": row,
                }
            )
        if candidates:
            candidates.sort(key=lambda item: (-item["confidence"], item["matched_standard_name"]))
            trusted[key] = candidates[0]
    return trusted


def old_positive_relation_is_safe(raw_name: str, matched_name: str, relation_type: str) -> bool:
    raw_simple = simplified_key(raw_name)
    matched_simple = simplified_key(matched_name)
    if not raw_simple or not matched_simple:
        return False
    if raw_simple == matched_simple:
        return True
    raw_key = normalize_key(raw_name)
    matched_key = normalize_key(matched_name)
    curated_standard = CURATED_ALIAS_TO_STANDARD.get(raw_key) or CURATED_ALIAS_TO_STANDARD.get(raw_simple)
    if curated_standard and normalize_key(curated_standard) == normalize_key(matched_name):
        return True
    if relation_type in {"same_ingredient", "nutrient_form", "ingredient_group"}:
        if len(matched_simple) >= 2 and matched_simple in raw_simple:
            return True
        if len(matched_key) >= 3 and matched_key in raw_key:
            return True
    if relation_type == "marker_compound":
        trusted_marker_pairs = {
            ("실리마린", "밀크씨슬추출물"),
            ("지아잔틴", "마리골드꽃추출물지아잔틴함유"),
            ("루테인", "마리골드꽃추출물지아잔틴함유"),
        }
        return (raw_simple, matched_simple) in trusted_marker_pairs
    return False


def find_embedding_cache(cache_dir: Path, expected_rows: int) -> Path | None:
    if not cache_dir.exists():
        return None
    for path in sorted(cache_dir.glob("*.npz"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            import numpy as np

            vectors = np.load(path, allow_pickle=False)["vectors"]
            if int(vectors.shape[0]) == expected_rows:
                return path
        except Exception:
            continue
    return None


def build_vectors_for_records(records: list[dict[str, Any]], metadata_path: Path, cache_dir: Path, model_name: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    record_signature = hashlib.sha1(
        "|".join(f"{record['standard_name']}::{record['canonical_name']}" for record in records).encode("utf-8")
    ).hexdigest()[:16]
    filtered_cache_path = cache_dir / f"ingredient_match_v2_standard_vectors_{record_signature}.npz"
    if filtered_cache_path.exists():
        try:
            cached = np.load(filtered_cache_path, allow_pickle=False)["vectors"].astype("float32")
            if int(cached.shape[0]) == len(records):
                return cached
        except Exception:
            pass

    all_metadata = pickle.load(metadata_path.open("rb"))
    all_index_by_standard: dict[str, int] = {
        readable(item.get("standard_name")): idx
        for idx, item in enumerate(all_metadata)
        if readable(item.get("standard_name"))
    }
    cache_path = find_embedding_cache(cache_dir, len(all_metadata))
    vectors = None
    if cache_path:
        all_vectors = np.load(cache_path, allow_pickle=False)["vectors"].astype("float32")
        indexes = [all_index_by_standard.get(record["standard_name"], -1) for record in records]
        if all(index >= 0 for index in indexes):
            vectors = all_vectors[indexes]
    if vectors is None:
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            [record["embedding_text"] for record in records],
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
    np.savez_compressed(filtered_cache_path, vectors=vectors)
    return vectors


def lexical_candidates(raw_name: str, normalized_key: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    raw_keys = {
        normalized_key,
        normalize_key(raw_name),
        normalize_key(strip_parenthetical(strip_approval_numbers(raw_name))),
        simplified_key(raw_name),
    }
    raw_keys = {key for key in raw_keys if key}
    scored: dict[str, dict[str, Any]] = {}
    for record in records:
        best = 0.0
        alias_terms = record.get("alias_terms")
        if not alias_terms:
            alias_terms = build_alias_terms(record["aliases"])
        for alias_term in alias_terms:
            alias_key = alias_term["key"]
            alias_simple = alias_term["simple"]
            if not alias_key:
                continue
            if alias_key in raw_keys:
                best = max(best, 1.0)
            elif alias_simple and alias_simple in raw_keys:
                best = max(best, 0.95)
            elif len(alias_key) >= 4 and any(alias_key in key for key in raw_keys):
                best = max(best, 0.78)
            elif any(len(key) >= 4 and key in alias_key for key in raw_keys):
                best = max(best, 0.7)
        if best <= 0:
            continue
        if is_specificity_conflict(raw_name, record["canonical_name"]):
            best = min(best, 0.3)
        scored[record["canonical_name"]] = {"record": record, "lexical_score": best}
    return sorted(scored.values(), key=lambda item: (-item["lexical_score"], item["record"]["canonical_name"]))[:top_k]


def candidate_payload(record: dict[str, Any], rank: int, embedding_score: float = 0.0, lexical_score: float = 0.0) -> dict[str, Any]:
    return {
        "rank": rank,
        "standard_name": record["canonical_name"],
        "category_main": record.get("category_main", ""),
        "category_sub": record.get("category_sub", ""),
        "claim_text": truncate(record.get("claim_text", ""), 220),
        "synonyms": truncate(record.get("synonyms_joined", ""), 220),
        "embedding_score": round(float(embedding_score or 0.0), 6),
        "lexical_score": round(float(lexical_score or 0.0), 6),
    }


def build_prompt(items: list[dict[str, Any]]) -> str:
    compact_items = []
    for item in items:
        compact_items.append(
            {
                "id": item["ingredient_id"],
                "raw_ingredient": item["raw_ingredient"],
                "normalized_key": item["normalized_raw"],
                "mention_count": item["mention_count"],
                "product_count": item["product_count"],
                "candidates": item["rag_candidates"],
            }
        )
    return (
        "너는 건강기능식품 원재료명을 기능성 원료 DB의 표준 원료명에 매칭하는 판정자다.\n"
        "각 input_ingredient에 대해 candidates 안에서 기존 표준 원료 매칭 여부를 판단하라.\n\n"
        "판정 규칙:\n"
        "- 후보 중 원료 자체가 같고 제형 차이(분말/추출물/농축액), 인정번호 포함/미포함, 한글/영문 동의어 차이만 있으면 decision=existing_match.\n"
        "- 같은 기능 영역이지만 원료 자체가 다르면 existing_match로 두지 말고 family_signal 또는 no_match로 둔다.\n"
        "- 콜라겐분말처럼 일반명인데 후보가 홍어껍질콜라겐펩타이드/AP 콜라겐처럼 특정 원료이면 existing_match 금지. 필요하면 family_signal 또는 new_standard_candidate.\n"
        "- 상어연골추출물분말처럼 관절/연골 표준 원료(예: 뮤코다당.단백)와 계열상 연결되면 family_signal 또는 낮은 confidence의 existing_match로 판단한다.\n"
        "- 쇠무릅분말은 우슬 계열 약한 신호일 수 있으나 우슬 등 복합물(HL-Joint100)과 동일 원료로 판정하지 않는다. family_signal 또는 new_standard_candidate를 우선 고려한다.\n"
        "- 부형제/첨가물/향료/색소/일반 가공식품/광고문구는 decision=non_functional.\n"
        "- 후보가 없거나 후보가 부적절하지만 실제 기능성 원료 후보로 보이면 decision=new_standard_candidate.\n"
        "- 불확실하면 decision=uncertain_review, needs_human_review=true.\n\n"
        "허용 decision: existing_match, family_signal, new_standard_candidate, non_functional, no_match, uncertain_review\n"
        "허용 relation_type: same_ingredient, ingredient_group, marker_compound, nutrient_form, same_function_only, family_signal, excipient, unrelated, unknown\n"
        "confidence는 0.00~1.00 숫자로 쓴다. matched_standard_name은 candidates에 있는 표준명만 사용하라. 없으면 빈 문자열.\n"
        "반드시 JSON만 출력하라.\n\n"
        "출력 스키마:\n"
        '{"results":[{"id":"ing_000001","decision":"existing_match","matched_standard_name":"",'
        '"relation_type":"same_ingredient","confidence":0.0,"category_main":"","is_new_standard_candidate":false,'
        '"reason":"짧은 한국어 근거","needs_human_review":false}]}\n\n'
        "입력 items:\n"
        f"{json.dumps({'items': compact_items}, ensure_ascii=False)}"
    )


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingredient_match_cache_v2 (
            ingredient_id TEXT PRIMARY KEY,
            raw_ingredient TEXT NOT NULL,
            normalized_raw TEXT NOT NULL,
            representative_ingredient TEXT NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 0,
            product_count INTEGER NOT NULL DEFAULT 0,
            variant_samples_json TEXT NOT NULL DEFAULT '[]',
            decision TEXT NOT NULL,
            matched_standard_name TEXT NOT NULL DEFAULT '',
            relation_type TEXT NOT NULL DEFAULT 'unknown',
            confidence REAL NOT NULL DEFAULT 0,
            category_main TEXT NOT NULL DEFAULT '',
            category_sub TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            match_method TEXT NOT NULL DEFAULT '',
            rag_candidates_json TEXT NOT NULL DEFAULT '[]',
            llm_response_json TEXT NOT NULL DEFAULT '{}',
            needs_human_review INTEGER NOT NULL DEFAULT 0,
            cache_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ingredient_match_cache_v2_normalized_raw ON ingredient_match_cache_v2(normalized_raw)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ingredient_match_cache_v2_decision ON ingredient_match_cache_v2(decision)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ingredient_match_cache_v2_matched_standard ON ingredient_match_cache_v2(matched_standard_name)")
    conn.commit()


def upsert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        row.setdefault("created_at", now)
        row["updated_at"] = now
        columns = [
            "ingredient_id",
            "raw_ingredient",
            "normalized_raw",
            "representative_ingredient",
            "mention_count",
            "product_count",
            "variant_samples_json",
            "decision",
            "matched_standard_name",
            "relation_type",
            "confidence",
            "category_main",
            "category_sub",
            "reason",
            "match_method",
            "rag_candidates_json",
            "llm_response_json",
            "needs_human_review",
            "cache_version",
            "created_at",
            "updated_at",
        ]
        placeholders = ",".join(["?"] * len(columns))
        update_columns = [column for column in columns if column not in {"ingredient_id", "created_at"}]
        conn.execute(
            f"""
            INSERT INTO ingredient_match_cache_v2 ({",".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(ingredient_id) DO UPDATE SET
            {",".join([f"{column}=excluded.{column}" for column in update_columns])}
            """,
            [row.get(column, "") for column in columns],
        )
    conn.commit()


def export_sqlite(conn: sqlite3.Connection, output_dir: Path) -> None:
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM ingredient_match_cache_v2 ORDER BY ingredient_id")]
    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        write_csv(output_dir / "ingredient_match_cache_v2.csv", fieldnames, rows)
        review_rows = [row for row in rows if int(row.get("needs_human_review") or 0) or row.get("decision") in {"family_signal", "new_standard_candidate", "uncertain_review"}]
        write_csv(output_dir / "ingredient_match_v2_review.csv", fieldnames, review_rows)
        new_rows = [row for row in rows if row.get("decision") == "new_standard_candidate"]
        write_csv(output_dir / "new_ingredient_additions_candidates.csv", fieldnames, new_rows)


def prepare(args: argparse.Namespace) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / "ingredient_match_cache_v2.sqlite"
    batch_jsonl_path = output_dir / "ingredient_match_v2_gemini_batch.jsonl"

    unique_rows = load_csv(Path(args.unique_csv))
    if args.limit:
        unique_rows = unique_rows[: args.limit]
    category_rows = load_csv(Path(args.category_map))
    category_by_name, category_by_key = build_category_indexes(category_rows)
    standard_records, alias_map, exact_name_map, by_canonical = build_standard_records(category_rows, Path(args.metadata))
    trusted_old = load_old_positive_cache(Path(args.old_cache), category_by_name, category_by_key)

    prepared_rows: list[dict[str, Any]] = []
    pending_items: list[dict[str, Any]] = []
    summary_counter: Counter[str] = Counter()

    for index, row in enumerate(unique_rows, start=1):
        raw = readable(row.get("representative_ingredient"))
        normalized = normalize_key(row.get("normalized_key") or raw)
        ingredient_id = f"ing_{index:06d}"
        base = {
            "ingredient_id": ingredient_id,
            "raw_ingredient": raw,
            "normalized_raw": normalized,
            "representative_ingredient": raw,
            "mention_count": int(float(row.get("mention_count") or 0)),
            "product_count": int(float(row.get("product_count") or 0)),
            "variant_samples_json": row.get("variant_samples_json") or "[]",
            "cache_version": CACHE_VERSION,
        }
        auto_record = resolve_alias_match(raw, normalized, alias_map, exact_name_map, by_canonical)
        if auto_record:
            prepared = {
                **base,
                "decision": "existing_match",
                "matched_standard_name": auto_record["canonical_name"],
                "relation_type": "same_ingredient",
                "confidence": 0.99,
                "category_main": auto_record.get("category_main", ""),
                "category_sub": auto_record.get("category_sub", ""),
                "reason": "기능성 원료 DB 표준명/동의어와 정규화 exact match",
                "match_method": "auto_exact_alias_v2",
                "rag_candidates_json": "[]",
                "llm_response_json": "{}",
                "needs_human_review": 0,
            }
            prepared_rows.append(prepared)
            summary_counter[prepared["match_method"]] += 1
            continue

        curated = resolve_curated_alias(raw, normalized, category_by_name, category_by_key)
        if curated:
            prepared = {
                **base,
                "decision": "existing_match",
                "matched_standard_name": curated["functional_ingredient_name"],
                "relation_type": "nutrient_form",
                "confidence": 0.97,
                "category_main": curated.get("category_main", ""),
                "category_sub": curated.get("category_sub", ""),
                "reason": "영양성분 염/형태/고시형 표기 curated alias 매칭",
                "match_method": "curated_alias_v2",
                "rag_candidates_json": "[]",
                "llm_response_json": "{}",
                "needs_human_review": 0,
            }
            prepared_rows.append(prepared)
            summary_counter[prepared["match_method"]] += 1
            continue

        if is_non_functional_rule(raw, normalized):
            prepared = {
                **base,
                "decision": "non_functional",
                "matched_standard_name": "",
                "relation_type": "excipient",
                "confidence": 0.94,
                "category_main": "",
                "category_sub": "",
                "reason": "부형제/첨가물/일반 가공 원료 규칙에 해당",
                "match_method": "auto_non_functional_rule_v2",
                "rag_candidates_json": "[]",
                "llm_response_json": "{}",
                "needs_human_review": 0,
            }
            prepared_rows.append(prepared)
            summary_counter[prepared["match_method"]] += 1
            continue

        old = trusted_old.get(normalized)
        if old and normalized not in GENERIC_RAW_KEYS_REQUIRING_LLM:
            prepared = {
                **base,
                "decision": "existing_match",
                "matched_standard_name": old["matched_standard_name"],
                "relation_type": old["relation_type"],
                "confidence": min(0.98, float(old["confidence"])),
                "category_main": old.get("category_main", ""),
                "category_sub": old.get("category_sub", ""),
                "reason": f"기존 고신뢰 positive 캐시 승계: {old.get('reason', '')}",
                "match_method": "trusted_old_positive_cache_v2",
                "rag_candidates_json": "[]",
                "llm_response_json": json.dumps(old.get("old_cache_row", {}), ensure_ascii=False),
                "needs_human_review": 0,
            }
            prepared_rows.append(prepared)
            summary_counter[prepared["match_method"]] += 1
            continue

        pending_items.append(base)

    if pending_items:
        vectors = build_vectors_for_records(standard_records, Path(args.metadata), Path(args.embedding_cache_dir), args.model_name)
        model = SentenceTransformer(args.model_name)
        queries = [item["raw_ingredient"] for item in pending_items]
        query_vectors = model.encode(
            queries,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        top_k = max(1, int(args.top_k))
        standard_matrix = np.asarray(vectors, dtype="float32")
        pending_after_vector: list[dict[str, Any]] = []
        total_candidate_chunks = math.ceil(len(pending_items) / 512)
        for chunk_index, start in enumerate(range(0, len(pending_items), 512), start=1):
            end = min(start + 512, len(pending_items))
            scores = np.dot(query_vectors[start:end], standard_matrix.T)
            for offset, item in enumerate(pending_items[start:end]):
                row_scores = scores[offset]
                candidate_indices = np.argsort(row_scores)[::-1][:top_k]
                merged: dict[str, dict[str, Any]] = {}
                lexical = lexical_candidates(item["raw_ingredient"], item["normalized_raw"], standard_records, top_k)
                for lexical_item in lexical:
                    record = lexical_item["record"]
                    merged[record["canonical_name"]] = {
                        "record": record,
                        "embedding_score": 0.0,
                        "lexical_score": lexical_item["lexical_score"],
                    }
                for idx in candidate_indices:
                    record = standard_records[int(idx)]
                    existing = merged.setdefault(
                        record["canonical_name"],
                        {"record": record, "embedding_score": 0.0, "lexical_score": 0.0},
                    )
                    existing["embedding_score"] = max(existing["embedding_score"], float(row_scores[int(idx)]))

                sorted_candidates = sorted(
                    merged.values(),
                    key=lambda candidate: (
                        -max(candidate["lexical_score"], candidate["embedding_score"]),
                        -candidate["lexical_score"],
                        candidate["record"]["canonical_name"],
                    ),
                )[:top_k]
                payloads = [
                    candidate_payload(
                        candidate["record"],
                        rank=rank,
                        embedding_score=candidate["embedding_score"],
                        lexical_score=candidate["lexical_score"],
                    )
                    for rank, candidate in enumerate(sorted_candidates, start=1)
                ]

                top1 = payloads[0] if payloads else {}
                top2 = payloads[1] if len(payloads) >= 2 else {}
                top1_score = float(top1.get("embedding_score") or 0.0)
                top2_score = float(top2.get("embedding_score") or 0.0)
                top1_lex = float(top1.get("lexical_score") or 0.0)
                matched_name = str(top1.get("standard_name") or "")
                raw_simple = simplified_key(item["raw_ingredient"])
                matched_simple = simplified_key(matched_name)
                lexical_same_ingredient = (
                    top1_lex >= 0.95
                    and raw_simple
                    and matched_simple
                    and (raw_simple == matched_simple or matched_simple in raw_simple)
                )
                safe_exact = (
                    matched_name
                    and not is_specificity_conflict(item["raw_ingredient"], matched_name)
                    and (
                        lexical_same_ingredient
                        or (
                            top1_score >= float(args.auto_vector_threshold)
                            and (top1_score - top2_score) >= float(args.auto_vector_margin)
                            and raw_simple == matched_simple
                        )
                    )
                )
                if safe_exact:
                    category = resolve_category_row(matched_name, category_by_name, category_by_key) or {}
                    prepared = {
                        **item,
                        "decision": "existing_match",
                        "matched_standard_name": matched_name,
                        "relation_type": "same_ingredient",
                        "confidence": 0.96 if top1_lex >= 0.95 else 0.93,
                        "category_main": category.get("category_main", top1.get("category_main", "")),
                        "category_sub": category.get("category_sub", top1.get("category_sub", "")),
                        "reason": "RAG 후보가 lexical/embedding 고신뢰 exact 조건을 통과",
                        "match_method": "auto_high_confidence_rag_v2",
                        "rag_candidates_json": json.dumps(payloads, ensure_ascii=False),
                        "llm_response_json": "{}",
                        "needs_human_review": 0,
                    }
                    prepared_rows.append(prepared)
                    summary_counter[prepared["match_method"]] += 1
                    continue

                pending = {
                    **item,
                    "decision": "pending_llm",
                    "matched_standard_name": "",
                    "relation_type": "unknown",
                    "confidence": 0.0,
                    "category_main": "",
                    "category_sub": "",
                    "reason": "자동 판정 불충분, Gemini Batch LLM 판정 대기",
                    "match_method": "pending_gemini_batch_v2",
                    "rag_candidates": payloads,
                    "rag_candidates_json": json.dumps(payloads, ensure_ascii=False),
                    "llm_response_json": "{}",
                    "needs_human_review": 1,
                }
                pending_after_vector.append(pending)
            print(
                f"candidate_loop {chunk_index}/{total_candidate_chunks} "
                f"({chunk_index / total_candidate_chunks:.1%})",
                flush=True,
            )

        prepared_rows.extend(pending_after_vector)
        summary_counter["pending_gemini_batch_v2"] += len(pending_after_vector)

    conn = sqlite3.connect(sqlite_path)
    ensure_schema(conn)
    conn.execute("DELETE FROM ingredient_match_cache_v2")
    conn.commit()
    upsert_rows(conn, prepared_rows)
    export_sqlite(conn, output_dir)

    pending_for_batch = [
        row
        for row in prepared_rows
        if row.get("decision") == "pending_llm"
    ]
    group_size = max(1, int(args.group_size))
    with batch_jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(pending_for_batch), group_size):
            group = pending_for_batch[start : start + group_size]
            key = f"ingredient_match_v2_{start + 1:06d}_{start + len(group):06d}"
            request = {
                "key": key,
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": build_prompt(group)}]}],
                    "generation_config": {
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                    },
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_version": CACHE_VERSION,
        "unique_input_count": len(unique_rows),
        "standard_record_count": len(standard_records),
        "prepared_row_count": len(prepared_rows),
        "pending_llm_count": len(pending_for_batch),
        "batch_request_count": math.ceil(len(pending_for_batch) / group_size) if pending_for_batch else 0,
        "group_size": group_size,
        "method_counts": dict(summary_counter),
        "outputs": {
            "sqlite": str(sqlite_path),
            "cache_csv": str(output_dir / "ingredient_match_cache_v2.csv"),
            "review_csv": str(output_dir / "ingredient_match_v2_review.csv"),
            "new_standard_candidates_csv": str(output_dir / "new_ingredient_additions_candidates.csv"),
            "batch_jsonl": str(batch_jsonl_path),
        },
    }
    (output_dir / "ingredient_match_v2_prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("text"), str):
        return response["text"]
    texts: list[str] = []
    for candidate in response.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        for part in ((candidate.get("content") or {}).get("parts") or []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "\n".join(texts).strip()


def parse_model_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output root must be an object")
    return value


def response_from_batch_line(line_obj: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(line_obj.get("response"), dict):
        return line_obj["response"]
    inline_response = line_obj.get("inlineResponse") or line_obj.get("inline_response")
    if isinstance(inline_response, dict) and isinstance(inline_response.get("response"), dict):
        return inline_response["response"]
    if isinstance(line_obj.get("candidates"), list):
        return line_obj
    return None


def normalize_decision_fields(
    decision: str,
    relation: str,
    matched: str,
    category: dict | None,
    category_main_from_model: str,
    confidence: float,
) -> tuple[str, str, str, dict | None, str, str, float]:
    """Make LLM fields internally consistent before persisting them."""
    if decision == "existing_match":
        if not matched or not category:
            return "uncertain_review", "unknown", "", None, "", "", 0.0
        if relation not in POSITIVE_RELATIONS:
            relation = "same_ingredient"
        return decision, relation, matched, category, category.get("category_main", ""), category.get("category_sub", ""), confidence

    if decision == "family_signal":
        if not matched or not category:
            return "uncertain_review", "unknown", "", None, "", "", 0.0
        return decision, "family_signal", matched, category, category.get("category_main", ""), category.get("category_sub", ""), min(confidence, 0.6)

    if decision == "new_standard_candidate":
        return decision, "unknown", "", None, "", "", 0.0

    if decision == "non_functional":
        return decision, "excipient", "", None, "", "", confidence

    if decision == "no_match":
        return decision, "unrelated", "", None, "", "", 0.0

    return "uncertain_review", "unknown", "", None, "", "", 0.0


def apply_results(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    sqlite_path = output_dir / "ingredient_match_cache_v2.sqlite"
    result_path = Path(args.result_jsonl)
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)
    if not result_path.exists():
        raise FileNotFoundError(result_path)

    category_rows = load_csv(Path(args.category_map))
    category_by_name, category_by_key = build_category_indexes(category_rows)
    conn = sqlite3.connect(sqlite_path)
    ensure_schema(conn)
    conn.row_factory = sqlite3.Row

    result_count = 0
    error_rows: list[dict[str, Any]] = []
    with result_path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                line_obj = json.loads(line)
                if line_obj.get("error"):
                    raise ValueError(json.dumps(line_obj["error"], ensure_ascii=False))
                response = response_from_batch_line(line_obj)
                if not response:
                    raise ValueError("missing response")
                text = extract_response_text(response)
                parsed = parse_model_json(text)
                results = parsed.get("results")
                if not isinstance(results, list):
                    raise ValueError("missing results array")
            except Exception as exc:
                error_rows.append({"line_number": line_number, "error": str(exc), "raw": line[:4000]})
                continue

            for item in results:
                if not isinstance(item, dict):
                    error_rows.append({"line_number": line_number, "error": "result item is not object", "raw": str(item)[:4000]})
                    continue
                ingredient_id = readable(item.get("id"))
                if not ingredient_id:
                    error_rows.append({"line_number": line_number, "error": "missing id", "raw": json.dumps(item, ensure_ascii=False)})
                    continue
                row = conn.execute(
                    "SELECT * FROM ingredient_match_cache_v2 WHERE ingredient_id=?",
                    (ingredient_id,),
                ).fetchone()
                if not row:
                    error_rows.append({"line_number": line_number, "error": f"unknown id {ingredient_id}", "raw": json.dumps(item, ensure_ascii=False)})
                    continue

                decision = readable(item.get("decision"))
                if decision not in ALLOWED_DECISIONS:
                    decision = "uncertain_review"
                relation = readable(item.get("relation_type"))
                if relation not in ALLOWED_RELATIONS:
                    relation = "unknown"
                matched = readable(item.get("matched_standard_name"))
                category = resolve_category_row(matched, category_by_name, category_by_key) if matched else None
                if decision == "existing_match" and (not matched or not category):
                    decision = "uncertain_review"
                    relation = "unknown"
                if decision == "family_signal" and relation == "unknown":
                    relation = "family_signal"
                if decision == "non_functional" and relation == "unknown":
                    relation = "excipient"
                if decision == "no_match" and relation == "unknown":
                    relation = "unrelated"
                try:
                    confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
                except ValueError:
                    confidence = 0.0
                decision, relation, matched, category, category_main, category_sub, confidence = normalize_decision_fields(
                    decision,
                    relation,
                    matched,
                    category,
                    readable(item.get("category_main")),
                    confidence,
                )
                reason = truncate(item.get("reason", ""), 500)
                needs_review = bool(item.get("needs_human_review")) or decision in {
                    "family_signal",
                    "new_standard_candidate",
                    "uncertain_review",
                }
                now = datetime.now().isoformat(timespec="seconds")
                conn.execute(
                    """
                    UPDATE ingredient_match_cache_v2
                    SET decision=?,
                        matched_standard_name=?,
                        relation_type=?,
                        confidence=?,
                        category_main=?,
                        category_sub=?,
                        reason=?,
                        match_method=?,
                        llm_response_json=?,
                        needs_human_review=?,
                        updated_at=?
                    WHERE ingredient_id=?
                    """,
                    (
                        decision,
                        matched,
                        relation,
                        confidence,
                        category_main,
                        category_sub,
                        reason,
                        "gemini_batch_v2",
                        json.dumps(item, ensure_ascii=False),
                        int(needs_review),
                        now,
                        ingredient_id,
                    ),
                )
                result_count += 1
    conn.commit()
    export_sqlite(conn, output_dir)

    rows = [dict(row) for row in conn.execute("SELECT decision, match_method FROM ingredient_match_cache_v2")]
    decision_counts = Counter(row["decision"] for row in rows)
    method_counts = Counter(row["match_method"] for row in rows)
    pending_count = int(decision_counts.get("pending_llm", 0))
    summary = {
        "applied_at": datetime.now().isoformat(timespec="seconds"),
        "result_jsonl": str(result_path),
        "updated_result_count": result_count,
        "error_count": len(error_rows),
        "pending_llm_count": pending_count,
        "decision_counts": dict(decision_counts),
        "method_counts": dict(method_counts),
        "outputs": {
            "sqlite": str(sqlite_path),
            "cache_csv": str(output_dir / "ingredient_match_cache_v2.csv"),
            "review_csv": str(output_dir / "ingredient_match_v2_review.csv"),
            "new_standard_candidates_csv": str(output_dir / "new_ingredient_additions_candidates.csv"),
            "errors_csv": str(output_dir / "ingredient_match_v2_apply_errors.csv"),
        },
    }
    write_csv(output_dir / "ingredient_match_v2_apply_errors.csv", ["line_number", "error", "raw"], error_rows)
    (output_dir / "ingredient_match_v2_apply_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ingredient_match_cache_v2 using exact/cache/RAG/Gemini Batch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--unique-csv", default=str(DEFAULT_UNIQUE_CSV))
    prepare_parser.add_argument("--category-map", default=str(DEFAULT_CATEGORY_MAP))
    prepare_parser.add_argument("--old-cache", default=str(DEFAULT_OLD_CACHE))
    prepare_parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    prepare_parser.add_argument("--embedding-cache-dir", default=str(DEFAULT_EMBEDDING_CACHE_DIR))
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    prepare_parser.add_argument("--top-k", type=int, default=8)
    prepare_parser.add_argument("--group-size", type=int, default=10)
    prepare_parser.add_argument("--auto-vector-threshold", type=float, default=0.92)
    prepare_parser.add_argument("--auto-vector-margin", type=float, default=0.08)
    prepare_parser.add_argument("--limit", type=int, default=None)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--result-jsonl", required=True)
    apply_parser.add_argument("--category-map", default=str(DEFAULT_CATEGORY_MAP))
    apply_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "apply":
        apply_results(args)


if __name__ == "__main__":
    main()
