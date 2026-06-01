from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CATEGORY_MAP = ROOT_DIR / "output" / "cache_db_excel_export" / "temp_csv" / "functional_category_map.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "ingredient_match_v2"
DEFAULT_ENRICHMENT_DIR = DEFAULT_OUTPUT_DIR / "new_standard_enrichment"
DEFAULT_RUNTIME_SOURCE_SQLITE = Path(r"D:\ec2_cache_snapshot\ingredient_match_cache_rebuilt_item_class_i0050_final.sqlite")

CACHE_TABLE_COLUMNS = [
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


def normalize_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", str(value or "").strip().lower())


def readable(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_category_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        name = readable(row.get("functional_ingredient_name"))
        if not name:
            continue
        lookup[name] = row
        lookup[normalize_key(name)] = row
    return lookup


def resolve_category(name: str, lookup: dict[str, dict[str, str]]) -> dict[str, str]:
    if not name:
        return {}
    return lookup.get(readable(name)) or lookup.get(normalize_key(name)) or {}


def existing_action(standard_name: str, relation_type: str, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "curated_action": "existing_match",
        "matched_standard_name": standard_name,
        "relation_type": relation_type,
        "confidence": confidence,
        "curated_reason": reason,
    }


def family_action(standard_name: str, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "curated_action": "family_signal",
        "matched_standard_name": standard_name,
        "relation_type": "family_signal",
        "confidence": min(float(confidence or 0.0), 0.6),
        "curated_reason": reason,
    }


def nonfunctional_action(reason: str, confidence: float = 0.94) -> dict[str, Any]:
    return {
        "curated_action": "non_functional",
        "matched_standard_name": "",
        "relation_type": "excipient",
        "confidence": confidence,
        "curated_reason": reason,
    }


def review_action(reason: str, decision: str = "uncertain_review") -> dict[str, Any]:
    return {
        "curated_action": decision,
        "matched_standard_name": "",
        "relation_type": "unknown",
        "confidence": 0.0,
        "curated_reason": reason,
    }


ADD_READY_NAME_ACTIONS: dict[str, dict[str, Any]] = {
    "Bifidobacterium animalis spp. lactis": existing_action(
        "프로바이오틱스",
        "ingredient_group",
        0.86,
        "일반 균주/종 표기이며 기존 고시형 프로바이오틱스 범주로 처리한다.",
    ),
    "Bifidobacterium animalis ssp. lactis": existing_action(
        "프로바이오틱스",
        "ingredient_group",
        0.86,
        "일반 균주/종 표기이며 기존 고시형 프로바이오틱스 범주로 처리한다.",
    ),
    "Lactobacillus acidophilus": existing_action(
        "프로바이오틱스",
        "ingredient_group",
        0.84,
        "일반 종명은 특정 개별인정 균주가 아니라 기존 프로바이오틱스 범주로 처리한다.",
    ),
    "Lactobacillus (고시형)": existing_action(
        "프로바이오틱스",
        "ingredient_group",
        0.86,
        "고시형 Lactobacillus 표기는 신규 표준이 아니라 프로바이오틱스 범주 표기다.",
    ),
    "Lactobacillus 복합물 HY7601 + KY1032": existing_action(
        "L. curvatus HY7601와 L. plantarum KY1032의 프로바이오틱스 복합물",
        "same_ingredient",
        0.96,
        "기존 DB에 동일한 균주 조합 원료가 존재한다.",
    ),
    "Limosilactobacillus fermentum(고시형)": existing_action(
        "프로바이오틱스",
        "ingredient_group",
        0.86,
        "고시형 균종 표기는 신규 표준이 아니라 프로바이오틱스 범주 표기다.",
    ),
    "건조효모(크롬함유)": existing_action(
        "크롬",
        "nutrient_form",
        0.96,
        "건조효모는 크롬 공급원/제형이므로 기존 크롬 표준 원료로 처리한다.",
    ),
    "과채유래유산균(L.plantarum CJLP133)": existing_action(
        "Lactiplantibacillus plantarum CJLP133 프로바이오틱스",
        "same_ingredient",
        0.96,
        "기존 DB에 동일 균주 CJLP133 원료가 존재한다.",
    ),
    "락툴로즈": existing_action(
        "락추로스 파우더(Lactulose Powder)(기능성원료인정제2009-40호)",
        "same_ingredient",
        0.94,
        "락툴로즈/락추로스는 동일 성분 표기 변형이다.",
    ),
    "레티닐 아세트산염": existing_action(
        "비타민 A",
        "nutrient_form",
        0.96,
        "레티닐 아세트산염은 비타민 A 공급원/제형이다.",
    ),
    "비타민혼합분말": review_action(
        "복수 비타민의 혼합제제라 단일 기능성 표준 원료로 추가하면 과매칭 위험이 크다.",
    ),
    "비타민 혼합분말": review_action(
        "복수 비타민의 혼합제제라 단일 기능성 표준 원료로 추가하면 과매칭 위험이 크다.",
    ),
    "비피도박테리움 롱굼": existing_action(
        "프로바이오틱스",
        "ingredient_group",
        0.84,
        "일반 균종 표기는 특정 개별인정 균주가 아니라 프로바이오틱스 범주로 처리한다.",
    ),
    "아스코르빈산칼슘": existing_action(
        "비타민 C",
        "nutrient_form",
        0.96,
        "아스코르빈산칼슘은 비타민 C 공급원/제형이다.",
    ),
    "요오드화칼륨": existing_action(
        "요오드",
        "nutrient_form",
        0.96,
        "요오드화칼륨은 요오드 공급원/제형이다.",
    ),
    "치커리 뿌리 식이섬유": existing_action(
        "이눌린/치커리추출물",
        "ingredient_group",
        0.90,
        "치커리 유래 식이섬유/이눌린 계열은 기존 이눌린/치커리추출물 표준으로 처리한다.",
    ),
    "치커리 뿌리 추출물 분말 (식이섬유 90% 이상)": existing_action(
        "이눌린/치커리추출물",
        "ingredient_group",
        0.91,
        "치커리 유래 식이섬유/이눌린 계열은 기존 이눌린/치커리추출물 표준으로 처리한다.",
    ),
    "치커리 뿌리 추출 분말 (이눌린 식이섬유 80% 이상)": existing_action(
        "이눌린/치커리추출물",
        "same_ingredient",
        0.92,
        "치커리 이눌린 표기이므로 기존 이눌린/치커리추출물 표준으로 처리한다.",
    ),
    "치커리 식이섬유": existing_action(
        "이눌린/치커리추출물",
        "ingredient_group",
        0.90,
        "치커리 유래 식이섬유/이눌린 계열은 기존 이눌린/치커리추출물 표준으로 처리한다.",
    ),
    "치커리 이눌린": existing_action(
        "이눌린/치커리추출물",
        "same_ingredient",
        0.92,
        "치커리 이눌린 표기이므로 기존 이눌린/치커리추출물 표준으로 처리한다.",
    ),
    "치커리 추출물 분말 (화이브로우즈F90, 식이섬유 85%)": existing_action(
        "이눌린/치커리추출물",
        "ingredient_group",
        0.90,
        "치커리 유래 식이섬유 원료로 기존 이눌린/치커리추출물 표준 범주에 속한다.",
    ),
    "크롬효모": existing_action(
        "크롬",
        "nutrient_form",
        0.96,
        "크롬효모는 크롬 공급원/제형이다.",
    ),
    "탄산칼슘혼합제제": existing_action(
        "칼슘",
        "nutrient_form",
        0.96,
        "탄산칼슘 혼합제제는 칼슘 공급원/제형이다.",
    ),
}

CONFLICT_ID_ACTIONS: dict[str, dict[str, Any]] = {
    "ing_006400": existing_action("비타민 E", "nutrient_form", 0.97, "d-α-tocopherol은 비타민 E 계열 공급원이다."),
    "ing_006601": existing_action("판토텐산", "nutrient_form", 0.96, "판토텐산 함유 건조효모는 판토텐산 공급원/제형이다."),
    "ing_006654": nonfunctional_action("결정셀룰로오스는 통상 부형제이며 식이섬유 표기만으로 기능성 표준 원료로 올리면 과매칭된다."),
    "ing_006822": existing_action("칼슘", "nutrient_form", 0.95, "굴껍질/패각 분말은 칼슘 공급원이다."),
    "ing_007537": nonfunctional_action("향미 분말 혼합제제로 기능성 표준 원료가 아니다.", 0.96),
    "ing_008681": existing_action("상엽추출물", "same_ingredient", 0.94, "뽕나무잎은 상엽 표기와 같은 원료 계열이다."),
    "ing_008685": existing_action("상엽추출물", "same_ingredient", 0.95, "뽕나무잎추출물은 상엽추출물과 동일 표기 계열이다."),
    "ing_008842": existing_action("피쉬 콜라겐펩타이드", "same_ingredient", 0.93, "생선콜라겐펩타이드는 피쉬 콜라겐펩타이드와 동일 계열이다."),
    "ing_008843": existing_action("피쉬 콜라겐펩타이드", "same_ingredient", 0.94, "피쉬콜라겐 100% 생선콜라겐펩타이드 표기다."),
    "ing_009648": review_action("오자 혼합 추출물은 구기자 등 단일 원료로 환원하기 어려워 별도 검토 후보로 유지한다.", "new_standard_candidate"),
    "ing_009767": existing_action("요오드", "nutrient_form", 0.96, "요오드칼륨은 요오드 공급원/제형이다."),
    "ing_009947": existing_action("홍삼", "ingredient_group", 0.88, "유산균발효 홍삼 농축액은 홍삼 기반 원료로 기존 홍삼 범주에 연결한다."),
    "ing_009956": existing_action("아연", "nutrient_form", 0.95, "아연 함유 유산균배양액 분말은 아연 공급원/제형으로 처리한다."),
    "ing_010169": existing_action("자일리톨", "same_ingredient", 0.98, "자일리통은 자일리톨 오기/표기 변형으로 판단한다."),
    "ing_010710": existing_action("코엔자임Q10", "same_ingredient", 0.98, "코큐텐은 코엔자임Q10의 일반 표기다."),
    "ing_010711": existing_action("코엔자임Q10", "same_ingredient", 0.97, "코큐텐분말은 코엔자임Q10 분말 제형 표기다."),
    "ing_011070": nonfunctional_action("향미/혼합제제 전체명으로, 포함된 미량 비타민을 대표 기능성 원료로 매칭하면 과매칭된다.", 0.96),
    "ing_011307": existing_action("헛개나무", "ingredient_group", 0.88, "헛개나무 열매 농축추출물은 기존 헛개나무 범주에 연결한다."),
    "ing_011390": existing_action("호프추출물", "same_ingredient", 0.94, "호프초임계추출물은 호프추출물 제형/추출 방식 표기다."),
}


def action_for_add_ready_name(standard_name: str) -> dict[str, Any]:
    action = ADD_READY_NAME_ACTIONS.get(readable(standard_name))
    if action:
        return action
    return ADD_READY_NAME_ACTIONS.get(normalize_key(standard_name), review_action("수동 검토 규칙에 없는 add_ready 항목이다."))


def enrich_action_row(
    source_row: dict[str, Any],
    action: dict[str, Any],
    category_lookup: dict[str, dict[str, str]],
    *,
    review_group: str,
) -> dict[str, Any]:
    matched = readable(action.get("matched_standard_name"))
    category = resolve_category(matched, category_lookup)
    final_decision = action["curated_action"]
    return {
        **source_row,
        "review_group": review_group,
        "curated_action": final_decision,
        "curated_decision": final_decision,
        "curated_matched_standard_name": matched,
        "curated_relation_type": action.get("relation_type", "unknown"),
        "curated_confidence": action.get("confidence", 0.0),
        "curated_category_main": category.get("category_main", ""),
        "curated_category_sub": category.get("category_sub", ""),
        "curated_reason": action.get("curated_reason", ""),
    }


def make_cache_override(
    row: dict[str, Any],
    action: dict[str, Any],
    category_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    matched = readable(action.get("matched_standard_name"))
    category = resolve_category(matched, category_lookup)
    decision = action["curated_action"]
    relation = action.get("relation_type", "unknown")
    confidence = float(action.get("confidence") or 0.0)
    needs_review = decision in {"new_standard_candidate", "uncertain_review"}
    if decision in {"existing_match", "family_signal"} and not category:
        decision = "uncertain_review"
        matched = ""
        relation = "unknown"
        confidence = 0.0
        needs_review = True
    if decision == "family_signal":
        relation = "family_signal"
        confidence = min(confidence, 0.6)
        needs_review = True
    return {
        "ingredient_id": row.get("ingredient_id", ""),
        "raw_ingredient": row.get("raw_ingredient", ""),
        "normalized_raw": row.get("normalized_raw", ""),
        "representative_ingredient": row.get("representative_ingredient") or row.get("raw_ingredient", ""),
        "decision": decision,
        "matched_standard_name": matched if decision in {"existing_match", "family_signal"} else "",
        "relation_type": relation,
        "confidence": confidence,
        "category_main": category.get("category_main", "") if decision in {"existing_match", "family_signal"} else "",
        "category_sub": category.get("category_sub", "") if decision in {"existing_match", "family_signal"} else "",
        "reason": f"curated_review_20260601: {action.get('curated_reason', '')}",
        "match_method": "curated_review_v2_20260601",
        "rag_candidates_json": "[]",
        "llm_response_json": json.dumps(
            {
                "curated_action": action,
                "source_standard_candidate": row.get("standard_ingredient_name", ""),
            },
            ensure_ascii=False,
        ),
        "needs_human_review": int(needs_review),
    }


def is_long_mixed_raw(raw: str) -> bool:
    text = str(raw or "")
    return len(text) >= 240 or text.count(",") >= 5


def build_domain_guard_overrides(
    output_dir: Path,
    category_lookup: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], set[str]]:
    cache_csv = output_dir / "ingredient_match_cache_v2.csv"
    if not cache_csv.exists():
        return [], set()
    rows = load_csv(cache_csv)
    overrides: list[dict[str, Any]] = []
    removed_names: set[str] = set()

    for row in rows:
        raw = readable(row.get("raw_ingredient"))
        key = normalize_key(raw)
        if not raw or not key:
            continue

        action: dict[str, Any] | None = None
        if "상어연골" in raw:
            removed_names.update(
                {
                    raw,
                    readable(row.get("representative_ingredient")),
                    readable(row.get("matched_standard_name")),
                }
            )
            if is_long_mixed_raw(raw):
                action = review_action("상어연골이 포함된 장문 복합 원재료 문자열이다. 단일 원료 매칭으로 확정하지 않는다.")
            elif "뮤코다당" in raw or "뮤코다당단백" in key:
                action = existing_action(
                    "뮤코다당.단백",
                    "same_ingredient",
                    0.94,
                    "상어연골 원료에 뮤코다당.단백 지표가 명시되어 기존 뮤코다당.단백으로 매칭한다.",
                )
            else:
                action = family_action(
                    "뮤코다당.단백",
                    0.55,
                    "상어연골 계열은 관절/연골 원료인 뮤코다당.단백과 약한 계열 신호로만 반영한다.",
                )

        if "쇠무릅" in raw or "쇠무릎" in raw:
            removed_names.update(
                {
                    raw,
                    readable(row.get("representative_ingredient")),
                    readable(row.get("matched_standard_name")),
                }
            )
            if is_long_mixed_raw(raw):
                action = review_action("쇠무릅/쇠무릎이 포함된 장문 복합 원재료 문자열이다. 단일 원료 매칭으로 확정하지 않는다.")
            elif "hljoint100" in key or "우슬등복합물" in key:
                action = existing_action(
                    "우슬 등 복합물(HL-Joint100)(제2018-13호)",
                    "same_ingredient",
                    0.95,
                    "HL-Joint100/우슬 등 복합물 표기가 명시되어 기존 표준 원료와 동일하게 처리한다.",
                )
            else:
                action = family_action(
                    "우슬 등 복합물(HL-Joint100)(제2018-13호)",
                    0.45,
                    "쇠무릅/쇠무릎은 우슬 계열 관절 보조 원료 신호지만 HL-Joint100과 동일 원료로 확정하지 않는다.",
                )

        if not action:
            continue
        override = make_cache_override(row, action, category_lookup)
        if override.get("ingredient_id"):
            overrides.append(override)

    deduped = {row["ingredient_id"]: row for row in overrides}
    return list(deduped.values()), {name for name in removed_names if name}


def export_v2_sqlite(conn: sqlite3.Connection, output_dir: Path) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM ingredient_match_cache_v2 ORDER BY ingredient_id")]
    write_csv(output_dir / "ingredient_match_cache_v2.csv", CACHE_TABLE_COLUMNS, rows)
    review_rows = [
        row
        for row in rows
        if int(row.get("needs_human_review") or 0)
        or row.get("decision") in {"family_signal", "new_standard_candidate", "uncertain_review", "no_match"}
    ]
    write_csv(output_dir / "ingredient_match_v2_review.csv", CACHE_TABLE_COLUMNS, review_rows)
    new_candidate_rows = [row for row in rows if row.get("decision") == "new_standard_candidate"]
    write_csv(output_dir / "new_ingredient_additions_candidates.csv", CACHE_TABLE_COLUMNS, new_candidate_rows)
    return {
        "row_count": len(rows),
        "review_row_count": len(review_rows),
        "new_standard_candidate_count": len(new_candidate_rows),
        "decision_counts": dict(Counter(row.get("decision", "") for row in rows)),
        "method_counts": dict(Counter(row.get("match_method", "") for row in rows)),
    }


def apply_overrides_to_v2_cache(output_dir: Path, overrides: list[dict[str, Any]]) -> dict[str, Any]:
    sqlite_path = output_dir / "ingredient_match_cache_v2.sqlite"
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)
    now = datetime.now().isoformat(timespec="seconds")
    updated = 0
    missing: list[dict[str, Any]] = []
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.row_factory = sqlite3.Row
        for override in overrides:
            ingredient_id = readable(override.get("ingredient_id"))
            if not ingredient_id:
                continue
            existing = conn.execute(
                "SELECT ingredient_id FROM ingredient_match_cache_v2 WHERE ingredient_id=?",
                (ingredient_id,),
            ).fetchone()
            if not existing:
                missing.append(override)
                continue
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
                    rag_candidates_json=?,
                    llm_response_json=?,
                    needs_human_review=?,
                    updated_at=?
                WHERE ingredient_id=?
                """,
                (
                    override["decision"],
                    override["matched_standard_name"],
                    override["relation_type"],
                    override["confidence"],
                    override["category_main"],
                    override["category_sub"],
                    override["reason"],
                    override["match_method"],
                    override["rag_candidates_json"],
                    override["llm_response_json"],
                    override["needs_human_review"],
                    now,
                    ingredient_id,
                ),
            )
            updated += 1
        conn.commit()
        export_summary = export_v2_sqlite(conn, output_dir)
    finally:
        conn.close()
    return {
        "sqlite": str(sqlite_path),
        "updated_count": updated,
        "missing_count": len(missing),
        "missing_overrides": missing[:20],
        **export_summary,
    }


def filter_curated_rag_documents(
    enrichment_dir: Path,
    removed_standard_names: set[str],
    accepted_addition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_docs = enrichment_dir / "new_standard_candidate_rag_documents.csv"
    source_jsonl = enrichment_dir / "new_standard_candidate_rag_documents.jsonl"
    source_metadata = enrichment_dir / "functional_ingredient_metadata_v2_provisional.pkl"
    if not source_docs.exists():
        raise FileNotFoundError(source_docs)
    doc_rows = load_csv(source_docs)
    removed_keys = {normalize_key(name) for name in removed_standard_names if readable(name)}
    accepted_keys = {normalize_key(row.get("functional_ingredient_name")) for row in accepted_addition_rows}

    curated_doc_rows = []
    removed_doc_rows = []
    for row in doc_rows:
        key = normalize_key(row.get("standard_name"))
        if key in removed_keys and key not in accepted_keys:
            removed_doc_rows.append(row)
            continue
        curated_doc_rows.append(row)

    doc_fieldnames = list(doc_rows[0].keys()) if doc_rows else [
        "id",
        "standard_name",
        "synonyms_joined",
        "search_text",
        "function_text",
        "sources_joined",
        "embedding_text",
    ]
    curated_docs_csv = enrichment_dir / "new_standard_candidate_rag_documents_curated.csv"
    curated_docs_jsonl = enrichment_dir / "new_standard_candidate_rag_documents_curated.jsonl"
    removed_docs_csv = enrichment_dir / "new_standard_candidate_rag_documents_removed_by_review.csv"
    write_csv(curated_docs_csv, doc_fieldnames, curated_doc_rows)
    write_csv(removed_docs_csv, doc_fieldnames, removed_doc_rows)
    with curated_docs_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in curated_doc_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata_count = 0
    curated_metadata_count = 0
    metadata_path = enrichment_dir / "functional_ingredient_metadata_v2_curated.pkl"
    metadata_jsonl_path = enrichment_dir / "functional_ingredient_metadata_v2_curated.jsonl"
    if source_metadata.exists():
        metadata = pickle.load(source_metadata.open("rb"))
        metadata_count = len(metadata)
        curated_metadata = []
        for item in metadata:
            key = normalize_key(item.get("standard_name"))
            source = str(item.get("sources_joined", ""))
            is_new_candidate_doc = "llm_new_standard_candidate" in source
            if is_new_candidate_doc and key in removed_keys and key not in accepted_keys:
                continue
            curated_metadata.append(item)
        with metadata_path.open("wb") as handle:
            pickle.dump(curated_metadata, handle)
        with metadata_jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in curated_metadata:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        curated_metadata_count = len(curated_metadata)

    return {
        "source_document_count": len(doc_rows),
        "curated_document_count": len(curated_doc_rows),
        "removed_document_count": len(removed_doc_rows),
        "source_metadata_count": metadata_count,
        "curated_metadata_count": curated_metadata_count,
        "outputs": {
            "curated_docs_csv": str(curated_docs_csv),
            "curated_docs_jsonl": str(curated_docs_jsonl),
            "removed_docs_csv": str(removed_docs_csv),
            "curated_metadata_pkl": str(metadata_path),
            "curated_metadata_jsonl": str(metadata_jsonl_path),
        },
    }


def create_runtime_sqlite(
    source_sqlite: Path,
    target_sqlite: Path,
    overrides: list[dict[str, Any]],
    accepted_addition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not source_sqlite.exists():
        raise FileNotFoundError(source_sqlite)
    target_sqlite.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sqlite, target_sqlite)

    upserted_cache = 0
    inserted_custom = 0
    conn = sqlite3.connect(target_sqlite)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_custom_functional_ingredient_map (
                functional_ingredient_name TEXT PRIMARY KEY,
                category_main TEXT,
                category_sub TEXT,
                claim_text TEXT,
                source TEXT,
                confidence REAL,
                notes TEXT,
                categories_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                function_text TEXT DEFAULT ''
            )
            """
        )
        for override in overrides:
            normalized_raw = readable(override.get("normalized_raw"))
            raw = readable(override.get("raw_ingredient"))
            if not normalized_raw or not raw:
                continue
            conn.execute(
                """
                INSERT INTO ingredient_match_cache (
                    raw_ingredient, normalized_raw, matched_standard_name, relation_type,
                    confidence, reason, rag_candidates_json, gpt_response_json,
                    created_at, updated_at, match_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(normalized_raw) DO UPDATE SET
                    raw_ingredient=excluded.raw_ingredient,
                    matched_standard_name=excluded.matched_standard_name,
                    relation_type=excluded.relation_type,
                    confidence=excluded.confidence,
                    reason=excluded.reason,
                    rag_candidates_json=excluded.rag_candidates_json,
                    gpt_response_json=excluded.gpt_response_json,
                    updated_at=CURRENT_TIMESTAMP,
                    match_method=excluded.match_method
                """,
                (
                    raw,
                    normalized_raw,
                    override.get("matched_standard_name", ""),
                    override.get("relation_type", ""),
                    float(override.get("confidence") or 0.0),
                    override.get("reason", ""),
                    override.get("rag_candidates_json", "[]"),
                    override.get("llm_response_json", "{}"),
                    "curated_review_v2_20260601",
                ),
            )
            upserted_cache += 1

        for row in accepted_addition_rows:
            conn.execute(
                """
                INSERT INTO runtime_custom_functional_ingredient_map (
                    functional_ingredient_name, category_main, category_sub, function_text,
                    claim_text, source, confidence, notes, categories_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(functional_ingredient_name) DO UPDATE SET
                    category_main=excluded.category_main,
                    category_sub=excluded.category_sub,
                    function_text=excluded.function_text,
                    claim_text=excluded.claim_text,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    notes=excluded.notes,
                    categories_json=excluded.categories_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    row.get("functional_ingredient_name", ""),
                    row.get("category_main", ""),
                    row.get("category_sub", ""),
                    row.get("function_text", ""),
                    row.get("claim_text", ""),
                    "curated_new_standard_candidate",
                    float(row.get("confidence") or 0.0),
                    row.get("notes", ""),
                    row.get("categories_json", "[]"),
                ),
            )
            inserted_custom += 1
        conn.commit()
        cache_count = conn.execute("SELECT COUNT(*) FROM ingredient_match_cache").fetchone()[0]
        custom_count = conn.execute("SELECT COUNT(*) FROM runtime_custom_functional_ingredient_map").fetchone()[0]
    finally:
        conn.close()

    return {
        "runtime_sqlite": str(target_sqlite),
        "cache_overrides_upserted": upserted_cache,
        "custom_functional_rows_upserted": inserted_custom,
        "ingredient_match_cache_count": cache_count,
        "runtime_custom_functional_ingredient_map_count": custom_count,
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    enrichment_dir = Path(args.enrichment_dir)
    base_category_map = Path(args.base_category_map)
    source_runtime_sqlite = Path(args.runtime_source_sqlite)
    target_runtime_sqlite = Path(args.runtime_target_sqlite)

    category_rows = load_csv(base_category_map)
    category_lookup = build_category_lookup(category_rows)
    enriched_rows = load_csv(enrichment_dir / "new_standard_candidates_enriched.csv")
    unique_add_ready_rows = load_csv(enrichment_dir / "functional_category_map_new_candidates_add_ready.csv")
    conflict_rows = load_csv(enrichment_dir / "new_standard_candidates_existing_conflicts.csv")

    add_ready_unique_review: list[dict[str, Any]] = []
    for row in unique_add_ready_rows:
        standard_name = readable(row.get("functional_ingredient_name"))
        action = action_for_add_ready_name(standard_name)
        add_ready_unique_review.append(enrich_action_row(row, action, category_lookup, review_group="add_ready_unique_22"))

    add_ready_raw_review: list[dict[str, Any]] = []
    conflict_review: list[dict[str, Any]] = []
    overrides_by_id: dict[str, dict[str, Any]] = {}
    removed_new_candidate_names: set[str] = set()

    for row in enriched_rows:
        if row.get("review_status") != "add_ready":
            continue
        standard_name = readable(row.get("standard_ingredient_name"))
        action = action_for_add_ready_name(standard_name)
        reviewed = enrich_action_row(row, action, category_lookup, review_group="add_ready_raw")
        add_ready_raw_review.append(reviewed)
        removed_new_candidate_names.add(standard_name)
        override = make_cache_override(row, action, category_lookup)
        if override.get("ingredient_id"):
            overrides_by_id[override["ingredient_id"]] = override

    for row in conflict_rows:
        ingredient_id = readable(row.get("ingredient_id"))
        action = CONFLICT_ID_ACTIONS.get(ingredient_id, review_action("수동 충돌 해소 규칙에 없는 항목이다."))
        reviewed = enrich_action_row(row, action, category_lookup, review_group="existing_conflict")
        conflict_review.append(reviewed)
        if action["curated_action"] != "new_standard_candidate":
            removed_new_candidate_names.add(readable(row.get("standard_ingredient_name")))
        override = make_cache_override(row, action, category_lookup)
        if override.get("ingredient_id"):
            overrides_by_id[override["ingredient_id"]] = override

    domain_guard_overrides, domain_removed_names = build_domain_guard_overrides(output_dir, category_lookup)
    for override in domain_guard_overrides:
        overrides_by_id[override["ingredient_id"]] = override
    removed_new_candidate_names.update(domain_removed_names)

    overrides = list(overrides_by_id.values())
    existing_overrides = [row for row in overrides if row["decision"] == "existing_match"]
    family_overrides = [row for row in overrides if row["decision"] == "family_signal"]
    nonfunctional_overrides = [row for row in overrides if row["decision"] == "non_functional"]
    review_overrides = [row for row in overrides if row["decision"] in {"new_standard_candidate", "uncertain_review"}]

    accepted_addition_rows: list[dict[str, Any]] = []
    accepted_addition_fieldnames = [
        "functional_ingredient_name",
        "category_main",
        "category_sub",
        "claim_text",
        "source",
        "confidence",
        "notes",
        "categories_json",
        "function_text",
    ]

    category_fieldnames = list(category_rows[0].keys()) if category_rows else [
        "functional_ingredient_name",
        "category_main",
        "category_sub",
        "claim_text",
        "source",
        "confidence",
        "notes",
        "categories_json",
    ]
    curated_category_map = enrichment_dir / "functional_category_map_v2_curated.csv"
    curated_additions = enrichment_dir / "functional_category_map_v2_curated_additions.csv"
    write_csv(curated_category_map, category_fieldnames, category_rows)
    write_csv(curated_additions, accepted_addition_fieldnames, accepted_addition_rows)

    review_fieldnames = sorted(
        set().union(*(row.keys() for row in add_ready_unique_review + add_ready_raw_review + conflict_review))
    )
    write_csv(enrichment_dir / "add_ready_22_curated_review.csv", review_fieldnames, add_ready_unique_review)
    write_csv(enrichment_dir / "add_ready_raw_curated_review.csv", review_fieldnames, add_ready_raw_review)
    write_csv(enrichment_dir / "existing_conflicts_curated_resolution.csv", review_fieldnames, conflict_review)
    write_csv(enrichment_dir / "curated_existing_match_overrides.csv", CACHE_TABLE_COLUMNS, existing_overrides)
    write_csv(enrichment_dir / "curated_family_signal_overrides.csv", CACHE_TABLE_COLUMNS, family_overrides)
    write_csv(enrichment_dir / "curated_non_functional_overrides.csv", CACHE_TABLE_COLUMNS, nonfunctional_overrides)
    write_csv(enrichment_dir / "curated_review_required_overrides.csv", CACHE_TABLE_COLUMNS, review_overrides)
    write_csv(enrichment_dir / "curated_new_standard_additions.csv", accepted_addition_fieldnames, accepted_addition_rows)

    v2_summary = apply_overrides_to_v2_cache(output_dir, overrides)
    rag_summary = filter_curated_rag_documents(enrichment_dir, removed_new_candidate_names, accepted_addition_rows)
    runtime_summary = create_runtime_sqlite(source_runtime_sqlite, target_runtime_sqlite, overrides, accepted_addition_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_category_map": str(base_category_map),
        "reviewed_add_ready_unique_count": len(add_ready_unique_review),
        "reviewed_add_ready_raw_count": len(add_ready_raw_review),
        "reviewed_existing_conflict_count": len(conflict_review),
        "override_count": len(overrides),
        "existing_match_override_count": len(existing_overrides),
        "family_signal_override_count": len(family_overrides),
        "non_functional_override_count": len(nonfunctional_overrides),
        "review_required_override_count": len(review_overrides),
        "domain_guard_override_count": len(domain_guard_overrides),
        "accepted_new_standard_addition_count": len(accepted_addition_rows),
        "removed_new_candidate_name_count": len(removed_new_candidate_names),
        "v2_cache": v2_summary,
        "rag": rag_summary,
        "runtime": runtime_summary,
        "outputs": {
            "add_ready_22_review": str(enrichment_dir / "add_ready_22_curated_review.csv"),
            "existing_conflicts_review": str(enrichment_dir / "existing_conflicts_curated_resolution.csv"),
            "curated_category_map": str(curated_category_map),
            "curated_additions": str(curated_additions),
            "runtime_sqlite": str(target_runtime_sqlite),
        },
    }
    write_json(enrichment_dir / "curated_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate add_ready/new-standard conflicts and wire reviewed outputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--enrichment-dir", default=str(DEFAULT_ENRICHMENT_DIR))
    parser.add_argument("--base-category-map", default=str(DEFAULT_BASE_CATEGORY_MAP))
    parser.add_argument("--runtime-source-sqlite", default=str(DEFAULT_RUNTIME_SOURCE_SQLITE))
    parser.add_argument(
        "--runtime-target-sqlite",
        default=str(DEFAULT_ENRICHMENT_DIR / "runtime_curated.sqlite"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
