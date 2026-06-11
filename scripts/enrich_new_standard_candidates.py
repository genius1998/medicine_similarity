from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT_DIR / "output" / "ingredient_match_v2" / "new_ingredient_additions_candidates.csv"
DEFAULT_CATEGORY_MAP = ROOT_DIR / "output" / "cache_db_excel_export" / "temp_csv" / "functional_category_map.csv"
DEFAULT_METADATA = Path(
    os.environ.get(
        "FUNCTIONAL_INGREDIENT_METADATA",
        ROOT_DIR / "output" / "ingredient_match_v2_promoted" / "functional_ingredient_metadata_v2_promoted.pkl",
    )
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "ingredient_match_v2" / "new_standard_enrichment"
DEFAULT_HEALTH_BATCH_DIR = Path(os.environ.get("HEALTH_BATCH_DIR", ROOT_DIR / "output" / "batch_jobs"))
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ALLOWED_REVIEW_STATUS = {"add_ready", "review_required", "reject_non_functional"}
ALLOWED_EVIDENCE_STATUS = {
    "official_approved",
    "general_literature",
    "traditional_use",
    "review_required",
    "insufficient",
}
MAP_FIELDS = [
    "functional_ingredient_name",
    "category_main",
    "category_sub",
    "claim_text",
    "source",
    "confidence",
    "notes",
    "categories_json",
]
ENRICHED_FIELDS = [
    "ingredient_id",
    "raw_ingredient",
    "normalized_raw",
    "representative_ingredient",
    "standard_ingredient_name",
    "aliases_json",
    "category_main",
    "category_sub",
    "function_description",
    "claim_text",
    "evidence_status",
    "review_status",
    "source",
    "confidence",
    "existing_standard_conflict",
    "existing_standard_name",
    "existing_alias_conflict",
    "existing_alias_name",
    "notes",
    "rag_search_text",
    "embedding_text",
    "mention_count",
    "product_count",
    "needs_human_review",
    "llm_response_json",
]


def readable(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def multiline(value: Any) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(value or "").strip())


def normalize_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", str(value or "").strip().lower())


def strip_parenthetical(value: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", str(value or ""))
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return readable(text)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_categories(category_map: Path) -> list[str]:
    rows = read_csv(category_map)
    categories = sorted({readable(row.get("category_main")) for row in rows if readable(row.get("category_main"))})
    for extra in ["기타", "검토필요"]:
        if extra not in categories:
            categories.append(extra)
    return categories


def load_existing_name_index(category_map: Path) -> dict[str, str]:
    rows = read_csv(category_map)
    index: dict[str, str] = {}
    for row in rows:
        name = readable(row.get("functional_ingredient_name"))
        key = normalize_key(name)
        if key:
            index.setdefault(key, name)
    return index


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        try:
            parsed = json.loads(str(value or "[]"))
            items = parsed if isinstance(parsed, list) else []
        except Exception:
            items = []
    return [readable(item) for item in items if readable(item)]


def build_aliases(source: dict[str, str], model_aliases: list[str], standard_name: str) -> list[str]:
    aliases: list[str] = []
    for value in [
        standard_name,
        source.get("raw_ingredient", ""),
        source.get("representative_ingredient", ""),
        *parse_json_list(source.get("variant_samples_json")),
        *model_aliases,
    ]:
        text = readable(value)
        if text:
            aliases.append(text)
        stripped = strip_parenthetical(text)
        if stripped:
            aliases.append(stripped)
        key_removed = re.sub(r"(추출물분말|추출분말|농축분말|분말|추출물|농축액|농축)", "", stripped).strip()
        if key_removed:
            aliases.append(key_removed)

    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        key = normalize_key(alias)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(alias)
    return result[:12]


def build_prompt(items: list[dict[str, Any]], categories: list[str]) -> str:
    compact_items = [
        {
            "id": item["ingredient_id"],
            "raw_ingredient": item["raw_ingredient"],
            "normalized_raw": item["normalized_raw"],
            "representative_ingredient": item.get("representative_ingredient", ""),
            "mention_count": item.get("mention_count", ""),
            "product_count": item.get("product_count", ""),
        }
        for item in items
    ]
    return (
        "너는 건강기능식품 원료 표준 DB를 보강하기 위한 신규 원료 후보 설명을 작성하는 전문가다.\n"
        "아래 원료들은 기존 functional_category_map에 일치하는 표준 원료가 없다고 판정된 new_standard_candidate이다.\n"
        "기존 후보 원료와 억지로 연결하지 말고, 입력 원료 자체를 독립적인 신규 표준 후보로 다룬다.\n\n"
        "작성 원칙:\n"
        "- 공식 인정 기능성을 확실히 모르면 승인된 기능성처럼 단정하지 않는다.\n"
        "- 기능 설명은 RAG 검색 보조용이다. 의학적 효능을 과장하지 말고 '가능성', '검토 필요' 표현을 사용한다.\n"
        "- 단순 향료, 착색료, 가공식품, 일반 과즙/맛분말처럼 기능성 원료로 보기 어려우면 review_status=reject_non_functional.\n"
        "- 기능성 원료일 수 있으나 근거가 부족하거나 범용 식물/식품 원료이면 review_status=review_required.\n"
        "- 원료명, 동의어, 기능 설명, 카테고리, 검토 상태는 이후 벡터 임베딩과 LLM 매칭 보조에 사용된다.\n"
        f"- category_main은 다음 중 하나를 우선 사용한다: {', '.join(categories)}\n\n"
        "허용 review_status: add_ready, review_required, reject_non_functional\n"
        "허용 evidence_status: official_approved, general_literature, traditional_use, review_required, insufficient\n"
        "반드시 JSON만 출력한다.\n\n"
        "출력 스키마:\n"
        '{"results":[{"id":"ing_000001","standard_ingredient_name":"원료명","aliases":["동의어1"],'
        '"category_main":"기타","category_sub":"식물추출물/검토필요","function_description":"RAG용 기능 설명",'
        '"claim_text":"주의적 기능 설명 또는 빈 문자열","evidence_status":"review_required",'
        '"review_status":"review_required","confidence":0.5,"notes":"검토 사유","needs_human_review":true}]}\n\n'
        "입력 items:\n"
        f"{json.dumps({'items': compact_items}, ensure_ascii=False)}"
    )


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv(Path(args.input_csv)) if row.get("decision") == "new_standard_candidate"]
    if args.limit:
        rows = rows[: args.limit]
    categories = load_categories(Path(args.category_map))

    requests: list[dict[str, Any]] = []
    request_map: dict[str, list[str]] = {}
    for start in range(0, len(rows), args.group_size):
        group = rows[start : start + args.group_size]
        key = f"new_standard_enrich_{start + 1:06d}_{start + len(group):06d}"
        request_map[key] = [row["ingredient_id"] for row in group]
        requests.append(
            {
                "key": key,
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": build_prompt(group, categories)}]}],
                    "generation_config": {"temperature": 0.0, "response_mime_type": "application/json"},
                },
            }
        )

    batch_jsonl = output_dir / "new_standard_enrichment_batch.jsonl"
    write_jsonl(batch_jsonl, requests)
    (output_dir / "new_standard_enrichment_request_map.json").write_text(
        json.dumps(request_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(args.input_csv),
        "candidate_count": len(rows),
        "request_count": len(requests),
        "group_size": args.group_size,
        "batch_jsonl": str(batch_jsonl),
    }
    (output_dir / "new_standard_enrichment_prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_command(command: list[str], cwd: Path, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")
    return proc


def extract_job_name(output: str) -> str:
    match = re.search(r"Batch job name:\s*(\S+)", output)
    if not match:
        raise RuntimeError("Batch job name not found in submit output")
    return match.group(1)


def extract_state(output: str) -> str:
    match = re.search(r"(JOB_STATE_[A-Z_]+)", output)
    return match.group(1) if match else "UNKNOWN"


def write_progress(output_dir: Path, **payload: Any) -> None:
    progress = {"updated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    (output_dir / "new_standard_enrichment_progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_batch(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    batch_jsonl = output_dir / "new_standard_enrichment_batch.jsonl"
    result_jsonl = output_dir / "new_standard_enrichment_result.jsonl"
    job_path = output_dir / "new_standard_enrichment.job.txt"
    health_batch_dir = Path(args.health_batch_dir)
    if result_jsonl.exists() and result_jsonl.stat().st_size > 0 and not args.force:
        print(f"result already exists: {result_jsonl}")
        return

    if job_path.exists() and not args.force:
        job_name = job_path.read_text(encoding="utf-8").strip()
    else:
        last_output = ""
        for attempt in range(1, args.max_submit_attempts + 1):
            write_progress(output_dir, phase="submit", attempt=attempt, percent=5)
            proc = run_command(
                [
                    "py",
                    "-3.12",
                    "submit_gemini_batch.py",
                    "--jsonl",
                    str(batch_jsonl),
                    "--name",
                    f"new-standard-enrich-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                ],
                cwd=health_batch_dir,
                allow_failure=True,
            )
            last_output = proc.stdout or ""
            (output_dir / "new_standard_enrichment_submit.log").write_text(last_output, encoding="utf-8")
            if proc.returncode == 0:
                job_name = extract_job_name(last_output)
                job_path.write_text(job_name + "\n", encoding="utf-8")
                break
            if "RESOURCE_EXHAUSTED" not in last_output:
                raise RuntimeError("Gemini Batch submit failed")
            time.sleep(args.submit_retry_seconds)
        else:
            raise RuntimeError(f"Gemini Batch submit retry exceeded: {last_output[-1000:]}")

    while True:
        proc = run_command(["py", "-3.12", "check_gemini_batch.py", "--job", job_name], cwd=health_batch_dir)
        state = extract_state(proc.stdout or "")
        write_progress(output_dir, phase="batch", job=job_name, state=state, percent=50)
        if state == "JOB_STATE_SUCCEEDED":
            run_command(
                [
                    "py",
                    "-3.12",
                    "check_gemini_batch.py",
                    "--job",
                    job_name,
                    "--download-output",
                    str(result_jsonl),
                ],
                cwd=health_batch_dir,
            )
            write_progress(output_dir, phase="downloaded", job=job_name, state=state, percent=100)
            return
        if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
            raise RuntimeError(f"Gemini Batch job ended with {state}")
        time.sleep(args.poll_seconds)


def response_from_batch_line(line_obj: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(line_obj.get("response"), dict):
        return line_obj["response"]
    inline_response = line_obj.get("inlineResponse") or line_obj.get("inline_response")
    if isinstance(inline_response, dict) and isinstance(inline_response.get("response"), dict):
        return inline_response["response"]
    if isinstance(line_obj.get("candidates"), list):
        return line_obj
    return None


def extract_response_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "\n".join(texts).strip()


def parse_model_json(text: str) -> dict[str, Any]:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("model output root must be object")
    return parsed


def fallback_row(source: dict[str, str], reason: str) -> dict[str, Any]:
    standard_name = readable(source.get("raw_ingredient"))
    aliases = build_aliases(source, [], standard_name)
    function_description = f"{standard_name} 원료 후보입니다. 건강기능식품 기능성 원료로 추가할지 근거 검토가 필요합니다."
    return build_enriched_row(
        source,
        {
            "standard_ingredient_name": standard_name,
            "aliases": aliases,
            "category_main": "검토필요",
            "category_sub": "신규 원료 후보/검토필요",
            "function_description": function_description,
            "claim_text": function_description,
            "evidence_status": "review_required",
            "review_status": "review_required",
            "confidence": 0.0,
            "notes": reason,
            "needs_human_review": True,
        },
    )


def build_search_text(row: dict[str, Any], aliases: list[str]) -> str:
    return multiline(
        f"""신규 표준 후보 원료명: {row['standard_ingredient_name']}

이 문서는 기존 기능성 원료 DB에 직접 매칭되지 않은 원료를 RAG 검색 보조용 신규 후보로 정리한 것이다.
검토 상태가 완료되기 전까지 공식 인정 기능성 원료로 단정하지 않는다.

대표 원료명:
{row['standard_ingredient_name']}

동의어 후보:
{', '.join(aliases) if aliases else '-'}

카테고리:
{row['category_main']} / {row['category_sub']}

기능 설명:
{row['function_description']}

기능성 표현/claim:
{row['claim_text'] or '-'}

근거 상태:
{row['evidence_status']}

검토 상태:
{row['review_status']}

메모:
{row['notes'] or '-'}
"""
    )


def build_enriched_row(source: dict[str, str], item: dict[str, Any]) -> dict[str, Any]:
    standard_name = readable(item.get("standard_ingredient_name") or item.get("standard_name") or source.get("raw_ingredient"))
    aliases = build_aliases(source, parse_json_list(item.get("aliases")), standard_name)
    review_status = readable(item.get("review_status"))
    if review_status not in ALLOWED_REVIEW_STATUS:
        review_status = "review_required"
    evidence_status = readable(item.get("evidence_status"))
    if evidence_status not in ALLOWED_EVIDENCE_STATUS:
        evidence_status = "review_required"
    category_main = readable(item.get("category_main"))
    category_sub = readable(item.get("category_sub"))
    if review_status == "reject_non_functional":
        category_main = ""
        category_sub = ""
        evidence_status = "insufficient"
    elif not category_main:
        category_main = "검토필요"
    if not category_sub and category_main:
        category_sub = "신규 원료 후보/검토필요"
    function_description = multiline(item.get("function_description"))
    if not function_description:
        function_description = f"{standard_name} 원료 후보입니다. 기능성 원료 여부와 기능 설명은 추가 검토가 필요합니다."
    claim_text = multiline(item.get("claim_text")) or function_description
    notes = multiline(item.get("notes"))
    try:
        confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
    except Exception:
        confidence = 0.0
    row = {
        "ingredient_id": source.get("ingredient_id", ""),
        "raw_ingredient": source.get("raw_ingredient", ""),
        "normalized_raw": source.get("normalized_raw", ""),
        "representative_ingredient": source.get("representative_ingredient", ""),
        "standard_ingredient_name": standard_name,
        "aliases_json": json.dumps(aliases, ensure_ascii=False),
        "category_main": category_main,
        "category_sub": category_sub,
        "function_description": function_description,
        "claim_text": claim_text,
        "evidence_status": evidence_status,
        "review_status": review_status,
        "source": "llm_new_standard_candidate",
        "confidence": confidence,
        "existing_standard_conflict": 0,
        "existing_standard_name": "",
        "existing_alias_conflict": 0,
        "existing_alias_name": "",
        "notes": notes,
        "mention_count": source.get("mention_count", ""),
        "product_count": source.get("product_count", ""),
        "needs_human_review": 1,
        "llm_response_json": json.dumps(item, ensure_ascii=False),
    }
    row["rag_search_text"] = build_search_text(row, aliases)
    row["embedding_text"] = (
        f"원료명: {standard_name}\n"
        f"동의어: {', '.join(aliases)}\n"
        f"카테고리: {category_main} / {category_sub}\n"
        f"기능 설명: {function_description}\n"
        f"근거 상태: {evidence_status}\n"
        f"검토 상태: {review_status}"
    )
    return row


def annotate_existing_conflicts(rows: list[dict[str, Any]], category_map: Path) -> list[dict[str, Any]]:
    existing = load_existing_name_index(category_map)
    for row in rows:
        standard_key = normalize_key(row.get("standard_ingredient_name"))
        if standard_key in existing:
            row["existing_standard_conflict"] = 1
            row["existing_standard_name"] = existing[standard_key]

        alias_hit = ""
        for alias in parse_json_list(row.get("aliases_json")):
            alias_key = normalize_key(alias)
            if alias_key in existing:
                alias_hit = existing[alias_key]
                break
        if alias_hit and not row.get("existing_standard_name"):
            row["existing_alias_conflict"] = 1
            row["existing_alias_name"] = alias_hit
    return rows


def row_priority(row: dict[str, Any]) -> tuple[int, float, int]:
    review_rank = {"add_ready": 2, "review_required": 1, "reject_non_functional": 0}.get(str(row.get("review_status")), 0)
    try:
        confidence = float(row.get("confidence") or 0)
    except Exception:
        confidence = 0.0
    try:
        product_count = int(float(row.get("product_count") or 0))
    except Exception:
        product_count = 0
    return review_rank, confidence, product_count


def dedupe_standard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = normalize_key(row.get("standard_ingredient_name")) or str(row.get("ingredient_id"))
        grouped.setdefault(key, []).append(row)

    deduped: list[dict[str, Any]] = []
    for group in grouped.values():
        chosen = dict(sorted(group, key=row_priority, reverse=True)[0])
        alias_values: list[str] = []
        for row in group:
            alias_values.extend(parse_json_list(row.get("aliases_json")))
            alias_values.append(readable(row.get("raw_ingredient")))
            alias_values.append(readable(row.get("representative_ingredient")))
        seen: set[str] = set()
        aliases: list[str] = []
        for alias in alias_values:
            key = normalize_key(alias)
            if key and key not in seen:
                seen.add(key)
                aliases.append(alias)
        chosen["aliases_json"] = json.dumps(aliases[:20], ensure_ascii=False)
        if len(group) > 1:
            note = readable(chosen.get("notes"))
            chosen["notes"] = readable(f"{note} merged_duplicate_count={len(group)}")
            chosen["rag_search_text"] = build_search_text(chosen, aliases[:20])
            chosen["embedding_text"] = (
                f"원료명: {chosen.get('standard_ingredient_name', '')}\n"
                f"동의어: {', '.join(aliases[:20])}\n"
                f"카테고리: {chosen.get('category_main', '')} / {chosen.get('category_sub', '')}\n"
                f"기능 설명: {chosen.get('function_description', '')}\n"
                f"근거 상태: {chosen.get('evidence_status', '')}\n"
                f"검토 상태: {chosen.get('review_status', '')}"
            )
        deduped.append(chosen)
    return sorted(deduped, key=lambda row: normalize_key(row.get("standard_ingredient_name")) or str(row.get("ingredient_id")))


def load_request_map(output_dir: Path) -> dict[str, list[str]]:
    path = output_dir / "new_standard_enrichment_request_map.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def enriched_to_map_row(row: dict[str, Any]) -> dict[str, Any]:
    category_main = readable(row.get("category_main"))
    categories = [category_main] if category_main else []
    return {
        "functional_ingredient_name": row.get("standard_ingredient_name", ""),
        "category_main": category_main,
        "category_sub": row.get("category_sub", ""),
        "claim_text": row.get("claim_text", ""),
        "source": row.get("source", "llm_new_standard_candidate"),
        "confidence": row.get("confidence", 0.0),
        "notes": f"review_status={row.get('review_status')}; evidence_status={row.get('evidence_status')}; {row.get('notes', '')}".strip(),
        "categories_json": json.dumps(categories, ensure_ascii=False),
    }


def enriched_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    name = row.get("standard_ingredient_name", "")
    aliases = parse_json_list(row.get("aliases_json"))
    return {
        "id": f"functional_ingredient::{name}",
        "standard_name": name,
        "synonym_count": len(aliases),
        "synonyms_joined": ", ".join(aliases),
        "search_text": row.get("rag_search_text", ""),
        "sources_joined": row.get("source", "llm_new_standard_candidate"),
        "function_text": row.get("function_description", ""),
        "caution_text": "신규 후보 원료이며 공식 기능성 인정 여부는 검토 필요",
        "embedding_text": row.get("embedding_text", ""),
        "canonical_priority": 0,
        "specific_penalty": 0,
    }


def save_artifacts(args: argparse.Namespace, enriched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    category_map_rows = read_csv(Path(args.category_map))
    enriched_rows = annotate_existing_conflicts(enriched_rows, Path(args.category_map))
    usable_rows = [row for row in enriched_rows if row.get("review_status") != "reject_non_functional"]
    usable_unique_rows = dedupe_standard_rows(usable_rows)
    add_ready_rows = [
        row
        for row in usable_unique_rows
        if row.get("review_status") == "add_ready"
        and not int(row.get("existing_standard_conflict") or 0)
        and not int(row.get("existing_alias_conflict") or 0)
    ]
    conflict_rows = [
        row
        for row in enriched_rows
        if int(row.get("existing_standard_conflict") or 0) or int(row.get("existing_alias_conflict") or 0)
    ]
    map_new_rows = [enriched_to_map_row(row) for row in usable_unique_rows]
    map_add_ready_rows = [enriched_to_map_row(row) for row in add_ready_rows]

    enriched_csv = output_dir / "new_standard_candidates_enriched.csv"
    map_new_csv = output_dir / "functional_category_map_new_candidates_only.csv"
    map_add_ready_csv = output_dir / "functional_category_map_new_candidates_add_ready.csv"
    map_provisional_csv = output_dir / "functional_category_map_v2_provisional.csv"
    map_add_ready_provisional_csv = output_dir / "functional_category_map_v2_add_ready_provisional.csv"
    conflict_csv = output_dir / "new_standard_candidates_existing_conflicts.csv"
    rag_jsonl = output_dir / "new_standard_candidate_rag_documents.jsonl"
    rag_csv = output_dir / "new_standard_candidate_rag_documents.csv"
    metadata_jsonl = output_dir / "functional_ingredient_metadata_v2_provisional.jsonl"
    metadata_pkl = output_dir / "functional_ingredient_metadata_v2_provisional.pkl"

    write_csv(enriched_csv, ENRICHED_FIELDS, enriched_rows)
    write_csv(map_new_csv, MAP_FIELDS, map_new_rows)
    write_csv(map_add_ready_csv, MAP_FIELDS, map_add_ready_rows)
    write_csv(map_provisional_csv, MAP_FIELDS, category_map_rows + map_new_rows)
    write_csv(map_add_ready_provisional_csv, MAP_FIELDS, category_map_rows + map_add_ready_rows)
    write_csv(conflict_csv, ENRICHED_FIELDS, conflict_rows)
    metadata_rows = [enriched_to_metadata(row) for row in usable_unique_rows]
    write_jsonl(rag_jsonl, metadata_rows)
    write_csv(
        rag_csv,
        ["id", "standard_name", "synonyms_joined", "search_text", "function_text", "sources_joined", "embedding_text"],
        metadata_rows,
    )

    combined_metadata: list[dict[str, Any]] = []
    if Path(args.metadata).exists():
        with Path(args.metadata).open("rb") as file:
            combined_metadata = pickle.load(file)
    combined_metadata.extend(metadata_rows)
    with metadata_pkl.open("wb") as file:
        pickle.dump(combined_metadata, file)
    write_jsonl(metadata_jsonl, combined_metadata)

    outputs = {
        "enriched_csv": str(enriched_csv),
        "new_candidates_map_csv": str(map_new_csv),
        "add_ready_map_csv": str(map_add_ready_csv),
        "provisional_category_map_csv": str(map_provisional_csv),
        "add_ready_provisional_category_map_csv": str(map_add_ready_provisional_csv),
        "existing_conflicts_csv": str(conflict_csv),
        "rag_jsonl": str(rag_jsonl),
        "rag_csv": str(rag_csv),
        "metadata_pkl": str(metadata_pkl),
        "metadata_jsonl": str(metadata_jsonl),
        "usable_new_candidate_count": len(usable_rows),
        "unique_usable_new_candidate_count": len(usable_unique_rows),
        "add_ready_new_candidate_count": len(add_ready_rows),
        "existing_conflict_count": len(conflict_rows),
    }
    if args.build_embeddings:
        outputs.update(build_embeddings(args, metadata_rows, combined_metadata))
    return outputs


def build_embeddings(args: argparse.Namespace, new_metadata: list[dict[str, Any]], combined_metadata: list[dict[str, Any]]) -> dict[str, str]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    output_dir = Path(args.output_dir)
    model = SentenceTransformer(args.model_name)
    new_vectors = model.encode(
        [row.get("embedding_text") or row.get("search_text") or row.get("standard_name", "") for row in new_metadata],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    combined_vectors = model.encode(
        [row.get("embedding_text") or row.get("search_text") or row.get("standard_name", "") for row in combined_metadata],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    new_path = output_dir / "new_standard_candidate_vectors.npz"
    combined_path = output_dir / "functional_ingredient_metadata_v2_provisional_vectors.npz"
    np.savez_compressed(new_path, vectors=new_vectors.astype("float32"))
    np.savez_compressed(combined_path, vectors=combined_vectors.astype("float32"))
    return {
        "new_candidate_vectors_npz": str(new_path),
        "provisional_metadata_vectors_npz": str(combined_path),
    }


def apply_results(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    result_jsonl = Path(args.result_jsonl)
    source_rows = {row["ingredient_id"]: row for row in read_csv(Path(args.input_csv)) if row.get("decision") == "new_standard_candidate"}
    request_map = load_request_map(output_dir)
    enriched_by_id: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    with result_jsonl.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            key = ""
            try:
                line_obj = json.loads(line)
                key = readable(line_obj.get("key"))
                if line_obj.get("error"):
                    raise ValueError(json.dumps(line_obj["error"], ensure_ascii=False))
                response = response_from_batch_line(line_obj)
                if not response:
                    raise ValueError("missing response")
                parsed = parse_model_json(extract_response_text(response))
                results = parsed.get("results")
                if not isinstance(results, list):
                    raise ValueError("missing results array")
            except Exception as exc:
                errors.append({"line_number": line_number, "key": key, "error": str(exc), "raw": line[:4000]})
                for ingredient_id in request_map.get(key, []):
                    source = source_rows.get(ingredient_id)
                    if source and ingredient_id not in enriched_by_id:
                        enriched_by_id[ingredient_id] = fallback_row(source, f"batch_parse_error: {exc}")
                continue

            for item in results:
                if not isinstance(item, dict):
                    errors.append({"line_number": line_number, "key": key, "error": "result item is not object", "raw": str(item)[:4000]})
                    continue
                ingredient_id = readable(item.get("id"))
                source = source_rows.get(ingredient_id)
                if not source:
                    errors.append({"line_number": line_number, "key": key, "error": f"unknown id {ingredient_id}", "raw": json.dumps(item, ensure_ascii=False)})
                    continue
                enriched_by_id[ingredient_id] = build_enriched_row(source, item)

    for ingredient_id, source in source_rows.items():
        if ingredient_id not in enriched_by_id:
            enriched_by_id[ingredient_id] = fallback_row(source, "missing_from_batch_result")

    enriched_rows = [enriched_by_id[key] for key in sorted(enriched_by_id)]
    outputs = save_artifacts(args, enriched_rows)
    write_csv(output_dir / "new_standard_enrichment_errors.csv", ["line_number", "key", "error", "raw"], errors)
    summary = {
        "applied_at": datetime.now().isoformat(timespec="seconds"),
        "input_candidate_count": len(source_rows),
        "enriched_count": len(enriched_rows),
        "error_count": len(errors),
        "review_status_counts": dict(Counter(row.get("review_status", "") for row in enriched_rows)),
        "evidence_status_counts": dict(Counter(row.get("evidence_status", "") for row in enriched_rows)),
        "category_counts": dict(Counter(row.get("category_main", "") for row in enriched_rows)),
        "outputs": outputs,
    }
    (output_dir / "new_standard_enrichment_apply_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Enrich ingredient_match_v2 new_standard_candidate rows for provisional RAG DB use.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    prepare_parser.add_argument("--category-map", default=str(DEFAULT_CATEGORY_MAP))
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument("--group-size", type=int, default=10)
    prepare_parser.add_argument("--limit", type=int, default=None)

    run_parser = subparsers.add_parser("run-batch")
    run_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run_parser.add_argument("--health-batch-dir", default=str(DEFAULT_HEALTH_BATCH_DIR))
    run_parser.add_argument("--poll-seconds", type=int, default=30)
    run_parser.add_argument("--submit-retry-seconds", type=int, default=120)
    run_parser.add_argument("--max-submit-attempts", type=int, default=8)
    run_parser.add_argument("--force", action="store_true")

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    apply_parser.add_argument("--category-map", default=str(DEFAULT_CATEGORY_MAP))
    apply_parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    apply_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    apply_parser.add_argument("--result-jsonl", default=str(DEFAULT_OUTPUT_DIR / "new_standard_enrichment_result.jsonl"))
    apply_parser.add_argument("--build-embeddings", action="store_true")
    apply_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-batch":
        run_batch(args)
    elif args.command == "apply":
        apply_results(args)


if __name__ == "__main__":
    main()
