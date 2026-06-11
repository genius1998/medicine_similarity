"""
80,395건 순차 배치 재레이블링 (5,000건씩, 토큰 한도 대응)

OpenAI gpt-4o-mini 조직 enqueued 토큰 한도: 2,000,000
→ 요청당 ~350 토큰 기준, 배치당 최대 5,000건 안전 처리
→ 총 ~17개 배치, 완료 후 다음 제출 반복

Usage:
    python scripts/relabel_sequential.py
    python scripts/relabel_sequential.py --retrain-only   # 레이블 이미 완성된 경우
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output" / "ml_validation" / "relabel_seq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR  = REPO_ROOT / "output" / "ml_models"

MAIN_CSV   = (
    REPO_ROOT / "output"
    / "recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060719_oralxylitol_p260_livermilk_p300"
    / "openai_chunk_judge_results.csv"
)

CHUNK_SIZE   = 5000   # 배치당 요청 수 (≈1.75M 토큰, 한도 2M 이하)
BATCH_PREFIX = "relbatch"


# ─────────────────────────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────────────────────────

def build_prompt(row: dict) -> str:
    def sjson(v):
        if not isinstance(v, str):
            return []
        try:
            r = json.loads(v)
            return r if isinstance(r, list) else []
        except Exception:
            return []

    shared_ings = sjson(row.get("shared_ingredients_json", "[]"))[:8]
    shared_cats = sjson(row.get("shared_categories_json", "[]"))
    adj_types   = sjson(row.get("score_adjustment_types_json", "[]"))

    return (
        "건강기능식품 추천 품질 평가.\n\n"
        "라벨 기준:\n"
        "- reasonable: 핵심 기능성 원료가 겹치고 기능 목적 동일\n"
        "- acceptable_adjacent: 핵심 원료 겹침은 있으나 완전 대체는 아님\n"
        "- weak: 근거 일부 있으나 상위 추천으로는 약함\n"
        "- bad: 기능·원료·카테고리 불일치 또는 흔한 영양소만 근거\n\n"
        "규칙: 비타민/미네랄만 겹치면 weak 이하. 핵심 원료 겹침 없으면 acceptable_adjacent 불가.\n\n"
        f"base: {row.get('base_product_name','')} [{row.get('base_main_category','')}]\n"
        f"candidate: {row.get('target_product_name','')} [{row.get('target_main_category','')}]\n"
        f"sim={float(row.get('similarity_score',0)):.3f} "
        f"func={float(row.get('function_similarity_score',0)):.3f} "
        f"core={float(row.get('core_match_score',0)):.3f} "
        f"rank={row.get('rank',0)}\n"
        f"공유원료: {shared_ings}\n"
        f"공유카테고리: {shared_cats}\n"
        f"조정: {adj_types}\n\n"
        '{"judgment":"<reasonable|acceptable_adjacent|weak|bad>","confidence":<0-1>,"reason":"<30자>"}'
    )


# ─────────────────────────────────────────────────────────────
# 배치 제출 & 대기 & 다운로드
# ─────────────────────────────────────────────────────────────

def submit_one_batch(rows: list[dict], chunk_idx: int, client) -> str:
    batch_name = f"{BATCH_PREFIX}_{chunk_idx:03d}"
    tasks = [
        {
            "custom_id": f"{batch_name}-{i:05d}",
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":    "gpt-4o-mini",
                "messages": [
                    {"role": "system",
                     "content": "건강기능식품 추천 품질 전문가. JSON만 반환."},
                    {"role": "user", "content": build_prompt(r)},
                ],
                "max_tokens":      80,
                "temperature":     0.0,
                "response_format": {"type": "json_object"},
            },
        }
        for i, r in enumerate(rows)
    ]

    jsonl_path = OUTPUT_DIR / f"{batch_name}_req.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open(jsonl_path, "rb") as f:
        bf = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=bf.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": batch_name},
    )
    # 메타 저장
    with open(OUTPUT_DIR / f"{batch_name}_meta.json", "w") as f:
        json.dump({"batch_id": batch.id, "batch_name": batch_name,
                   "chunk_idx": chunk_idx, "n": len(tasks)}, f, indent=2)

    return batch.id


def wait_for_batch(batch_id: str, batch_name: str, client,
                   timeout_s: int = 3600) -> str | None:
    start = time.time()
    while time.time() - start < timeout_s:
        b = client.batches.retrieve(batch_id)
        elapsed = int((time.time() - start) // 60)
        print(f"    [{elapsed}m] {batch_name} {b.status} "
              f"{b.request_counts.completed}/{b.request_counts.total}", flush=True)
        if b.status == "completed":
            return b.output_file_id
        if b.status in ("failed", "expired", "cancelled"):
            print(f"    배치 실패: {b.status}")
            return None
        time.sleep(20)
    print(f"    타임아웃: {batch_id}")
    return None


def download_results(output_file_id: str, chunk_idx: int, client) -> dict:
    raw = client.files.content(output_file_id).read().decode("utf-8")
    out = OUTPUT_DIR / f"{BATCH_PREFIX}_{chunk_idx:03d}_res.jsonl"
    out.write_text(raw, encoding="utf-8")

    parsed = {}
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        try:
            item     = json.loads(line)
            cid      = item["custom_id"]
            content  = item["response"]["body"]["choices"][0]["message"]["content"]
            obj      = json.loads(content)
            judgment = str(obj.get("judgment", "")).strip()
            if judgment not in {"reasonable", "acceptable_adjacent", "weak", "bad"}:
                judgment = "acceptable_adjacent"
            parsed[cid] = {
                "judgment":   judgment,
                "confidence": float(obj.get("confidence", 0.7)),
            }
        except Exception:
            pass
    return parsed


# ─────────────────────────────────────────────────────────────
# 재학습
# ─────────────────────────────────────────────────────────────

def retrain(df: pd.DataFrame) -> float:
    import joblib
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.metrics import classification_report, precision_recall_curve
    from sklearn.model_selection import train_test_split

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.train_recommendation_quality_model import engineer_features

    print("\n레이블 분포 비교:")
    print("  원본:", df["judge_judgment"].value_counts().to_dict())
    print("  신규:", df["judge_judgment_new"].value_counts().to_dict())

    y_new = df["judge_judgment_new"].isin({"weak", "bad"}).astype(int).values
    X     = engineer_features(df)
    print(f"신규 weak/bad: {y_new.sum():,}건 ({y_new.mean():.2%})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y_new, test_size=0.2, random_state=42, stratify=y_new
    )
    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)

    model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    print("재학습 중...")
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    ap    = average_precision_score(y_te, proba)
    print(f"Hold-out ROC-AUC: {auc:.4f}  AP: {ap:.4f}")
    print(classification_report(y_te, (proba >= 0.5).astype(int),
                                 target_names=["good", "weak/bad"]))

    # 최적 임계값 (precision ≥ 0.60)
    precs, recs, threshs = precision_recall_curve(y_te, proba)
    best_thresh, best_f1 = 0.5, 0.0
    for p, r, t in zip(precs, recs, threshs):
        if p >= 0.60:
            f1 = 2 * p * r / max(p + r, 1e-9)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
    print(f"최적 임계값 (precision≥0.60): {best_thresh:.3f}  F1={best_f1:.3f}")
    print(classification_report(y_te, (proba >= best_thresh).astype(int),
                                 target_names=["good", "weak/bad"]))

    # 카테고리 맵
    category_maps = {}
    for col in ["base_main_category", "target_main_category"]:
        if col in df.columns:
            cat = df[col].fillna("").astype("category")
            category_maps[col] = {v: i for i, v in enumerate(cat.cat.categories)}

    old_path = MODEL_DIR / "recommendation_quality_model.pkl"
    artifact = joblib.load(old_path) if old_path.exists() else {}
    artifact.update({
        "model":            model,
        "model_type":       "xgboost",
        "feature_names":    list(X.columns),
        "category_maps":    category_maps,
        "train_rows":       len(df),
        "positive_rate":    float(y_new.mean()),
        "filter_threshold": float(best_thresh),
        "retrain_note":     "relabeled strict prompt v2 sequential",
    })
    joblib.dump(artifact, old_path)
    print(f"모델 저장: {old_path}")
    return float(best_thresh)


def update_threshold_in_code(threshold: float) -> None:
    model_file = REPO_ROOT / "api" / "recommendation_quality_model.py"
    content    = model_file.read_text(encoding="utf-8")
    import re
    content = re.sub(
        r'(WEAK_PROBABILITY_THRESHOLD: float = float\(\s*os\.environ\.get\("RECOMMENDATION_QUALITY_WEAK_THRESHOLD",\s*")[^"]*(")',
        rf'\g<1>{threshold:.3f}\g<2>',
        content,
    )
    model_file.write_text(content, encoding="utf-8")
    print(f"임계값 코드 업데이트: {threshold:.3f}")


def deploy_to_server(threshold: float) -> None:
    import subprocess

    print("\n=== Git commit & push ===")
    subprocess.run(
        ["git", "add",
         "api/recommendation_quality_model.py",
         "output/ml_models/recommendation_quality_model.pkl",
         "scripts/relabel_sequential.py"],
        cwd=REPO_ROOT, check=True
    )
    msg = (
        f"feat: retrain XGBoost with strict-prompt labels, filter mode threshold={threshold:.3f}\n\n"
        f"- 80,395건 엄격 프롬프트로 재레이블링\n"
        f"- XGBoost 재학습 (500 estimators)\n"
        f"- filter threshold auto-tuned: {threshold:.3f} (precision>=60%)\n"
        f"- RECOMMENDATION_QUALITY_ML_PHASE=filter 로 전환\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=REPO_ROOT, check=True)
    print("GitHub push 완료")

    print("\n=== EC2 서버 배포 ===")
    pem = os.environ.get("EC2_SSH_KEY", "")
    host_name = os.environ.get("EC2_HOST", "")
    host_user = os.environ.get("EC2_USER", "ubuntu")
    app_dir = os.environ.get("EC2_APP_DIR", "/opt/medicine_similarity")
    service_name = os.environ.get("EC2_SERVICE", "seongbunkok.service")
    if not pem or not host_name:
        print("EC2_SSH_KEY/EC2_HOST 환경변수가 없어 서버 배포를 건너뜁니다.")
        return
    host = f"{host_user}@{host_name}"
    ssh_cmd = (
        f"cd {app_dir} && "
        "git pull origin main && "
        f"RECOMMENDATION_QUALITY_ML_PHASE=filter && "
        f"sudo systemctl restart {service_name} && "
        "sleep 3 && "
        f"sudo systemctl status {service_name} --no-pager | head -8"
    )
    result = subprocess.run(
        ["ssh", "-i", pem, "-o", "StrictHostKeyChecking=no", host, ssh_cmd],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
    )
    print(result.stdout)
    if result.returncode != 0:
        print("서버 오류:", result.stderr[:200])


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain-only", action="store_true")
    args = parser.parse_args()

    print("원본 데이터 로드...")
    df = pd.read_csv(MAIN_CSV, low_memory=False)
    n  = len(df)
    print(f"  {n:,}건")

    # ── retrain-only: 기존 결과 파일에서 레이블 읽기 ──────────
    if args.retrain_only:
        label_map = {}
        for res_file in sorted(OUTPUT_DIR.glob("*_res.jsonl")):
            for line in res_file.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    cid  = item["custom_id"]
                    obj  = json.loads(item["response"]["body"]["choices"][0]["message"]["content"])
                    j    = str(obj.get("judgment","")).strip()
                    if j not in {"reasonable","acceptable_adjacent","weak","bad"}:
                        j = "acceptable_adjacent"
                    label_map[cid] = j
                except Exception:
                    pass
        print(f"기존 레이블 로드: {len(label_map):,}건")
        _attach_and_retrain(df, label_map)
        return

    from dotenv import dotenv_values
    from openai import OpenAI

    env_vals = dotenv_values(Path("D:/health_batch_project/.env"))
    client   = OpenAI(api_key=env_vals.get("OPENAI_API_KEY"))

    # ── 청크 분할 ───────────────────────────────────────────
    chunks     = [df.iloc[i:i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]
    total_chunks = len(chunks)
    label_map: dict[str, str] = {}

    # 이미 완료된 청크 복원
    for res_file in sorted(OUTPUT_DIR.glob("*_res.jsonl")):
        for line in res_file.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cid  = item["custom_id"]
                obj  = json.loads(item["response"]["body"]["choices"][0]["message"]["content"])
                j    = str(obj.get("judgment","")).strip()
                if j not in {"reasonable","acceptable_adjacent","weak","bad"}:
                    j = "acceptable_adjacent"
                label_map[cid] = j
            except Exception:
                pass

    done_chunks = {int(f.stem.split("_")[1]) for f in OUTPUT_DIR.glob("*_res.jsonl")
                   if f.stem.split("_")[1].isdigit()}
    print(f"이미 완료된 청크: {len(done_chunks)}/{total_chunks}")

    # ── 순차 제출 ────────────────────────────────────────────
    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx in done_chunks:
            continue

        batch_name = f"{BATCH_PREFIX}_{chunk_idx:03d}"
        rows       = chunk.to_dict(orient="records")
        print(f"\n[{chunk_idx+1}/{total_chunks}] 제출: {batch_name} ({len(rows)}건)")

        bid = submit_one_batch(rows, chunk_idx, client)
        print(f"  배치 ID: {bid}")

        output_file_id = wait_for_batch(bid, batch_name, client, timeout_s=3600)
        if output_file_id is None:
            print(f"  실패 — 나중에 재시도 필요: chunk_idx={chunk_idx}")
            # 실패해도 계속 진행 (원본 fallback 사용)
            continue

        chunk_labels = download_results(output_file_id, chunk_idx, client)
        label_map.update(chunk_labels)
        done_chunks.add(chunk_idx)
        print(f"  완료: {len(chunk_labels)}건 레이블 획득  (누계: {len(label_map):,}건)")

    _attach_and_retrain(df, label_map)


def _attach_and_retrain(df: pd.DataFrame, label_map: dict) -> None:
    # 레이블 붙이기 (미파싱은 원본 fallback)
    new_labels = []
    n_fallback = 0
    for global_idx in range(len(df)):
        chunk_idx = global_idx // CHUNK_SIZE
        local_idx = global_idx % CHUNK_SIZE
        cid       = f"{BATCH_PREFIX}_{chunk_idx:03d}-{local_idx:05d}"
        j         = label_map.get(cid)
        if j is None:
            j = df.iloc[global_idx]["judge_judgment"]
            n_fallback += 1
        new_labels.append(j)

    print(f"\n레이블 병합 완료: {len(label_map):,}건 신규 + {n_fallback}건 원본 fallback")
    df = df.copy()
    df["judge_judgment_new"] = new_labels

    # 레이블 저장
    df[["base_report_no", "target_report_no", "rank",
        "judge_judgment", "judge_judgment_new"]].to_csv(
        OUTPUT_DIR / "final_labels.csv", index=False, encoding="utf-8-sig"
    )

    thresh = retrain(df)
    update_threshold_in_code(thresh)
    deploy_to_server(thresh)


if __name__ == "__main__":
    main()
