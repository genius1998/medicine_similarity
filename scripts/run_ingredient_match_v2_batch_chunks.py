from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "ingredient_match_v2"
DEFAULT_CATEGORY_MAP = ROOT / "output" / "cache_db_excel_export" / "temp_csv" / "functional_category_map.csv"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
CHUNK_DIR = OUTPUT_DIR / "batch_chunks"
HEALTH_BATCH_DIR = Path(os.environ.get("HEALTH_BATCH_DIR", ROOT / "output" / "batch_jobs"))
PROGRESS_PATH = OUTPUT_DIR / "ingredient_match_v2_batch_progress.json"
FINAL_RESULT_JSONL = OUTPUT_DIR / "ingredient_match_v2_gemini_batch_result.jsonl"
APPLY_SUMMARY = OUTPUT_DIR / "ingredient_match_v2_apply_summary.json"
CATEGORY_MAP = DEFAULT_CATEGORY_MAP

POLL_SECONDS = 30
SUBMIT_RETRY_SECONDS = 120
MAX_SUBMIT_ATTEMPTS = 8


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def write_progress(**kwargs) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **kwargs,
    }
    PROGRESS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(command: list[str], cwd: Path, *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
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


def chunk_paths(index: int) -> dict[str, Path]:
    return {
        "jsonl": CHUNK_DIR / f"ingredient_match_v2_chunk_{index:03d}.jsonl",
        "result": CHUNK_DIR / f"ingredient_match_v2_chunk_{index:03d}_result.jsonl",
        "submit_log": CHUNK_DIR / f"chunk_{index:03d}_submit.log",
        "job": CHUNK_DIR / f"chunk_{index:03d}.job.txt",
    }


def discover_chunks() -> list[Path]:
    chunks = sorted(CHUNK_DIR.glob("ingredient_match_v2_chunk_*.jsonl"))
    return [path for path in chunks if not path.name.endswith("_result.jsonl")]


def split_batch_jsonl(batch_jsonl_path: Path, lines_per_chunk: int) -> list[Path]:
    if not batch_jsonl_path.exists():
        return []
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    existing = discover_chunks()
    if existing:
        return existing
    chunk_index = 0
    line_count = 0
    current = None
    current_path = None
    try:
        with batch_jsonl_path.open("r", encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                if current is None or line_count >= lines_per_chunk:
                    if current is not None:
                        current.close()
                    chunk_index += 1
                    line_count = 0
                    current_path = CHUNK_DIR / f"ingredient_match_v2_chunk_{chunk_index:03d}.jsonl"
                    current = current_path.open("w", encoding="utf-8", newline="\n")
                current.write(line.rstrip("\n") + "\n")
                line_count += 1
    finally:
        if current is not None:
            current.close()
    return discover_chunks()


def extract_job_name(output: str) -> str:
    match = re.search(r"Batch job name:\s*(\S+)", output)
    if not match:
        raise RuntimeError("Batch job name not found in submit output")
    return match.group(1)


def extract_state(output: str) -> str:
    match = re.search(r"(JOB_STATE_[A-Z_]+)", output)
    if not match:
        return "UNKNOWN"
    return match.group(1)


def submit_chunk(index: int, jsonl_path: Path) -> str:
    name = f"ingredient-match-v2-c{index:03d}-20260601"
    paths = chunk_paths(index)
    if paths["job"].exists():
        job_name = paths["job"].read_text(encoding="utf-8").strip()
        if job_name:
            log(f"[{index}] reuse job {job_name}")
            return job_name

    last_output = ""
    for attempt in range(1, MAX_SUBMIT_ATTEMPTS + 1):
        log(f"[{index}] submit attempt {attempt}/{MAX_SUBMIT_ATTEMPTS}")
        proc = run_command(
            [
                "py",
                "-3.12",
                "submit_gemini_batch.py",
                "--jsonl",
                str(jsonl_path),
                "--name",
                name,
            ],
            cwd=HEALTH_BATCH_DIR,
            allow_failure=True,
        )
        last_output = proc.stdout or ""
        paths["submit_log"].write_text(last_output, encoding="utf-8")
        if proc.returncode == 0:
            job_name = extract_job_name(last_output)
            paths["job"].write_text(job_name + "\n", encoding="utf-8")
            return job_name
        if "RESOURCE_EXHAUSTED" not in last_output:
            raise RuntimeError(f"submit failed for chunk {index}")
        log(f"[{index}] quota exhausted; wait {SUBMIT_RETRY_SECONDS}s")
        time.sleep(SUBMIT_RETRY_SECONDS)
    raise RuntimeError(f"submit retry exceeded for chunk {index}: {last_output[-1000:]}")


def check_or_download(job_name: str, result_path: Path, *, download: bool = False) -> str:
    command = ["py", "-3.12", "check_gemini_batch.py", "--job", job_name]
    if download:
        command.extend(["--download-output", str(result_path)])
    proc = run_command(command, cwd=HEALTH_BATCH_DIR)
    return extract_state(proc.stdout or "")


def merge_results(chunks: list[Path]) -> int:
    count = 0
    with FINAL_RESULT_JSONL.open("w", encoding="utf-8", newline="\n") as out:
        for result_path in chunks:
            with result_path.open("r", encoding="utf-8-sig") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line.rstrip("\n") + "\n")
                        count += 1
    return count


def apply_results() -> None:
    run_command(
        [
            "python",
            "scripts\\build_ingredient_match_cache_v2.py",
            "apply",
            "--result-jsonl",
            str(FINAL_RESULT_JSONL),
            "--category-map",
            str(CATEGORY_MAP),
            "--output-dir",
            str(OUTPUT_DIR),
        ],
        cwd=ROOT,
    )


def main() -> None:
    global OUTPUT_DIR, CHUNK_DIR, HEALTH_BATCH_DIR, PROGRESS_PATH, FINAL_RESULT_JSONL, APPLY_SUMMARY, CATEGORY_MAP, POLL_SECONDS
    parser = argparse.ArgumentParser(description="Submit ingredient match v2 batch JSONL in chunks and apply results.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--category-map", default=str(DEFAULT_CATEGORY_MAP))
    parser.add_argument("--health-batch-dir", default=str(HEALTH_BATCH_DIR))
    parser.add_argument("--lines-per-chunk", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = (ROOT / OUTPUT_DIR).resolve()
    CHUNK_DIR = OUTPUT_DIR / "batch_chunks"
    HEALTH_BATCH_DIR = Path(args.health_batch_dir)
    if not HEALTH_BATCH_DIR.is_absolute():
        HEALTH_BATCH_DIR = (ROOT / HEALTH_BATCH_DIR).resolve()
    PROGRESS_PATH = OUTPUT_DIR / "ingredient_match_v2_batch_progress.json"
    FINAL_RESULT_JSONL = OUTPUT_DIR / "ingredient_match_v2_gemini_batch_result.jsonl"
    APPLY_SUMMARY = OUTPUT_DIR / "ingredient_match_v2_apply_summary.json"
    CATEGORY_MAP = Path(args.category_map)
    if not CATEGORY_MAP.is_absolute():
        CATEGORY_MAP = (ROOT / CATEGORY_MAP).resolve()
    POLL_SECONDS = max(5, int(args.poll_seconds))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    chunks = discover_chunks()
    if not chunks:
        chunks = split_batch_jsonl(OUTPUT_DIR / "ingredient_match_v2_gemini_batch.jsonl", max(1, int(args.lines_per_chunk)))
    if not chunks:
        raise SystemExit("no chunk JSONL files found and no batch JSONL to split")
    total = len(chunks)
    log(f"chunk_count={total}")

    completed = 0
    for index, chunk in enumerate(chunks, start=1):
        paths = chunk_paths(index)
        if paths["result"].exists() and paths["result"].stat().st_size > 0:
            completed += 1
            log(f"[{index}/{total}] result already exists")
            write_progress(
                phase="batch_chunks",
                chunk_index=index,
                chunk_count=total,
                chunk_state="already_done",
                completed_chunks=completed,
                percent=round(completed / total * 100, 2),
            )
            continue

        write_progress(
            phase="batch_chunks",
            chunk_index=index,
            chunk_count=total,
            chunk_state="submitting",
            completed_chunks=completed,
            percent=round(completed / total * 100, 2),
        )
        job_name = submit_chunk(index, chunk)
        log(f"[{index}/{total}] job={job_name}")

        while True:
            state = check_or_download(job_name, paths["result"], download=False)
            write_progress(
                phase="batch_chunks",
                chunk_index=index,
                chunk_count=total,
                chunk_state=state,
                completed_chunks=completed,
                current_job=job_name,
                percent=round((completed + 0.5) / total * 100, 2),
            )
            if state == "JOB_STATE_SUCCEEDED":
                check_or_download(job_name, paths["result"], download=True)
                completed += 1
                write_progress(
                    phase="batch_chunks",
                    chunk_index=index,
                    chunk_count=total,
                    chunk_state="downloaded",
                    completed_chunks=completed,
                    current_job=job_name,
                    percent=round(completed / total * 100, 2),
                )
                break
            if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                raise RuntimeError(f"chunk {index} ended with {state}")
            log(f"[{index}/{total}] state={state}; wait {POLL_SECONDS}s")
            time.sleep(POLL_SECONDS)

    result_paths = [chunk_paths(index)["result"] for index in range(1, total + 1)]
    write_progress(phase="merge_results", chunk_count=total, completed_chunks=completed, percent=100)
    merged_lines = merge_results(result_paths)
    log(f"merged_result_lines={merged_lines}")

    write_progress(phase="apply_results", chunk_count=total, completed_chunks=completed, percent=100)
    apply_results()
    summary = json.loads(APPLY_SUMMARY.read_text(encoding="utf-8"))
    write_progress(
        phase="complete",
        chunk_count=total,
        completed_chunks=completed,
        percent=100,
        apply_summary=summary,
    )
    log("complete")


if __name__ == "__main__":
    main()
