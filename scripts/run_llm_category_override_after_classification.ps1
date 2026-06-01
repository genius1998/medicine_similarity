param(
    [int]$ClassificationPid = 0,
    [string]$ClassificationCsv = "output/llm_main_sub_category_classification_all.csv",
    [double]$OverrideMinConfidence = 0.9,
    [string]$OverrideDecisions = "auto_override_candidate"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$OutputDir = Join-Path $RepoRoot "output"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$WatcherLog = Join-Path $OutputDir "llm_main_sub_category_rebuild_watcher.log"
$BuildLog = Join-Path $OutputDir "build_product_function_profile_main_sub_override_all.log"
$BackupDir = Join-Path $OutputDir "llm_main_sub_category_profile_backups"

function Write-RunLog {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $Line | Tee-Object -FilePath $WatcherLog -Append
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

Write-RunLog "watcher started; classification_pid=$ClassificationPid; csv=$ClassificationCsv"

if ($ClassificationPid -gt 0) {
    $Process = Get-Process -Id $ClassificationPid -ErrorAction SilentlyContinue
    if ($Process) {
        Write-RunLog "waiting for classification process $ClassificationPid"
        Wait-Process -Id $ClassificationPid
    }
}

$ValidationScript = @'
import json
import sys
from pathlib import Path

import pandas as pd

from scripts.classify_product_main_categories_llm import load_joined_rows

csv_path = Path(sys.argv[1])
if not csv_path.exists():
    print(json.dumps({"ok": False, "error": f"classification csv not found: {csv_path}"}, ensure_ascii=False))
    raise SystemExit(2)

classified_df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False).fillna("")
if "report_no" not in classified_df.columns:
    print(json.dumps({"ok": False, "error": "classification csv missing report_no"}, ensure_ascii=False))
    raise SystemExit(2)

expected_rows = load_joined_rows(
    Path("output/product_function_profile.csv"),
    Path("output/cache_db_excel_export/temp_csv/catalog_product_master.csv"),
)
expected = {str(row.get("report_no", "")).strip() for row in expected_rows if str(row.get("report_no", "")).strip()}
actual = set(classified_df["report_no"].astype(str).str.strip())
missing = sorted(expected - actual)
summary = {
    "ok": not missing,
    "expected_unique_report_nos": len(expected),
    "actual_unique_report_nos": len(actual),
    "rows": int(len(classified_df)),
    "missing_count": len(missing),
    "sample_missing": missing[:10],
}
print(json.dumps(summary, ensure_ascii=False))
raise SystemExit(0 if not missing else 2)
'@

$ValidationOutput = $ValidationScript | python - $ClassificationCsv
Write-RunLog "validation: $ValidationOutput"
if ($LASTEXITCODE -ne 0) {
    Write-RunLog "validation failed; rebuild skipped"
    exit $LASTEXITCODE
}

$ProfileCsv = Join-Path $OutputDir "product_function_profile.csv"
$ProfileJsonl = Join-Path $OutputDir "product_function_profile.jsonl"
$TempProfileCsv = Join-Path $OutputDir "cache_db_excel_export\temp_csv\product_function_profile.csv"

foreach ($Path in @($ProfileCsv, $ProfileJsonl, $TempProfileCsv)) {
    if (Test-Path $Path) {
        $Name = Split-Path -Leaf $Path
        Copy-Item $Path (Join-Path $BackupDir "$($Name)_before_main_sub_override_$RunStamp")
    }
}

Write-RunLog "profile backups written to $BackupDir"
Write-RunLog "starting product function profile rebuild"

$BuildArgs = @(
    "scripts/build_product_function_profile.py",
    "--main-category-overrides", $ClassificationCsv,
    "--override-min-confidence", "$OverrideMinConfidence",
    "--override-decisions", $OverrideDecisions
)

& python @BuildArgs 2>&1 | Tee-Object -FilePath $BuildLog
if ($LASTEXITCODE -ne 0) {
    Write-RunLog "build failed; see $BuildLog"
    exit $LASTEXITCODE
}

if (Test-Path $ProfileCsv) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $TempProfileCsv) | Out-Null
    Copy-Item $ProfileCsv $TempProfileCsv -Force
    Write-RunLog "copied rebuilt profile csv to temp export path"
}

Write-RunLog "rebuild completed"
