# Recommendation Quality Validation

## Current Decision

Keep `semantic_weighted_jaccard_v2_9` without adding another cap or gate.

The latest OpenAI `gpt-5-nano` judge validation passes the quality gate:

| Metric | Value |
| --- | ---: |
| Labels | 8,041 / 8,041 |
| Products | 1,504 |
| Batch requests | 1,109 |
| Weak/bad count | 639 |
| Weak/bad rate | 7.95% |
| High-score weak/bad count | 92 |
| High-score weak/bad rate | 1.14% |
| Actionable pattern count | 0 |

Gate decision:

```text
pass_continue_validation_without_algorithm_change
```

Recommended next action:

```text
stop_sampling_keep_current_algorithm
```

The candidate cap patterns are not actionable because they affect too many reasonable or acceptable-adjacent recommendations relative to the weak/bad cases they catch.

Latest high-score weak diagnostics also support keeping the current algorithm:

- High-score rows: `6,262`
- High-score weak/bad rows: `92`
- Within-high-score weak/bad rate: `1.47%`
- Overall high-score weak/bad rate: `1.14%`
- `function_similarity < 0.40` would catch `38` weak/bad rows but also affect `921` non-weak rows.
- `not_same_primary` would catch `31` weak/bad rows but also affect `749` non-weak rows.
- `same_primary_set` appears in `61` high-score weak/bad rows, so primary-set equality alone is not a reliable accept signal or reject signal.

## Validation Inputs

Current merged validation output:

```text
output/recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063
```

Included samples:

```text
output/recommendation_quality_judge_v2_8_openai_chunk_*_*
output/recommendation_quality_judge_v2_9_openai_chunk_*_*
output/recommendation_quality_judge_v2_9_openai_seed20260605_p5
output/recommendation_quality_judge_v2_9_openai_targeted_highweak_seed20260605
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606051_p5
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606052
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606053
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606054_p5
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed20260606
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606061
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606062_p5
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606063_p25
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606063_p10
```

Retry replacements:

```text
output/recommendation_quality_judge_v2_9_openai_chunk_*_retry_*
```

## Reproduce Summary

Run the full validation workflow:

```powershell
python scripts\recommendation_quality_judge_batch.py validate-results `
  --parts-glob "output\recommendation_quality_judge_v2_8_openai_chunk_*_*" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_chunk_*_*" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_seed20260605_p5" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_highweak_seed20260605" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606051_p5" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606052" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606053" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606054_p5" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed20260606" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606061" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606062_p5" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606063_p25" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606063_p10" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_chunk_*_retry_*" `
  --output-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063 `
  --high-score-threshold 0.65
```

This workflow also writes `high_score_weak_diagnostics.json` from the generated pattern features and `validation_status.json` from the quality gate plus diagnostics.

Write the Markdown report:

```powershell
python scripts\recommendation_quality_judge_batch.py validation-report `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063 `
  --top-categories 10 `
  --top-patterns 7
```

The report includes `high_score_weak_diagnostics.json` automatically when that file exists. To regenerate only the diagnostic JSON:

```powershell
python scripts\recommendation_quality_judge_batch.py diagnose-high-score-weak `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063 `
  --high-score-threshold 0.65
```

Regenerate only the stop/continue status:

```powershell
python scripts\recommendation_quality_judge_batch.py validation-status `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063
```

## OpenAI Batch Safety Check

Before submitting another OpenAI Batch job, check whether any active jobs are already running:

```powershell
python scripts\recommendation_quality_judge_batch.py openai-list `
  --env-path D:\health_batch_project\.env `
  --active-only `
  --limit 20
```

Treat these statuses as active:

```text
validating
in_progress
finalizing
cancelling
```

Do not resubmit the same validation sample while an active job is still present. The `openai-submit` command also reuses the existing job file by default unless `--force` is explicitly provided.

When using `openai-submit` directly, pass `--require-no-active` and the current `validation_status.json` so it refuses to create a new Batch job while another one is still validating, running, finalizing, or cancelling, and also refuses new submissions after a stop decision:

```powershell
python scripts\recommendation_quality_judge_batch.py openai-submit `
  --output-dir output\recommendation_quality_judge_v2_9_openai_targeted_next_seedYYYYMMDD `
  --env-path D:\health_batch_project\.env `
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063\validation_status.json `
  --require-no-active
```

Existing job files are still reused without checking active jobs. Job files must contain a non-empty job id; an empty job file is treated as an error instead of being reused.

After submitting a job, wait for completion and download outputs with:

```powershell
python scripts\recommendation_quality_judge_batch.py openai-check `
  --output-dir output\recommendation_quality_judge_v2_9_openai_targeted_next_seedYYYYMMDD `
  --env-path D:\health_batch_project\.env `
  --watch `
  --poll-seconds 60 `
  --timeout-seconds 7200 `
  --download `
  --download-errors
```

Then finalize the downloaded result locally:

```powershell
python scripts\recommendation_quality_judge_batch.py openai-finalize `
  --output-dir output\recommendation_quality_judge_v2_9_openai_targeted_next_seedYYYYMMDD `
  --high-score-threshold 0.65
```

For a prepared output directory, the submit/watch/download/finalize sequence can be run as one command:

```powershell
python scripts\recommendation_quality_judge_batch.py openai-run `
  --output-dir output\recommendation_quality_judge_v2_9_openai_targeted_next_seedYYYYMMDD `
  --env-path D:\health_batch_project\.env `
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063\validation_status.json `
  --require-no-active `
  --poll-seconds 60 `
  --timeout-seconds 7200 `
  --high-score-threshold 0.65
```

`openai-run` reuses the existing job file by default. `--require-no-active` only blocks a new submission when there is no reusable job file. Use `--force` only when intentionally submitting a replacement job.

When `--validation-status-json` points to a status whose `next_action` is `stop_sampling_keep_current_algorithm`, `openai-submit` and `openai-run` refuse to submit a new OpenAI Batch job. Existing job files can still be reused for download/finalization, even if the original JSONL is no longer present. Use `--allow-after-validation-stop` only when intentionally overriding the stop decision.

The `plan-next-sample` output also includes `openai_run_command_powershell` so the next targeted sample can be prepared first and then run with the same safety preflight.

When a validation status is available, pass it to `plan-next-sample` so a stop decision suppresses new sample commands:

```powershell
python scripts\recommendation_quality_judge_batch.py plan-next-sample `
  --summary-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063\openai_chunk_judge_summary.json `
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606063\validation_status.json `
  --output-dir output\recommendation_quality_judge_v2_9_openai_next_sample_plan_after_stop
```

## Decision Rule

The quality gate requires at least `50` judge labels by default. Smaller smoke runs are useful for checking OpenAI Batch submission, download, and finalize wiring, but they should not be treated as an accept/reject decision for the recommendation algorithm.

Continue with the current algorithm when all are true:

- Judge label count is at least `50`.
- Overall weak/bad rate is at or below `10%`.
- High-score weak/bad rate is at or below `2%`.
- No candidate pattern has high weak/bad concentration with low non-weak blast radius.

Stop routine sampling and keep the current algorithm when all are true:

- Quality gate decision is `pass_continue_validation_without_algorithm_change`.
- Judge label count is at least `5,000`.
- High-score diagnostics find no low-blast-radius condition candidate.

Only consider an algorithm change when a pattern meets all of these:

- Weak/bad count is at least `5`.
- Weak/bad rate is at least `50%`.
- Non-weak affected count is at most `5`.

If the aggregate rates fail but no low-blast-radius pattern appears, collect more targeted samples instead of adding a broad cap.

If the label count is below `50`, collect a larger targeted sample before interpreting aggregate weak/bad rates.
