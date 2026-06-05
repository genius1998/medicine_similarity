# Recommendation Quality Validation

## Current Decision

Keep `semantic_weighted_jaccard_v2_9` without adding another cap or gate.

The latest OpenAI `gpt-5-nano` judge validation passes the quality gate:

| Metric | Value |
| --- | ---: |
| Labels | 4,227 / 4,227 |
| Products | 802 |
| Batch requests | 575 |
| Weak/bad count | 315 |
| Weak/bad rate | 7.45% |
| High-score weak/bad count | 49 |
| High-score weak/bad rate | 1.16% |
| Actionable pattern count | 0 |

Gate decision:

```text
pass_continue_validation_without_algorithm_change
```

The candidate cap patterns are not actionable because they affect too many reasonable or acceptable-adjacent recommendations relative to the weak/bad cases they catch.

## Validation Inputs

Current merged validation output:

```text
output/recommendation_quality_judge_v2_9_openai_validation_current_plus_holdout2
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
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_chunk_*_retry_*" `
  --output-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_holdout2 `
  --high-score-threshold 0.65
```

Write the Markdown report:

```powershell
python scripts\recommendation_quality_judge_batch.py validation-report `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_holdout2 `
  --top-categories 10 `
  --top-patterns 7
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
  --require-no-active `
  --poll-seconds 60 `
  --timeout-seconds 7200 `
  --high-score-threshold 0.65
```

`openai-run` reuses the existing job file by default. `--require-no-active` only blocks a new submission when there is no reusable job file. Use `--force` only when intentionally submitting a replacement job.

The `plan-next-sample` output also includes `openai_run_command_powershell` so the next targeted sample can be prepared first and then run with the same safety preflight.

## Decision Rule

Continue with the current algorithm when all are true:

- Overall weak/bad rate is at or below `10%`.
- High-score weak/bad rate is at or below `2%`.
- No candidate pattern has high weak/bad concentration with low non-weak blast radius.

Only consider an algorithm change when a pattern meets all of these:

- Weak/bad count is at least `5`.
- Weak/bad rate is at least `50%`.
- Non-weak affected count is at most `5`.

If the aggregate rates fail but no low-blast-radius pattern appears, collect more targeted samples instead of adding a broad cap.
