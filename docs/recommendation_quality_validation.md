# Recommendation Quality Validation

## Current Decision

Keep `semantic_weighted_jaccard_v2_9` without adding another cap or gate.

The latest OpenAI `gpt-5-nano` judge validation passes the quality gate:

| Metric | Value |
| --- | ---: |
| Labels | 48,873 / 48,873 |
| Products | 9,494 |
| Batch requests | 6,823 |
| Weak/bad count | 4,156 |
| Weak/bad rate | 8.50% |
| High-score weak/bad count | 641 |
| High-score weak/bad rate | 1.31% |
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

The targeted p50 sample alone pushed the aggregate weak/bad rate just over the review gate, but it still had no low-blast-radius pattern candidate. Neutral all-category p10, p15, p20, p25, p30, p35, and p40 holdouts were added next. The p40 holdout was split into two OpenAI Batch jobs because the full prepared JSONL exceeded the current `gpt-5-nano` 2M enqueued-token limit. Part 1 had a `4.76%` weak/bad rate, and part 2 had a `10.32%` weak/bad rate, but neither found an actionable low-blast-radius pattern.

Two additional targeted samples were then run against the highest weak-rate categories: 영양보충, 여성 건강, 피로개선, 수면/긴장완화, 혈당, 혈중지질, 혈압, and 체지방. These targeted samples had elevated standalone weak/bad rates of `15.01%` and `13.60%`, as expected from the selected categories, but their high-score weak/bad rates stayed at `1.13%` and `1.65%` and no actionable low-blast-radius pattern was found.

A neutral all-category p45 holdout was added after that and split into three OpenAI Batch jobs to stay under the `gpt-5-nano` 2M enqueued-token limit. The three parts had standalone weak/bad rates of `4.16%`, `8.50%`, and `10.09%`; part 3 slightly exceeded the standalone weak-rate gate, but its high-score weak/bad rate was `1.32%` and it also found no actionable low-blast-radius pattern.

A neutral all-category p50 holdout was then added and split into four OpenAI Batch jobs. The four parts had standalone weak/bad rates of `5.36%`, `7.04%`, `10.91%`, and `11.42%`. Parts 3 and 4 exceeded the standalone weak-rate gate, but the merged validation remains inside the overall gate and still has no actionable low-blast-radius pattern. After retrying malformed judge outputs from earlier holdouts, the merged validation has complete label coverage.

Latest high-score weak diagnostics also support keeping the current algorithm:

- High-score rows: `38,232`
- High-score weak/bad rows: `640`
- Within-high-score weak/bad rate: `1.67%`
- Overall high-score weak/bad rate: `1.31%`
- `function_similarity < 0.40` would catch `263` weak/bad rows but also affect `4,435` non-weak rows.
- `not_same_primary` would catch `156` weak/bad rows but also affect `3,926` non-weak rows.
- `same_primary_set` appears in `484` high-score weak/bad rows, so primary-set equality alone is not a reliable accept signal or reject signal.

## Validation Inputs

Current merged validation output:

```text
output/recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50
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
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606064_p30
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606064_p10
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606065_p30
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606066_p40
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606067_p50
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606067_p10
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606068_p15
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606069_p20
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060610_p25
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060611_p30
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060612_p35
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060613_p40_part1
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060613_p40_part2
output/recommendation_quality_judge_v2_9_openai_targeted_after_p40_seed2026060614_p35
output/recommendation_quality_judge_v2_9_openai_targeted_after_p40_seed2026060615_p35
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060616_p45_part1
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060616_p45_part2
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060616_p45_part3
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part1
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part2
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part3
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part4
```

Retry replacements:

```text
output/recommendation_quality_judge_v2_9_openai_chunk_*_retry_*
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606066_p40_retry_*
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606067_p50_retry_*
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606068_p15_retry_*
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060611_p30_retry_*
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
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606064_p30" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606064_p10" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606065_p30" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606066_p40" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606067_p50" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606067_p10" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606068_p15" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606069_p20" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060610_p25" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060611_p30" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060612_p35" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060613_p40_part1" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060613_p40_part2" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_after_p40_seed2026060614_p35" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_after_p40_seed2026060615_p35" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060616_p45_part1" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060616_p45_part2" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060616_p45_part3" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part1" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part2" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part3" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060617_p50_part4" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_chunk_*_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606066_p40_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606067_p50_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606068_p15_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060611_p30_retry_*" `
  --output-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50 `
  --high-score-threshold 0.65
```

This workflow also writes `high_score_weak_diagnostics.json` from the generated pattern features and `validation_status.json` from the quality gate plus diagnostics.

Write the Markdown report:

```powershell
python scripts\recommendation_quality_judge_batch.py validation-report `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50 `
  --top-categories 10 `
  --top-patterns 7
```

The report includes `high_score_weak_diagnostics.json` automatically when that file exists. To regenerate only the diagnostic JSON:

```powershell
python scripts\recommendation_quality_judge_batch.py diagnose-high-score-weak `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50 `
  --high-score-threshold 0.65
```

Regenerate only the stop/continue status:

```powershell
python scripts\recommendation_quality_judge_batch.py validation-status `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50
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
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50\validation_status.json `
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
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50\validation_status.json `
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
  --summary-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50\openai_chunk_judge_summary.json `
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060617_p50\validation_status.json `
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
