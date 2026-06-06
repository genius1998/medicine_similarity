# Recommendation Quality Validation

## Current Decision

Keep `semantic_weighted_jaccard_v2_9` without adding another cap or gate.

The latest OpenAI `gpt-5-nano` judge validation passes the quality gate:

| Metric | Value |
| --- | ---: |
| Labels | 63,972 / 63,972 |
| Products | 12,828 |
| Batch requests | 8,988 |
| Weak/bad count | 5,450 |
| Weak/bad rate | 8.52% |
| High-score weak/bad count | 854 |
| High-score weak/bad rate | 1.33% |
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

One more targeted p45 sample was run after p50 against the highest weak-rate categories: 여성 건강, 영양보충, 수면/긴장완화, 피로개선, 혈중지질, 혈압, 혈당, and 피부 건강. This targeted sample had a standalone weak/bad rate of `12.02%`, as expected from the selected high-risk categories, but its high-score weak/bad rate stayed within the gate at `1.84%` and it found no actionable low-blast-radius pattern. The merged validation still passes the quality gate.

A follow-up targeted p50 sample was then run against the same high-weak categories with a new seed. This sample had a standalone weak/bad rate of `13.75%` and two bad labels, but its high-score weak/bad rate stayed low at `1.25%` and no actionable low-blast-radius pattern was found. The merged validation continues to pass the quality gate, so adding another broad cap would still overfilter many reasonable or acceptable-adjacent recommendations.

A mid-coverage targeted p50 sample was then run against less-recently stressed categories: 체지방, 운동/근력, 기억력, 항산화, 눈 건강, 간 건강, 장 건강, and 혈행. It had a standalone weak/bad rate of `6.35%` and high-score weak/bad rate of `1.30%`, with no actionable low-blast-radius pattern. One malformed OpenAI response in this sample hit `max_output_tokens`; the affected base product was retried separately and the merged validation again has complete label coverage.

A low-risk/control p50 sample was then run against 관절/연골, 구강 건강, 남성 건강, 면역, 뼈 건강, 기타, 운동/근력, and 영양보충. It had a standalone weak/bad rate of `3.75%` and high-score weak/bad rate of `1.78%`, with complete label coverage and no actionable low-blast-radius pattern. This supports the current algorithm from the overfiltering side as well: broad caps would remove many reasonable or acceptable-adjacent rows in these lower-risk categories.

An additional neutral all-category p20 holdout with seed `202606075` was run as a moderate-size continuation sample. It covered `447` products, produced `316` OpenAI Batch requests, and had a standalone weak/bad rate of `7.10%` and high-score weak/bad rate of `1.01%`. One base product was retried to fix a single missing judge label, after which the merged validation again had complete label coverage and still found no actionable low-blast-radius pattern.

A targeted high-weak p35 sample with seed `202606076` was then run against 여성 건강, 수면/긴장완화, 혈중지질, 피로개선, 혈당, 혈압, 체지방, and 피부 건강. It covered `280` products and produced `206` OpenAI Batch requests. Its standalone weak/bad rate was elevated at `14.46%`, as expected for the selected high-weak categories, but the high-score weak/bad rate stayed within the gate at `1.56%`. One duplicate-label response was retried for complete coverage. The merged validation still passes with no actionable low-blast-radius pattern, so no algorithm cap or gate was added.

A low-coverage/high-weak p40 sample with seed `202606077` was then run against 영양보충, 기타, 운동/근력, 여성 건강, 혈압, and 피부 건강. It covered `240` products and produced `119` OpenAI Batch requests. Its standalone weak/bad rate was `11.74%` and standalone high-score weak/bad rate was `2.35%`, slightly over the standalone gate, but the `14` high-score weak/bad rows were spread across categories, ranks, and function-similarity buckets. The merged validation still passes, and the candidate patterns still affect hundreds or thousands of non-weak recommendations, so no algorithm cap or gate was added.

A follow-up low-coverage/high-weak p60 sample with seed `202606078` used the same categories with a larger sample. It covered `360` products and produced `169` OpenAI Batch requests. Its standalone weak/bad rate was `12.18%`, but standalone high-score weak/bad rate dropped back inside the gate at `1.33%`. The `11` high-score weak rows were spread across 여성 건강, 피부 건강, 운동/근력, and 혈압, and no actionable low-blast-radius condition was found.

A neutral all-category p20 holdout with seed `202606079` was then run to check the whole-distribution behavior after the targeted high-weak samples. It covered `447` products, produced `315` OpenAI Batch requests, and had a standalone weak/bad rate of `5.50%` and high-score weak/bad rate of `1.17%`. The sample had complete label coverage and found no actionable low-blast-radius pattern, which supports keeping the current algorithm from a neutral distribution side.

Latest high-score weak diagnostics also support keeping the current algorithm:

- High-score rows: `49,899`
- High-score weak/bad rows: `853`
- Within-high-score weak/bad rate: `1.71%`
- Overall high-score weak/bad rate: `1.33%`
- `function_similarity < 0.40` would catch `347` weak/bad rows but also affect `5,772` non-weak rows.
- `not_same_primary` would catch `220` weak/bad rows but also affect `5,280` non-weak rows.
- `same_primary_set` appears in `633` high-score weak/bad rows, so primary-set equality alone is not a reliable accept signal or reject signal.

## Validation Inputs

Current merged validation output:

```text
output/recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20
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
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606071
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606072
output/recommendation_quality_judge_v2_9_openai_targeted_midcoverage_seed202606073_p50
output/recommendation_quality_judge_v2_9_openai_targeted_lowrisk_seed202606074_p50
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606075_p20
output/recommendation_quality_judge_v2_9_openai_targeted_highweak_seed202606076_p35
output/recommendation_quality_judge_v2_9_openai_targeted_lowcoverage_highweak_seed202606077_p40
output/recommendation_quality_judge_v2_9_openai_targeted_lowcoverage_highweak_seed202606078_p60
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606079_p20
```

Retry replacements:

```text
output/recommendation_quality_judge_v2_9_openai_chunk_*_retry_*
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606066_p40_retry_*
output/recommendation_quality_judge_v2_9_openai_targeted_next_seed202606067_p50_retry_*
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606068_p15_retry_*
output/recommendation_quality_judge_v2_9_openai_holdout_seed2026060611_p30_retry_*
output/recommendation_quality_judge_v2_9_openai_targeted_midcoverage_seed202606073_p50_retry_*
output/recommendation_quality_judge_v2_9_openai_holdout_seed202606075_p20_retry_*
output/recommendation_quality_judge_v2_9_openai_targeted_highweak_seed202606076_p35_retry_*
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
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606071" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606072" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_midcoverage_seed202606073_p50" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_lowrisk_seed202606074_p50" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606075_p20" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_highweak_seed202606076_p35" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_lowcoverage_highweak_seed202606077_p40" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_targeted_lowcoverage_highweak_seed202606078_p60" `
  --parts-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606079_p20" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_chunk_*_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606066_p40_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_targeted_next_seed202606067_p50_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606068_p15_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed2026060611_p30_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_targeted_midcoverage_seed202606073_p50_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_holdout_seed202606075_p20_retry_*" `
  --retry-glob "output\recommendation_quality_judge_v2_9_openai_targeted_highweak_seed202606076_p35_retry_*" `
  --output-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20 `
  --high-score-threshold 0.65
```

This workflow also writes `high_score_weak_diagnostics.json` from the generated pattern features and `validation_status.json` from the quality gate plus diagnostics.

Write the Markdown report:

```powershell
python scripts\recommendation_quality_judge_batch.py validation-report `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20 `
  --top-categories 10 `
  --top-patterns 7
```

The report includes `high_score_weak_diagnostics.json` automatically when that file exists. To regenerate only the diagnostic JSON:

```powershell
python scripts\recommendation_quality_judge_batch.py diagnose-high-score-weak `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20 `
  --high-score-threshold 0.65
```

Regenerate only the stop/continue status:

```powershell
python scripts\recommendation_quality_judge_batch.py validation-status `
  --validation-dir output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20
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
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20\validation_status.json `
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
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20\validation_status.json `
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
  --summary-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20\openai_chunk_judge_summary.json `
  --validation-status-json output\recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout202606079_neutral_p20\validation_status.json `
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
