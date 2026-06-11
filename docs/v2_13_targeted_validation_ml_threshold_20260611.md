# v2.13 Targeted Validation and ML Threshold Rationale

## Summary

- Date: 2026-06-11
- Runtime algorithm: `semantic_weighted_jaccard_v2_13`
- OpenAI judge model: `gpt-5-nano`
- Main targeted category: `기타`
- Primary Batch job: `batch_6a2a8c8edb8c8190a799ccf2ef474381`
- Primary validation output: `output/recommendation_quality_judge_v13_targeted_gita_20260611_p400`
- Merged continuity output: `output/recommendation_quality_judge_v13_validation_plus_gita_p400_20260611`
- XGBoost holdout report: `output/ml_validation/xgboost_holdout_threshold_report_20260611_185052.md`
- p400 XGBoost external check: `output/ml_validation/xgboost_v13_gita_p400_only_summary_20260611.json`

Decision:

```text
Do not change runtime recommendation logic or production ML threshold yet.
```

The larger targeted Batch passed the current quality gate by itself. The merged continuity report is useful for trend inspection, but it contains overlapping rows from earlier targeted runs, so it should not be treated as 549 independent labels.

## Primary OpenAI Batch Result

This run targeted the `기타` category because the earlier v2.13 sample concentrated its weak labels there.

| Metric | Value |
| --- | ---: |
| Selected base products | 360 |
| Products with display recommendations | 125 |
| OpenAI Batch requests | 125 |
| Completed requests | 125 |
| Failed requests | 0 |
| Judge labels | 382 / 382 |
| Reasonable labels | 27 |
| Acceptable-adjacent labels | 348 |
| Weak labels | 7 |
| Bad labels | 0 |
| Weak/bad rate | 1.83% |
| High-score weak/bad labels | 7 |
| High-score weak/bad rate | 1.83% |
| Actionable low-blast-radius patterns | 0 |

Gate decision:

```text
pass_continue_validation_without_algorithm_change
```

Reason:

- Overall weak/bad rate and high-score weak/bad rate are both inside the current gate.
- No candidate pattern was narrow enough to justify a hard cap.
- The sample is still below the 5,000-label stop-sampling target, so further validation is useful, but not because the current logic failed.

## Pattern Diagnosis

No new algorithmic cap is justified by the p400 targeted sample.

| Candidate condition | Matched | Weak/bad | Weak/bad rate | Non-weak affected |
| --- | ---: | ---: | ---: | ---: |
| `shared_core_0` | 364 | 5 | 1.37% | 359 |
| `function_lt_0_30` | 6 | 1 | 16.67% | 5 |
| `function_lt_0_40` | 7 | 1 | 14.29% | 6 |
| `primary_primary_overlap_0` | 26 | 3 | 11.54% | 23 |
| `not_same_primary` | 28 | 3 | 10.71% | 25 |
| `shared_core_le1_and_fn_lt_0_40` | 5 | 0 | 0.00% | 5 |

The closest-looking rules are still not actionable:

- `shared_core_0` catches 5 weak rows but would affect 359 non-weak rows.
- `primary_primary_overlap_0` catches only 3 weak rows and would affect 23 non-weak rows.
- `function_lt_0_30` and `function_lt_0_40` are too small and still not clean enough.

The observed weak rows are therefore better treated as review signals and model-training candidates, not deterministic filtering rules.

## Merged Continuity Report

The previous v13 sample, the first small targeted sample, and the p400 targeted sample were merged into:

```text
output/recommendation_quality_judge_v13_validation_plus_gita_p400_20260611
```

| Metric | Value |
| --- | ---: |
| Reported merged labels | 549 |
| Unique `(base_report_no, target_report_no, rank)` keys | 382 |
| Weak/bad labels | 14 |
| Weak/bad rate | 2.55% |
| High-score weak/bad labels | 13 |
| High-score weak/bad rate | 2.37% |
| Actionable pattern count | 0 |

Merged gate decision:

```text
review_collect_more_targeted_samples
```

This merged result crosses the 2% high-score weak/bad gate, but the overlap means it is not independent evidence. It is useful as a warning that `기타` remains the category to watch, but the p400-only result is the cleaner decision basis.

## XGBoost Holdout Basis

The XGBoost quality model was evaluated on a fresh 80/20 stratified split from the 80,395-row judge dataset.

| Metric | Value |
| --- | ---: |
| Total rows | 80,395 |
| Weak/bad positives | 6,709 |
| Positive rate | 8.35% |
| Holdout rows | 16,079 |
| Holdout ROC-AUC | 0.9637 |
| Holdout Average Precision | 0.7098 |

Operating threshold: `0.834` rounded (`0.8344295` in the saved artifact).

| Threshold | Precision | Recall | Filter rate | False filter / all |
| ---: | ---: | ---: | ---: | ---: |
| 0.500 | 45.40% | 90.01% | 16.55% | 9.04% |
| 0.700 | 54.84% | 82.71% | 12.59% | 5.68% |
| 0.800 | 61.48% | 75.04% | 10.19% | 3.92% |
| 0.834 | 64.27% | 69.30% | 9.00% | 3.22% |
| 0.900 | 72.02% | 54.47% | 6.31% | 1.77% |
| 0.950 | 79.71% | 32.49% | 3.40% | 0.69% |

The current threshold remains defensible as a conservative suppressor. It is not tuned to maximize recall; it is tuned to avoid broad false filtering.

## p400 External XGBoost Check

The p400 validation set was scored separately after merging the p400 pattern features. External-only legacy score columns that do not exist in the p400 result were filled conservatively with current score / zero delta.

| Metric | Value |
| --- | ---: |
| Rows | 382 |
| Weak/bad labels | 7 |
| Weak/bad rate | 1.83% |
| ROC-AUC | 0.8680 |
| Average Precision | 0.2979 |
| Filtered at operating threshold | 0 |
| Weak/bad caught at operating threshold | 0 / 7 |

Probability bands:

| XGBoost probability band | Rows | Weak/bad | Weak/bad rate |
| --- | ---: | ---: | ---: |
| `0-0.1` | 348 | 3 | 0.86% |
| `0.1-0.3` | 27 | 1 | 3.70% |
| `0.3-0.5` | 5 | 2 | 40.00% |
| `0.5-0.7` | 2 | 1 | 50.00% |
| `0.7-0.834` | 0 | 0 | 0.00% |
| `0.834+` | 0 | 0 | 0.00% |

Threshold check on p400:

| Threshold | TP | FP | FN | TN | Precision | Recall | Filter rate | False filter / all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.500 | 1 | 1 | 6 | 374 | 50.00% | 14.29% | 0.52% | 0.26% |
| 0.600 | 1 | 0 | 6 | 375 | 100.00% | 14.29% | 0.26% | 0.00% |
| 0.700 | 0 | 0 | 7 | 375 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.800 | 0 | 0 | 7 | 375 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.834 | 0 | 0 | 7 | 375 | 0.00% | 0.00% | 0.00% | 0.00% |

The external check shows two things at once:

- The model ranks some weak rows higher than the background rows, so it is still useful as a monitoring signal.
- The current threshold is too conservative to catch the p400 weak rows, but lowering it now would be based on only 7 positives and would still catch only 1 of them at `0.5-0.6`.

The weakest blind spot is not a simple threshold problem. Some high-score weak rows involving broad ingredient overlap, such as artichoke-related pairs, receive very low XGBoost probabilities. That points to feature engineering and richer ingredient-role semantics, not an immediate threshold change.

## Recommendation

Keep v2.13 runtime logic and the current XGBoost production threshold.

Do not add a new deterministic rule from the current targeted results. Every apparent rule either affects too many acceptable-adjacent rows or has too few weak examples.

Recommended next work:

1. Continue OpenAI Batch validation until at least 1,000 independent v2.13 labels, then ideally 5,000 labels.
2. Deduplicate validation rows before gate calculation when merging overlapping targeted runs.
3. Track a shadow XGBoost threshold band around `0.5-0.7`, but do not use it for production suppression yet.
4. Add feature candidates for ingredient role mismatch, primary-vs-secondary ingredient asymmetry, and same-ingredient/different-functional-claim cases.
5. Retrain only after those feature candidates are labeled on fresh, independent rows.

Current engineering conclusion:

```text
Validation improved confidence in v2.13. More Batch validation is useful, but it does not yet justify changing the algorithm or threshold.
```
