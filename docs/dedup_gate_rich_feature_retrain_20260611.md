# Deduplicated Validation Gate and Rich Feature Retrain

## Summary

- Date: 2026-06-11
- Validation gate change: duplicate recommendation labels are removed by `(base_report_no, target_report_no, rank)` before pattern analysis and quality-gate scoring.
- Deduplication strategy: `last` label wins by default.
- Model feature engineering version: `role_semantic_v2`
- Model artifact: `output/ml_models/recommendation_quality_model.pkl`
- Operating threshold: unchanged at `0.834`

## Validation Gate Result

Recomputed validation output:

```text
output/recommendation_quality_judge_v13_validation_plus_gita_p400_dedup_20260611
```

| Metric | Value |
| --- | ---: |
| Raw labels | 549 |
| Deduplicated gate labels | 382 |
| Duplicate labels removed | 167 |
| Raw coverage | 549 / 549 |
| Coverage OK | true |
| Weak labels | 7 |
| Bad labels | 0 |
| Weak/bad rate | 1.83% |
| High-score weak/bad rate | 1.83% |
| Actionable patterns | 0 |

Gate decision:

```text
pass_continue_validation_without_algorithm_change
```

The previous merged review signal was caused by counting overlapping samples as independent labels. After deduplication, the merged gate matches the cleaner p400-only interpretation.

## Rich Feature Engineering

The XGBoost feature set increased from the previous baseline to 58 features. New feature groups focus on role and semantic mismatch:

- Cross-role primary overlap: `primary_cross_role_overlap_count`, `primary_cross_role_overlap_ratio`, `primary_role_mismatch_ratio`
- Primary overlap shape: `primary_overlap_gap`, `primary_count_delta_abs`
- Semantic asymmetry: `semantic_unshared_total`, `semantic_unshared_ratio`, `semantic_unshared_gap_abs`
- High-score interaction flags: `high_score_no_core`, `high_score_low_function`, `high_score_cross_role_no_primary`
- Same-primary risk interactions: `same_primary_no_core`, `same_primary_low_function`
- Score gap features: `function_core_gap_abs`, `sim_function_gap_positive`, `sim_core_gap_positive`

The API inference wrapper now mirrors these features and reads the saved artifact threshold unless `RECOMMENDATION_QUALITY_WEAK_THRESHOLD` overrides it.

## Retrain Result

Training source:

```text
output/recommendation_quality_judge_v2_9_openai_validation_current_plus_targeted_holdout2026060719_oralxylitol_p260_livermilk_p300
```

Training summary:

| Metric | Value |
| --- | ---: |
| Train rows | 80,395 |
| Weak/bad positives | 6,709 |
| Positive rate | 8.35% |
| Feature count | 58 |
| Final train ROC-AUC | 0.9849 |
| Final train Average Precision | 0.8441 |

Top new signal among feature importances:

```text
high_score_no_core
target_primary_in_base_secondary_count
primary_count_delta_abs
```

## Holdout and External Check

Report:

```text
output/ml_validation/xgboost_rich_feature_holdout_external_report_20260611_2120.md
```

Holdout:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9638 |
| Average Precision | 0.7096 |
| Precision at 0.834 | 64.68% |
| Recall at 0.834 | 68.78% |
| Filter rate at 0.834 | 8.87% |
| False filter / all at 0.834 | 3.13% |

External dedup p400:

| Metric | Value |
| --- | ---: |
| Rows | 382 |
| Weak/bad positives | 7 |
| ROC-AUC | 0.9002 |
| Average Precision | 0.3623 |
| Filtered at 0.834 | 5 |
| Weak caught at 0.834 | 3 / 7 |
| False filters at 0.834 | 2 |

The retrained model now catches more of the p400 weak rows at the unchanged threshold. The tradeoff is that it also filters two acceptable-adjacent rows in this small external sample.

False-filter rows at threshold `0.834`:

| Base | Target | Rank | XGBoost probability | Judge label |
| --- | --- | ---: | ---: | --- |
| `테아닌 B` | `울랄라 익스텐션` | 7 | 0.9318 | acceptable_adjacent |
| `테아닌 B` | `스트레스케어 테아닌 250` | 8 | 0.9090 | acceptable_adjacent |

Both false filters are high-score, same-ingredient adjacent rows with `core_match_score = 0`. This should be watched because the richer model is intentionally more sensitive to high-score/no-core cases.

## Engineering Decision

This is a reasonable model upgrade, but it should still be monitored as a conservative filter rather than treated as a complete quality model.

Recommended next checks before production promotion:

1. Run the API recommendation smoke tests against the updated model artifact.
2. Inspect the two p400 false-filter rows from `xgboost_rich_feature_external_p400_scored_rows_20260611_2120.csv`.
3. Continue collecting independent v2.13 labels until at least 1,000 rows, then rerun the deduplicated gate.
