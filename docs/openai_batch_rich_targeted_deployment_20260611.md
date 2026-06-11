# OpenAI Batch Rich Targeted Validation and Deployment Decision

## Summary

- Date: 2026-06-11
- OpenAI Batch job: `batch_6a2ab1cc9e5c81909eece6413b34c017`
- Judge model: `gpt-5-nano`
- Validation output: `output/recommendation_quality_judge_v13_rich_targeted_validation_20260611`
- Label count: 451 / 451
- Error count: 0
- Final model threshold: `0.95`

## Batch Composition

The sample was selected from a 3,806-row recommendation pool after excluding previously judged base-target pairs.

| Target signal | Labels |
| --- | ---: |
| `filtered_at_0_834` | 51 |
| `shadow_0_7_0_834` | 80 |
| `mid_0_5_0_7` | 140 |
| `high_score_no_core_control` | 120 |
| `background_control` | 60 |

## Judge Result

| Metric | Value |
| --- | ---: |
| Labels | 451 |
| Reasonable | 76 |
| Acceptable-adjacent | 337 |
| Weak | 38 |
| Bad | 0 |
| Weak/bad rate | 8.43% |
| High-score weak/bad count | 2 |
| High-score weak/bad rate | 0.44% |
| Quality gate decision | `pass_continue_validation_without_algorithm_change` |

The runtime recommendation algorithm still passes the gate.

## Filter Threshold Finding

The previous `0.834` model threshold was rejected for production filtering.

On this targeted sample:

| Threshold | TP | FP | Precision | Recall | False filter / all |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.834 | 8 | 43 | 15.69% | 21.05% | 9.53% |
| 0.900 | 4 | 15 | 21.05% | 10.53% | 3.33% |
| 0.950 | 0 | 1 | 0.00% | 0.00% | 0.22% |

This showed that the `role_semantic_v2` model was overfiltering acceptable-adjacent rows at `0.834`.

## Final Retrain

The final artifact was retrained with:

- Existing 80,395-row judge dataset
- Deduplicated p400 v2.13 labels
- New 451-label rich targeted Batch

Artifact:

```text
output/ml_models/recommendation_quality_model.pkl
```

Saved threshold:

```text
0.95
```

The API inference wrapper uses the artifact threshold unless `RECOMMENDATION_QUALITY_WEAK_THRESHOLD` is explicitly set.

## Deployment Decision

Decision:

```text
deploy_conservative_filter_threshold_0_95
```

Reason:

- v2.13 algorithm gate passes after deduplication.
- `0.834` was too aggressive and was not deployed.
- `0.95` is conservative enough to avoid broad false filtering.
- In the targeted 30% holdout experiment, `0.95` reached 77.78% precision and 63.64% recall with 1.47% false filter / all.

Final artifact check:

| Dataset | Precision | Recall | Filter rate | False filter / all |
| --- | ---: | ---: | ---: | ---: |
| p400 dedup | 0.00% | 0.00% | 0.00% | 0.00% |
| rich targeted full | 69.70% | 60.53% | 7.32% | 2.22% |

The p400 result means this threshold does not add risk to the earlier stable validation set. The rich targeted result means it can still catch a subset of concentrated weak cases.

## Outputs

- Decision JSON: `output/ml_validation/xgboost_final_extra_label_threshold095_decision_20260611.json`
- Decision report: `output/ml_validation/xgboost_final_extra_label_threshold095_decision_20260611.md`
- Batch decision metrics: `output/recommendation_quality_judge_v13_rich_targeted_validation_20260611/rich_targeted_validation_decision_metrics.json`
