# PDFAssembly Error Sweep: 70 Reads

Generated: 2026-07-17 (Asia/Tokyo)

## What This Tested

This group tests artificial read error rates at 70 reads, comparing D off and D on.

## Output CSVs

- `../../pdf_ab_vs_abd_error2_raw_mi_grid.csv`
- `../../pdf_ab_vs_abd_error3_raw_mi_grid.csv`
- `../../pdf_ab_vs_abd_error5_raw_mi_grid.csv`
- `../../pdf_ab_vs_abd_error2_nmi_grid.csv`
- `../../pdf_ab_vs_abd_error3_nmi_grid.csv`
- `../../pdf_ab_vs_abd_error5_nmi_grid.csv`

## Parameter Interpretation

These CSVs do not use a `model` column. The model is inferred from coefficients:

| Parameter | Meaning |
|---|---|
| `degree_penalty=120.0` | A on |
| `length_penalty=2e-06` | B on |
| `gc_penalty=0.0` | C off |
| `mi_reward_scale=0.0` | D off, so the model is A+B |
| `mi_reward_scale>0.0` | D on, so the model is A+B+D |

`mi_score_mode=raw_mi` or `nmi` only matters when D is on. If `mi_reward_scale=0.0`, the D term contributes zero regardless of the recorded score mode.

## Compared Cases

| Error rate | D score | D strengths tested | Result |
|---:|---|---|---|
| 2% | raw_mi | 0, 1, 2.5, 5, 10 | all 5/5 ground |
| 3% | raw_mi | 0, 1, 2.5, 5, 10 | all 5/5 ground |
| 5% | raw_mi | 0, 5, 10 | all 5/5 ground |
| 2% | nmi | 0, 2, 5, 10, 20 | all 5/5 ground |
| 3% | nmi | 0, 2, 5, 10, 20 | all 5/5 ground |
| 5% | nmi | 0, 10, 20 | scale 10 gives 4/5 ground; scale 0 and 20 give 5/5 ground; all are 5/5 recoverable |

## Conclusion

The clean interpretation is that A+B is already strong in this 70-read error sweep. D was tested both off and on, but current evidence does not show that D is necessary. Raw MI did not hurt in these settings. NMI at 5% error and scale 10 caused one exact ground miss, but recoverability remained 5/5.

## Included In Main Conclusion

Yes, but only as robustness evidence for A+B and as a scan of D. Do not present this as evidence that D is the main improvement.

## Limitations

All cases here use 70 reads and C is off. These results do not establish how C or D behave at larger read counts.

## User Supplements

None yet.
