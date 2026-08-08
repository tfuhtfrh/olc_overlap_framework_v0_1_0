# PDFAssembly Pilot Runs

Generated: 2026-07-17 (Asia/Tokyo)

## What This Tested

These were preliminary single-case pilot runs before the formal grid experiments.

## Output CSVs

- `../../pdf_abd_error2_raw_mi_pilot.csv`
- `../../pdf_abd_error5_raw_mi_pilot.csv`

## Parameter Clarification

Despite the `abd` name, the recorded coefficients show:

| Parameter | Value | Meaning |
|---|---:|---|
| `degree_penalty` | 120.0 | A on |
| `length_penalty` | 2e-06 | B on |
| `gc_penalty` | 0.0 | C off |
| `mi_reward_scale` | 0.0 | D off |

Therefore these pilot rows are A+B by actual parameters, not A+B+D.

## Result Statistics

| Pilot | Result | Interpretation |
|---|---|---|
| 2% error raw_mi pilot | 70 reads, 273 candidate edges, ground=true, valid=true, recoverable=true | successful single-case A+B pilot |
| 5% error raw_mi pilot | error before QUBO solve: candidate DAG lacks Hamiltonian path covering all 70 reads; longest path has 59 edges | graph-processing failure under this pilot setup |

## Conclusion

Mark these as trial/pilot results. The 5% pilot failure should not be counted as Hamiltonian failure. The conclusion is that, under that high artificial error pilot setup, graph processing failed because the candidate graph did not contain the full true path.

## Included In Main Conclusion

No. Use these only as historical context and as a warning that high artificial error can break the graph-processing stage before Hamiltonian optimization becomes meaningful.

## User Supplements

None yet.
