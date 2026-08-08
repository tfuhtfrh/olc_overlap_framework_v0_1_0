# PDF A+B Scale Sweep: GC 0.65

Generated: 2026-07-17 (Asia/Tokyo)

## What This Tested

This group tests whether the PDF A+B configuration continues to reach ground state as read count and variable count grow.

## Output CSVs

- `../../pdf_path_ab_more_reads_120_140_160_180_gc065.csv`
- `../../pdf_path_ab_more_reads_200_220_gc065.csv`
- `../../pdf_path_ab_180_gc065_multiseed.csv`
- `../../pdf_path_ab_200_gc065_multiseed_recoverability.csv`

## Result Statistics

| Reads | Candidate edges / variables | Result |
|---:|---:|---|
| 120 | 476 | 2/2 ground |
| 140 | 557 | 2/2 ground |
| 160 | 637 | 2/2 ground |
| 180 | 717 | 2/2 ground |
| 200 | 798 | 2/2 ground |
| 220 | 879 | 2/2 ground |

Multi-seed checks:

| Setting | Result |
|---|---|
| 180 reads, seeds 42-49 | 8/8 ground |
| 200 reads, seeds 40-49 | 10/10 ground, 10/10 valid, 10/10 recoverable |

## Conclusion

A+B is currently the strongest scaling result. The most important confirmed point is 200 reads / 798 variables with 10/10 ground, valid, and recoverable. The 220-read result is promising but has only two seeds in the current CSV set.

## Included In Main Conclusion

Yes.

## Limitations

The 220-read case needs more seeds before it should be treated as equally strong as the 200-read result.

## User Supplements

None yet.
