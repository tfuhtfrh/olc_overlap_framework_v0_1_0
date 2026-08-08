# PDFAssembly vs Old EdgePathDAG: GC 0.65

Generated: 2026-07-17 (Asia/Tokyo)

## What This Tested

This group compares the older `EdgePathDAGQUBOHamiltonian` against PDFAssembly variants under GC fraction 0.65.

## Output CSVs

- `../../pdf_vs_edge_path_gc065_more_reads.csv`
- `../../pdf_vs_edge_path_optimized_gc065_80_90_100.csv`

## Result Statistics

| Reads | old edge-path DAG | PDF A+B | PDF A+B+C+D |
|---:|---:|---:|---:|
| 50 | 1/2 ground | 2/2 ground | 2/2 ground |
| 60 | 1/2 ground | 1/2 ground | 2/2 ground |
| 70 | 0/2 ground | 2/2 ground | 2/2 ground |
| 80/90/100 combined | 0/6 ground | 6/6 ground | not tested in that CSV |

## Conclusion

PDF A+B clearly outperforms old edge-path DAG at larger read counts in these CSVs. The strongest contrast is the 80/90/100 read comparison: old edge-path DAG is 0/6 while PDF A+B is 6/6.

## Included In Main Conclusion

Yes. This is core evidence that adding the B length term overcomes the old edge-path scaling issue.

## Limitations

The 50/60/70 CSV includes A+B+C+D, while the 80/90/100 optimized CSV compares only old edge-path DAG and A+B.

## User Supplements

None yet.
