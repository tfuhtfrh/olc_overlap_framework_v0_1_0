# PDFAssembly Ablation: 35/40/45 Reads

Generated: 2026-07-17 (Asia/Tokyo)

## What This Tested

This experiment tested which PDFAssembly terms matter by comparing old edge-path DAG against combinations of A, B, C, and D.

## Output CSV

- `../../pdf_vs_edge_path_ablation_35_40_45.csv`

## Terms

- A: edge-path skeleton.
- B: assembly length prior.
- C: GC composition prior.
- D: MI/NMI reward.

## Result Statistics

| Model | Ground hits | Interpretation |
|---|---:|---|
| old edge-path DAG | 1/6 | weak baseline in this sweep |
| PDF A only | 0/6 | A alone is not enough |
| PDF A+C | 0/6 | GC alone does not rescue A |
| PDF A+D | 0/6 | MI reward alone does not rescue A |
| PDF A+B | 6/6 | length prior is decisive |
| PDF A+B+C | 4/6 | adding C can perturb some cases |
| PDF A+B+D | 5/6 | adding D can perturb some cases |
| PDF A+B+C+D | 6/6 | full combination works in this sweep |

## Conclusion

B is the key term in this ablation. The correct summary is not "PDF terms help equally"; it is "A+B is the minimal consistently successful combination in this sweep."

## Included In Main Conclusion

Yes. This is one of the core pieces of evidence that the assembly length prior is the main breakthrough.

## Limitations

Only 35/40/45 reads and two seeds per read count were tested here.

## User Supplements

None yet.
