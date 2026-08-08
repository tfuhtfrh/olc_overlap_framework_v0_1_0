# PDFAssembly QUBO Experiment Notes

Generated: 2026-07-17 (Asia/Tokyo)

This folder records the interpretation layer for PDFAssembly QUBO experiments. Current CSV files remain in `olc_overlap_framework/debug/qubo/` and are referenced from these notes rather than moved.

## Hamiltonian Terms

- A: edge-path skeleton. Current implementation reuses the `EdgePathDAG` constraints: select `N-1` edges and penalize incoming/outgoing degree conflicts.
- B: assembly length prior. It encourages total read length minus selected overlap lengths to match the target genome length.
- C: GC composition prior. It encourages assembled GC fraction to match the target GC fraction.
- D: MI/NMI reward. It rewards selected edges with high overlap information score.

## Current Main Conclusions

1. The B length term is the clearest breakthrough.
2. A+B is already very strong; D is not currently proven necessary.
3. A+B scales to 200 reads with 10/10 ground, valid, and recoverable in the current multi-seed result.
4. Error sweeps should be interpreted as A+B versus A+B+D because C is off in those CSVs.
5. Pilot failures caused by missing Hamiltonian path in the candidate graph are graph-processing failures, not Hamiltonian ground-state failures.

## Experiment Groups

- `ablation_35_40_45/notes.md`: A/B/C/D ablation on 35, 40, 45 reads.
- `compare_old_edge_path_gc065/notes.md`: comparison with old edge-path DAG at 50-100 reads.
- `scale_sweep_gc065/notes.md`: A+B scale tests from 120 to 220 reads, including multi-seed checks.
- `error_sweep_70_reads/notes.md`: 70-read artificial error sweep, D off/on comparison.
- `pilots/notes.md`: pilot runs, including one graph-processing failure at high artificial error.

## Maintenance

Do not silently rewrite old experiment interpretations. If new data changes a conclusion, add a dated correction or a superseding section. User supplements should be kept in the relevant experiment note.
