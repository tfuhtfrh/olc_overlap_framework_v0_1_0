# OLC Overlap Framework

Version: 0.1.0

This is a modular reproduction-code framework for OLC-style read overlap experiments.

Pipeline:

1. Random read generation with mismatch / insertion / deletion errors and shuffled input order.
2. all-vs-all overlap candidate finding with minimap2.
3. candidate refinement with a Parasail-compatible refiner interface.
4. Shannon mutual-information scoring from refined alignments.
5. Optional layout solver interface for OR-Tools plus a QUBO layout solver with built-in simulated annealing.
6. Edge-level evaluation against simulated ground truth.

## Module map

```text
src/olc_pipeline/
  data.py              Shared dataclasses and interfaces between stages.
  simulator.py         Random genome/read simulator with error injection.
  io_utils.py          FASTA/text helpers.
  candidate_finder.py  Minimap2CandidateFinder + OriginalCandidateFinder stub.
  refiner.py           ParasailOverlapRefiner using parasail semi-global alignment.
  mi_scorer.py         Shannon MI / NMI scorer over aligned columns.
  evaluator.py         Edge and layout evaluation helpers.
  layout_solver.py     Dummy/OR-Tools/QUBO layout solver interfaces and built-in binary SA.
  pipeline.py          High-level experiment runner.
```

## Installation

Create a virtual environment, then install the local package in editable mode:

```bash
pip install -e .
```

Install minimap2 separately. For example with conda:

```bash
conda install -c bioconda minimap2
```

Required Python package for DP refinement:

```bash
pip install parasail
```

Optional packages for future modules:

```bash
pip install ortools
```

Optional QUBO / simulated annealing backend:

```bash
pip install ".[qubo]"
# or:
pip install dwave-samplers dimod
```

Version 0.1.0 uses `parasail` for semi-global DP refinement. The demo also requires command-line `minimap2`.

## Run demos

From PowerShell, enter the WSL project environment with:

```powershell
wsl bash -lc "cd /mnt/d/Pytnon/olc_overlap_framework_v0_1_0/olc_overlap_framework && source .venv/bin/activate && exec bash -i"
```

Then run any demo from that activated WSL shell, for example:

```bash
python demo.py
python demo_exact_vs_minimap.py
```

To include the C++ version of the original exact-overlap algorithm in the speed
comparison, compile it in WSL and pass the resulting binary to the comparison
demo:

```bash
g++ -O3 -std=c++17 tests/test1_overlap.cpp -o tests/test1_overlap
python demo_exact_vs_minimap.py --cpp-bin tests/test1_overlap
```

The all-outgoing C++ variant emits every exact suffix-prefix candidate, so it can
produce jump-correct edges:

```bash
g++ -O3 -std=c++17 tests/test1_all_outgoing.cpp -o tests/test1_all_outgoing
python demo_exact_vs_minimap.py --cpp-bin tests/test1_overlap --cpp-all-bin tests/test1_all_outgoing
```

The exact-overlap vs minimap2 comparison demo should be run in WSL because the
bundled minimap2 build is a Linux executable. If minimap2 is not on the WSL
`PATH`, the comparison demo will try the sibling source build at
`../minimap2/minimap2`.

## QUBO layout solver

`QUBOLayoutSolver` consumes refined `OverlapEdge` objects and solves the OLC
layout order with binary variables `x[v,j]`, meaning read `v` appears at layout
position `j`. The default Hamiltonian is:

```text
A_read * sum_v (1 - sum_j x[v,j])^2
+ A_pos * sum_j (1 - sum_v x[v,j])^2
+ B_missing * sum_(u,v not in E) sum_j x[u,j] x[v,j+1]
```

The first two coefficients should be high because they penalize invalid binary
layouts. The missing-edge coefficient is the editable edge term.

```python
from olc_pipeline.layout_solver import (
    DWaveAnnealingConfig,
    DWaveSimulatedAnnealer,
    MissingEdgeHamiltonianConfig,
    MissingEdgeQUBOHamiltonian,
    QUBOLayoutSolver,
)

solver = QUBOLayoutSolver(
    hamiltonian=MissingEdgeQUBOHamiltonian(MissingEdgeHamiltonianConfig(
        read_once_penalty=100.0,
        position_once_penalty=100.0,
        missing_edge_penalty=1.0,
    )),
    annealer=DWaveSimulatedAnnealer(DWaveAnnealingConfig(
        num_reads=30,
        num_sweeps=1000,
        seed=42,
    )),
)
layout = solver.solve(reads, edges)
```

To test a different Hamiltonian, implement `QUBOHamiltonianBuilder.build()` and
pass it to `QUBOLayoutSolver(hamiltonian=...)`.

For experiments that reward strong overlaps instead of only penalizing missing
edges, use `WeightedOverlapQUBOHamiltonian`. It keeps the same binary layout
constraints and adds:

```text
+ C_missing * adjacent pairs not in E
- B_reward * reward(u, v) for adjacent pairs in E
```

The reward source is selectable with `score_mode`: `overlap_len`, `identity`,
`dp`, `mi`, `nmi`, `mapq`, `matches`, or `quality`. `demo.py` exposes these as
top-level constants:

```python
QUBO_HAMILTONIAN = "weighted_overlap"
QUBO_SCORE_MODE = "mi"
QUBO_EDGE_REWARD_SCALE = 2.0
QUBO_NORMALIZE_REWARDS = True
```

`demo.py` now runs a small QUBO layout demo after DP refinement. It uses the
D-Wave sampler backend when `dwave-samplers` and `dimod` are installed, and
falls back to the built-in binary simulated annealer otherwise.

The demo can optionally write a Graphviz DOT file for the overlap graph:

```python
WRITE_QUBO_GRAPH_DOT = True
QUBO_GRAPH_DOT_PATH = Path("debug/qubo/overlap_graph.dot")
```

Render it in WSL/Linux with:

```bash
dot -Tsvg debug/qubo/overlap_graph.dot -o debug/qubo/overlap_graph.svg
dot -Tpng debug/qubo/overlap_graph.dot -o debug/qubo/overlap_graph.png
```

Render it in Windows PowerShell with:

```powershell
dot -Tsvg debug\qubo\overlap_graph.dot -o debug\qubo\overlap_graph.svg
dot -Tpng debug\qubo\overlap_graph.dot -o debug\qubo\overlap_graph.png
```

The QUBO demo also has an optional permutation-aware polish step:

```python
QUBO_POLISH_LAYOUT = True
QUBO_POLISH_MAX_PASSES = 20
```

D-Wave's simulated annealer updates individual binary variables. For the
one-hot layout encoding `x[v,j]`, moving between two valid read permutations
usually requires coordinated bit changes, so bit-level annealing can get stuck
near a valid but poor permutation. The polish step decodes the binary sample to
an order, then tries swap and insert moves that always stay inside valid
permutation space while evaluating the same QUBO energy. This does not change
the Hamiltonian; it only improves the search path after annealing.

Conda deployment on Windows:

```powershell
& 'D:\Applications\Anaconda\shell\condabin\conda-hook.ps1'
conda activate base
cd D:\Pytnon\olc_overlap_framework_v0_1_0\olc_overlap_framework
pip install -e ".[qubo]"
$env:PYTHONPATH='src'
python -m pytest
```

WSL deployment:

```bash
cd /mnt/d/Pytnon/olc_overlap_framework_v0_1_0/olc_overlap_framework
source .venv/bin/activate
python -m pip install -e ".[qubo]"
python demo.py
```

## Notes

- `OriginalCandidateFinder` is a reserved interface for the custom/original overlap algorithm.
- `ORToolsLayoutSolver` is intentionally left unimplemented.
- The first version only handles same-strand (`+`) suffix-prefix overlaps. Reverse-complement handling should be added in a later version.
