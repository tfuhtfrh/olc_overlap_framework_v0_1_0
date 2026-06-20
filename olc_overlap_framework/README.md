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

Optional simulated quantum annealing and real D-Wave QPU backends:

```bash
pip install ".[sqa]"   # OpenJij simulated quantum annealing
pip install ".[qpu]"   # D-Wave Ocean QPU interface
pip install ".[qa]"    # both OpenJij SQA and D-Wave QPU support
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

### DAG edge-path Hamiltonian

`EdgePathDAGQUBOHamiltonian` uses one binary variable for each unique valid
directed overlap edge:

```text
y[u,v] = 1 when overlap edge u -> v is selected
```

For `N` reads it minimizes:

```text
A_count * (sum_e y[e] - (N - 1))^2
+ A_degree * incoming/outgoing pair conflicts
- B_reward * sum_e normalized_reward[e] * y[e]
```

The degree terms penalize selecting more than one incoming or outgoing edge at
any read. If the candidate graph is a DAG, selecting exactly `N-1` edges with
these degree limits produces one path covering every read. The builder checks
that the candidate graph is acyclic and, by default, that it contains a
Hamiltonian path before annealing.

This reduces the variable count from `N^2` to `|E|`, although the exact edge
count constraint still introduces dense pairwise couplings between edge
variables. Select it in `demo.py` with:

```python
QUBO_HAMILTONIAN = "edge_path_dag"
QUBO_EDGE_COUNT_PENALTY = 100.0
QUBO_EDGE_DEGREE_PENALTY = 120.0
QUBO_EDGE_REWARD_SCALE = 20.0
```

For the current 35-read, 131-edge zero-error demo, a 10-seed scan
(`seed=40..49`, `100 reads`, `1000 sweeps`, `trotter=32`) found:

```text
edge count penalty = 100
degree penalty     = 120
ground-state hits = 10 / 10
valid path hits   = 10 / 10
```

The nearby `degree=115` setting reached `8/10`. Lower degree penalties tended
to leave one incoming/outgoing conflict; higher degree penalties tended to
select only 33 of the required 34 edges. The dedicated sweep command is:

```bash
python demo_edge_path_parameter_sweep.py
```

It writes `debug/qubo/edge_path_dag_parameter_sweep.csv` by default.

The same script can change the simulated read density and report larger QUBO
sizes before running SQA:

```bash
python demo_edge_path_parameter_sweep.py \
  --step 250 \
  --read-counts 16,35,50,69 \
  --describe-only
```

Current zero-error scale measurements with a 20,000 bp genome and 3,000 bp
reads are:

```text
step  reads  edge variables  quadratic terms
500      35             131            8,515
350      49             315           49,455
250      69             577          166,176
200      86             954          454,581
```

The dense quadratic growth comes mainly from
`(sum_e y[e] - (N - 1))^2`. With `100 reads`, `1000 sweeps`, `trotter=32`,
and multiple annealer seeds, the observed penalty trend was:

```text
variables  count/degree  ground hits
       54       100/120           5/5
      131       100/120         10/10
      217       100/120           5/5
      315       120/140          4/10
      405       140/160          2/10
      577       about 190/210     0/3
```

The preferred penalties move upward as the edge-variable model grows, but
coefficient tuning alone does not preserve success rate. Beyond roughly 300
variables in this dense formulation, increasing annealing effort or replacing
the global edge-count square with a less densely coupled encoding becomes more
important than further penalty scaling.

`demo.py` now runs a small QUBO layout demo after DP refinement. It uses the
D-Wave sampler backend when `dwave-samplers` and `dimod` are installed, and
falls back to the built-in binary simulated annealer otherwise.

Annealer backend selection:

```python
QUBO_ANNEALER_BACKEND = "dwave_sa"     # local classical SA via dwave-samplers
QUBO_ANNEALER_BACKEND = "openjij_sqa"  # local simulated quantum annealing
QUBO_ANNEALER_BACKEND = "dwave_qpu"    # real D-Wave QPU via dwave-system
```

All backends consume the same `QUBOModel` and Hamiltonian builders. The QPU
backend is reserved for real D-Wave access and may require `dwave config create`
or explicit token/endpoint settings.

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

Run a small SQA/SA parameter sweep:

```bash
python demo_sqa_parameter_sweep.py
```

By default this scans a high-Trotter curve with fixed representative QUBO
coefficients:

```text
read/position penalty = 100
missing edge penalty  = 50
edge reward scale     = 30
trotter               = 16,32,48,64,96,128
```

Useful focused scans:

```bash
python demo_sqa_parameter_sweep.py --max-reads 16 --num-reads 50 --num-sweeps 1000
python demo_sqa_parameter_sweep.py --backend dwave-sa --max-reads 16 --num-reads 50 --num-sweeps 1000
python demo_sqa_parameter_sweep.py --trotters 16,32,48,64,96,128
python demo_sqa_parameter_sweep.py --trotters none,16,32 --read-penalties 100,200,500 --missing-penalties 30,50,100 --reward-scales 10,20,30
```

The sweep writes `debug/qubo/sqa_parameter_sweep.csv`. First check whether
`valid` becomes `True` and whether read/position violations go to zero. For
layout quality, prefer rows with small `missing_edge_count`; `connected_layout`
means zero missing adjacent graph edges, and `acceptable_connected_layout` uses
the `--acceptable-missing-edges` threshold. When one or two missing edges are
acceptable, compare `legal_true_adjacent_count` next: it counts true adjacent
read pairs among layout adjacencies that are also legal overlap-graph edges.

Current SQA starting point from the 16-read sweep:

```python
QUBO_READ_ONCE_PENALTY = 100.0
QUBO_POSITION_ONCE_PENALTY = 100.0
QUBO_MISSING_EDGE_PENALTY = 120.0
QUBO_EDGE_REWARD_SCALE = 20.0
QUBO_OPENJIJ_TROTTER = 32
```

The first 16-read grid around `trotter=32` found a single seed-sensitive
ground-state hit at `75/75/60/40`. A later multi-seed scan prioritized connected
overlap layouts instead of exact ground-state hits. The current starting point
above was valid for all tested seeds and usually kept missing adjacent overlap
edges to one or two. Treat it as a local tuning result for the current demo
scale, not a universal optimum.

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

Quick WSL shell from VSCode:

- Open the terminal dropdown and choose `OLC WSL venv`.
- Or run `Ctrl+Shift+P` -> `Tasks: Run Task` -> `OLC: WSL venv shell`.
- Or run from Windows PowerShell:

```powershell
.\olc_overlap_framework\tools\enter_wsl_venv.ps1
```

All three entry points run:

```bash
cd /mnt/d/Pytnon/olc_overlap_framework_v0_1_0/olc_overlap_framework
source .venv/bin/activate
```

## Notes

- `OriginalCandidateFinder` is a reserved interface for the custom/original overlap algorithm.
- `ORToolsLayoutSolver` is intentionally left unimplemented.
- The first version only handles same-strand (`+`) suffix-prefix overlaps. Reverse-complement handling should be added in a later version.
