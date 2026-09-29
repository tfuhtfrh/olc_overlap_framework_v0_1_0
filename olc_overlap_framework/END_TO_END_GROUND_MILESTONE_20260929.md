# End-to-end structured SQA ground-state milestone — 2026-09-29

## Main milestone

CHM13 complex144 now reaches the independently certified weighted Hamilton-path
ground state

    W* = 1,343,093

from both zero starts and random full-space starts using an SQA-assisted
structured sampler.

Successful architecture:

    full-space OpenJij SQA
        -> structured whole-worldline topology repair
        -> feasible whole-worldline R3 / R2-pair refinement
        -> certified ground

No target path or target edge set is used by the optimizer. The certified score
is used only after the run to label a ground-state hit.

Correct description:

> SQA-assisted hybrid optimization with structured nonlocal path-integral
> update kernels.

It is not a claim that native single-spin SQA alone solves the benchmark.

## 1. First end-to-end test without immediate cyclic repair

The first integrated version used:
- OpenJij P=8 SQA on the full 261-edge binary space;
- weight + degree + count BQM;
- adaptive weak exact current-cycle cut or adaptive cycle-averaged/rank-ensemble bias;
- once any Hamilton path appeared, whole-worldline R3 and disjoint R2-pair refinement.

Three seeds per start/method:

| method | start | HP hits | ground hits |
|---|---:|---:|---:|
| exact cut | zero | 2/3 | 2/3 |
| exact cut | random | 3/3 | 3/3 |
| rank ensemble | zero | 2/3 | 2/3 |
| rank ensemble | random | 1/3 | 1/3 |

Example first feasible scores that were later refined to ground:
- 1,266,838
- 1,285,031
- 1,290,542
- 1,303,331

Thus refinement did not rely on first landing near the runner-up.

## 2. Exact feasible-path landscape under structured moves

All 192 exact Hamilton paths were enumerated.

Feasible direct moves:
- one genuine R3 reconnect;
- one simultaneous pair of disjoint R2 reconnects.

Results:
- exact Hamilton paths: 192
- structured local maxima: 1
- unique local maximum: certified ground 1,343,093
- non-ground paths with at least one improving structured neighbor: 191/191
- greedy best-improvement basin of ground: 192/192
- maximum greedy macro steps from any Hamilton path to ground: 6

Step-count distribution:
    0: 1 path
    1: 12
    2: 47
    3: 72
    4: 47
    5: 12
    6: 1

So for complex144, once any Hamilton path is reached, this structured move basis
contains no secondary feasible local optimum. This is instance-specific
evidence, not a general theorem for arbitrary OLC graphs.

## 3. Immediate topology-repair kernel

v2 inserts a whole-worldline repair episode whenever OpenJij returns

    selected edges = N-1
    degree conflicts = 0
    cycle count > 0.

Static symmetric proposal families:
1. same-head sink transfer
2. same-tail source transfer
3. R2 reconnect
4. R3 reconnect

Every proposal is applied to all Trotter slices together, so the Trotter
interaction change is exactly zero for the macro move.

After an HP is reached, refinement switches to feasible R3 / disjoint-R2-pair
proposals.

## 4. v2 result: zero/random starts both reach ground

Three seeds per method/start:

| method | start | HP hits | ground hits |
|---|---:|---:|---:|
| adaptive exact cut | zero | 3/3 | 3/3 |
| adaptive exact cut | random | 3/3 | 3/3 |
| adaptive rank ensemble | zero | 3/3 | 3/3 |
| adaptive rank ensemble | random | 3/3 | 3/3 |

Several zero-start runs reached a degree-correct cyclic state after the first
OpenJij outer solve, then structured repair produced an HP and feasible
refinement reached ground.

Examples:
- zero -> 3-cycle state -> runner-up 1,338,209 -> ground
- zero -> 3-cycle state -> ground directly
- zero -> 1-cycle state -> ground directly
- random -> 1-cycle state -> 1,326,265 -> ground
- random -> 2-cycle state -> ground directly

## 5. Six-seed robustness expansion

Compared:
- no explicit acyclicity pressure;
- adaptive exact cycle cut;
- adaptive rank-ensemble/cycle-average pressure;

from both zero and random starts.

Six seeds for each of 3 methods x 2 start modes = 36 end-to-end runs.

| method | start | ground hits |
|---|---|---:|
| none | zero | 6/6 |
| none | random | 6/6 |
| exact cut | zero | 6/6 |
| exact cut | random | 6/6 |
| rank ensemble | zero | 6/6 |
| rank ensemble | random | 6/6 |

Total:
    36 / 36 ground hits

All final scores were 1,343,093.

Interpretation:
On this benchmark, once topology-aware whole-worldline proposals are present,
move geometry dominates the remaining practical difficulty.

The fact that no explicit cycle bias also succeeds 6/6 does not make
acyclicity constraints mathematically unnecessary. Earlier exact diagnostics
showed a degree-correct cyclic state whose rank-free weight+degree energy is
lower than the ground Hamilton path. Complex144 is simply too favorable to
discriminate the long-run value of the three acyclicity representations once
the move kernel is strong.

## 6. Runner-up -> ground remains the cleanest SQA dynamics diagnostic

Runner-up:
    1,338,209

Ground:
    1,343,093

They differ by exactly 3 removed + 3 added edges, one alternating 6-cycle, i.e.
one genuine R3 and one single-read relocation.

Direct R3:
- rank-free Delta H = -0.290870109
- stale-rank A=4 Delta H = +3.709129891

At beta=4, P=8, Gamma=0.03:
- direct single-slice action estimate: +50.2519
- whole-worldline action: -1.16348

Direct path-integral benchmark:
- single-slice R3: 17/24 ground hits
- whole-worldline R3: 24/24 ground hits
- median whole-worldline first ground proposal: 37

## 7. R2-only runner-up route and why acyclicity still matters

Runner-up can also reach ground in two R2 moves:

    runner HP
      -> path + cycle
      -> ground HP

There is no feasible-only R2 route.

Rank-free energy changes:
1. runner -> cyclic intermediate: Delta H = -0.305282592
2. cyclic intermediate -> ground: Delta H = +0.014412483

Hence the cyclic intermediate is slightly lower in the rank-free
weight+degree/count energy than the true Hamilton-path ground.

This is a concrete counterexample to treating
H_weight + H_degree + H_count
as a complete Hamilton-path objective.

## 8. Penalty-scale policy

Previous experiments often used order/cycle penalties of order 1, 4, 16
without relating them to the local objective scale. That is no longer a good
default.

For the runner-up R2 route, the relevant cycle-vs-ground barrier is only
0.0144125. An exact full-cycle penalty only needs to be slightly larger than
this local difference to reverse that particular preference.

The new adaptive policy estimates the cheapest locally available
cycle-reducing R2/R3 move and chooses a small exact-cut scale

    A_exact ~= max(0.02, 1.25 * positive_local_barrier + 0.005)

with an upper clip.

Observed successful repair episodes typically used exact scales around
0.02--0.023.

For cycle-averaged rank pressure on an L-cycle, total cycle pressure is scaled
by L so that removing one cycle edge produces approximately the same local
energy pressure:

    A_ensemble,total ~= L * A_exact
    per-edge pressure ~= A_exact

Future penalty experiments should report coefficients relative to:
- local structured-move Delta H;
- smallest positive topology-repair barrier;
- typical weighted-path gaps;
- coefficient distribution of the base BQM.

Raw penalty values alone are not sufficiently interpretable.

## 9. Current algorithmic interpretation

Global full-space phase:
    OpenJij SQA on H_weight + H_degree + H_count
    with optional weak current-state acyclicity pressure.

Topology-repair phase:
    whole-worldline source/sink transfer + R2 + R3.

Feasible refinement phase:
    whole-worldline R3 + disjoint R2-pair.

If acyclicity pressure is used, it should act on cycle/SCC structure, not an
arbitrarily chosen backward edge.

## 10. Status against the research target

The immediate target

    random start or zero start reaches the certified ground state

has now been achieved on CHM13 complex144.

Current evidence:
- zero start: ground reached
- random start: ground reached
- exact-cut version: ground reached
- rank-ensemble version: ground reached
- six-seed robustness expansion including no-cycle-bias control: 36/36 total ground hits

Next scientific questions are whether the same architecture remains effective
when:
1. topology is larger or contains multiple harder complex cores;
2. weights strongly favor cyclic decompositions;
3. feasible-path structured neighborhoods have multiple local maxima;
4. cardinality/degree repair becomes the dominant barrier;
5. structured worldline kernels are compared fairly with classical structured
   MC/Tabu at matched proposal budgets.
