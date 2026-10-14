# Standard-SQA hybrid: current direction and next tests — 2026-09-30

## 1. Current main line

The current main line is **SQA hybrid optimization with a standard SQA inner sampler**.

For this project, "standard SQA" means:

- OpenJij `SQASampler`;
- built-in `updater="single spin flip"`;
- Suzuki–Trotter replicas handled internally by OpenJij;
- no endpoint-transfer, R2, R3, multi-bit, cluster, or whole-worldline custom updater;
- no hard Hamilton-path-manifold filtering.

Outer classical control is allowed. In particular, we may:

- rebuild or perturb the Hamiltonian between SQA calls;
- maintain several explorers;
- change warm starts;
- keep a best-feasible incumbent archive;
- allocate a fixed read budget among several standard-SQA probes.

Therefore the target architecture is an **SQA-classical / QA-compatible hybrid**, not a custom structure-aware Monte Carlo kernel.

The earlier endpoint/R2/R3/whole-worldline experiments remain useful as diagnostics of the landscape and proposal geometry, but their success must not be counted as standard-SQA ground discovery.

---

## 2. Current complex144 status

Benchmark:

- CHM13 complex144;
- (N=144);
- 261 edge variables in the current edge-selection space;
- certified weighted Hamilton-path ground score:
  (W^*=1,343,093);
- frequent runner-up:
  (W_2=1,338,209).

The practical state of the solver is now approximately:

> Standard SQA often reaches valid, high-quality Hamilton paths and frequently reaches or approaches the runner-up region, but direct ground-state discovery is still a low-probability event.

Thus the main bottleneck is no longer simply "find any Hamilton path". It is:

[
	ext{high-quality HP basin} ightarrow 	ext{certified ground basin}.
]

The certified ground score is used only for offline evaluation. It is not used for steering, proposal selection, or early stopping.

---

## 3. Results so far

### 3.1 Fixed-QUBO standard SQA

Base Hamiltonian:

[
H_0 = H_{mathrm{weight}} + H_{mathrm{degree}} + H_{mathrm{count}}.
]

Simple sweeps over beta / gamma / Trotter number / schedule variants did not remove the main difficulty.

The fixed-(H_0) standard-SQA baseline can produce feasible HPs, but the tested parameter screens did not produce a certified ground hit.

Interpretation:

- physical/schedule tuning alone is not currently enough;
- the difficulty appears more structural than a simple bad choice of beta/gamma/P.

### 3.2 Hamiltonian-adaptation hybrid with standard SQA

Rank, reachability, and cycle-related probes were tested while keeping every inner solve as standard SingleSpinFlip SQA.

Important observations:

- deterministic projected-rank guidance can be harmful;
- rank dropout / rank mixture are less destructive;
- sparse reachability guidance is more promising than a fixed total order;
- one reachability-guided random-start run directly produced the certified ground as a standard SQA output.

This is a genuine standard-SQA inner-solver ground hit: the ground was not produced by R2/R3 repair or post-processing.

### 3.3 Population / multi-explorer hybrid

A rank-free population-size ablation used a fixed budget:

- 8 standard SQA reads per outer iteration;
- 16 outer iterations;
- beta=5, gamma=1, P=8;
- 1400 sweeps/read;
- no rank, reachability, or custom updater.

Results:

| population | start | runs with HP incumbent | ground |
|---|---|---:|---:|
| K=1 | zero | 11/24 | 0/24 |
| K=1 | random | 10/24 | 0/24 |
| K=2 | zero | 20/24 | 0/24 |
| K=2 | random | 20/24 | 0/24 |
| K=4 | zero | 22/24 | 1/24 |
| K=4 | random | 24/24 | 0/24 |

The K=4 zero-start case therefore provides another direct ground hit from standard SingleSpinFlip SQA.

The useful signal is not only the single hit: larger populations also substantially increased the probability of maintaining feasible/high-quality regions.

Current interpretation:

> maintaining several distinct basins appears more useful than repeatedly exploiting one incumbent.

### 3.4 Reverse-style SQA from the runner-up

Reverse-like / reverse-style schedules were tested from the exact runner-up with standard SingleSpinFlip SQA.

Observed behavior:

- shallow reverse schedules mostly return the runner-up;
- deeper schedules increasingly destroy the runner-up;
- no tested schedule produced a ground hit.

Thus reverse annealing is currently not a promising primary direction.

The exact six-bit runner-up/ground spectral diagnostic is consistent with this result:

- the runner-up-connected excited branch stays strongly runner-like over a broad field range;
- strong field eventually mixes it with the target, but also washes out the incumbent identity.

So there is no clear intermediate reverse depth that preferentially transfers runner-up -> ground.

---

## 4. Current interpretation

The present evidence suggests:

1. Standard SQA can reach the correct low-energy region.
2. Feasibility is no longer the dominant bottleneck on complex144.
3. The search often collapses into one strong near-optimal basin.
4. Warm-start exploitation of the runner-up does not solve the transition problem.
5. Diversity across independent standard-SQA episodes is currently more promising than deeper exploitation of a single incumbent.

Therefore the optimization target should be formulated as increasing

[
P(	ext{a standard SQA read produces }x^*)
]

under a fixed total resource budget.

The final explorer does not need to remain at the ground. A classical outer controller may keep the best valid HP observed so far; this is non-oracular and QA-compatible.

---

## 5. Recommended next tests

### Priority A — population diversity

Continue the K=4 / multi-explorer direction, but replace simple "lowest-(H_0) distinct states" selection with explicit diversity preservation.

Possible selectors:

- one elite by base energy + remaining members by max-min Hamming distance;
- quality-diversity tradeoff;
- topology-aware diversity using cycle pattern, endpoints, SCC pattern, or selected-edge signatures.

Goal:

> prevent several explorers from becoming different bitstrings inside the same effective basin.

### Priority B — elite-frequency-guided diversification

Instead of repelling every edge of one incumbent, estimate edge frequencies over an elite archive:

[
f_e = rac{#{	ext{elite states containing }e}}{#	ext{elite states}}.
]

Interpretation:

- (f_e approx 1): stable backbone, normally keep it;
- intermediate (f_e): uncertain / competing edge region;
- use weak probe biases mainly on uncertain edges.

This should be less destructive than uniform incumbent repulsion, especially because runner-up and ground share almost all edges.

### Priority C — alternating-cycle Hamiltonian probes

Use the topology knowledge learned from R2/R3 diagnostics **without changing the proposal kernel**.

From a current high-quality HP, identify short non-oracular alternating-edge candidates, for example 4- or 6-edge reconnection structures suggested by high-reward unused edges.

Construct weak probe Hamiltonians such as

[
H_k = H_0 + epsilon H_{mathrm{alt},k},
]

then solve each (H_k) with ordinary `SQASampler(..., updater="single spin flip")`.

Important distinction:

- allowed: encode a weak structural preference in a static Hamiltonian and run standard SQA;
- not mainline: directly execute an R2/R3 multi-bit Monte Carlo move.

This is currently one of the most direct ways to transfer the useful topology insight from the custom-updater experiments back into the standard-SQA framework.

### Priority D — reduce the dense count-term burden

Current cardinality term:

[
A_{mathrm{count}}left(sum_e x_e-(N-1)ight)^2
]

creates dense quadratic couplings among edge variables.

Possible tests:

1. replace the square temporarily by an outer-updated chemical-potential term,
   [
   H = H_{mathrm{weight}} + H_{mathrm{degree}} + mu sum_e x_e,
   ]
   and adjust (mu) between SQA episodes;

2. test a sparse auxiliary encoding of cardinality, accepting extra variables in exchange for fewer dense couplers.

Question:

> for standard SingleSpinFlip SQA, is a somewhat larger but sparser BQM easier than the current smaller but highly dense BQM?

### Priority E — multi-Hamiltonian islands

Maintain several standard-SQA islands simultaneously, for example:

- rank-free (H_0);
- weak reachability probe;
- one or more weak alternating-cycle probes.

Each island uses only standard SQA. Warm starts or population members may be exchanged periodically by the outer controller.

This is preferable to forcing all search pressure into one adaptive Hamiltonian.

---

## 6. Evaluation protocol for the next stage

Ground hits are still rare enough that 1/24-style observations should be treated as signals, not strong statistical conclusions.

Future comparisons should use the same total resource budget and report at least:

- HP hit rate;
- runner-up hit rate;
- ground hit rate;
- distribution of best feasible incumbent score;
- first-ground read / outer iteration;
- population diversity;
- source branch of each incumbent improvement;
- total reads and total sweeps.

Larger seed counts are needed before claiming one hybrid policy is better than another.

---

## 7. Working priority

Current recommended order:

1. population diversity;
2. elite-frequency-guided diversification;
3. alternating-cycle Hamiltonian probes;
4. count-term sparsification / chemical-potential experiments;
5. multi-Hamiltonian island variants.

Reverse-style SQA remains useful as a negative/dynamics diagnostic, but is not the current primary optimization route.
