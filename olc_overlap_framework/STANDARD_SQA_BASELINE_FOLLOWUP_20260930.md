# Standard SQA baseline follow-up — 2026-09-30

## Scope

This note resets the baseline after the structure-aware PIMC experiments.

Definition of standard SQA in this project:
- OpenJij SQASampler;
- built-in SingleSpinFlip updater only;
- no endpoint, R2, R3, short-cluster, or whole-worldline custom proposal;
- OpenJij standard optimization output is used:
  each read returns its lowest-classical-energy Trotter slice;
  SampleSet.first is the lowest-energy returned read.

Ground score 1,343,093 is never used to stop, steer, or select proposals.
It is used only offline to certify a hit.

Outer Hamiltonian rebuilding (current cycle cuts / rank ensemble) is labeled
a hybrid at the outer level even though every inner solve is standard SQA.

---

## 1. Baseline A — fixed QUBO, standard SQA parameter sweep

Fixed QUBO:
    H_weight + H_degree + H_count

No acyclicity term.
No custom updater.

4 zero-start and 4 random-start runs per configuration.
8 reads/run, 1400 sweeps/read.

| setting | zero HP | random HP | ground |
|---|---:|---:|---:|
| beta=5, gamma=1, P=8 old default | 1/4 | 1/4 | 0 |
| beta=6, gamma=1, P=8 | 0/4 | 0/4 | 0 |
| beta=6, gamma=2, P=8 | 0/4 | 0/4 | 0 |
| beta=6, gamma=4, P=8 | 0/4 | 0/4 | 0 |
| beta=8, gamma=2, P=8 | 0/4 | 0/4 | 0 |
| beta=6, gamma=2, P=16 | 1/4 | 1/4 | 0 |
| tested pause/quench variants | 0/4 | 0/4 | 0 |

Best HP scores:
- old default zero: 1,275,205
- old default random: 1,295,426
- P16 zero: 1,285,115
- P16 random: 1,258,457

Conclusion:
standard-SQA parameter tuning alone did not solve the fixed
weight+degree+count Hamiltonian. Larger gamma and the tested pause/quench
schedules often degraded degree/count repair. The old default was at least as
good as the hand-tuned variants in this screening.

This does not establish a limitation of QA itself; it is a limitation of this
finite-P local-update SQA baseline and of an incomplete acyclicity Hamiltonian.

---

## 2. Known local barriers with vanilla standard SQA

24 seeds, 8 reads/seed, 2000 sweeps.
No custom proposals.

### Stage-2 degree-correct path+cycle checkpoint

Target: any valid Hamilton path.

| Hamiltonian / setting | primary HP hits |
|---|---:|
| no cycle cut, beta=5 gamma=1 P=8 | 2/24 |
| no cycle cut, beta=6 gamma=2 P=16 | 4/24 |
| weak exact current-cycle cut A=0.02, P=8 | 2/24 |
| weak exact current-cycle cut A=0.02, P=16 | 2/24 |

The weak exact cut did not improve this local vanilla-SQA barrier.

### Stage-3 runner-up 1,338,209 warm start

Target: certified ground 1,343,093.

| setting | ground hits | HP outputs |
|---|---:|---:|
| beta=5 gamma=1 P=8 | 0/24 | 1/24 |
| beta=6 gamma=2 P=16 | 0/24 | 2/24 |

No one of the 8 standard reads in any of the 48 runs produced ground.

Most outputs fell back into degree-correct cyclic states.

This is a clean contrast with the non-standard structure-aware R3 experiments,
where the same runner-up -> ground collective transition was highly accessible.

Interpretation:
the collective transition exists in the problem landscape, but vanilla
single-spin finite-P PIMC does not sample it efficiently under these settings.
This does not prove or disprove a corresponding high-order transition on a
physical QA device.

---

## 3. Baseline B — standard SQA inner solver + acyclicity Hamiltonian variants

Every inner episode:
- OpenJij SQASampler;
- SingleSpinFlip only;
- 8 standard reads;
- fixed 12 outer episodes;
- no ground early stop.

Starts:
- zero
- random.

Acyclicity methods:
1. none;
2. weak current exact cycle cut A=0.02;
3. current rank-ensemble with per-cycle-edge pressure 0.02.

Schedules:
- old_default: beta=5, gamma=1, P=8;
- P16: beta=6, gamma=2, P=16.

### old_default

| method | zero HP runs | random HP runs | best HP | ground |
|---|---:|---:|---:|---:|
| none | 3/4 | 1/4 | 1,319,792 | 0 |
| exact cut | 3/4 | 3/4 | 1,318,086 | 0 |
| rank ensemble | 3/4 | 2/4 | **1,343,093** | **1 random run** |

The weak exact cut improved random-start HP incidence from 1/4 to 3/4 in this
small batch, but did not reach ground.

Rank ensemble produced the first certified ground hit obtained with:
- no custom PIMC updater;
- no ground oracle;
- standard OpenJij SingleSpinFlip inner SQA.

The full algorithm is still an outer Hamiltonian-adaptation hybrid.

### P16

| method | zero HP runs | random HP runs | best HP | ground |
|---|---:|---:|---:|---:|
| none | 2/4 | 3/4 | 1,308,819 | 0 |
| exact cut | 2/4 | 2/4 | 1,324,816 | 0 |
| rank ensemble | 3/4 | 3/4 | 1,308,819 | 0 |

P=16 did not improve ground-state performance in this batch.

---

## 4. Exact trace of the standard-SQA ground hit

Configuration:
    old_default
    rank_ensemble
    random start
    seed 202619302

Outer trace:

| iteration | selected | degree conflicts | cycles | HP score |
|---:|---:|---:|---:|---:|
| 0 | 142 | 0 | 3 | — |
| 1 | 143 | 0 | 4 | — |
| 2 | 142 | 0 | 1 | — |
| 3 | 143 | 0 | 1 | — |
| 4 | 143 | 0 | 0 | **1,343,093** |
| 5 | 143 | 0 | 3 | — |
| 6 | 143 | 0 | 3 | — |
| 7 | 143 | 0 | 2 | — |
| 8 | 143 | 0 | 2 | — |
| 9 | 143 | 0 | 2 | — |
| 10 | 143 | 0 | 1 | — |
| 11 | 143 | 0 | 3 | — |

At iteration 4, the standard OpenJij primary output itself is the certified
ground; this is not merely a hidden read.

At iteration 5 the state immediately leaves the HP and returns to a three-cycle
state.

Reason:
the rank-ensemble is based on current cycles. At the HP, current cycle count is
zero, so the next episode has no acyclicity bias. The base
weight+degree+count Hamiltonian again permits lower-energy cyclic states.

Thus:
- search can discover the ground;
- current-state-only acyclicity guidance does not dynamically retain it.

---

## 5. Incumbent retention without an oracle

There are two different meanings of "retain the best result."

### Physical / sampler-state retention

Does the next SQA episode remain in the ground state?

Current answer: no in the observed successful trace.

### Optimization incumbent retention

Maintain:
    best_valid_HP = argmax path_score
over all valid HP samples observed so far.

This uses only:
- Hamilton-path feasibility, which is known;
- the overlap objective, which is known.

It does NOT use the unknown optimal score 1,343,093.

Therefore best-feasible incumbent storage is a legitimate non-oracle solver
mechanism. Under this rule the iteration-4 ground is retained as final output
even though later SQA states leave it.

Future solver reports should distinguish:
- final Markov-chain state;
- best feasible incumbent seen during the fixed budget.

---

## 6. Retention variants

A separate fresh four-seed batch tested:

1. accumulated weak exact cycle cuts A=0.02;
2. sticky rank-ensemble, where the most recent cycle guidance remains active
   when the current state becomes acyclic.

Results:

| method | start | ground ever | final ground | best incumbent |
|---|---|---:|---:|---:|
| accumulated exact | zero | 0/4 | 0/4 | 1,322,872 |
| accumulated exact | random | 0/4 | 0/4 | 1,298,149 |
| sticky ensemble | zero | 0/4 | 0/4 | 1,296,875 |
| sticky ensemble | random | 0/4 | 0/4 | 1,303,094 |

These variants did not improve reproducibility in this small fresh-seed batch.
Accumulated exact cuts reached up to 16 active cuts and likely reintroduced
auxiliary/coupler sampling overhead. Sticky linear ensemble pressure is not an
exact acyclicity encoding and may distort valid paths.

This does not invalidate the single standard-SQA ground hit; it shows that the
hit is not yet robust.

---

## 7. Current interpretation

The evidence now separates three claims:

### A. Fixed-QUBO vanilla SQA
No ground hit in the tested parameter sweep.

### B. Standard SQA + better Hamiltonian guidance
One certified ground hit has now occurred with current rank-ensemble guidance,
using only OpenJij SingleSpinFlip in each inner episode.

### C. Non-standard structure-aware PIMC
Collective endpoint/R2/R3 updates make the known topology transitions vastly
more accessible and can solve the instance much more reliably.

The appropriate interpretation of C is diagnostic:
it demonstrates that collective low-dimensional topology transitions connect
the relevant states and exposes limitations of local PIMC mixing. It can
motivate hypotheses about high-order QA transitions, but it is not itself a
QPU dynamics model.

---

## 8. Baseline status

Preserve all three result families:

1. historical standard-SQA rank/cycle-cut results;
2. new vanilla standard-SQA baselines and the single no-oracle standard-SQA
   ground hit;
3. non-standard structure-aware PIMC/SQA experiments as sampler-dynamics
   extensions.

The next standard-SQA work should focus on:
- increasing statistical confidence around the rank-ensemble ground hit;
- non-oracle incumbent reporting;
- Hamiltonians that retain acyclicity semantics without large auxiliary
  overhead;
rather than further modifying the Monte Carlo updater.
