# SQA experiment taxonomy and baseline reset — 2026-09-30

## Purpose

Separate three conceptually different layers that had become mixed during the
sampler-dynamics experiments:

1. standard SQA;
2. non-standard structure-aware PIMC/SQA updater experiments;
3. outer Hamiltonian-adaptation hybrids.

The goal is to preserve all previous results while preventing claims about
standard SQA or physical QA from inheriting performance that came from a
custom Monte Carlo transition kernel.

## A. Standard SQA

Definition used in this project:

- OpenJij SQASampler;
- built-in SingleSpinFlip updater only;
- Suzuki-Trotter replicas handled internally by OpenJij;
- no endpoint-transfer, R2, R3, multi-bit, short-cluster, or whole-worldline
  custom update;
- returned optimization sample follows OpenJij's standard rule: for each read,
  choose the lowest classical problem-energy Trotter slice; SampleSet.first
  then selects the lowest-energy returned read.

Changing beta, gamma, Trotter number P, or the annealing schedule s(t) remains
inside the standard-SQA category.

Previous standard-SQA results are retained, including the rank/cycle-cut
OpenJij experiments where every inner solve used SQASampler SingleSpinFlip.
Those outer algorithms may still be hybrid because the QUBO was rebuilt
between SQA episodes.

Important historical result:
- no previous standard OpenJij SQA experiment reached the certified weighted
  HP ground 1,343,093 on complex144;
- the strongest earlier fixed/current/rank/cycle-cut runs reached feasible HPs
  but ground_hit remained 0.

## B. Non-standard structure-aware PIMC/SQA updater experiments

These experiments keep the transverse-field path-integral structure but modify
the Monte Carlo transition kernel.

Examples:
- endpoint-transfer;
- R2;
- R3;
- disjoint R2-pair;
- applying a graph move to L=1,2,4 Trotter slices;
- whole-worldline L=P updates;
- hard HP-manifold filtering.

These should be described as, depending on the exact experiment:

- structure-aware PIMC;
- non-standard SQA updater;
- SQA-assisted structured Monte Carlo;
- constrained path-integral Monte Carlo.

They should not be reported as vanilla OpenJij SQA.

Scientific purpose:
- diagnose proposal-geometry / mixing failures of local PIMC;
- test whether collective graph-theoretic transitions connect relevant
  low-energy states in the same classical problem landscape;
- provide hypotheses about collective/high-order transitions that a physical QA
  system might or might not realize.

They do NOT demonstrate that a QPU executes endpoint/R2/R3 transitions.

## C. Hamiltonian-adaptation hybrids

Examples:
- projected rank recomputation;
- current-cycle cut generation;
- adaptive cycle-cut penalty;
- rank-ensemble recomputation.

Within each inner episode the Hamiltonian is static and can be solved by
standard SQASampler, but the total outer algorithm is a hybrid because the
Hamiltonian is rebuilt between episodes.

This layer should be kept separate from updater modifications.

## OpenJij slice output rule

OpenJij's transverse-Ising get_solution implementation evaluates the
classical problem energy of each final Trotter slice and returns the slice
with minimum classical energy.

Therefore:
- all-slice consensus is a useful equilibration/freezing diagnostic;
- fixed-slice readout is a useful physical-marginal diagnostic;
- but lowest-energy slice is the correct standard OpenJij optimization output
  and should be used for the standard-SQA baseline.

With num_reads=R, there are R returned optimization samples. It is useful to
report both:
- SampleSet.first, the standard best returned read;
- whether any of the R standard read outputs was an HP/ground.

The ground score must never be used for early stopping or proposal selection.
It is only an offline certificate.

## Baseline-A: fixed QUBO, standard SQA parameter sweep

QUBO:
    H_weight + H_degree + H_count

No acyclicity term.
No outer Hamiltonian rebuild.
No custom updater.

4 zero-start and 4 random-start runs per setting, 8 reads/run, 1400 sweeps.

Results from the first screening:

| setting | zero HP | random HP | ground |
|---|---:|---:|---:|
| beta=5,gamma=1,P=8 old default | 1/4 | 1/4 | 0 |
| beta=6,gamma=1,P=8 | 0/4 | 0/4 | 0 |
| beta=6,gamma=2,P=8 | 0/4 | 0/4 | 0 |
| beta=6,gamma=4,P=8 | 0/4 | 0/4 | 0 |
| beta=8,gamma=2,P=8 | 0/4 | 0/4 | 0 |
| beta=6,gamma=2,P=16 | 1/4 | 1/4 | 0 |
| tested pause/quench variants | 0/4 | 0/4 | 0 |

The old default was at least as good as the hand-tuned alternatives in this
small screening.

Interpretation:
physical/schedule parameter tuning alone did not remove the principal
difficulty of the fixed weight+degree+count Hamiltonian. The fixed QUBO also
does not encode acyclicity completely, so this baseline should not be treated
as a complete HP solver.

## Current baseline plan

A. Fixed-QUBO vanilla standard SQA:
   identify what beta/gamma/P/schedule tuning alone can do.

B. Standard-SQA inner solver + Hamiltonian acyclicity variants:
   compare weak exact current-cycle cuts and rank-ensemble, with no custom
   updater and no oracle early stop.

C. Compare B against the already completed non-standard structured updater
   experiments at matched problem/start conditions.

This separation allows the project to ask two different questions cleanly:

1. Is the Hamiltonian representation useful under a standard transverse-field
   SQA solver?
2. How much of the observed difficulty is caused by the local Monte Carlo
   transition kernel rather than by the energy landscape itself?
