# Post-Hamiltonian hybrid SQA routes — 2026-09-30

## Scope

After rank/reachability Hamiltonian tuning reached diminishing returns, three post-Hamiltonian directions were tested while keeping the inner sampler as standard OpenJij SQASampler with built-in SingleSpinFlip:
1. reverse-style incumbent-centered schedules;
2. population / multi-explorer hybrid search;
3. exact six-bit transverse-field spectral diagnostics around the certified runner-up -> ground barrier.

No result below uses the certified ground score for steering or early stopping.

## Reverse-style SQA from runner-up

Warm start: runner-up W2 = 1,338,209. Target W* = 1,343,093.
All runs used beta=5, gamma=1, P=8, 8 reads/seed, 2000 MC steps/read, 24 seeds/config.

Moderate reverse depths s_min = 0.9,0.8,0.7,0.6,0.5,0.4:
- 24/24 returned an HP;
- 24/24 returned the exact runner-up;
- 0/24 reached ground.

Deeper reverse:
- s_min=0.30: 23/24 HP, 23/24 runner, 0 ground
- s_min=0.25: 24/24 HP, 24/24 runner, 0 ground
- s_min=0.20: 21/24 HP, 21/24 runner, 0 ground
- s_min=0.15: 20/24 HP, 20/24 runner, 0 ground
- s_min=0.10: 16/24 HP, 16/24 runner, 0 ground
- s_min=0.05: 11/24 HP, 10/24 runner, 0 ground

Interpretation: shallow reverse preserves runner-up; deep reverse destroys it; no tested depth preferentially transfers to target.

## Population / multi-explorer hybrid

Fixed 8 standard reads/outer and 16 outer iterations.

Two-explorer 12-seed screening:
- single8 zero/random: 0 ground;
- beam2_energy zero: 1/12 ground, source no_rank;
- beam2_diverse zero: 1/12 ground, source no_rank;
- beam2_combo zero: 1/12 ground, source no_rank;
- beam2_combo random: 12/12 HP, best 1,327,035.

Population-size ablation, 24 seeds/start:
- K=1 zero: 11/24 HP, 0 ground, best 1,326,265
- K=1 random: 10/24 HP, 0 ground, best 1,336,620
- K=2 zero: 20/24 HP, 0 ground, best 1,322,872
- K=2 random: 20/24 HP, 0 ground, best 1,336,620
- K=4 zero: 22/24 HP, 1/24 ground, best 1,343,093
- K=4 random: 24/24 HP, 0 ground, best 1,338,209

Population K=4 preserves substantial basin diversity (mean final pairwise Hamming distance about 22 bits).

## Exact six-bit transverse-field spectrum

Runner-up and ground differ in exactly six edge bits. Freeze every other variable and enumerate all 64 states under H0 = H_weight + H_degree + H_count.

E_runner = 63.6000833779
E_target = 63.3092132689
E_target - E_runner = -0.2908701090

Within this six-bit cube the target HP is the classical global minimum.

Exact diagonalization of H(g) = H0 - g sum_i X_i gives minimum ground/first-excited gap about 0.276618 at g about 8.8.

Target probability in the quantum ground state:
- s=0.8, g=0.25: 0.99981
- s=0.6, g=0.667: 0.99869
- s=0.5, g=1: 0.99701
- s=0.3, g=2.33: 0.98408
- s=0.2, g=4: 0.95628
- s=0.1, g=9: 0.83079
- s=0.05, g=19: 0.35738

## Runner-connected excited eigenbranch

At g=0 the runner-up is an excited classical eigenstate. Tracking the continuously connected eigenbranch:
- s=0.8: runner prob 0.999811, target ~3e-21
- s=0.6: runner 0.998624, target ~5e-16
- s=0.5: runner 0.996990, target ~5e-14
- s=0.3: runner 0.984176, target ~1e-9
- s=0.2: runner 0.956017, target ~7e-7
- s=0.1: runner 0.830272, target 0.004455
- s=0.05: runner 0.377672, target 0.255664

Across g in [0,20], maximum target probability on the runner-connected branch is about 0.2613 near g=20. Minimum nearest-level separation is about 0.276618 near g=8.825.

Interpretation: the runner-connected eigenstate stays runner-like throughout moderate field and mixes significantly with target only at very strong field, where incumbent identity is already washed out. This matches reverse-style SQA behavior.

## Current direction

Population / multi-explorer rank-free search has stronger evidence than reverse-style incumbent annealing.
Recommended standard-SQA hybrid direction: population explorers + sparse topology probes + non-oracle best-feasible incumbent archive.
Reverse-style SQA remains useful as a QA-relevant diagnostic, but did not improve ground discovery under the tested schedules.
