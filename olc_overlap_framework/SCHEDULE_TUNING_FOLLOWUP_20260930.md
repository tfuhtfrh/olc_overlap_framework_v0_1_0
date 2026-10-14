# Temperature / transverse-field schedule tuning — 2026-09-30

## Goal

Reduce Trotter freezing in Stage 2/3 so that structured updates do not need
full whole-worldline flips.

Representative diagnostics:
- Stage 2: degree-correct path+cycle -> any Hamilton path via endpoint transfer.
- Stage 3: exact runner-up -> certified ground via R3.

P=8 throughout.

## Key relation

K(beta,Gamma) = -0.5 log tanh(beta Gamma / P).

Important consequence:
- lowering Gamma too much at the *start* makes K large immediately and freezes
  short clusters;
- raising the temperature (lower beta) does not automatically reduce Trotter
  freezing, because it can increase K;
- the most effective intervention was to keep Gamma_start large while raising
  Gamma_floor.

## First schedule comparison

Baseline:
    beta = 4
    Gamma: 3 -> 0.03
    K: 0.0498 -> 2.0999

Schedules with small Gamma_start (~0.5-1) failed badly for L=1/2/4 because K
was already O(1) at the start. Whole-worldline L=8 remained mobile.

Keeping Gamma_start=3 and raising the final floor:
    3 -> 0.10 : K_final ~ 1.50
    3 -> 0.30 : K_final ~ 0.95
    3 -> 0.50 : K_final ~ 0.70

increased short-cluster mobility substantially.

## Coarse grid

Grid:
- beta in {3,4,5,6}
- Gamma_start in {2,3,4}
- Gamma_floor in {0.2,0.3,0.5,0.8}
- L in {1,2,4}
- 4 seeds/config, 2000 attempts

Promising region:
- Stage 2: beta ~ 5-6, Gamma_start ~ 3-4, Gamma_floor ~ 0.5-0.8
- Stage 3: beta ~ 6, Gamma_start ~ 4, Gamma_floor ~ 0.6-0.8

Examples:
Stage 2 L=4:
    beta=6, Gamma 3->0.5
    4/4 any-HP, 3/4 all-slice HP

Stage 3 L=1:
    beta=6, Gamma 4->0.8
    4/4 any-ground
    late acceptance ~ 0.22 in the coarse 4-seed batch

## Fine search

Fine Stage-2 neighborhood:
- beta 5.5-7
- Gamma_start 3-3.5
- Gamma_floor 0.4-0.8

Fine Stage-3 neighborhood:
- beta 5.5-7
- Gamma_start 3.5-5
- Gamma_floor 0.6-1.0

8 seeds/config, 2500 attempts.

Best stable patterns:

Stage 2:
- L=1:
    beta=6.5, Gamma 3->0.6
    8/8 any HP
    mean final HP slices ~5.0/8
- L=2:
    beta=6-6.5, Gamma 3-3.5->0.6
    8/8 any HP
    4/8 all-slice HP
    mean final HP slices ~6.75/8
- L=4:
    many configs 8/8 any HP, typically 4/8 all-slice HP

Stage 3:
- L=1:
    beta=6.5, Gamma 4->0.8
    8/8 any ground
    median first ground ~59 proposals
- L=4:
    beta=5.5, Gamma 4->0.8
    8/8 any ground
    6/8 all-slice ground
    mean final ground slices ~6.5/8

## 24-seed validation

### Stage 2

Tuned L=2:
    beta=6.0, Gamma 3.5->0.6
    24/24 any HP
    10/24 all-slice HP
    mean final HP slices = 5.25/8
    late acceptance = 0.00370

Tuned L=2:
    beta=6.5, Gamma 3.0->0.6
    24/24 any HP
    10/24 all-slice HP
    mean final HP slices = 5.67/8
    late acceptance = 0.00341

Tuned L=1:
    beta=6.5, Gamma 3.0->0.6
    24/24 any HP
    3/24 all-slice HP
    mean final HP slices = 4.67/8

Baseline L=1:
    beta=4, Gamma 3->0.03
    19/24 any HP
    1/24 all-slice HP
    mean final HP slices = 3.04/8

Baseline L=2:
    21/24 any HP
    9/24 all-slice HP
    mean final HP slices = 4.33/8

Thus tuned L=1/2 removes the need for whole-worldline updates to *reach* an HP
on this diagnostic.

### Stage 3

Tuned L=4:
    beta=6.0, Gamma 4->1.0
    24/24 any ground
    15/24 all-slice ground
    mean final ground slices = 4.17/8
    mean max ground slices = 7.0/8
    late acceptance = 0.03225

Tuned L=4:
    beta=5.5, Gamma 4->0.8
    23/24 any ground
    12/24 all-slice ground

Tuned L=1:
    beta=6.5, Gamma 4->0.8
    24/24 any ground
    0/24 all-slice ground
    mean final ground slices = 2.54/8
    late acceptance = 0.1425

Tuned L=2:
    beta=6.5, Gamma 4->1.0
    24/24 any ground
    1/24 all-slice ground

Baseline L=1:
    beta=4, Gamma 3->0.03
    20/24 any ground

Baseline L=4:
    beta=4, Gamma 3->0.03
    17/24 any ground
    late acceptance = 0

Thus the best current compromise for Stage 3 is not single-slice but L=4:
it retains short-cluster character while giving much stronger replica
consensus than L=1/2.

## Current recommended schedules

Stage 2:
    beta ~ 6-6.5
    Gamma_start ~ 3-3.5
    Gamma_floor ~ 0.6
    cluster length L ~ 2
    (L=1 is viable when only any-slice HP discovery matters)

Stage 3:
    beta ~ 6
    Gamma_start ~ 4
    Gamma_floor ~ 1.0
    cluster length L ~ 4

These are search schedules, not final readout schedules.

A final quench to small Gamma should be studied separately. The tuned search
stage intentionally stops at finite Gamma to avoid late Trotter freezing; after
the important topology move has occurred, a separate readout/quench stage can
reduce Gamma toward the classical limit.

## Interpretation

The experiments weaken the need for topology-specific whole-worldline updates:

- Stage 2 tuned L=1/2: 24/24 target hits
- Stage 3 tuned L=4: 24/24 target hits

So full L=P updates are no longer necessary for target reachability on these
diagnostics.

However, short clusters still show weaker all-slice consensus than L=8.
The next question is therefore not "can short clusters reach the target?" but:

> can a finite-Gamma short-cluster search be followed by a generic,
> non-topology-oracle quench/readout stage without losing the solution?
