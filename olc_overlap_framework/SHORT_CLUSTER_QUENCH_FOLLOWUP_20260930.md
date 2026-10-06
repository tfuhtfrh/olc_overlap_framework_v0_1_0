# Short-cluster search and generic quench follow-up — 2026-09-30

## 1. Trotter slice count vs structured cluster length

P remains fixed at 8 throughout.

A structured update with cluster length L=2 means:
- choose one contiguous two-slice segment, e.g. slices {3,4};
- apply the same graph move to those two slices only.

It does NOT mean that the simulation has only four slices.

The Trotter discretization remains P=8, and the coupling
    K = -0.5 log tanh(beta Gamma / P)
is still computed with P=8.

Only if one artificially partitioned the worldline into four fixed,
non-overlapping two-slice blocks would one obtain something resembling four
coarse blocks. The current sampler does not do this: the cluster start is
random, clusters overlap, and repeated proposals can affect all eight slices.

## 2. Parameters with and without direct QA analogues

SQA/PIMC parameters:
- beta: inverse temperature of the simulated quantum Gibbs distribution;
- Gamma: transverse-field strength in the simulated Hamiltonian;
- P: Suzuki-Trotter discretization count;
- L: Monte Carlo imaginary-time cluster length.

Physical QA analogy:
- Gamma has a direct qualitative analogue in the QPU tunneling/transverse
  energy A(s) relative to problem energy B(s);
- beta corresponds to physical temperature relative to the Hamiltonian energy
  scale, but is generally not an independently programmable annealing schedule
  on a QPU;
- P and L are numerical Monte Carlo parameters only and have no direct QPU
  counterpart.

Thus beta/Gamma schedule tuning changes both the simulated physical model and
the efficiency of its PIMC representation, while L changes only the sampler.

## 3. Tuned short-cluster schedules

24-seed validation:

Stage 2:
    beta=6.0--6.5
    Gamma_start=3.0--3.5
    Gamma_floor=0.6
    L=2

Results:
    24/24 any-HP hit
    ~10/24 all-slice HP
    mean final HP slices ~5.3--5.7 / 8

Single-slice L=1 with beta=6.5, Gamma 3->0.6:
    24/24 any-HP hit

Stage 3:
    beta=6.0
    Gamma 4->1.0
    L=4

Results:
    24/24 any-ground hit
    15/24 all-slice ground
    late acceptance ~3.2%

Single-slice L=1 with beta=6.5, Gamma 4->0.8:
    24/24 any-ground hit
    but weak replica consensus.

Therefore whole-worldline L=8 is not required for target reachability in these
representative diagnostics.

## 4. Generic final quench

A topology-agnostic final quench was then tested.

Search stage:
- Stage 2: beta=6.5, Gamma 3->0.6, L=2 endpoint-transfer search;
- Stage 3: beta=6.0, Gamma 4->1.0, L=4 R3 search.

Quench stage:
- disable endpoint/R2/R3 structured proposals completely;
- use generic single-bit PIMC Metropolis updates;
- full weight+degree+count BQM;
- Gamma floor -> 0.03;
- compare 250, 1000, 4000 quench attempts.

Result:
    mean accepted quench moves = 0
for every tested quench length in both Stage 2 and Stage 3.

Hence the quench did not generate replica consensus. It simply froze and
preserved the finite-Gamma endpoint distribution.

Stage 2 final search distribution:
    23/24 runs had at least one HP slice at the end;
    mean HP slices = 5.33 / 8;
    fixed slice-0 HP hits = 14/24;
    random-slice HP probability ~0.667.

Stage 3 final search distribution:
    14/24 runs had at least one ground slice at the end;
    mean ground slices = 4.125 / 8;
    fixed slice-0 ground hits = 12/24;
    random-slice ground probability ~0.516.

The lower Stage-3 final-any count compared with the earlier 24/24 result is not
a contradiction: the earlier metric was "ever reached ground"; this quench
experiment deliberately measures the state present at the END of a fixed
search budget, without ground-score early stopping.

## 5. Interpretation

All-slice consensus is a useful PIMC convergence diagnostic but is not a
necessary definition of QA readout success. Trotter slices are not independent
physical copies of the QPU. A slice marginal is the more appropriate analogue
for a classical configuration readout.

The current hierarchy is therefore:

1. whole-worldline structured update:
   strongest algorithmic convenience; target and consensus both easy;

2. tuned L=2/L=4 structured updates:
   target reachability remains high without L=P;

3. fixed-slice / random-slice readout:
   stricter and no postselection; success probability is lower but nonzero;

4. generic single-bit final quench:
   currently freezes immediately under the large degree/count penalties and
   does not improve consensus.

The next sampler improvement should therefore target generic imaginary-time
cluster updates or a pause/quench schedule, not restore topology-specific
whole-worldline moves.
