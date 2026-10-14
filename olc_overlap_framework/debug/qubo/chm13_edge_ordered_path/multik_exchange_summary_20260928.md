# Multi-k fixed-cardinality exchange follow-up (2026-09-28)

## Design

Keep the successful two-phase schedule:

1. outer iterations 0..11: exchange-only degree repair;
2. outer iterations 12..17: 50/50 exchange + directed reconnect.

Compare exchange move-size distributions:
- k1: 100% 1<->1 exchange
- k12: 70% k=1, 30% k=2
- k123: 60% k=1, 30% k=2, 10% k=3
- k123_wide: 40% k=1, 40% k=2, 20% k=3

A k-exchange selects k currently-on and k currently-off edge bits uniformly
and flips all 2k bits.  Because the Hamming weight is fixed at 143, the proposal
is symmetric:
  p_k / [C(143,k) C(118,k)]
in both directions.

Settings:
- 8 Trotter slices
- 1500 sweeps
- 2 reads
- 18 outer iterations
- 4 seeds/configuration
- beta=4
- gamma 3 -> 0.03

## Results

No configuration reached a Hamilton path in this particular 4-seed batch.

Aggregate minimum degree conflicts:
- k1: 1
- k12: 1
- k123: 1
- k123_wide: 0 (one run reached degree-correct state but retained one cycle)

Mean degree conflicts at the end of Phase A (iteration 11):
- k1: 2.5
- k12: 2.5
- k123: 2.5
- k123_wide: 3.5

Acceptance over the full runs:
- k1 exchange: ~0.99%
- k2 exchange: ~0.39-0.44%
- k3 exchange: ~0.26-0.27%
- eligible reconnect: ~8-9%

Thus naive larger-k exchange does not improve degree repair at these proposal
weights.  Larger-k proposals are accepted substantially less often, and using
them too heavily (k123_wide) worsens the mean Phase-A degree residual, although
one seed did reach degree=0.

This does not rule out k>1.  It suggests that uniform random k-edge exchange is
too unstructured.  More promising next variants are:
- retain k1 as the dominant degree-repair kernel;
- invoke k2/k3 only late, when k1 acceptance collapses;
- bias k2/k3 proposal construction toward known degree-conflict vertices while
  preserving forward/reverse proposal symmetry (or include Hastings correction);
- keep reconnect as the topology/cycle kernel after degree repair.

## Important terminology

Multi-bit MCMC proposals by themselves do NOT imply a non-standard quantum
driver.  If the full binary state space is retained and the same
Suzuki-Trotter effective distribution for the standard transverse-field
Hamiltonian is sampled with detailed balance, multi-bit/cluster updates are
only Monte-Carlo acceleration.

The non-standard aspect of the present benchmark is the hard restriction to
the fixed-Hamming-weight sector.  Standard independent transverse field does
not preserve that sector.
