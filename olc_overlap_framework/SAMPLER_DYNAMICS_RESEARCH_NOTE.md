# Active Research Note: Sampler Dynamics, Effective Exchange, and Reverse Annealing

> **Status:** active research note for branch `benchmark/qubo-solvers-20260922`  
> **Scope:** CHM13 complex144 and follow-up OLC Hamilton-path solver work  
> **Purpose:** preserve the reasoning that led from static-QUBO failures to sampler-level optimization, and record the two main ideas that should be revisited later: **effective exchange dynamics under a strong count penalty** and **reverse annealing as local refinement**.

## 1. Why this note exists

The original experiments focused on finding a compact static QUBO for a weighted Hamilton path on sparse OLC graphs.  Several formulations were tested:

- edge-position encoding;
- independent edge-selection plus edge-position certificate;
- bounded vertex-rank encodings;
- strict latent-rank comparator encodings;
- endpoint-free exact-count encodings.

A repeated empirical pattern emerged:

> a graph-theoretically small path change can require many coordinated QUBO-bit changes.

Static full-QUBO SQA and generic block solvers therefore became trapped even when the formulation itself had a correct ground state.

This motivated the projected edge-only hybrid:

1. only real edge-selection bits are sampled;
2. rank/order information is projected classically;
3. the edge-only QUBO is rebuilt between outer iterations.

On CHM13 complex144 this changed the practical regime completely:
- projected OpenJij SQA repeatedly found valid Hamilton paths;
- projected Tabu reached the certified weighted optimum;
- a feasible SQA path only ~1.2% below optimum could be repaired to the optimum by two legal relocation moves.

The next research question therefore became:

> **Can we keep the Hamiltonian essentially fixed and improve the transition kernel so that one Monte-Carlo move corresponds better to a meaningful OLC topology move?**

That question produced the fixed-cardinality exchange/reconnect experiments and the two ideas recorded below.

---

## 2. Two sampler routes that should be kept in parallel

### Route A — full binary space, standard problem Hamiltonian

State space:

[
Omega = {0,1}^{M}.
]

Keep the count penalty in the Hamiltonian:

[
H_{m count}
=
A_{m count}
left(sum_e x_e-(N-1)ight)^2.
]

Use a mixture of symmetric Monte-Carlo kernels, for example:

- single-bit flips, which change cardinality;
- (1leftrightarrow1) exchange;
- (kleftrightarrow k) exchange;
- directed reconnect / 2-switch moves.

If the kernels are symmetric (or use the correct Metropolis-Hastings factor) and the target path-integral action is unchanged, multi-bit proposals are a **sampling acceleration**, not a change of the target Hamiltonian.

This route is the higher-priority route because it retains the full state space and is the cleanest conceptual comparison with a standard transverse-field QPU formulation.

### Route B — hard fixed-cardinality state space

Restrict every sampled slice to

[
Omega_{N-1}
=
left{
xin{0,1}^{M}:
sum_e x_e=N-1
ight}.
]

Then (H_{m count}equiv0) on the accessible state space and can be removed computationally.

This route has two practical advantages:
- the dense count-square coupling disappears;
- exchange/reconnect kernels operate directly on the low-energy cardinality manifold.

However, this restricted state space is not the full configuration space of the ordinary independent transverse-field driver.  It should therefore be treated as a **constrained path-integral / SQA-inspired algorithmic branch**, not automatically identified with standard transverse-field SQA.

Both routes should be preserved because they answer different questions:
- Route A: can better MCMC dynamics improve a QPU-compatible Hamiltonian?
- Route B: how much performance is gained by building a known hard constraint directly into the sampler geometry?

---

## 3. How the fixed-cardinality idea appeared

The endpoint-free projected formulation used

[
H_{m count}
=
A_{m count}
left(sum_e x_e-(N-1)ight)^2.
]

For CHM13 complex144, (N=144), (M=261), so a feasible Hamilton path has exactly 143 selected edges.

Expanding the count square creates a dense all-to-all edge coupling.  The observation was:

> if the sampler starts with exactly 143 selected edges and every proposal switches the same number of 1-bits and 0-bits, then the count constraint is satisfied identically and the dense count penalty is unnecessary.

This led to exchange moves:

[
k	ext{ selected edges }1	o0,
qquad
k	ext{ unselected edges }0	o1.
]

At fixed cardinality the uniform (k)-exchange proposal is symmetric:

[
q_k(x	o x')
=
rac{p_k}
{inom{143}{k}inom{118}{k}}
=
q_k(x'	o x).
]

A second complementary kernel was the directed reconnect / 2-switch:

[
(a	o b,;c	o d)
longleftrightarrow
(a	o d,;c	o b).
]

Reconnect preserves:
- selected-edge count;
- every vertex's in-degree;
- every vertex's out-degree.

This made the empirical roles of the two kernels distinct:

- **exchange:** repairs the degree profile;
- **reconnect:** changes topology/cycles while preserving an already good degree profile.

One successful CHM13 run showed exactly this sequence:

[
	ext{random 143-edge state}
	o
	ext{degree repair by exchange}
	o
	ext{Hamilton path of score }1{,}326{,}008
	o
	ext{reconnect phase}
	o
	ext{certified optimum }1{,}343{,}093.
]

This is why future kernel scheduling should be phase-aware rather than using reconnect heavily from the beginning.

---

## 4. Effective second-order exchange under the full-space Hamiltonian

This idea arose when comparing the hard fixed-cardinality sampler with the full-space, QPU-compatible formulation.

Consider the ordinary count-penalized Hamiltonian plus a standard transverse-field driver:

[
H
=
H_P
-
Gammasum_isigma_i^x,
]

with

[
H_P
supset
A_{m count}
left(sum_i x_i-Kight)^2.
]

Suppose the system is in a low-energy state with exactly (K) selected bits.  A single transverse-field flip leaves the cardinality sector:

[
10	o00
]

or

[
10	o11.
]

Those intermediate states pay a count-penalty gap (Delta_{m count}).  A second flip can return to the original cardinality sector:

[
10
	o
00
	o
01,
]

or equivalently through the (11) intermediate state.

Therefore, after eliminating high-energy off-cardinality states, the low-energy effective dynamics can contain an exchange-like process

[
10leftrightarrow01.
]

Perturbatively, its scale is expected to be of order

[
J_{m eff}
sim
rac{Gamma^2}{Delta_{m count}},
]

up to model-dependent signs, factors, and contributions from the rest of (H_P).

### Why this is useful

This gives a conceptual bridge between the two routes:

- hard fixed-cardinality exchange explicitly performs a (10leftrightarrow01) move;
- the full-space transverse-field model can generate a related low-energy process indirectly through virtual off-cardinality states.

The two dynamics are **not claimed to be identical**.  The point is that the constrained exchange kernel may be viewed as directly accelerating a transition that the strongly count-penalized full model can only realize through higher-order motion.

### Follow-up directions

1. Derive the effective low-energy Hamiltonian more carefully using degenerate perturbation theory / Schrieffer-Wolff reduction for the count-penalty sector.
2. Quantify how the effective exchange scale depends on (A_{m count}), (Gamma), and other local energy terms.
3. Compare:
   - standard single-bit full-space SQA;
   - full-space SQA with multi-bit MCMC acceleration;
   - hard fixed-cardinality exchange sampler.
4. Test whether too-large (A_{m count}) suppresses the effective exchange rate by making (Delta_{m count}) excessively large.
5. For future QPU work, test whether moderate count penalties plus reverse annealing outperform extremely strong count penalties.

---

## 5. Multi-bit MCMC does not automatically imply a non-standard quantum driver

A critical distinction was clarified during this discussion.

Using a multi-bit Monte-Carlo proposal does **not**, by itself, mean that the simulated quantum Hamiltonian has changed.

If the sampler still targets the same Suzuki-Trotter effective distribution for the standard transverse-field Hamiltonian and maintains detailed balance/ergodicity, one may use:
- single-spin updates;
- worldline updates;
- cluster updates;
- symmetric multi-spin proposals.

These are different MCMC kernels for the same target distribution.

The non-standard step in the present constrained experiments is instead:

> **forbidding all states outside the fixed-Hamming-weight sector.**

The standard independent driver

[
-Gammasum_isigma_i^x
]

does not conserve Hamming weight, so the hard-constrained route is not automatically an exact realization of that driver.

This is why the full-space route should remain the primary QPU-compatible line of work.

---

## 6. Reverse annealing: how the idea entered the project

Reverse annealing came up only after the solver could already produce valid Hamilton paths.

The motivating observation was:

- global feasibility and local weighted refinement are different tasks;
- once a good Hamilton path exists, restarting a full forward anneal is often wasteful;
- complex144 has many valid Hamilton paths, including separated local-move basins;
- a good path may need a coordinated transition through temporarily worse/off-manifold states to reach a better path.

This is precisely the regime in which reverse annealing is conceptually attractive.

### Important distinction from the current code

The current projected SQA runs are **not reverse annealing**.

They may warm-start the next outer iteration from a previous sample, but each inner anneal still uses a forward-style schedule with large fluctuations first and small fluctuations at the end.

A genuine reverse schedule starts from a classical state at (approximately) zero transverse field, increases the fluctuation strength to partially delocalize around that seed, optionally pauses, and then returns to the classical end of the schedule.

Conceptually:

[
x_0
	o
	ext{increase quantum fluctuations}
	o
	ext{local/basin exploration}
	o
	ext{return to a classical sample}.
]

### Intended future use

Reverse annealing should **not** be the first-choice mechanism for finding the first Hamilton path from a random state.

More natural use cases are:

1. **weighted refinement:** start from a valid but suboptimal Hamilton path;
2. **stagnation escape:** start from a path repeatedly rediscovered by forward search;
3. **multi-seed local refinement:** reverse-anneal several distinct valid paths;
4. **basin-crossing study:** use complex144's known multi-path structure to study how reverse depth affects transitions between path basins.

Useful future parameters include:
- reverse depth / minimum anneal parameter;
- pause duration;
- number of reverse cycles;
- count/degree penalty strengths during reverse annealing;
- seed path quality.

A particularly interesting future comparison is:

[
	ext{classical reconnect/LNS}
quad	ext{vs}quad
	ext{reverse SQA}
quad	ext{vs}quad
	ext{QPU reverse annealing}.
]

---

## 7. Current empirical kernel results that motivate the next tests

### Fixed-cardinality exchange/reconnect

Initial random fixed-cardinality experiments showed:
- exchange is much better than reconnect at repairing degree conflicts;
- reconnect is useful only after the degree profile is already close to path-like;
- a time-phased exchange-then-reconnect run reached the certified optimum.

### Naive (k>1) exchange

Uniform random (k=2,3) exchange was tested.

Observed acceptance rates were approximately:
- (k=1): ~1%;
- (k=2): ~0.4%;
- (k=3): ~0.26%.

Using large-(k) proposals heavily from the beginning did not improve the average degree-repair result.

The next experiment should therefore **not** simply increase (k).  Instead:
- keep (k=1) dominant early;
- activate (k=2,3) late, when (k=1) acceptance collapses;
- make several independent (k=2,3) proposal attempts rather than selecting the best of many candidates;
- if proposals are biased toward conflict vertices, either construct a symmetric rule or include the Hastings ratio.

Repeated independent attempts are preferable to “sample many and take the best” because each attempt remains a legitimate reversible MCMC transition.

---

## 8. Immediate experimental roadmap

### Priority 1 — full-space multi-k sampler

Restore

[
H_{m count}
]

and retain the full ({0,1}^{261}) state space.

Compare:
1. standard single-bit SQA;
2. single-bit + (1leftrightarrow1) exchange;
3. single-bit + exchange + reconnect;
4. late repeated (k=2,3) exchange.

This is the main route for a clean future SQA/QPU comparison.

### Priority 2 — optimized constrained sampler

Keep the hard 143-edge manifold as an algorithmic control.

Use:
- (k=1) exchange early;
- several independent (k=2,3) attempts late;
- reconnect after the degree profile is repaired.

Measure native proposal counts in addition to wall time so that extra proposal trials are not hidden.

### Priority 3 — reverse annealing later

Once forward/full-space kernels are understood, revisit reverse annealing using good Hamilton paths as seeds.

Do not mix this into the current kernel benchmark yet; it is a separate question.

---

## 9. Experimental reporting rules

For all future sampler comparisons, keep separate:

- Hamiltonian / objective;
- accessible state space;
- proposal kernel;
- annealing schedule;
- outer classical projection;
- native proposal count;
- wall time.

Report at least:

[
P_{m feasible},
qquad
Delta W=W_0-W,
qquad
P_{m ground}mid	ext{feasible},
]

plus:
- degree residual;
- cycle count;
- acceptance rate by move type;
- number of attempted/eligible moves;
- iteration-to-feasible;
- best score.

The goal is not to prove that every run reaches the ground state.  For the assembly application, near-ground solutions can be useful if the reconstructed sequence remains correct.  Ground-state distance is currently a controlled proxy because a reference/certificate is available for the benchmark.

---

## 10. Related benchmark files on this branch

- `benchmark_chm13_projected_edge_sqa.py`
- `benchmark_chm13_projected_edge_hybrid.py`
- `benchmark_chm13_fixed_cardinality_kernels.py`
- `benchmark_chm13_fixed_cardinality_phase.py`
- `benchmark_chm13_multik_exchange.py`

Detailed results are under:

`debug/qubo/chm13_edge_ordered_path/`

This note should be updated when:
- the first full-space multi-k benchmark is complete;
- a formal effective-exchange derivation is added;
- reverse SQA or QPU reverse annealing is actually tested.
