# Cyclic Hamiltonian experiments

Generated: 2026-09-04 (Asia/Tokyo)

## What this tested

This experiment area records candidate Hamiltonians for directed cyclic OLC
graphs and tests whether the existing DAG-only edge-cycle Hamiltonian can be
reused after deliberately cutting a known graph Hamilton cycle.

## Why it was tested

The accepted phi174 graph is cyclic, while the current
`EdgeCycleCoverDAGQUBOHamiltonian` requires the candidate read-read graph to be
a DAG.  Before implementing a new cyclic formulation, a non-strict test can
separate Hamiltonian/model behavior from cyclic graph preprocessing.

## Recorded native cyclic Hamiltonian (not implemented)

Nüßlein et al. (arXiv:2204.13539v1) describe an edge-position encoding for a
Hamiltonian cycle.  Every directed edge receives a binary-encoded position
`P_e`; `P_e = 0` means that the edge is not selected, while nonzero positions
place selected edges around the cycle.  Penalties enforce consecutive
positions for incident selected edges, reject incompatible edges sharing a
tail or head, and anchor one outgoing edge at position 1 and one incoming edge
at position N.  The position constraints prevent disconnected subtours.

For the accepted 812-node, 8,385-edge phi174 graph, the direct binary position
encoding would require `ceil(log2(812 + 1)) = 10` bits per edge, or at most
83,850 binary variables before any auxiliary-variable changes.  A weighted OLC
objective also needs a careful adaptation because edge selection is represented
implicitly by `P_e > 0`.  This formulation is recorded only as a candidate
native-cyclic baseline; no project module has been added or modified for it.

## Inputs and parameters

- accepted graph pipeline: minimap2 PAF direct, identity >= 0.990, overlap >=
  80 bp, full-coverage reduction, and the approved one-pass 34-node deletion;
- graph certificate: `debug/phi174_cycle742_graph/cycle742_hamilton_cycle.tsv`;
- cut selection: fixed pseudorandom seed 20260904;
- DAG projection: rotate the certified cycle after the cut and retain every
  candidate edge `u -> v` for which `rank(u) < rank(v)`;
- existing Hamiltonian: `EdgeCycleCoverDAGQUBOHamiltonian`;
- degree penalty 100, normalized `overlap_len_power2` reward scale 20.

## Output files

- `cycle742_cut_dag_edge_cycle_audit.txt`: graph, DAG, QUBO, witness, and
  optional low-budget OpenJij result;
- `../../../demo_phi174_cycle742_cut_dag_qubo.py`: reproducible demo.

## Result statistics

With seed 20260904, the selected cut is certificate position 122:
`SRR27862880.107001_m1(+) -> SRR27862880.9001_m2(-)`.  Both orientations
match the accepted global one-orientation projection.  Rotating after this
edge makes `SRR27862880.9001_m2` the path start and
`SRR27862880.107001_m1` the path end.

- original graph: 812 nodes and 8,385 unique directed edges;
- forward-rank DAG: 812 nodes and 6,186 unique directed edges;
- removed backward/cut edges: 2,199;
- retained certificate path edges: 811/811;
- QUBO: 7,810 variables and 721,978 quadratic terms;
- normalized edge reward range: 0.284444444444 to 1;
- QUBO build time in the recorded WSL runs: about 0.24--0.28 seconds;
- encoded certificate: 813 selected variables, energy -13641.9013333, one
  void source, one void sink, and zero read/void constraint violations.

A single-sample OpenJij SQA trial with 100 sweeps and trotter 4 took about
7.6--8.1 seconds.  It did not find a valid edge cycle: energy 19874.496, 943 real
edges selected, 162 read-in and 164 read-out constraint violations, while both
void constraints were satisfied.

## Interpretation

The known cycle is used only to define the cut and the forward topological
rank.  Consequently, this is an intentionally non-strict feasibility and
resource-scale test.  It cannot show that a production pipeline can infer a
good cut without already knowing a Hamilton cycle.

The formulation check passed: the existing Hamiltonian represents the full
cut path with no constraint violation.  The small SQA trial failed to recover
a feasible sample at this scale, but that does not contradict model
feasibility because the explicit witness is valid and has much lower energy.

Removing only the selected cycle edge would not make the full candidate graph
acyclic.  The forward-rank projection is necessary for the existing builder's
DAG precondition and preserves all 811 consecutive edges of the cut
certificate path by construction.

## Whether it is part of the main conclusion

No.  It is a diagnostic upper-bound test for the existing implementation.

## Limitations

- The test leaks a known feasible graph cycle into preprocessing.
- It does not evaluate reference order unless a later evaluation stage is
  explicitly added.
- A low-budget annealing failure would describe optimizer scaling, not QUBO
  infeasibility; the encoded known witness is the formulation check.

## User supplements

- Temporarily record the native cyclic Hamiltonian without implementing it.
- For this non-strict test, cut the single cycle at a conflict-free location
  and try the existing edge-cycle DAG Hamiltonian.

## Fixed-endpoint optimizer follow-up (2026-09-04)

The first full-model OpenJij SQA trial left 162 read-in and 164 read-out
constraint violations.  A demo-only exact variable substitution was therefore
tested under the same known-cut assumption.  All 812 void-to-read source
variables and 812 read-to-void sink variables are fixed: only
`void -> path_start` and `path_end -> void` are 1.  This does not alter the
Hamiltonian on the remaining variables.

The reduction changes the QUBO from 7,810 variables and 721,978 quadratic
terms to 6,186 variables and 51,074 quadratic terms.  The encoded certificate
has energy -13641.9013333 before and after substitution, confirming exact
energy preservation.

Same-scale optimizer observations:

- fixed-endpoint OpenJij SQA, one read and 100 sweeps: invalid, with 153
  read-in and 161 read-out violations;
- fixed-endpoint D-Wave classical SA, one read and 100 sweeps: invalid, with
  30 read-in and 30 read-out violations;
- the same classical SA with edge reward disabled: invalid, with 38 read-in
  and 34 read-out violations, so the overlap reward was useful guidance rather
  than the principal cause of failure;
- fixed-endpoint D-Wave tabu, one read and a 2,000 ms tabu timeout: valid, with
  811 real edges and zero read/void constraint violations.  Its energy is
  -13641.9013333 and its order exactly matches the rotated graph certificate.
  The measured solve stage, including BQM conversion, was about 5.45 seconds.

The successful order is written to `cycle742_cut_dag_solved_order.tsv`.  This
resolves feasibility within the previous wall-clock scale, but it remains a
non-strict upper-bound experiment: the known Hamilton cycle determines both
the DAG rank and its fixed endpoints.  Since the preserved consecutive edges
make that rank a Hamilton path, the DAG has a unique topological order; the
recovered order is therefore the existing arbitrary feasibility witness, not
a newly optimized biological order.

## Fixed-endpoint OpenJij SQA sweep (2026-09-04)

The reduced 6,186-variable, 51,074-quadratic-term QUBO was also tested directly
with OpenJij SQA.  The complete sweep is stored in
`cycle742_fixed_endpoint_sqa_sweep.tsv`.  The installed sampler defaults to
`beta=5.0`, `trotter=4`, one read, and 1,000 sweeps when values are not
overridden.

Increasing from 100 to 1,000 sweeps improved the default-beta result from
153/161 to 133/129 read-in/read-out violations.  Ten reads returned the same
best sample.  Explicit beta tuning was substantially more useful: with 1,000
sweeps and trotter 4, beta values 0.05, 0.5, 1.0, 2.0, and 3.0 produced total
in/out violation pairs of 571/553, 43/43, 25/25, 13/13, and 40/40,
respectively.  Raising trotter from 4 to 8 at beta 2.0 gave 14/14 and did not
improve the result.

Separating the annealer seed from the fixed cut seed and trying SQA seeds 1,
2, and 3 at beta 2.0 gave 12/12 violations in each case, the best pure-SQA
result in this sweep.  Increasing to 10,000 sweeps was unstable and produced a
much worse terminal sample rather than a monotonic improvement.

Pure SQA therefore approached but did not reach feasibility in the tested
range.  The canonical audit remains the zero-violation tabu result; the SQA
sweep is retained as a separate optimizer comparison and does not supersede
it.

## SQA Trotter-m and penalty/reward follow-up (2026-09-04)

The `trotter` parameter is the Suzuki--Trotter slice count, denoted here by
`m`.  OpenJij rejects `m=1`, so `m=2` is the legal lower bound.  With beta 2.0,
1,000 sweeps, annealer seed 3, degree penalty 100, and reward scale 20, the
read-in/read-out violations for `m=2,4,8,16,32` were respectively `8/8`,
`12/12`, `16/16`, `21/21`, and `35/35`.  Increasing simulation precision did
not improve optimization quality on this instance; it made the returned
terminal sample progressively worse and slower.

At `m=2`, beta values 1.5, 1.8, 2.0, 2.2, and 2.5 confirmed beta 2.0 as the
best tested region.  Annealer seeds 1, 2, 3, 4, and 5 produced 10/10, 10/10,
8/8, 9/9, and 9/9 violations.  Reducing the overlap reward scale from 20 to 10
at the best `m/beta/seed` reduced the result to 6/6 violations and 817 selected
edges, the best pure-SQA sample so far.  Reward scales 5 and 12.5, and degree
penalty 200, were worse.  Ten reads did not improve over the one-read result.

Thus the updated best pure-SQA parameters are `m=2`, `beta=2.0`, 1,000
sweeps, degree penalty 100, reward scale 10, and annealer seed 3.  The result is
near-feasible but still not a Hamilton path.  Higher `m` should not be treated
as automatically better optimization: it improves the imaginary-time
discretization of the simulation, not the ability of a fixed schedule to
escape this QUBO's local minima.

## SQA gamma, schedule, and local-minimum diagnosis (2026-09-04)

The OpenJij adapter now exposes optional `gamma` and custom two- or
three-column schedules.  Existing calls are backward compatible.  The demo
also exposes quartic, linear, pause, beta-ramp, and classical-tail schedules,
plus optional selection of the best recorded Trotter slice.  Detailed results
are stored in `cycle742_fixed_endpoint_sqa_schedule_sweep.tsv`.

At the best `m=2`, beta 2.0, reward 10 configuration, gamma values 0.5, 0.8,
1.0, 1.2, 2.0, 10, and 100 gave in/out violation pairs of 16/16, 11/11, 6/6,
7/7, 8/8, 8/8, and 9/9.  Linear, pause, varying-beta, and classical-tail
schedules did not improve on 6/6.  A low-beta classical tail damaged the
partial solution and was the only tested terminal sample that was not already
a one-flip local minimum.  Raising Trotter `m` to 4 produced four identical-
energy candidate slices and a worse 12/12 result; selecting the best slice did
not recover hidden solution quality.

The best 6/6 sample selects 817 real edges.  Because the fixed source and sink
make the required count 811, all six in violations and all six out violations
are degree-two surplus conflicts; there are no degree-zero deficits.  Exact
one-bit delta evaluation found zero improving or neutral flips, with minimum
delta +5.67511111111.  D-Wave's deterministic steepest descent therefore made
zero flips and no energy change.

`cycle742_fixed_endpoint_sqa_local_minimum.txt` records the conflict structure.
There are six out-conflict and six in-conflict reads.  The selected sample has
12 edges touching only an out-conflict endpoint, 12 touching only an
in-conflict endpoint, and zero selected edges directly joining an out-conflict
tail to an in-conflict head.  Removing any one of those 24 boundary edges
merely moves a degree violation to its regular endpoint while losing the edge
reward, so every single deletion raises energy.  Repair requires a coordinated
alternating multi-edge exchange.  OpenJij's installed SQA implementation only
offers the single-spin-flip updater; parameter tuning changes which local basin
is reached but does not remove this neighborhood barrier.  The successful tabu
run is consistent with this diagnosis because tabu search can cross temporary
energy increases.

## Linear edge bias and initial-state diagnosis (2026-09-04)

The edge-cycle configuration now has an optional non-negative
`edge_selection_penalty`, defaulting to zero. It adds only
`lambda * sum(real_edge_variables)` and therefore adds no variables or
quadratic couplings. Every feasible fixed-endpoint Hamilton path uses exactly
811 real edges, so the term shifts every feasible path by the same constant
and does not alter feasible-path ranking.

The sweep in `cycle742_fixed_endpoint_sqa_edge_bias_sweep.tsv` shows that this
term controls cardinality but does not remove the matching-neighborhood trap.
At lambda 8 the raw sample had exactly 811 selected edges but still had 2
read-in and 4 read-out violations; steepest descent ended at 810 edges and
3/3 violations. Lambda 12 changed the previous overfull 817-edge local
minimum into an underfull 804-edge local minimum with 7/7 violations. Ten
reads and 2,000 sweeps at lambda 8 still ended at 3/3 after postprocessing.

Uniformly scaling all coefficients also did not solve the instance. Scaling
100:10:8 to 1:0.1:0.08 caused thermal disorder (54/52 violations before the
final one-flip minimum); intermediate scales 0.05, 0.1, 0.2, and 0.5 ended at
roughly 20/20, 17/17, 11/13, and 8/4 postprocessed violations. The original
scale remained best among these tests.

The OpenJij adapter now exposes an optional binary `initial_state`. A standard
forward schedule starting from the encoded certificate destroyed that state
and ended at 10/10 violations, because the early transverse-field stage erases
the classical initialization. A diagnostic late-linear schedule starting at
`s=0.8` retained the certificate exactly: 811 edges, zero degree violations,
energy -6820.95066667, and minimum one-flip delta +190. This confirms that the
valid basin exists and is deep; random forward SQA fails to enter it with its
single-spin neighborhood. This certificate-initialized run is diagnostic and
must not be reported as an end-to-end solution.

## External Plan-B cyclic compression diagnosis (2026-09-04)

The full 812-node graph is one closed strongly connected component. The
external Plan-B compressor requires a proper region with at least one external
input and output port and, by default, requires cycle closure so macro-edge
replacement cannot recreate a cycle through the outside graph. The whole SCC
has no external ports, while small proper regions close back through the rest
of the SCC and fail the cycle-closed condition.

Disabling cycle closure alone was insufficient. In a reduced search over four
high-degree seeds, no candidate appeared with symmetric port limits 4, 8, or
16. At port limit 32, candidates appeared, but even a two-node region had
about 28--30 incoming and 18--20 outgoing boundary edges. Such a region can
create roughly 500--600 input-output macro alternatives, so replacing two
nodes may expand rather than compress the problem. Therefore the warning is
a valid structural safeguard, not a threshold that should simply be relaxed.
Native support for this input needs a separate top-level closed-SCC/cycle
branch (for example, an explicit virtual cut or a native cycle formulation).
