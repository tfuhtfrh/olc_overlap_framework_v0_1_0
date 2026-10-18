# Rank-expression hybrid follow-up — 2026-09-30

## Motivation

The recent standard-SQA baselines suggest a consistent tension:

- rank-free H_weight + H_degree + H_count explores broadly and can reach good
  Hamilton paths, but it frequently returns to cyclic states;
- a deterministic projected total rank can reduce exploration and inject an
  arbitrary backward-edge direction;
- stronger rank pressure does not reliably retain a good HP under a fresh
  forward-SQA episode.

The goal is therefore not simply to increase A_rank. The rank representation
itself should preserve uncertainty and should be separated from incumbent
retention.

All experiments below use vanilla OpenJij SQASampler SingleSpinFlip as the
inner solver. No endpoint/R2/R3/cluster/worldline custom updater is used.

Ground score 1,343,093 is used only for offline certification.

---

## 1. Rank dropout and multi-hypothesis rank

Matched screening budget:
- 4 standard SQA reads / outer iteration;
- 12 outer iterations;
- A_rank = 0.02;
- 4 zero-start and 4 random-start runs.

Methods:

none:
    no rank penalty.

gated_rank:
    one deterministic projected total rank when the current state is
    N-1-edge, degree-correct, and cyclic.

rank_dropout:
    during a gated episode, 2 reads use no rank and 2 reads use the projected
    rank.

rank_mixture:
    during a gated episode, four alternative rank hypotheses are created by
    rotating the ordering inside nontrivial SCCs; one standard SQA read is run
    under each hypothesis.

All candidate outputs are compared with the same non-oracle topology/weight
merit for propagation, and the best valid HP is archived as an incumbent.

Results:

| method | start | HP-incumbent runs | best incumbent |
|---|---|---:|---:|
| none | zero | 1/4 | 1,293,265 |
| none | random | 4/4 | 1,305,467 |
| deterministic gated rank | zero | 1/4 | 1,265,067 |
| deterministic gated rank | random | 1/4 | 1,291,734 |
| rank dropout | zero | 3/4 | 1,313,703 |
| rank dropout | random | 4/4 | 1,320,179 |
| rank mixture | zero | 4/4 | 1,325,446 |
| rank mixture | random | 4/4 | 1,327,035 |

No configuration hit certified ground in this small matched screening.

Interpretation:
- deterministic rank is clearly harmful in this batch;
- retaining rank-free reads restores exploration;
- representing rank uncertainty as separate SQA modes is better than forcing
  one arbitrary ordering.

The mixture interpretation is conceptually different from averaging penalties:

    H_avg(x) = H0(x) + A E_r[B_r(x)]

penalizes all ambiguous directions fractionally.

A rank-mixture instead approximates sampling from

    p(x) proportional to sum_r exp[-beta(H0(x)+A B_r(x))],

which retains separate low-energy modes for different plausible orders.

---

## 2. SCC partial-order rank

A second test removed total ordering inside each SCC.

SCCs are ordered using the condensation DAG, but vertices inside the same SCC
are treated as incomparable and receive no within-SCC backward penalty.

Methods:
- none;
- weak exact current-cycle cut;
- full projected rank;
- SCC partial rank;
- SCC partial rank + exact cut.

Important results:

full rank:
- zero: 0/4 HP incumbent
- random: 1/4

partial rank:
- zero: 0/4
- random: 2/4

partial rank + cut:
- zero: 3/4 HP incumbent, best 1,320,179
- random: 1/4

Thus simply removing arbitrary within-SCC ordering is not enough to break a
current cycle. It is more naturally a preventive / orientation aid, while a
cycle-level mechanism handles the SCC itself.

---

## 3. Reachability-based partial order

A stronger partial-order formulation avoids choosing any topological order.

Let C(u) be the SCC of u in the current graph and D the condensation DAG.
Penalize a candidate edge u->v only if

    C(v) -> ... -> C(u)

already exists in D.

Then adding u->v would definitely close a directed cycle between SCCs.

Hamiltonian term:

    H_reach = A * sum_{(u,v): C(v) reaches C(u)} x_uv

with A=0.02 in the screening.

Properties:
- no arbitrary ordering between incomparable SCCs;
- no arbitrary edge choice inside a current SCC;
- on an HP, backward/chord edges that would close a cycle are still
  recognized, so guidance does not disappear simply because cycle_count=0.

Screening results:

| method | start | HP-incumbent runs | best incumbent | ground |
|---|---|---:|---:|---:|
| none | zero | 1/4 | 1,327,035 | 0 |
| none | random | 2/4 | 1,319,675 | 0 |
| current exact cut | zero | 1/4 | 1,274,948 | 0 |
| current exact cut | random | 3/4 | 1,319,675 | 0 |
| reachability | zero | 0/4 | — | 0 |
| reachability | random | 2/4 | **1,343,093** | **1/4** |
| reachability + cut | zero | 1/4 | 1,274,948 | 0 |
| reachability + cut | random | 4/4 | 1,319,675 | 0 |

The reachability-only random-start configuration produced a certified ground
hit using only vanilla SingleSpinFlip SQA internally.

The combined reachability+cut variant increased random-start HP incidence to
4/4 but did not reach ground. This is another example of the feasibility /
exploration tradeoff: more acyclicity pressure can make HPs easier to obtain
while reducing access to the best weighted HP.

Ground-hit trace for reachability/random:

    iter 0: HP 1,319,675
    iter 1-4: cyclic states
    iter 5: HP 1,282,308
    iter 6-8: cyclic states
    iter 9: certified ground 1,343,093
    iter 10-11: cyclic states again

The ground is therefore discovered but not dynamically retained.

---

## 4. Reachability-strength calibration

To test whether retention could be fixed by increasing the rank coefficient,
controlled one-episode diagnostics were run from:

- the exact runner-up HP, 1,338,209;
- the exact ground HP, 1,343,093.

A_reach was swept over:

    0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50

with 12 seeds per point and 8 standard reads per seed.

### Starting from runner-up

No A value produced the ground.

HP outputs were rare:
- A=0, 0.01, 0.02: 1/12 HP;
- A=0.05, 0.10, 0.50: 0/12 HP;
- A=0.20: 1/12 HP.

### Starting from ground

No A value retained the exact ground even once.

Best HP-retention incidence:
- A=0.05: 4/12 runs remained on some HP;
- A=0.02: 3/12;
- A=0 or 0.01: 2/12;
- stronger A was not monotonic and eventually became worse.

Thus there is no evidence that simply increasing a static reachability-rank
coefficient solves retention.

A forward SQA episode is itself a new global annealing search; a warm start plus
linear rank bias is not a reliable "stay near the incumbent" mechanism.

---

## 5. Updated role of rank

The evidence supports a narrower role:

> rank should provide sparse topology guidance, not act as a mechanism for
> storing the best solution.

Recommended separation:

### Explorer
Use predominantly rank-free

    H0 = H_weight + H_degree + H_count

to preserve broad weight/topology exploration.

### Guided probes
Allocate only part of the read budget to:
- reachability partial-order bias; or
- multiple rank hypotheses.

Do not average all uncertain directions into one strong Hamiltonian.

### Incumbent archive
Maintain

    x_inc = argmax W(x)

over all observed valid Hamilton paths.

This is non-oracular:
- Hamilton-path feasibility is part of the problem definition;
- W(x) is the known optimization objective;
- the unknown optimum score is never used.

The sampler state can leave an HP while the solver still retains the best
feasible solution found so far.

This is preferable to increasing A_rank merely to force state retention.

---

## 6. Current preferred rank formulations

### Primary candidate: reachability partial order

    H_reach = A sum_{(u,v): C(v) reaches C(u)} x_uv

Advantages:
- no arbitrary total order;
- no arbitrary within-SCC backward edge;
- persists on acyclic states;
- produced a standard-SQA certified ground hit in screening.

### Exploration-preserving alternative: rank mixture / dropout

Use a mixture of:
- rank-free reads;
- reads under different plausible ranks.

This performed substantially better than a deterministic gated rank in the
matched screening.

### Cycle cut

Keep as a separate semantic acyclicity mechanism when needed.

The experiments suggest not to stack every form of acyclicity bias
simultaneously: reachability+cut improved HP incidence but reduced best-weight
performance in the small screening.

---

## 7. Current hybrid design direction

A promising architecture is:

    rank-free explorer
        +
    sparse guided SQA probes
        +
    non-oracle best-feasible incumbent archive

rather than

    one persistent total-rank Hamiltonian.

This directly matches the observed tradeoff:
- no-rank gives exploration;
- weak rank/partial-order probes can repair topology;
- the incumbent archive provides retention without distorting the Hamiltonian
  merely to hold the current state.
