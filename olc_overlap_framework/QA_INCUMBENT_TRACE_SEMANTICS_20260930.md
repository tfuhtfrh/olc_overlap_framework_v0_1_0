# QA incumbent and hybrid trace semantics — 2026-09-30

## 1. Historical-best incumbent is not intrinsic SQA state

The current solver stores

    x_inc = argmax W(x)

over all observed valid Hamilton paths.

This is not a memory mechanism inside SQA/PIMC. It is classical outer-controller state.

For optimization this is non-oracular:
- Hamilton-path feasibility is known from the problem definition;
- W(x) is the known overlap objective;
- the unknown optimum score is never used.

The same distinction applies to QA:
- one physical anneal produces one measured classical sample;
- repeated anneal/read cycles produce a collection of samples;
- a host controller can keep the best valid sample seen so far.

Thus best-observed-solution retention is compatible with QA-based optimization, but the memory lives outside the QPU.

If the Hamiltonian is adaptively changed between batches of anneals, the overall method should be called a quantum-classical hybrid, not one-shot QA.

For sampling-distribution studies, incumbent filtering should not replace raw sample statistics. For optimization, it is an appropriate output rule.

## 2. Trace levels

### A. Intra-read SQA trajectory

Inside one OpenJij SQA read:
- Suzuki-Trotter state evolves through SingleSpinFlip sweeps;
- this internal sweep-by-sweep path is NOT what the current outer trace shows.

### B. Standard SQA call

One SQASampler call can contain multiple reads.

For the standard baseline:
- typically 8 reads;
- typically 1400 sweeps/read;
- each read returns OpenJij's minimum-classical-energy Trotter slice.

These are the standard SQA outputs of one complete call.

### C. Outer hybrid trace

The displayed iter 0 -> iter 1 -> iter 2 -> ... is an OUTER trajectory.

A typical current-rank/reachability iteration is:

    x^(t)
      -> construct H^(t+1) from x^(t)
      -> run a complete standard-SQA call
      -> collect its returned read samples
      -> choose explorer x^(t+1)
      -> update best-feasible incumbent

Only then is the next outer-trace row produced.

Therefore an entry such as

    iter 9: score = 1,343,093

means a complete newly launched SQA episode produced the certified ground as one of its standard outputs. It does not mean the ground was observed halfway through an anneal and used to stop that anneal.

## 3. Dual-track trace

For the current dual-track reachability hybrid, one outer iteration can contain two separate standard-SQA calls, for example:

    4 rank-free reads
    +
    4 reachability-guided reads

Both branches start from the same current explorer state but use different static Hamiltonians.

Their standard outputs are pooled.

Then:
- the explorer for the next outer iteration is selected using rank-free H0 energy;
- any valid HP among all outputs can update the incumbent.

This is a quantum-classical / SQA-classical hybrid architecture.

## 4. Evaluation metrics going forward

Primary optimization metrics:
- whether any valid HP was observed during the fixed budget;
- best feasible incumbent score;
- whether incumbent equals the certified ground (offline evaluation only);
- first iteration/read at which the incumbent was improved.

Secondary dynamics diagnostics:
- final explorer HP/ground status;
- cycle count and degree conflicts of the final explorer;
- how often an HP is subsequently lost;
- rank-guided vs rank-free branch contribution.

This avoids requiring the final Markov/SQA state to remain at the best solution.

## 5. Current implication

The fact that a fresh forward-SQA episode often leaves a ground-state warm start is no longer a solver failure by itself.

It means:
- the sampler does not dynamically retain that state;
- but the optimization controller can still return the best valid solution it observed.

The remaining scientific question is therefore shifted toward discovery rate: how often and under what Hamiltonian guidance does standard SQA/QA generate the optimum at least once within a fixed resource budget?
