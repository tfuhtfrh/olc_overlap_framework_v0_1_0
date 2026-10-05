# 第4章改稿メモ — 2026-10-05

対象: 段階報告の第4章。目的は **循環 OLC グラフに対する QUBO アルゴリズムの構成** を成果として説明すること。
研究過程を時系列で列挙せず、最終的な設計判断だけを残す。

## 章の論理

1. 第3章の weighted Hamiltonian path / edge-variable QUBO は DAG では成立するが、循環グラフでは `one path + cycles` を排除できない。
2. 評価データとして CHM13 complex144 を先に定義する。144 reads、261 candidate edges、最大 SCC 112、exact Hamilton paths 192。
3. Nüßlein et al. の order-variable Hamiltonian-cycle formulation を出発点として、weighted Hamiltonian path 用の順序表現を構成する。
4. 代表 formulation は latent strict-rank comparator: selected edge `u->v` に `P_v > P_u` のみを要求する。`+1` の連続順位は不要。
5. complex144 では 3789 variables、33376 quadratic terms、max |J|=576。既知経路の audit は成立するが、solver は edge topology と rank/comparator bits を同時に整合させにくい。
6. global acyclicity をすべて QUBO 変数に保持せず、edge-only 261-variable QUBO に戻す。selected graph の cycle/SCC 判定は古典計算で安価に実行できる。
7. 現在の cycle に対して均等な弱い penaltyを加え、QUBO optimization と classical cycle detection を反復する。
8. 正式評価では 3789-variable strict-rank comparator と 261-variable iterative edge-QUBO を同程度の SQA 計算量で比較する。

## 掲載しない内容

- R2 / R3 / endpoint transfer / whole-worldline proposal kernel
- 192 HP の詳細な state-neighborhood 解析
- reverse annealing
- reachability partial order
- alternative-rank / 4+2+2 の read allocation
- multiple explorers / population search
- bounded `+1` formulation

これらは研究上の診断としては有用だが、第4章の成果説明には含めない。

## complex144 の記述上の注意

データ package の README に合わせる。

- official CHM13 v1.1 HiFi primary-alignment data を出発点とする。
- benchmark の整理、骨格生成、orientation normalization には reference alignment を利用している。
- final graph は 144 physical reads / 261 directed edges。
- edge weight は read-read overlap のみから計算し、reference coordinate reward は含めない。
- main weight: `w = M - 49 d`, `d=B-M`。
- exact enumeration: 192 Hamilton paths。

「reference は graph construction に一切使っていない」とは書かない。

## strict-rank comparator の主要式

Vertex rank:

`P_v = sum_k 2^k p[v,k]`, `K=ceil(log2 N)`.

Selected edge condition:

`x_uv=1 => P_v>P_u`.

Borrow comparator:

`b[e,K]=1 <=> P_v<=P_u`.

Gate:

`H_gate = A_gate sum_e x_e b[e,K]`.

Variable count for complex144:

`Q=M+2N+NK+MK=3789`.

## iterative edge-QUBO

Base:

`H0 = H_weight + H_degree + H_count`.

For detected cycle `C`:

`H_cyc(C) = A_cyc/|C| * sum_{e in C} x_e`.

Update:

`H^(t+1) = H0 + sum_{C in C^(t)} H_cyc(C)`.

- past cycle penalties are not accumulated;
- current selected graph is re-analysed each iteration;
- no auxiliary variables are added;
- logical variables remain 261.

## 正式 benchmark — 固定条件

Compare only:

1. strict-rank comparator: 3789 variables, fixed Hamiltonian;
2. iterative edge-QUBO: 261 variables + classical current-cycle feedback.

No fixed-H0 baseline in the report.

SQA settings for both:

- OpenJij SQASampler, standard SingleSpinFlip
- beta = 5
- gamma = 1
- Trotter P = 8
- num_sweeps = 1400
- num_reads = 8 per iteration
- 16 iterations per run
- first state: all zero
- 24 independent runs per method

Between iterations:

- choose the minimum-QUBO-energy sample among the 8 returned samples as the next state;
- strict-rank: Hamiltonian remains fixed;
- iterative edge-QUBO: detect current cycles and rebuild the uniform cycle penalty before the next solve.

Uniform penalty strength fixed before formal evaluation:

`A_cyc = 4`.

## 統計

For every returned sample, decode selected edges and audit graph validity separately from auxiliary-variable consistency.

Per run record:

- whether any valid Hamilton path was observed;
- best Hamilton-path score observed;
- whether certified optimum was observed;
- strict-rank constraint residuals;
- iterative edge-QUBO cycle counts by iteration;
- runtime (secondary).

Aggregate over 24 runs:

- HP-hit runs / 24;
- median best score;
- IQR of best score;
- overall best score;
- optimum-hit count;
- runtime summary if useful.

The chapter should not make optimum hit a prerequisite for method validity; the report-wide evaluation criterion will be handled earlier in the document later.

## Writing policy

- 成果を説明し、試行錯誤の時系列を書かない。
- 1 paragraph = 1 technical point。不要な「一方で」「さらに」の連鎖を避ける。
- solver-specific implementation details are minimized; SQA/QA/Tabu are optimizers for the QUBO, not the conceptual center of the method.
- internal class/script names are not used in the report body.
- “reads” is reserved for DNA reads. SQA output uses “samples” or 「独立試行」.
- numerical claims must correspond to recorded repository data or the forthcoming formal benchmark.
