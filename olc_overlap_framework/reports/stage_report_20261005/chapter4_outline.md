# 第4章改稿メモ — 2026-10-06

## 位置づけ

第4章の目的は、循環を含む OLC グラフに対する QUBO アルゴリズムを示すことである。
研究過程ではなく、採用する定式化と数値評価を順に説明する。

## 現在の構成

1. DAG 用の辺変数型 QUBO が循環グラフでは「一本の経路 + サイクル」を許すことを示す。
2. CHM13 complex144 の抽出条件とグラフの特徴を説明する。
3. Nüßlein et al. の順序変数型定式化を示し、連続順位 P_{bc}=P_{ab}+1 とその二乗制約を紹介する。
4. 本研究では各 read に順位 P_v を持たせ、selected edge に P_u<P_v のみを要求する。
5. P_v-P_u-1 の二進減算を桁借り変数で表し、比較器、始点・終点、次数、重みを含む完全な QUBO を式で与える。
6. complex144 上で順位比較型の変数数、係数幅、既知解 audit、QBSolv による予備確認を示す。
7. 順位・桁借り変数を除き、261 個の辺変数だけを残す。サイクルは古典計算で検出する。
8. 検出したサイクルの各辺に同一の線形ペナルティ A_cyc を与え、QUBO を反復更新する。
9. 正式評価では 3789 変数の順位比較型と 261 変数の反復更新型へ同一の SQA 予算を与える。

## 重要な式

### 順位比較型

`P_v = sum_k 2^k p[v,k]`

`x_uv=1 => P_u<P_v`

各候補辺 e=(u,v) について P_v-P_u-1 を減算する。
b[e,0]=1 とし、

`b[e,k+1]=1 <=> p[u,k]+b[e,k] > p[v,k]`

を二次式 R で表す。

`H_cmp = A_cmp sum_e sum_k R(...)`

`H_ord = A_ord sum_e x_e b[e,K]`

次数・始点終点:

`H_deg = A_deg sum_v [(1-s_v-sum_in x)^2 + (1-t_v-sum_out x)^2]`

`H_end = A_end[(1-sum s)^2+(1-sum t)^2] + A_diff sum_v s_v t_v`

重みは complex144 の `w_e=M_e-49d_e` を規格化した edge cost を用いる。

完全形:

`H_rank = H_weight + H_deg + H_end + H_cmp + H_ord`

### 反復更新型

辺変数だけの基本 Hamiltonian:

`H0 = H_weight + H_degree + H_count`

各反復で非自明 SCC からサイクルを検出し、

`H_cyc^(t) = A_cyc sum_{C in C^(t)} sum_{e in C} x_e`

とする。以前の `A/|C|` 正規化は使用しない。
各 cycle edge の追加コストを cycle length に依存させないためである。

`H^(t+1)=H0+H_cyc^(t)`

過去の cycle penalty は蓄積しない。

## 正式評価

両定式化に同じ SQA 呼び出し回数・samples・sweeps を与える。

- OpenJij SQASampler
- beta = 5
- gamma = 1
- Trotter P = 8
- num_reads = 8
- 順位比較型: 20800 sweeps を 1 回 / independent trial
- 反復更新型: 1300 sweeps × 16 回 / independent trial
- 24 independent trials / formulation

両定式化で 1 trial 当たりの sweep 数を 20800 に揃える。

- 順位比較型: Hamiltonian を固定して 1 回の長い SQA を実行
- 反復更新型: 各 call の最低 QUBO energy sample を次回の initial state とし、call 後に cycle を検出して線形 penalty を更新

A_cyc は per-edge penalty への変更後、正式計算前に固定し、全試行で共通とする。

記録:
- Hamilton path を得た試行数
- 各試行の最大 path weight
- median / IQR / overall best
- known optimum hit count
- constraint violations（順位比較型）
- cycle count by iteration（反復更新型）
- runtime

## 本文に入れない内容

R2/R3、sampling-kernel modification、reverse annealing、reachability、alternative rank、4+2+2、multiple explorers。

## 文体

- 会話中の shorthand や実装名を本文へ持ち込まない。
- 数式は入力変数から完全な Hamiltonian まで追える順序で出す。
- 未定義の H 項を最後に突然置かない。
- 一段落一論点を基本とする。
- 研究の試行錯誤ではなく、採用した定式化と確認結果を書く。
