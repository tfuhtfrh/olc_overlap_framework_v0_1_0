# QbSolv（tabu）求解阶段报告中的顺序变量 QUBO

- 生成日期：2026-11-25
- 运行：GitHub Actions run 45533943184（`report-qbsolv-rank.yml`，提交 30a127f），结果在 `run_45533943184/summary.json`。
- 第一次运行 45533512140 的结果相同，但提交结果那一步失败了：`pip install -e` 改动了仓库中被跟踪的缓存文件，`git pull --rebase` 被拒绝。修正 workflow 后重新运行。

## 测试内容和原因

为 M1 研究相談会的发表，把提案手法（二進減算による順序変数 QUBO）的 SQA 结果和经典求解器作比较。

QUBO 和阶段报告中 SQA 用的完全相同：
- 构建：`experimental_chm13_latent_rank_comparator_qubo.build_model()` 的默认配置
  - degree 144、void 144、comparator 288、order gate 144、endpoint distinct 144
- 规模：3,789 个变量，33,520 个二次项

## 参数

- dwave-qbsolv 0.3.4（内部为经典 tabu，energy-impact 分解），Python 3.10，dimod 0.10.17
- num_repeats = 50，子问题大小 = 47（均为默认值），每个 seed 上限 600 秒
- seed 与阶段报告相同的 6 个（2026101801–06）
- 不修复解；参考路径在全部采样结束后才读取，只用于离线评价

## 结果

| | SQA（阶段报告表 6） | QbSolv |
|---|---|---|
| 得到合法路径的试行 | 1 / 6 | 0 / 6 |
| 最大权重 | 1,238,236 | — |
| 每次试行的计算时间 | — | 16–23 秒 |

各 seed 中 QbSolv 返回的最低能量解：
- 选中 136–138 条边（合法路径需要 143 条），断成 6–8 段路径；
- 度数残差平方和 8–14；
- 借位（comparator）违反 0 处，最终借位不一致 0 处；
- 违反顺序条件的选中边：56–61 条；
- 能量 9,572–10,867，最优路径的能量为 63.3。

## 解读

- 经典的 tabu（QbSolv）同样没有得到合法路径，最低能量远高于最优路径。所以不是罚项有漏洞、使非法解能量更低，而是搜索没有到达低能量区域。
- 借位电路始终满足；失败在于"选边"和"全局一致地排列顺位"难以同时满足。这与 2026-09-26 的快速测试（`chm13_edge_ordered_path/latent_rank_comparator_summary_20260926.md`）的倾向一致。
- 定位：发表用的参考结果，不属于正式评估。

## 局限

- QbSolv 只用了默认参数这一种设置，没有调子问题大小、num_repeats、时间预算或罚项系数。
- 和 SQA 的计算量条件没有对齐（SQA 每次试行 20,800 sweeps）。
- 只测了 complex144 一个数据集。

## 用户补充

（待补）
