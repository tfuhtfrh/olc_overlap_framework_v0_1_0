# Plan B 复原指标在 CHM13 complex144 上的试用

生成日期：2026-11-20

## 测试内容和原因

共同研究者提供了 Plan B 的 DNA 复原评价材料（`PLANB_METRICS_FOR_COLLABORATOR_20261120`，未放入本仓库）。它的流程是：在原图上独立审计路径 → 按原 overlap 证据从原始 read 拼出 FASTA → 冻结 SHA256 → 用 QUAST 5.3.0 对参考评价。主指标为 Genome fraction (GF)，辅助指标为 misassemblies、NGA50、Duplication ratio、mismatch/indel 率等。

这里检查两件事：一是这套指标能否直接用于本仓库的 layout 结果；二是它能否区分路径分数不同的解。

## 输入与参数

- 数据：`test_data/CHM13/CHM13_complex144`。
  - 图用 `graph_with_rc.graphml`：Plan B 的审计要求每条 read 的 +/- 两个方向都在图里。
  - 用 `paf_line`（`overlaps.exact.paf.gz` 的 1-based 行号）给每条边补上 PAF 坐标（q/qstart/qend/qlen/t/…）。
  - reads 用 `reads.fasta`，参考用 `reference.fasta`。
  - 转换脚本是 `prepare_complex144.py`，它不读取参考。
- 候选路径：
  - `weighted_certificate.json` 的前 3 名（1,343,093 / 1,338,209 / 1,336,620）。
  - Actions 39137698403 中 6 个 seed 的不同最好解（1,320,804 / 1,273,384 / 1,272,255）。路径取自复核运行 39145356481 的日志；两个 1,273,384 是同一条路径。
- 评价：用包内 `run_metrics.py`。
  - 每条路径各自做 freeze 和 evaluate，冻结清单未经人工改动。
  - QUAST 5.3.0（GitHub tag `quast_5.3.0`）；参数为 `--min-identity 95 --extensive-mis-size 1000 --local-mis-size 200 --ambiguity-usage one --eukaryote --threads 2`。
  - 哈希见 `frozen_inputs.json`。

## 结果

6 条路径的独立审计全部 PASS（144 条物理 read 各用一次），评价状态都是 PASS。完整表格见 `results.csv`，QUAST 原始报告见 `quast_reports/`。

| 路径 | 分数 | GF % | 总长 | misassemblies | NGA50 | mismatch/100kbp |
|---|---:|---:|---:|---:|---:|---:|
| 认证第 1（最优） | 1,343,093 | 99.577 | 1,640,191 | 1 | 1,317,932 | 9.69 |
| 认证第 2 | 1,338,209 | 99.577 | 1,640,189 | 1 | 1,317,930 | 9.75 |
| 认证第 3 | 1,336,620 | 99.577 | 1,640,189 | 1 | 1,317,930 | 9.88 |
| SQA 最好 | 1,320,804 | 99.577 | 1,640,163 | 1 | 1,317,873 | 12.38 |
| SQA | 1,273,384 | 99.577 | 1,694,589 | 3 | 939,604 | 26.97 |
| SQA | 1,272,255 | 99.572 | 1,694,630 | 7 | 897,805 | 50.88 |

参考长度为 1,647,338 bp。所有路径的 indel 率都在 182–211 /100kbp，这是直接拼接原始 HiFi read、未做 consensus 的结果。

## 解读

1. **GF 在这里没有区分力。** 每条合法路径都用上了全部 144 条 read，所以覆盖范围几乎相同。分数相差 70,000 的路径，GF 也只差 0.005 个百分点。
2. **区分力来自 misassemblies、NGA50 和总长。**
   - 前三名和 SQA 最好解（1,320,804）在这些指标上与最优解几乎没有差别，和 10/18 的序列重建结论（相差 188 edit）一致。
   - 1,27x,xxx 档的两条 SQA 解多出约 54 kb，有 3 个和 7 个 relocation，NGA50 从 1.32 Mb 降到 0.90–0.94 Mb。两条都有两处约 27 kb 的重复；1,272,255 另外还有约 1.1 Mb 大块的前后调换。
3. **连认证最优路径也有 1 个 relocation，原因不在 read 顺序，而在边证据。**
   - 位置：路径第 113→114 条 read 之间，即 `53346526/ccs+` → `93652138/ccs+`，参考坐标约 1,300 kb 处。
   - 按 BAM 坐标，两条 read 真实重叠约 7,504 bp，对应 PAF 第 14157 行：block 7,511，matches 7,482，weight 6,061。
   - 图中这条边选的却是第 14153 行：block 14,468，matches 14,323，weight 7,218。这是错开约 7 kb 重复单元的 overlap。
   - 规则"同一节点对取 weight 最大的证据"在这里选中了重复造成的错位证据。拼出的序列因此塌缩了一个约 7 kb 的重复单元：GF 少 0.42%，并报 1 个 relocation。所有候选路径都经过这条边，所以 6 条路径都带着这个误接。
   - 这意味着：complex144 的"参考路径即最优"在 read **顺序**层面成立；但在 overlap **证据**层面，至少有一条边的权重偏向错误证据。

## 是否属于主结论

暂不属于。这是对外部评价标准的试用，以及对 complex144 边证据的一项诊断。第 3 点可能影响"分数最优＝生物学最好"这一叙述，需要和用户讨论后再决定是否写进报告。

## 局限

- QUAST 只在本次 sandbox 中运行过一次，没有在 GitHub Actions 的冻结环境中复现。
- 包内 README 写明，PDF 所依据的 R74 原入口没有找回。本次用的是 R262 部署快照的评价代码，其中 `genome_fraction_evaluation.py` 的 SHA 与包内声明一致。
- GF 和 misassemblies 依赖参考、`--ambiguity-usage one` 等设置。CHM13 chr2 这一区间有长重复，检测结果不保证完整。
- 只检查了第 113→114 这一条边；其他边是否也选中了重复错位证据，还没有系统检查。

## 用户补充

（待补）
