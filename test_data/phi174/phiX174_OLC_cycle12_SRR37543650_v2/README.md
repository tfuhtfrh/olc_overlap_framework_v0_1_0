# φX174 read-level OLC 单环验证集（SRR37543650，v2）

## 定位

本包面向“由 reads 构造 OLC 图并求 layout”，而不是仅验证组装器能否输出正确 unitig。

旧的56-read v1包包含大量 contained/unsupported reads，只能作为预处理来源，不能当作56节点 Hamiltonian benchmark。本v2包取代v1。

本数据适合验证：

- 高质量 dovetail overlap 检出；
- 反向互补方向处理；
- 传递边约简；
- 环检测及 Hamiltonian cycle 求解；
- 起点和方向简并的处理。

它仍然是简单单环，不包含 bubble 或重复分支。

## 数据来源与 scrubbing

- 原始 ONT run：`SRR37543650`
  - https://www.ncbi.nlm.nih.gov/sra/SRR37543650
- 研究：*A simple library preparation modification significantly reduces barcode crosstalk in ONT multiplexed sequencing*
  - https://www.biorxiv.org/content/10.1101/2025.11.19.689316v1
- 参考：φX174 `NC_001422.1`，5,386 bp
  - https://www.ncbi.nlm.nih.gov/nuccore/NC_001422.1

该run来自条码串扰研究，不能直接作为纯φX输入。这里使用参考辅助 scrubbing：

1. Q≥10，排除近全长 read；
2. 找到 identity≥95%、block≥100 bp 的可信φX片段；
3. 排除方向翻转的嵌合 read；
4. 仅裁掉可信片段外的非φX/文库尾部；
5. 不修改可信片段内部的任何碱基，不做共识或模拟；
6. 选择一个每条read都至少稳定推进100 bp、相邻真实overlap≥200 bp的闭环。

结果为12条真实、仅做尾部裁剪的reads。

## read质量与覆盖

| 指标 | 结果 |
|---|---:|
| reads | 12 |
| 总碱基 | 13,381 bp |
| read长度 | 434–2,791 bp |
| 最长read/基因组 | 51.82% |
| 名义深度 | 2.484× |
| 裁剪后query coverage | 100%（全部12条） |
| 加权read identity | 98.616% |
| 加权read错误率 | 1.384% |
| 最差read identity | 97.872% |
| 参考覆盖 | 100% |
| 参考深度 | 最低1×、中位2×、平均2.488×、最高5× |

## contained判定

按参考坐标展开一圈后，12条read的start和end都严格递增；每条read相对前一条至少推进108 bp，闭环边推进184 bp，因此没有任何一个所选参考区间被另一条完全包含。

miniasm的明显-contained预筛选同样报告：

```text
dropped 0 contained reads
```

但miniasm默认的两轮read-selection是为常规冗余覆盖组装设计的。在这套刻意压缩到2.48×的最小布局集上，第二轮启发式会错误去掉6条。因此本benchmark在已经完成显式scrubbing和containment审计后使用`-1 -2`跳过两轮read-selection。不能把默认miniasm清理结果当作本数据的节点真值。

## OLC构建

对同向版本执行精确 all-vs-all：

```bash
minimap2 -x ava-ont -k9 -w5 -m20 -n2 -s40 -c --eqx -X \
  SRR37543650.cycle12.scrubbed.forward_oriented.fastq.gz \
  SRR37543650.cycle12.scrubbed.forward_oriented.fastq.gz \
  > all_vs_all.forward_oriented.paf
```

显式保留：

```text
alignment_block >= 200 bp
matches / alignment_block >= 0.96
```

得到16个物理overlap：12条Hamiltonian环边和4条传递弦。随后执行：

```bash
miniasm -R -1 -2 -m200 -s200 -c1 -o200 -h1000 -I0.8 \
  -g100 -d0 -e1 -n0 -F1.0 -p sg \
  reliable_overlaps.i0.96.o200.paf > string_graph.cycle12.gfa
```

传递约简后恰好得到：

- 12个物理节点；
- 12条物理边；
- 每个节点唯一入边、唯一出边；
- 一个物理单环；
- GFA的正、反方向镜像表现为两个12节点oriented SCC。

阈值稳定性：identity 0.955–0.963、最短overlap 100–250 bp的20组组合全部得到同一个12节点单环。identity提高到0.964会删掉最弱真实边；最短overlap提高到300 bp会删掉可靠短边。

12条环边实际：

- overlap block：257–1,299 bp；
- pairwise identity：96.373%–99.015%。

## 已知Hamiltonian解

原始read方向下：

```text
1041(-) → 2309(+) → 88(+) → 947(-)
→ 1945(+) → 2000(-) → 1728(-) → 78(-)
→ 1552(+) → 1343(+) → 182(+) → 1384(-) → 1041(-)
```

完整ID、方向、参考展开坐标和逐边质量见`truth_cycle.tsv`。

该解在固定起点与方向后唯一。若QUBO不固定起点与方向，则有12个循环平移×2个方向，即24个等价位置编码解；它们代表同一个物理layout。

## 两个FASTQ版本

- `original_orientation.fastq.gz`：保留每条真实read的原始方向，用于支持bidirected/反向互补处理的完整OLC流程。
- `forward_oriented.fastq.gz`：将参考负链reads反向互补，使12条read方向统一；序列错误仍完全来自真实reads。

当前 `olc_overlap_framework_v0_1_0` 的 `Minimap2CandidateFinder` 会跳过`strand != '+'`的PAF，因此立即接入当前代码时应使用`forward_oriented.fastq.gz`。该版本只适合验证overlap、图构建和Hamiltonian排序，不适合评价方向推断能力。

建议当前框架至少使用：

```python
Minimap2Config(
    preset="ava-ont",
    min_overlap=200,
    max_error_rate_hint=0.04,
    overhang_tolerance=20,
    min_mapq=0,
    extra_args=("-k", "9", "-w", "5", "-m", "20", "-n", "2",
                "-s", "40", "-c", "--eqx", "-X"),
)
```

## 文件

- `SRR37543650.cycle12.scrubbed.original_orientation.fastq.gz`
- `SRR37543650.cycle12.scrubbed.forward_oriented.fastq.gz`
- `scrubbing_metadata.tsv`
- `all_vs_all.forward_oriented.paf`
- `reliable_overlaps.i0.96.o200.paf`
- `string_graph.cycle12.gfa`
- `truth_cycle.tsv`
- `directed_cycle.graphml`
- `reads_to_reference.audit_only.paf`
- `NC_001422.1.fasta`
- `SHA256SUMS`
