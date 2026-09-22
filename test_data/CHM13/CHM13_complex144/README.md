# CHM13 complex144：大核心、多解、参考最优的 read-level OLC layout 测试

## 推荐输入与已验证结果

推荐使用 `graph.graphml`：它包含 144 条物理 reads，每条已选择一个方向，保留 261 条通过质量与 dovetail 几何筛选的有向边。目标是自由起终点的 Hamilton path，不固定首尾。

| 指标 | 结果 |
|---|---:|
| 物理 read 节点 | 144 |
| 有向边 | 261 |
| 出度>1 / 入度>1 节点 | 86 / 87 |
| 非平凡 SCC | 112, 11 节点 |
| 全部 Hamilton paths | 192 |
| 参考路径分数 | 1343093 |
| 第二名分数 | 1338209 |
| 参考领先 | 4884 |
| 参与位置变化的 reads | 113 |
| 解集合使用的非参考邻接边 | 27 |

穷尽求解在自由起终点条件下找到 192 条路径并以 UNSAT 结束，因此这里是精确总数。若能量定义为 `E=-sum(weight)`，参考路径是严格唯一的最低能量顺序；整条路径的反向互补镜像不在单方向图内重复计算。

最大 SCC 有 112 个节点。它内部自由端点有 288 条 Hamilton paths；加入核心前后各一个参考相邻端口后仍有 96 条。第二个 11 节点 SCC 的对应数量为 31 和 2。因此这一档的困难性来自一个长距离重复核心中的真实全局顺序选择，不只是单条回边把很多节点包进 SCC。

该图仍可被经典 SMT 在较短时间内穷尽，适合验证更复杂 layout、能量排序和子环约束；它不构成经典困难性或量子优势证据。

## 文件

- `reads.fasta`：144 条未改碱基的原始 HiFi reads，原始测序方向，顺序以固定种子打乱。
- `reads.normalized.fasta`：相同 reads，按源 BAM 参考方向统一；负链 reads 仅做反向互补。
- `reference.fasta`：CHM13 v1.1 chr2 [89565221,91212559)，长度 1647338 bp。
- `graph.graphml`：推荐的方向归一 layout 图；节点名是 `read_id+sign`，sign 指相对原始 FASTA 的选择。
- `graph_normalized.graphml`：与前者同构，节点名仅为 read ID，对应 `reads.normalized.fasta`。
- `graph_with_rc.graphml`：推荐图及其完整反向互补镜像；不含方向预处理排除的跨相位边。
- `graph_full_oriented.graphml`：方向预处理前的全部 ± 候选边，仅供审计。
- `nodes.tsv`、`edges.tsv`：推荐图的节点和边表，便于不使用 GraphML 时直接读取。
- `WEIGHT_SPEC.md`、`weight_spec.json`：人类可读及机器可读的权重定义、缩放关系和得分约定。
- `overlaps.exact.paf.gz`：在本包 144 条原始 reads 上重新运行的 exact all-vs-all 结果。
- `reference_path.json`、`truth.json`：已知参考顺序、方向和源 BAM 坐标，仅用于答案与审计。
- `weighted_certificate.json`：192 条完整路径的穷尽证书、分数直方图和前十名路径。
- `solution_variation.json`：不同解中位置变化的 reads 与实际使用过的邻接边。
- `final_summary.json`、`reference_audit.json`、`containment_audit.json`、`full_oriented_audit.json`：拓扑、阈值、误差、包含和方向审计。
- `topology.png`：按参考顺序标注的静态拓扑概览；参考坐标只用于画图横轴。
- `verification/`：从 PAF 重建图并重新穷尽全部路径的脚本。

解压后在数据目录运行：

```bash
PYTHONPATH=verification/deps python3 verification/verify_complex144.py .
```

安装等价版本的 `networkx` 和 `z3-solver` 后也可直接运行，不必使用包内依赖。

## overlap 和权重

候选比对：

```bash
minimap2 -x map-hifi -X -k19 -w10 -m100 -n3 -s100 -K 500k -c --eqx -t 4 reads.fasta reads.fasta -o overlaps.exact.paf
```

每条保留证据满足：

- exact-CIGAR identity >= 98.0%；
- `min(query_span,target_span) >= 300 bp`；
- 方向归一后为 suffix-to-prefix dovetail；
- 两条 read 的未比对外侧均超过端点容差 `min(100,max(5,floor(0.02*min_span)))`。

同一有向节点对若有多条证据，选择下式最大的证据：

`weight = matching_bases - 49*errors`

其中 `errors = alignment_block - matching_bases`。这与旧尺度严格等价：

`1000*matching_bases - 980*alignment_block = 20*weight`

除以 20 不会改变任何路径排序、最优解或简并度，只会减小整数系数。它衡量超过 98% 质量门槛的证据量：一个错误会抵消 49 个正确碱基的收益，避免“更长但明显更差”的重复 overlap 仅靠长度占优。权重只来自 read-read 比对，没有参考坐标奖励，也没有添加参考边。更严格的 98.5% 版本使用其自身最大公约数约简后的 `3*M-197*d`，保留 48 条完整路径，参考仍严格最优；不同 identity 阈值下的绝对分数不可直接横向比较。

最短 overlap 改成 500 bp 时仍有 192 条路径，参考仍最优；改成 1000 bp 时有 48 条。主结果使用 300 bp，因为可靠短边不应被长度阈值先验删除。

没有进行传递约简、tip 修剪、bubble 弹出或 unitig 压缩。

## 为什么先处理方向

这一段含倒置重复。完整 ± 图共有 288 个定向节点、702 条边；直接把方向和顺序同时交给同一个 matching-bases/质量余量目标时，审计结果为：发现更高分的混合方向路径。这些路径可以在重复拷贝之间反复改变 read 方向，不代表参考上的单分子顺序。

本数据的目标是继续测试 layout，因此用源 BAM 的 primary alignment strand 预先选择每条 read 的方向。该步骤明确使用参考，是 benchmark 整理的一部分。`graph_full_oriented.graphml` 保留原证据，便于以后单独研究 orientation synchronization；不要把它与 `graph.graphml` 的“参考严格最优”结论混用。

## 数据质量、包含关系与筛选披露

- 参考对齐加权 identity：99.801163018%，表观错误率 0.198837%。
- 最低单-read identity：99.505592%；最低 query coverage：99.593120%。
- 参考区间覆盖：100.000000%；选择后数据量 3302988 bp，名义骨架深度 2.005x。
- read 长度 min/median/max：{'min': 15361, 'median': 22879.5, 'max': 31859}。
- 穷尽物理 read 对和两个方向的 full-read infix 审计：identity>=98% 的 contained 配对 0，identity>=98.5% 的 contained 配对 0，identity>=99% 的 contained 配对 0。

原始来源为官方 CHM13 v1.1 HiFi 主比对 BAM。先从 chr2 [88500000,91200000) 取得 5075 条 primary reads；按源 BAM 近似 identity>=99.5% 初筛后，参考辅助贪心生成相邻参考 overlap>=10 kb 的 253-read 代表骨架。随后对完整 253-read 骨架执行 exact all-vs-all，用图拓扑定位核心，取连续骨架索引 [100:253]。最后按全对 full-read infix 审计删除 9 条在 98% identity 下被另一条 read 完整包含的查询 read，得到本包 144 条。删除清单与原命中证据在 `selection.json`。复杂区间定位来自 overlap 图；骨架构造和方向归一使用了参考。

因此这是参考辅助整理的真实 read benchmark，适合测试已知答案下的 layout，不适合当作位置盲、无偏的从头组装性能评测。

## 来源

- 官方项目：https://github.com/marbl/CHM13
- v1.1 说明：https://github.com/marbl/CHM13/blob/master/Previous_assembly_release_CHM13.md
- HiFi BAM：https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/alignments/chm13.draft_v1.1.hifi_20k.wm_2.01.pri.bam
- 参考序列：https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chm13.v1.1.fasta
- T2T-CHM13 论文：https://www.science.org/doi/10.1126/science.abj6987

权重归一化整理日期：2026-09-08。
