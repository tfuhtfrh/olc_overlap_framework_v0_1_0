# Plan B 复原评价链（外部代码，未修改）

来源：共同研究者交付的 `PLANB_METRICS_FOR_COLLABORATOR_20261120`（R262 部署快照的评价源码）。整个包没有放进本仓库，这里只取 freeze / audit / spell / QUAST 评价链实际依赖的 8 个文件，**内容未做任何修改**。下表的 SHA256 与原包 `SHA256SUMS.txt` 逐一核对一致（2026-11-20）。

| 文件 | 作用 |
|---|---|
| `run_metrics.py` | CLI：freeze / evaluate / parse |
| `code/independent_final_audit.py` | 在原图上独立审计路径（每条物理 read 只用一次、边来自原图、无 mate 冲突） |
| `code/spell_read_path.py` | 按原 overlap 证据从原始 read 拼出 FASTA，并输出 spans 证明 |
| `code/genome_fraction_evaluation.py` | 核对冻结清单和拼接来源，调用 QUAST，读取 report.tsv |
| `code/posthoc_reference_gate.py` | 打开参考之前要求审计 PASS（只用其中两个函数） |
| `code/runtime_utils.py` | SHA256 |
| `code/regenerate_hifi_overlaps.py` | FASTA 读取（`records`） |
| `code/kernels/graph_common.py` | 读图、物理 read 与方向的映射、mate map |

QUAST 参数固定在 `genome_fraction_evaluation.py` 中：`--min-identity 95 --extensive-mis-size 1000 --local-mis-size 200 --ambiguity-usage one`，另由调用方指定 `--eukaryote` 和线程数。QUAST 版本为 5.3.0（从 GitHub tag `quast_5.3.0` 安装）。

注意：
- `graph_common.load_graph` 会读取 pickle，只能用于可信文件。本项目只传 GraphML。
- `posthoc_reference_gate.py` 里会懒加载 `upstream/hinge_layout_audit.py`，但那条路径评价链用不到，该文件不在此处。
- 本项目的适配和扩展代码放在 `olc_overlap_framework/assembly_eval/`，不改这里的文件。

```
38ec7271e99b887279f12854f2fdd0f9ea0032bb1f751a2427d72f4c81611c27  ./code/genome_fraction_evaluation.py
f87a35f0d98dc9ed5fac4ce7368b961aa4ebd2609058026d8ea512c4e3913b6c  ./code/independent_final_audit.py
3e7274ce080a8b64989182aa87a68ffcf16b35b710a7102b3c56aee53e47b5fc  ./code/kernels/graph_common.py
6d3ae4c90f358bce3e54691b0acfc552a157b5da33bf2ef39ca138277fa568a5  ./code/posthoc_reference_gate.py
45b941a94c1acb7f76ce32899e68dcaa2d994144786280b00021999c389a603e  ./code/regenerate_hifi_overlaps.py
f23bb79aacbf44551999c8d00877b4213d1a5d11700846881f1235467a57fafa  ./code/runtime_utils.py
c7b416bebb5a67fbad68f73739f821182883de06117ca5eb49bd3f0f52b6d140  ./code/spell_read_path.py
2c11b5862ff1e4d86a5d17e1b2b7bb6f1e7a36436f77cbbd8f7ce6d182bd588f  ./run_metrics.py
```
