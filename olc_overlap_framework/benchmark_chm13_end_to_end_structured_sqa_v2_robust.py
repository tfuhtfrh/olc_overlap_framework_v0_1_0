"""Expanded robustness batch for end-to-end structured SQA v2.

Compare no acyclicity pressure, adaptive exact cycle cut, and adaptive
rank-ensemble cycle pressure from both zero and random starts over six seeds.
"""
import benchmark_chm13_end_to_end_structured_sqa as base
import benchmark_chm13_end_to_end_structured_sqa_v2 as v2
from pathlib import Path

base.METHODS=("none","exact_cut","rank_ensemble")
base.STARTS=("zero","random")
base.SEEDS=(202609301,202609302,202609303,202609304,202609305,202609306)
v2.OUT=Path("debug/qubo/chm13_end_to_end_structured_sqa_v2_robust_20260929.json")

if __name__=="__main__":
    v2.main()
