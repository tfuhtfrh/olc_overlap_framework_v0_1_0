"""
olc_pipeline.pipeline
Version: 0.1.0

High-level experiment pipeline. It connects candidate finding, overlap
refinement, MI scoring, edge filtering, and edge-level evaluation. Layout solving
is optional and can remain disabled until OR-Tools/SA/QUBO modules are ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from .data import Read, OverlapCandidate, OverlapEdge, EdgeEvaluationReport, RefinementEvaluationReport
from .candidate_finder import OverlapCandidateFinder
from .refiner import OverlapRefiner
from .mi_scorer import AlignmentMIScorer
from .evaluator import EdgeEvaluator

MODULE_VERSION = "0.1.0"


@dataclass
class PipelineResult:
    """All major outputs from one overlap experiment run."""

    candidates: list[OverlapCandidate]
    edges: list[OverlapEdge]
    candidate_report: EdgeEvaluationReport
    edge_report: EdgeEvaluationReport
    refinement_report: RefinementEvaluationReport
    timings: dict[str, float]


class OverlapExperimentPipeline:
    """
    Main pipeline for overlap-graph experiments.

    Public interface:
        run(reads) -> PipelineResult
    """

    def __init__(
        self,
        candidate_finder: OverlapCandidateFinder,
        refiner: OverlapRefiner,
        mi_scorer: Optional[AlignmentMIScorer] = None,
        edge_evaluator: Optional[EdgeEvaluator] = None,
    ):
        self.candidate_finder = candidate_finder
        self.refiner = refiner
        self.mi_scorer = mi_scorer or AlignmentMIScorer()
        self.edge_evaluator = edge_evaluator or EdgeEvaluator()

    def run(self, reads: list[Read]) -> PipelineResult:
        reads_by_id = {read.rid: read for read in reads}
        timings: dict[str, float] = {}

        t0 = perf_counter()
        candidates = self.candidate_finder.find_candidates(reads)
        timings["candidate_finding_sec"] = perf_counter() - t0

        t1 = perf_counter()
        edges: list[OverlapEdge] = []
        for candidate in candidates:
            edge = self.refiner.refine(candidate, reads_by_id)
            if edge.accepted:
                if edge.alignment is not None:
                    self.mi_scorer.attach_to_edge(edge)
                edges.append(edge)
        edges = self._deduplicate_edges(edges)
        timings["refinement_and_mi_sec"] = perf_counter() - t1

        t2 = perf_counter()
        candidate_report = self.edge_evaluator.evaluate_candidates(reads, candidates)
        edge_report = self.edge_evaluator.evaluate(reads, edges)
        refinement_report = self.edge_evaluator.evaluate_refinement(reads, edges)
        timings["evaluation_sec"] = perf_counter() - t2

        return PipelineResult(
            candidates=candidates,
            edges=edges,
            candidate_report=candidate_report,
            edge_report=edge_report,
            refinement_report=refinement_report,
            timings=timings,
        )

    @staticmethod
    def _deduplicate_edges(edges: list[OverlapEdge]) -> list[OverlapEdge]:
        """Keep the highest DP-weight edge for each directed read pair."""
        best: dict[tuple[str, str], OverlapEdge] = {}
        for edge in edges:
            key = (edge.left_id, edge.right_id)
            score = edge.weight_dp if edge.weight_dp is not None else edge.dp_score
            old = best.get(key)
            old_score = old.weight_dp if old and old.weight_dp is not None else (old.dp_score if old else float("-inf"))
            if old is None or score > old_score:
                best[key] = edge
        return list(best.values())
