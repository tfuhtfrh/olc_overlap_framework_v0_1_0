"""Shared reads-only execution helpers for the two phi174 demos."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config
from olc_pipeline.data import Read
from olc_pipeline.mi_scorer import AlignmentMIScorer
from olc_pipeline.pipeline import OverlapExperimentPipeline, PipelineResult
from olc_pipeline.refiner import (
    PafCandidateEdgeAdapter,
    ParasailOverlapRefiner,
    RefinerConfig,
)
from olc_pipeline.reference_evaluator import CircularReferenceEvaluator, read_fasta_sequence
from olc_pipeline.io_utils import read_fastq


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "test_data" / "phi174"


def run_reads_only_pipeline(
    reads: list[Read],
    *,
    minimap2_bin: str,
    min_overlap: int,
    min_paf_identity: float,
    max_error_rate_hint: float,
    max_error_rate: float,
    overhang_tolerance: int,
    extra_args: tuple[str, ...],
    full_coverage_min_identity: Optional[float] = None,
    edge_mode: Literal["paf", "dp"] = "paf",
) -> tuple[Minimap2CandidateFinder, PipelineResult]:
    """Run reads-only overlap processing without passing a reference.

    ``edge_mode="paf"`` is the current phi174 preprocessing path: accepted
    minimap2 candidates become graph edges without DP realignment.  ``"dp"``
    retains the experimental Parasail path for later comparison.
    """
    finder = Minimap2CandidateFinder(Minimap2Config(
        minimap2_bin=minimap2_bin,
        preset="ava-ont",
        threads=1,
        min_overlap=min_overlap,
        max_error_rate_hint=max_error_rate_hint,
        overhang_tolerance=overhang_tolerance,
        support_reverse_strand=True,
        include_reverse_complement_edges=True,
        min_paf_identity=min_paf_identity,
        full_coverage_min_identity=full_coverage_min_identity,
        extra_args=extra_args,
    ))
    if edge_mode == "paf":
        refiner = PafCandidateEdgeAdapter()
    elif edge_mode == "dp":
        refiner = ParasailOverlapRefiner(RefinerConfig(
            min_overlap=min_overlap,
            max_error_rate=max_error_rate,
            margin=max(20, overhang_tolerance * 2),
        ))
    else:
        raise ValueError(f"unsupported edge_mode: {edge_mode!r}")
    result = OverlapExperimentPipeline(
        candidate_finder=finder,
        refiner=refiner,
        mi_scorer=AlignmentMIScorer(),
    ).run(reads)
    return finder, result


def evaluate_against_circular_reference(
    reads: list[Read],
    reference_path: Path,
    layout,
    *,
    minimap2_bin: str,
    mapping_preset: str,
):
    """Map reads after layout solving and evaluate circular order."""
    evaluator = CircularReferenceEvaluator(
        read_fasta_sequence(reference_path),
        minimap2_bin=minimap2_bin,
        preset=mapping_preset,
    )
    placements = evaluator.map_reads(reads)
    oriented_order = layout.metadata.get("oriented_order", [])
    report = evaluator.evaluate(oriented_order, placements)
    return evaluator, placements, report


def print_run_summary(finder, result: PipelineResult) -> None:
    print(f"candidate_records: {len(result.candidates)}")
    print(f"preprocessed_edges: {len(result.edges)}")
    print(f"candidate_finding_sec: {result.timings['candidate_finding_sec']:.3f}")
    print(f"edge_processing_sec: {result.timings['edge_processing_sec']:.3f}")
    print(f"minimap2_filter_counts: {dict(finder.last_filter_counts)}")


def print_reference_summary(evaluator, placements, report) -> None:
    print(f"reference_length: {evaluator.reference_length}")
    print(f"mapped_predicted_reads: {report.mapped_predicted_reads}/{report.total_predicted_reads}")
    print(f"reference_coverage_bp: {report.reference_coverage_bp}/{evaluator.reference_length}")
    print(f"orientation_accuracy: {report.orientation_accuracy}")
    print(f"forward_edge_accuracy: {report.forward_edge_accuracy:.4f}")
    print(f"reverse_edge_accuracy: {report.reverse_edge_accuracy:.4f}")
    print(f"exact_forward_cycle: {report.exact_forward_cycle}")
    print(f"exact_reverse_equivalent_cycle: {report.exact_reverse_equivalent_cycle}")
    if report.notes:
        print(f"evaluation_notes: {'; '.join(report.notes)}")
