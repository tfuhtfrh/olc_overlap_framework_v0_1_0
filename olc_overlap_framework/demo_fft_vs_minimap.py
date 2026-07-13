"""
Compare the FFT no-gap overlap prototype with minimap2 on mismatch-only reads.

This demo uses simulated reads with substitutions only:
    mismatch_rate > 0
    ins_rate = 0
    del_rate = 0

The FFT prototype scores ungapped suffix-prefix overlaps. It is therefore a
good fit for this dataset and should be compared with minimap2 at the raw
candidate stage, before Parasail gap-aware refinement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from time import perf_counter
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
TESTS_DIR = PROJECT_DIR / "tests"
for path in (SRC_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config
from olc_pipeline.data import EdgeEvaluationReport, OverlapCandidate, Read
from olc_pipeline.evaluator import EdgeEvaluator
from olc_pipeline.io_utils import write_reads_fasta
from olc_pipeline.simulator import RandomReadSimulator, SimulationConfig
from test2 import best_suffix_prefix_overlap


GENOME_LEN = 12_000
READ_LEN = 1_500
STEP = 500
MISMATCH_RATE = 0.03
MIN_OVERLAP = 400
MIN_IDENTITY = 0.90
MAX_ERROR_RATE_HINT = 0.12
SEED = 20260618
THREADS = 1
DEBUG_DIR = Path("debug/fft_vs_minimap")


@dataclass(frozen=True)
class MethodResult:
    name: str
    candidates: list[OverlapCandidate]
    seconds: Optional[float]
    report: EdgeEvaluationReport
    note: str = ""


class FFTNoGapOverlapFinder:
    """All-vs-all adapter around tests/test2.py's ungapped FFT overlap scorer."""

    def __init__(self, min_overlap: int, min_identity: float) -> None:
        self.min_overlap = min_overlap
        self.min_identity = min_identity
        self.last_pair_count = 0

    def find_candidates(self, reads: list[Read]) -> list[OverlapCandidate]:
        candidates: list[OverlapCandidate] = []
        self.last_pair_count = 0
        for left in reads:
            for right in reads:
                if left.rid == right.rid:
                    continue
                self.last_pair_count += 1
                result = best_suffix_prefix_overlap(
                    left.seq,
                    right.seq,
                    min_overlap=self.min_overlap,
                    min_identity=self.min_identity,
                )
                if result is None:
                    continue
                candidates.append(self._to_candidate(left, right, result))
        return candidates

    @staticmethod
    def _to_candidate(left: Read, right: Read, result: dict[str, object]) -> OverlapCandidate:
        overlap_len = int(result["overlap"])
        matches = int(result["matches"])
        left_len = len(left.seq)
        right_len = len(right.seq)
        left_start = left_len - overlap_len

        return OverlapCandidate(
            left_id=left.rid,
            right_id=right.rid,
            source="fft_no_gap",
            query_id=left.rid,
            target_id=right.rid,
            strand="+",
            q_len=left_len,
            q_st=left_start,
            q_en=left_len,
            t_len=right_len,
            t_st=0,
            t_en=overlap_len,
            n_match=matches,
            aln_block_len=overlap_len,
            mapq=255,
            left_start_hint=left_start,
            left_end_hint=left_len,
            right_start_hint=0,
            right_end_hint=overlap_len,
            rough_overlap_len=overlap_len,
            rough_shift=left_start,
        )


def run_finder(finder, reads: list[Read]) -> tuple[list[OverlapCandidate], float]:
    start = perf_counter()
    candidates = finder.find_candidates(reads)
    return candidates, perf_counter() - start


def evaluate_method(
    name: str,
    reads: list[Read],
    candidates: list[OverlapCandidate],
    seconds: Optional[float],
    note: str = "",
) -> MethodResult:
    return MethodResult(
        name=name,
        candidates=candidates,
        seconds=seconds,
        report=EdgeEvaluator().evaluate_candidates(reads, candidates),
        note=note,
    )


def best_outgoing_by_left(candidates: list[OverlapCandidate]) -> list[OverlapCandidate]:
    best: dict[str, OverlapCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.left_id)
        if current is None or candidate.rough_overlap_len > current.rough_overlap_len:
            best[candidate.left_id] = candidate
    return list(best.values())


def candidate_edge_set(candidates: list[OverlapCandidate]) -> set[tuple[str, str]]:
    return {(candidate.left_id, candidate.right_id) for candidate in candidates}


def resolve_minimap2_bin(user_value: Optional[str]) -> Optional[str]:
    if user_value:
        return user_value
    from_path = shutil.which("minimap2")
    if from_path is not None:
        return from_path
    sibling_build = PROJECT_DIR.parent / "minimap2" / "minimap2"
    if sibling_build.exists():
        return str(sibling_build)
    return None


def write_truth_table(reads: list[Read], path: Path) -> None:
    reads_sorted = sorted(reads, key=lambda read: read.true_start)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("rid\ttrue_start\ttrue_end\tlength\n")
        for read in reads_sorted:
            handle.write(
                f"{read.rid}\t{read.true_start}\t{read.true_end}\t{len(read.seq)}\n"
            )


def true_adjacent_overlap_lengths(reads: list[Read]) -> list[int]:
    reads_sorted = sorted(reads, key=lambda read: read.true_start)
    overlaps: list[int] = []
    for left, right in zip(reads_sorted, reads_sorted[1:]):
        overlap_len = min(left.true_end, right.true_end) - max(left.true_start, right.true_start)
        if overlap_len > 0:
            overlaps.append(overlap_len)
    return overlaps


def read_lengths(reads: list[Read]) -> list[int]:
    return [len(read.seq) for read in reads]


def fmt_int_stats(values: list[int]) -> str:
    if not values:
        return "n/a"
    avg = sum(values) / len(values)
    return f"{min(values):,} / {avg:,.1f} / {max(values):,}"


def fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}s"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def print_kv(label: str, value: object) -> None:
    print(f"{label:<30} {value}")


def print_report(result: MethodResult) -> None:
    report = result.report
    print_section(result.name)
    print_kv("Candidates", len(result.candidates))
    print_kv("Runtime", fmt_seconds(result.seconds))
    print_kv("Adjacent correct", report.adjacent_correct)
    print_kv("Jump correct", report.jump_correct)
    print_kv("Wrong", report.wrong_edges)
    print_kv("Missing adjacent", report.missing_adjacent_edges)
    print_kv("Precision", fmt_pct(report.edge_precision))
    print_kv("Adjacent recall", fmt_pct(report.adjacent_recall))
    if result.note:
        print_kv("Note", result.note)
    print_top_candidates(result.candidates)


def print_top_candidates(candidates: list[OverlapCandidate], limit: int = 8) -> None:
    if not candidates:
        return
    print("Top overlaps:")
    for candidate in sorted(candidates, key=lambda c: c.rough_overlap_len, reverse=True)[:limit]:
        match_frac = candidate.n_match / candidate.aln_block_len if candidate.aln_block_len else 0.0
        score_label = "identity" if candidate.source == "fft_no_gap" else "match_frac"
        print(
            f"  {candidate.left_id} -> {candidate.right_id} "
            f"L={candidate.rough_overlap_len} "
            f"{score_label}={match_frac:.3f} "
            f"shift={candidate.rough_shift}"
        )


def print_summary(results: list[MethodResult]) -> None:
    print_section("Comparison Summary")
    print("Method                         Cand  Adj  Jump  Wrong  MissAdj  Prec    Recall  Runtime")
    print("-" * 91)
    for result in results:
        report = result.report
        print(
            f"{result.name:<30}"
            f"{len(result.candidates):>5} "
            f"{report.adjacent_correct:>4} "
            f"{report.jump_correct:>5} "
            f"{report.wrong_edges:>6} "
            f"{report.missing_adjacent_edges:>8} "
            f"{fmt_pct(report.edge_precision):>7} "
            f"{fmt_pct(report.adjacent_recall):>7} "
            f"{fmt_seconds(result.seconds):>8}"
        )


def print_edge_delta(label: str, left: list[OverlapCandidate], right: list[OverlapCandidate]) -> None:
    left_edges = candidate_edge_set(left)
    right_edges = candidate_edge_set(right)
    only_left = sorted(left_edges - right_edges)
    only_right = sorted(right_edges - left_edges)
    print_section(label)
    print_kv("Common edges", len(left_edges & right_edges))
    print_kv("Only left", len(only_left))
    print_kv("Only right", len(only_right))
    if only_left:
        print_kv("Only left examples", only_left[:8])
    if only_right:
        print_kv("Only right examples", only_right[:8])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate no-gap substitution reads and compare FFT overlap with minimap2."
    )
    parser.add_argument("--genome-len", type=int, default=GENOME_LEN)
    parser.add_argument("--read-len", type=int, default=READ_LEN)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--mismatch-rate", type=float, default=MISMATCH_RATE)
    parser.add_argument("--gc-fraction", type=float, default=0.5)
    parser.add_argument("--min-overlap", type=int, default=MIN_OVERLAP)
    parser.add_argument("--min-identity", type=float, default=MIN_IDENTITY)
    parser.add_argument("--max-error-rate-hint", type=float, default=MAX_ERROR_RATE_HINT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--debug-dir", type=Path, default=DEBUG_DIR)
    parser.add_argument("--minimap2-bin", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        genome_len=args.genome_len,
        read_len=args.read_len,
        step=args.step,
        mismatch_rate=args.mismatch_rate,
        ins_rate=0.0,
        del_rate=0.0,
        gc_fraction=args.gc_fraction,
        seed=args.seed,
        shuffle_reads=True,
    )
    genome, reads = RandomReadSimulator().simulate(config)

    debug_dir = PROJECT_DIR / args.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    reads_path = debug_dir / "reads.fa"
    truth_path = debug_dir / "truth.tsv"
    write_reads_fasta(reads, reads_path)
    write_truth_table(reads, truth_path)

    expected_pairwise_identity = (1.0 - args.mismatch_rate) ** 2 + (args.mismatch_rate ** 2) / 3.0

    print_section("Generated Test Data")
    print_kv("Genome length", f"{len(genome):,} bp")
    print_kv("Reads", len(reads))
    print_kv("Read length / step", f"{args.read_len:,} / {args.step:,} bp")
    print_kv("Read length min/mean/max", fmt_int_stats(read_lengths(reads)))
    print_kv("Adjacent overlap min/mean/max", fmt_int_stats(true_adjacent_overlap_lengths(reads)))
    print_kv("Error model", f"mismatch={args.mismatch_rate:.3f}, ins=0.0, del=0.0")
    print_kv("Expected pairwise identity", f"{expected_pairwise_identity:.3f}")
    print_kv("Min overlap", args.min_overlap)
    print_kv("FFT min identity", f"{args.min_identity:.3f}")
    print_kv("Reads FASTA", reads_path)
    print_kv("Truth TSV", truth_path)

    results: list[MethodResult] = []

    fft_finder = FFTNoGapOverlapFinder(
        min_overlap=args.min_overlap,
        min_identity=args.min_identity,
    )
    fft_candidates, fft_seconds = run_finder(fft_finder, reads)
    fft_raw_result = evaluate_method(
        "FFT Raw",
        reads,
        fft_candidates,
        fft_seconds,
        note="All pairwise ungapped suffix-prefix candidates from tests/test2.py.",
    )
    results.append(fft_raw_result)
    print_report(fft_raw_result)
    print_kv("Pairs scanned", fft_finder.last_pair_count)

    fft_best = best_outgoing_by_left(fft_candidates)
    fft_best_result = evaluate_method(
        "FFT Best",
        reads,
        fft_best,
        None,
        note="Longest outgoing FFT candidate per left read.",
    )
    results.append(fft_best_result)
    print_report(fft_best_result)

    minimap2_bin = resolve_minimap2_bin(args.minimap2_bin)
    if minimap2_bin is None:
        print_section("Minimap2")
        print("minimap2 was not found. Pass --minimap2-bin or run in the configured WSL environment.")
        print_summary(results)
        return

    minimap_finder = Minimap2CandidateFinder(Minimap2Config(
        minimap2_bin=minimap2_bin,
        preset="ava-ont",
        threads=args.threads,
        min_overlap=args.min_overlap,
        max_error_rate_hint=args.max_error_rate_hint,
        overhang_tolerance=80,
        min_mapq=0,
        debug_dir=debug_dir,
    ))
    minimap_candidates, minimap_seconds = run_finder(minimap_finder, reads)
    minimap_raw_result = evaluate_method(
        "Minimap2 Raw",
        reads,
        minimap_candidates,
        minimap_seconds,
        note="Same-strand suffix-prefix candidates parsed from PAF.",
    )
    results.append(minimap_raw_result)
    print_report(minimap_raw_result)

    print_section("Minimap2 Filter Counts")
    for key, value in sorted(minimap_finder.last_filter_counts.items()):
        print_kv(key.replace("_", " "), value)

    minimap_best = best_outgoing_by_left(minimap_candidates)
    minimap_best_result = evaluate_method(
        "Minimap2 Best",
        reads,
        minimap_best,
        None,
        note="Longest outgoing minimap2 candidate per left read.",
    )
    results.append(minimap_best_result)
    print_report(minimap_best_result)

    print_edge_delta("Raw Edge Set Delta: FFT vs Minimap2", fft_candidates, minimap_candidates)
    print_edge_delta("Best Edge Set Delta: FFT vs Minimap2", fft_best, minimap_best)
    print_summary(results)


if __name__ == "__main__":
    main()
