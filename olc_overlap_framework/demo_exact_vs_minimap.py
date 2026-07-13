"""
Compare the original exact-overlap prototype with minimap2 candidates.

This demo intentionally uses error-free simulated reads. The original prototype
only detects exact same-strand suffix-prefix overlaps, so noisy reads with
mismatches/insertions/deletions are outside its valid input domain.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import DefaultDict, Optional


# ============================================================
# Experiment settings
#
# For everyday use, edit this block and run:
#   python demo_exact_vs_minimap.py
#
# Command-line flags still work as temporary overrides.
# ============================================================

USE_RANDOM_LAYOUT = False

GENOME_LEN = 100_000
MIN_OVERLAP = 200
SEED = 42
THREADS = 1

# Used only when USE_RANDOM_LAYOUT = False.
READ_LEN = 3_0000
STEP = 5000

# Used only when USE_RANDOM_LAYOUT = True.
READ_LEN_MIN = 2_000
READ_LEN_MAX = 8_000
OVERLAP_FRACTION_MIN = 0.10
OVERLAP_FRACTION_MAX = 0.80

# Optional compiled C++ benchmark binaries. Set to paths such as
# "tests/test1_overlap" and "tests/test1_all_outgoing" after compiling in WSL.
CPP_BIN: Optional[str] = None
CPP_ALL_BIN: Optional[str] = None
MINIMAP2_BIN: Optional[str] = None


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config
from olc_pipeline.data import EdgeEvaluationReport, OverlapCandidate, Read
from olc_pipeline.evaluator import EdgeEvaluator
from olc_pipeline.io_utils import write_reads_fasta
from olc_pipeline.simulator import RandomReadSimulator, SimulationConfig


CODE = {"A": 1, "G": 2, "T": 3, "C": 4}
MOD1 = 1_000_000_007
MOD2 = 1_000_000_009
BASE1 = 911_382_323
BASE2 = 972_663_749


@dataclass(frozen=True)
class CompressionResult:
    unique_seqs: list[str]
    class_members: list[list[int]]
    read_to_class: list[int]
    seq_to_class: dict[str, int]


@dataclass(frozen=True)
class RollingHashPowers:
    pow1: list[int]
    pow2: list[int]


@dataclass(frozen=True)
class MethodResult:
    name: str
    candidates: list[OverlapCandidate]
    seconds: Optional[float]
    report: EdgeEvaluationReport
    note: str = ""


@dataclass(frozen=True)
class CppBenchmarkResult:
    binary: str
    reads: int
    unique_classes: int
    candidates: int
    seconds: float
    edges: tuple[tuple[int, int, int], ...] = ()


class DoubleRollingHash:
    def __init__(self, seq: str, powers: RollingHashPowers) -> None:
        self.seq = seq
        self.n = len(seq)
        self.pow1 = powers.pow1
        self.pow2 = powers.pow2
        self.pref1 = [0] * (self.n + 1)
        self.pref2 = [0] * (self.n + 1)

        for idx, base in enumerate(seq):
            value = CODE[base]
            self.pref1[idx + 1] = (self.pref1[idx] * BASE1 + value) % MOD1
            self.pref2[idx + 1] = (self.pref2[idx] * BASE2 + value) % MOD2

    def substring_hash(self, start: int, end: int) -> tuple[int, int]:
        h1 = (self.pref1[end] - self.pref1[start] * self.pow1[end - start]) % MOD1
        h2 = (self.pref2[end] - self.pref2[start] * self.pow2[end - start]) % MOD2
        return h1, h2

    def prefix_hash(self, length: int) -> tuple[int, int]:
        return self.substring_hash(0, length)

    def suffix_hash(self, length: int) -> tuple[int, int]:
        return self.substring_hash(self.n - length, self.n)


class OriginalExactBestOutgoingFinder:
    """
    Adapter for the original rolling-hash prototype.

    Input:
        list[Read] containing uppercase A/C/G/T reads without sequencing errors.

    Output:
        list[OverlapCandidate], at most one outgoing suffix-prefix candidate per
        source read. This mirrors the prototype's best_outgoing behavior rather
        than minimap2's all-candidate behavior.
    """

    def __init__(self, min_overlap: int, allow_full_length: bool = False) -> None:
        self.min_overlap = min_overlap
        self.allow_full_length = allow_full_length
        self.last_unique_classes = 0
        self.last_total_classes = 0

    def find_candidates(self, reads: list[Read]) -> list[OverlapCandidate]:
        comp = self._compress_reads([read.seq for read in reads])
        self.last_unique_classes = len(comp.unique_seqs)
        self.last_total_classes = len(reads)

        best_by_class = self._find_best_by_class(
            comp.unique_seqs,
            min_overlap=self.min_overlap,
            allow_full_length=self.allow_full_length,
        )
        return self._expand_to_candidates(reads, comp, best_by_class)

    @staticmethod
    def _compress_reads(seqs: list[str]) -> CompressionResult:
        seq_to_class: dict[str, int] = {}
        unique_seqs: list[str] = []
        class_members: list[list[int]] = []
        read_to_class: list[int] = []

        for read_idx, seq in enumerate(seqs):
            for base in seq:
                if base not in CODE:
                    raise ValueError(
                        "Original exact finder only accepts uppercase A/C/G/T reads; "
                        f"read index {read_idx} contains {base!r}."
                    )

            class_id = seq_to_class.get(seq)
            if class_id is None:
                class_id = len(unique_seqs)
                seq_to_class[seq] = class_id
                unique_seqs.append(seq)
                class_members.append([])

            class_members[class_id].append(read_idx)
            read_to_class.append(class_id)

        return CompressionResult(
            unique_seqs=unique_seqs,
            class_members=class_members,
            read_to_class=read_to_class,
            seq_to_class=seq_to_class,
        )

    @staticmethod
    def _build_powers(max_len: int) -> RollingHashPowers:
        pow1 = [1] * (max_len + 1)
        pow2 = [1] * (max_len + 1)
        for idx in range(max_len):
            pow1[idx + 1] = (pow1[idx] * BASE1) % MOD1
            pow2[idx + 1] = (pow2[idx] * BASE2) % MOD2
        return RollingHashPowers(pow1=pow1, pow2=pow2)

    def _find_best_by_class(
        self,
        unique_seqs: list[str],
        min_overlap: int,
        allow_full_length: bool,
    ) -> dict[int, tuple[int, int]]:
        if not unique_seqs:
            return {}

        lengths = [len(seq) for seq in unique_seqs]
        powers = self._build_powers(max(lengths))
        hashers = [DoubleRollingHash(seq, powers) for seq in unique_seqs]
        best_out: dict[int, tuple[int, int]] = {}

        for overlap_len in range(max(lengths), min_overlap - 1, -1):
            prefix_map: DefaultDict[tuple[int, int], list[int]] = defaultdict(list)
            suffix_map: DefaultDict[tuple[int, int], list[int]] = defaultdict(list)

            for class_id, hasher in enumerate(hashers):
                read_len = lengths[class_id]
                if read_len < overlap_len:
                    continue
                if not allow_full_length and read_len == overlap_len:
                    continue

                prefix_map[hasher.prefix_hash(overlap_len)].append(class_id)
                if class_id not in best_out:
                    suffix_map[hasher.suffix_hash(overlap_len)].append(class_id)

            for hash_value, src_classes in suffix_map.items():
                dst_classes = prefix_map.get(hash_value)
                if not dst_classes:
                    continue

                for src_class in src_classes:
                    if src_class in best_out:
                        continue
                    suffix = unique_seqs[src_class][-overlap_len:]

                    for dst_class in dst_classes:
                        if dst_class == src_class:
                            continue
                        if suffix != unique_seqs[dst_class][:overlap_len]:
                            continue
                        best_out[src_class] = (dst_class, overlap_len)
                        break

        return best_out

    @staticmethod
    def _expand_to_candidates(
        reads: list[Read],
        comp: CompressionResult,
        best_by_class: dict[int, tuple[int, int]],
    ) -> list[OverlapCandidate]:
        candidates: list[OverlapCandidate] = []

        for src_class, member_indices in enumerate(comp.class_members):
            best = best_by_class.get(src_class)
            if best is None:
                continue

            dst_class, overlap_len = best
            dst_read_idx = comp.class_members[dst_class][0]
            right = reads[dst_read_idx]

            for src_read_idx in member_indices:
                left = reads[src_read_idx]
                if left.rid == right.rid:
                    continue

                left_len = len(left.seq)
                right_len = len(right.seq)
                left_start = left_len - overlap_len
                candidates.append(OverlapCandidate(
                    left_id=left.rid,
                    right_id=right.rid,
                    source="original_exact_best",
                    query_id=left.rid,
                    target_id=right.rid,
                    strand="+",
                    q_len=left_len,
                    q_st=left_start,
                    q_en=left_len,
                    t_len=right_len,
                    t_st=0,
                    t_en=overlap_len,
                    n_match=overlap_len,
                    aln_block_len=overlap_len,
                    mapq=255,
                    left_start_hint=left_start,
                    left_end_hint=left_len,
                    right_start_hint=0,
                    right_end_hint=overlap_len,
                    rough_overlap_len=overlap_len,
                    rough_shift=left_start,
                ))

        return candidates


def best_outgoing_by_left(candidates: list[OverlapCandidate]) -> list[OverlapCandidate]:
    best: dict[str, OverlapCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.left_id)
        if current is None or candidate.rough_overlap_len > current.rough_overlap_len:
            best[candidate.left_id] = candidate
    return list(best.values())


def resolve_minimap2_bin(user_value: Optional[str]) -> Optional[str]:
    if user_value:
        return user_value
    from_path = shutil.which("minimap2")
    if from_path is not None:
        return from_path

    sibling_build = PROJECT_DIR.parent / "minimap2" / "minimap2"
    if os.name != "nt" and sibling_build.exists():
        return str(sibling_build)

    return None


def run_finder(finder, reads: list[Read]) -> tuple[list[OverlapCandidate], float]:
    start = perf_counter()
    candidates = finder.find_candidates(reads)
    return candidates, perf_counter() - start


def run_cpp_benchmark(
    cpp_bin: str,
    reads: list[Read],
    min_overlap: int,
    emit_edges: bool = False,
) -> CppBenchmarkResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "reads.fa"
        write_reads_fasta(reads, fasta_path)
        cmd = [
            cpp_bin,
            "--benchmark",
            str(fasta_path),
            "--min-overlap",
            str(min_overlap),
            "--allow-full-length",
            "0",
            "--verify-exact",
            "1",
        ]
        if emit_edges:
            cmd.extend(["--emit-edges", "1"])

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            "C++ benchmark failed with return code "
            f"{proc.returncode}:\n{proc.stderr}"
        )

    values = parse_key_value_output(proc.stdout)
    edges = parse_cpp_edges(proc.stdout)
    return CppBenchmarkResult(
        binary=cpp_bin,
        reads=int(values["reads"]),
        unique_classes=int(values["unique_classes"]),
        candidates=int(values["candidates"]),
        seconds=float(values["seconds"]),
        edges=tuple(edges),
    )


def parse_key_value_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            values[parts[0]] = parts[1]
    return values


def parse_cpp_edges(text: str) -> list[tuple[int, int, int]]:
    edges: list[tuple[int, int, int]] = []
    in_edges = False
    for line in text.splitlines():
        if line == "edges_begin":
            in_edges = True
            continue
        if line == "edges_end":
            break
        if not in_edges:
            continue

        src_text, dst_text, overlap_text = line.split("\t")
        edges.append((int(src_text), int(dst_text), int(overlap_text)))
    return edges


def cpp_edges_to_candidates(
    reads: list[Read],
    edges: tuple[tuple[int, int, int], ...],
    source: str,
) -> list[OverlapCandidate]:
    candidates: list[OverlapCandidate] = []
    for src_read, dst_read, overlap_len in edges:
        left = reads[src_read]
        right = reads[dst_read]
        left_len = len(left.seq)
        right_len = len(right.seq)
        left_start = left_len - overlap_len
        candidates.append(OverlapCandidate(
            left_id=left.rid,
            right_id=right.rid,
            source=source,
            query_id=left.rid,
            target_id=right.rid,
            strand="+",
            q_len=left_len,
            q_st=left_start,
            q_en=left_len,
            t_len=right_len,
            t_st=0,
            t_en=overlap_len,
            n_match=overlap_len,
            aln_block_len=overlap_len,
            mapq=255,
            left_start_hint=left_start,
            left_end_hint=left_len,
            right_start_hint=0,
            right_end_hint=overlap_len,
            rough_overlap_len=overlap_len,
            rough_shift=left_start,
        ))
    return candidates


def evaluate_method(
    name: str,
    reads: list[Read],
    candidates: list[OverlapCandidate],
    seconds: Optional[float],
    note: str = "",
) -> MethodResult:
    report = EdgeEvaluator().evaluate_candidates(reads, candidates)
    return MethodResult(
        name=name,
        candidates=candidates,
        seconds=seconds,
        report=report,
        note=note,
    )


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
    print_kv("Shift error mean/median/max", (
        f"{fmt_float(report.mean_shift_error, 2)} / "
        f"{fmt_float(report.median_shift_error, 2)} / "
        f"{report.max_shift_error if report.max_shift_error is not None else 'n/a'} bp"
    ))
    if result.note:
        print_kv("Note", result.note)
    print_top_candidates(result.candidates)


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
    print()
    print("Read this table as:")
    print("  Adj/Recall  = recovery of true neighboring suffix-prefix overlaps.")
    print("  Jump        = correct but non-adjacent overlaps, common in raw all-vs-all output.")
    print("  Wrong       = candidate direction/geometry is not supported by simulated truth.")
    if any(result.name == "Minimap2 Best" for result in results):
        print("  Minimap2 Best is the fairer row for the original script, because both are one outgoing edge per read.")


def print_parameter_diagnostics(
    reads: list[Read],
    min_overlap: int,
    read_len_min: int,
    random_layout: bool,
) -> None:
    adjacent_overlaps = true_adjacent_overlap_lengths(reads)
    min_true_overlap = min(adjacent_overlaps) if adjacent_overlaps else None

    notes: list[str] = []
    if min_overlap < 20:
        notes.append(
            "min_overlap is very small; exact all-outgoing will include many random A/C/G/T prefix-suffix matches."
        )
    if random_layout and read_len_min < 100:
        notes.append(
            "read_len_min is very small; minimap2 may miss short reads/short overlaps because its seeding is not tuned for this regime."
        )
    if min_true_overlap is not None and min_overlap < max(20, min_true_overlap // 4):
        notes.append(
            f"min_overlap is far below the shortest true adjacent overlap ({min_true_overlap} bp), so weak random overlaps are allowed."
        )

    if not notes:
        return

    print_section("Parameter Diagnostics")
    for note in notes:
        print(f"- {note}")


def print_cpp_benchmark(result: CppBenchmarkResult, python_seconds: float, python_candidates: int) -> None:
    print_section("C++ Original Benchmark")
    print_kv("Binary", result.binary)
    print_kv("Reads", result.reads)
    print_kv("Unique classes", result.unique_classes)
    print_kv("Candidates", result.candidates)
    print_kv("Runtime", fmt_seconds(result.seconds))
    print_kv("Speedup vs Python Original", f"{python_seconds / result.seconds:.2f}x")
    if result.candidates != python_candidates:
        print_kv("Warning", f"C++ candidates ({result.candidates}) != Python candidates ({python_candidates})")


def count_true_adjacent_edges(reads: list[Read]) -> int:
    return len(true_adjacent_overlap_lengths(reads))


def true_adjacent_overlap_lengths(reads: list[Read]) -> list[int]:
    reads_sorted = sorted(reads, key=lambda read: read.true_start)
    overlaps: list[int] = []
    for left, right in zip(reads_sorted, reads_sorted[1:]):
        overlap_len = min(left.true_end, right.true_end) - max(left.true_start, right.true_start)
        if overlap_len > 0:
            overlaps.append(overlap_len)
    return overlaps


def read_lengths(reads: list[Read]) -> list[int]:
    return [read.true_end - read.true_start for read in reads]


def fmt_int_stats(values: list[int]) -> str:
    if not values:
        return "n/a"
    avg = sum(values) / len(values)
    return f"{min(values):,} / {avg:,.1f} / {max(values):,}"


def print_top_candidates(candidates: list[OverlapCandidate], limit: int = 8) -> None:
    if not candidates:
        return
    print("Top overlaps:")
    for cand in sorted(candidates, key=lambda c: c.rough_overlap_len, reverse=True)[:limit]:
        print(
            f"  {cand.left_id} -> {cand.right_id} "
            f"L={cand.rough_overlap_len} shift={cand.rough_shift}"
        )


def fmt_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}s"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def print_kv(label: str, value: object) -> None:
    print(f"{label:<28} {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact best-outgoing overlaps with minimap2 on error-free reads."
    )
    parser.add_argument("--genome-len", type=int, default=GENOME_LEN)
    parser.add_argument("--read-len", type=int, default=READ_LEN)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--read-len-min", type=int, default=READ_LEN_MIN)
    parser.add_argument("--read-len-max", type=int, default=READ_LEN_MAX)
    parser.add_argument("--overlap-fraction-min", type=float, default=OVERLAP_FRACTION_MIN)
    parser.add_argument("--overlap-fraction-max", type=float, default=OVERLAP_FRACTION_MAX)
    parser.add_argument("--gc-fraction", type=float, default=0.5)
    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument("--random-layout", dest="random_layout", action="store_true")
    layout_group.add_argument("--fixed-layout", dest="random_layout", action="store_false")
    parser.set_defaults(random_layout=None)
    parser.add_argument("--min-overlap", type=int, default=MIN_OVERLAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--cpp-bin",
        default=CPP_BIN,
        help="Optional compiled C++ original-overlap benchmark binary.",
    )
    parser.add_argument(
        "--cpp-all-bin",
        default=CPP_ALL_BIN,
        help="Optional compiled C++ all-outgoing original-overlap benchmark binary.",
    )
    parser.add_argument(
        "--minimap2-bin",
        default=MINIMAP2_BIN,
        help="Path/name for minimap2. Defaults to minimap2 from PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_random_layout = USE_RANDOM_LAYOUT if args.random_layout is None else args.random_layout

    if not use_random_layout:
        config = SimulationConfig(
            genome_len=args.genome_len,
            read_len=args.read_len,
            step=args.step,
            mismatch_rate=0.0,
            ins_rate=0.0,
            del_rate=0.0,
            gc_fraction=args.gc_fraction,
            seed=args.seed,
            shuffle_reads=True,
        )
    else:
        config = SimulationConfig(
            genome_len=args.genome_len,
            read_len=args.read_len,
            step=args.step,
            read_len_min=args.read_len_min,
            read_len_max=args.read_len_max,
            overlap_fraction_min=args.overlap_fraction_min,
            overlap_fraction_max=args.overlap_fraction_max,
            adjacent_overlap_min=args.min_overlap,
            mismatch_rate=0.0,
            ins_rate=0.0,
            del_rate=0.0,
            gc_fraction=args.gc_fraction,
            seed=args.seed,
            shuffle_reads=True,
        )
    genome, reads = RandomReadSimulator().simulate(config)

    print_section("Input")
    print_kv("Genome length", f"{len(genome):,} bp")
    print_kv("Reads", len(reads))
    if not use_random_layout:
        print_kv("Layout", "fixed read length and fixed step")
        print_kv("Read length / step", f"{config.read_len:,} / {config.step:,} bp")
    else:
        print_kv("Layout", "random read length and random adjacent overlap")
        print_kv("Configured read length", f"{args.read_len_min:,}-{args.read_len_max:,} bp")
        print_kv("Configured overlap frac", f"{args.overlap_fraction_min:.2f}-{args.overlap_fraction_max:.2f}")
        print_kv("Generated overlap lower bound", f">= min-overlap ({args.min_overlap:,} bp)")
    print_kv("Read length min/mean/max", fmt_int_stats(read_lengths(reads)))
    print_kv("Adjacent overlap min/mean/max", fmt_int_stats(true_adjacent_overlap_lengths(reads)))
    print_kv("Error rates", "mismatch=0.0%, ins=0.0%, del=0.0%")
    print_kv("Min overlap", args.min_overlap)
    print_kv("True adjacent edges", count_true_adjacent_edges(reads))
    print_kv("Original method domain", "exact A/C/G/T, same-strand, best outgoing only")
    print_parameter_diagnostics(
        reads,
        min_overlap=args.min_overlap,
        read_len_min=args.read_len_min,
        random_layout=use_random_layout,
    )

    results: list[MethodResult] = []

    exact_finder = OriginalExactBestOutgoingFinder(min_overlap=args.min_overlap)
    exact_candidates, exact_seconds = run_finder(exact_finder, reads)
    exact_result = evaluate_method(
        "Original Exact Best",
        reads,
        exact_candidates,
        exact_seconds,
        note="One exact outgoing suffix-prefix overlap per read.",
    )
    results.append(exact_result)
    print_report(exact_result)
    print_kv("Unique sequence classes", exact_finder.last_unique_classes)

    if args.cpp_bin:
        cpp_result = run_cpp_benchmark(args.cpp_bin, reads, args.min_overlap)
        print_cpp_benchmark(cpp_result, exact_seconds, len(exact_candidates))

    if args.cpp_all_bin:
        cpp_all_result = run_cpp_benchmark(
            args.cpp_all_bin,
            reads,
            args.min_overlap,
            emit_edges=True,
        )
        cpp_all_candidates = cpp_edges_to_candidates(
            reads,
            cpp_all_result.edges,
            source="cpp_original_all",
        )
        cpp_all_method_result = evaluate_method(
            "C++ Original All",
            reads,
            cpp_all_candidates,
            cpp_all_result.seconds,
            note="All exact outgoing suffix-prefix overlaps from the C++ variant.",
        )
        results.append(cpp_all_method_result)
        print_cpp_benchmark(cpp_all_result, exact_seconds, len(cpp_all_candidates))
        print_report(cpp_all_method_result)

    minimap2_bin = resolve_minimap2_bin(args.minimap2_bin)
    if minimap2_bin is None:
        print_section("Minimap2")
        print("minimap2 was not found on PATH, so the minimap2 side was skipped.")
        print("Run this demo in WSL/Linux, or pass --minimap2-bin with a Linux minimap2 path.")
        print_summary(results)
        return

    minimap_finder = Minimap2CandidateFinder(Minimap2Config(
        minimap2_bin=minimap2_bin,
        preset="ava-ont",
        threads=args.threads,
        min_overlap=args.min_overlap,
        max_error_rate_hint=0.05,
        overhang_tolerance=80,
        min_mapq=0,
        debug_dir=PROJECT_DIR / "debug" / "exact_vs_minimap",
    ))

    minimap_candidates, minimap_seconds = run_finder(minimap_finder, reads)
    minimap_raw_result = evaluate_method(
        "Minimap2 Raw",
        reads,
        minimap_candidates,
        minimap_seconds,
        note="All accepted same-strand suffix-prefix candidates from PAF.",
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
        note="Post-processed raw minimap2 candidates: longest outgoing edge per left read.",
    )
    results.append(minimap_best_result)
    print_report(minimap_best_result)
    print_summary(results)


if __name__ == "__main__":
    main()
