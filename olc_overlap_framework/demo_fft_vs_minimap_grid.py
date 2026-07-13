"""
Run a parameter grid for the FFT no-gap overlap prototype vs minimap2.

The grid varies:
    - genome length, which changes the number of reads
    - read step, which changes true adjacent overlap length
    - substitution rate, with insertions/deletions fixed at zero

The output CSV is intended for quick inspection of accuracy and runtime trends.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from demo_fft_vs_minimap import (
    FFTNoGapOverlapFinder,
    best_outgoing_by_left,
    candidate_edge_set,
    evaluate_method,
    resolve_minimap2_bin,
    run_finder,
)
from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config
from olc_pipeline.data import OverlapCandidate, Read
from olc_pipeline.io_utils import write_reads_fasta
from olc_pipeline.simulator import RandomReadSimulator, SimulationConfig


READ_LEN = 1_500
GENOME_LENS = (6_000, 12_000, 18_000)
STEPS = (1_000, 500, 250)
MISMATCH_RATES = (0.0, 0.03, 0.06)
MIN_OVERLAP = 400
MIN_IDENTITY = 0.90
MAX_ERROR_RATE_HINT = 0.20
SEED = 20260618
THREADS = 1
OUTPUT_CSV = Path("debug/fft_vs_minimap_grid/grid.csv")


@dataclass(frozen=True)
class GridRow:
    genome_len: int
    seed: int
    read_len: int
    step: int
    true_adjacent_overlap: int
    mismatch_rate: float
    expected_pairwise_identity: float
    reads: int
    pairs_scanned: int
    min_overlap: int
    min_identity: float
    fft_raw_candidates: int
    fft_raw_adjacent: int
    fft_raw_jump: int
    fft_raw_wrong: int
    fft_raw_missing_adjacent: int
    fft_raw_precision: float
    fft_raw_recall: float
    fft_best_candidates: int
    fft_best_adjacent: int
    fft_best_wrong: int
    fft_best_missing_adjacent: int
    fft_best_precision: float
    fft_best_recall: float
    fft_seconds: float
    c_fft_raw_candidates: Optional[int]
    c_fft_raw_adjacent: Optional[int]
    c_fft_raw_jump: Optional[int]
    c_fft_raw_wrong: Optional[int]
    c_fft_raw_missing_adjacent: Optional[int]
    c_fft_raw_precision: Optional[float]
    c_fft_raw_recall: Optional[float]
    c_fft_best_candidates: Optional[int]
    c_fft_best_adjacent: Optional[int]
    c_fft_best_wrong: Optional[int]
    c_fft_best_missing_adjacent: Optional[int]
    c_fft_best_precision: Optional[float]
    c_fft_best_recall: Optional[float]
    c_fft_seconds: Optional[float]
    speedup_python_fft_vs_c_fft: Optional[float]
    c_raw_common_edges: Optional[int]
    c_raw_only_python_fft: Optional[int]
    c_raw_only_c_fft: Optional[int]
    c_best_common_edges: Optional[int]
    c_best_only_python_fft: Optional[int]
    c_best_only_c_fft: Optional[int]
    c_fft_status: str
    minimap_raw_candidates: Optional[int]
    minimap_raw_adjacent: Optional[int]
    minimap_raw_jump: Optional[int]
    minimap_raw_wrong: Optional[int]
    minimap_raw_missing_adjacent: Optional[int]
    minimap_raw_precision: Optional[float]
    minimap_raw_recall: Optional[float]
    minimap_best_candidates: Optional[int]
    minimap_best_adjacent: Optional[int]
    minimap_best_wrong: Optional[int]
    minimap_best_missing_adjacent: Optional[int]
    minimap_best_precision: Optional[float]
    minimap_best_recall: Optional[float]
    minimap_seconds: Optional[float]
    speedup_minimap_vs_fft: Optional[float]
    raw_common_edges: Optional[int]
    raw_only_fft: Optional[int]
    raw_only_minimap: Optional[int]
    best_common_edges: Optional[int]
    best_only_fft: Optional[int]
    best_only_minimap: Optional[int]
    minimap_status: str
    minimap_extra_args: str


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def expected_pairwise_identity(mismatch_rate: float) -> float:
    return (1.0 - mismatch_rate) ** 2 + (mismatch_rate ** 2) / 3.0


def resolve_c_fft_bin(user_value: Optional[str]) -> Optional[str]:
    if user_value:
        return user_value
    candidate = PROJECT_DIR / "tests" / "test2_fft_overlap"
    if candidate.exists():
        return str(candidate)
    return None


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line in {"edges_begin", "edges_end"}:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            values[parts[0]] = parts[1]
    return values


def c_fft_edges_to_candidates(
    reads: list[Read],
    stdout: str,
) -> list[OverlapCandidate]:
    candidates: list[OverlapCandidate] = []
    in_edges = False
    for line in stdout.splitlines():
        if line == "edges_begin":
            in_edges = True
            continue
        if line == "edges_end":
            break
        if not in_edges:
            continue

        src_text, dst_text, overlap_text, matches_text, _identity_text = line.split("\t")
        left = reads[int(src_text)]
        right = reads[int(dst_text)]
        overlap_len = int(overlap_text)
        matches = int(matches_text)
        left_len = len(left.seq)
        right_len = len(right.seq)
        left_start = left_len - overlap_len
        candidates.append(OverlapCandidate(
            left_id=left.rid,
            right_id=right.rid,
            source="c_fft_no_gap",
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
        ))
    return candidates


def run_c_fft_benchmark(
    c_fft_bin: str,
    reads: list[Read],
    min_overlap: int,
    min_identity: float,
) -> tuple[list[OverlapCandidate], float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        reads_path = Path(tmpdir) / "reads.fa"
        write_reads_fasta(reads, reads_path)
        proc = subprocess.run(
            [
                c_fft_bin,
                "--benchmark",
                str(reads_path),
                "--min-overlap",
                str(min_overlap),
                "--min-identity",
                str(min_identity),
                "--emit-edges",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"C FFT exited with {proc.returncode}")

    values = parse_key_values(proc.stdout)
    return c_fft_edges_to_candidates(reads, proc.stdout), float(values["seconds"])


def run_grid_row(
    genome_len: int,
    read_len: int,
    step: int,
    mismatch_rate: float,
    min_overlap: int,
    min_identity: float,
    c_fft_bin: Optional[str],
    minimap2_bin: Optional[str],
    max_error_rate_hint: float,
    minimap_extra_args: tuple[str, ...],
    threads: int,
    seed: int,
    gc_fraction: float,
) -> GridRow:
    config = SimulationConfig(
        genome_len=genome_len,
        read_len=read_len,
        step=step,
        mismatch_rate=mismatch_rate,
        ins_rate=0.0,
        del_rate=0.0,
        gc_fraction=gc_fraction,
        seed=seed,
        shuffle_reads=True,
    )
    _, reads = RandomReadSimulator().simulate(config)

    fft_finder = FFTNoGapOverlapFinder(
        min_overlap=min_overlap,
        min_identity=min_identity,
    )
    fft_candidates, fft_seconds = run_finder(fft_finder, reads)
    fft_raw = evaluate_method("FFT Raw", reads, fft_candidates, fft_seconds)
    fft_best_candidates = best_outgoing_by_left(fft_candidates)
    fft_best = evaluate_method("FFT Best", reads, fft_best_candidates, None)

    c_fft_candidates = None
    c_fft_seconds = None
    c_fft_raw = None
    c_fft_best = None
    c_fft_best_candidates = None
    c_fft_status = "skipped"

    if c_fft_bin is not None:
        try:
            c_fft_candidates, c_fft_seconds = run_c_fft_benchmark(
                c_fft_bin,
                reads,
                min_overlap,
                min_identity,
            )
            c_fft_raw = evaluate_method("C FFT Raw", reads, c_fft_candidates, c_fft_seconds)
            c_fft_best_candidates = best_outgoing_by_left(c_fft_candidates)
            c_fft_best = evaluate_method("C FFT Best", reads, c_fft_best_candidates, None)
            c_fft_status = "ok"
        except Exception as exc:  # pragma: no cover - diagnostic demo path
            c_fft_status = f"error: {exc}"

    minimap_candidates = None
    minimap_seconds = None
    minimap_raw = None
    minimap_best = None
    minimap_best_candidates = None
    minimap_status = "skipped"

    if minimap2_bin is not None:
        try:
            minimap_finder = Minimap2CandidateFinder(Minimap2Config(
                minimap2_bin=minimap2_bin,
                preset="ava-ont",
                threads=threads,
                min_overlap=min_overlap,
                max_error_rate_hint=max_error_rate_hint,
                overhang_tolerance=80,
                min_mapq=0,
                extra_args=minimap_extra_args,
            ))
            minimap_candidates, minimap_seconds = run_finder(minimap_finder, reads)
            minimap_raw = evaluate_method("Minimap2 Raw", reads, minimap_candidates, minimap_seconds)
            minimap_best_candidates = best_outgoing_by_left(minimap_candidates)
            minimap_best = evaluate_method("Minimap2 Best", reads, minimap_best_candidates, None)
            minimap_status = "ok"
        except Exception as exc:  # pragma: no cover - diagnostic demo path
            minimap_status = f"error: {exc}"

    raw_common = raw_only_fft = raw_only_minimap = None
    best_common = best_only_fft = best_only_minimap = None
    speedup = None
    if minimap_candidates is not None and minimap_best_candidates is not None and minimap_seconds is not None:
        fft_edges = candidate_edge_set(fft_candidates)
        minimap_edges = candidate_edge_set(minimap_candidates)
        raw_common = len(fft_edges & minimap_edges)
        raw_only_fft = len(fft_edges - minimap_edges)
        raw_only_minimap = len(minimap_edges - fft_edges)

        fft_best_edges = candidate_edge_set(fft_best_candidates)
        minimap_best_edges = candidate_edge_set(minimap_best_candidates)
        best_common = len(fft_best_edges & minimap_best_edges)
        best_only_fft = len(fft_best_edges - minimap_best_edges)
        best_only_minimap = len(minimap_best_edges - fft_best_edges)
        speedup = fft_seconds / minimap_seconds if minimap_seconds > 0 else None

    c_raw_common = c_raw_only_python = c_raw_only_c = None
    c_best_common = c_best_only_python = c_best_only_c = None
    c_speedup = None
    if c_fft_candidates is not None and c_fft_best_candidates is not None and c_fft_seconds is not None:
        py_edges = candidate_edge_set(fft_candidates)
        c_edges = candidate_edge_set(c_fft_candidates)
        c_raw_common = len(py_edges & c_edges)
        c_raw_only_python = len(py_edges - c_edges)
        c_raw_only_c = len(c_edges - py_edges)

        py_best_edges = candidate_edge_set(fft_best_candidates)
        c_best_edges = candidate_edge_set(c_fft_best_candidates)
        c_best_common = len(py_best_edges & c_best_edges)
        c_best_only_python = len(py_best_edges - c_best_edges)
        c_best_only_c = len(c_best_edges - py_best_edges)
        c_speedup = fft_seconds / c_fft_seconds if c_fft_seconds > 0 else None

    return GridRow(
        genome_len=genome_len,
        seed=seed,
        read_len=read_len,
        step=step,
        true_adjacent_overlap=read_len - step,
        mismatch_rate=mismatch_rate,
        expected_pairwise_identity=expected_pairwise_identity(mismatch_rate),
        reads=len(reads),
        pairs_scanned=fft_finder.last_pair_count,
        min_overlap=min_overlap,
        min_identity=min_identity,
        fft_raw_candidates=len(fft_candidates),
        fft_raw_adjacent=fft_raw.report.adjacent_correct,
        fft_raw_jump=fft_raw.report.jump_correct,
        fft_raw_wrong=fft_raw.report.wrong_edges,
        fft_raw_missing_adjacent=fft_raw.report.missing_adjacent_edges,
        fft_raw_precision=fft_raw.report.edge_precision,
        fft_raw_recall=fft_raw.report.adjacent_recall,
        fft_best_candidates=len(fft_best_candidates),
        fft_best_adjacent=fft_best.report.adjacent_correct,
        fft_best_wrong=fft_best.report.wrong_edges,
        fft_best_missing_adjacent=fft_best.report.missing_adjacent_edges,
        fft_best_precision=fft_best.report.edge_precision,
        fft_best_recall=fft_best.report.adjacent_recall,
        fft_seconds=fft_seconds,
        c_fft_raw_candidates=len(c_fft_candidates) if c_fft_candidates is not None else None,
        c_fft_raw_adjacent=c_fft_raw.report.adjacent_correct if c_fft_raw else None,
        c_fft_raw_jump=c_fft_raw.report.jump_correct if c_fft_raw else None,
        c_fft_raw_wrong=c_fft_raw.report.wrong_edges if c_fft_raw else None,
        c_fft_raw_missing_adjacent=c_fft_raw.report.missing_adjacent_edges if c_fft_raw else None,
        c_fft_raw_precision=c_fft_raw.report.edge_precision if c_fft_raw else None,
        c_fft_raw_recall=c_fft_raw.report.adjacent_recall if c_fft_raw else None,
        c_fft_best_candidates=len(c_fft_best_candidates) if c_fft_best_candidates is not None else None,
        c_fft_best_adjacent=c_fft_best.report.adjacent_correct if c_fft_best else None,
        c_fft_best_wrong=c_fft_best.report.wrong_edges if c_fft_best else None,
        c_fft_best_missing_adjacent=c_fft_best.report.missing_adjacent_edges if c_fft_best else None,
        c_fft_best_precision=c_fft_best.report.edge_precision if c_fft_best else None,
        c_fft_best_recall=c_fft_best.report.adjacent_recall if c_fft_best else None,
        c_fft_seconds=c_fft_seconds,
        speedup_python_fft_vs_c_fft=c_speedup,
        c_raw_common_edges=c_raw_common,
        c_raw_only_python_fft=c_raw_only_python,
        c_raw_only_c_fft=c_raw_only_c,
        c_best_common_edges=c_best_common,
        c_best_only_python_fft=c_best_only_python,
        c_best_only_c_fft=c_best_only_c,
        c_fft_status=c_fft_status,
        minimap_raw_candidates=len(minimap_candidates) if minimap_candidates is not None else None,
        minimap_raw_adjacent=minimap_raw.report.adjacent_correct if minimap_raw else None,
        minimap_raw_jump=minimap_raw.report.jump_correct if minimap_raw else None,
        minimap_raw_wrong=minimap_raw.report.wrong_edges if minimap_raw else None,
        minimap_raw_missing_adjacent=minimap_raw.report.missing_adjacent_edges if minimap_raw else None,
        minimap_raw_precision=minimap_raw.report.edge_precision if minimap_raw else None,
        minimap_raw_recall=minimap_raw.report.adjacent_recall if minimap_raw else None,
        minimap_best_candidates=len(minimap_best_candidates) if minimap_best_candidates is not None else None,
        minimap_best_adjacent=minimap_best.report.adjacent_correct if minimap_best else None,
        minimap_best_wrong=minimap_best.report.wrong_edges if minimap_best else None,
        minimap_best_missing_adjacent=minimap_best.report.missing_adjacent_edges if minimap_best else None,
        minimap_best_precision=minimap_best.report.edge_precision if minimap_best else None,
        minimap_best_recall=minimap_best.report.adjacent_recall if minimap_best else None,
        minimap_seconds=minimap_seconds,
        speedup_minimap_vs_fft=speedup,
        raw_common_edges=raw_common,
        raw_only_fft=raw_only_fft,
        raw_only_minimap=raw_only_minimap,
        best_common_edges=best_common,
        best_only_fft=best_only_fft,
        best_only_minimap=best_only_minimap,
        minimap_status=minimap_status,
        minimap_extra_args=" ".join(minimap_extra_args),
    )


def write_csv(rows: list[GridRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def fmt_seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def fmt_ratio(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}x"


def print_row(row: GridRow) -> None:
    print(
        f"genome={row.genome_len:<6} step={row.step:<4} "
        f"ovl={row.true_adjacent_overlap:<4} err={row.mismatch_rate:.2f} "
        f"reads={row.reads:<3} "
        f"FFT best recall={row.fft_best_recall:.2f} "
        f"C best recall={row.c_fft_best_recall if row.c_fft_best_recall is not None else 'n/a'} "
        f"MM best recall={row.minimap_best_recall if row.minimap_best_recall is not None else 'n/a'} "
        f"FFT={fmt_seconds(row.fft_seconds)} "
        f"C={fmt_seconds(row.c_fft_seconds)} "
        f"MM={fmt_seconds(row.minimap_seconds)} "
        f"C speedup={fmt_ratio(row.speedup_python_fft_vs_c_fft)} "
        f"MM speedup={fmt_ratio(row.speedup_minimap_vs_fft)} "
        f"raw delta FFT/MM={row.raw_only_fft}/{row.raw_only_minimap} "
        f"FFT/C={row.c_raw_only_python_fft}/{row.c_raw_only_c_fft}"
    )


def print_summary(rows: list[GridRow], output_csv: Path) -> None:
    minimap_rows = [row for row in rows if row.minimap_seconds is not None]
    print("\n== Grid Summary ==")
    print(f"Rows: {len(rows)}")
    print(f"CSV: {output_csv}")
    c_rows = [row for row in rows if row.c_fft_seconds is not None]
    if c_rows:
        avg_c_speedup = sum(row.speedup_python_fft_vs_c_fft or 0.0 for row in c_rows) / len(c_rows)
        max_c_speedup = max(row.speedup_python_fft_vs_c_fft or 0.0 for row in c_rows)
        print(f"Average C FFT speedup vs Python FFT: {avg_c_speedup:.1f}x")
        print(f"Max C FFT speedup vs Python FFT: {max_c_speedup:.1f}x")
        c_differing = [
            row for row in c_rows
            if row.c_raw_only_python_fft != 0 or row.c_raw_only_c_fft != 0
        ]
        print(f"Rows with Python/C FFT edge-set differences: {len(c_differing)}")
    if minimap_rows:
        avg_speedup = sum(row.speedup_minimap_vs_fft or 0.0 for row in minimap_rows) / len(minimap_rows)
        max_speedup = max(row.speedup_minimap_vs_fft or 0.0 for row in minimap_rows)
        print(f"Average minimap2 speedup vs FFT: {avg_speedup:.1f}x")
        print(f"Max minimap2 speedup vs FFT: {max_speedup:.1f}x")

        differing = [
            row for row in minimap_rows
            if row.raw_only_fft != 0 or row.raw_only_minimap != 0
        ]
        print(f"Rows with raw edge-set differences: {len(differing)}")
        if differing:
            print("First differences:")
            for row in differing[:8]:
                print_row(row)

    weak_fft = [row for row in rows if row.fft_best_recall < 1.0 or row.fft_best_wrong > 0]
    print(f"Rows where FFT Best is imperfect: {len(weak_fft)}")
    for row in weak_fft[:8]:
        print_row(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid sweep for FFT no-gap overlap vs minimap2."
    )
    parser.add_argument("--read-len", type=int, default=READ_LEN)
    parser.add_argument("--genome-lens", default=",".join(str(v) for v in GENOME_LENS))
    parser.add_argument("--steps", default=",".join(str(v) for v in STEPS))
    parser.add_argument("--mismatch-rates", default=",".join(str(v) for v in MISMATCH_RATES))
    parser.add_argument("--min-overlap", type=int, default=MIN_OVERLAP)
    parser.add_argument("--min-identity", type=float, default=MIN_IDENTITY)
    parser.add_argument("--max-error-rate-hint", type=float, default=MAX_ERROR_RATE_HINT)
    parser.add_argument("--gc-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--c-fft-bin", default=None)
    parser.add_argument("--skip-c-fft", action="store_true")
    parser.add_argument("--minimap2-bin", default=None)
    parser.add_argument("--minimap-extra-args", default="")
    parser.add_argument("--skip-minimap2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genome_lens = parse_int_list(args.genome_lens)
    steps = parse_int_list(args.steps)
    mismatch_rates = parse_float_list(args.mismatch_rates)
    seeds = parse_int_list(args.seeds) if args.seeds else (args.seed,)
    minimap_extra_args = tuple(shlex.split(args.minimap_extra_args))
    c_fft_bin = None if args.skip_c_fft else resolve_c_fft_bin(args.c_fft_bin)
    minimap2_bin = None if args.skip_minimap2 else resolve_minimap2_bin(args.minimap2_bin)

    rows: list[GridRow] = []
    total = len(genome_lens) * len(steps) * len(mismatch_rates) * len(seeds)
    current = 0
    for genome_len in genome_lens:
        for step in steps:
            if step >= args.read_len:
                raise ValueError("Every step must be smaller than read_len so adjacent reads overlap.")
            for mismatch_rate in mismatch_rates:
                for seed in seeds:
                    current += 1
                    print(
                        f"[{current}/{total}] running genome={genome_len}, "
                        f"step={step}, mismatch={mismatch_rate}, seed={seed}"
                    )
                    row = run_grid_row(
                        genome_len=genome_len,
                        read_len=args.read_len,
                        step=step,
                        mismatch_rate=mismatch_rate,
                        min_overlap=args.min_overlap,
                        min_identity=args.min_identity,
                        c_fft_bin=c_fft_bin,
                        minimap2_bin=minimap2_bin,
                        max_error_rate_hint=args.max_error_rate_hint,
                        minimap_extra_args=minimap_extra_args,
                        threads=args.threads,
                        seed=seed,
                        gc_fraction=args.gc_fraction,
                    )
                    rows.append(row)
                    print_row(row)

    write_csv(rows, PROJECT_DIR / args.output_csv)
    print_summary(rows, PROJECT_DIR / args.output_csv)


if __name__ == "__main__":
    main()
