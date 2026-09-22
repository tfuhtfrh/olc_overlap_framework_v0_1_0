"""Audit physical read placement and mapping error against a reference."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from olc_pipeline.io_utils import read_fastq
from olc_pipeline.reference_evaluator import CircularReferenceEvaluator, read_fasta_sequence


def write_report(reads, placements, path: Path) -> dict[str, float | int]:
    rows = []
    for read in reads:
        placement = placements.get(read.rid)
        if placement is None:
            continue
        rows.append((
            read.rid,
            placement.orientation,
            placement.reference_start,
            placement.reference_end,
            placement.alignment_length,
            placement.identity,
            1.0 - placement.identity,
            placement.mapq,
        ))
    rows.sort(key=lambda row: (row[2], row[0]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "read_id\torientation\tref_start\tref_end\talignment_length\t"
            "identity\terror_rate\tmapq\n"
        )
        for row in rows:
            handle.write(
                f"{row[0]}\t{row[1]:+d}\t{row[2]}\t{row[3]}\t{row[4]}\t"
                f"{row[5]:.8f}\t{row[6]:.8f}\t{row[7]}\n"
            )

    errors = [row[6] for row in rows]
    total_span = sum(row[4] for row in rows)
    weighted_identity = sum(row[5] * row[4] for row in rows)
    return {
        "input_reads": len(reads),
        "mapped_reads": len(rows),
        "unmapped_reads": len(reads) - len(rows),
        "mean_error_rate": statistics.mean(errors) if errors else 0.0,
        "median_error_rate": statistics.median(errors) if errors else 0.0,
        "max_error_rate": max(errors) if errors else 0.0,
        "length_weighted_error_rate": (
            1.0 - weighted_identity / total_span if total_span else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reads", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimap2-bin", default="minimap2")
    parser.add_argument("--preset", default="map-ont")
    args = parser.parse_args()

    reads = read_fastq(args.reads)
    evaluator = CircularReferenceEvaluator(
        read_fasta_sequence(args.reference),
        minimap2_bin=args.minimap2_bin,
        preset=args.preset,
    )
    placements = evaluator.map_reads(reads)
    summary = write_report(reads, placements, args.output)
    print(f"reference_length: {evaluator.reference_length}")
    for key, value in summary.items():
        if key.endswith("error_rate"):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")
    print(f"mapping_report: {args.output}")


if __name__ == "__main__":
    main()
