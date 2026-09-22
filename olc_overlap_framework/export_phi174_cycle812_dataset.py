"""Export the current 812-read phiX174 graph as a data-only package."""

from __future__ import annotations

import csv
import gzip
import shutil
import zipfile
from pathlib import Path

from demo_phi174_cycle742_cut_dag_qubo import DATASET, _build_accepted_graph


PACKAGE_NAME = "SRR27862880_phiX174_OLC_graph812_i0.990_o80"
PACKAGE_DIR = DATASET.parent / PACKAGE_NAME
ARCHIVE_PATH = PACKAGE_DIR.parent / f"{PACKAGE_NAME}.zip"


def _raw_interval(
    start: int,
    end: int,
    length: int,
    orientation: int,
) -> tuple[int, int]:
    if orientation == 1:
        return start, end
    if orientation == -1:
        return length - end, length - start
    raise ValueError(f"invalid orientation: {orientation}")


def _write_reads_subset(
    source: Path,
    destination: Path,
    retained_ids: set[str],
) -> int:
    found: set[str] = set()
    with gzip.open(source, "rt", encoding="utf-8") as input_handle:
        with gzip.open(destination, "wt", encoding="utf-8", newline="\n") as output_handle:
            while True:
                header = input_handle.readline()
                if not header:
                    break
                sequence = input_handle.readline()
                plus = input_handle.readline()
                quality = input_handle.readline()
                if not sequence or not plus or not quality:
                    raise ValueError(f"incomplete FASTQ record in {source}")
                read_id = header[1:].strip().split()[0]
                if read_id in retained_ids:
                    output_handle.write(header)
                    output_handle.write(sequence)
                    output_handle.write(plus)
                    output_handle.write(quality)
                    found.add(read_id)
    if found != retained_ids:
        missing = sorted(retained_ids - found)
        raise ValueError(f"retained reads missing from source FASTQ: {missing}")
    return len(found)


def _write_graph_tsv(path: Path, graph) -> None:
    fieldnames = [
        "left_id",
        "left_orientation",
        "right_id",
        "right_orientation",
        "left_start",
        "left_end",
        "right_start",
        "right_end",
        "overlap_len",
        "shift",
        "matches",
        "mismatches",
        "edit_distance",
        "identity",
        "error_rate",
        "mapq",
        "dp_score",
        "weight_dp",
        "candidate_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for edge in graph.edges:
            writer.writerow({name: getattr(edge, name) for name in fieldnames})


def _write_graph_paf(path: Path, graph) -> None:
    lengths = {read.rid: len(read.seq) for read in graph.reads}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for edge in graph.edges:
            query_length = lengths[edge.left_id]
            target_length = lengths[edge.right_id]
            query_start, query_end = _raw_interval(
                edge.left_start,
                edge.left_end,
                query_length,
                edge.left_orientation,
            )
            target_start, target_end = _raw_interval(
                edge.right_start,
                edge.right_end,
                target_length,
                edge.right_orientation,
            )
            strand = (
                "+"
                if edge.left_orientation == edge.right_orientation
                else "-"
            )
            fields = [
                edge.left_id,
                str(query_length),
                str(query_start),
                str(query_end),
                strand,
                edge.right_id,
                str(target_length),
                str(target_start),
                str(target_end),
                str(edge.matches),
                str(edge.overlap_len),
                str(edge.mapq),
                "tp:A:P",
                f"lo:i:{edge.left_orientation}",
                f"ro:i:{edge.right_orientation}",
                f"sh:i:{edge.shift}",
                f"de:f:{edge.error_rate:.8f}",
            ]
            handle.write("\t".join(fields) + "\n")


def _write_graph_gfa(path: Path, graph) -> None:
    lengths = {read.rid: len(read.seq) for read in graph.reads}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\n")
        for read in graph.reads:
            orientation = graph.orientation_by_read[read.rid]
            sign = "+" if orientation == 1 else "-"
            handle.write(
                f"S\t{read.rid}\t*\tLN:i:{lengths[read.rid]}\tOR:A:{sign}\n"
            )
        for edge in graph.edges:
            left_sign = "+" if edge.left_orientation == 1 else "-"
            right_sign = "+" if edge.right_orientation == 1 else "-"
            handle.write(
                f"L\t{edge.left_id}\t{left_sign}\t{edge.right_id}\t"
                f"{right_sign}\t*\tOL:i:{edge.overlap_len}\t"
                f"SH:i:{edge.shift}\tNM:i:{edge.edit_distance}\t"
                f"ID:f:{edge.identity:.8f}\n"
            )


def main() -> None:
    graph = _build_accepted_graph("minimap2")
    retained_ids = {read.rid for read in graph.reads}
    edge_keys = {
        (
            edge.left_id,
            edge.left_orientation,
            edge.right_id,
            edge.right_orientation,
        )
        for edge in graph.edges
    }
    if len(graph.reads) != 812:
        raise ValueError(f"expected 812 reads, got {len(graph.reads)}")
    if len(edge_keys) != 8385:
        raise ValueError(f"expected 8385 unique edges, got {len(edge_keys)}")
    if any(
        edge.left_orientation != graph.orientation_by_read[edge.left_id]
        or edge.right_orientation != graph.orientation_by_read[edge.right_id]
        for edge in graph.edges
    ):
        raise ValueError("graph edge orientation disagrees with node orientation")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    reads_path = PACKAGE_DIR / "reads.fastq.gz"
    graph_tsv_path = PACKAGE_DIR / "graph_edges.tsv"
    graph_paf_path = PACKAGE_DIR / "graph_edges.paf"
    graph_gfa_path = PACKAGE_DIR / "graph.gfa"
    reference_path = PACKAGE_DIR / "reference_NC_001422.1.fasta"

    exported_reads = _write_reads_subset(
        DATASET / "SRR27862880.stride500.unique.fastq.gz",
        reads_path,
        retained_ids,
    )
    _write_graph_tsv(graph_tsv_path, graph)
    _write_graph_paf(graph_paf_path, graph)
    _write_graph_gfa(graph_gfa_path, graph)
    shutil.copyfile(DATASET / "NC_001422.1.fasta", reference_path)

    package_files = [
        reads_path,
        graph_tsv_path,
        graph_paf_path,
        graph_gfa_path,
        reference_path,
    ]
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_files:
            archive.write(path, arcname=f"{PACKAGE_NAME}/{path.name}")

    print(f"package_dir: {PACKAGE_DIR}")
    print(f"archive: {ARCHIVE_PATH}")
    print(f"reads: {exported_reads}")
    print(f"unique_directed_edges: {len(edge_keys)}")
    print(f"files: {len(package_files)}")


if __name__ == "__main__":
    main()
