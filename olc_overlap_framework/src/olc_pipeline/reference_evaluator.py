"""Reference-only evaluation for circular real-read layouts.

This module is deliberately downstream of overlap discovery and layout
solving.  It maps already-selected reads to a doubled reference sequence only
to determine their reference order, orientation, and coverage.  The reference
is never supplied to a candidate finder or a Hamiltonian builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional

from .data import Read
from .io_utils import write_reads_fasta

MODULE_VERSION = "0.1.0"


@dataclass(frozen=True)
class ReferencePlacement:
    """Best reference placement for one physical read."""

    read_id: str
    orientation: int
    reference_start: int
    reference_end: int
    alignment_length: int
    identity: float
    mapq: int


@dataclass(frozen=True)
class CircularOrderEvaluation:
    """Start-invariant evaluation of an oriented circular read order."""

    total_predicted_reads: int
    mapped_predicted_reads: int
    reference_coverage_bp: int
    orientation_accuracy: Optional[float]
    forward_edge_accuracy: float
    reverse_edge_accuracy: float
    exact_forward_cycle: bool
    exact_reverse_equivalent_cycle: bool
    truth_order: tuple[str, ...]
    predicted_order: tuple[str, ...]
    notes: tuple[str, ...] = ()


def read_fasta_sequence(path: Path) -> str:
    """Read and concatenate the sequence records in a FASTA file."""
    sequence: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                sequence.append(line)
    value = "".join(sequence).upper()
    if not value:
        raise ValueError(f"FASTA contains no sequence: {path}")
    return value


def parse_oriented_read_token(token: str) -> tuple[str, int]:
    """Split a layout token such as ``read42-`` into ID and orientation."""
    if not token or token[-1] not in "+-":
        raise ValueError(f"oriented read token must end with +/-: {token!r}")
    return token[:-1], +1 if token[-1] == "+" else -1


class CircularReferenceEvaluator:
    """Map reads to a doubled circular reference for evaluation only."""

    def __init__(
        self,
        reference_sequence: str,
        minimap2_bin: str = "minimap2",
        preset: str = "map-ont",
        threads: int = 1,
    ):
        self.reference_sequence = reference_sequence.upper()
        if not self.reference_sequence:
            raise ValueError("reference_sequence must not be empty")
        self.minimap2_bin = minimap2_bin
        self.preset = preset
        self.threads = threads
        self.last_command: list[str] = []
        self.last_stderr = ""

    @property
    def reference_length(self) -> int:
        return len(self.reference_sequence)

    def map_reads(self, reads: list[Read]) -> dict[str, ReferencePlacement]:
        """Return one best PAF placement per read against reference + reference."""
        if shutil.which(self.minimap2_bin) is None:
            raise RuntimeError(
                f"Cannot find minimap2 binary '{self.minimap2_bin}' for reference evaluation."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reads_path = tmp / "reads.fa"
            reference_path = tmp / "reference.circular_x2.fa"
            paf_path = tmp / "placements.paf"
            write_reads_fasta(reads, reads_path)
            with reference_path.open("w", encoding="utf-8") as handle:
                handle.write(">circular_reference_x2\n")
                doubled = self.reference_sequence + self.reference_sequence
                for start in range(0, len(doubled), 80):
                    handle.write(doubled[start:start + 80] + "\n")

            command = [
                self.minimap2_bin,
                "-x", self.preset,
                "-t", str(self.threads),
                "-c", "--eqx",
                str(reference_path),
                str(reads_path),
            ]
            self.last_command = command
            with paf_path.open("w", encoding="utf-8") as stdout:
                process = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            self.last_stderr = process.stderr
            if process.returncode != 0:
                raise RuntimeError(
                    "minimap2 reference mapping failed with return code "
                    f"{process.returncode}:\n{process.stderr}"
                )
            return self._parse_paf(paf_path)

    def evaluate(
        self,
        oriented_order: Iterable[tuple[str, int] | str],
        placements: dict[str, ReferencePlacement],
    ) -> CircularOrderEvaluation:
        """Compare a predicted order with the reference order up to rotation.

        The forward truth cycle sorts selected reads by circular-reference
        start coordinate.  The reverse-equivalent cycle reverses that order and
        flips every read orientation.  Both are biologically the same circular
        molecule, while a pure cyclic rotation is always ignored as an
        arbitrary choice of origin.
        """
        predicted = [
            parse_oriented_read_token(item) if isinstance(item, str) else item
            for item in oriented_order
        ]
        predicted_ids = [read_id for read_id, _ in predicted]
        mapped_ids = [read_id for read_id in predicted_ids if read_id in placements]
        selected_placements = {
            read_id: placements[read_id]
            for read_id in mapped_ids
        }
        truth_forward = sorted(
            (
                read_id,
                selected_placements[read_id].orientation,
            )
            for read_id in selected_placements
        )
        truth_forward.sort(
            key=lambda item: (selected_placements[item[0]].reference_start, item[0])
        )
        truth_reverse = [
            (read_id, -orientation)
            for read_id, orientation in reversed(truth_forward)
        ]

        mapped_set = set(mapped_ids)
        predicted_mapped = [item for item in predicted if item[0] in mapped_set]
        orientation_matches = [
            orientation == placements[read_id].orientation
            for read_id, orientation in predicted_mapped
        ]
        forward_edges = self._cycle_edge_accuracy(predicted_mapped, truth_forward)
        reverse_edges = self._cycle_edge_accuracy(predicted_mapped, truth_reverse)
        exact_forward = self._cyclic_equal(predicted_mapped, truth_forward)
        exact_reverse = self._cyclic_equal(predicted_mapped, truth_reverse)

        notes: list[str] = []
        if not predicted:
            notes.append("No oriented cycle order was produced by the layout solver.")
        if len(mapped_ids) != len(predicted_ids):
            notes.append("Some predicted reads did not obtain a reference placement.")
        if len(set(predicted_ids)) != len(predicted_ids):
            notes.append("Predicted order contains duplicate physical read IDs.")
        return CircularOrderEvaluation(
            total_predicted_reads=len(predicted_ids),
            mapped_predicted_reads=len(mapped_ids),
            reference_coverage_bp=self._coverage_bp(selected_placements.values()),
            orientation_accuracy=(
                sum(orientation_matches) / len(orientation_matches)
                if orientation_matches else None
            ),
            forward_edge_accuracy=forward_edges,
            reverse_edge_accuracy=reverse_edges,
            exact_forward_cycle=exact_forward,
            exact_reverse_equivalent_cycle=exact_reverse,
            truth_order=tuple(read_id for read_id, _ in truth_forward),
            predicted_order=tuple(read_id for read_id, _ in predicted_mapped),
            notes=tuple(notes),
        )

    def _parse_paf(self, path: Path) -> dict[str, ReferencePlacement]:
        best: dict[str, ReferencePlacement] = {}
        best_key: dict[str, tuple[int, float, int, int]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) < 12:
                continue
            read_id = fields[0]
            q_en = int(fields[3])
            strand = fields[4]
            t_st = int(fields[7])
            t_en = int(fields[8])
            matches = int(fields[9])
            block = int(fields[10])
            mapq = int(fields[11])
            if block <= 0:
                continue
            start = t_st % self.reference_length
            span = t_en - t_st
            placement = ReferencePlacement(
                read_id=read_id,
                orientation=+1 if strand == "+" else -1,
                reference_start=start,
                reference_end=start + span,
                alignment_length=span,
                identity=matches / block,
                mapq=mapq,
            )
            key = (span, placement.identity, mapq, q_en)
            if key > best_key.get(read_id, (-1, -1.0, -1, -1)):
                best[read_id] = placement
                best_key[read_id] = key
        return best

    def _coverage_bp(self, placements: Iterable[ReferencePlacement]) -> int:
        intervals: list[tuple[int, int]] = []
        length = self.reference_length
        for placement in placements:
            start = placement.reference_start
            end = placement.reference_end
            if end - start >= length:
                return length
            if end <= length:
                intervals.append((start, end))
            else:
                intervals.append((start, length))
                intervals.append((0, end - length))
        intervals = [(start, end) for start, end in intervals if end > start]
        if not intervals:
            return 0
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return min(length, sum(end - start for start, end in merged))

    @staticmethod
    def _cycle_edge_accuracy(
        predicted: list[tuple[str, int]],
        truth: list[tuple[str, int]],
    ) -> float:
        if not predicted:
            return 0.0
        truth_successor = {
            left: truth[(index + 1) % len(truth)]
            for index, left in enumerate(truth)
        } if truth else {}
        correct = sum(
            truth_successor.get(left) == right
            for left, right in zip(predicted, predicted[1:] + predicted[:1])
        )
        return correct / len(predicted)

    @staticmethod
    def _cyclic_equal(
        predicted: list[tuple[str, int]],
        truth: list[tuple[str, int]],
    ) -> bool:
        if len(predicted) != len(truth):
            return False
        if not predicted:
            return False
        doubled = truth + truth
        size = len(truth)
        return any(doubled[offset:offset + size] == predicted for offset in range(size))
