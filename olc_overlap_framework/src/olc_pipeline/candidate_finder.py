"""
olc_pipeline.candidate_finder
Version: 0.1.0

Candidate overlap finders. The main implemented finder calls command-line
minimap2 in all-vs-all overlap mode and parses PAF records into directed
suffix-prefix overlap candidates.

The OriginalCandidateFinder is intentionally a stub so the user's custom
algorithm can later be inserted behind the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional

from .data import ContainmentEvidence, FullCoverageEvidence, Read, OverlapCandidate
from .io_utils import write_reads_fasta

MODULE_VERSION = "0.1.0"


class OverlapCandidateFinder(ABC):
    """
    Interface for all candidate pair generators.

    Input:
        list[Read]
    Output:
        list[OverlapCandidate]
    """

    @abstractmethod
    def find_candidates(self, reads: list[Read]) -> list[OverlapCandidate]:
        raise NotImplementedError


@dataclass(frozen=True)
class Minimap2Config:
    """Runtime and filtering options for minimap2 all-vs-all overlap."""

    minimap2_bin: str = "minimap2"
    preset: str = "ava-ont"
    threads: int = 1
    min_overlap: int = 500
    max_error_rate_hint: float = 0.25
    overhang_tolerance: int = 50
    min_mapq: int = 0
    support_reverse_strand: bool = False
    include_reverse_complement_edges: bool = False
    min_paf_identity: Optional[float] = None
    full_coverage_min_identity: Optional[float] = None
    containment_terminal_tolerance: int = 2
    extra_args: tuple[str, ...] = ()
    debug_dir: Optional[Path] = None


class Minimap2CandidateFinder(OverlapCandidateFinder):
    """
    Use command-line minimap2 to find all-vs-all read overlaps.

    This finder does not perform base-level refinement. It only converts PAF
    records into directed suffix-prefix candidates. Downstream refiner modules
    should use parasail/edlib/WFA to compute exact edge quality if needed.
    """

    def __init__(self, config: Optional[Minimap2Config] = None):
        self.config = config or Minimap2Config()
        self.last_command: list[str] = []
        self.last_stderr: str = ""
        self.last_filter_counts: Counter[str] = Counter()
        self.last_containment_evidence: list[ContainmentEvidence] = []
        self.last_full_coverage_evidence: list[FullCoverageEvidence] = []
        self.last_debug_paf_path: Optional[Path] = None

    def find_candidates(self, reads: list[Read]) -> list[OverlapCandidate]:
        if shutil.which(self.config.minimap2_bin) is None:
            raise RuntimeError(
                f"Cannot find minimap2 binary '{self.config.minimap2_bin}'. "
                "Install minimap2 or set Minimap2Config.minimap2_bin."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fasta_path = tmp / "reads.fa"
            paf_path = tmp / "overlaps.paf"
            write_reads_fasta(reads, fasta_path)

            cmd = [
                self.config.minimap2_bin,
                "-x", self.config.preset,
                "-t", str(self.config.threads),
                *self.config.extra_args,
                str(fasta_path),
                str(fasta_path),
            ]
            self.last_command = cmd

            with open(paf_path, "w", encoding="utf-8") as stdout:
                proc = subprocess.run(
                    cmd,
                    stdout=stdout,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            self.last_stderr = proc.stderr

            if proc.returncode != 0:
                raise RuntimeError(
                    "minimap2 failed with return code "
                    f"{proc.returncode}:\n{proc.stderr}"
                )

            self._save_debug_files(fasta_path, paf_path)
            return list(self._parse_paf(paf_path))

    def _save_debug_files(self, fasta_path: Path, paf_path: Path) -> None:
        self.last_debug_paf_path = None
        if self.config.debug_dir is None:
            return

        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_fasta = debug_dir / "reads.fa"
        debug_paf = debug_dir / "overlaps.paf"
        shutil.copyfile(fasta_path, debug_fasta)
        shutil.copyfile(paf_path, debug_paf)
        self.last_debug_paf_path = debug_paf

    def _parse_paf(self, paf_path: Path) -> Iterable[OverlapCandidate]:
        with open(paf_path, "r", encoding="utf-8") as handle:
            yield from self._parse_paf_lines(handle)

    def _parse_paf_lines(self, lines: Iterable[str]) -> Iterable[OverlapCandidate]:
        counts: Counter[str] = Counter()
        containment_keys: set[tuple[str, str, int, int]] = set()
        full_coverage_keys: set[tuple[str, str]] = set()
        self.last_containment_evidence = []
        self.last_full_coverage_evidence = []
        for line in lines:
            if not line.strip():
                continue
            counts["raw_records"] += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                counts["skipped_malformed"] += 1
                continue

            q_id = fields[0]
            q_len = int(fields[1])
            q_st = int(fields[2])
            q_en = int(fields[3])
            strand = fields[4]
            t_id = fields[5]
            t_len = int(fields[6])
            t_st = int(fields[7])
            t_en = int(fields[8])
            n_match = int(fields[9])
            aln_block_len = int(fields[10])
            mapq = int(fields[11])
            tags = self._parse_optional_tags(fields[12:])

            if q_id == t_id:
                counts["skipped_self"] += 1
                continue
            if mapq < self.config.min_mapq:
                counts["skipped_mapq"] += 1
                continue
            if aln_block_len < self.config.min_overlap:
                counts["skipped_min_overlap"] += 1
                continue
            if aln_block_len <= 0:
                counts["skipped_empty_alignment"] += 1
                continue

            # Capture the node-reduction evidence before the directed-edge
            # quality/geometry filters.  Full-coverage evidence is governed
            # by its own identity threshold and must not disappear merely
            # because the same PAF record cannot form a positive-shift edge.
            full_coverage = self._paf_to_full_coverage_evidence(
                q_id=q_id,
                q_len=q_len,
                q_st=q_st,
                q_en=q_en,
                strand=strand,
                t_id=t_id,
                t_len=t_len,
                t_st=t_st,
                t_en=t_en,
                n_match=n_match,
                aln_block_len=aln_block_len,
                mapq=mapq,
            )
            full_coverage_threshold = self.config.full_coverage_min_identity
            if full_coverage_threshold is None:
                full_coverage_threshold = self.config.min_paf_identity
            if (
                full_coverage is not None
                and full_coverage_threshold is not None
                and n_match / aln_block_len >= full_coverage_threshold
            ):
                key = tuple(sorted((full_coverage.first_id, full_coverage.second_id)))
                if key not in full_coverage_keys:
                    full_coverage_keys.add(key)
                    self.last_full_coverage_evidence.append(full_coverage)
                    counts["full_coverage_evidence"] += 1

            if (
                self.config.min_paf_identity is not None
                and n_match / aln_block_len < self.config.min_paf_identity
            ):
                counts["skipped_paf_identity"] += 1
                continue

            error_rate_hint = self._error_rate_hint(n_match, aln_block_len, tags)
            if error_rate_hint > self.config.max_error_rate_hint:
                counts["skipped_error_hint"] += 1
                continue

            containment = self._paf_to_containment_evidence(
                q_id=q_id,
                q_len=q_len,
                q_st=q_st,
                q_en=q_en,
                strand=strand,
                t_id=t_id,
                t_len=t_len,
                t_st=t_st,
                t_en=t_en,
                n_match=n_match,
                aln_block_len=aln_block_len,
                mapq=mapq,
            )
            if containment is not None:
                key = (
                    containment.contained_id,
                    containment.container_id,
                    containment.contained_orientation,
                    containment.container_orientation,
                )
                if key not in containment_keys:
                    containment_keys.add(key)
                    self.last_containment_evidence.append(containment)
                    counts["containment_evidence"] += 1

            if strand != "+" and not self.config.support_reverse_strand:
                counts["skipped_reverse_strand"] += 1
                continue

            cand = self._paf_to_directed_candidate(
                q_id=q_id,
                q_len=q_len,
                q_st=q_st,
                q_en=q_en,
                strand=strand,
                t_id=t_id,
                t_len=t_len,
                t_st=t_st,
                t_en=t_en,
                n_match=n_match,
                aln_block_len=aln_block_len,
                mapq=mapq,
            )
            if cand is not None:
                counts["accepted"] += 1
                yield cand
                if self.config.include_reverse_complement_edges:
                    counts["accepted_reverse_complement"] += 1
                    yield self._reverse_complement_candidate(cand)
            else:
                counts["skipped_geometry"] += 1
        self.last_filter_counts = counts

    @staticmethod
    def _paf_to_full_coverage_evidence(
        q_id: str,
        q_len: int,
        q_st: int,
        q_en: int,
        strand: str,
        t_id: str,
        t_len: int,
        t_st: int,
        t_en: int,
        n_match: int,
        aln_block_len: int,
        mapq: int,
    ) -> Optional[FullCoverageEvidence]:
        """Recognize exact full-read coverage, including coextensive pairs."""
        q_full = q_st == 0 and q_en == q_len
        t_full = t_st == 0 and t_en == t_len
        q_is_covered = q_full and q_len < t_len
        t_is_covered = t_full and t_len < q_len
        coextensive = q_full and t_full and q_len == t_len
        if not (q_is_covered or t_is_covered or coextensive):
            return None
        return FullCoverageEvidence(
            first_id=q_id,
            second_id=t_id,
            first_orientation=+1,
            second_orientation=+1 if strand == "+" else -1,
            alignment_length=aln_block_len,
            identity=n_match / aln_block_len,
            mapq=mapq,
        )

    def _paf_to_containment_evidence(
        self,
        q_id: str,
        q_len: int,
        q_st: int,
        q_en: int,
        strand: str,
        t_id: str,
        t_len: int,
        t_st: int,
        t_en: int,
        n_match: int,
        aln_block_len: int,
        mapq: int,
    ) -> Optional[ContainmentEvidence]:
        """Recognize full-read/internal-target PAF containment evidence."""
        tolerance = self.config.containment_terminal_tolerance
        target_orientation = +1 if strand == "+" else -1
        q_full = q_st <= tolerance and q_len - q_en <= tolerance
        t_internal = t_st > tolerance and t_len - t_en > tolerance
        if q_full and t_internal and q_len < t_len:
            return ContainmentEvidence(
                contained_id=q_id,
                container_id=t_id,
                contained_orientation=+1,
                container_orientation=target_orientation,
                alignment_length=aln_block_len,
                identity=n_match / aln_block_len,
                mapq=mapq,
            )

        t_full = t_st <= tolerance and t_len - t_en <= tolerance
        q_internal = q_st > tolerance and q_len - q_en > tolerance
        if t_full and q_internal and t_len < q_len:
            return ContainmentEvidence(
                contained_id=t_id,
                container_id=q_id,
                contained_orientation=target_orientation,
                container_orientation=+1,
                alignment_length=aln_block_len,
                identity=n_match / aln_block_len,
                mapq=mapq,
            )
        return None

    @staticmethod
    def _parse_optional_tags(fields: list[str]) -> dict[str, str]:
        tags: dict[str, str] = {}
        for field in fields:
            parts = field.split(":", 2)
            if len(parts) == 3:
                tags[parts[0]] = parts[2]
        return tags

    @staticmethod
    def _error_rate_hint(
        n_match: int,
        aln_block_len: int,
        tags: dict[str, str],
    ) -> float:
        """
        Estimate divergence for coarse minimap2 filtering.

        In all-vs-all overlap mode, minimap2 may report chaining-oriented PAF
        fields where n_match / aln_block_len is much lower than the `dv` tag.
        Prefer minimap2's divergence tag when it is available, and fall back to
        the mandatory PAF columns for records without tags.
        """
        for tag in ("dv", "de"):
            value = tags.get(tag)
            if value is not None:
                try:
                    return float(value)
                except ValueError:
                    pass
        return 1.0 - (n_match / aln_block_len)

    def _near_start(self, pos: int) -> bool:
        return pos <= self.config.overhang_tolerance

    def _near_end(self, pos: int, length: int) -> bool:
        return (length - pos) <= self.config.overhang_tolerance

    def _paf_to_directed_candidate(
        self,
        q_id: str,
        q_len: int,
        q_st: int,
        q_en: int,
        strand: str,
        t_id: str,
        t_len: int,
        t_st: int,
        t_en: int,
        n_match: int,
        aln_block_len: int,
        mapq: int,
    ) -> Optional[OverlapCandidate]:
        """
        Convert PAF query/target geometry to left->right suffix-prefix direction.

        Case A:
            target suffix aligns query prefix: target -> query
        Case B:
            query suffix aligns target prefix: query -> target
        """
        target_orientation = +1 if strand == "+" else -1
        # Convert target coordinates into the orientation that aligns to the
        # query.  For a reverse PAF strand, the target interval is mirrored.
        oriented_t_st = t_st if target_orientation == +1 else t_len - t_en
        oriented_t_en = t_en if target_orientation == +1 else t_len - t_st

        target_to_query_shift = oriented_t_st - q_st
        query_to_target_shift = q_st - oriented_t_st

        # When both reads are nearly full-length aligned, both endpoint tests
        # can pass within the overhang tolerance.  Prefer the direction that
        # advances the layout and reject zero/negative-shift duplicate edges.
        # target suffix -> query prefix
        if (
            self._near_end(oriented_t_en, t_len)
            and self._near_start(q_st)
            and target_to_query_shift > 0
        ):
            return OverlapCandidate(
                left_id=t_id,
                right_id=q_id,
                source="minimap2",
                query_id=q_id,
                target_id=t_id,
                strand=strand,
                q_len=q_len,
                q_st=q_st,
                q_en=q_en,
                t_len=t_len,
                t_st=t_st,
                t_en=t_en,
                n_match=n_match,
                aln_block_len=aln_block_len,
                mapq=mapq,
                left_start_hint=max(0, oriented_t_st),
                left_end_hint=t_len,
                right_start_hint=0,
                right_end_hint=min(q_len, q_en),
                rough_overlap_len=aln_block_len,
                rough_shift=oriented_t_st - q_st,
                left_orientation=target_orientation,
                right_orientation=+1,
            )

        # query suffix -> target prefix
        if (
            self._near_end(q_en, q_len)
            and self._near_start(oriented_t_st)
            and query_to_target_shift > 0
        ):
            return OverlapCandidate(
                left_id=q_id,
                right_id=t_id,
                source="minimap2",
                query_id=q_id,
                target_id=t_id,
                strand=strand,
                q_len=q_len,
                q_st=q_st,
                q_en=q_en,
                t_len=t_len,
                t_st=t_st,
                t_en=t_en,
                n_match=n_match,
                aln_block_len=aln_block_len,
                mapq=mapq,
                left_start_hint=max(0, q_st),
                left_end_hint=q_len,
                right_start_hint=0,
                right_end_hint=min(t_len, oriented_t_en),
                rough_overlap_len=aln_block_len,
                rough_shift=q_st - oriented_t_st,
                left_orientation=+1,
                right_orientation=target_orientation,
            )

        return None

    @staticmethod
    def _reverse_complement_candidate(candidate: OverlapCandidate) -> OverlapCandidate:
        """Return the reciprocal edge on the reverse-complement strand."""
        left_len = candidate.q_len if candidate.left_id == candidate.query_id else candidate.t_len
        right_len = candidate.q_len if candidate.right_id == candidate.query_id else candidate.t_len
        left_start = right_len - candidate.right_end_hint
        left_end = right_len - candidate.right_start_hint
        right_start = left_len - candidate.left_end_hint
        right_end = left_len - candidate.left_start_hint
        return OverlapCandidate(
            left_id=candidate.right_id,
            right_id=candidate.left_id,
            source=candidate.source,
            query_id=candidate.target_id,
            target_id=candidate.query_id,
            strand=candidate.strand,
            q_len=candidate.t_len,
            q_st=candidate.t_len - candidate.t_en,
            q_en=candidate.t_len - candidate.t_st,
            t_len=candidate.q_len,
            t_st=candidate.q_len - candidate.q_en,
            t_en=candidate.q_len - candidate.q_st,
            n_match=candidate.n_match,
            aln_block_len=candidate.aln_block_len,
            mapq=candidate.mapq,
            # The reverse candidate's left read is the original right read,
            # and its right read is the original left read.  Clip each hint
            # against the corresponding reverse-candidate read length.
            left_start_hint=max(0, left_start),
            left_end_hint=min(right_len, left_end),
            right_start_hint=max(0, right_start),
            right_end_hint=min(left_len, right_end),
            rough_overlap_len=candidate.rough_overlap_len,
            rough_shift=left_start - right_start,
            left_orientation=-candidate.right_orientation,
            right_orientation=-candidate.left_orientation,
        )


class OriginalCandidateFinder(OverlapCandidateFinder):
    """
    Placeholder for the user's original all-vs-all overlap candidate algorithm.

    Implement this class later so it returns the same OverlapCandidate objects as
    Minimap2CandidateFinder. Then the rest of the pipeline can compare minimap2
    and the original method under identical refinement/evaluation conditions.
    """

    def find_candidates(self, reads: list[Read]) -> list[OverlapCandidate]:
        raise NotImplementedError(
            "OriginalCandidateFinder is a reserved interface. "
            "Implement the custom candidate algorithm here."
        )
