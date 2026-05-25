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

from .data import Read, OverlapCandidate
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

            error_rate_hint = self._error_rate_hint(n_match, aln_block_len, tags)
            if error_rate_hint > self.config.max_error_rate_hint:
                counts["skipped_error_hint"] += 1
                continue

            # Version 0.1.0 only handles same-strand directed overlaps.
            # Reverse-complement handling is left for a later module update.
            if strand != "+":
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
            else:
                counts["skipped_geometry"] += 1
        self.last_filter_counts = counts

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
        # target suffix -> query prefix
        if self._near_end(t_en, t_len) and self._near_start(q_st):
            rough_shift = t_st - q_st
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
                left_start_hint=max(0, t_st),
                left_end_hint=t_len,
                right_start_hint=0,
                right_end_hint=min(q_len, q_en),
                rough_overlap_len=aln_block_len,
                rough_shift=rough_shift,
            )

        # query suffix -> target prefix
        if self._near_end(q_en, q_len) and self._near_start(t_st):
            rough_shift = q_st - t_st
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
                right_end_hint=min(t_len, t_en),
                rough_overlap_len=aln_block_len,
                rough_shift=rough_shift,
            )

        return None


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
