"""
olc_pipeline.io_utils
Version: 0.1.0

Small file I/O helpers shared by candidate finders and demos.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from .data import Read

MODULE_VERSION = "0.1.0"


def write_reads_fasta(reads: list[Read], path: Path, line_width: int = 80) -> None:
    """Write reads to FASTA for minimap2 or other external tools."""
    with open(path, "w", encoding="utf-8") as handle:
        for read in reads:
            handle.write(f">{read.rid}\n")
            for start in range(0, len(read.seq), line_width):
                handle.write(read.seq[start:start + line_width] + "\n")


def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def read_fastq(path: Path, max_reads: int | None = None) -> list[Read]:
    """Read FASTQ/FASTQ.GZ records as physical :class:`Read` objects.

    Quality strings are intentionally ignored because the current OLC modules
    operate on read sequence and overlap evidence only.  No reference or
    precomputed graph information is consulted here.
    """
    opener = gzip.open if Path(path).suffix == ".gz" else open
    reads: list[Read] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            sequence = handle.readline().rstrip("\r\n")
            plus = handle.readline()
            quality = handle.readline().rstrip("\r\n")
            if not plus or not plus.startswith("+") or len(sequence) != len(quality):
                raise ValueError(f"Malformed FASTQ record in {path}")
            if not header.startswith("@"):
                raise ValueError(f"FASTQ header does not start with @ in {path}")
            reads.append(Read(rid=header[1:].strip().split()[0], seq=sequence))
            if max_reads is not None and len(reads) >= max_reads:
                break
    return reads
