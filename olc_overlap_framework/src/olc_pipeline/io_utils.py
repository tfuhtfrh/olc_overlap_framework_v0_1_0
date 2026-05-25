"""
olc_pipeline.io_utils
Version: 0.1.0

Small file I/O helpers shared by candidate finders and demos.
"""

from __future__ import annotations

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
