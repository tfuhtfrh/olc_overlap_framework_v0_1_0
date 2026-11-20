"""Small runtime utilities shared by the canonical total program."""
from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

__all__=["sha256_file"]
