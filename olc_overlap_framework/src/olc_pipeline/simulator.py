"""
olc_pipeline.simulator
Version: 0.1.0

Random read simulator for controlled OLC overlap experiments.
Generates a linear reference genome, samples reads with long overlaps, injects
mismatch/insertion/deletion errors, and shuffles the input order. Ground-truth
coordinates are retained in Read objects for later evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

from .data import Read

MODULE_VERSION = "0.1.0"
DNA = "ACGT"


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for random read generation."""

    genome_len: int = 20_000
    read_len: int = 2_000
    step: int = 800
    read_len_min: Optional[int] = None
    read_len_max: Optional[int] = None
    overlap_fraction_min: Optional[float] = None
    overlap_fraction_max: Optional[float] = None
    adjacent_overlap_min: Optional[int] = None
    mismatch_rate: float = 0.02
    ins_rate: float = 0.01
    del_rate: float = 0.01
    seed: int = 42
    shuffle_reads: bool = True

    def uses_random_layout(self) -> bool:
        return any(
            value is not None
            for value in (
                self.read_len_min,
                self.read_len_max,
                self.overlap_fraction_min,
                self.overlap_fraction_max,
                self.adjacent_overlap_min,
            )
        )

    def validate(self) -> None:
        if self.genome_len <= 0:
            raise ValueError("genome_len must be positive")
        if self.read_len <= 0:
            raise ValueError("read_len must be positive")
        if not self.uses_random_layout() and self.read_len > self.genome_len:
            raise ValueError("read_len must not exceed genome_len")
        if self.step <= 0:
            raise ValueError("step must be positive")
        for name in ("mismatch_rate", "ins_rate", "del_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if self.uses_random_layout():
            missing = [
                name
                for name in (
                    "read_len_min",
                    "read_len_max",
                    "overlap_fraction_min",
                    "overlap_fraction_max",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "random read layout requires all range fields: "
                    + ", ".join(missing)
                )

            assert self.read_len_min is not None
            assert self.read_len_max is not None
            assert self.overlap_fraction_min is not None
            assert self.overlap_fraction_max is not None

            if self.read_len_min <= 1:
                raise ValueError("read_len_min must be greater than 1")
            if self.read_len_max < self.read_len_min:
                raise ValueError("read_len_max must be >= read_len_min")
            if self.read_len_max > self.genome_len:
                raise ValueError("read_len_max must not exceed genome_len")
            if not 0.0 < self.overlap_fraction_min <= 1.0:
                raise ValueError("overlap_fraction_min must be in (0, 1]")
            if not 0.0 < self.overlap_fraction_max <= 1.0:
                raise ValueError("overlap_fraction_max must be in (0, 1]")
            if self.overlap_fraction_max < self.overlap_fraction_min:
                raise ValueError("overlap_fraction_max must be >= overlap_fraction_min")
            if self.adjacent_overlap_min is not None:
                if self.adjacent_overlap_min <= 0:
                    raise ValueError("adjacent_overlap_min must be positive")
                max_possible_for_shortest_read = round(self.read_len_min * self.overlap_fraction_max)
                if self.adjacent_overlap_min > max_possible_for_shortest_read:
                    raise ValueError(
                        "adjacent_overlap_min cannot be satisfied by the shortest read "
                        "and overlap_fraction_max"
                    )


class RandomReadSimulator:
    """
    Generate simulated reads and preserve their true genome coordinates.

    Public interface:
        simulate(config) -> tuple[genome, reads]
    """

    def simulate(self, config: SimulationConfig) -> tuple[str, list[Read]]:
        config.validate()
        rng = random.Random(config.seed)
        genome = self._random_dna(config.genome_len, rng)

        if config.uses_random_layout():
            reads = self._simulate_random_layout(genome, config, rng)
        else:
            reads = self._simulate_fixed_layout(genome, config, rng)

        if config.shuffle_reads:
            rng.shuffle(reads)

        return genome, reads

    def _simulate_fixed_layout(
        self,
        genome: str,
        config: SimulationConfig,
        rng: random.Random,
    ) -> list[Read]:
        reads: list[Read] = []
        for idx, start in enumerate(range(0, config.genome_len - config.read_len + 1, config.step)):
            raw = genome[start:start + config.read_len]
            noisy, ref_coords = self._mutate_read(
                raw,
                rng,
                ref_start=start,
                mismatch_rate=config.mismatch_rate,
                ins_rate=config.ins_rate,
                del_rate=config.del_rate,
            )
            reads.append(Read(
                rid=f"read_{idx}",
                seq=noisy,
                true_start=start,
                true_end=start + config.read_len,
                strand=+1,
                ref_coords=ref_coords,
            ))

        return reads

    def _simulate_random_layout(
        self,
        genome: str,
        config: SimulationConfig,
        rng: random.Random,
    ) -> list[Read]:
        assert config.read_len_min is not None
        assert config.read_len_max is not None
        assert config.overlap_fraction_min is not None
        assert config.overlap_fraction_max is not None

        reads: list[Read] = []
        start = 0
        read_len = rng.randint(config.read_len_min, config.read_len_max)
        idx = 0

        while start + read_len <= config.genome_len:
            raw = genome[start:start + read_len]
            noisy, ref_coords = self._mutate_read(
                raw,
                rng,
                ref_start=start,
                mismatch_rate=config.mismatch_rate,
                ins_rate=config.ins_rate,
                del_rate=config.del_rate,
            )
            reads.append(Read(
                rid=f"read_{idx}",
                seq=noisy,
                true_start=start,
                true_end=start + read_len,
                strand=+1,
                ref_coords=ref_coords,
            ))

            next_start, next_read_len = self._next_random_layout_position(
                current_start=start,
                current_read_len=read_len,
                genome_len=config.genome_len,
                read_len_min=config.read_len_min,
                read_len_max=config.read_len_max,
                overlap_fraction_min=config.overlap_fraction_min,
                overlap_fraction_max=config.overlap_fraction_max,
                adjacent_overlap_min=config.adjacent_overlap_min,
                rng=rng,
            )
            if next_start is None or next_read_len is None:
                break

            start = next_start
            read_len = next_read_len
            idx += 1

        return reads

    @staticmethod
    def _next_random_layout_position(
        current_start: int,
        current_read_len: int,
        genome_len: int,
        read_len_min: int,
        read_len_max: int,
        overlap_fraction_min: float,
        overlap_fraction_max: float,
        adjacent_overlap_min: Optional[int],
        rng: random.Random,
    ) -> tuple[Optional[int], Optional[int]]:
        for _ in range(1000):
            next_read_len = rng.randint(read_len_min, read_len_max)
            shorter_len = min(current_read_len, next_read_len)
            min_overlap_len = max(1, round(shorter_len * overlap_fraction_min))
            if adjacent_overlap_min is not None:
                min_overlap_len = max(min_overlap_len, adjacent_overlap_min)
            max_overlap_len = min(round(shorter_len * overlap_fraction_max), shorter_len - 1)
            if min_overlap_len > max_overlap_len:
                continue

            overlap_len = rng.randint(min_overlap_len, max_overlap_len)
            next_start = current_start + current_read_len - overlap_len

            if next_start <= current_start:
                continue
            if next_start + next_read_len <= genome_len:
                return next_start, next_read_len

        return None, None

    @staticmethod
    def true_order(reads: list[Read]) -> list[str]:
        """Return read ids sorted by ground-truth coordinate."""
        return [r.rid for r in sorted(reads, key=lambda r: r.true_start)]

    @staticmethod
    def _random_dna(length: int, rng: random.Random) -> str:
        return "".join(rng.choice(DNA) for _ in range(length))

    @staticmethod
    def _mutate_read(
        seq: str,
        rng: random.Random,
        ref_start: int,
        mismatch_rate: float,
        ins_rate: float,
        del_rate: float,
    ) -> tuple[str, tuple[int, ...]]:
        """
        Apply a simple independent mismatch/insertion/deletion error model.

        Returns the noisy read and, for each emitted base, the reference
        coordinate it came from. Inserted bases use -1.
        """
        out: list[str] = []
        ref_coords: list[int] = []
        for offset, ch in enumerate(seq):
            if rng.random() < ins_rate:
                out.append(rng.choice(DNA))
                ref_coords.append(-1)

            if rng.random() < del_rate:
                continue

            if rng.random() < mismatch_rate:
                out.append(rng.choice([base for base in DNA if base != ch]))
            else:
                out.append(ch)
            ref_coords.append(ref_start + offset)

        if rng.random() < ins_rate:
            out.append(rng.choice(DNA))
            ref_coords.append(-1)
        return "".join(out), tuple(ref_coords)
