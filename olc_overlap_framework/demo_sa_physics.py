"""
Standalone simulated-annealing sanity check on a simple physical system.

System:
    1D ferromagnetic Ising chain with open boundary conditions.

Hamiltonian:
    H = -J * sum_i s_i s_{i+1}, where s_i in {-1, +1}

The exact ground states are all spins aligned:
    000...0 or 111...1 in binary variables x_i = (s_i + 1) / 2

This demo converts the Ising chain to a QUBO and solves it with the same
annealer interface used by the layout QUBO code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from olc_pipeline.layout_solver import (  # noqa: E402
    BinaryAnnealingConfig,
    BinarySimulatedAnnealer,
    DWaveAnnealingConfig,
    DWaveSimulatedAnnealer,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOModel,
)


def build_ferromagnetic_ising_chain(num_spins: int, coupling: float) -> QUBOModel:
    """
    Build QUBO for H = -J * sum s_i s_{i+1}.

    With s_i = 2*x_i - 1:
        -J*s_i*s_j = -4J*x_i*x_j + 2J*x_i + 2J*x_j - J
    """
    model = QUBOModel(read_ids=[f"s{i}" for i in range(num_spins)], linear=[0.0] * num_spins)
    for left in range(num_spins - 1):
        right = left + 1
        model.add_constant(-coupling)
        model.add_linear(left, 2.0 * coupling)
        model.add_linear(right, 2.0 * coupling)
        model.add_quadratic(left, right, -4.0 * coupling)
    return model


def exact_ground_energy(num_spins: int, coupling: float) -> float:
    return -coupling * (num_spins - 1)


def spin_string(sample: list[int]) -> str:
    return "".join("+" if bit else "-" for bit in sample)


def count_domain_walls(sample: list[int]) -> int:
    return sum(1 for left, right in zip(sample, sample[1:]) if left != right)


def solve_with_backend(args: argparse.Namespace, model: QUBOModel):
    if args.backend == "dwave-sa":
        annealer = DWaveSimulatedAnnealer(DWaveAnnealingConfig(
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        ))
        return annealer.solve(model)
    if args.backend == "openjij-sqa":
        annealer = OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        ))
        return annealer.solve(model)

    annealer = BinarySimulatedAnnealer(BinaryAnnealingConfig(
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        cooling_rate=args.cooling_rate,
        sweeps_per_temperature=args.sweeps_per_temperature,
        seed=args.seed,
        random_restarts=args.num_reads,
        start_from_valid_permutation=False,
        swap_move_probability=0.0,
    ))
    return annealer.solve(model)


def backend_names(args: argparse.Namespace) -> list[str]:
    if args.backend == "all":
        return ["builtin", "dwave-sa", "openjij-sqa"]
    return [args.backend]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test SA on a 1D ferromagnetic Ising chain converted to QUBO."
    )
    parser.add_argument("--spins", type=int, default=16)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--backend", choices=("all", "dwave-sa", "openjij-sqa", "builtin"), default="all")
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Built-in backend parameters.
    parser.add_argument("--initial-temperature", type=float, default=5.0)
    parser.add_argument("--final-temperature", type=float, default=0.01)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--sweeps-per-temperature", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.spins <= 1:
        raise ValueError("--spins must be greater than 1")
    if args.coupling <= 0.0:
        raise ValueError("--coupling must be positive for this ferromagnetic test")

    model = build_ferromagnetic_ising_chain(args.spins, args.coupling)
    target_energy = exact_ground_energy(args.spins, args.coupling)

    print("== SA Physics Sanity Check ==")
    print(f"System                    1D ferromagnetic Ising chain")
    print(f"Spins                     {args.spins}")
    print(f"Coupling J                {args.coupling}")
    print(f"Reads / sweeps            {args.num_reads} / {args.num_sweeps}")
    print(f"Exact ground energy       {target_energy:.6f}")
    print()

    failures = 0
    for backend in backend_names(args):
        backend_args = argparse.Namespace(**vars(args))
        backend_args.backend = backend
        print(f"-- Backend: {backend} --")
        try:
            result = solve_with_backend(backend_args, model)
        except RuntimeError as exc:
            print(f"Status                    skipped")
            print(f"Reason                    {exc}")
            print()
            continue

        domain_walls = count_domain_walls(result.sample)
        reached_ground = abs(result.energy - target_energy) < 1e-9
        print(f"Backend result            {result.backend}")
        print(f"Found energy              {result.energy:.6f}")
        print(f"Reached ground state      {reached_ground}")
        print(f"Domain walls              {domain_walls}")
        print(f"Spin configuration        {spin_string(result.sample)}")
        print()
        if not reached_ground:
            failures += 1

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
