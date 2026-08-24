#!/usr/bin/env python3
"""Guard deterministic Cartoon constraint-region history reductions."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()

    source = (args.source_root / "src/outputs/history.cpp").read_text()
    label = '"Cartoon constraint axis off-axis and layer sums"'
    label_position = source.index(label)
    begin = source.rfind("Kokkos::parallel_reduce(", 0, label_position)
    end = source.index(
        "const std::array<ConstraintMaximum, kCartoonConstraintFamilies>", begin
    )
    block = source[begin:end]

    required = (
        "Kokkos::parallel_reduce(",
        "cartoon_history_sum::DiagnosticSum &thread_sums",
        "Kokkos::Sum<cartoon_history_sum::DiagnosticSum>(diagnostic_sums)",
    )
    for token in required:
        if token not in block:
            raise SystemExit(f"missing deterministic history reduction token: {token}")
    if "Kokkos::atomic_add" in block:
        raise SystemExit("Cartoon constraint-region history still uses unordered atomics")

    print("PASS: Cartoon constraint-region history uses a dedicated reduction")


if __name__ == "__main__":
    main()
