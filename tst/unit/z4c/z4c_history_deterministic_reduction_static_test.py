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

    min_lapse_label = 'pdata->label[min_lapse_index] = "minLapse";'
    min_lapse_begin = source.index('"Z4cHistoryMinLapse"')
    min_lapse_end = source.index("if (opt.telegraph_lapse)", min_lapse_begin)
    min_lapse_block = source[min_lapse_begin:min_lapse_end]
    for token in (
        min_lapse_label,
        "pdata->reduction[min_lapse_index] = HistoryData::Reduction::min;",
    ):
        if token not in source:
            raise SystemExit(f"missing slice-minimum lapse history token: {token}")
    for token in (
        "canonical_diagnostic_owner",
        "z4c.alpha(m, k, j, i)",
        "Kokkos::Min<Real>(min_lapse)",
        "pdata->hdata[min_lapse_index] = min_lapse;",
    ):
        if token not in min_lapse_block:
            raise SystemExit(f"missing slice-minimum lapse history token: {token}")

    print("PASS: Cartoon history reductions include deterministic slice min lapse")


if __name__ == "__main__":
    main()
