#!/usr/bin/env python3
"""Aggregate the matched warmed Ref-GH/Z4c performance discriminator."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def history_time(path: Path) -> tuple[int, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cycle_matches = re.findall(r"^time=.* cycle=(\d+)$", text, re.MULTILINE)
    cpu_matches = re.findall(r"^cpu time used\s*=\s*([0-9.eE+-]+)$", text,
                             re.MULTILINE)
    if not cycle_matches or not cpu_matches:
        raise RuntimeError(f"missing cycle/cpu time in {path}")
    return int(cycle_matches[-1]), float(cpu_matches[-1])


def kernel_table(path: Path) -> dict[str, tuple[int, float]]:
    result: dict[str, tuple[int, float]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result[row["kernel"]] = (int(row["calls"]),
                                     float(row["total_seconds"]))
    return result


def average(table: dict[str, tuple[int, float]], name: str) -> float:
    calls, seconds = table.get(name, (0, 0.0))
    return 0.0 if calls == 0 else seconds / calls


def sum_per_stage(table: dict[str, tuple[int, float]], names: tuple[str, ...],
                  stage_calls: int) -> float:
    return sum(table.get(name, (0, 0.0))[1] for name in names) / stage_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run

    cases = ("refgh_dynamic", "refgh_static", "z4c")
    complete: dict[str, float] = {}
    method: dict[str, int] = {}
    for case in cases:
        warm_cycles, warm_cpu = history_time(run / f"{case}_warmup" / "run.log")
        measured_cycles, measured_cpu = history_time(run / case / "run.log")
        cycle_delta = measured_cycles - warm_cycles
        if cycle_delta <= 0:
            raise RuntimeError(f"nonpositive warmed cycle interval for {case}")
        stages = 4 * cycle_delta
        complete[case] = (measured_cpu - warm_cpu) / stages
        method[f"{case}_warmup_cycles"] = warm_cycles
        method[f"{case}_measured_cycles"] = measured_cycles

    dynamic = kernel_table(run / "refgh_dynamic_kernels.tsv")
    z4c = kernel_table(run / "z4c_kernels.tsv")
    profile_stages = dynamic["ref_gh RK update"][0]
    z4c_profile_stages = z4c["z4c RK update"][0]

    psi = average(dynamic, "ref_gh psi rhs")
    scalar_pi = average(dynamic, "ref_gh scalar source and pi rhs")
    compatible_phi = average(dynamic, "ref_gh compatible phi rhs")
    standard_phi = average(dynamic, "ref_gh standard phi rhs")
    gamma2 = average(dynamic, "ref_gh gamma2 reduction damping")
    main_rhs = psi + scalar_pi + compatible_phi + standard_phi + gamma2
    dissipation = average(dynamic, "ref_gh dissipation")
    reference = average(dynamic, "ref_gh analytic radial-q stage reference")
    q_control = average(dynamic, "ref_gh compact current-q reduction")
    boundary = average(dynamic, "ref_gh projected trumpet metric boundaries")
    rk_update = average(dynamic, "ref_gh RK update")
    communication = sum_per_stage(dynamic, ("SendBuff", "RecvBuff"),
                                  profile_stages + 1)
    z4c_rhs = average(z4c, "z4c rhs loop")
    z4c_dissipation = average(z4c, "K-O Dissipation")

    complete_dynamic = complete["refgh_dynamic"]
    seconds = {
        "complete_warmed_dynamic_stage": complete_dynamic,
        "complete_warmed_static_stage": complete["refgh_static"],
        "complete_warmed_z4c_stage": complete["z4c"],
        "analytic_reference": reference,
        "q_control": q_control,
        "main_rhs_without_dissipation": main_rhs,
        "dissipation": dissipation,
        "physical_boundary": boundary,
        "rk_update": rk_update,
        "communication_device_pack_unpack": communication,
        "z4c_rhs_without_dissipation": z4c_rhs,
        "z4c_dissipation": z4c_dissipation,
    }
    ratios = {
        "dynamic_q_over_static_complete_time":
            complete_dynamic / complete["refgh_static"],
        "refgh_over_z4c_complete_time":
            complete_dynamic / complete["z4c"],
        "refgh_over_z4c_rhs_without_dissipation": main_rhs / z4c_rhs,
        "refgh_over_z4c_rhs_with_dissipation":
            (main_rhs + dissipation) / (z4c_rhs + z4c_dissipation),
    }
    fractions = {
        name: value / complete_dynamic
        for name, value in seconds.items()
        if name not in ("complete_warmed_static_stage",
                        "complete_warmed_z4c_stage",
                        "z4c_rhs_without_dissipation", "z4c_dissipation")
    }
    result = {
        "method": {
            **method,
            "rk_stages_per_cycle": 4,
            "profile_refgh_stages": profile_stages,
            "profile_z4c_stages": z4c_profile_stages,
            "complete_stage_method":
                "(measured cpu - warmup cpu) / (cycle delta * 4)",
            "profile_method": "separate globally fenced Kokkos kernel profile",
        },
        "seconds_per_stage": seconds,
        "fractions_of_complete_dynamic_stage": fractions,
        "ratios": ratios,
        "targets": {
            "q_control_lt_0.02_stage": fractions["q_control"] < 0.02,
            "reference_lt_0.10_stage": fractions["analytic_reference"] < 0.10,
            "dynamic_q_lt_1.10_static":
                ratios["dynamic_q_over_static_complete_time"] < 1.10,
            "refgh_rhs_le_2_z4c_rhs":
                ratios["refgh_over_z4c_rhs_without_dissipation"] <= 2.0,
        },
        "limitations": [
            "One MPI rank on one PVC tile; no inter-rank communication is present.",
            "Synchronized kernel profiles are separate from unfenced throughput.",
            "Device SendBuff/RecvBuff timings omit host and MPI overhead.",
            "Z4c is a matched grid/hardware control, not equal arithmetic work.",
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
