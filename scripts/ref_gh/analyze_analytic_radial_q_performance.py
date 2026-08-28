#!/usr/bin/env python3
"""Aggregate the matched Aurora Ref-GH/Z4c timing checkpoint."""

import argparse
import json
import re
from pathlib import Path


CPU_RE = re.compile(r"cpu time used\s*=\s*([0-9.eE+-]+)")


def cpu_seconds(root, case):
    match = CPU_RE.search((root / case / "run.log").read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"missing cpu time in {case}")
    return float(match.group(1))


def kernels(path):
    result = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "kernel\tcalls\ttotal_seconds\taverage_seconds\tpercent":
        raise RuntimeError(f"unexpected kernel profile header in {path}")
    for line in lines[1:]:
        name, calls, total, _, _ = line.split("\t")
        result[name] = (int(calls), float(total))
    return result


def total(profile, names):
    return sum(profile[name][1] for name in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory

    stages = (100 - 20) * 4
    dynamic_stage = (
        cpu_seconds(root, "refgh_dynamic")
        - cpu_seconds(root, "refgh_dynamic_warmup")
    ) / stages
    static_stage = (
        cpu_seconds(root, "refgh_static")
        - cpu_seconds(root, "refgh_static_warmup")
    ) / stages
    z4c_stage = (
        cpu_seconds(root, "z4c") - cpu_seconds(root, "z4c_warmup")
    ) / stages

    dynamic = kernels(root / "refgh_dynamic_kernels.tsv")
    z4c = kernels(root / "z4c_kernels.tsv")
    dynamic_profile_stages = dynamic["ref_gh RK update"][0]
    z4c_profile_stages = z4c["z4c RK update"][0]

    rhs_names = [
        "ref_gh psi rhs",
        "ref_gh scalar source and pi rhs",
        "ref_gh compatible phi rhs",
    ]
    refgh_rhs = total(dynamic, rhs_names) / dynamic["ref_gh psi rhs"][0]
    refgh_diss = dynamic["ref_gh dissipation"][1] / dynamic["ref_gh dissipation"][0]
    z4c_rhs = z4c["z4c rhs loop"][1] / z4c["z4c rhs loop"][0]
    z4c_diss = z4c["K-O Dissipation"][1] / z4c["K-O Dissipation"][0]

    categories = {
        "q_control": dynamic["ref_gh compact current-q reduction"][1]
        / dynamic["ref_gh compact current-q reduction"][0],
        "analytic_reference": dynamic["ref_gh analytic radial-q stage reference"][1]
        / dynamic["ref_gh analytic radial-q stage reference"][0],
        "main_rhs_without_dissipation": refgh_rhs,
        "dissipation": refgh_diss,
        "rk_update": dynamic["ref_gh RK update"][1] / dynamic_profile_stages,
        "communication_device_pack_unpack": total(dynamic, ["SendBuff", "RecvBuff"])
        / dynamic_profile_stages,
        "physical_boundary": dynamic[
            "ref_gh projected trumpet metric boundaries"
        ][1]
        / dynamic["ref_gh projected trumpet metric boundaries"][0],
        "complete_warmed_stage": dynamic_stage,
    }
    result = {
        "method": {
            "active_cells": 64**3,
            "rk_stages_per_cycle": 4,
            "warmup_cycles": 20,
            "measured_cycles": 100,
            "profile_cycles": dynamic_profile_stages // 4,
            "complete_stage_method": "(100-cycle cpu - 20-cycle cpu) / 320 stages",
            "profile_method": "separate globally fenced Kokkos kernel profile",
        },
        "seconds_per_stage": categories,
        "fractions_of_complete_dynamic_stage": {
            name: value / dynamic_stage for name, value in categories.items()
        },
        "ratios": {
            "dynamic_q_over_static_complete_time": dynamic_stage / static_stage,
            "refgh_over_z4c_complete_time": dynamic_stage / z4c_stage,
            "refgh_over_z4c_rhs_without_dissipation": refgh_rhs / z4c_rhs,
            "refgh_over_z4c_rhs_with_dissipation": (refgh_rhs + refgh_diss)
            / (z4c_rhs + z4c_diss),
        },
        "targets": {
            "q_control_lt_0.02_stage": categories["q_control"] < 0.02 * dynamic_stage,
            "reference_lt_0.10_stage": categories["analytic_reference"]
            < 0.10 * dynamic_stage,
            "dynamic_q_lt_1.10_static": dynamic_stage < 1.10 * static_stage,
            "refgh_rhs_le_2_z4c_rhs": refgh_rhs <= 2.0 * z4c_rhs,
        },
        "limitations": [
            "One MPI rank on one PVC tile; no inter-rank communication is present.",
            "Communication timing is device SendBuff/RecvBuff only; host overhead is in the complete-stage residual.",
            "The synchronized kernel profiles are separate from the unfenced throughput measurements.",
            "Z4c is a matched grid/hardware execution control, not equal arithmetic work.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
