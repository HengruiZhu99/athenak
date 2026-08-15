#!/usr/bin/env python3
"""Summarize the enlarged-domain level-20 controls without mutation."""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
CHI_REJECTION = re.compile(
    r"Z4c chi .*? rejected (\d+) parent stencils and (\d+) limited sibling groups"
)
TIME_CYCLE = re.compile(
    r"(?:time|t)=\s*([+\-0-9.eE]+).*?(?:cycle|ncycle)=\s*(\d+)", re.I
)
SELECTED = (
    "time",
    "dt",
    "C-norm2",
    "H-norm2",
    "M-norm2",
    "Z-norm2",
    "max_abs_K",
    "nmb_total",
    "maxRefLev",
    "cycle",
    "axisLapse",
    "axisTau",
    "axisKret",
    "maxAbsKret",
)


def read_history(path: Path) -> tuple[dict[str, int], list[list[float]]]:
    labels: dict[str, int] = {}
    rows: list[list[float]] = []
    if not path.exists():
        return labels, rows
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#"):
            for index, name in HEADER.findall(line):
                labels[name] = int(index) - 1
            continue
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            rows.append(values)
    return labels, rows


def row_dict(labels: dict[str, int], row: list[float]) -> dict[str, float]:
    return {name: row[labels[name]] for name in SELECTED if name in labels}


def find_history(case_dir: Path) -> Path | None:
    paths = sorted(case_dir.glob("*.z4c.user.hst"))
    return paths[0] if len(paths) == 1 else None


def summarize_case(
    case_dir: Path,
    name: str,
    tau: float,
    kappa: float,
    dissipation: float,
    shift_condition: str,
) -> dict:
    exit_path = case_dir / "exit-status.txt"
    exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
    log_path = case_dir / "run.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    history_path = find_history(case_dir)
    labels, rows = read_history(history_path) if history_path else ({}, [])
    last = row_dict(labels, rows[-1]) if rows else None
    max_level = (
        int(max(row[labels["maxRefLev"]] for row in rows))
        if rows and "maxRefLev" in labels
        else None
    )
    max_s_mu = (
        max(row[labels["dt"]] * row[labels["max_abs_K"]] / tau for row in rows)
        if rows and {"dt", "max_abs_K"} <= labels.keys()
        else None
    )
    max_s_beta = (
        max(2.0 * row[labels["dt"]] for row in rows)
        if rows and "dt" in labels
        else None
    )
    chi_matches = [tuple(map(int, match)) for match in CHI_REJECTION.findall(log)]
    fatals = [
        line.strip()
        for line in log.splitlines()
        if "FATAL" in line or "Kokkos ERROR" in line or "TIME LIMIT" in line
    ]
    terminal_candidates = TIME_CYCLE.findall(log)
    bindings = []
    for path in sorted((case_dir / "bindings").glob("rank_binding_*.json")):
        bindings.append(json.loads(path.read_text()))
    return {
        "name": name,
        "tau": tau,
        "kappa": kappa,
        "kappa_over_tau": kappa / tau,
        "ko_dissipation": dissipation,
        "shift_condition": shift_condition,
        "constraint_damping": False,
        "damp_kappa1": 0.0,
        "damp_kappa2": 0.0,
        "target_kappa1": 0.0,
        "damp_kappa1_max_K": False,
        "exit_code": exit_code,
        "history_path": str(history_path) if history_path else None,
        "history_rows": len(rows),
        "last_history": last,
        "max_refinement_level_reached": max_level,
        "max_dt_mu_over_tau": max_s_mu,
        "max_dt_eta_shift": max_s_beta,
        "chi_rejections": [
            {"invalid_parent_stencils": parent, "invalid_limited_groups": limited}
            for parent, limited in chi_matches
        ],
        "fatal_lines": fatals[-8:],
        "last_log_time_cycle": (
            {"time": float(terminal_candidates[-1][0]), "cycle": int(terminal_candidates[-1][1])}
            if terminal_candidates
            else None
        ),
        "rank_bindings": bindings,
        "reached_target_t20": bool(last and last.get("time", -math.inf) >= 20.0 - 1e-10),
        "qualification_claim": False,
    }


def write_case(args: argparse.Namespace) -> None:
    result = summarize_case(
        args.case_dir,
        args.name,
        args.tau,
        args.kappa,
        args.dissipation,
        args.shift_condition,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def comparison_result(cases: list[dict]) -> dict:
    assert [case["name"] for case in cases] == [
        "fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64",
        "zero_shift_tau1_kappa1_l20_nocd_ko05_domain64",
    ]
    assert [(case["tau"], case["kappa"]) for case in cases] == [
        (1.0, 1.0),
        (1.0, 1.0),
    ]
    assert [case["ko_dissipation"] for case in cases] == [0.5, 0.5]
    assert [case["shift_condition"] for case in cases] == [
        "fixed_gamma_driver_eta2",
        "zero_shift",
    ]
    for case in cases:
        assert case["constraint_damping"] is False
        assert case["damp_kappa1"] == case["damp_kappa2"] == 0.0
        assert case["target_kappa1"] == 0.0
        assert case["damp_kappa1_max_K"] is False
    all_bindings = all(
        len(case["rank_bindings"]) == 1
        and all(
            binding.get("binding_verified") is True
            and binding.get("gpu_name") == "NVIDIA A100-SXM4-80GB"
            for binding in case["rank_bindings"]
        )
        for case in cases
    )
    return {
        "schema": "athenak_brill_domain64_l20_tau1_nocd_ko05_shift_pair_cuda1_v1",
        "source_commit": "2a8ad80e02279769a99fe279b7a33516bc6c8d0d",
        "exact_common_changes": {
            "mesh_refinement/num_levels": 21,
            "mesh/nx1": 256,
            "mesh/nx2": 512,
            "mesh/x1min": 0.0,
            "mesh/x1max": 64.0,
            "mesh/x2min": -64.0,
            "mesh/x2max": 64.0,
            "mesh_refinement/max_nmb_per_rank": 16384,
            "time/nmb_total_limit": 16384,
            "z4c/shift_eta_max_K": False,
            "z4c_amr/max_ref_lev": 20,
        },
        "domain_difference_from_predecessor": {
            "mesh/nx1": [64, 256],
            "mesh/nx2": [128, 512],
            "mesh/x1max": [16.0, 64.0],
            "mesh/x2min": [-16.0, -64.0],
            "mesh/x2max": [16.0, 64.0],
            "base_dx": [0.25, 0.25],
            "mesh_refinement/max_nmb_per_rank": [2048, 16384],
            "time/nmb_total_limit": [8192, 16384],
        },
        "constraint_damping_control": {
            "z4c/damp_kappa1": 0.0,
            "z4c/damp_kappa2": 0.0,
            "z4c/target_kappa1": 0.0,
            "z4c/damp_kappa1_max_K": False,
            "z4c/roll_kappa": False,
        },
        "case_parameter_differences": {
            "case1_to_case2": "fixed Gamma-driver eta=2 -> zero shift",
        },
        "cases": cases,
        "all_cases_attempted": all(case["exit_code"] is not None for case in cases),
        "all_rank_bindings_verified": all_bindings,
        "qualification_claim": False,
    }


def write_comparison(args: argparse.Namespace) -> None:
    result = comparison_result(
        [json.loads(path.read_text()) for path in args.case_results]
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        case = Path(tmp_name)
        (case / "exit-status.txt").write_text("1\n")
        (case / "run.log").write_text(
            "time=1.25 cycle=8\n"
            "Z4c chi boundary prolongation rejected 3 parent stencils and 0 limited sibling groups\n"
        )
        (case / "synthetic.z4c.user.hst").write_text(
            "# [1]=time [2]=dt [3]=C-norm2 [4]=H-norm2 [5]=M-norm2 "
            "[6]=Z-norm2 [7]=max_abs_K [8]=nmb_total [9]=maxRefLev "
            "[10]=cycle [11]=axisLapse [12]=axisTau [13]=axisKret [14]=maxAbsKret\n"
            "1.25 0.01 2 3 4 5 6 7 8 9 0.5 1.0 10 11\n"
        )
        result = summarize_case(
            case, "synthetic", 2.0, 2.0, 0.02, "fixed_gamma_driver_eta2"
        )
        assert result["history_rows"] == 1
        assert result["max_refinement_level_reached"] == 8
        assert result["max_dt_mu_over_tau"] == 0.03
        assert result["constraint_damping"] is False
        assert result["damp_kappa1"] == result["damp_kappa2"] == 0.0
        assert result["chi_rejections"][0]["invalid_parent_stencils"] == 3
        assert result["last_log_time_cycle"] == {"time": 1.25, "cycle": 8}
        first = dict(result)
        first.update(
            name="fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64",
            tau=1.0,
            kappa=1.0,
            ko_dissipation=0.5,
            rank_bindings=[{
                "binding_verified": True,
                "gpu_name": "NVIDIA A100-SXM4-80GB",
            }],
        )
        second = dict(first)
        second.update(
            name="zero_shift_tau1_kappa1_l20_nocd_ko05_domain64",
            shift_condition="zero_shift",
        )
        comparison = comparison_result([first, second])
        assert comparison["all_cases_attempted"] is True
        assert comparison["all_rank_bindings_verified"] is True
        assert comparison["qualification_claim"] is False
        wrong_gpu = dict(second, rank_bindings=[{
            "binding_verified": True,
            "gpu_name": "NVIDIA A100-SXM4-40GB",
        }])
        assert comparison_result([first, wrong_gpu])["all_rank_bindings_verified"] is False
        mutated = dict(second, damp_kappa1=0.02)
        try:
            comparison_result([first, mutated])
        except AssertionError:
            pass
        else:
            raise AssertionError("nonzero constraint damping was accepted")
    print("ANALYZE_NOCD_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    case.add_argument("--case-dir", type=Path, required=True)
    case.add_argument("--name", required=True)
    case.add_argument("--tau", type=float, required=True)
    case.add_argument("--kappa", type=float, required=True)
    case.add_argument("--dissipation", type=float, required=True)
    case.add_argument(
        "--shift-condition",
        choices=("fixed_gamma_driver_eta2", "zero_shift"),
        required=True,
    )
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(func=write_case)
    comparison = sub.add_parser("comparison")
    comparison.add_argument("--case-results", type=Path, nargs=2, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.set_defaults(func=write_comparison)
    test = sub.add_parser("self-test")
    test.set_defaults(func=lambda _args: self_test())
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
