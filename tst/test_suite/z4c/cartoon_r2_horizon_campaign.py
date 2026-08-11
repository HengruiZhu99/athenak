#!/usr/bin/env python3
"""Prepare, run, and analyze the bounded Cartoon R2 horizon campaign.

This script composes with the R1 evidence helpers and drives one frozen AthenaK
executable.  It is deliberately limited to the declared Kerr-puncture and
angular-basis cases; it is not a general horizon-testing framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import cartoon_r1_campaign as r1


SCHEMA = "athenak_cartoon_r2_horizon_campaign_v1"
ANALYSIS_SCHEMA = "athenak_cartoon_r2_horizon_analysis_v1"
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR / "cartoon_r2_horizon.athinput"
BASIS_LEVELS = {
    "l2n8": {"lmax": 2, "ntheta": 8},
    "l4n16": {"lmax": 4, "ntheta": 16},
    "l6n24": {"lmax": 6, "ntheta": 24},
}
PHYSICAL_CASES = {
    "schwarzschild": {"chi": 0.0, "z_h": 0.0, "expected_branch": "origin"},
    "spin99_centered": {"chi": 0.99, "z_h": 0.0,
                        "expected_branch": "origin"},
    "spin99_plus": {"chi": 0.99, "z_h": 2.0, "expected_branch": "plus"},
    "spin99_minus": {"chi": 0.99, "z_h": -2.0, "expected_branch": "minus"},
}
HORIZON_COLUMNS = (
    "cycle", "time", "branch", "accepted", "center_z", "axis_extremum_z",
    "center_lapse", "area", "irreducible_mass", "horizon_mass", "spin_z",
    "mean_radius", "minimum_radius", "direct_residual", "flow_residual",
    "failure",
)
HORIZON_HEADER = "# " + " ".join(HORIZON_COLUMNS) + " a_l0..."
BRANCHES = {"origin", "plus", "minus"}
FAILURE_CODES = {
    "none", "nonfinite_coefficient", "coverage_or_radius",
    "nonfinite_integral", "iteration_limit", "axis_lapse_scan_coverage",
}
FASTFLOW_KEYS = (
    "fastflow_schema", "fastflow_coefficient_count", "fastflow_coefficients",
    "fastflow_surface_mode", "fastflow_selected_branch", "fastflow_center_count",
    "fastflow_center_z0", "fastflow_center_z1", "fastflow_status",
    "fastflow_failure_code", "fastflow_last_search_cycle",
    "fastflow_last_search_time", "fastflow_converged",
)
RESTART_CONTINUATION_KEYS = (*r1.CENTRAL_KEYS, *FASTFLOW_KEYS)
FIELD_MARKERS = {
    "constraints": "con_C",
    "weyl": "weyl_rpsi4",
    "curvature": "z4c_Kretschmann",
}
DIRECT_TOLERANCE = 3.0e-2
FLOW_TOLERANCE = 3.0e-2
MASS_RELATIVE_TOLERANCE = 5.0e-2
SPIN_ABSOLUTE_TOLERANCE = 8.0e-2
AREA_RELATIVE_TOLERANCE = 1.0e-1
RADIUS_RELATIVE_TOLERANCE = 3.0e-1
SHAPE_RELATIVE_TOLERANCE = 2.0e-1
BASIS_RELATIVE_TOLERANCE = 5.0e-2
CENTER_TOLERANCE = 1.0e-1


def attempt_name(physical: str, basis: str) -> str:
    return f"{physical}_{basis}"


def expected_fresh() -> tuple[str, ...]:
    return tuple(attempt_name(physical, basis)
                 for physical in PHYSICAL_CASES for basis in BASIS_LEVELS)


def expected_cases() -> tuple[str, ...]:
    fresh = expected_fresh()
    restarts = tuple(f"{physical}_restart" for physical in PHYSICAL_CASES)
    return fresh + restarts


def theory(chi: float, mass: float = 1.0) -> dict[str, float]:
    root = math.sqrt(1.0 - chi * chi)
    irreducible = mass * math.sqrt(0.5 * (1.0 + root))
    return {
        "mass": mass,
        "spin_z": chi * mass * mass,
        "area": 16.0 * math.pi * irreducible * irreducible,
        "irreducible_mass": irreducible,
        "coordinate_radius": 0.25 * mass * (1.0 + root),
    }


def declared_thresholds() -> dict[str, float]:
    return {
        "direct_residual": DIRECT_TOLERANCE,
        "flow_residual": FLOW_TOLERANCE,
        "mass_relative": MASS_RELATIVE_TOLERANCE,
        "spin_absolute": SPIN_ABSOLUTE_TOLERANCE,
        "area_relative": AREA_RELATIVE_TOLERANCE,
        "radius_relative": RADIUS_RELATIVE_TOLERANCE,
        "shape_relative": SHAPE_RELATIVE_TOLERANCE,
        "basis_relative": BASIS_RELATIVE_TOLERANCE,
        "center_absolute": CENTER_TOLERANCE,
    }


def validate_template() -> None:
    blocks = r1.parse_athinput(TEMPLATE)
    r1.require(blocks["problem"].get("pgen_name") == "kerr_puncture",
               "R2 template does not select kerr_puncture")
    r1.require(blocks["z4c"].get("symmetry") == "cartoon_so2" and
               blocks["z4c"].get("coordinate_map") ==
               "signed_rho_z_suppressed_y_v1",
               "R2 template does not select signed-rho Cartoon")
    r1.require(blocks["mesh"].get("nx3") == "1" and
               float(blocks["mesh"]["x1min"]) ==
               -float(blocks["mesh"]["x1max"]),
               "R2 template violates the collapsed signed-rho topology")
    r1.require(blocks["mesh_refinement"].get("refinement") == "adaptive" and
               blocks["mesh_refinement"].get("num_levels") == "3" and
               blocks["z4c_amr"].get("max_ref_lev") == "2",
               "R2 template omits the short two-level AMR refinement")
    r1.require(blocks["mesh"].get("nx1") == "128" and
               blocks["mesh"].get("nx2") == "128" and
               blocks["meshblock"].get("nx1") == "32" and
               blocks["meshblock"].get("nx2") == "32" and
               blocks["time"].get("nlim") == "3",
               "R2 short-run grid/cycle contract changed")
    fastflow = blocks["fastflow"]
    expected_fastflow = {
        "num_horizons", "lmax", "ntheta", "flow_iterations_0",
        "find_interval_0", "start_time_0", "stop_time_0", "initial_radius_0",
        "flow_alpha_beta_const_0", "dimensionless_hrms_tol_0",
        "mass_relative_tol_0", "cartoon_direct_residual_tol_0",
        "cartoon_pair_relative_tol_0", "cartoon_center_z_0",
        "cartoon_axis_search_bound_0", "cartoon_axis_search_samples_0",
        "cartoon_surface_mode_0",
    }
    r1.require(set(fastflow) == expected_fastflow and
               fastflow["num_horizons"] == "1" and
               fastflow["cartoon_surface_mode_0"] == "single",
               "R2 FastFlow input inventory changed")
    r1.require(float(fastflow["dimensionless_hrms_tol_0"]) == FLOW_TOLERANCE and
               float(fastflow["cartoon_direct_residual_tol_0"]) ==
               DIRECT_TOLERANCE,
               "R2 input residual thresholds changed")
    outputs = [values for block, values in blocks.items()
               if block.startswith("output")]
    r1.require([values.get("file_type") for values in outputs].count("hst") == 1 and
               [values.get("file_type") for values in outputs].count("rst") == 1,
               "R2 template requires exactly one history and restart output")
    r1.require({values.get("variable") for values in outputs} >=
               {"con", "weyl", "z4c_diag"},
               "R2 template omits constraints, Weyl, or curvature output")
    r1.require(all(values.get("single_file_per_rank") == "true"
                   for values in outputs if values.get("file_type") == "bin"),
               "R2 binary diagnostics cannot prove four-rank ownership")


def validate_source_contract(source: Path) -> None:
    """Bind the campaign to the in-tree m=0 adapter, not legacy FastFlow."""
    r1.validate_source_contract(source)
    cpp = (source / "src/z4c/cartoon_m0_fastflow.cpp").read_text(
        encoding="utf-8")
    header = (source / "src/z4c/cartoon_m0_fastflow.hpp").read_text(
        encoding="utf-8")
    fastflow = (source / "src/z4c/fastflow.cpp").read_text(encoding="utf-8")
    fastflow_header = (source / "src/z4c/fastflow.hpp").read_text(encoding="utf-8")
    cmake = (source / "src/CMakeLists.txt").read_text(encoding="utf-8")
    allocation = (source / "src/mesh/meshblock_pack.cpp").read_text(
        encoding="utf-8")
    markers = (
        'GetOrAddInteger("fastflow", "lmax", 4)',
        'GetOrAddInteger("fastflow", "ntheta", 12)',
        '"cartoon_direct_residual_tol_"',
        '"cartoon_pair_relative_tol_"',
        '"cartoon_center_z_"',
        '"cartoon_axis_search_bound_"',
        '"cartoon_axis_search_samples_"',
        '"cartoon_surface_mode_"',
        '".cartoon_m0_horizon_"',
        'candidate.direct_residual, candidate.flow_residual',
        'state.selected_branch = selected_.size() == 2 ? "plus_minus"',
    )
    for marker in markers:
        r1.require(marker in cpp, f"m=0 FastFlow schema marker disappeared: {marker}")
    r1.require("class CartoonM0FastFlow" in header and
               "std::unique_ptr<z4c::CartoonM0FastFlow>" in fastflow_header,
               "FastFlow adapter ownership marker disappeared")
    r1.require("cartoon_m0->Find(iter, time)" in fastflow and
               "cartoon_m0->Write(iter, time)" in fastflow,
               "FastFlow no longer delegates find/write to the m=0 adapter")
    r1.require("z4c/cartoon_m0_fastflow.cpp" in cmake,
               "m=0 FastFlow adapter is absent from the build graph")
    r1.require("FastFlow before the m=0 Cartoon adapter is integrated" not in
               allocation,
               "pre-allocation policy still rejects the integrated m=0 FastFlow")
    header_anchor = 'std::fprintf(output_, "# cycle'
    r1.require(header_anchor in cpp, "horizon output header disappeared")
    header_start = cpp.index(header_anchor)
    r1.require(");" in cpp[header_start:], "horizon output header is malformed")
    header_end = cpp.index(");", header_start)
    emitted_header = "".join(re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"',
                                         cpp[header_start:header_end]))
    emitted_header = emitted_header.replace("\\n", "\n")
    r1.require(emitted_header == HORIZON_HEADER + "\n",
               "exact horizon output-column contract changed")


def render_attempt(physical: str, basis: str) -> str:
    spec = PHYSICAL_CASES[physical]
    angular = BASIS_LEVELS[basis]
    radius = theory(float(spec["chi"]))["coordinate_radius"]
    overrides = {
        "job/basename": attempt_name(physical, basis),
        "problem/chi": format(float(spec["chi"]), ".17g"),
        "problem/z_h": format(float(spec["z_h"]), ".17g"),
        "fastflow/lmax": str(angular["lmax"]),
        "fastflow/ntheta": str(angular["ntheta"]),
        "fastflow/initial_radius_0": format(radius, ".17g"),
        "fastflow/cartoon_center_z_0":
            format(max(abs(float(spec["z_h"])), radius), ".17g"),
    }
    return r1.render_athinput(r1.parse_athinput(TEMPLATE), overrides)


def contract_payload(state: dict[str, Any]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    dynamic = {"status", "exit_code", "elapsed_seconds", "command",
               "log_path", "log_sha256", "restart_path", "restart_sha256"}
    for name, case in state["cases"].items():
        cases[name] = {key: value for key, value in case.items()
                       if key not in dynamic}
    return {"schema": state["schema"], "source": state["source"],
            "executable": state["executable"], "backend": state["backend"],
            "ranks": state["ranks"], "inputs": state["inputs"],
            "cases": cases, "thresholds": state["thresholds"]}


def contract_digest(state: dict[str, Any]) -> str:
    encoded = json.dumps(contract_payload(state), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_state(state: dict[str, Any]) -> None:
    allowed = {"schema", "source", "executable", "backend", "ranks", "inputs",
               "cases", "thresholds", "contract_sha256", "analysis"}
    r1.require(set(state) <= allowed and allowed - {"analysis"} ==
               set(state) - ({"analysis"} & set(state)),
               "R2 campaign-state field inventory is malformed")
    r1.require(state.get("schema") == SCHEMA and state.get("backend") == "Cuda" and
               state.get("ranks") == 4, "unknown R2 campaign resource contract")
    r1.require(set(state["source"]) ==
               {"path", "commit", "tree", "kokkos_commit"} and
               set(state["executable"]) == {"path", "sha256"},
               "R2 source/executable provenance inventory is malformed")
    r1.require(set(state["inputs"]) == set(expected_fresh()) and
               set(state["cases"]) == set(expected_cases()),
               "R2 input/case inventory changed")
    threshold_keys = {
        "direct_residual", "flow_residual", "mass_relative", "spin_absolute",
        "area_relative", "radius_relative", "shape_relative", "basis_relative",
        "center_absolute",
    }
    r1.require(set(state["thresholds"]) == threshold_keys and
               state["thresholds"] == declared_thresholds(),
               "R2 threshold inventory is malformed")
    input_keys = {"path", "sha256", "template_sha256", "spec"}
    spec_keys = {"physical", "basis", "chi", "z_h", "expected_branch",
                 "lmax", "ntheta"}
    r1.require(all(set(record) == input_keys and
                   set(record["spec"]) == spec_keys
                   for record in state["inputs"].values()),
               "R2 generated-input metadata is malformed")
    dynamic = {"status", "exit_code", "elapsed_seconds", "command",
               "log_path", "log_sha256", "restart_path", "restart_sha256"}
    for name, case in state["cases"].items():
        base = {"mode", "input"} if case.get("mode") == "fresh" else {
            "mode", "restart_from", "checkpoint_cycle"}
        r1.require(base | {"status"} <= set(case) <= base | dynamic and
                   case["status"] in {"pending", "complete", "failed"},
                   f"R2 case metadata is malformed: {name}")
        if case["mode"] == "fresh":
            r1.require(case["input"] == name,
                       f"R2 fresh case/input binding changed: {name}")
        else:
            physical = name.removesuffix("_restart")
            r1.require(case["restart_from"] == attempt_name(physical, "l6n24") and
                       case["checkpoint_cycle"] == 2,
                       f"R2 restart dependency changed: {name}")
    if "analysis" in state:
        r1.require(set(state["analysis"]) == {"path", "sha256", "verdict"},
                   "R2 analysis metadata is malformed")
    r1.require(state["contract_sha256"] == contract_digest(state),
               "R2 campaign contract changed after preparation")
    executable = Path(state["executable"]["path"])
    r1.require(executable.is_file() and r1.sha256(executable) ==
               state["executable"]["sha256"], "bound executable changed")
    source = Path(state["source"]["path"])
    r1.require(r1.git(source, "rev-parse", "HEAD") == state["source"]["commit"] and
               r1.git(source, "rev-parse", "HEAD^{tree}") ==
               state["source"]["tree"] and
               r1.git(source, "rev-parse", "HEAD:kokkos") ==
               state["source"]["kokkos_commit"] and
               r1.git(source, "status", "--porcelain") == "",
               "bound source identity or cleanliness changed")
    for name, record in state["inputs"].items():
        path = Path(record["path"])
        physical = record["spec"]["physical"]
        basis = record["spec"]["basis"]
        r1.require(physical in PHYSICAL_CASES and basis in BASIS_LEVELS,
                   f"unknown generated-input specification: {name}")
        r1.require(path.is_file() and r1.sha256(path) == record["sha256"] and
                   name == attempt_name(physical, basis) and
                   {key: record["spec"][key] for key in PHYSICAL_CASES[physical]} ==
                   PHYSICAL_CASES[physical] and
                   {key: record["spec"][key] for key in BASIS_LEVELS[basis]} ==
                   BASIS_LEVELS[basis],
                   f"bound generated input changed: {name}")


def prepare(args: argparse.Namespace) -> None:
    validate_template()
    source = args.source.resolve()
    validate_source_contract(source)
    executable = args.executable.resolve()
    output = args.output.resolve()
    r1.require(executable.is_file(), f"missing Athena executable: {executable}")
    r1.require(args.ranks == 4 and args.backend == "Cuda",
               "R2 qualification requires CUDA and exactly four MPI ranks")
    r1.require(r1.git(source, "status", "--porcelain") == "",
               "R2 preparation requires a clean source checkout")
    r1.require(not output.exists() or not any(output.iterdir()),
               "R2 output root already contains evidence")
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, Any] = {}
    for physical in PHYSICAL_CASES:
        for basis in BASIS_LEVELS:
            name = attempt_name(physical, basis)
            path = input_dir / f"{name}.athinput"
            path.write_text(render_attempt(physical, basis), encoding="utf-8")
            inputs[name] = {
                "path": str(path.resolve()), "sha256": r1.sha256(path),
                "template_sha256": r1.sha256(TEMPLATE),
                "spec": {"physical": physical, "basis": basis,
                         **PHYSICAL_CASES[physical], **BASIS_LEVELS[basis]},
            }
    cases: dict[str, Any] = {}
    for name in expected_fresh():
        cases[name] = {"mode": "fresh", "input": name, "status": "pending"}
    for physical in PHYSICAL_CASES:
        parent = attempt_name(physical, "l6n24")
        cases[f"{physical}_restart"] = {
            "mode": "restart", "restart_from": parent,
            "checkpoint_cycle": 2, "status": "pending",
        }
    thresholds = declared_thresholds()
    state: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {"path": str(source),
                   "commit": r1.git(source, "rev-parse", "HEAD"),
                   "tree": r1.git(source, "rev-parse", "HEAD^{tree}"),
                   "kokkos_commit": r1.git(source, "rev-parse", "HEAD:kokkos")},
        "executable": {"path": str(executable), "sha256": r1.sha256(executable)},
        "backend": args.backend, "ranks": args.ranks,
        "inputs": inputs, "cases": cases, "thresholds": thresholds,
    }
    state["contract_sha256"] = contract_digest(state)
    r1.atomic_json(output / "campaign_state.json", state)


def run_case(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = r1.strict_load(state_path)
    verify_state(state)
    r1.require(args.case in state["cases"], f"unknown R2 case {args.case}")
    case = state["cases"][args.case]
    if case["status"] == "complete":
        r1.require(args.resume, f"case {args.case} is already complete")
        log = Path(case["log_path"])
        r1.require(log.is_file() and r1.sha256(log) == case["log_sha256"],
                   f"completed evidence changed: {args.case}")
        return
    r1.require(case["status"] == "pending",
               f"case {args.case} has terminal status {case['status']}")
    root = state_path.parent
    run_dir = root / "runs" / args.case
    r1.require(not run_dir.exists() or not any(run_dir.iterdir()),
               f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    executable = state["executable"]["path"]
    if case["mode"] == "fresh":
        athena = [executable, "-i", state["inputs"][case["input"]]["path"],
                  "-d", str(run_dir)]
    else:
        dependency = state["cases"][case["restart_from"]]
        r1.require(dependency["status"] == "complete",
                   f"restart dependency is incomplete: {case['restart_from']}")
        restart = r1.select_restart(root / "runs" / case["restart_from"],
                                    int(case["checkpoint_cycle"]))
        athena = [executable, "-r", str(restart), "-d", str(run_dir)]
        case["restart_path"] = str(restart)
        case["restart_sha256"] = r1.sha256(restart)
    wrapper = (Path(state["source"]["path"]) /
               "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")
    r1.require(wrapper.is_file(), "established rank-binding wrapper is missing")
    command = [
        "srun", "--nodes=1", "--ntasks=4", "--ntasks-per-node=4",
        "--cpus-per-task=8", "--gpus-per-task=1",
        "--gpu-bind=map_gpu:0,1,2,3", "--cpu-bind=cores", "--exact",
        "--kill-on-bad-exit=1", sys.executable, str(wrapper),
        "--evidence-dir", str(run_dir / "bindings"), "--require-cuda", "--",
        *athena,
    ]
    case["command"] = command
    log = run_dir / "run.log"
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "8", "KOKKOS_NUM_THREADS": "8",
                        "MPICH_GPU_SUPPORT_ENABLED": "1",
                        "MPICH_GPU_IPC_ENABLED": "0",
                        "MPICH_OFI_NIC_POLICY": "GPU"})
    started = time.monotonic()
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=root, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT,
                                check=False)
    case["elapsed_seconds"] = time.monotonic() - started
    case["exit_code"] = result.returncode
    case["log_path"] = str(log)
    case["log_sha256"] = r1.sha256(log)
    case["status"] = "complete" if result.returncode == 0 else "failed"
    r1.atomic_json(state_path, state)
    r1.require(result.returncode == 0,
               f"case {args.case} failed; raw evidence was preserved")


def parse_horizon(path: Path, lmax: int) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    r1.require(lines and lines[0] == HORIZON_HEADER,
               f"unexpected horizon header: {path}")
    records: list[dict[str, Any]] = []
    keys: set[tuple[int, str]] = set()
    for number, line in enumerate(lines[1:], 2):
        r1.require(line.strip() and not line.startswith("#"),
                   f"malformed horizon record at {path}:{number}")
        fields = line.split()
        r1.require(len(fields) >= len(HORIZON_COLUMNS),
                   f"short horizon record at {path}:{number}")
        branch, failure = fields[2], fields[15]
        r1.require(branch in BRANCHES and failure in FAILURE_CODES,
                   f"unknown branch/failure at {path}:{number}")
        try:
            cycle = int(fields[0])
            accepted_integer = int(fields[3])
            numeric = [float(token) for token in fields[1:2] + fields[4:15]]
            coefficients = [float(token) for token in fields[16:]]
        except ValueError as error:
            raise RuntimeError(f"malformed horizon value at {path}:{number}") from error
        r1.require(cycle >= 0 and accepted_integer in (0, 1) and
                   all(math.isfinite(value) for value in numeric + coefficients),
                   f"nonfinite/malformed horizon record at {path}:{number}")
        accepted = accepted_integer == 1
        r1.require((accepted and failure == "none" and
                    len(coefficients) == lmax + 1) or
                   (not accepted and failure != "none" and
                    len(coefficients) in (0, lmax + 1)),
                   f"inconsistent horizon status at {path}:{number}")
        key = (cycle, branch)
        r1.require(key not in keys, f"duplicate horizon record {key} in {path}")
        keys.add(key)
        record = dict(zip(
            ("time", "center_z", "axis_extremum_z", "center_lapse", "area",
             "irreducible_mass", "horizon_mass", "spin_z", "mean_radius",
             "minimum_radius", "direct_residual", "flow_residual"), numeric))
        record.update({"cycle": cycle, "branch": branch, "accepted": accepted,
                       "failure": failure, "coefficients": coefficients})
        if accepted:
            r1.require(record["area"] > 0.0 and record["mean_radius"] > 0.0 and
                       record["minimum_radius"] > 0.0 and
                       record["irreducible_mass"] > 0.0 and
                       record["horizon_mass"] > 0.0 and
                       record["direct_residual"] >= 0.0 and
                       record["flow_residual"] >= 0.0,
                       f"nonphysical accepted horizon at {path}:{number}")
        records.append(record)
    r1.require(records, f"horizon output has no records: {path}")
    cycles = {record["cycle"] for record in records}
    for cycle in cycles:
        r1.require({record["branch"] for record in records
                    if record["cycle"] == cycle} == BRANCHES,
                   f"missing branch evidence at cycle {cycle} in {path}")
    return records


def legendre(l: int, x: float) -> float:
    if l == 0:
        return 1.0
    if l == 1:
        return x
    previous, current = 1.0, x
    for degree in range(2, l + 1):
        previous, current = current, (
            (2 * degree - 1) * x * current - (degree - 1) * previous
        ) / degree
    return current


def shape_metrics(coefficients: list[float]) -> dict[str, float]:
    radii = []
    for index in range(65):
        x = -1.0 + 2.0 * index / 64.0
        radius = sum(coefficient * math.sqrt((2 * degree + 1) /
                                             (4.0 * math.pi)) *
                     legendre(degree, x)
                     for degree, coefficient in enumerate(coefficients))
        radii.append(radius)
    r1.require(all(math.isfinite(value) and value > 0.0 for value in radii),
               "accepted coefficients reconstruct a nonpositive surface")
    mean = sum(radii) / len(radii)
    higher = math.sqrt(sum(value * value for value in coefficients[1:]))
    return {"sampled_minimum": min(radii), "sampled_maximum": max(radii),
            "relative_spread": (max(radii) - min(radii)) / mean,
            "relative_higher_mode_l2": higher /
                max(abs(coefficients[0]), 1.0e-300)}


def raw_manifest(run_dir: Path) -> dict[str, str]:
    return {str(path.relative_to(run_dir)): r1.sha256(path)
            for path in sorted(run_dir.rglob("*")) if path.is_file()}


def binary_summary(run_dir: Path, reader: Any) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel
    files = sorted(run_dir.glob("bin/rank_*/*.bin"))
    r1.require(files, f"binary diagnostics are missing: {run_dir}")
    groups = {name: {"cycles": set(), "ranks": set(), "max_abs": 0.0}
              for name in FIELD_MARKERS}
    for path in files:
        data = reader(path)
        matches = [name for name, marker in FIELD_MARKERS.items()
                   if marker in data["var_names"]]
        r1.require(len(matches) == 1, f"unexpected binary inventory: {path}")
        rank_match = re.fullmatch(r"rank_(\d{8})", path.parent.name)
        r1.require(rank_match is not None, f"malformed rank directory: {path}")
        summary = groups[matches[0]]
        summary["cycles"].add(int(data["cycle"]))
        summary["ranks"].add(int(rank_match.group(1)))
        for arrays in data["mb_data"].values():
            for array in arrays:
                values = np.asarray(array)
                r1.require(bool(np.all(np.isfinite(values))),
                           f"nonfinite binary diagnostics in {path}")
                if values.size:
                    summary["max_abs"] = max(summary["max_abs"],
                                             float(np.max(np.abs(values))))
    result = {}
    for name, summary in groups.items():
        r1.require(summary["cycles"] and summary["ranks"] == set(range(4)),
                   f"missing cycle/rank binary evidence for {name}")
        result[name] = {"cycles": sorted(summary["cycles"]),
                        "ranks": sorted(summary["ranks"]),
                        "max_abs": summary["max_abs"]}
    return result


def selected_row(rows: list[dict[str, Any]],
                 carrier: dict[str, str]) -> dict[str, Any] | None:
    r1.require(carrier["fastflow_surface_mode"] == "single",
               "latest restart changed the declared single-surface mode")
    if carrier["fastflow_converged"] in {"0", "false"}:
        r1.require(carrier["fastflow_status"] == "failed" and
                   carrier["fastflow_failure_code"] == "no_candidate" and
                   carrier["fastflow_selected_branch"] == "none" and
                   carrier["fastflow_center_count"] == "0" and
                   carrier["fastflow_coefficient_count"] == "0" and
                   carrier["fastflow_coefficients"] == "none",
                   "failed FastFlow restart state is malformed")
        return None
    r1.require(carrier["fastflow_converged"] in {"1", "true"} and
               carrier["fastflow_status"] == "accepted" and
               carrier["fastflow_failure_code"] == "none" and
               carrier["fastflow_center_count"] == "1",
               "latest restart has malformed accepted m=0 horizon state")
    cycle = int(carrier["fastflow_last_search_cycle"])
    branch = carrier["fastflow_selected_branch"]
    matches = [row for row in rows if row["cycle"] == cycle and
               row["branch"] == branch]
    r1.require(len(matches) == 1 and matches[0]["accepted"],
               "selected restart horizon has no unique accepted output row")
    selected = matches[0]
    try:
        count = int(carrier["fastflow_coefficient_count"])
        coefficients = [float(token) for token in
                        carrier["fastflow_coefficients"].split(",")]
        center = float(carrier["fastflow_center_z0"])
        search_time = float(carrier["fastflow_last_search_time"])
    except ValueError as error:
        raise RuntimeError("malformed selected FastFlow restart values") from error
    r1.require(count == len(selected["coefficients"]) == len(coefficients) and
               all(math.isfinite(value) for value in coefficients) and
               all(left == right for left, right in
                   zip(coefficients, selected["coefficients"])) and
               center == selected["center_z"] and search_time == selected["time"],
               "selected horizon output and restart carrier disagree")
    return selected


def collect_case(state: dict[str, Any], name: str, reader: Any) -> dict[str, Any]:
    case = state["cases"][name]
    run_dir = Path(case["log_path"]).parent
    if case["mode"] == "fresh":
        input_name = case["input"]
    else:
        input_name = state["cases"][case["restart_from"]]["input"]
    spec = state["inputs"][input_name]["spec"]
    horizon_files = sorted(run_dir.glob("*.cartoon_m0_horizon_0.txt"))
    histories = sorted(run_dir.glob("*.hst"))
    r1.require(len(horizon_files) == 1 and len(histories) == 1,
               f"expected one horizon/history file in {run_dir}")
    rows = parse_horizon(horizon_files[0], int(spec["lmax"]))
    r1.require(histories[0].read_text(encoding="utf-8").count(
                   "# Athena++ history data") == 1,
               f"duplicate/missing history header: {histories[0]}")
    history = r1.read_history(histories[0])
    r1.require(tuple(history) == r1.HISTORY_KEYS,
               f"unexpected history inventory: {histories[0]}")
    restart_path, carrier = r1.latest_restart(run_dir)
    selected = selected_row(rows, carrier)
    if selected is not None:
        selected["shape"] = shape_metrics(selected["coefficients"])
    return {
        "status": "complete", "spec": spec, "history": history,
        "horizon_rows": rows, "selected": selected,
        "selection_status": "accepted" if selected is not None else "failed",
        "restart_path": str(restart_path),
        "restart_sha256": r1.sha256(restart_path),
        "restart_carrier": carrier,
        "binary": binary_summary(run_dir, reader),
        "rank_bindings": r1.binding_summary(run_dir, "Cuda", 4),
        "elapsed_seconds": case["elapsed_seconds"],
        "evidence_files": raw_manifest(run_dir),
    }


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def append_gate(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def validate_observations(observations: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    append_gate(failures, set(observations) == set(expected_cases()),
                "missing or unexpected R2 case evidence")
    if set(observations) != set(expected_cases()):
        return {"verdict": "fail", "failures": failures}
    for name, record in observations.items():
        if record.get("status") != "complete":
            failures.append(f"{name}: {record.get('reason', 'process failure')}")
            continue
        r1.require_finite_json(record)
        selected = record["selected"]
        spec = record["spec"]
        expected = theory(float(spec["chi"]))
        required_selection = (name not in expected_fresh() or
                              spec["basis"] in {"l4n16", "l6n24"})
        append_gate(failures, not required_selection or selected is not None,
                    f"{name}: mandatory angular basis found no horizon")
        if selected is not None and name in expected_fresh() and \
                spec["basis"] in {"l4n16", "l6n24"}:
            append_gate(failures, selected["branch"] == spec["expected_branch"],
                        f"{name}: selected wrong center branch")
            append_gate(failures,
                        selected["direct_residual"] <= DIRECT_TOLERANCE,
                        f"{name}: direct residual exceeds threshold")
            append_gate(failures, selected["flow_residual"] <= FLOW_TOLERANCE,
                        f"{name}: flow residual exceeds threshold")
        if selected is not None and name in expected_fresh() and \
                spec["basis"] == "l6n24":
            append_gate(failures,
                        relative(selected["horizon_mass"], expected["mass"]) <=
                        MASS_RELATIVE_TOLERANCE,
                        f"{name}: Christodoulou mass misses analytic Kerr")
            append_gate(failures,
                        abs(selected["spin_z"] - expected["spin_z"]) <=
                        SPIN_ABSOLUTE_TOLERANCE,
                        f"{name}: spin misses analytic Kerr")
            append_gate(failures,
                        relative(selected["area"], expected["area"]) <=
                        AREA_RELATIVE_TOLERANCE,
                        f"{name}: area misses analytic Kerr")
            append_gate(failures,
                        relative(selected["mean_radius"],
                                 expected["coordinate_radius"]) <=
                        RADIUS_RELATIVE_TOLERANCE,
                        f"{name}: coordinate radius misses analytic Kerr")
            append_gate(failures,
                        abs(selected["center_z"] - float(spec["z_h"])) <=
                        CENTER_TOLERANCE,
                        f"{name}: axial center recovery misses tolerance")
            append_gate(failures,
                        selected["shape"]["relative_spread"] <=
                        SHAPE_RELATIVE_TOLERANCE,
                        f"{name}: reconstructed surface shape is irregular")
        history = record["history"]
        append_gate(failures,
                    history["ahStatus"][-1] ==
                    (1.0 if selected is not None else 0.0) and
                    (selected is None or
                     int(history["ahLastCyc"][-1]) == selected["cycle"]),
                    f"{name}: history and selected horizon disagree")
        append_gate(failures,
                    all(history[key][-1] >= 0.0 for key in
                        ("C-norm2", "H-norm2", "M-norm2", "Z-norm2",
                         "max_abs_K", "maxAbsKret", "axisLapse", "axisKret")),
                    f"{name}: invalid constraint/curvature/axis diagnostics")
        append_gate(failures, len(record["rank_bindings"]) == 4 and
                    len({item["selected_uuid"]
                         for item in record["rank_bindings"]}) == 4,
                    f"{name}: CUDA rank/device binding is incomplete")
        append_gate(failures, bool(record["evidence_files"]),
                    f"{name}: raw checksum inventory is empty")
    for physical in PHYSICAL_CASES:
        medium_name = attempt_name(physical, "l4n16")
        fine_name = attempt_name(physical, "l6n24")
        if observations[medium_name].get("status") != "complete" or \
                observations[fine_name].get("status") != "complete":
            continue
        medium = observations[medium_name]["selected"]
        fine = observations[fine_name]["selected"]
        if medium is None or fine is None:
            failures.append(f"{physical}: angular-basis sequence lacks a horizon")
            continue
        for key in ("area", "horizon_mass", "spin_z", "mean_radius", "center_z"):
            absolute = key in {"spin_z", "center_z"}
            tolerance = CENTER_TOLERANCE if absolute else \
                BASIS_RELATIVE_TOLERANCE
            difference = abs(medium[key] - fine[key]) if absolute else \
                relative(medium[key], fine[key])
            append_gate(failures, difference <= tolerance,
                        f"{physical}: angular-basis {key} has not stabilized")
        restart_name = f"{physical}_restart"
        restart = observations[restart_name]
        if restart.get("status") == "complete":
            append_gate(failures,
                        {key: fine_record for key, fine_record in
                         observations[fine_name]["restart_carrier"].items()
                         if key in RESTART_CONTINUATION_KEYS} ==
                        {key: restart_record for key, restart_record in
                         restart["restart_carrier"].items()
                         if key in RESTART_CONTINUATION_KEYS},
                        f"{physical}: restart continuation changed diagnostic/horizon state")
    plus = observations[attempt_name("spin99_plus", "l6n24")]
    minus = observations[attempt_name("spin99_minus", "l6n24")]
    if plus.get("status") == "complete" and minus.get("status") == "complete":
        psel, msel = plus["selected"], minus["selected"]
        append_gate(failures, psel is not None and msel is not None,
                    "displaced mirror evidence is one-sided")
        if psel is None or msel is None:
            return {"verdict": "fail", "failures": failures,
                    "attempted_angular_bases": [dict(value) for value in
                                                 BASIS_LEVELS.values()],
                    "physical_cases": list(PHYSICAL_CASES),
                    "failed_candidate_rows": sum(
                        1 for record in observations.values()
                        if record.get("status") == "complete"
                        for row in record["horizon_rows"]
                        if not row["accepted"])}
        append_gate(failures, psel["branch"] == "plus" and
                    msel["branch"] == "minus",
                    "displaced mirror selected the wrong branches")
        append_gate(failures, abs(psel["center_z"] + msel["center_z"]) <=
                    CENTER_TOLERANCE, "displaced centers do not reflect")
        for key in ("area", "horizon_mass", "spin_z", "mean_radius",
                    "direct_residual", "flow_residual"):
            absolute = key in {"spin_z", "direct_residual", "flow_residual"}
            difference = abs(psel[key] - msel[key]) if absolute else \
                relative(psel[key], msel[key])
            tolerance = DIRECT_TOLERANCE if absolute else \
                BASIS_RELATIVE_TOLERANCE
            append_gate(failures, difference <= tolerance,
                        f"displaced mirror {key} mismatch")
        reflected = [((-1) ** degree) * value
                     for degree, value in enumerate(msel["coefficients"])]
        coefficient_error = max(abs(left - right)
                                for left, right in zip(psel["coefficients"],
                                                       reflected))
        append_gate(failures, coefficient_error <=
                    BASIS_RELATIVE_TOLERANCE * psel["mean_radius"],
                    "displaced mirror shape coefficients mismatch")
    return {"verdict": "pass" if not failures else "fail",
            "failures": failures,
            "attempted_angular_bases": [dict(value) for value in
                                         BASIS_LEVELS.values()],
            "physical_cases": list(PHYSICAL_CASES),
            "failed_candidate_rows": sum(
                1 for record in observations.values()
                if record.get("status") == "complete"
                for row in record["horizon_rows"] if not row["accepted"]),
    }


def analyze(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = r1.strict_load(state_path)
    verify_state(state)
    r1.require(all(case["status"] in {"complete", "failed"}
                   for case in state["cases"].values()),
               "R2 analysis requires every declared attempt to reach a terminal state")
    reader = r1.import_binary_reader(Path(state["source"]["path"]))
    observations: dict[str, Any] = {}
    for name, case in state["cases"].items():
        if case["status"] == "failed":
            run_dir = Path(case["log_path"]).parent
            observations[name] = {"status": "failed", "reason":
                                  f"process exit {case['exit_code']}",
                                  "evidence_files": raw_manifest(run_dir)}
            continue
        try:
            if case["mode"] == "restart":
                r1.verify_restart_lineage(state, name)
            observations[name] = collect_case(state, name, reader)
        except RuntimeError as error:
            run_dir = Path(case["log_path"]).parent
            observations[name] = {"status": "invalid", "reason": str(error),
                                  "evidence_files": raw_manifest(run_dir)}
    summary = validate_observations(observations)
    report = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_contract_sha256": state["contract_sha256"],
        "provenance": {"source": state["source"],
                       "executable": state["executable"],
                       "inputs": state["inputs"], "backend": state["backend"],
                       "ranks": state["ranks"]},
        "thresholds": state["thresholds"], "summary": summary,
        "cases": observations,
    }
    output = state_path.parent / "r2_horizon_analysis.json"
    r1.atomic_json(output, report)
    state["analysis"] = {"path": str(output), "sha256": r1.sha256(output),
                         "verdict": summary["verdict"]}
    r1.atomic_json(state_path, state)
    r1.require(summary["verdict"] == "pass",
               f"R2 qualification failed; evidence preserved in {output}")


def synthetic_selected(spec: dict[str, Any]) -> dict[str, Any]:
    expected = theory(float(spec["chi"]))
    radius = expected["coordinate_radius"]
    coefficients = [radius * math.sqrt(4.0 * math.pi)] + \
        [0.0] * int(spec["lmax"])
    return {
        "cycle": 3, "time": 0.1, "branch": spec["expected_branch"],
        "accepted": True, "center_z": float(spec["z_h"]),
        "axis_extremum_z": float(spec["z_h"]), "center_lapse": 0.2,
        "area": expected["area"],
        "irreducible_mass": expected["irreducible_mass"],
        "horizon_mass": expected["mass"], "spin_z": expected["spin_z"],
        "mean_radius": radius, "minimum_radius": radius,
        "direct_residual": 1.0e-4, "flow_residual": 2.0e-4,
        "failure": "none", "coefficients": coefficients,
        "shape": shape_metrics(coefficients),
    }


def synthetic_observations() -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for name in expected_cases():
        if name.endswith("_restart"):
            physical = name.removesuffix("_restart")
            basis = "l6n24"
        else:
            physical = next(item for item in PHYSICAL_CASES
                            if name.startswith(item + "_"))
            basis = name.removeprefix(physical + "_")
        spec = {"physical": physical, "basis": basis,
                **PHYSICAL_CASES[physical], **BASIS_LEVELS[basis]}
        selected = synthetic_selected(spec)
        rows = []
        for branch in sorted(BRANCHES):
            if branch == selected["branch"]:
                rows.append(dict(selected))
            else:
                failed = dict(selected)
                failed.update({"branch": branch, "accepted": False,
                               "failure": "iteration_limit"})
                rows.append(failed)
        history = {key: [value] for key, value in {
            "time": 0.1, "dt": 0.01, "C-norm2": 1.0e-6,
            "H-norm2": 1.0e-7, "M-norm2": 1.0e-7,
            "Z-norm2": 1.0e-8, "Mx-norm2": 1.0e-7,
            "My-norm2": 1.0e-7, "Mz-norm2": 1.0e-7,
            "Theta-norm": 1.0e-8, "Volume": 1.0, "max_abs_K": 0.1,
            "nmb_total": 16.0, "maxAbsKret": 2.0, "maxRefLev": 2.0,
            "maxNmbRank": 4.0, "ahStatus": 1.0, "ahLastCyc": 3.0,
            "cycle": 3.0, "axisLapse": 0.2, "axisTau": 0.1,
            "axisKret": 2.0,
        }.items()}
        carrier = {key: "0" for key in r1.RESTART_KEYS}
        carrier.update({"fastflow_schema": "1",
                        "fastflow_coefficient_count": str(len(selected["coefficients"])),
                        "fastflow_coefficients": ",".join(
                            format(value, ".17g") for value in selected["coefficients"]),
                        "fastflow_surface_mode": "single",
                        "fastflow_selected_branch": selected["branch"],
                        "fastflow_center_count": "1",
                        "fastflow_center_z0": format(selected["center_z"], ".17g"),
                        "fastflow_center_z1": "0", "fastflow_status": "accepted",
                        "fastflow_failure_code": "none",
                        "fastflow_last_search_cycle": "3",
                        "fastflow_last_search_time": "0.1",
                        "fastflow_converged": "1"})
        observations[name] = {
            "status": "complete", "spec": spec, "history": history,
            "horizon_rows": rows, "selected": selected,
            "selection_status": "accepted",
            "restart_path": f"{name}.rst", "restart_sha256": "a" * 64,
            "restart_carrier": carrier,
            "binary": {group: {"cycles": [3], "ranks": list(range(4)),
                               "max_abs": 1.0} for group in FIELD_MARKERS},
            "rank_bindings": [{"selected_uuid": f"GPU-{rank}"}
                              for rank in range(4)],
            "elapsed_seconds": 1.0, "evidence_files": {"run.log": "a" * 64},
        }
    return observations


def self_test() -> None:
    validate_template()
    for physical, physical_spec in PHYSICAL_CASES.items():
        for basis, angular in BASIS_LEVELS.items():
            rendered = r1.parse_athinput_text(render_attempt(physical, basis),
                                              f"synthetic:{physical}:{basis}")
            r1.require(float(rendered["problem"]["chi"]) ==
                       physical_spec["chi"] and
                       float(rendered["problem"]["z_h"]) ==
                       physical_spec["z_h"] and
                       int(rendered["fastflow"]["lmax"]) == angular["lmax"] and
                       int(rendered["fastflow"]["ntheta"]) == angular["ntheta"] and
                       float(rendered["fastflow"]["initial_radius_0"]) ==
                       theory(float(physical_spec["chi"]))["coordinate_radius"],
                       f"R2 rendered-input contract changed: {physical}:{basis}")
    synthetic_summary = validate_observations(synthetic_observations())
    r1.require(synthetic_summary["verdict"] == "pass" and
               synthetic_summary["failed_candidate_rows"] ==
               2 * len(expected_cases()),
               "synthetic R2 qualification/failure archive fixture failed")
    coarse_failure = synthetic_observations()
    coarse = coarse_failure[attempt_name("schwarzschild", "l2n8")]
    coarse["selected"] = None
    coarse["selection_status"] = "failed"
    coarse["history"]["ahStatus"] = [0.0]
    for row in coarse["horizon_rows"]:
        row["accepted"] = False
        row["failure"] = "iteration_limit"
    coarse["restart_carrier"].update({
        "fastflow_coefficient_count": "0", "fastflow_coefficients": "none",
        "fastflow_selected_branch": "none", "fastflow_center_count": "0",
        "fastflow_status": "failed", "fastflow_failure_code": "no_candidate",
        "fastflow_converged": "0",
    })
    r1.require(validate_observations(coarse_failure)["verdict"] == "pass" and
               selected_row(coarse["horizon_rows"],
                            coarse["restart_carrier"]) is None,
               "archived coarse-basis failure was not retained as nullable evidence")
    one_sided = synthetic_observations()
    del one_sided[attempt_name("spin99_minus", "l6n24")]
    r1.require(validate_observations(one_sided)["verdict"] == "fail",
               "one-sided mirror evidence was accepted")
    with tempfile.TemporaryDirectory(prefix="cartoon-r2-selftest-") as directory:
        path = Path(directory) / "synthetic.cartoon_m0_horizon_0.txt"
        spec = {**PHYSICAL_CASES["schwarzschild"], **BASIS_LEVELS["l2n8"]}
        selected = synthetic_selected(spec)
        lines = [HORIZON_HEADER]
        for branch in ("origin", "plus", "minus"):
            accepted = branch == "origin"
            row = dict(selected)
            row.update({"branch": branch, "accepted": accepted,
                        "failure": "none" if accepted else "iteration_limit"})
            values = [str(row["cycle"]), format(row["time"], ".17e"), branch,
                      "1" if accepted else "0"]
            values.extend(format(row[key], ".17e") for key in
                          HORIZON_COLUMNS[4:15])
            values.append(row["failure"])
            values.extend(format(value, ".17e") for value in row["coefficients"])
            lines.append(" ".join(values))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        r1.require(len(parse_horizon(path, 2)) == 3,
                   "exact synthetic horizon fixture was not parsed")
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        try:
            parse_horizon(path, 2)
        except RuntimeError as error:
            r1.require("missing branch evidence" in str(error),
                       "wrong missing-branch failure")
        else:
            raise RuntimeError("missing horizon branch was accepted")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.write_text(path.read_text(encoding="utf-8") + lines[1] + "\n",
                        encoding="utf-8")
        try:
            parse_horizon(path, 2)
        except RuntimeError as error:
            r1.require("duplicate horizon record" in str(error),
                       "wrong duplicate-record failure")
        else:
            raise RuntimeError("duplicate horizon record was accepted")
        path.write_text("\n".join(lines).replace("1.00000000000000000e+00",
                                                 "nan", 1) + "\n",
                        encoding="utf-8")
        try:
            parse_horizon(path, 2)
        except RuntimeError as error:
            r1.require("nonfinite" in str(error), "wrong nonfinite-row failure")
        else:
            raise RuntimeError("nonfinite horizon record was accepted")
        strict = Path(directory) / "strict.json"
        try:
            r1.atomic_json(strict, {"bad": float("inf")})
        except RuntimeError:
            pass
        else:
            raise RuntimeError("R2 strict JSON emitted Infinity")
    dummy = {"schema": SCHEMA, "source": {}, "executable": {},
             "backend": "Cuda", "ranks": 4, "inputs": {}, "cases": {},
             "thresholds": {}}
    dummy["contract_sha256"] = contract_digest(dummy)
    r1.require(dummy["contract_sha256"] !=
               contract_digest(dummy | {"ranks": 2}),
               "R2 contract digest did not bind the MPI rank count")
    print("Cartoon R2 horizon campaign tooling self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    make = subparsers.add_parser("prepare")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--executable", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--backend", choices=("Cuda",), default="Cuda")
    make.add_argument("--ranks", type=int, default=4)
    run = subparsers.add_parser("run-case")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--case", choices=expected_cases(), required=True)
    run.add_argument("--resume", action="store_true")
    inspect = subparsers.add_parser("analyze")
    inspect.add_argument("--state", type=Path, required=True)
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run-case":
        run_case(args)
    elif args.action == "analyze":
        analyze(args)
    else:
        self_test()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(f"FAIL: {error}") from error
