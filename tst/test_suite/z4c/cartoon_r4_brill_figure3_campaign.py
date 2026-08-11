#!/usr/bin/env python3
"""Run the prospectively frozen off-center Brill Figure-3 analogue campaign.

This is deliberately a campaign driver, not a scheduler framework.  It binds
one input-selected AthenaK executable to the accepted IrisK 48x32 handoff,
renders exactly three 2-D Cartoon inputs, launches them in the declared order,
and emits machine-readable central-curvature and completeness evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR / "cartoon_r4_brill_figure3.athinput"
DESIGN_PATH = SCRIPT_DIR / "cartoon_r4_brill_figure3_campaign.json"
STATE_SCHEMA = "athenak_cartoon_r4_brill_figure3_campaign_v1"
ANALYSIS_SCHEMA = "athenak_cartoon_r4_brill_figure3_analysis_v1"
PAYLOAD_SENTINEL = "__IRISK_BRILL_FIGURE3_PAYLOAD__"


def load_r1() -> Any:
    path = SCRIPT_DIR / "cartoon_r1_campaign.py"
    spec = importlib.util.spec_from_file_location("cartoon_r1_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load established Cartoon campaign utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r1 = load_r1()


def require(condition: bool, message: str) -> None:
    r1.require(condition, message)


def sha256(path: Path) -> str:
    return r1.sha256(path)


def strict_load(path: Path) -> dict[str, Any]:
    return r1.strict_load(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    r1.atomic_json(path, value)


def git(root: Path, *arguments: str) -> str:
    return r1.git(root, *arguments)


def design() -> dict[str, Any]:
    result = strict_load(DESIGN_PATH)
    require(set(result) == {"schema", "qualification_claim", "paper_reference",
                            "initial_data", "execution", "resolutions",
                            "fixed_policy", "acceptance"},
            "Figure-3 design inventory changed")
    require(result["schema"] == "athenak_cartoon_r4_brill_figure3_design_v1" and
            result["qualification_claim"] == "current_gauge_analogue_only",
            "Figure-3 design schema/claim changed")
    require(result["execution"] == {
                "order": ["n128", "n192", "n256"], "mpi_ranks": 4,
                "backend": "Cuda", "one_executable": True,
                "cmake_option": "Athena_ENABLE_IRISK_INTERPOLATOR:BOOL=ON",
                "required_import_coordinate_map":
                    "CartoonIrisInterpolationCoordinates:(x1,0,x2)"},
            "prospective execution contract changed")
    expected = {"n128": (128, 128), "n192": (192, 192),
                "n256": (256, 256)}
    require(set(result["resolutions"]) == set(expected),
            "resolution pool changed")
    for name, shape in expected.items():
        record = result["resolutions"][name]
        require((record["nx1"], record["nx2"]) == shape and
                record["meshblock_nx1"] == 32 and
                record["meshblock_nx2"] == 32,
                f"resolution contract changed for {name}")
    return result


def validate_template() -> None:
    spec = design()
    blocks = r1.parse_athinput(TEMPLATE)
    require(blocks["problem"].get("pgen_name") == "z4c_irisk_xcts" and
            blocks["problem"].get("irisk_adm_spectral_file") == PAYLOAD_SENTINEL,
            "template does not select the external IrisK handoff")
    require(blocks["z4c"].get("symmetry") == "cartoon_so2" and
            blocks["z4c"].get("coordinate_map") ==
            "signed_rho_z_suppressed_y_v1" and
            blocks["mesh"].get("nx3") == "1" and
            blocks["meshblock"].get("nx3") == "1",
            "template is not collapsed signed-rho Cartoon")
    require(float(blocks["mesh"]["x1min"]) ==
            -float(blocks["mesh"]["x1max"]) and
            int(blocks["mesh"]["nx1"]) % int(blocks["meshblock"]["nx1"]) == 0 and
            (int(blocks["mesh"]["nx1"]) //
             int(blocks["meshblock"]["nx1"])) % 2 == 0,
            "template violates the internal-axis MeshBlock topology")
    policy = spec["fixed_policy"]
    require(blocks["mesh_refinement"].get("refinement") == "adaptive" and
            blocks["z4c_amr"].get("method") == policy["amr_method"] and
            float(blocks["z4c_amr"]["dchi_max"]) == policy["dchi_max"] and
            int(blocks["z4c_amr"]["max_ref_lev"]) ==
            policy["max_refinement_level"], "AMR policy changed")
    require(blocks["z4c"].get("telegraph_lapse") == "true" and
            blocks["z4c"].get("telegraph_max_K") == "true" and
            blocks["z4c"].get("lapse_oplog") == "0.0" and
            blocks["z4c"].get("lapse_harmonic") == "0.0",
            "current-gauge contract changed")
    outputs = [value for key, value in blocks.items() if key.startswith("output")]
    require([value.get("file_type") for value in outputs].count("hst") == 1 and
            [value.get("file_type") for value in outputs].count("rst") == 1,
            "history/restart inventory changed")
    require({value.get("variable") for value in outputs} >=
            {"z4c", "con", "adm", "weyl", "z4c_diag"},
            "state evidence inventory changed")


def validate_source_contract(source: Path) -> None:
    cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")
    dispatch = (source / "src/pgen/pgen.cpp").read_text(encoding="utf-8")
    importer = (source / "src/pgen/z4c_irisk_xcts.cpp").read_text(
        encoding="utf-8")
    symmetry = (source / "src/z4c/z4c_symmetry.cpp").read_text(
        encoding="utf-8")
    require("option(Athena_ENABLE_IRISK_INTERPOLATOR" in cmake,
            "IrisK interpolator feature option is absent")
    require('pgen_fun_name.compare("z4c_irisk_xcts")' in dispatch and
            "Z4cIrisXcts(pin, is_restart)" in dispatch,
            "input-selected IrisK dispatch is absent")
    require(importer.count("IrisAthenakSpectralOpen") == 1 and
            importer.count("IrisAthenakSpectralInterpolateCartesian") == 1,
            "IrisK importer boundary changed")
    require('input.problem_generator == "z4c_irisk_xcts"' in symmetry,
            "Cartoon admission for the IrisK importer is absent")
    # Mesh x2 is physical z and mesh x3 is the suppressed y direction in the
    # signed-rho map.  A Cartesian interpolator must therefore see (x1,0,x2),
    # not the ordinary Cartesian mesh tuple (x1,x2,x3).  The 01a base does not
    # yet meet this consuming production contract, so preparation must stop.
    require("cartoon_so2" in importer and
            "signed_rho_z_suppressed_y_v1" in importer and
            "CartoonIrisInterpolationCoordinates" in importer,
            "IrisK importer lacks reviewed signed-rho (x1,0,x2) coordinate mapping")


def load_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith(("#", "//")):
            continue
        require("=" in raw and ":" in raw.split("=", 1)[0],
                f"malformed CMake cache record at {path}:{number}")
        typed, value = raw.split("=", 1)
        key, _ = typed.split(":", 1)
        require(key not in values, f"duplicate CMake cache key {key}")
        values[key] = value
    return values


def validate_cache(path: Path, source: Path) -> dict[str, str]:
    require(path.is_file(), f"missing CMakeCache.txt: {path}")
    values = load_cache(path)
    require(values.get("Athena_ENABLE_IRISK_INTERPOLATOR") == "ON",
            "Athena_ENABLE_IRISK_INTERPOLATOR is not ON")
    require(values.get("Kokkos_ENABLE_CUDA") == "ON",
            "future Figure-3 executable is not CUDA-enabled")
    require(Path(values.get("CMAKE_HOME_DIRECTORY", "")).resolve() == source,
            "CMake cache is not bound to the requested source checkout")
    return values


def require_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"malformed SHA-256 for {label}")
    return value


def validate_handoff(payload: Path, sidecar: Path,
                     expected: dict[str, Any]) -> dict[str, Any]:
    require(payload.is_file(), f"missing Brill payload: {payload}")
    require(sidecar.is_file(), f"missing Brill sidecar: {sidecar}")
    require(sidecar.name == payload.name + ".manifest.json",
            "Brill sidecar is not adjacent-name bound to the payload")
    require(sha256(payload) == expected["payload_sha256"],
            "accepted Brill payload hash mismatch")
    require(sha256(sidecar) == expected["sidecar_sha256"],
            "accepted Brill sidecar hash mismatch")
    manifest = strict_load(sidecar)
    require(set(manifest) == {"schema_version", "artifact_format", "family",
                              "branch", "amplitude", "source_faithful",
                              "solver", "residuals", "export", "artifacts"},
            "Brill sidecar inventory is not exact")
    require(manifest["schema_version"] == 1 and
            manifest["artifact_format"] ==
            "IRIS_ATHENAK_SPECTRAL_ADM_PROVENANCE_V1" and
            manifest["family"] == expected["family"] and
            manifest["branch"] == expected["branch"] and
            manifest["amplitude"] == expected["amplitude"] and
            manifest["source_faithful"] is True,
            "Brill sidecar family/branch/amplitude/source contract mismatch")
    solver = manifest["solver"]
    require(isinstance(solver, dict) and set(solver) ==
            {"converged", "radial_points", "angular_points", "radial_scale",
             "adm_mass", "reciprocal_condition", "forward_error",
             "backward_error"} and solver.get("converged") is True and
            solver.get("radial_points") == expected["radial_points"] and
            solver.get("angular_points") == expected["angular_points"] and
            math.isfinite(solver.get("adm_mass", math.nan)) and
            solver["adm_mass"] > 0.0,
            "Brill solver record is not the accepted 48x32 solution")
    residuals = manifest["residuals"]
    require(isinstance(residuals, dict) and set(residuals) ==
            {"nonlinear", "hamiltonian", "momentum_max", "momentum_r",
             "momentum_theta", "maximal_lapse", "minimum_lapse",
             "oversampled_hamiltonian", "oversampled_momentum_r",
             "oversampled_momentum_theta", "oversampled_maximal_lapse"},
            "Brill residual inventory is not exact")
    export = manifest["export"]
    require(isinstance(export, dict) and set(export) ==
            {"order", "block_count", "nodes_per_block", "variable_count",
             "reconciled_traces"} and export["reconciled_traces"] is True and
            export["variable_count"] == 17,
            "Brill export inventory is not exact")
    artifacts = manifest["artifacts"]
    require(isinstance(artifacts, dict) and set(artifacts) ==
            {"source", "executable", "input", "coefficients", "adm_payload"},
            "Brill artifact inventory is not exact")
    for label in ("source", "executable", "input", "coefficients", "adm_payload"):
        require(isinstance(artifacts[label], dict),
                f"malformed Brill artifact {label}")
        require_hash(artifacts[label].get("sha256"), label)
    require(set(artifacts["source"]) == {"identifier", "path", "sha256"} and
            set(artifacts["executable"]) == {"path", "sha256"} and
            set(artifacts["input"]) == {"path", "sha256"} and
            set(artifacts["coefficients"]) ==
            {"description", "schema", "path", "sha256"} and
            set(artifacts["adm_payload"]) == {"path", "sha256"},
            "Brill nested artifact schema is not exact")
    require(artifacts["adm_payload"]["sha256"] == expected["payload_sha256"] and
            Path(artifacts["adm_payload"]["path"]).name == payload.name and
            artifacts["coefficients"]["sha256"] ==
            expected["coefficient_sha256"] and
            expected["irisk_commit"] in artifacts["source"].get("identifier", ""),
            "Brill payload/coefficient/source binding mismatch")
    for label in ("source", "executable", "input"):
        bound = Path(artifacts[label]["path"])
        require(bound.is_file() and sha256(bound) == artifacts[label]["sha256"],
                f"external Brill provenance artifact changed: {label}")
    return manifest


def validate_paper_figure(path: Path, expected: str) -> None:
    require(path.is_file() and sha256(path) == expected,
            "official Figure-3 vector PDF hash mismatch")


def render_input(name: str, payload: Path, spec: dict[str, Any]) -> str:
    require(not any(character.isspace() for character in str(payload)),
            "payload path contains whitespace unsupported by Athena input")
    blocks = r1.parse_athinput(TEMPLATE)
    resolution = spec["resolutions"][name]
    text = r1.render_athinput(blocks, {
        "job/basename": f"cartoon_r4_brill_figure3_{name}",
        "mesh/nx1": str(resolution["nx1"]),
        "mesh/nx2": str(resolution["nx2"]),
        "meshblock/nx1": str(resolution["meshblock_nx1"]),
        "meshblock/nx2": str(resolution["meshblock_nx2"]),
        "problem/irisk_adm_spectral_file": str(payload),
        "problem/constraint_summary_file":
            f"cartoon_r4_brill_figure3_{name}-constraints.dat",
    })
    rendered = r1.parse_athinput_text(text, f"rendered-{name}")
    require(rendered["mesh"]["nx3"] == "1" and
            rendered["meshblock"]["nx3"] == "1" and
            int(rendered["mesh"]["nx1"]) //
            int(rendered["meshblock"]["nx1"]) % 2 == 0,
            f"rendered {name} topology is invalid")
    return text


def contract_payload(state: dict[str, Any]) -> dict[str, Any]:
    result = {key: state[key] for key in
              ("schema", "source", "executable", "cmake_cache", "design",
               "initial_data", "paper_figure", "backend", "ranks", "inputs",
               "execution_order")}
    # Runtime status, commands, and hashes are evidence rather than prospective
    # contract fields.  Bind only the exact case inventory here so state updates
    # cannot either invalidate or expand the declared pool.
    result["case_inventory"] = list(state["cases"])
    return result


def contract_digest(state: dict[str, Any]) -> str:
    encoded = json.dumps(contract_payload(state), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_state(state: dict[str, Any]) -> None:
    require(state.get("schema") == STATE_SCHEMA and
            state.get("backend") == "Cuda" and state.get("ranks") == 4,
            "unknown or altered Figure-3 campaign state")
    require(state.get("contract_sha256") == contract_digest(state),
            "Figure-3 campaign contract changed after preparation")
    source = Path(state["source"]["path"])
    require(git(source, "rev-parse", "HEAD") == state["source"]["commit"] and
            git(source, "rev-parse", "HEAD^{tree}") == state["source"]["tree"] and
            git(source, "rev-parse", "HEAD:kokkos") ==
            state["source"]["kokkos_commit"], "bound source identity changed")
    executable = Path(state["executable"]["path"])
    require(executable.is_file() and sha256(executable) ==
            state["executable"]["sha256"], "bound executable changed")
    cache = Path(state["cmake_cache"]["path"])
    require(cache.is_file() and sha256(cache) == state["cmake_cache"]["sha256"],
            "bound CMake cache changed")
    for group in ("initial_data", "paper_figure"):
        for key in ("archive_path",):
            path = Path(state[group][key])
            require(path.is_file() and sha256(path) == state[group]["sha256"],
                    f"bound {group} changed")
    sidecar = Path(state["initial_data"]["sidecar_archive_path"])
    require(sidecar.is_file() and sha256(sidecar) ==
            state["initial_data"]["sidecar_sha256"], "bound sidecar changed")
    for record in state["inputs"].values():
        path = Path(record["path"])
        require(path.is_file() and sha256(path) == record["sha256"],
                f"bound rendered input changed: {path}")


def prepare(args: argparse.Namespace) -> None:
    validate_template()
    spec = design()
    source = args.source.resolve()
    executable = args.executable.resolve()
    cache = args.cmake_cache.resolve()
    payload = args.payload.resolve()
    sidecar = args.sidecar.resolve()
    figure = args.paper_figure.resolve()
    output = args.output.resolve()
    validate_source_contract(source)
    require(git(source, "status", "--porcelain") == "",
            "campaign preparation requires a clean source checkout")
    require(executable.is_file(), f"missing AthenaK executable: {executable}")
    validate_cache(cache, source)
    manifest = validate_handoff(payload, sidecar, spec["initial_data"])
    validate_paper_figure(figure, spec["paper_reference"]["figure_sha256"])
    require(not output.exists() or not any(output.iterdir()),
            "campaign output root already contains evidence")
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    archived_payload = provenance / payload.name
    archived_sidecar = provenance / sidecar.name
    archived_figure = provenance / "arxiv2607.10843v1_figure3.pdf"
    shutil.copyfile(payload, archived_payload)
    shutil.copyfile(sidecar, archived_sidecar)
    shutil.copyfile(figure, archived_figure)
    input_dir = output / "inputs"
    input_dir.mkdir()
    inputs: dict[str, dict[str, str]] = {}
    for name in spec["execution"]["order"]:
        path = input_dir / f"cartoon_r4_brill_figure3_{name}.athinput"
        path.write_text(render_input(name, archived_payload.resolve(), spec),
                        encoding="utf-8")
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256(path),
                        "template_sha256": sha256(TEMPLATE)}
    cases = {name: {"status": "pending"} for name in spec["execution"]["order"]}
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "source": {"path": str(source), "commit": git(source, "rev-parse", "HEAD"),
                   "tree": git(source, "rev-parse", "HEAD^{tree}"),
                   "kokkos_commit": git(source, "rev-parse", "HEAD:kokkos")},
        "executable": {"path": str(executable), "sha256": sha256(executable)},
        "cmake_cache": {"path": str(cache), "sha256": sha256(cache)},
        "design": {"path": str(DESIGN_PATH), "sha256": sha256(DESIGN_PATH)},
        "initial_data": {
            "source_path": str(payload), "archive_path": str(archived_payload.resolve()),
            "sha256": sha256(archived_payload),
            "sidecar_source_path": str(sidecar),
            "sidecar_archive_path": str(archived_sidecar.resolve()),
            "sidecar_sha256": sha256(archived_sidecar),
            "manifest": manifest,
        },
        "paper_figure": {"source_path": str(figure),
                         "archive_path": str(archived_figure.resolve()),
                         "sha256": sha256(archived_figure),
                         "machine_readable_curve_available": False},
        "backend": "Cuda", "ranks": 4, "inputs": inputs,
        "execution_order": list(spec["execution"]["order"]), "cases": cases,
    }
    state["contract_sha256"] = contract_digest(state)
    atomic_json(output / "campaign_state.json", state)


def process_record(environment: dict[str, str] | None = None) -> dict[str, Any]:
    selected = os.environ if environment is None else environment
    keys = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_NODELIST",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "KOKKOS_NUM_THREADS",
            "MPICH_GPU_SUPPORT_ENABLED", "MPICH_GPU_IPC_ENABLED",
            "MPICH_OFI_NIC_POLICY")
    return {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "environment": {key: selected.get(key) for key in keys}}


def run_case(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = strict_load(state_path)
    verify_state(state)
    require(args.case in state["cases"], f"unknown case {args.case}")
    expected_next = next((name for name in state["execution_order"]
                          if state["cases"][name]["status"] != "complete"), None)
    require(args.case == expected_next,
            f"cases must run in prospective order; next is {expected_next}")
    case = state["cases"][args.case]
    require(case["status"] == "pending", f"case {args.case} is not pending")
    root = state_path.parent
    run_dir = root / "runs" / args.case
    require(not run_dir.exists() or not any(run_dir.iterdir()),
            f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    wrapper = Path(state["source"]["path"]) / (
        "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")
    require(wrapper.is_file(), "established rank/GPU evidence wrapper is missing")
    athena = [state["executable"]["path"], "-i",
              state["inputs"][args.case]["path"], "-d", str(run_dir)]
    command = ["srun", "--nodes=1", "--ntasks=4", "--ntasks-per-node=4",
               "--cpus-per-task=8", "--gpus-per-task=1",
               "--gpu-bind=map_gpu:0,1,2,3", "--cpu-bind=cores", "--exact",
               "--kill-on-bad-exit=1", sys.executable, str(wrapper),
               "--evidence-dir", str(run_dir / "bindings"), "--require-cuda",
               "--", *athena]
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "8", "KOKKOS_NUM_THREADS": "8",
                        "MPICH_GPU_SUPPORT_ENABLED": "1",
                        "MPICH_GPU_IPC_ENABLED": "0",
                        "MPICH_OFI_NIC_POLICY": "GPU"})
    case["command"] = command
    started = time.perf_counter()
    atomic_json(run_dir / "process_start.json",
                {"schema": "athenak_process_evidence_v1",
                 **process_record(environment),
                 "command": command})
    log = run_dir / "run.log"
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=root, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    atomic_json(run_dir / "process_end.json",
                {"schema": "athenak_process_evidence_v1",
                 **process_record(environment), "exit_code": result.returncode,
                 "elapsed_seconds": time.perf_counter() - started})
    case.update({"exit_code": result.returncode, "log_path": str(log),
                 "log_sha256": sha256(log),
                 "status": "complete" if result.returncode == 0 else "failed"})
    atomic_json(state_path, state)
    require(result.returncode == 0, f"case {args.case} failed; evidence retained")


def linear_value(x: list[float], y: list[float], point: float) -> float:
    require(len(x) == len(y) and len(x) >= 2 and x[0] <= point <= x[-1],
            "interpolation point is outside the curve")
    low, high = 0, len(x) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if x[middle] <= point:
            low = middle
        else:
            high = middle
    width = x[high] - x[low]
    require(width > 0.0, "curve proper-time coordinate is not strictly increasing")
    fraction = (point - x[low]) / width
    return y[low] + fraction * (y[high] - y[low])


def curve_error(reference: dict[str, list[float]],
                trial: dict[str, list[float]], start: float, end: float,
                samples: int) -> float:
    require(start < end, "central curves have no common proper-time interval")
    points = [start + (end - start) * index / (samples - 1)
              for index in range(samples)]
    ref = [linear_value(reference["axisTau"], reference["axisKret"], point)
           for point in points]
    other = [linear_value(trial["axisTau"], trial["axisKret"], point)
             for point in points]
    scale = max(max(abs(value) for value in ref), sys.float_info.min)
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(ref, other)) /
                     samples) / scale


def evidence_hashes(run_dir: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        require(relative not in records, f"duplicate evidence path {relative}")
        records[relative] = sha256(path)
    return records


def collect_case(state: dict[str, Any], name: str, reader: Any) -> dict[str, Any]:
    root = Path(state["cases"][name]["log_path"]).parent
    histories = sorted(root.glob("*.hst"))
    require(len(histories) == 1, f"expected one history for {name}")
    history = r1.read_history(histories[0])
    require(tuple(history) == r1.HISTORY_KEYS,
            f"unexpected history inventory for {name}")
    count = len(history["axisTau"])
    require(count >= 2 and all(len(values) == count for values in history.values()),
            f"incomplete history columns for {name}")
    require(all(history["axisTau"][index] < history["axisTau"][index + 1]
                for index in range(count - 1)),
            f"proper time is not strictly increasing for {name}")
    require(all(value >= 0.0 for value in history["axisKret"]),
            f"central absolute Kretschmann is negative for {name}")
    groups = r1.final_binary_groups(root, reader)
    blocks = r1.parse_athinput(Path(state["inputs"][name]["path"]))
    root_blocks_x1 = int(blocks["mesh"]["nx1"]) // int(
        blocks["meshblock"]["nx1"])
    tree = r1.tree_summary(groups["adm"], root_blocks_x1)
    bindings = r1.binding_summary(root, "Cuda", 4)
    restart, carrier = r1.latest_restart(root)
    require(Path(state["initial_data"]["archive_path"]).name in
            Path(state["inputs"][name]["path"]).read_text(encoding="utf-8"),
            f"rendered {name} input lost the accepted payload")
    normalized_h: list[float | None] = []
    normalized_m: list[float | None] = []
    for volume, max_k, h2, m2 in zip(history["Volume"], history["max_abs_K"],
                                     history["H-norm2"], history["M-norm2"]):
        require(volume > 0.0 and h2 >= 0.0 and m2 >= 0.0,
                f"invalid constraint integral for {name}")
        if max_k == 0.0:
            normalized_h.append(None)
            normalized_m.append(None)
        else:
            normalized_h.append(math.sqrt(h2 / volume) / (max_k * max_k))
            normalized_m.append(math.sqrt(m2 / volume) / (max_k * max_k))
    return {"history": history, "normalized_h": normalized_h,
            "normalized_m": normalized_m, "tree": tree,
            "rank_bindings": bindings, "restart": {"path": str(restart),
                "sha256": sha256(restart), "central": carrier},
            "evidence_files": evidence_hashes(root)}


def write_curve_csv(path: Path, cases: dict[str, Any], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("resolution", "cycle", "coordinate_time", "proper_time",
                         "abs_kretschmann_I", "log10_abs_kretschmann_I",
                         "axis_lapse", "normalized_H", "normalized_M",
                         "meshblocks", "max_refinement_level",
                         "max_meshblocks_per_rank"))
        for name in order:
            case = cases[name]
            history = case["history"]
            for index, value in enumerate(history["axisKret"]):
                log_value: str | float = "" if value == 0.0 else math.log10(value)
                writer.writerow((name, history["cycle"][index],
                                 history["time"][index], history["axisTau"][index],
                                 value, log_value, history["axisLapse"][index],
                                 "" if case["normalized_h"][index] is None else
                                 case["normalized_h"][index],
                                 "" if case["normalized_m"][index] is None else
                                 case["normalized_m"][index],
                                 history["nmb_total"][index],
                                 history["maxRefLev"][index],
                                 history["maxNmbRank"][index]))


def analyze(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = strict_load(state_path)
    verify_state(state)
    require(all(state["cases"][name]["status"] == "complete"
                for name in state["execution_order"]),
            "all three prospective cases must complete before analysis")
    reader = r1.import_binary_reader(Path(state["source"]["path"]))
    cases = {name: collect_case(state, name, reader)
             for name in state["execution_order"]}
    spec = design()
    acceptance = spec["acceptance"]
    common_start = max(cases[name]["history"]["axisTau"][0]
                       for name in state["execution_order"])
    common_end = min(cases[name]["history"]["axisTau"][-1]
                     for name in state["execution_order"])
    samples = max(acceptance["minimum_common_samples"], 256)
    low_medium = curve_error(cases["n192"]["history"],
                             cases["n128"]["history"], common_start, common_end,
                             samples)
    medium_high = curve_error(cases["n256"]["history"],
                              cases["n192"]["history"], common_start, common_end,
                              samples)
    constraint_values = [value for name in state["execution_order"]
                         for key in ("normalized_h", "normalized_m")
                         for value in cases[name][key] if value is not None]
    require(constraint_values, "all normalized constraint records are undefined")
    constraint_max = max(constraint_values)
    failures: list[str] = []
    if common_end < acceptance["minimum_common_proper_time"]:
        failures.append("insufficient_common_proper_time")
    if medium_high > acceptance["fine_medium_normalized_l2_max"]:
        failures.append("fine_medium_curve_difference")
    if medium_high > acceptance["required_error_contraction"] * low_medium:
        failures.append("central_curve_error_did_not_contract")
    if constraint_max > acceptance["normalized_constraint_linf_max"]:
        failures.append("normalized_constraint_limit")
    curve_path = state_path.parent / "plot_data" / "figure3_current_gauge.csv"
    write_curve_csv(curve_path, cases, state["execution_order"])
    comparison = {
        "claim": "current_gauge_analogue_only",
        "paper_curve_machine_readable": False,
        "paper_figure_sha256": state["paper_figure"]["sha256"],
        "warning": ("No published machine-readable curve is available; the paper "
                    "PDF is retained but is not digitized or represented as data."),
        "common_proper_time_start": common_start,
        "common_proper_time_end": common_end, "sample_count": samples,
        "low_medium_normalized_l2": low_medium,
        "medium_high_normalized_l2": medium_high,
        "normalized_constraint_linf": constraint_max,
    }
    report = {"schema": ANALYSIS_SCHEMA,
              "qualification_claim": "current_gauge_analogue_only",
              "verdict": "passed" if not failures else "failed",
              "failures": failures, "provenance": contract_payload(state),
              "comparison": comparison, "cases": cases,
              "plot_data": {"path": str(curve_path),
                            "sha256": sha256(curve_path)}}
    output = state_path.parent / "figure3_analysis.json"
    atomic_json(output, report)
    state["analysis"] = {"path": str(output), "sha256": sha256(output),
                         "verdict": report["verdict"],
                         "plot_data_sha256": sha256(curve_path)}
    atomic_json(state_path, state)
    require(not failures, "Figure-3 analogue gates failed: " + ", ".join(failures))


def synthetic_manifest(payload: Path, source: Path, executable: Path,
                       input_path: Path, payload_hash: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_format": "IRIS_ATHENAK_SPECTRAL_ADM_PROVENANCE_V1",
        "family": "brill_gaussian", "branch": "unique_time_symmetric",
        "amplitude": -0.047, "source_faithful": True,
        "solver": {"converged": True, "radial_points": 48,
                   "angular_points": 32, "radial_scale": 3.0,
                   "adm_mass": 2.6, "reciprocal_condition": 1.0e-10,
                   "forward_error": 1.0e-10, "backward_error": 1.0e-16},
        "residuals": {"nonlinear": None, "hamiltonian": 1.0e-12,
                      "momentum_max": None, "momentum_r": None,
                      "momentum_theta": None, "maximal_lapse": None,
                      "minimum_lapse": None, "oversampled_hamiltonian": None,
                      "oversampled_momentum_r": None,
                      "oversampled_momentum_theta": None,
                      "oversampled_maximal_lapse": None},
        "export": {"order": 4, "block_count": 25, "nodes_per_block": 64,
                   "variable_count": 17, "reconciled_traces": True},
        "artifacts": {
            "source": {"identifier":
                       "irisk:2a069fd0497ef4352d4ecd28c6879ac47b84a5a1",
                       "path": str(source), "sha256": sha256(source)},
            "executable": {"path": str(executable),
                           "sha256": sha256(executable)},
            "input": {"path": str(input_path), "sha256": sha256(input_path)},
            "coefficients": {
                "description":
                    "psi_minus_one_rational_sine_even_radial_major_angular_fast",
                "schema": "IRIS_BRILL_WAVE_STATE_V1", "path": None,
                "sha256": "4" * 64},
            "adm_payload": {"path": str(payload), "sha256": payload_hash},
        },
    }


def expect_failure(action: Any, token: str) -> None:
    try:
        action()
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        require(token in str(error), f"wrong failure for {token}: {error}")
    else:
        raise RuntimeError(f"synthetic {token} case was accepted")


def self_test() -> None:
    validate_template()
    spec = design()
    rendered = r1.parse_athinput_text(
        render_input("n192", Path("/immutable/brill.adm_spectral"), spec),
        "synthetic-render")
    require(rendered["mesh"]["nx1"] == "192" and
            rendered["mesh"]["nx2"] == "192",
            "runtime resolution selection failed")
    invalid = r1.render_athinput(rendered, {"mesh/nx3": "2"})
    expect_failure(lambda: require(
        r1.parse_athinput_text(invalid, "invalid-topology")["mesh"]["nx3"] == "1",
        "topology"), "topology")
    with tempfile.TemporaryDirectory(prefix="cartoon-r4-brill-") as directory:
        root = Path(directory)
        payload = root / "brill.adm_spectral"
        sidecar = root / "brill.adm_spectral.manifest.json"
        source = root / "source.md"
        executable = root / "iris"
        input_path = root / "input.txt"
        for path, content in ((payload, b"payload"), (source, b"source"),
                              (executable, b"exe"), (input_path, b"input")):
            path.write_bytes(content)
        expected = dict(spec["initial_data"])
        expected.update({"payload_sha256": sha256(payload),
                         "sidecar_sha256": "", "coefficient_sha256": "4" * 64})
        manifest = synthetic_manifest(payload, source, executable, input_path,
                                      expected["payload_sha256"])
        sidecar.write_text(json.dumps(manifest, sort_keys=True) + "\n",
                           encoding="utf-8")
        expected["sidecar_sha256"] = sha256(sidecar)
        validate_handoff(payload, sidecar, expected)
        missing = root / "missing.adm_spectral"
        expect_failure(lambda: validate_handoff(missing, sidecar, expected),
                       "missing Brill payload")
        payload.write_bytes(b"changed")
        expect_failure(lambda: validate_handoff(payload, sidecar, expected),
                       "payload hash mismatch")
        payload.write_bytes(b"payload")
        sidecar.write_text('{"amplitude": NaN}\n', encoding="utf-8")
        bad = dict(expected)
        bad["sidecar_sha256"] = sha256(sidecar)
        expect_failure(lambda: validate_handoff(payload, sidecar, bad),
                       "nonfinite JSON")
        sidecar.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
        bad["sidecar_sha256"] = sha256(sidecar)
        expect_failure(lambda: validate_handoff(payload, sidecar, bad),
                       "duplicate JSON key")
        sidecar.write_text('{broken\n', encoding="utf-8")
        bad["sidecar_sha256"] = sha256(sidecar)
        expect_failure(lambda: validate_handoff(payload, sidecar, bad),
                       "invalid strict JSON")
        cache = root / "CMakeCache.txt"
        cache.write_text("A:BOOL=ON\nA:BOOL=OFF\n", encoding="utf-8")
        expect_failure(lambda: load_cache(cache), "duplicate CMake cache key")
    synthetic_curve = {"axisTau": [0.0, 1.0, 2.0],
                       "axisKret": [1.0, 2.0, 1.0]}
    require(curve_error(synthetic_curve, synthetic_curve, 0.0, 2.0, 5) == 0.0,
            "curve comparison oracle failed")
    synthetic_state = {
        "schema": STATE_SCHEMA, "source": {}, "executable": {},
        "cmake_cache": {}, "design": {}, "initial_data": {},
        "paper_figure": {}, "backend": "Cuda", "ranks": 4,
        "inputs": {}, "execution_order": ["n128"],
        "cases": {"n128": {"status": "pending"}},
    }
    before = contract_digest(synthetic_state)
    synthetic_state["cases"]["n128"]["status"] = "complete"
    require(before == contract_digest(synthetic_state),
            "runtime evidence incorrectly changes the prospective contract")
    synthetic_state["cases"]["extra"] = {"status": "pending"}
    require(before != contract_digest(synthetic_state),
            "prospective contract did not bind the exact case inventory")
    print("Cartoon R4 Brill Figure-3 campaign tooling self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    make = subparsers.add_parser("prepare")
    make.add_argument("--source", required=True, type=Path)
    make.add_argument("--executable", required=True, type=Path)
    make.add_argument("--cmake-cache", required=True, type=Path)
    make.add_argument("--payload", required=True, type=Path)
    make.add_argument("--sidecar", required=True, type=Path)
    make.add_argument("--paper-figure", required=True, type=Path)
    make.add_argument("--output", required=True, type=Path)
    run = subparsers.add_parser("run-case")
    run.add_argument("--state", required=True, type=Path)
    run.add_argument("--case", required=True)
    inspect = subparsers.add_parser("analyze")
    inspect.add_argument("--state", required=True, type=Path)
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
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
