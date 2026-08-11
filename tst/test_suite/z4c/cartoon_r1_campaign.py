#!/usr/bin/env python3
"""Prepare, run, and analyze the focused Cartoon R1 evolution campaign.

The script uses one input-selected AthenaK executable.  It is intentionally
limited to the fixed-grid, mirrored-AMR, derefinement, and restart-equivalence
cases declared below; it is not a general scheduler or test framework.
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
from typing import Any


SCHEMA = "athenak_cartoon_r1_campaign_v1"
ANALYSIS_SCHEMA = "athenak_cartoon_r1_analysis_v1"
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATES = {
    "fixed": SCRIPT_DIR / "cartoon_r1_fixed.athinput",
    "amr": SCRIPT_DIR / "cartoon_r1_amr.athinput",
    "derefine": SCRIPT_DIR / "cartoon_r1_derefine.athinput",
}
FIELD_GROUPS = {
    "z4c": "z4c_chi",
    "constraints": "con_C",
    "adm": "adm_gxx",
    "weyl": "weyl_rpsi4",
    "curvature": "z4c_Kretschmann",
}
CENTRAL_KEYS = (
    "central_initialized",
    "central_proper_time",
    "central_previous_lapse",
    "central_constraint_norm",
    "central_abs_kretschmann",
    "central_sample_gid",
    "central_sample_level",
    "central_last_cycle",
    "central_last_time",
)
RESTART_KEYS = (
    "carrier_schema", "symmetry", "coordinate_map", "symmetry_schema",
    "requested_spatial_order", "effective_spatial_order", "stencil_width",
    "mesh_nx1", "mesh_nx2", "mesh_nx3", "meshblock_nx1", "meshblock_nx2",
    "meshblock_nx3", "central_schema", *CENTRAL_KEYS, "fastflow_schema",
    "fastflow_coefficient_count", "fastflow_coefficients", "fastflow_surface_mode",
    "fastflow_selected_branch", "fastflow_center_count", "fastflow_center_z0",
    "fastflow_center_z1", "fastflow_status", "fastflow_failure_code",
    "fastflow_last_search_cycle", "fastflow_last_search_time", "fastflow_converged",
)
# HistoryOutput writes labels with %.10s; the in-memory `Theta-norm2` label is
# therefore intentionally `Theta-norm` in the exact on-disk inventory.
HISTORY_KEYS = (
    "time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2", "Mx-norm2",
    "My-norm2", "Mz-norm2", "Theta-norm", "Volume", "max_abs_K", "nmb_total",
    "maxAbsKret", "maxRefLev", "maxNmbRank", "ahStatus", "ahLastCyc", "cycle",
    "axisLapse", "axisTau", "axisKret",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(token: str) -> None:
    raise ValueError(f"nonfinite JSON token {token!r}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def require_finite_json(value: Any, where: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"nonfinite value at {where}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{where}[{index}]")
        return
    require(isinstance(value, dict), f"unsupported JSON value at {where}")
    for key, item in value.items():
        require(isinstance(key, str), f"non-string JSON key at {where}")
        require_finite_json(item, f"{where}.{key}")


def strict_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid strict JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    require_finite_json(value)
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    require_finite_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def parse_athinput_text(text: str, label: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            name = line[1:-1].strip()
            require(name and name not in blocks,
                    f"duplicate/malformed block at {label}:{number}")
            current = {}
            blocks[name] = current
            continue
        require(current is not None and "=" in line,
                f"malformed input record at {label}:{number}")
        key, value = (part.strip() for part in line.split("=", 1))
        require(key and value and key not in current,
                f"duplicate/malformed key at {label}:{number}")
        current[key] = value
    require(blocks, f"empty Athena input {label}")
    return blocks


def parse_athinput(path: Path) -> dict[str, dict[str, str]]:
    return parse_athinput_text(path.read_text(encoding="utf-8"), str(path))


def render_athinput(blocks: dict[str, dict[str, str]],
                    overrides: dict[str, str]) -> str:
    rendered = {block: dict(values) for block, values in blocks.items()}
    for qualified, value in overrides.items():
        require("/" in qualified, f"unqualified input override {qualified!r}")
        block, key = qualified.split("/", 1)
        require(block in rendered and key in rendered[block],
                f"override does not target an existing key: {qualified}")
        rendered[block][key] = value
    lines: list[str] = []
    for block, values in rendered.items():
        lines.append(f"<{block}>")
        lines.extend(f"{key} = {value}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


def validate_inputs() -> None:
    fixed = parse_athinput(TEMPLATES["fixed"])
    amr = parse_athinput(TEMPLATES["amr"])
    overlay = parse_athinput(TEMPLATES["derefine"])
    for name, blocks in (("fixed", fixed), ("amr", amr)):
        require(blocks["problem"].get("pgen_name") == "kerr_puncture",
                f"{name} input does not select kerr_puncture")
        require(blocks["z4c"].get("symmetry") == "cartoon_so2" and
                blocks["z4c"].get("coordinate_map") ==
                "signed_rho_z_suppressed_y_v1",
                f"{name} input does not select signed-rho Cartoon")
        require(blocks["mesh"].get("nx3") == "1" and
                int(blocks["mesh"]["nx1"]) % 2 == 0 and
                float(blocks["mesh"]["x1min"]) ==
                -float(blocks["mesh"]["x1max"]),
                f"{name} input violates collapsed signed-rho topology")
        require(blocks["z4c"].get("excise_chi") == "0.0",
                f"{name} volume oracle requires unexcised integral cells")
        outputs = [values for block, values in blocks.items()
                   if block.startswith("output")]
        types = [values.get("file_type") for values in outputs]
        require(types.count("hst") == 1 and types.count("rst") == 1,
                f"{name} input requires exactly one history and restart output")
        require({"z4c", "con", "adm", "weyl", "z4c_diag"}.issubset(
                    {values.get("variable") for values in outputs}),
                f"{name} input omits a state-equality output family")
        require(all(values.get("single_file_per_rank") == "true"
                    for values in outputs if values.get("file_type") == "bin"),
                f"{name} input cannot prove rank ownership")
    require(fixed["mesh_refinement"].get("refinement") == "none",
            "fixed input unexpectedly enables AMR")
    require(amr["mesh_refinement"].get("refinement") == "adaptive" and
            amr["mesh_refinement"].get("refinement_interval") == "1" and
            amr["z4c_amr"].get("method") == "dchi_max",
            "AMR input does not exercise per-cycle production refinement")
    require(float(overlay["z4c_amr"]["dchi_max"]) >= 1.0e9,
            "derefinement overlay does not force the structural gate")


def validate_source_contract(source: Path) -> None:
    """Bind every non-generic input key/value to its production consumer."""
    pgen = (source / "src/pgen/pgen.cpp").read_text(encoding="utf-8")
    kerr = (source / "src/pgen/z4c/kerr_puncture.cpp").read_text(encoding="utf-8")
    symmetry = (source / "src/z4c/z4c_symmetry.cpp").read_text(encoding="utf-8")
    amr = (source / "src/z4c/z4c_amr.cpp").read_text(encoding="utf-8")
    refinement = (source / "src/mesh/mesh_refinement.cpp").read_text(encoding="utf-8")
    mesh = (source / "src/mesh/mesh.cpp").read_text(encoding="utf-8")
    tree = (source / "src/mesh/build_tree.cpp").read_text(encoding="utf-8")
    outputs = (source / "src/outputs/outputs.cpp").read_text(encoding="utf-8")
    variables = (source / "src/outputs/basetype_output.cpp").read_text(
        encoding="utf-8")
    history = (source / "src/outputs/history.cpp").read_text(encoding="utf-8")
    restart = (source / "src/z4c/z4c_restart.cpp").read_text(encoding="utf-8")
    z4c = (source / "src/z4c/z4c.cpp").read_text(encoding="utf-8")
    binary = (source / "src/outputs/binary.cpp").read_text(encoding="utf-8")
    restart_output = (source / "src/outputs/restart.cpp").read_text(encoding="utf-8")
    binary_reader = (source / "vis/python/bin_convert.py").read_text(encoding="utf-8")
    rank_wrapper = (source / "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py").read_text(
        encoding="utf-8")
    for marker, text in (
        ('pgen_fun_name.compare("kerr_puncture") == 0', pgen),
        ("ConfigureKerrPuncture", pgen),
        ('GetOrAddReal("problem", "M"', kerr),
        ('GetOrAddReal("problem", "chi"', kerr),
        ('GetOrAddReal("problem", "z_h"', kerr),
        ('GetOrAddString("problem", "initial_gauge"', kerr),
        ('input.problem_generator == "kerr_puncture"', symmetry),
        ('GetOrAddString("z4c_amr", "method"', amr),
        ('GetOrAddReal("z4c_amr", "dchi_max"', amr),
        ('GetOrAddInteger("z4c_amr", "max_ref_lev"', amr),
        ('GetOrAddString("mesh_refinement","refinement","none")', mesh),
        ('GetOrAddReal("mesh_refinement", "refinement_interval"', refinement),
        ('DoesParameterExist("mesh_refinement", "max_nmb_per_rank"', tree),
        ('GetOrAddInteger("mesh_refinement", "num_levels"', tree),
        ('GetInteger(opar.block_name,"dcycle")', outputs),
        ('GetOrAddBoolean(opar.block_name,\n          "single_file_per_rank"', outputs),
        ('file_type.compare("hst")', outputs),
        ('file_type.compare("bin")', outputs),
        ('file_type.compare("rst")', outputs),
        ('variable.compare("z4c")', variables),
        ('variable.compare("con")', variables),
        ('variable.compare("adm")', variables),
        ('variable.compare("weyl")', variables),
        ('variable.compare("z4c_diag")', variables),
        ('pdata->label[0] = "C-norm2"', history),
        ('pdata->label[central_lapse_index] = "axisLapse"', history),
        ('pdata->label[central_proper_time_index] = "axisTau"', history),
        ('pdata->label[central_kretschmann_index] = "axisKret"', history),
        ('GetOrAddBoolean("z4c", "history_kretschmann"', z4c),
        ('GetOrAddReal("z4c", "excise_chi"', z4c),
        ('"central_constraint_norm"', restart),
        ('"central_abs_kretschmann"', restart),
        ('mkdir("bin",0775)', binary),
        ('"bin/rank_%08d/"', binary),
        ('mkdir("rst",0775)', restart_output),
        ('fname = std::string("rst/") + out_params.file_basename', restart_output),
        ('fname.assign(out_params.file_basename)', history),
        ('mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))', binary_reader),
        ('"CUDA_VISIBLE_DEVICES is unset"', rank_wrapper),
        ('"selected_uuid": uuid', rank_wrapper),
        ('os.O_WRONLY | os.O_CREAT | os.O_EXCL', rank_wrapper),
    ):
        require(marker in text, f"Athena input/evidence schema marker disappeared: {marker}")
    require(binary.index("(loc.lx1)") < binary.index("(loc.lx2)") <
            binary.index("(loc.lx3)") < binary.index("(loc.level-pm->root_level)"),
            "binary logical tuple order changed")
    start = restart.index("for (const char *key : {")
    stop = restart.index("}) {", start)
    production_restart_keys = tuple(re.findall(r'"([a-z0-9_]+)"',
                                                restart[start:stop]))
    require(production_restart_keys == RESTART_KEYS,
            "campaign restart inventory differs from production RequireKeys")


def contract_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state[key] for key in (
        "schema", "source", "executable", "backend", "ranks", "inputs", "cases"
    ) if key != "cases"} | {
        "cases": {
            name: {key: value for key, value in case.items() if key not in {
                "status", "exit_code", "log_path", "log_sha256", "command",
                "restart_path", "restart_sha256"
            }}
            for name, case in state["cases"].items()
        }
    }


def contract_digest(state: dict[str, Any]) -> str:
    payload = json.dumps(
        contract_payload(state), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_state(state: dict[str, Any]) -> None:
    allowed_top = {"schema", "source", "executable", "backend", "ranks", "inputs",
                   "cases", "contract_sha256", "analysis"}
    require(set(state).issubset(allowed_top) and
            allowed_top - {"analysis"} == set(state) - ({"analysis"} & set(state)),
            "campaign-state field inventory is malformed")
    require(state.get("schema") == SCHEMA, "unknown R1 campaign-state schema")
    require(set(state["source"]) == {"path", "commit", "tree", "kokkos_commit"} and
            set(state["executable"]) == {"path", "sha256"},
            "campaign provenance inventory is malformed")
    require(set(state["inputs"]) == set(TEMPLATES) and
            all(set(record) == {"path", "sha256", "template_sha256"}
                for record in state["inputs"].values()),
            "campaign input inventory is malformed")
    require(set(state["cases"]) == {"fixed_fresh", "fixed_restart", "amr_fresh",
                                     "amr_restart_pre", "amr_restart_post",
                                     "amr_derefine"},
            "campaign case inventory is malformed")
    common_dynamic = {"status", "exit_code", "log_path", "log_sha256", "command",
                      "restart_path", "restart_sha256"}
    for name, case in state["cases"].items():
        base = ({"mode", "input"} if case.get("mode") == "fresh" else
                {"mode", "restart_from", "checkpoint_cycle"})
        if name == "amr_derefine":
            base.add("overlay")
        require(base | {"status"} <= set(case) <= base | common_dynamic and
                case["status"] in {"pending", "complete", "failed"},
                f"campaign case metadata is malformed: {name}")
    if "analysis" in state:
        require(set(state["analysis"]) == {"path", "sha256", "verdict"},
                "campaign analysis metadata is malformed")
    require(state.get("contract_sha256") == contract_digest(state),
            "campaign contract changed after preparation")
    executable = Path(state["executable"]["path"])
    require(executable.is_file() and sha256(executable) ==
            state["executable"]["sha256"], "bound executable changed")
    source = Path(state["source"]["path"])
    require(git(source, "rev-parse", "HEAD") == state["source"]["commit"] and
            git(source, "rev-parse", "HEAD^{tree}") == state["source"]["tree"] and
            git(source, "rev-parse", "HEAD:kokkos") ==
            state["source"]["kokkos_commit"], "bound source identity changed")
    for record in state["inputs"].values():
        path = Path(record["path"])
        require(path.is_file() and sha256(path) == record["sha256"],
                f"bound input changed: {path}")


def prepare(args: argparse.Namespace) -> None:
    validate_inputs()
    source = args.source.resolve()
    validate_source_contract(source)
    executable = args.executable.resolve()
    output = args.output.resolve()
    require(executable.is_file(), f"missing Athena executable: {executable}")
    require(args.ranks == 4, "R1 qualification is fixed to four MPI ranks")
    require(git(source, "status", "--porcelain") == "",
            "campaign preparation requires a clean source checkout")
    require(not output.exists() or not any(output.iterdir()),
            "campaign output root already contains evidence")
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, str]] = {}
    for name in ("fixed", "amr", "derefine"):
        parsed = parse_athinput(TEMPLATES[name])
        rendered = render_athinput(parsed, {})
        path = input_dir / TEMPLATES[name].name
        path.write_text(rendered, encoding="utf-8")
        generated[name] = {"path": str(path.resolve()), "sha256": sha256(path),
                           "template_sha256": sha256(TEMPLATES[name])}
    cases = {
        "fixed_fresh": {"mode": "fresh", "input": "fixed", "status": "pending"},
        "fixed_restart": {"mode": "restart", "restart_from": "fixed_fresh",
                          "checkpoint_cycle": 2, "status": "pending"},
        "amr_fresh": {"mode": "fresh", "input": "amr", "status": "pending"},
        "amr_restart_pre": {"mode": "restart", "restart_from": "amr_fresh",
                            "checkpoint_cycle": 0, "status": "pending"},
        "amr_restart_post": {"mode": "restart", "restart_from": "amr_fresh",
                             "checkpoint_cycle": 2, "status": "pending"},
        "amr_derefine": {"mode": "restart", "restart_from": "amr_fresh",
                         "checkpoint_cycle": 2, "overlay": "derefine",
                         "status": "pending"},
    }
    state: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {"path": str(source), "commit": git(source, "rev-parse", "HEAD"),
                   "tree": git(source, "rev-parse", "HEAD^{tree}"),
                   "kokkos_commit": git(source, "rev-parse", "HEAD:kokkos")},
        "executable": {"path": str(executable), "sha256": sha256(executable)},
        "backend": args.backend,
        "ranks": args.ranks,
        "inputs": generated,
        "cases": cases,
    }
    state["contract_sha256"] = contract_digest(state)
    atomic_json(output / "campaign_state.json", state)


def restart_metadata(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    marker = b"<par_end>\n"
    require(marker in raw, f"restart lacks <par_end>: {path}")
    text = raw[:raw.index(marker)].decode("utf-8", errors="strict")
    blocks = parse_athinput_text(text, f"{path}:ParameterDump")
    require("z4c_restart" in blocks, f"restart lacks z4c_restart carrier: {path}")
    carrier = blocks["z4c_restart"]
    require(set(carrier) == set(RESTART_KEYS),
            f"restart carrier inventory is not exact: {path}")
    require(carrier["symmetry"] == "cartoon_so2" and
            carrier["coordinate_map"] == "signed_rho_z_suppressed_y_v1" and
            carrier["symmetry_schema"] == "1" and carrier["central_schema"] == "2",
            f"restart symmetry/schema metadata changed: {path}")
    for key in ("central_proper_time", "central_previous_lapse",
                "central_constraint_norm", "central_abs_kretschmann",
                "central_last_time", "fastflow_center_z0", "fastflow_center_z1",
                "fastflow_last_search_time"):
        require(math.isfinite(float(carrier[key])),
                f"restart contains nonfinite {key}: {path}")
    require(carrier["central_initialized"] in {"1", "true"},
            f"restart central state is not initialized: {path}")
    for key in ("carrier_schema", "symmetry_schema", "requested_spatial_order",
                "effective_spatial_order", "stencil_width", "mesh_nx1", "mesh_nx2",
                "mesh_nx3", "meshblock_nx1", "meshblock_nx2", "meshblock_nx3",
                "central_schema", "central_sample_gid", "central_sample_level",
                "central_last_cycle", "fastflow_schema", "fastflow_coefficient_count",
                "fastflow_center_count", "fastflow_last_search_cycle"):
        try:
            int(carrier[key])
        except ValueError as error:
            raise RuntimeError(f"restart contains malformed integer {key}: {path}") from error
    require(carrier["fastflow_converged"] in {"0", "1", "false", "true"},
            f"restart contains malformed fastflow_converged: {path}")
    return carrier


def select_restart(run_dir: Path, cycle: int) -> Path:
    matches: list[Path] = []
    for path in sorted(run_dir.glob("rst/*.rst")):
        if int(restart_metadata(path)["central_last_cycle"]) == cycle:
            matches.append(path)
    require(len(matches) == 1,
            f"expected exactly one cycle-{cycle} restart in {run_dir}, got {matches}")
    return matches[0]


def run_case(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = strict_load(state_path)
    verify_state(state)
    require(args.case in state["cases"], f"unknown campaign case {args.case}")
    case = state["cases"][args.case]
    if case["status"] == "complete":
        require(args.resume, f"case {args.case} is already complete")
        log = Path(case["log_path"])
        require(log.is_file() and sha256(log) == case["log_sha256"],
                f"completed case evidence changed: {args.case}")
        return
    require(case["status"] == "pending",
            f"case {args.case} has terminal status {case['status']}")
    root = state_path.parent
    run_dir = root / "runs" / args.case
    require(not run_dir.exists() or not any(run_dir.iterdir()),
            f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    executable = state["executable"]["path"]
    athena: list[str]
    if case["mode"] == "fresh":
        athena = [executable, "-i", state["inputs"][case["input"]]["path"],
                  "-d", str(run_dir)]
    else:
        dependency = state["cases"][case["restart_from"]]
        require(dependency["status"] == "complete",
                f"restart dependency is incomplete: {case['restart_from']}")
        restart = select_restart(root / "runs" / case["restart_from"],
                                 int(case["checkpoint_cycle"]))
        athena = [executable, "-r", str(restart), "-d", str(run_dir)]
        if "overlay" in case:
            athena.extend(["-i", state["inputs"][case["overlay"]]["path"]])
        case["restart_path"] = str(restart)
        case["restart_sha256"] = sha256(restart)
    rank_wrapper = (Path(state["source"]["path"]) /
                    "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")
    require(rank_wrapper.is_file(), "established rank-binding wrapper is missing")
    wrapper = [sys.executable, str(rank_wrapper),
               "--evidence-dir", str(run_dir / "bindings")]
    if state["backend"] == "Cuda":
        wrapper.append("--require-cuda")
        command = [
            "srun", "--nodes=1", "--ntasks=4", "--ntasks-per-node=4",
            "--cpus-per-task=8", "--gpus-per-task=1",
            "--gpu-bind=map_gpu:0,1,2,3", "--cpu-bind=cores", "--exact",
            "--kill-on-bad-exit=1", *wrapper, "--", *athena,
        ]
    else:
        command = [args.mpiexec, "-n", str(state["ranks"]),
                   *wrapper, "--", *athena]
    case["command"] = command
    log = run_dir / "run.log"
    environment = os.environ.copy()
    if state["backend"] == "Cuda":
        environment.update({"OMP_NUM_THREADS": "8", "KOKKOS_NUM_THREADS": "8",
                            "MPICH_GPU_SUPPORT_ENABLED": "1",
                            "MPICH_GPU_IPC_ENABLED": "0",
                            "MPICH_OFI_NIC_POLICY": "GPU"})
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=root, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    case["exit_code"] = result.returncode
    case["log_path"] = str(log)
    case["log_sha256"] = sha256(log)
    case["status"] = "complete" if result.returncode == 0 else "failed"
    atomic_json(state_path, state)
    require(result.returncode == 0, f"case {args.case} failed; evidence preserved")


def verify_restart_lineage(state: dict[str, Any], name: str) -> None:
    case = state["cases"][name]
    if case["mode"] == "fresh":
        return
    required = {"restart_path", "restart_sha256", "restart_from", "checkpoint_cycle"}
    require(required.issubset(case), f"restart lineage is incomplete for {name}")
    path = Path(case["restart_path"])
    parent = Path(state["cases"][case["restart_from"]]["log_path"]).parent / "rst"
    require(path.is_file() and path.parent.resolve() == parent.resolve() and
            sha256(path) == case["restart_sha256"],
            f"restart lineage hash/path changed for {name}")
    metadata = restart_metadata(path)
    require(int(metadata["central_last_cycle"]) == int(case["checkpoint_cycle"]),
            f"restart lineage cycle changed for {name}")


def read_history(path: Path) -> dict[str, list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [index for index, line in enumerate(lines)
               if line == "# Athena++ history data"]
    require(headers, f"history header is missing: {path}")
    header = headers[-1] + 1
    names = re.findall(r"\[\d+\]=(\S+)", lines[header])
    require(names and len(names) == len(set(names)),
            f"history inventory is malformed: {path}")
    result = {name: [] for name in names}
    for line in lines[header + 1:]:
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        require(len(values) == len(names), f"malformed history row: {path}")
        for name, token in zip(names, values):
            value = float(token)
            require(math.isfinite(value), f"nonfinite {name} in {path}")
            result[name].append(value)
    require(result[names[0]], f"history has no records: {path}")
    return result


def import_binary_reader(source: Path):
    sys.path.insert(0, str(source / "vis/python"))
    try:
        import bin_convert  # pylint: disable=import-outside-toplevel
    except (ImportError, OSError) as error:
        raise RuntimeError("cannot import repository binary reader") from error
    return bin_convert.read_binary


def final_binary_groups(run_dir: Path, reader) -> dict[str, list[tuple[int, Any]]]:
    files = sorted(run_dir.glob("bin/rank_*/*.bin"))
    require(files, f"per-rank binary evidence is missing: {run_dir}")
    loaded = [(path, reader(path)) for path in files]
    final_cycle = max(int(data["cycle"]) for _, data in loaded)
    groups: dict[str, list[tuple[int, Any]]] = {name: [] for name in FIELD_GROUPS}
    for path, data in loaded:
        if int(data["cycle"]) != final_cycle:
            continue
        rank_match = re.fullmatch(r"rank_(\d{8})", path.parent.name)
        require(rank_match is not None, f"malformed rank directory: {path.parent}")
        rank = int(rank_match.group(1))
        matches = [name for name, marker in FIELD_GROUPS.items()
                   if marker in data["var_names"]]
        require(len(matches) == 1, f"ambiguous binary variable inventory: {path}")
        groups[matches[0]].append((rank, data))
    for name, records in groups.items():
        require(records, f"final binary group {name} is missing: {run_dir}")
    return groups


def block_map(records: list[tuple[int, Any]]) -> dict[tuple[int, ...], tuple[int, Any, int]]:
    blocks: dict[tuple[int, ...], tuple[int, Any, int]] = {}
    for rank, data in records:
        for index, logical in enumerate(data["mb_logical"]):
            key = tuple(int(value) for value in logical)
            require(key not in blocks, f"duplicate logical MeshBlock {key}")
            blocks[key] = (rank, data, index)
    return blocks


def snapshot_digest(records: list[tuple[int, Any]]) -> str:
    digest = hashlib.sha256()
    blocks = block_map(records)
    for key in sorted(blocks):
        _, data, index = blocks[key]
        digest.update(repr(key).encode("ascii"))
        digest.update(data["mb_geometry"][index].tobytes())
        for name in sorted(data["var_names"]):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(data["mb_data"][name][index].tobytes())
    return digest.hexdigest()


def summarize_tree(block_owners: dict[tuple[int, ...], int],
                   root_blocks_x1: int) -> dict[str, Any]:
    blocks = block_owners
    locations = set(blocks)
    levels = [key[3] for key in locations]
    for lx1, lx2, lx3, level in locations:
        mirror = (root_blocks_x1 * (2 ** level) - 1 - lx1, lx2, lx3, level)
        require(mirror in locations, f"asymmetric signed-rho AMR tree at {(lx1, lx2, level)}")
    maximum = max(levels)
    radial = root_blocks_x1 * (2 ** maximum)
    axis_adjacent = any(level == maximum and lx1 in {radial // 2 - 1, radial // 2}
                        for lx1, _, _, level in locations)
    ownership: dict[str, int] = {}
    for rank in blocks.values():
        ownership[str(rank)] = ownership.get(str(rank), 0) + 1
    return {"locations": [list(key) for key in sorted(locations)],
            "max_level": maximum, "axis_adjacent_refinement": axis_adjacent,
            "ownership": ownership}


def tree_summary(records: list[tuple[int, Any]], root_blocks_x1: int) -> dict[str, Any]:
    blocks = block_map(records)
    return summarize_tree({key: value[0] for key, value in blocks.items()},
                          root_blocks_x1)


def cylindrical_volume(adm_records: list[tuple[int, Any]]) -> float:
    import numpy as np  # pylint: disable=import-outside-toplevel
    total = 0.0
    for _, data in adm_records:
        required = ("adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz",
                    "adm_gzz")
        require(all(name in data["mb_data"] for name in required),
                "ADM output lacks spatial metric components")
        for block, geometry in enumerate(data["mb_geometry"]):
            arrays = [data["mb_data"][name][block].astype(np.float64)
                      for name in required]
            gxx, gxy, gxz, gyy, gyz, gzz = arrays
            determinant = (-gxz * gxz * gyy + 2.0 * gxy * gxz * gyz -
                           gyz * gyz * gxx - gxy * gxy * gzz + gxx * gyy * gzz)
            require(bool(np.isfinite(determinant).all()) and
                    bool((determinant > 0.0).all()),
                    "ADM volume oracle found invalid spatial determinant")
            nx1 = gxx.shape[-1]
            nx2 = gxx.shape[-2]
            dx1 = (geometry[1] - geometry[0]) / nx1
            dx2 = (geometry[3] - geometry[2]) / nx2
            rho = geometry[0] + (np.arange(nx1, dtype=np.float64) + 0.5) * dx1
            weights = np.where(rho > 0.0, 2.0 * math.pi * rho * dx1 * dx2, 0.0)
            total += float(np.sum(np.sqrt(determinant) * weights[None, None, :]))
    require(math.isfinite(total) and total > 0.0, "invalid cylindrical volume oracle")
    return total


def binding_summary(run_dir: Path, backend: str, ranks: int) -> list[dict[str, Any]]:
    records = [strict_load(path) for path in sorted((run_dir / "bindings").glob("*.json"))]
    binding_keys = {"rank", "local_rank", "hostname", "cuda_visible_devices",
                    "visible_device_token", "selected_uuid", "gpu_name",
                    "binding_verified"}
    require(len(records) == ranks, f"expected {ranks} rank-binding records in {run_dir}")
    require({record.get("rank") for record in records} == set(range(ranks)) and
            {record.get("local_rank") for record in records} == set(range(ranks)),
            f"rank/local-rank inventory is incomplete in {run_dir}")
    require(all(set(record) == binding_keys and
                isinstance(record.get("hostname"), str) and record["hostname"]
                for record in records),
            f"rank-binding metadata is malformed in {run_dir}")
    if backend == "Cuda":
        uuids = [record.get("selected_uuid") for record in records]
        require(all(record.get("binding_verified") is True for record in records) and
                all(isinstance(uuid, str) and uuid for uuid in uuids) and
                all(isinstance(record.get("cuda_visible_devices"), str) and
                    "," not in record["cuda_visible_devices"] for record in records) and
                len(set(uuids)) == ranks,
                f"CUDA UUID binding is missing or duplicated in {run_dir}")
    return records


def latest_restart(run_dir: Path) -> tuple[Path, dict[str, str]]:
    records = [(path, restart_metadata(path)) for path in run_dir.glob("rst/*.rst")]
    require(records, f"restart evidence is missing: {run_dir}")
    return max(records, key=lambda item: int(item[1]["central_last_cycle"]))


def collect_case(state: dict[str, Any], name: str, reader) -> dict[str, Any]:
    root = Path(state["source"]["path"])
    run_dir = Path(state["cases"][name]["log_path"]).parent
    histories = sorted(run_dir.glob("*.hst"))
    require(len(histories) == 1, f"expected one history file in {run_dir}")
    history = read_history(histories[0])
    require(tuple(history) == HISTORY_KEYS,
            f"unexpected history inventory in {histories[0]}")
    groups = final_binary_groups(run_dir, reader)
    fixed_blocks = int(parse_athinput(TEMPLATES["fixed"])["mesh"]["nx1"]) // int(
        parse_athinput(TEMPLATES["fixed"])["meshblock"]["nx1"])
    restart_path, carrier = latest_restart(run_dir)
    volume = cylindrical_volume(groups["adm"])
    logged_volume = history["Volume"][-1]
    volume_relative_error = abs(volume - logged_volume) / max(abs(volume), 1.0)
    log_text = Path(state["cases"][name]["log_path"]).read_text(
        encoding="utf-8", errors="replace")
    created = [int(value) for value in re.findall(r"(\d+) MeshBlocks created", log_text)]
    deleted = [int(value) for _, value in re.findall(
        r"(\d+) MeshBlocks created, (\d+) MeshBlocks deleted", log_text)]
    evidence_files = {
        str(path.relative_to(run_dir)): sha256(path)
        for path in sorted(run_dir.rglob("*")) if path.is_file()
    }
    return {
        "history": history,
        "snapshots": {group: snapshot_digest(records) for group, records in groups.items()},
        "tree": tree_summary(groups["z4c"], fixed_blocks),
        "rank_bindings": binding_summary(run_dir, state["backend"], state["ranks"]),
        "central": {key: carrier[key] for key in CENTRAL_KEYS},
        "restart_path": str(restart_path),
        "restart_sha256": sha256(restart_path),
        "volume_oracle": volume,
        "volume_relative_error": volume_relative_error,
        "amr_created": max(created, default=0),
        "amr_deleted": max(deleted, default=0),
        "evidence_files": evidence_files,
        "source_path": str(root),
    }


def validate_observations(observations: dict[str, Any], backend: str,
                          ranks: int) -> dict[str, Any]:
    required_cases = {"fixed_fresh", "fixed_restart", "amr_fresh",
                      "amr_restart_pre", "amr_restart_post", "amr_derefine"}
    require(set(observations) == required_cases, "observation case inventory changed")
    for name, record in observations.items():
        require_finite_json(record)
        history = record["history"]
        count = len(history["cycle"])
        require(count > 0 and all(len(values) == count for values in history.values()),
                f"history lengths disagree for {name}")
        require(all(value >= 0.0 for value in history["axisLapse"]) and
                all(value >= 0.0 for value in history["axisKret"]) and
                all(history["axisTau"][index] <= history["axisTau"][index + 1]
                    for index in range(count - 1)),
                f"central history is invalid for {name}")
        require(all(history[key][-1] >= 0.0 for key in
                    ("C-norm2", "H-norm2", "M-norm2", "Z-norm2", "Volume",
                     "max_abs_K", "maxAbsKret")),
                f"constraint/curvature history is invalid for {name}")
        require(record["volume_relative_error"] <= 5.0e-5,
                f"cylindrical volume oracle mismatch for {name}")
        require(record.get("evidence_files") and
                all(re.fullmatch(r"[0-9a-f]{64}", digest)
                    for digest in record["evidence_files"].values()),
                f"raw evidence checksums are incomplete for {name}")
        require(set(record["tree"]["ownership"]) == {str(rank) for rank in range(ranks)}
                and all(value > 0 for value in record["tree"]["ownership"].values()),
                f"MeshBlock rank ownership is incomplete for {name}")
        require(len(record["rank_bindings"]) == ranks,
                f"rank binding inventory is incomplete for {name}")
        if backend == "Cuda":
            uuids = [binding["selected_uuid"] for binding in record["rank_bindings"]]
            require(len(set(uuids)) == ranks and all(uuids),
                    f"CUDA rank binding is not one-GPU-per-rank for {name}")
        central = record["central"]
        require(central["central_initialized"] in {"1", "true"} and
                int(central["central_last_cycle"]) == int(history["cycle"][-1]),
                f"central restart metadata does not match history for {name}")
        for carrier_key, history_key in (("central_previous_lapse", "axisLapse"),
                                         ("central_proper_time", "axisTau"),
                                         ("central_abs_kretschmann", "axisKret")):
            actual = float(central[carrier_key])
            recorded = history[history_key][-1]
            require(abs(actual - recorded) <=
                    5.0e-6 * max(1.0, abs(actual), abs(recorded)),
                    f"central restart/history mismatch for {name}:{carrier_key}")
    require(observations["fixed_fresh"]["tree"]["max_level"] == 0 and
            observations["fixed_restart"]["tree"]["max_level"] == 0,
            "fixed-grid campaign unexpectedly refined")
    amr = observations["amr_fresh"]
    require(amr["tree"]["max_level"] >= 1 and
            amr["tree"]["axis_adjacent_refinement"] and amr["amr_created"] > 0,
            "AMR campaign did not refine across the internal axis")
    derefine = observations["amr_derefine"]
    require(derefine["tree"]["max_level"] == 0 and derefine["amr_deleted"] > 0,
            "forced structural derefinement did not return to the root tree")
    pairs = (("fixed_fresh", "fixed_restart"),
             ("amr_fresh", "amr_restart_pre"),
             ("amr_fresh", "amr_restart_post"))
    for reference, resumed in pairs:
        require(observations[reference]["snapshots"] ==
                observations[resumed]["snapshots"],
                f"accepted state differs after restart: {reference}/{resumed}")
        require(observations[reference]["central"] ==
                observations[resumed]["central"],
                f"central state differs after restart: {reference}/{resumed}")
        require(observations[reference]["tree"]["locations"] ==
                observations[resumed]["tree"]["locations"],
                f"AMR tree differs after restart: {reference}/{resumed}")
    return {"verdict": "pass", "state_equal_pairs": [list(pair) for pair in pairs],
            "fixed_max_level": observations["fixed_fresh"]["tree"]["max_level"],
            "amr_max_level": amr["tree"]["max_level"],
            "derefine_final_level": derefine["tree"]["max_level"]}


def analyze(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = strict_load(state_path)
    verify_state(state)
    require(all(case["status"] == "complete" for case in state["cases"].values()),
            "analysis requires every declared case to complete")
    for name in state["cases"]:
        verify_restart_lineage(state, name)
    reader = import_binary_reader(Path(state["source"]["path"]))
    observations = {name: collect_case(state, name, reader)
                    for name in state["cases"]}
    summary = validate_observations(observations, state["backend"], state["ranks"])
    report = {"schema": ANALYSIS_SCHEMA,
              "campaign_contract_sha256": state["contract_sha256"],
              "backend": state["backend"], "ranks": state["ranks"],
              "provenance": {"source": state["source"],
                             "executable": state["executable"],
                             "inputs": state["inputs"]},
              "summary": summary, "cases": observations}
    output = state_path.parent / "r1_analysis.json"
    atomic_json(output, report)
    state["analysis"] = {"path": str(output), "sha256": sha256(output),
                         "verdict": summary["verdict"]}
    atomic_json(state_path, state)


def synthetic_observations(backend: str = "Cuda") -> dict[str, Any]:
    names = ("fixed_fresh", "fixed_restart", "amr_fresh", "amr_restart_pre",
             "amr_restart_post", "amr_derefine")
    history = {key: [value, value] for key, value in {
        "time": 0.1, "dt": 0.01, "C-norm2": 1.0e-8, "H-norm2": 1.0e-9,
        "M-norm2": 1.0e-9, "Z-norm2": 1.0e-10, "Mx-norm2": 1.0e-9,
        "My-norm2": 1.0e-9, "Mz-norm2": 1.0e-9, "Theta-norm": 1.0e-10,
        "Volume": 12.0, "max_abs_K": 0.2, "nmb_total": 4.0,
        "maxAbsKret": 2.0, "maxRefLev": 0.0, "maxNmbRank": 1.0,
        "ahStatus": 0.0, "ahLastCyc": -1.0, "cycle": 4.0,
        "axisLapse": 0.5, "axisTau": 0.1, "axisKret": 2.0,
    }.items()}
    bindings = [{"rank": rank, "local_rank": rank,
                 "selected_uuid": f"GPU-{rank}" if backend == "Cuda" else None,
                 "binding_verified": backend == "Cuda"} for rank in range(4)]
    central = {"central_initialized": "1", "central_proper_time": "0.1",
               "central_previous_lapse": "0.5", "central_constraint_norm": "1e-4",
               "central_abs_kretschmann": "2", "central_sample_gid": "0",
               "central_sample_level": "0", "central_last_cycle": "4",
               "central_last_time": "0.1"}
    result: dict[str, Any] = {}
    for name in names:
        is_amr = name.startswith("amr") and name != "amr_derefine"
        locations = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]]
        if is_amr:
            locations = [[lx1, lx2, 0, 1] for lx1 in range(4) for lx2 in range(4)]
        ownership = {str(rank): 1 for rank in range(4)}
        local_history = {key: list(values) for key, values in history.items()}
        local_history["maxRefLev"] = [float(is_amr), float(is_amr)]
        result[name] = {"history": local_history,
                        "snapshots": {group: "amr-state" if is_amr else "fixed-state"
                                      for group in FIELD_GROUPS},
                        "tree": {"locations": locations, "max_level": int(is_amr),
                                 "axis_adjacent_refinement": is_amr,
                                 "ownership": ownership},
                        "rank_bindings": bindings, "central": dict(central),
                        "restart_path": f"{name}.rst", "restart_sha256": "a" * 64,
                        "volume_oracle": 12.0, "volume_relative_error": 0.0,
                        "amr_created": 12 if name == "amr_fresh" else 0,
                        "amr_deleted": 12 if name == "amr_derefine" else 0,
                        "evidence_files": {"run.log": "a" * 64},
                        "source_path": "/synthetic"}
    return result


def self_test() -> None:
    validate_inputs()
    validate_source_contract(SCRIPT_DIR.parents[2])
    import numpy as np  # pylint: disable=import-outside-toplevel
    flat = np.ones((1, 4, 4), dtype=np.float64)
    zero = np.zeros_like(flat)
    flat_data = {"mb_geometry": [np.array([-2.0, 2.0, -3.0, 3.0, -0.5, 0.5])],
                 "mb_data": {"adm_gxx": [flat], "adm_gxy": [zero],
                             "adm_gxz": [zero], "adm_gyy": [flat],
                             "adm_gyz": [zero], "adm_gzz": [flat]}}
    require(abs(cylindrical_volume([(0, flat_data)]) - 24.0 * math.pi) < 1.0e-12,
            "flat-cylinder analytic volume fixture failed")
    fixed = parse_athinput(TEMPLATES["fixed"])
    rendered_path = SCRIPT_DIR / ".cartoon_r1_render_selftest.athinput"
    try:
        rendered_path.write_text(
            render_athinput(fixed, {"job/basename": "rendered", "time/nlim": "9"}),
            encoding="utf-8")
        rendered = parse_athinput(rendered_path)
        require(rendered["job"]["basename"] == "rendered" and
                rendered["time"]["nlim"] == "9", "input rendering lost overrides")
    finally:
        if rendered_path.exists():
            rendered_path.unlink()
    for malformed in ("<mesh>\nnx1 = 4\n<mesh>\nnx2 = 4\n",
                      "<mesh>\nnx1 = 4\nnx1 = 8\n", "orphan = value\n"):
        try:
            parse_athinput_text(malformed, "synthetic-malformed")
        except RuntimeError:
            pass
        else:
            raise RuntimeError("strict Athena input parser accepted malformed records")
    validate_observations(synthetic_observations(), "Cuda", 4)
    mirrored = {(lx1, lx2, 0, 1): (lx1 + lx2) % 4
                for lx1 in range(4) for lx2 in range(4)}
    require(summarize_tree(mirrored, 2)["axis_adjacent_refinement"],
            "synthetic mirrored tree missed the internal axis")
    asymmetric = dict(mirrored)
    asymmetric.pop((3, 0, 0, 1))
    try:
        summarize_tree(asymmetric, 2)
    except RuntimeError as error:
        require("asymmetric signed-rho" in str(error), "wrong tree failure")
    else:
        raise RuntimeError("synthetic asymmetric tree was accepted")
    broken = synthetic_observations()
    broken["amr_restart_post"]["snapshots"]["z4c"] = "different"
    try:
        validate_observations(broken, "Cuda", 4)
    except RuntimeError as error:
        require("accepted state differs" in str(error), "wrong equality failure")
    else:
        raise RuntimeError("synthetic restart mismatch was accepted")
    duplicate_gpu = synthetic_observations()
    duplicate_gpu["fixed_fresh"]["rank_bindings"][1]["selected_uuid"] = "GPU-0"
    try:
        validate_observations(duplicate_gpu, "Cuda", 4)
    except RuntimeError as error:
        require("CUDA rank binding" in str(error), "wrong binding failure")
    else:
        raise RuntimeError("synthetic duplicate GPU binding was accepted")
    with tempfile.TemporaryDirectory(prefix="cartoon-r1-restart-") as directory:
        history_path = Path(directory) / "synthetic.z4c.user.hst"
        header = "#  " + "    ".join(
            f"[{index}]={name}" for index, name in enumerate(HISTORY_KEYS, 1))
        history_path.write_text(
            "# Athena++ history data\n" + header + "\n" +
            " ".join("1" for _ in HISTORY_KEYS) + "\n", encoding="utf-8")
        require(tuple(read_history(history_path)) == HISTORY_KEYS and
                "Theta-norm" in read_history(history_path),
                "exact emitted history-header fixture changed")
        restart_path = Path(directory) / "synthetic.rst"
        carrier = {
            "carrier_schema": "1", "symmetry": "cartoon_so2",
            "coordinate_map": "signed_rho_z_suppressed_y_v1", "symmetry_schema": "1",
            "requested_spatial_order": "6", "effective_spatial_order": "6",
            "stencil_width": "4", "mesh_nx1": "64", "mesh_nx2": "64",
            "mesh_nx3": "1", "meshblock_nx1": "32", "meshblock_nx2": "32",
            "meshblock_nx3": "1",
            "central_schema": "2", "central_initialized": "1",
            "central_proper_time": "0.1", "central_previous_lapse": "0.5",
            "central_constraint_norm": "1e-4", "central_abs_kretschmann": "2",
            "central_sample_gid": "0", "central_sample_level": "0",
            "central_last_cycle": "4", "central_last_time": "0.1",
            "fastflow_schema": "1", "fastflow_coefficient_count": "0",
            "fastflow_coefficients": "none", "fastflow_surface_mode": "none",
            "fastflow_selected_branch": "none", "fastflow_center_count": "0",
            "fastflow_center_z0": "0", "fastflow_center_z1": "0",
            "fastflow_status": "not_started", "fastflow_failure_code": "none",
            "fastflow_last_search_cycle": "-1", "fastflow_last_search_time": "0",
            "fastflow_converged": "0",
        }
        require(tuple(carrier) == RESTART_KEYS,
                "synthetic restart fixture does not match exact production inventory")
        restart_path.write_bytes(
            ("<z4c_restart>\n" + "".join(f"{key} = {value}\n"
             for key, value in carrier.items()) + "<par_end>\n").encode("utf-8") + b"raw")
        require(restart_metadata(restart_path)["central_last_cycle"] == "4",
                "synthetic restart metadata was not recovered")
        restart_path.write_bytes(restart_path.read_bytes().replace(b"1e-4", b" nan"))
        try:
            restart_metadata(restart_path)
        except RuntimeError as error:
            require("nonfinite central_constraint_norm" in str(error),
                    "wrong nonfinite restart failure")
        else:
            raise RuntimeError("nonfinite synthetic restart metadata was accepted")
    strict_path = SCRIPT_DIR / ".cartoon_r1_json_selftest.json"
    try:
        for payload in ('{"a": NaN}', '{"a": 1, "a": 2}'):
            strict_path.write_text(payload, encoding="utf-8")
            try:
                strict_load(strict_path)
            except RuntimeError:
                pass
            else:
                raise RuntimeError("strict JSON accepted malformed evidence")
        try:
            atomic_json(strict_path, {"bad": float("inf")})
        except RuntimeError:
            pass
        else:
            raise RuntimeError("strict JSON emitted a nonfinite value")
    finally:
        if strict_path.exists():
            strict_path.unlink()
    state = {"schema": SCHEMA, "source": {}, "executable": {}, "backend": "Cuda",
             "ranks": 4, "inputs": {}, "cases": {}}
    state["contract_sha256"] = contract_digest(state)
    require(state["contract_sha256"] != contract_digest(state | {"ranks": 2}),
            "contract digest did not bind rank count")
    print("Cartoon R1 campaign tooling self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    make = subparsers.add_parser("prepare")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--executable", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--backend", choices=("Serial", "Cuda"), required=True)
    make.add_argument("--ranks", type=int, default=4)
    run = subparsers.add_parser("run-case")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--case", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--mpiexec", default="mpiexec")
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
