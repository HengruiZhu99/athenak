#!/usr/bin/env python3
"""Fail-closed runner for the half-plane spin-0.5 Kerr qualification deck."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


CASES = (
    ("moving_puncture", "h32", 96, 192, 1.0 / 32.0),
    ("moving_puncture", "h48", 144, 288, 1.0 / 48.0),
    ("moving_puncture", "h64", 192, 384, 1.0 / 64.0),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError(f"nonfinite JSON token {value}")))


def write_json(path: Path, value) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def replace_once(text: str, old: str, new: str) -> str:
    require(text.count(old) == 1, f"template expected exactly one {old!r}")
    return text.replace(old, new)


def render_input(template: str, gauge: str, spacing: str,
                 nx1: int, nx2: int) -> str:
    basename = f"kerr_half_plane_{spacing}_{gauge}"
    rendered = replace_once(template,
                            "basename = kerr_half_plane_h32_moving_puncture",
                            f"basename = {basename}")
    rendered = replace_once(rendered, "nx1 = 96\n", f"nx1 = {nx1}\n")
    rendered = replace_once(rendered, "nx2 = 192\n", f"nx2 = {nx2}\n")
    require(gauge == "moving_puncture", f"unsupported gauge {gauge}")
    return rendered


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify_state(root: Path) -> dict:
    state_path = root / "campaign_state.json"
    state = load_json(state_path)
    require(state.get("schema") == "athenak_cartoon_half_plane_kerr_campaign_v2",
            "campaign state schema changed")
    contract = state.get("contract")
    require(isinstance(contract, dict), "campaign immutable contract is missing")
    require(state.get("contract_sha256") == contract_digest(contract),
            "campaign immutable contract digest changed")
    source = Path(contract["source"]["path"])
    executable = Path(contract["executable"]["path"])
    require(source.is_dir() and not source.is_symlink(), "source root is unavailable")
    require(executable.is_file() and not executable.is_symlink(),
            "campaign executable is unavailable")
    require(git(source, "rev-parse", "HEAD^{commit}") == contract["source"]["commit"],
            "campaign source commit changed")
    require(git(source, "rev-parse", "HEAD^{tree}") == contract["source"]["tree"],
            "campaign source tree changed")
    require(git(source, "status", "--porcelain") == "", "campaign source is dirty")
    require(sha256(executable) == contract["executable"]["sha256"],
            "campaign executable bytes changed")
    for case in contract["cases"]:
        path = Path(case["input_path"])
        require(path.is_file() and not path.is_symlink(),
                f"case input unavailable: {case['name']}")
        require(sha256(path) == case["input_sha256"],
                f"case input bytes changed: {case['name']}")
    return state


def prepare(arguments) -> None:
    output = arguments.output.resolve()
    require(not output.exists() and not output.is_symlink(),
            "campaign output must be absent")
    source = arguments.source_dir.resolve()
    executable = arguments.executable.resolve()
    template = arguments.template.resolve()
    analyzer = arguments.analyzer.resolve()
    require(git(source, "status", "--porcelain") == "",
            "final campaign source must be clean")
    for path, kind in ((executable, "executable"), (template, "template"),
                       (analyzer, "analyzer")):
        require(path.is_file() and not path.is_symlink(), f"{kind} is unavailable")

    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    template_text = template.read_text(encoding="utf-8")
    case_contracts = []
    dynamic_cases = {}
    for gauge, spacing, nx1, nx2, h in CASES:
        name = f"{gauge}_{spacing}"
        input_path = inputs / f"{name}.athinput"
        input_path.write_text(render_input(template_text, gauge, spacing, nx1, nx2),
                              encoding="utf-8")
        case_contracts.append({
            "name": name,
            "gauge": gauge,
            "spacing": spacing,
            "finest_spacing_M": h,
            "root_nx1": nx1,
            "root_nx2": nx2,
            "input_path": str(input_path),
            "input_sha256": sha256(input_path),
        })
        dynamic_cases[name] = {"status": "pending", "returncode": None,
                               "command_sha256": None}
    contract = {
        "source": {"path": str(source),
                   "commit": git(source, "rev-parse", "HEAD^{commit}"),
                   "tree": git(source, "rev-parse", "HEAD^{tree}")},
        "executable": {"path": str(executable), "sha256": sha256(executable)},
        "template": {"path": str(template), "sha256": sha256(template)},
        "analyzer": {"path": str(analyzer), "sha256": sha256(analyzer)},
        "physics": {"mass": 1.0, "dimensionless_spin": 0.5,
                    "target_time_M": 5.0, "spatial_order": 6,
                    "integrator": "rk4", "precollapsed_lapse": True,
                    "chi_floor": False, "dchi_max": 0.02,
                    "gauge": {
                        "name": "athenak_default_moving_puncture",
                        "lapse": "advective_1_plus_log",
                        "lapse_oplog": 2.0,
                        "lapse_harmonicf": 1.0,
                        "lapse_harmonic": 0.0,
                        "lapse_advect": 1.0,
                        "slow_start_lapse": False,
                        "telegraph_lapse": False,
                        "shift": "advective_Gamma_driver",
                        "shift_Gamma": 1.0,
                        "shift_eta": 2.0,
                        "shift_advect": 1.0,
                        "shift_alpha2Gamma": 0.0,
                        "shift_H": 0.0,
                        "shift_eta_max_K": False,
                        "sss_damping_amp": 0.0,
                    }},
        "execution": {"ranks": 4, "gpus": 4, "cpus_per_rank": 8,
                      "one_node_per_case": True},
        "cases": case_contracts,
    }
    state = {"schema": "athenak_cartoon_half_plane_kerr_campaign_v2",
             "contract": contract, "contract_sha256": contract_digest(contract),
             "cases": dynamic_cases, "analysis": None}
    write_json(output / "campaign_state.json", state)
    verify_state(output)
    print(f"prepared {len(CASES)} immutable half-plane Kerr cases at {output}")


def find_case(state: dict, name: str) -> dict:
    matches = [case for case in state["contract"]["cases"] if case["name"] == name]
    require(len(matches) == 1, f"unknown or duplicate case {name}")
    return matches[0]


def run_case(arguments) -> None:
    root = arguments.output.resolve()
    state = verify_state(root)
    case = find_case(state, arguments.case)
    dynamic = state["cases"][arguments.case]
    require(dynamic["status"] == "pending", "case is not pending")
    launcher = json.loads(arguments.launcher_json)
    require(isinstance(launcher, list) and launcher and
            all(isinstance(item, str) and item for item in launcher),
            "launcher JSON must be a nonempty string list")
    case_root = root / case["gauge"] / case["spacing"]
    require(not case_root.exists() and not case_root.is_symlink(),
            "case output must be absent")
    case_root.mkdir(parents=True)
    command = launcher + [state["contract"]["executable"]["path"], "-i",
                          case["input_path"]]
    write_json(case_root / "command.json", {"argv": command})
    with (case_root / "stdout.log").open("wb") as stdout, \
         (case_root / "stderr.log").open("wb") as stderr:
        completed = subprocess.run(command, cwd=case_root, stdout=stdout, stderr=stderr,
                                   check=False)
    (case_root / "status.txt").write_text(f"{completed.returncode}\n", encoding="utf-8")
    dynamic["status"] = "complete" if completed.returncode == 0 else "failed"
    dynamic["returncode"] = completed.returncode
    dynamic["command_sha256"] = sha256(case_root / "command.json")
    write_json(root / "campaign_state.json", state)
    if completed.returncode != 0:
        raise RuntimeError(f"case {arguments.case} failed with {completed.returncode}")
    print(f"case {arguments.case} completed")


def analyze(arguments) -> None:
    root = arguments.output.resolve()
    state = verify_state(root)
    require(all(state["cases"][case["name"]]["status"] == "complete"
                for case in state["contract"]["cases"]),
            "analysis requires complete case inventory")
    analysis_path = root / "qualification_analysis.json"
    command = ["python3", state["contract"]["analyzer"]["path"],
               "--run-root", str(root), "--output", str(analysis_path),
               "--gauges", "moving_puncture"]
    completed = subprocess.run(command, check=False)
    require(completed.returncode == 0 and analysis_path.is_file(), "analyzer failed")
    result = load_json(analysis_path)
    state["analysis"] = {
        "path": str(analysis_path), "sha256": sha256(analysis_path),
        "qualification_claim": result["qualification_claim"]}
    write_json(root / "campaign_state.json", state)
    print(json.dumps(state["analysis"], sort_keys=True))


def self_test(template: Path) -> None:
    text = template.read_text(encoding="utf-8")
    rendered = {}
    for gauge, spacing, nx1, nx2, h in CASES:
        value = render_input(text, gauge, spacing, nx1, nx2)
        name = f"{gauge}_{spacing}"
        require(f"basename = kerr_half_plane_{spacing}_{gauge}" in value,
                f"{name}: basename mismatch")
        require(f"nx1 = {nx1}" in value and f"nx2 = {nx2}" in value,
                f"{name}: root resolution mismatch")
        require("x1min = 0.0" in value and "ix1_bc = axis" in value,
                f"{name}: half-plane geometry missing")
        require("dchi_max = 0.02" in value and "tlim = 5.0" in value and
                "spatial_order = 6" in value and "initial_gauge = precollapsed" in value,
                f"{name}: science contract changed")
        require(all(token in value for token in
                    ("lapse_oplog = 2.0", "lapse_harmonicf = 1.0",
                     "lapse_harmonic = 0.0", "lapse_advect = 1.0",
                     "slow_start_lapse = false", "telegraph_lapse = false",
                     "shift_Gamma = 1.0", "shift_eta = 2.0",
                     "shift_advect = 1.0", "shift_alpha2Gamma = 0.0",
                     "shift_H = 0.0", "shift_eta_max_K = false",
                     "sss_damping_amp = 0.0")),
                "default moving-puncture gauge contract changed")
        rendered[name] = hashlib.sha256(value.encode()).hexdigest()
    require(len(set(rendered.values())) == len(CASES), "rendered cases are not distinct")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "strict.json"
        write_json(path, {"rendered": rendered})
        require(load_json(path)["rendered"] == rendered, "strict JSON round trip failed")
    print("Cartoon half-plane Kerr campaign self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-dir", type=Path, required=True)
    prepare_parser.add_argument("--executable", type=Path, required=True)
    prepare_parser.add_argument("--template", type=Path, required=True)
    prepare_parser.add_argument("--analyzer", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run-case")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--case", choices=[f"{gauge}_{spacing}"
                                                for gauge, spacing, *_ in CASES],
                            required=True)
    run_parser.add_argument("--launcher-json", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, required=True)
    self_parser = subparsers.add_parser("self-test")
    self_parser.add_argument("--template", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments)
    elif arguments.command == "verify":
        verify_state(arguments.output.resolve())
        print("Cartoon half-plane Kerr campaign state verified")
    elif arguments.command == "run-case":
        run_case(arguments)
    elif arguments.command == "analyze":
        analyze(arguments)
    else:
        self_test(arguments.template.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
