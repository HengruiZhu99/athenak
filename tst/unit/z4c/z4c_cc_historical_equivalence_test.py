#!/usr/bin/env python3
"""Compare the candidate CC path to the frozen pre-VC source authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import z4c_cc_selector_equivalence_test as cc


HISTORICAL_COMMIT = "6daa774d7451dbc5f7cac640c6e32a6fd11de7f9"
CONFIG_KEYS = (
    "Athena_BUILD_UNIT_TESTS", "Athena_ENABLE_IRISK_INTERPOLATOR",
    "Athena_ENABLE_MPI", "Athena_ENABLE_OPENMP", "Athena_SINGLE_PRECISION",
    "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "Kokkos_ENABLE_CUDA",
    "Kokkos_ENABLE_HIP", "Kokkos_ENABLE_OPENMP", "Kokkos_ENABLE_SERIAL",
    "Kokkos_ENABLE_SYCL", "PROBLEM",
)


def cache_values(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        typed_key, value = line.split("=", 1)
        key = typed_key.split(":", 1)[0]
        if key in CONFIG_KEYS:
            cc.require(key not in parsed, f"duplicate cache key {key} in {path}")
            parsed[key] = value
    cc.require(set(parsed) == set(CONFIG_KEYS),
               f"missing reviewed cache keys in {path}: {set(CONFIG_KEYS) - set(parsed)}")
    parsed["CMAKE_CXX_COMPILER"] = str(Path(parsed["CMAKE_CXX_COMPILER"]).resolve())
    return parsed


def compare_outputs(left_root: Path, right_root: Path, source_dir: Path) -> dict:
    sys.path.insert(0, str(source_dir / "vis/python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel

    basename = "z4c_kerr_half_plane_init"
    evidence = {
        "history_sha256": cc.sha256_file(left_root / f"{basename}.z4c.user.hst"),
        "timestep_contract_sha256": cc.sha256_file(
            left_root / "z4c_timestep_contract.csv"),
        "binary_payload_sha256": {},
        "restart_payload_sha256": {},
    }
    for name, expected in ((f"{basename}.z4c.user.hst", cc.HISTORY_SHA256),
                           ("z4c_timestep_contract.csv", cc.TIMESTEP_SHA256)):
        left = (left_root / name).read_bytes()
        right = (right_root / name).read_bytes()
        cc.require(left == right, f"candidate/historical {name} bytes changed")
        cc.require(cc.sha256_bytes(left) == expected,
                   f"candidate/historical authoritative {name} SHA-256 changed")

    for variable in ("z4c", "con"):
        for output in (0, 1, 2):
            relative = Path("bin") / f"{basename}.{variable}.{output:05d}.bin"
            digest = cc.compare_binary(
                bin_convert.read_binary(str(left_root / relative)),
                bin_convert.read_binary(str(right_root / relative)), str(relative))
            evidence["binary_payload_sha256"][str(relative)] = digest

    left_restarts = sorted((left_root / "rst").glob(f"{basename}.*.rst"))
    right_restarts = sorted((right_root / "rst").glob(f"{basename}.*.rst"))
    cc.require(left_restarts and
               [path.name for path in left_restarts] ==
               [path.name for path in right_restarts],
               "candidate/historical restart inventories differ")
    for left_path, right_path in zip(left_restarts, right_restarts):
        left_payload = cc.restart_payload(left_path)
        right_payload = cc.restart_payload(right_path)
        cc.require(left_payload == right_payload,
                   f"candidate/historical restart payload changed for {left_path.name}")
        evidence["restart_payload_sha256"][left_path.name] = \
            cc.sha256_bytes(left_payload)
    return evidence


def git_identity(source: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], check=True,
        text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--short"], check=True,
        text=True, capture_output=True).stdout
    cc.require(not status, f"source authority is dirty: {source}\n{status}")
    return {"commit": commit, "tree": tree}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-athena", type=Path, required=True)
    parser.add_argument("--historical-athena", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--historical-cache", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--historical-source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_identity = git_identity(args.candidate_source)
    historical_identity = git_identity(args.historical_source)
    cc.require(historical_identity["commit"] == HISTORICAL_COMMIT,
               "historical source is not the frozen pre-VC authority")
    candidate_cache = cache_values(args.candidate_cache)
    historical_cache = cache_values(args.historical_cache)
    cc.require(candidate_cache == historical_cache,
               f"candidate/historical build configurations differ:\n"
               f"candidate={candidate_cache}\nhistorical={historical_cache}")

    root = args.work_dir
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    input_text = args.input.read_text(encoding="utf-8")
    cc.require("grid_centering" not in input_text,
               "historical comparison input must use the implicit CC selector")
    input_text = input_text.rstrip() + "\n" + cc.RESTART_OUTPUT
    candidate_run = root / "candidate"
    historical_run = root / "historical"
    cc.run_case(args.candidate_athena, input_text, candidate_run)
    cc.run_case(args.historical_athena, input_text, historical_run)

    evidence = {
        "schema": 1,
        "disposition": "pass",
        "historical_source": historical_identity,
        "candidate_source": candidate_identity,
        "reviewed_build_configuration": candidate_cache,
        "comparison": compare_outputs(candidate_run, historical_run,
                                      args.candidate_source),
    }
    (root / "cc_historical_equivalence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS: candidate CC path is exact against historical pre-VC authority")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
