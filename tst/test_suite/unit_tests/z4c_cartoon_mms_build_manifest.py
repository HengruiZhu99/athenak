#!/usr/bin/env python3
"""Create/check immutable Cartoon-MMS configure/build provenance.

The script never configures or builds.  It consumes checksum-bound records produced by
an outer `/usr/bin/time -v` wrapper and the transparent per-TU timing launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


SCHEMA = "athenak_z4c_cartoon_mms_build_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_record(path: Path, label: str) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "command", "started_at_utc", "finished_at_utc",
                "elapsed_seconds", "peak_rss_kib", "exit_code", "log_path",
                "gnu_time_path"}
    require(required.issubset(record), f"{label} timing record is incomplete")
    require(isinstance(record["command"], list) and record["command"],
            f"{label} command is not an argv list")
    require(record["exit_code"] == 0 and float(record["elapsed_seconds"]) > 0.0 and
            int(record["peak_rss_kib"]) > 0, f"{label} did not complete successfully")
    log = Path(str(record["log_path"])).resolve()
    require(log.is_file(), f"{label} log does not exist")
    record["log_path"] = str(log)
    record["log_sha256"] = sha256(log)
    raw_time = Path(str(record["gnu_time_path"])).resolve()
    require(raw_time.is_file(), f"{label} raw GNU-time output does not exist")
    record["gnu_time_path"] = str(raw_time)
    record["gnu_time_sha256"] = sha256(raw_time)
    return record


def cache_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        left, value = line.split("=", 1)
        key, _ = left.split(":", 1)
        values[key] = value
    return values


def validate_cache(path: Path, backend: str) -> dict[str, str]:
    values = cache_values(path)
    required = {"CMAKE_BUILD_TYPE": "Release", "Athena_ENABLE_MPI": "ON",
                "Athena_ENABLE_OPENMP": "OFF", "Athena_BUILD_UNIT_TESTS": "OFF",
                "PROBLEM": "built_in_pgens", "Kokkos_ENABLE_OPENMP": "OFF",
                "Kokkos_ENABLE_TESTS": "OFF"}
    required.update({"Kokkos_ENABLE_CUDA": "ON", "Kokkos_ENABLE_CUDA_LAMBDA": "ON",
                     "Kokkos_ENABLE_CUDA_CONSTEXPR": "ON",
                     "Kokkos_ENABLE_SERIAL": "OFF", "Kokkos_ARCH_AMPERE80": "ON"}
                    if backend == "Cuda" else
                    {"Kokkos_ENABLE_CUDA": "OFF", "Kokkos_ENABLE_OPENMP": "OFF",
                     "Kokkos_ENABLE_SERIAL": "ON", "Kokkos_ENABLE_TESTS": "OFF"})
    for key, expected in required.items():
        require(values.get(key) == expected, f"cache requires {key}={expected}")
    return {key: values[key] for key in sorted(set(required) | {
        "CMAKE_CXX_COMPILER", "Kokkos_ENABLE_CUDA", "Kokkos_ENABLE_SERIAL"})
            if key in values}


def load_tus(path: Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    require(records, "per-TU timing evidence is empty")
    required = {"schema", "argv", "source", "object", "wall_seconds",
                "max_rss_kib", "exit_code"}
    require(all(required.issubset(record) and
                record["schema"] == "athenak_z4c_cartoon_mms_tu_timing_v1" and
                isinstance(record["argv"], list) and record["source"] and
                record["source"] in record["argv"] and record["object"] and
                record["exit_code"] == 0 and
                float(record["wall_seconds"]) >= 0.0 and int(record["max_rss_kib"]) > 0
                for record in records), "per-TU timing record is incomplete or failed")
    require(len({str(record["object"]) for record in records}) == len(records),
            "per-TU timing evidence contains duplicate object paths")
    require(not any("cartoon_derivatives_test" in str(record["source"]) or
                    "/tst/unit/" in str(record["source"]) for record in records),
            "routine build unexpectedly compiled the monolithic/unit-test target")
    return records


def source_identity(root: Path) -> dict[str, object]:
    return {"source_commit": git(root, "rev-parse", "HEAD"),
            "source_tree": git(root, "rev-parse", "HEAD^{tree}"),
            "kokkos_commit": git(root, "rev-parse", "HEAD:kokkos"),
            "source_clean": git(root, "status", "--porcelain") == ""}


def create(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    executable = args.executable.resolve()
    cache = args.cache.resolve()
    timing = args.timing_jsonl.resolve()
    compiler = args.compiler_version.resolve()
    runtime_path = args.kokkos_runtime.resolve()
    for path in (executable, cache, timing, compiler, runtime_path):
        require(path.is_file(), f"required build evidence is missing: {path}")
    identity = source_identity(source)
    require(identity["source_clean"], "build provenance requires a clean source checkout")
    configure = load_record(args.configure_record.resolve(), "configure")
    build = load_record(args.build_record.resolve(), "build")
    exact_flags = ["-DCMAKE_BUILD_TYPE=Release", "-DPROBLEM=built_in_pgens",
                   "-DAthena_ENABLE_MPI=ON", "-DAthena_ENABLE_OPENMP=OFF",
                   "-DAthena_BUILD_UNIT_TESTS=OFF", "-DKokkos_ENABLE_OPENMP=OFF",
                   "-DKokkos_ENABLE_TESTS=OFF"]
    exact_flags += (["-DCMAKE_C_COMPILER=cc", "-DCMAKE_CXX_COMPILER=CC",
                     "-DKokkos_ENABLE_CUDA=ON", "-DKokkos_ENABLE_CUDA_LAMBDA=ON",
                     "-DKokkos_ENABLE_CUDA_CONSTEXPR=ON",
                     "-DKokkos_ENABLE_SERIAL=OFF", "-DKokkos_ARCH_AMPERE80=ON"]
                    if args.backend == "Cuda" else
                    ["-DKokkos_ENABLE_CUDA=OFF", "-DKokkos_ENABLE_SERIAL=ON"])
    require(all(flag in configure["command"] for flag in exact_flags),
            "configure argv differs from the exact steering backend flags")
    require("--target" in build["command"] and
            build["command"][build["command"].index("--target") + 1] == "athena",
            "reviewed routine build must target only athena")
    require("--parallel" in build["command"] and
            build["command"][build["command"].index("--parallel") + 1] == "8",
            "reviewed build command must use --parallel 8")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    require(runtime.get("backend") == args.backend and
            runtime.get("default_execution_space") == args.backend and
            runtime.get("kokkos_version"), "Kokkos runtime evidence conflicts with backend")
    tus = load_tus(timing)
    slowest = sorted(tus, key=lambda item: (-float(item["wall_seconds"]),
                                            str(item["object"])))[:20]
    manifest = {"schema": SCHEMA, **identity, "backend": args.backend,
                "executable_path": str(executable), "executable_sha256": sha256(executable),
                "configure_cache_path": str(cache),
                "configure_cache_sha256": sha256(cache),
                "configure_cache_contract": validate_cache(cache, args.backend),
                "compiler": {"version_path": str(compiler),
                             "version_sha256": sha256(compiler),
                             "version_text": compiler.read_text(encoding="utf-8")},
                "kokkos_runtime": runtime,
                "kokkos_runtime_path": str(runtime_path),
                "kokkos_runtime_sha256": sha256(runtime_path),
                "configure": configure, "build": build,
                "translation_units": len(tus), "tu_timing_sha256": sha256(timing),
                "tu_timing_path": str(timing), "slowest_tus": slowest}
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, args.output)


def check(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    require(manifest.get("schema") == SCHEMA, "unknown build-manifest schema")
    source = args.source.resolve()
    identity = source_identity(source)
    require(identity["source_clean"] and all(manifest.get(key) == value
            for key, value in identity.items()), "source identity/cleanliness changed")
    pairs = (("executable_path", "executable_sha256"),
             ("configure_cache_path", "configure_cache_sha256"),
             ("tu_timing_path", "tu_timing_sha256"),
             ("kokkos_runtime_path", "kokkos_runtime_sha256"))
    for path_key, hash_key in pairs:
        path = Path(manifest[path_key])
        require(path.is_file() and sha256(path) == manifest[hash_key],
                f"build artifact changed: {path_key}")
    for phase in ("configure", "build"):
        log = Path(manifest[phase]["log_path"])
        require(log.is_file() and sha256(log) == manifest[phase]["log_sha256"],
                f"{phase} log changed")
        raw_time = Path(manifest[phase]["gnu_time_path"])
        require(raw_time.is_file() and
                sha256(raw_time) == manifest[phase]["gnu_time_sha256"],
                f"{phase} raw GNU-time output changed")
    compiler = Path(manifest["compiler"]["version_path"])
    require(compiler.is_file() and sha256(compiler) == manifest["compiler"]["version_sha256"],
            "compiler evidence changed")
    validate_cache(Path(manifest["configure_cache_path"]), manifest["backend"])
    require(len(load_tus(Path(manifest["tu_timing_path"]))) ==
            manifest["translation_units"], "translation-unit count changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    make = subparsers.add_parser("create")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--executable", type=Path, required=True)
    make.add_argument("--cache", type=Path, required=True)
    make.add_argument("--timing-jsonl", type=Path, required=True)
    make.add_argument("--configure-record", type=Path, required=True)
    make.add_argument("--build-record", type=Path, required=True)
    make.add_argument("--compiler-version", type=Path, required=True)
    make.add_argument("--kokkos-runtime", type=Path, required=True)
    make.add_argument("--backend", choices=("Serial", "Cuda"), required=True)
    make.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("check")
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "create":
        create(args)
    else:
        check(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(f"FAIL: {error}")
