#!/usr/bin/env python3
"""Immutable CPU/CUDA MPI campaign driver for the input-selected Cartoon MMS.

This driver intentionally has no configure or build capability.  One already-built
Athena executable is checksum-bound and reused for the complete backend matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid


SCHEMA = "athenak_z4c_cartoon_derivative_mms_campaign_v1"
REDUCTION_TOLERANCE_FACTOR = 4096.0
SATURATION_FACTOR = 4096.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, object]],
                     delimiter: str = ",") -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=delimiter,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def replace_parameter(text: str, block: str, key: str, value: object) -> str:
    pattern = re.compile(rf"(?ms)(^<{re.escape(block)}>\s*$)(.*?)(?=^<|\Z)")
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"input lacks <{block}>")
    body = match.group(2)
    key_pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
    if key_pattern.search(body):
        body = key_pattern.sub(rf"\g<1>{value}", body)
    else:
        body += f"{key} = {value}\n"
    return text[:match.start(2)] + body + text[match.end(2):]


def render_input(base: Path, order: int, resolution: int, phase: int,
                 basename: str, domain: tuple[float, float, float, float]) -> str:
    text = base.read_text(encoding="utf-8")
    nghost = order // 2 + 1
    for block, key, value in (
        ("job", "basename", basename), ("mesh", "nghost", nghost),
        ("mesh", "nx1", resolution), ("mesh", "nx2", resolution),
        ("z4c", "spatial_order", order), ("problem", "noise_phase", phase),
        ("problem", "output_directory", "."),
        ("mesh", "x1min", domain[0]), ("mesh", "x1max", domain[1]),
        ("mesh", "x2min", domain[2]), ("mesh", "x2max", domain[3]),
    ):
        text = replace_parameter(text, block, key, value)
    return text


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def launcher_command(launcher: str, ranks: int) -> list[str]:
    words = shlex.split(launcher)
    executable = Path(words[0]).name
    if executable in {"mpirun", "mpiexec"}:
        return words + ["-np", str(ranks)]
    if executable == "srun":
        return words + ["--ntasks", str(ranks)]
    return words + [str(ranks)]


def verified_complete(case: Path, identity: dict[str, object]) -> bool:
    manifest_path = case / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete" or manifest.get("identity") != identity:
        return False
    return all((case / name).is_file() and sha256(case / name) == digest
               for name, digest in manifest["files"].items())


def run_case(args: argparse.Namespace, root: Path, source: dict[str, str],
             order: int, resolution: int, phase: int) -> dict[str, object]:
    nghost = order // 2 + 1
    domain_hash = hashlib.sha256(json.dumps(args.domain).encode()).hexdigest()[:10]
    key = f"o{order}-ng{nghost}-n{resolution}-p{phase}-r{args.ranks}-d{domain_hash}"
    case_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL,
                               json.dumps([SCHEMA, source, key], sort_keys=True)))
    case = args.output / f"{key}-{case_uuid}"
    basename = "cartoon_mms"
    rendered = render_input(args.input, order, resolution, phase, basename, args.domain)
    wrapper = args.rank_wrapper
    wrapped = [sys.executable, str(wrapper), "--evidence-dir", "."]
    if args.require_backend == "Cuda":
        wrapped.append("--require-cuda")
    command = launcher_command(args.launcher, args.ranks) + wrapped + [
        "--", str(args.athena), "-i", "input.athinput"]
    identity = {"case_id": key, "uuid": case_uuid, "source": source,
                "athena_sha256": sha256(args.athena), "input_sha256":
                hashlib.sha256(rendered.encode()).hexdigest(), "command": command,
                "ranks": args.ranks, "backend_required": args.require_backend,
                "build_manifest_sha256": sha256(args.build_manifest),
                "rank_wrapper_sha256": sha256(args.rank_wrapper),
                "domain": args.domain}
    if case.exists():
        if verified_complete(case, identity):
            resumed = json.loads((case / "result.json").read_text(encoding="utf-8"))
            resumed["case_manifest_sha256"] = sha256(case / "manifest.json")
            return resumed
        raise RuntimeError(f"refusing incomplete or mismatched case directory: {case}")
    case.mkdir(parents=True)
    (case / "input.athinput").write_text(rendered, encoding="utf-8")
    write_atomic(case / "manifest.json", {"schema": SCHEMA, "state": "running",
                                           "identity": identity})
    environment = os.environ.copy()
    started = time.time()
    with (case / "stdout.txt").open("wb") as stdout, \
         (case / "stderr.txt").open("wb") as stderr:
        completed = subprocess.run(command, cwd=case, env=environment,
                                   stdout=stdout, stderr=stderr, check=False)
    if completed.returncode:
        raise RuntimeError(f"case {key} failed with exit {completed.returncode}: {case}")
    stdout_text = (case / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    final_states = re.findall(r"time=([^\s]+)\s+cycle=(\d+)", stdout_text)
    if not final_states or int(final_states[-1][1]) != 0 or float(final_states[-1][0]) != 0.0:
        raise RuntimeError(f"case {key} executed a physical evolution step")
    raw_result = case / f"{basename}.mms.json"
    raw_csv = case / f"{basename}.mms.csv"
    probes_csv = case / f"{basename}.mms.probes.csv"
    result = json.loads(raw_result.read_text(encoding="utf-8"))
    if result.get("status") != "pass" or result.get("operator_count") != 171:
        raise RuntimeError(f"case {key} did not produce the complete passing 171-series set")
    if (result.get("initial_cycle"), result.get("pgen_final_cycle"),
            result.get("initial_time"), result.get("pgen_final_time")) != (0, 0, 0, 0):
        raise RuntimeError(f"case {key} pgen cycle/time sentinel changed")
    if result.get("backend") != args.require_backend:
        raise RuntimeError(f"case {key} backend {result.get('backend')} != {args.require_backend}")
    if result.get("owned_cells") != resolution * resolution or \
       result.get("ownership_sequence") != "[0,N*N) exactly once":
        raise RuntimeError(f"case {key} failed exact MPI ownership proof")
    bindings = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(case.glob("rank_binding_*.json"))]
    if len(bindings) != args.ranks or sorted(item["rank"] for item in bindings) != \
       list(range(args.ranks)):
        raise RuntimeError(f"case {key} lacks one binding record per MPI rank")
    if args.require_backend == "Cuda":
        uuids = [item["selected_uuid"] for item in bindings]
        if (args.ranks != 4 or None in uuids or len(set(uuids)) != 4 or
                any("A100" not in (item.get("gpu_name") or "") for item in bindings)):
            raise RuntimeError(f"case {key} requires four distinct CUDA UUIDs")
    rows = list(csv.DictReader(raw_csv.open(encoding="utf-8")))
    operator_set = {row["operator"] for row in rows}
    operator_names = result.get("operator_names")
    if (not isinstance(operator_names, list) or len(operator_names) != 171 or
            len(set(operator_names)) != 171 or operator_set != set(operator_names) or
            any(int(row["nonfinite"]) for row in rows)):
        raise RuntimeError(f"case {key} has incomplete or nonfinite CSV series")
    noise_bound = float(result["noise_delta_bound"])
    if any(float(row["shared_delta_linfinity"]) > noise_bound or
           float(row["independent_delta_linfinity"]) > noise_bound for row in rows):
        raise RuntimeError(f"case {key} exceeds frozen direct noise-delta bound")
    probe_rows = list(csv.DictReader(probes_csv.open(encoding="utf-8")))
    if {row["operator"] for row in probe_rows} != operator_set or \
       any(not math.isfinite(float(row["raw_error"])) for row in probe_rows) or \
       any(not row["layer_index"] or not row["classification"] for row in probe_rows):
        raise RuntimeError(f"case {key} has incomplete raw probe/layer records")
    result.update({"case_id": key, "case_uuid": case_uuid, "phase": phase,
                   "resolution": resolution, "elapsed_seconds": time.time() - started,
                   "csv_sha256": sha256(raw_csv),
                   "probes_csv_sha256": sha256(probes_csv),
                   "operator_names": operator_names,
                   "rank_bindings": bindings,
                   "output_bytes": raw_csv.stat().st_size + probes_csv.stat().st_size})
    (case / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    files = {name: sha256(case / name) for name in
             ("input.athinput", "stdout.txt", "stderr.txt", "cartoon_mms.mms.json",
              "cartoon_mms.mms.csv", "result.json")}
    files["cartoon_mms.mms.probes.csv"] = sha256(probes_csv)
    for binding in sorted(case.glob("rank_binding_*.json")):
        files[binding.name] = sha256(binding)
    write_atomic(case / "manifest.json", {"schema": SCHEMA, "state": "complete",
                                           "identity": identity, "files": files})
    result["case_manifest_sha256"] = sha256(case / "manifest.json")
    return result


def convergence_gate(cases: list[dict[str, object]], output: Path,
                     series_manifest: Path) -> None:
    inventory = json.loads(series_manifest.read_text(encoding="utf-8"))
    classes = {item["name"]: item["classification"] for item in inventory["series"]}
    convergence_lanes = {item["name"]: item["convergence_lanes"]
                         for item in inventory["series"]}
    if inventory.get("count") != 171 or len(classes) != 171:
        raise RuntimeError("series manifest does not enumerate exactly 171 operators")
    grouped = {}
    exact_records = []
    failures = []
    for result in cases:
        case = output / f"{result['case_id']}-{result['case_uuid']}"
        rows = csv.DictReader((case / "cartoon_mms.mms.csv").open(encoding="utf-8"))
        for row in rows:
            operator = row["operator"]
            classification = classes.get(operator)
            if classification is None:
                failures.append(f"unknown operator {operator}")
                continue
            order = int(result["spatial_order"])
            resolution = int(result["resolution"])
            phase = int(result["phase"])
            mask = row["mask"]
            if classification != "truncating":
                bound = 2.0e-10 if classification == "exact_discrete" else (
                    SATURATION_FACTOR * sys.float_info.epsilon * resolution * resolution)
                passed = float(row["linfinity"]) <= bound
                exact_records.append({"order": order, "phase": phase,
                                      "resolution": resolution, "operator": operator,
                                      "mask": mask, "classification": classification,
                                      "value": float(row["linfinity"]), "bound": bound,
                                      "passed": passed})
                if not passed:
                    failures.append(f"exact gate failed {operator} {mask} N={resolution}")
                continue
            clean_norms = ["l1", "l2", "linfinity"]
            if int(row["cyl_count"]) > 0:
                clean_norms += ["cyl_l1", "cyl_l2", "cyl_linfinity"]
            for norm in clean_norms:
                grouped.setdefault((order, phase, operator + "|" + mask,
                                    "clean", norm), []).append(
                    (resolution, float(row[norm]), 0.0))
            for lane in ("shared", "independent"):
                if lane not in convergence_lanes[operator]:
                    continue
                for norm in ("l1", "l2", "linfinity"):
                    grouped.setdefault((order, phase, operator + "|" + mask,
                                        lane, norm), []).append(
                        (resolution, float(row[f"{lane}_{norm}"]),
                         float(row[f"{lane}_delta_{norm}"])))
    records = []
    epsilon = sys.float_info.epsilon
    for key, values in grouped.items():
        order, phase, series, lane, norm = key
        values.sort()
        if len(values) != 4:
            failures.append(f"{key}: expected four resolutions")
            continue
        mask = series.split("|")[-1]
        expected = order
        margin = 0.15 if lane == "clean" else 0.5
        if mask.startswith("raw_transition"):
            expected = order - 1
            margin = 0.25 if lane == "clean" else 0.5
        elif mask == "full_signed_plane":
            if norm in ("l1", "cyl_l1", "cyl_l2"):
                expected = order
            elif norm == "l2":
                expected = order - 0.5
            else:
                expected = order - 1
            margin = 0.25 if lane == "clean" else 0.5
        ratios = []
        for (coarse_n, coarse, coarse_delta), (fine_n, fine, fine_delta) in zip(
                values, values[1:]):
            arithmetic_floor = SATURATION_FACTOR * epsilon * max(1.0, coarse)
            noise_floor = 8.0 * max(coarse_delta, fine_delta) if lane != "clean" else 0.0
            if fine <= max(arithmetic_floor, noise_floor):
                ratios.append(None)
            elif fine > 0.0 and coarse >= fine:
                ratios.append(math.log(coarse / fine) / math.log(fine_n / coarse_n))
            else:
                ratios.append(float("-inf"))
        unsaturated = [rate for rate in ratios if rate is not None]
        passed = len(unsaturated) >= 2 and min(unsaturated[-2:]) >= expected - margin
        records.append({"order": order, "phase": phase, "series": series,
                        "lane": lane, "norm": norm, "expected": expected,
                        "samples": [{"resolution": n, "error": error,
                                     "direct_delta": delta}
                                    for n, error, delta in values],
                        "rates": ratios, "passed": passed})
        if not passed:
            failures.append(f"{key}: rates={ratios}, expected>={expected-margin}")
    records.sort(key=lambda item: (item["order"], item["phase"], item["series"],
                                   item["lane"], item["norm"]))
    exact_records.sort(key=lambda item: (item["order"], item["phase"],
                                         item["resolution"], item["operator"],
                                         item["mask"]))
    table_rows = []
    for record in records:
        samples = record["samples"]
        for index, rate in enumerate(record["rates"]):
            table_rows.append({
                "order": record["order"], "phase": record["phase"],
                "series": record["series"], "lane": record["lane"],
                "norm": record["norm"], "coarse_resolution":
                samples[index]["resolution"], "fine_resolution":
                samples[index + 1]["resolution"], "coarse_error":
                f"{samples[index]['error']:.17g}", "fine_error":
                f"{samples[index + 1]['error']:.17g}", "coarse_direct_delta":
                f"{samples[index]['direct_delta']:.17g}", "fine_direct_delta":
                f"{samples[index + 1]['direct_delta']:.17g}", "observed_rate":
                "saturated" if rate is None else f"{rate:.17g}",
                "expected_rate": f"{record['expected']:.17g}",
                "passed": int(record["passed"]),
            })
    fields = ["order", "phase", "series", "lane", "norm", "coarse_resolution",
              "fine_resolution", "coarse_error", "fine_error",
              "coarse_direct_delta", "fine_direct_delta", "observed_rate",
              "expected_rate", "passed"]
    csv_path = output / "convergence.csv"
    data_path = output / "convergence_rates.pgfplots.dat"
    write_csv_atomic(csv_path, fields, table_rows)
    # The whitespace-delimited PGFPlots source is intentionally a numeric projection;
    # series names remain in the checksum-bound CSV/JSON and cannot confuse TeX parsing.
    plot_rows = [{"order": row["order"], "phase": row["phase"],
                  "lane_id": {"clean": 0, "shared": 1, "independent": 2}[row["lane"]],
                  "fine_resolution": row["fine_resolution"],
                  "observed_rate": row["observed_rate"],
                  "expected_rate": row["expected_rate"], "passed": row["passed"]}
                 for row in table_rows if row["observed_rate"] != "saturated"]
    plot_fields = ["order", "phase", "lane_id", "fine_resolution",
                   "observed_rate", "expected_rate", "passed"]
    write_csv_atomic(data_path, plot_fields, plot_rows, delimiter=" ")
    plot_path = output / "convergence_plot.tex"
    plot_path.write_text(
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[xlabel={fine resolution},ylabel={observed order},"
        "xmode=log,log basis x=2,legend pos=south east]\n"
        "\\addplot+[only marks,mark size=.35pt] table[x=fine_resolution,"
        "y=observed_rate] {convergence_rates.pgfplots.dat};\n"
        "\\addlegendentry{all unsaturated reviewed series}\n"
        "\\end{axis}\n\\end{tikzpicture}\n", encoding="utf-8")
    write_atomic(output / "convergence.json", {"schema": SCHEMA,
                                                "series_manifest_sha256":
                                                sha256(series_manifest),
                                                "records": records,
                                                "exact_records": exact_records,
                                                "artifacts": {
                                                    "convergence.csv": sha256(csv_path),
                                                    "convergence_rates.pgfplots.dat":
                                                    sha256(data_path),
                                                    "convergence_plot.tex":
                                                    sha256(plot_path)},
                                                "failures": failures})
    if failures:
        raise RuntimeError("convergence gates failed; see convergence.json")


def compare_rank_campaigns(cases: list[dict[str, object]], output: Path,
                           reference_root: Path) -> dict[str, object]:
    reference_campaign_path = reference_root / "campaign.json"
    reference = json.loads(reference_campaign_path.read_text(encoding="utf-8"))
    reference_case_manifests = [item.get("case_manifest_sha256")
                                for item in reference.get("cases", [])]
    if (not reference_case_manifests or None in reference_case_manifests or
            any(not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in reference_case_manifests)):
        raise RuntimeError("reference campaign does not bind every case manifest")
    reference_cases = {(item["spatial_order"], item["resolution"], item["phase"]): item
                       for item in reference["cases"]}
    comparisons = []
    failures = []
    for current in cases:
        key = (current["spatial_order"], current["resolution"], current["phase"])
        other = reference_cases.get(key)
        if other is None:
            failures.append(f"missing reference case {key}")
            continue
        current_dir = output / f"{current['case_id']}-{current['case_uuid']}"
        other_dir = reference_root / f"{other['case_id']}-{other['case_uuid']}"
        current_rows = list(csv.DictReader(
            (current_dir / "cartoon_mms.mms.csv").open(encoding="utf-8")))
        other_rows = list(csv.DictReader(
            (other_dir / "cartoon_mms.mms.csv").open(encoding="utf-8")))
        indexed = {(row["operator"], row["mask"]): row for row in other_rows}
        current_index = {(row["operator"], row["mask"]): row for row in current_rows}
        if (len(current_index) != len(current_rows) or len(indexed) != len(other_rows) or
                set(current_index) != set(indexed) or
                current.get("operator_names") != other.get("operator_names")):
            failures.append(f"series/row inventory differs for {key}")
            continue
        for row in current_rows:
            row_key = (row["operator"], row["mask"])
            baseline = indexed.get(row_key)
            if baseline is None:
                failures.append(f"{key} missing row {row_key}")
                continue
            exact_fields = ("count", "nonfinite", "cyl_count", "linfinity",
                            "cyl_linfinity", "shared_linfinity",
                            "shared_delta_linfinity", "independent_linfinity",
                            "independent_delta_linfinity", "rotation_linfinity",
                            "target_abs_rho", "actual_abs_rho", "mask_xor")
            exact = all(row[field] == baseline[field] for field in exact_fields)
            tolerance = (REDUCTION_TOLERANCE_FACTOR * sys.float_info.epsilon *
                         max(1.0, math.log2(int(current["resolution"]) ** 2)))
            numeric = True
            worst = 0.0
            numeric_fields = ["l1", "l2", "shared_l1", "shared_l2",
                              "shared_delta_l1", "shared_delta_l2",
                              "independent_l1", "independent_l2",
                              "independent_delta_l1", "independent_delta_l2"]
            if int(row["cyl_count"]) > 0:
                numeric_fields += ["cyl_l1", "cyl_l2"]
            for field in numeric_fields:
                left, right = float(row[field]), float(baseline[field])
                difference = abs(left - right)
                worst = max(worst, difference)
                numeric &= difference <= tolerance * max(1.0, abs(left), abs(right))
            passed = exact and numeric
            comparisons.append({"case": key, "row": row_key, "passed": passed,
                                "worst_absolute_difference": worst,
                                "relative_tolerance": tolerance})
            if not passed:
                failures.append(f"rank comparison failed {key} {row_key}")
        if current["ownership_fnv1a64"] != other["ownership_fnv1a64"]:
            failures.append(f"ownership hash differs for {key}")
        current_probes = list(csv.DictReader(
            (current_dir / "cartoon_mms.mms.probes.csv").open(encoding="utf-8")))
        other_probes = list(csv.DictReader(
            (other_dir / "cartoon_mms.mms.probes.csv").open(encoding="utf-8")))
        probe_key = lambda row: (row["operator"], row["mask"], row["side"],
                                 row["layer_index"], row["classification"])
        current_probe_index = {probe_key(row): row for row in current_probes}
        other_probe_index = {probe_key(row): row for row in other_probes}
        if (len(current_probe_index) != len(current_probes) or
                len(other_probe_index) != len(other_probes) or
                set(current_probe_index) != set(other_probe_index)):
            failures.append(f"probe inventory differs for {key}")
        else:
            for probe_id, row in current_probe_index.items():
                baseline = other_probe_index[probe_id]
                if any(row[field] != baseline[field] for field in
                       ("target_rho", "actual_rho", "target_z", "actual_z",
                        "global_cell_id", "raw_error")):
                    failures.append(f"probe value differs for {key} {probe_id}")
                    break
        for label, result in (("current", current), ("reference", other)):
            bindings = result.get("rank_bindings", [])
            expected_ranks = int(result.get("mpi_ranks", 0))
            if (len(bindings) != expected_ranks or
                    sorted(item.get("rank") for item in bindings) !=
                    list(range(expected_ranks))):
                failures.append(f"{label} rank binding inventory incomplete for {key}")
    write_atomic(output / "rank_comparison.json", {"schema": SCHEMA,
                                                    "reference": str(reference_root),
                                                    "comparisons": comparisons,
                                                    "failures": failures})
    if failures:
        raise RuntimeError("2-rank/4-rank comparison failed; see rank_comparison.json")
    return {"rank_comparison.json": sha256(output / "rank_comparison.json"),
            "reference_campaign_sha256": sha256(reference_campaign_path),
            "reference_case_manifest_sha256": reference_case_manifests}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--ranks", type=int, choices=(2, 4), required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=(2, 4, 6))
    parser.add_argument("--resolutions", type=int, nargs="+", default=(32, 64, 128, 256))
    parser.add_argument("--phases", type=int, nargs="+", default=tuple(range(8)))
    parser.add_argument("--require-backend", choices=("Serial", "Cuda"))
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--rank-wrapper", type=Path)
    parser.add_argument("--x1min", type=float, default=-2.0)
    parser.add_argument("--x1max", type=float, default=2.0)
    parser.add_argument("--x2min", type=float, default=-2.0)
    parser.add_argument("--x2max", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-campaign", type=Path)
    args = parser.parse_args()
    if (set(args.orders) != {2, 4, 6} or len(args.orders) != 3 or
            set(args.resolutions) != {32, 64, 128, 256} or
            len(args.resolutions) != 4 or set(args.phases) != set(range(8)) or
            len(args.phases) != 8):
        raise RuntimeError("qualification requires exactly orders 2/4/6, resolutions "
                           "32/64/128/256, and phases 0..7")
    args.orders = [2, 4, 6]
    args.resolutions = [32, 64, 128, 256]
    args.phases = list(range(8))
    root = Path(__file__).resolve().parents[3]
    if args.input is None:
        args.input = root / "tst/inputs/z4c_cartoon_derivatives.athinput"
    args.athena = args.athena.resolve()
    args.rank_wrapper = (args.rank_wrapper or
                         (root / "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")).resolve()
    args.build_manifest = (args.build_manifest or
                           (args.athena.parent / "mms_build_manifest.json")).resolve()
    build = json.loads(args.build_manifest.read_text(encoding="utf-8"))
    required_build_keys = {"schema", "source_commit", "source_tree", "kokkos_commit",
                           "source_clean", "backend", "executable_sha256",
                           "configure_cache_sha256", "compiler", "kokkos_runtime",
                           "configure", "build", "translation_units", "slowest_tus"}
    if not required_build_keys.issubset(build) or not build["source_clean"]:
        raise RuntimeError("immutable build manifest is incomplete or not clean")
    if build["executable_sha256"] != sha256(args.athena):
        raise RuntimeError("Athena executable does not match immutable build manifest")
    if args.require_backend is None:
        args.require_backend = build["backend"]
    elif args.require_backend != build["backend"]:
        raise RuntimeError("requested backend conflicts with immutable build manifest")
    if args.require_backend == "Cuda" and args.ranks != 4:
        raise RuntimeError("CUDA+MPI qualification requires exactly four ranks")
    args.domain = (args.x1min, args.x1max, args.x2min, args.x2max)
    if not all(math.isfinite(value) for value in args.domain) or \
       not (args.x1min < args.x1max and args.x2min < args.x2max) or \
       abs(args.x1min + args.x1max) > 32 * sys.float_info.epsilon * \
       max(1.0, abs(args.x1min), abs(args.x1max)) or args.x1max <= 1.0:
        raise RuntimeError("domain must be finite, ordered, and signed-rho symmetric")
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.output.is_dir():
        raise RuntimeError("output is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source = {"commit": git_value(root, "rev-parse", "HEAD"),
              "tree": git_value(root, "rev-parse", "HEAD^{tree}"),
              "kokkos": git_value(root, "rev-parse", "HEAD:kokkos")}
    if git_value(root, "status", "--porcelain"):
        raise RuntimeError("campaign requires a clean source checkout")
    if (build["source_commit"] != source["commit"] or
            build["source_tree"] != source["tree"] or
            build["kokkos_commit"] != source["kokkos"]):
        raise RuntimeError("build manifest source identity does not match driver checkout")
    series_manifest = root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json"
    series_inventory = json.loads(series_manifest.read_text(encoding="utf-8"))
    expected_operators = [item["name"] for item in series_inventory.get("series", [])]
    if series_inventory.get("count") != 171 or len(expected_operators) != 171 or \
       len(set(expected_operators)) != 171:
        raise RuntimeError("frozen runtime series manifest is not exactly 171 unique entries")
    case_count = len(args.orders) * len(args.resolutions) * len(args.phases)
    estimated_bytes = 0
    for order in args.orders:
        nghost = order // 2 + 1
        csv_rows = 171 * (7 + 2 * nghost)
        probe_rows = 171 * (6 + 2 * nghost)
        estimated_bytes += len(args.resolutions) * len(args.phases) * (
            csv_rows * 768 + probe_rows * 384 + (2 + args.ranks) * 1024 * 1024)
    preflight = {"schema": SCHEMA, "state": "preflight", "case_count": case_count,
                 "estimated_output_bytes_upper_bound": estimated_bytes,
                 "free_bytes_before_campaign": shutil.disk_usage(args.output).free,
                 "orders": args.orders, "resolutions": args.resolutions,
                 "phases": args.phases, "ranks": args.ranks,
                 "series_manifest_sha256": sha256(series_manifest)}
    if preflight["free_bytes_before_campaign"] < 2 * estimated_bytes:
        raise RuntimeError("campaign output forecast exceeds half the available space")
    write_atomic(args.output / "preflight.json", preflight)
    cases = [run_case(args, root, source, order, resolution, phase)
             for order in args.orders for resolution in args.resolutions
             for phase in args.phases]
    for result in cases:
        if result.get("operator_names") != expected_operators:
            raise RuntimeError(f"{result['case_id']} operator ordering differs from frozen 171-series manifest")
    convergence_gate(cases, args.output, series_manifest)
    rank_evidence = None
    if args.compare_campaign:
        rank_evidence = compare_rank_campaigns(
            cases, args.output, args.compare_campaign.resolve())
    convergence_artifacts = {name: sha256(args.output / name) for name in
                             ("convergence.json", "convergence.csv",
                              "convergence_rates.pgfplots.dat",
                              "convergence_plot.tex", "preflight.json")}
    if rank_evidence is not None:
        convergence_artifacts.update(rank_evidence)
    write_atomic(args.output / "campaign.json", {"schema": SCHEMA, "source": source,
                                                  "build_manifest": build,
                                                  "build_manifest_sha256":
                                                  sha256(args.build_manifest),
                                                  "environment": {name: os.environ[name]
                                                                  for name in sorted(os.environ)
                                                                  if name.startswith(("SLURM_", "PMI_", "OMPI_", "CUDA_", "KOKKOS_"))},
                                                  "ranks": args.ranks,
                                                  "backend": args.require_backend,
                                                  "reduction_tolerance_factor":
                                                  REDUCTION_TOLERANCE_FACTOR,
                                                  "convergence_artifacts":
                                                  convergence_artifacts,
                                                  "cases": cases})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
