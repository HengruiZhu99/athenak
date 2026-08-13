#!/usr/bin/env python3
"""Create compact, hash-bound tables and plots from the Candidate-A/C campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import shutil
import sys
from collections.abc import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: pathlib.Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numeric_rows(path: pathlib.Path, delimiter: str | None = None) -> list[list[str]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        result.append(line.split(delimiter) if delimiter else line.split())
    return result


def row_at(rows: list[list[str]], target: float, time_index: int = 0) -> list[str]:
    matches = [row for row in rows if abs(float(row[time_index]) - target) <= 2e-12]
    if not matches:
        raise RuntimeError(f"no row at target {target}")
    return matches[-1]


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def matching_prefix(path: pathlib.Path, expected: str) -> tuple[int, int]:
    """Return the newline-terminated byte/line prefix with the expected SHA-256."""
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            byte_count += len(line)
            line_count += 1
            if digest.hexdigest() == expected:
                return byte_count, line_count
    raise RuntimeError(f"no append-only prefix of {path} matches {expected}")


def matching_prefix_lines(path: pathlib.Path, expected: str) -> list[bytes]:
    digest = hashlib.sha256()
    lines: list[bytes] = []
    with path.open("rb") as stream:
        for line in stream:
            lines.append(line)
            digest.update(line)
            if digest.hexdigest() == expected:
                return lines
    raise RuntimeError(f"no append-only prefix of {path} matches {expected}")


def parse_profile(path: pathlib.Path) -> tuple[float, list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    match = re.search(r"time=([0-9.eE+-]+)", lines[0])
    if match is None:
        raise RuntimeError(f"missing profile time in {path}")
    names = lines[1].lstrip("# ").split()[2:]
    rows = [[float(value) for value in line.split()[2:]] for line in lines[2:] if line.strip()]
    if not rows or any(len(row) != len(names) for row in rows):
        raise RuntimeError(f"malformed profile {path}")
    return float(match.group(1)), names, rows


def interpolate(xs: list[float], ys: list[float], x: float) -> float:
    if x < xs[0] or x > xs[-1]:
        raise RuntimeError("profile interpolation escaped source domain")
    if x == xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    weight = (x - xs[lo]) / (xs[hi] - xs[lo])
    return (1.0 - weight) * ys[lo] + weight * ys[hi]


def finite(values: Iterable[float], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"nonfinite {label}")


def resource_metrics(path: pathlib.Path) -> dict[str, object]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.strip().split(": ", 1)
            values[key] = value
    required = ("User time (seconds)", "System time (seconds)",
                "Maximum resident set size (kbytes)", "File system inputs",
                "File system outputs", "Exit status")
    if any(key not in values for key in required):
        raise RuntimeError(f"incomplete resource record {path}")
    return {"user_cpu_seconds": float(values[required[0]]),
            "system_cpu_seconds": float(values[required[1]]),
            "maximum_rss_kbytes": int(values[required[2]]),
            "filesystem_inputs": int(values[required[3]]),
            "filesystem_outputs": int(values[required[4]]),
            "resource_exit_status": int(values[required[5]])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--preflight-root", type=pathlib.Path, action="append", default=[])
    p.add_argument("--validation-root", action="append", default=[], metavar="NAME=PATH")
    args = p.parse_args()
    root = args.run_root.resolve()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"refusing existing output {out}")
    out.mkdir(parents=True)
    manifest = json.loads((root / "manifest.json").read_text())
    outcomes = json.loads((root / "outcomes.json").read_text())
    terminal = json.loads((root / "terminal.json").read_text())
    if terminal.get("schema") != manifest.get("schema") + "_TERMINAL":
        raise RuntimeError("terminal schema mismatch")
    if terminal.get("outcomes_sha256") != sha256(root / "outcomes.json"):
        raise RuntimeError("terminal/outcomes hash mismatch")
    if terminal.get("completed_schedule_items") != len(outcomes):
        raise RuntimeError("terminal outcome count mismatch")
    for index, outcome in enumerate(outcomes):
        if outcome.get("sequence") != index or manifest["schedule"][index] != {
                "gauge": outcome.get("gauge"), "resolution": outcome.get("resolution"),
                "sequence": index, "target": outcome.get("target")}:
            raise RuntimeError(f"schedule/outcome mismatch at sequence {index}")
    by_case: dict[str, list[dict[str, object]]] = {}
    runtime_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    bulk_rows: list[dict[str, object]] = []
    prefix_rows: list[dict[str, object]] = []
    profile_difference_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    gauge_comparison_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for outcome in outcomes:
        for key in ("command", "stdout", "stderr", "resource"):
            path = root / outcome[key]
            current_digest = sha256(path)
            recorded_digest = outcome[f"{key}_sha256"]
            hash_valid = current_digest == recorded_digest
            if not hash_valid and outcome["classification"] == "complete":
                raise RuntimeError(f"hash mismatch for completed outcome artifact {path}")
            artifact_rows.append({"sequence": outcome["sequence"], "role": key,
                                  "path": outcome[key], "bytes": path.stat().st_size,
                                  "recorded_sha256": recorded_digest,
                                  "current_sha256": current_digest,
                                  "hash_valid": hash_valid})
        if outcome["classification"] == "complete":
            if abs(float(outcome["validation"]["accepted_time"]) - float(outcome["target"])) > 2e-12:
                raise RuntimeError(f"accepted-time mismatch for sequence {outcome['sequence']}")
            checkpoint = root / "cases" / outcome["case_id"] / outcome["validation"]["checkpoint"]
            if sha256(checkpoint) != outcome["validation"]["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch {checkpoint}")
            artifact_rows.append({"sequence": outcome["sequence"], "role": "restart_checkpoint",
                                  "path": str(checkpoint.relative_to(root)),
                                  "bytes": checkpoint.stat().st_size,
                                  "recorded_sha256": outcome["validation"]["checkpoint_sha256"],
                                  "current_sha256": sha256(checkpoint), "hash_valid": True})
        by_case.setdefault(outcome["case_id"], []).append(outcome)
        resource = resource_metrics(root / outcome["resource"])
        runtime_rows.append({key: outcome.get(key) for key in (
            "sequence", "case_id", "gauge", "resolution", "target", "classification",
            "exit_status", "wall_seconds", "terminal_reason")} | resource)

    for case_id, case_outcomes in by_case.items():
        gauge = case_outcomes[0]["gauge"]
        resolution = case_outcomes[0]["resolution"]
        case = root / "cases" / case_id
        basename = f"z4c_{case_id}"
        history_rows = numeric_rows(case / f"{basename}.user.hst")
        constraints = numeric_rows(case / f"{basename}.z4c.user.hst")
        gauge_rows = numeric_rows(case / "gauge_source_diagnostics.csv", ",")
        horizons = numeric_rows(case / f"{basename}.horizon_summary_0.txt")
        shape_text = (case / f"{basename}.horizon_shape_0.txt").read_text(encoding="utf-8")
        shape_times = [float(x) for x in re.findall(r"Time = ([0-9.eE+-]+)", shape_text)]
        cumulative = {
            "history": case / f"{basename}.user.hst",
            "constraints": case / f"{basename}.z4c.user.hst",
            "gauge_diagnostics": case / "gauge_source_diagnostics.csv",
            "horizon": case / f"{basename}.horizon_summary_0.txt",
        }
        for role, path in cumulative.items():
            artifact_rows.append({"sequence": "final", "role": f"final_{role}",
                                  "path": str(path.relative_to(root)),
                                  "bytes": path.stat().st_size, "recorded_sha256": "",
                                  "current_sha256": sha256(path), "hash_valid": "post_terminal"})
        for suffix in ("horizon_shape_0.txt", "horizon_verbose_0.txt"):
            path = case / f"{basename}.{suffix}"
            if path.exists():
                artifact_rows.append({"sequence": "final", "role": suffix.removesuffix(".txt"),
                                      "path": str(path.relative_to(root)),
                                      "bytes": path.stat().st_size, "recorded_sha256": "",
                                      "current_sha256": sha256(path), "hash_valid": "post_terminal"})
        h0 = [float(x) for x in row_at(history_rows, 0.0)]
        c0 = [float(x) for x in row_at(constraints, 0.0)]
        g0_raw = row_at(gauge_rows, 0.0)
        g0 = [float(x) if i != 2 else math.nan for i, x in enumerate(g0_raw)]
        finite(h0, f"initial history {case_id}")
        finite(c0, f"initial constraints {case_id}")
        finite((value for i, value in enumerate(g0) if i != 2),
               f"initial gauge diagnostics {case_id}")
        initial = {"case_id": case_id, "gauge": gauge, "resolution": resolution,
                   "target": 0.0, "accepted_cycle": 0, "checkpoint_sha256": ""}
        state_rows.append(initial | {"dt": h0[1], "minimum_lapse": h0[2],
            "minimum_chi": h0[3], "minimum_metric_minor": h0[4],
            "maximum_shift": h0[5], "maximum_conformal_gamma": h0[6],
            "invalid_points": h0[7]})
        constraint_rows.append(initial | {"dt": c0[1], "collective_norm2": c0[2],
            "hamiltonian_norm2": c0[3], "momentum_norm2": c0[4], "z_norm2": c0[5],
            "mx_norm2": c0[6], "my_norm2": c0[7], "mz_norm2": c0[8],
            "theta_norm": c0[9], "volume": c0[10]})
        offset = 9
        for measure in ("coordinate", "proper"):
            for region in ("global", "core", "ah_shell", "exterior"):
                source_rows.append(initial | {"measure": measure, "region": region,
                    "volume": g0[offset], "gamma_rms": g0[offset+1],
                    "chi_gradient_rms": g0[offset+2], "lapse_gradient_rms": g0[offset+3],
                    "damping_rms": g0[offset+4]})
                offset += 5
        for outcome in case_outcomes:
            if outcome["classification"] != "complete":
                target = float(outcome["target"])
                h = [float(x) for x in row_at(history_rows, target)]
                c = [float(x) for x in row_at(constraints, target)]
                g_raw = row_at(gauge_rows, target)
                g = [float(x) if i != 2 else math.nan for i, x in enumerate(g_raw)]
                failure_rows.append({"sequence": outcome["sequence"], "case_id": case_id,
                    "gauge": gauge, "resolution": resolution, "target": target,
                    "classification": outcome["classification"],
                    "application_exit_status": outcome["exit_status"],
                    "terminal_reason": outcome.get("terminal_reason", ""),
                    "history_time": h[0], "dt": h[1], "minimum_lapse": h[2],
                    "minimum_chi": h[3], "minimum_metric_minor": h[4],
                    "maximum_shift": h[5], "maximum_conformal_gamma": h[6],
                    "invalid_points": h[7], "collective_norm2": c[2],
                    "hamiltonian_norm2": c[3], "momentum_norm2": c[4],
                    "gauge_profile": g_raw[2]})
                continue
            for role, path in cumulative.items():
                expected = outcome["validation"][f"{role}_sha256"]
                prefix_bytes, prefix_lines = matching_prefix(path, expected)
                prefix_rows.append({"sequence": outcome["sequence"], "case_id": case_id,
                                    "target": outcome["target"], "role": role,
                                    "path": str(path.relative_to(root)), "sha256": expected,
                                    "prefix_bytes": prefix_bytes, "prefix_lines": prefix_lines})
            target = float(outcome["target"])
            h = [float(x) for x in row_at(history_rows, target)]
            c = [float(x) for x in row_at(constraints, target)]
            g_raw = row_at(gauge_rows, target)
            g = [float(x) if i != 2 else math.nan for i, x in enumerate(g_raw)]
            ah = [float(x) for x in row_at(horizons, target, 1)]
            finite(h, f"history {case_id} T={target}")
            finite(c, f"constraints {case_id} T={target}")
            finite((value for i, value in enumerate(g) if i != 2),
                   f"gauge diagnostics {case_id} T={target}")
            finite(ah, f"horizon {case_id} T={target}")
            common = {"case_id": case_id, "gauge": gauge, "resolution": resolution,
                      "target": target, "accepted_cycle": outcome["validation"]["accepted_cycle"],
                      "checkpoint_sha256": outcome["validation"]["checkpoint_sha256"]}
            state_rows.append(common | {
                "dt": h[1], "minimum_lapse": h[2], "minimum_chi": h[3],
                "minimum_metric_minor": h[4], "maximum_shift": h[5],
                "maximum_conformal_gamma": h[6], "invalid_points": h[7]})
            constraint_rows.append(common | {
                "dt": c[1], "collective_norm2": c[2], "hamiltonian_norm2": c[3],
                "momentum_norm2": c[4], "z_norm2": c[5], "mx_norm2": c[6],
                "my_norm2": c[7], "mz_norm2": c[8], "theta_norm": c[9],
                "volume": c[10]})
            regions = ("global", "core", "ah_shell", "exterior")
            measures = ("coordinate", "proper")
            offset = 9
            for measure in measures:
                for region in regions:
                    source_rows.append(common | {"measure": measure, "region": region,
                        "volume": g[offset], "gamma_rms": g[offset+1],
                        "chi_gradient_rms": g[offset+2], "lapse_gradient_rms": g[offset+3],
                        "damping_rms": g[offset+4]})
                    offset += 5
            found = any(abs(value - target) <= 2e-12 for value in shape_times)
            horizon_rows.append(common | {"finder_authority": "found_shape_record" if found else "not_found",
                "mass": ah[2], "spin": ah[6], "area": ah[7], "hrms": ah[8],
                "hmean": ah[9], "mean_radius": ah[10], "minimum_radius": ah[11]})

        for path in sorted((case / "tab").glob(f"{basename}.z4c.*.tab")):
            parse_profile(path)

    profiles_by_key: dict[tuple[str, str, float], tuple[list[str], list[list[float]]]] = {}
    for case_id, case_outcomes in by_case.items():
        gauge = str(case_outcomes[0]["gauge"])
        resolution = str(case_outcomes[0]["resolution"])
        case = root / "cases" / case_id
        basename = f"z4c_{case_id}"
        for path in sorted((case / "tab").glob(f"{basename}.z4c.*.tab")):
            profile_time, names, rows = parse_profile(path)
            profiles_by_key[(gauge, resolution, profile_time)] = (names, rows)
    for gauge in ("standard", "candidate_a", "candidate_c"):
        times = sorted(set(key[2] for key in profiles_by_key if key[0] == gauge))
        for target in times:
            for coarse, fine in (("R0", "R1"), ("R1", "R2")):
                if (gauge, coarse, target) not in profiles_by_key or (gauge, fine, target) not in profiles_by_key:
                    continue
                cnames, crows = profiles_by_key[(gauge, coarse, target)]
                fnames, frows = profiles_by_key[(gauge, fine, target)]
                if cnames != fnames:
                    raise RuntimeError("profile field roster drift")
                cxs = [row[0] for row in crows]
                fxs = [row[0] for row in frows]
                for field_index, field in enumerate(cnames[1:], 1):
                    differences = [crow[field_index] - interpolate(
                        fxs, [row[field_index] for row in frows], crow[0]) for crow in crows]
                    finite(differences, f"{gauge} {coarse}/{fine} {field} profile difference")
                    profile_difference_rows.append({"gauge": gauge, "time": target,
                        "coarse": coarse, "fine": fine, "field": field,
                        "linf": max(abs(value) for value in differences),
                        "rms": math.sqrt(sum(value * value for value in differences) / len(differences))})

    constraint_metrics = ("collective_norm2", "hamiltonian_norm2", "momentum_norm2",
                          "z_norm2", "theta_norm")
    state_metrics = ("minimum_lapse", "minimum_chi", "minimum_metric_minor",
                     "maximum_shift", "maximum_conformal_gamma")
    horizon_metrics = ("mass", "area", "hrms", "mean_radius", "minimum_radius")
    for gauge in ("standard", "candidate_a", "candidate_c"):
        targets = sorted(set(row["target"] for row in state_rows if row["gauge"] == gauge))
        for target in targets:
            for source_name, rows, metrics in (("constraints", constraint_rows, constraint_metrics),
                                                ("state", state_rows, state_metrics),
                                                ("horizon", horizon_rows, horizon_metrics)):
                selected = {(row["resolution"]): row for row in rows
                            if row["gauge"] == gauge and row["target"] == target}
                if set(selected) != {"R0", "R1", "R2"}:
                    continue
                for metric in metrics:
                    values = [float(selected[level][metric]) for level in ("R0", "R1", "R2")]
                    finite(values, f"resolution trend {gauge} {target} {metric}")
                    resolution_rows.append({"gauge": gauge, "time": target,
                        "source": source_name, "metric": metric, "R0": values[0],
                        "R1": values[1], "R2": values[2],
                        "strictly_decreasing": values[0] > values[1] > values[2],
                        "absolute_error_to_one_decreasing": (abs(values[0]-1) > abs(values[1]-1)
                                                             > abs(values[2]-1)) if metric == "mass" else ""})
    for candidate in ("candidate_a", "candidate_c"):
        for resolution in ("R0", "R1", "R2"):
            targets = sorted(set(row["target"] for row in state_rows
                                 if row["gauge"] == candidate and row["resolution"] == resolution))
            for target in targets:
                for source_name, rows, metrics in (("constraints", constraint_rows, constraint_metrics),
                                                    ("state", state_rows, state_metrics),
                                                    ("horizon", horizon_rows, horizon_metrics)):
                    baseline_matches = [row for row in rows if row["gauge"] == "standard"
                                        and row["resolution"] == resolution and row["target"] == target]
                    trial_matches = [row for row in rows if row["gauge"] == candidate
                                     and row["resolution"] == resolution and row["target"] == target]
                    if len(baseline_matches) != 1 or len(trial_matches) != 1:
                        continue
                    baseline, trial = baseline_matches[0], trial_matches[0]
                    for metric in metrics:
                        lhs, rhs = float(trial[metric]), float(baseline[metric])
                        finite((lhs, rhs), f"gauge comparison {candidate} {metric}")
                        gauge_comparison_rows.append({"candidate": candidate,
                            "resolution": resolution, "time": target, "source": source_name,
                            "metric": metric, "candidate_value": lhs, "baseline_value": rhs,
                            "absolute_difference": lhs-rhs,
                            "relative_difference": (lhs-rhs)/rhs if rhs != 0 else ""})

    write_csv(out / "runtime.csv", list(runtime_rows[0]), runtime_rows)
    write_csv(out / "accepted_state.csv", list(state_rows[0]), state_rows)
    write_csv(out / "constraints.csv", list(constraint_rows[0]), constraint_rows)
    write_csv(out / "gauge_sources.csv", list(source_rows[0]), source_rows)
    write_csv(out / "horizons.csv", list(horizon_rows[0]), horizon_rows)
    write_csv(out / "artifact_index.csv", list(artifact_rows[0]), artifact_rows)
    write_csv(out / "prefix_hash_ledger.csv", list(prefix_rows[0]), prefix_rows)
    write_csv(out / "profile_differences.csv", list(profile_difference_rows[0]), profile_difference_rows)
    write_csv(out / "resolution_trends.csv", list(resolution_rows[0]), resolution_rows)
    write_csv(out / "gauge_vs_baseline.csv", list(gauge_comparison_rows[0]), gauge_comparison_rows)
    if failure_rows:
        write_csv(out / "failed_terminal_states.csv", list(failure_rows[0]), failure_rows)
    indexed_checkpoints = {row["path"] for row in artifact_rows
                           if row["role"] == "restart_checkpoint"}
    for path in sorted(root.glob("cases/*/*/*")):
        if not path.is_file() or path.suffix not in (".rst", ".bin", ".tab"):
            continue
        relative = str(path.relative_to(root))
        if relative in indexed_checkpoints:
            digest = next(row["current_sha256"] for row in artifact_rows if row["path"] == relative)
        else:
            digest = sha256(path)
        bulk_rows.append({"role": {".rst": "restart", ".bin": "volume", ".tab": "profile"}[path.suffix],
                          "path": relative, "bytes": path.stat().st_size, "sha256": digest,
                          "committed": path.suffix == ".tab"})
    write_csv(out / "bulk_artifact_index.csv", list(bulk_rows[0]), bulk_rows)
    shutil.copy2(root / "manifest.json", out / "manifest.json")
    shutil.copy2(root / "terminal.json", out / "terminal.json")
    shutil.copy2(root / "outcomes.json", out / "outcomes.json")

    preflight_index: list[dict[str, object]] = []
    if args.preflight_root:
        preflight_output = out / "preflights"
        preflight_output.mkdir()
        for preflight_root_argument in args.preflight_root:
            preflight_root = preflight_root_argument.resolve()
            destination = preflight_output / preflight_root.name
            destination.mkdir()
            for name in ("manifest.json", "outcomes.json", "terminal.json"):
                source = preflight_root / name
                if source.exists():
                    shutil.copy2(source, destination / name)
                    preflight_index.append({"root": str(preflight_root), "role": name,
                                            "source_path": str(source),
                                            "bytes": source.stat().st_size,
                                            "sha256": sha256(source)})
            for source in sorted(preflight_root.glob("cases/*/segments/*/*")):
                if not source.is_file() or source.suffix in (".rst", ".bin", ".tab"):
                    continue
                relative = source.relative_to(preflight_root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                preflight_index.append({"root": str(preflight_root),
                    "role": "segment_evidence", "source_path": str(source),
                    "bytes": source.stat().st_size, "sha256": sha256(source)})
        write_csv(out / "preflight_artifact_index.csv", list(preflight_index[0]), preflight_index)
    validation_index: list[dict[str, object]] = []
    if args.validation_root:
        validation_output = out / "validation_artifacts"
        validation_output.mkdir()
        for specification in args.validation_root:
            if "=" not in specification:
                raise RuntimeError("validation root must use NAME=PATH")
            name, source_text = specification.split("=", 1)
            source_root = pathlib.Path(source_text).resolve()
            destination = validation_output / name
            destination.mkdir()
            for source in sorted(source_root.rglob("*")):
                if not source.is_file() or source.suffix in (".rst", ".bin", ".tab"):
                    continue
                relative = source.relative_to(source_root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                validation_index.append({"name": name, "source_path": str(source),
                    "relative_path": str(relative), "bytes": source.stat().st_size,
                    "sha256": sha256(source)})
        write_csv(out / "validation_artifact_index.csv", list(validation_index[0]), validation_index)

    raw = out / "raw_text"
    raw.mkdir()
    for outcome in outcomes:
        segment = raw / f"sequence_{int(outcome['sequence']):03d}"
        segment.mkdir()
        for role in ("command", "stdout", "stderr", "resource"):
            shutil.copy2(root / outcome[role], segment / pathlib.Path(outcome[role]).name)
    for path in sorted(root.glob("orchestration_intervention_*.json")):
        shutil.copy2(path, raw / path.name)
    for path in sorted(root.glob("cases/*/segments/*interrupted*/*")):
        if path.is_file():
            destination = raw / "interrupted_attempts" / path.parent.name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / path.name)
    for case_id, case_outcomes in by_case.items():
        complete = [item for item in case_outcomes if item["classification"] == "complete"]
        if not complete:
            continue
        last = complete[-1]
        case = root / "cases" / case_id
        basename = f"z4c_{case_id}"
        destination = raw / case_id
        destination.mkdir()
        cumulative = {
            "history": case / f"{basename}.user.hst",
            "constraints": case / f"{basename}.z4c.user.hst",
            "gauge_diagnostics": case / "gauge_source_diagnostics.csv",
            "horizon": case / f"{basename}.horizon_summary_0.txt",
        }
        for role, path in cumulative.items():
            prefix = matching_prefix_lines(path, last["validation"][f"{role}_sha256"])
            (destination / path.name).write_bytes(b"".join(prefix))
        for suffix in ("horizon_shape_0.txt", "horizon_verbose_0.txt"):
            path = case / f"{basename}.{suffix}"
            if path.exists():
                shutil.copy2(path, destination / path.name)
        for path in sorted((case / "tab").glob(f"{basename}.z4c.*.tab")):
            shutil.copy2(path, destination / path.name)

    colors = {"standard": "black", "candidate_a": "tab:red", "candidate_c": "tab:blue"}
    styles = {"R0": ":", "R1": "--", "R2": "-"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for gauge in ("standard", "candidate_a", "candidate_c"):
        for resolution in ("R0", "R1", "R2"):
            rows = [r for r in state_rows if r["gauge"] == gauge and r["resolution"] == resolution]
            rows.sort(key=lambda r: r["target"])
            label = f"{gauge} {resolution}"
            axes[0,0].plot([r["target"] for r in rows], [r["minimum_lapse"] for r in rows],
                           color=colors[gauge], ls=styles[resolution], label=label)
            axes[0,1].plot([r["target"] for r in rows], [r["minimum_chi"] for r in rows],
                           color=colors[gauge], ls=styles[resolution])
            cr = [r for r in constraint_rows if r["gauge"] == gauge and r["resolution"] == resolution]
            cr.sort(key=lambda r: r["target"])
            axes[1,0].plot([r["target"] for r in cr], [r["hamiltonian_norm2"] for r in cr],
                           color=colors[gauge], ls=styles[resolution])
            axes[1,1].plot([r["target"] for r in cr], [r["momentum_norm2"] for r in cr],
                           color=colors[gauge], ls=styles[resolution])
    axes[0,0].set_ylabel("min lapse")
    axes[0,1].set_ylabel("min raw chi")
    axes[1,0].set_ylabel("AthenaK H norm2")
    axes[1,1].set_ylabel("AthenaK M norm2")
    for ax in axes.flat:
        ax.set_xlabel("accepted time / M")
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "state_and_constraints.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for gauge in colors:
        for resolution in styles:
            rows = [r for r in horizon_rows if r["gauge"] == gauge and r["resolution"] == resolution
                    and r["finder_authority"] == "found_shape_record"]
            rows.sort(key=lambda r: r["target"])
            axes[0].plot([r["target"] for r in rows], [r["mass"] for r in rows],
                         color=colors[gauge], ls=styles[resolution])
            axes[1].plot([r["target"] for r in rows], [r["area"] for r in rows],
                         color=colors[gauge], ls=styles[resolution])
            axes[2].plot([r["target"] for r in rows], [r["mean_radius"] for r in rows],
                         color=colors[gauge], ls=styles[resolution], label=f"{gauge} {resolution}")
    for ax, label in zip(axes, ("AH mass", "AH area", "AH coordinate mean radius")):
        ax.set_xlabel("accepted time / M"); ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "horizon_evolution.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    source_fields = ("gamma_rms", "chi_gradient_rms", "lapse_gradient_rms", "damping_rms")
    for gauge in colors:
        for resolution in styles:
            rows = [r for r in source_rows if r["gauge"] == gauge
                    and r["resolution"] == resolution and r["measure"] == "proper"
                    and r["region"] == "core"]
            rows.sort(key=lambda r: r["target"])
            for ax, field in zip(axes.flat, source_fields):
                ax.plot([r["target"] for r in rows], [r[field] for r in rows],
                        color=colors[gauge], ls=styles[resolution],
                        label=f"{gauge} {resolution}")
    for ax, field in zip(axes.flat, source_fields):
        ax.set_title(field); ax.set_xlabel("accepted time / M")
        if all(r[field] > 0 for r in source_rows if r["measure"] == "proper" and r["region"] == "core"):
            ax.set_yscale("log")
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7, ncol=2)
    fig.suptitle("Proper-volume RMS shift-source terms, puncture core r <= 0.25 M")
    fig.tight_layout(); fig.savefig(out / "core_gauge_source_evolution.png", dpi=180)
    plt.close(fig)

    common_profile_times = sorted(set.intersection(*(
        {key[2] for key in profiles_by_key if key[0] == gauge and key[1] == resolution}
        for gauge in colors for resolution in styles)))
    if common_profile_times:
        target = common_profile_times[-1]
        fields = ("z4c_alpha", "z4c_betax", "z4c_chi", "z4c_gxx", "z4c_gyy", "z4c_Khat")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for gauge in colors:
            for resolution in styles:
                names, rows = profiles_by_key[(gauge, resolution, target)]
                for ax, field in zip(axes.flat, fields):
                    index = names.index(field)
                    ax.plot([row[0] for row in rows], [row[index] for row in rows],
                            color=colors[gauge], ls=styles[resolution],
                            label=f"{gauge} {resolution}")
        for ax, field in zip(axes.flat, fields):
            ax.set_title(field); ax.set_xlabel("x / M"); ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"x-axis profiles at common time {target:g} M")
        fig.tight_layout(); fig.savefig(out / "gauge_profiles_last_common.png", dpi=180)
        plt.close(fig)

    analysis = {
        "schema": "ATHENAK_Z4C_CANDIDATE_AC_COMPACT_ANALYSIS", "schema_version": 1,
        "campaign_status": terminal["status"], "outcome_count": len(outcomes),
        "complete_outcomes": sum(r["classification"] == "complete" for r in outcomes),
        "failed_outcomes": sum(r["classification"] != "complete" for r in outcomes),
        "outcome_artifact_hash_drifts": sum(row["hash_valid"] is False for row in artifact_rows),
        "prefix_hash_records": len(prefix_rows),
        "profile_difference_records": len(profile_difference_rows),
        "manifest_sha256": sha256(root / "manifest.json"),
        "outcomes_sha256": sha256(root / "outcomes.json"),
        "terminal_sha256": sha256(root / "terminal.json"),
        "analyzer_path": str(pathlib.Path(__file__).resolve()),
        "analyzer_sha256": sha256(pathlib.Path(__file__).resolve()),
        "limitations": [
            "AthenaK history quantities retain their native normalization and labels.",
            "FastFlow found status is inferred only from matching committed shape records.",
            "The compact bundle excludes bulk restart and binary state files but indexes their hashes.",
            "No continuum convergence order is inferred from survival alone.",
        ],
    }
    write_json(out / "analysis.json", analysis)
    sums = []
    for path in sorted(out.rglob("*")):
        if path.name != "SHA256SUMS" and path.is_file():
            sums.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
