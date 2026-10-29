#!/usr/bin/env python3
"""Read-only comparison of the original Aurora Minkowski resolution controls.

Run from any directory. NumPy and Matplotlib are required. Missing new runs
remain pending; --require-all rejects partial or unbound evidence. Input CSVs are never
rewritten. --new-dir may select either the downloaded campaign/output directory
or a portable copy of its small records. No checkpoints are opened or admitted.

For separately launched controls, repeat --case-dir CASE OUTPUT_DIRECTORY,
for example --case-dir minkowski-n24 /path/to/followup/output. Each directory
uses its own parent/worker-status.json; worker records are never combined.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = next((p for p in HERE.parents if (p / "src/z4c").is_dir()), None)
SOURCE_SHA = "3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f"
BINARY_SHA = "6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef"
CASES = [
    ("original-n16", "n16, dt=0.0005", "minkowski", 16, .0005),
    ("minkowski-n24", "n24, dt=1/3000", "minkowski-n24", 24, 1 / 3000),
    ("minkowski-n32", "n32, dt=0.00025", "minkowski-n32", 32, .00025),
    ("minkowski-n16-half-dt", "n16, dt=0.00025", "minkowski-n16-half-dt", 16, .00025),
]
FIELDS = ("state_deviation", "H_L2", "M_L2", "Z_L2", "null_surface", "null_ghost")
COLORS = ("#222222", "#0072B2", "#D55E00", "#009E73")
SCIENCE_FIELDS = {"min_chi", "min_lapse", "max_state_deviation", "max_null_surface",
                  "max_null_ghost", "max_cfl", "H_L2", "M_L2", "M_physical_L2",
                  "H_exterior_L2", "M_exterior_L2", "Theta_L2", "Z_L2"}


def finite(value):
    return value is not None and np.isfinite(value)


def clean(value):
    """Represent missing/nonfinite diagnostics as JSON null, never zero."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def digest(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def binding(path, worker):
    evidence = digest(path)
    evidence["worker_bound"] = False
    if worker is None:
        return evidence
    matches = [record for case in worker.get("cases", [])
               for name, record in case.get("artifacts", {}).items()
               if Path(name).name == path.name]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous worker artifact: {path.name}")
    if matches:
        if any(evidence[key] != matches[0][key] for key in ("sha256", "bytes")):
            raise ValueError(f"Worker artifact mismatch: {path}")
        evidence["worker_bound"] = True
    return evidence


def load_worker(directory):
    path = directory.parent / "worker-status.json"
    if not path.exists():
        return None, None
    worker = json.loads(path.read_text())
    if worker.get("source_manifest_sha256") != SOURCE_SHA:
        raise ValueError(f"Unexpected numerical source in {path}")
    binaries = worker.get("binaries", {})
    if BINARY_SHA not in binaries.values() or any(v != BINARY_SHA for v in binaries.values()):
        raise ValueError(f"Unexpected executable in {path}")
    return worker, digest(path)


def scientific_gates(worker, prefix, report):
    """Keep worker science thresholds separate from completion/config checks."""
    result = {"status": "unassessed", "checks": [], "other_failed_checks": []}
    if worker is None or report is None:
        return result
    cases = [case for case in worker.get("cases", []) if case.get("name") == prefix]
    if len(cases) > 1:
        raise ValueError(f"Duplicate worker case: {prefix}")
    if not cases:
        return result
    case = cases[0]
    result["worker_case_status"] = case.get("status")
    result["worker_report_status"] = case.get("report_status")
    result["diagnostic_time"] = report.get("diagnostic_time")
    result["diagnostics_current"] = report.get("diagnostics_current")
    for record in case.get("checks", []):
        check = record["check"]
        value = report
        for part in check["path"]:
            value = value[part]
        op, threshold = check["op"], check.get("value")
        if op == "finite":
            passed = finite(value)
        elif op == "eq":
            passed = value == threshold
        elif op in ("ge", "le", "gt", "lt"):
            passed = finite(value) and {"ge": value >= threshold, "le": value <= threshold,
                                        "gt": value > threshold, "lt": value < threshold}[op]
        else:
            raise ValueError(f"Unsupported recorded check operator: {op}")
        if bool(passed) != record["passed"] or clean(value) != clean(record.get("observed")):
            raise ValueError(f"Worker/report check mismatch for {prefix}: {check}")
        verified = {"check": check, "observed": value, "passed": bool(passed)}
        if check["path"][0] in SCIENCE_FIELDS:
            result["checks"].append(verified)
        elif not passed:
            result["other_failed_checks"].append(verified)
    if result["checks"]:
        result["status"] = "failed" if any(not x["passed"] for x in result["checks"]) else "passed_recorded_checks"
    result["note"] = "Science thresholds apply to the report's stated diagnostics and running maxima; passing them alone does not establish endpoint completion or admit a trajectory."
    return clean(result)


def read_history(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(("t", "steps", "wall_seconds", *FIELDS)) <= set(reader.fieldnames or []):
            raise ValueError(f"Missing required history columns: {path}")
        records = list(reader)
    if not records or any(None in row or None in row.values() for row in records):
        raise ValueError(f"Empty or partial CSV record: {path}")
    data = {key: np.array([float(row[key]) for row in records]) for key in records[0]}
    for key in ("t", "steps", "wall_seconds"):
        if not np.all(np.isfinite(data[key])) or np.any(np.diff(data[key]) <= 0):
            raise ValueError(f"Nonfinite or nonincreasing {key}: {path}")
    if np.any(data["steps"] != np.floor(data["steps"])):
        raise ValueError(f"Noninteger accepted-step count: {path}")
    return data


def selected_interval(data, low, high):
    """Longest contiguous amplitude-window segment; earliest wins a tie."""
    y = data["state_deviation"]
    indexes = np.flatnonzero(np.isfinite(y) & (y >= low) & (y <= high) & (y > 0))
    if not len(indexes):
        return indexes
    groups = np.split(indexes, np.flatnonzero(np.diff(indexes) != 1) + 1)
    return max(groups, key=len)


def fit_growth(data, indexes, field):
    t, y = data["t"][indexes], data[field][indexes]
    valid = np.isfinite(y) & (y > 0)
    t, y = t[valid], y[valid]
    fit = {"points": len(t), "times": t, "values": y,
           "method": "unweighted OLS ln(value) = intercept + rate * coordinate_time"}
    initial = data[field][0]
    fit["precision"] = {
        "initial_recorded_magnitude": initial,
        "minimum_fit_value_over_initial": min(y) / initial if len(y) and finite(initial) and initial > 0 else None,
        "samples_within_100_times_initial": int(np.count_nonzero(y <= 100 * initial)) if finite(initial) and initial > 0 else None,
        "note": "The initial residual is an empirical comparison scale, not a rigorous roundoff bound. The factor100 is a sensitivity flag only; no values are removed or floored."}
    if field == "H_L2":
        fit["precision"]["hamiltonian_caveat"] = "H combines differentiated fields and cancellation of nonzero geometric terms. Near the initial roundoff residual, log slopes can reflect cancellation or seed amplitude rather than a resolved physical constraint mode. Compare state/M/Z growth and the later H samples; R² alone does not establish resolution."
    if len(t):
        fit.update(time_range=[t[0], t[-1]], value_range=[min(y), max(y)])
    if len(t) < 3:
        fit.update(status="insufficient_points", rate=None, r_squared=None)
        return clean(fit)
    x, z = t - np.mean(t), np.log(y)
    variance = np.dot(x, x)
    rate = np.dot(x, z - np.mean(z)) / variance
    predicted = np.mean(z) + rate * x
    residual = np.sum((z - predicted) ** 2)
    total = np.sum((z - np.mean(z)) ** 2)
    fit.update(status="fitted", rate=rate,
               intercept=np.mean(z) - rate * np.mean(t),
               rate_standard_error=np.sqrt(residual / (len(t) - 2) / variance),
               r_squared=1 - residual / total if total > 0 else None,
               log10_dynamic_range=np.log10(max(y) / min(y)),
               monotone_nondecreasing=bool(np.all(np.diff(y) >= 0)))
    fit["conditioning"] = {
        "centered_time_sum_of_squares": variance,
        "scaled_design_condition_number": np.linalg.cond(np.column_stack((np.ones(len(t)), x / np.ptp(t)))),
        "standard_error_caveat": "OLS residual standard error is descriptive only; diagnostic samples are correlated, so this is not an independent-sample confidence interval."}
    return clean(fit)


def throughput(data, cells):
    intervals = []
    for i in range(1, len(data["t"])):
        steps = int(data["steps"][i] - data["steps"][i - 1])
        seconds = data["wall_seconds"][i] - data["wall_seconds"][i - 1]
        seconds_per_step = seconds / steps
        intervals.append({"time_start": data["t"][i - 1], "time_end": data["t"][i],
                          "accepted_steps": steps, "seconds": seconds,
                          "seconds_per_step": seconds_per_step,
                          "active_cell_steps_per_second": cells / seconds_per_step,
                          "used": i > 1})
    used = [row["seconds_per_step"] for row in intervals if row["used"]]
    accepted = [row for row in intervals if row["used"]]
    median = float(np.median(used)) if used else None
    return clean({"method": "median history delta-wall/delta-accepted-steps; first interval discarded as startup; includes diagnostics and communication",
                  "intervals": intervals, "used_intervals": len(used),
                  "median_seconds_per_step": median,
                  "median_active_cell_steps_per_second": cells / median if median else None,
                  "accepted_steps_per_interval": sorted({row["accepted_steps"] for row in accepted}),
                  "median_coordinate_interval": np.median([row["time_end"] - row["time_start"] for row in accepted]) if accepted else None,
                  "median_coordinate_time_per_wall_second": np.median([(row["time_end"] - row["time_start"]) / row["seconds"] for row in accepted]) if accepted else None})


def analyze_case(spec, directory, low, high):
    key, label, prefix, n, dt = spec
    summary = {"case": key, "label": label, "n": n, "dt": dt,
               "n_times_dt": n * dt, "active_cells": 19 * n**3, "outcome": "pending",
               "input_directory": str(directory.resolve())}
    history_path, report_path = [directory / (prefix + suffix)
                                 for suffix in (".history.csv", ".report.json")]
    worker, worker_evidence = load_worker(directory)
    summary["worker_record"] = worker_evidence
    summary["inputs"] = [binding(p, worker) for p in (history_path, report_path) if p.exists()]
    summary["original_binary_evidence_verified"] = (worker is not None and len(summary["inputs"]) == 2
                                                     and all(p["worker_bound"] for p in summary["inputs"]))
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    data = read_history(history_path) if history_path.exists() else None
    summary["scientific_gates"] = scientific_gates(worker, prefix, report)
    if report:
        for field, expected in (("n", n), ("cells", 19*n**3), ("kind", "minkowski"),
                                ("backend", "SYCL"), ("mpi_size", 12)):
            if report.get(field) != expected:
                raise ValueError(f"Unexpected {field} in {report_path}")
        if not np.isclose(report["dt"], dt, rtol=1e-13, atol=0):
            raise ValueError(f"Unexpected dt in {report_path}")
        status = report.get("status")
        summary.update(report_status=status,
                       outcome={"completed": "completed", "failed": "failed",
                                "checkpointed": "bounded"}.get(status, "other_reported"),
                       reached_time=report.get("t"), accepted_steps=report.get("steps"),
                       failure_time=report.get("t") if status == "failed" else None,
                       error=report.get("error"), target_time=report.get("target_time"),
                       numerical_breakdown_reported=(status == "failed" and any(
                           text in report.get("error", "").lower() for text in
                           ("nonpositive", "nonfinite", "invalid geometry", "negative metric"))),
                       checkpoint_admitted=report.get("checkpoint_admitted"),
                       report_diagnostic_time=report.get("diagnostic_time"),
                       report_diagnostics_current=report.get("diagnostics_current"),
                       timers_seconds={k: report.get(k) for k in
                                       ("rhs_seconds", "exchange_seconds", "closure_seconds", "diagnostic_seconds")},
                       report_null_maxima={k: report.get(k) for k in ("max_null_surface", "max_null_ghost")})
        if status == "completed" and report["t"] < report["target_time"] - dt * .01:
            raise ValueError(f"Completed report did not reach its endpoint: {report_path}")
    elif data is not None:
        summary["outcome"] = "history_only_unreported"
    if data is None:
        summary["history_status"] = "missing"
        return clean(summary), None
    if report and (data["t"][-1] > report["t"] + dt * .01 or data["steps"][-1] > report["steps"]):
        raise ValueError(f"History extends beyond accepted report state: {history_path}")
    selected = selected_interval(data, low, high)
    summary.update(history_status="present", history_rows=len(data["t"]),
                   last_history_time=data["t"][-1],
                   last_history_values={field: data[field][-1] for field in FIELDS},
                   missing_history_values={field: int(np.count_nonzero(~np.isfinite(data[field]))) for field in FIELDS},
                   history_end_gap=report["t"] - data["t"][-1] if report else None,
                   selected_state_fit_times=data["t"][selected],
                   growth={field: fit_growth(data, selected, field) for field in FIELDS[:4]},
                   throughput=throughput(data, 19*n**3))
    return clean(summary), data


def common_samples(results, histories):
    available = [(r["case"], d) for r, d in zip(results, histories) if d is not None]
    if not available:
        return []
    rows = []
    for t in available[0][1]["t"]:
        values = {}
        for key, data in available:
            index = np.flatnonzero(np.isclose(data["t"], t, rtol=0, atol=1e-10))
            if len(index) != 1:
                break
            values[key] = {field: data[field][index[0]] for field in FIELDS}
        if len(values) == len(available):
            rows.append({"coordinate_time": t, "values": values})
    return clean(rows)


def number(value, digits=5):
    return f"{value:.{digits}g}" if finite(value) else "—"


def write_table(results, path, low, high):
    lines = ["# Original-binary Minkowski resolution controls", "",
             "All times and growth rates use coordinate time. Pending cases have no inferred outcome.", "",
             "| Case | Run outcome | Science gates | Last diagnostics | Reached / failure time | State rate | Points; fit time range; R² | Median s/step | Active cell-steps/s |",
             "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for r in results:
        f = r.get("growth", {}).get("state_deviation", {})
        tr = f.get("time_range", [None, None])
        fit = f"{f.get('points', 0)}; {number(tr[0])}–{number(tr[1])}; {number(f.get('r_squared'))}"
        speed = r.get("throughput", {})
        lines.append(f"| {r['label']} | {r['outcome']} | {r['scientific_gates']['status']} | {number(r.get('last_history_time'))} | {number(r.get('reached_time'))} | {number(f.get('rate'))} | {fit} | {number(speed.get('median_seconds_per_step'))} | {number(speed.get('median_active_cell_steps_per_second'))} |")
    unverified = [r["label"] for r in results if r.get("history_status") == "present" and not r["original_binary_evidence_verified"]]
    if unverified:
        lines[0] = "# Minkowski resolution controls: unverified input evidence"
        lines += ["", "Source/binary identity and complete worker artifact bindings are not verified for: " + "; ".join(unverified) + ". These cases are provisional and cannot support a fixed-source comparison."]
    cadence = "; ".join(r["label"] + ": " + "/".join(map(str, r.get("throughput", {}).get("accepted_steps_per_interval", [])))
                        for r in results if r.get("throughput", {}).get("used_intervals", 0))
    lines += ["", f"The fit uses the longest contiguous set of positive state deviations in [{low:g}, {high:g}], with at least three recorded samples. H, M and Z use the same time selection; their individual fits, exact values, standard errors and R² are in analysis.json. These are discrete growth estimates, not convergence orders. Fit windows can differ between grids.",
              "", "Throughput excludes the first history interval and includes diagnostic/communication overhead. Active cell-steps/s counts all 19 blocks across the node; it is not per GPU. Component timers remain separate in the JSON because closure includes communication and cumulative timers are MPI maxima, not an additive cost partition.",
              "", "Recorded accepted steps per measured diagnostic interval: " + (cadence or "unavailable") + ". Different diagnostic cadences change overhead per step; these timings do not isolate kernel efficiency. Median coordinate-time advance per wall second is also retained in the JSON.",
              "", "Plots end at recorded diagnostic times. No final-state value is imputed from stale report diagnostics, and no curve is extended to a later rejection time. Null curves are the recorded running maxima. A bounded checkpoint is not a completed endpoint or a stability admission.",
              "", "Run outcome and scientific gates are independent: a trajectory can reach its endpoint while failing a state/constraint tolerance, or fail numerically while its null closure remains precise. Recorded worker science checks are reevaluated against the report; missing checks remain unassessed. Completion, configuration and checkpoint checks are retained separately. Passing the recorded science checks alone is not a stability qualification.",
              "", "H is the recorded evolved-variable Hamiltonian; M and Z are the recorded Penrose-metric covector norms, all normalized by compactified coordinate volume. When tr_g A drifts, this H need not equal the Hamiltonian of fully reconstructed ADM data. Zero/nonfinite values are omitted on logarithmic axes without a floor; gaps remain gaps.",
              "", "H near the initial residual is sensitive to floating-point cancellation and differentiated roundoff. The JSON reports the minimum fitted value relative to its initial magnitude and flags samples within 100 times that empirical scale, without removing them. This factor is a sensitivity indicator, not a measured error bound. R² does not establish resolved constraint growth; OLS standard errors are descriptive because samples are correlated. Two points always remain insufficient, regardless of their apparent slope.",
              "", "Failure times alone cannot distinguish truncation error from a changed roundoff seed. These controls cannot identify a unique unstable operator. No assertion that spatial resolution is ruled out is generated automatically."]
    path.write_text("\n".join(lines) + "\n")


def plot(results, histories, out, low, high):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
    panels = [("state_deviation", "Stored-field deviation (maximum)"),
              ("H_L2", "Reported Hamiltonian RMS"), ("M_L2", "Momentum RMS (Penrose norm)"),
              ("Z_L2", "Z₄ RMS (Penrose norm)")]
    for r, data, color in zip(results, histories, COLORS):
        if data is None:
            continue
        style = "--" if r["case"] == "minkowski-n16-half-dt" else "-"
        for ax, (field, title) in zip(axes.flat, panels):
            y = np.where(np.isfinite(data[field]) & (data[field] > 0), data[field], np.nan)
            ax.semilogy(data["t"], y, style, color=color, marker=".", label=r["label"])
            ax.set_title(title)
        fit = r["growth"]["state_deviation"]
        if fit["status"] == "fitted":
            axes[0, 0].scatter(fit["times"], fit["values"], s=36, facecolors="none", edgecolors=color)
        axes[1, 1].plot(data["t"], data["null_surface"] / 1e-15, style, color=color, marker=".")
        axes[1, 1].plot(data["t"], data["null_ghost"] / 1e-15, ":", color=color)
        intervals = [row for row in r["throughput"]["intervals"] if row["used"]]
        axes[1, 2].plot([(x["time_start"] + x["time_end"]) / 2 for x in intervals],
                        [x["active_cell_steps_per_second"] / 1e6 for x in intervals],
                        style, color=color, marker=".")
    axes[0, 0].axhspan(low, high, color="grey", alpha=.08)
    axes[1, 1].set_title("Null closure: recorded running maxima")
    axes[1, 1].set_ylabel("Relative residual / 10⁻¹⁵")
    axes[1, 1].text(.02, .97, "Solid/dashed: surface; dotted: ghost", transform=axes[1, 1].transAxes,
                    va="top", fontsize=8)
    axes[1, 2].set_title("Full-step node throughput after startup")
    axes[1, 2].set_ylabel("Million active cell-steps / wall second")
    axes[1, 2].set_ylim(bottom=0)
    final_diagnostic = max((d["t"][-1] for d in histories if d is not None), default=1)
    for ax in axes.flat:
        ax.set_xlabel("Coordinate time t")
        ax.set_xlim(0, final_diagnostic * 1.025 if final_diagnostic > 0 else 1)
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, bbox_to_anchor=(.5, .945))
    pending = [r["case"] for r in results if r["outcome"] == "pending"]
    unverified = any(d is not None and not r["original_binary_evidence_verified"] for r, d in zip(results, histories))
    title = "Minkowski resolution controls: unverified inputs" if unverified else "Original-binary Minkowski resolution controls"
    fig.suptitle(title + "\n"
                 "12 MPI ranks / 12 GPU tiles • diagnostics only; no endpoint extrapolation", fontsize=14)
    fig.get_layout_engine().set(rect=(0, .055, 1, .88))
    note = "Pending: " + ", ".join(pending) if pending else "Outcomes and rejection times are separate from diagnostic curves; see summary.md."
    fig.text(.015, .012, note + "\nOpen circles: state-fit samples in the shaded amplitude window. Zero/nonfinite log values are not floored.", fontsize=9)
    fig.savefig(out / "resolution-controls.png", dpi=175)
    fig.savefig(out / "resolution-controls.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=HERE.parent / "stability-v1/output")
    downloaded = (REPO / "output/hyperbolic-core-shell/aurora/resolution-control-v1/output"
                  if REPO else HERE / "output")
    parser.add_argument("--new-dir", type=Path, default=HERE / "output" if (HERE / "output").exists() else downloaded)
    parser.add_argument("--case-dir", action="append", nargs=2, default=[],
                        metavar=("CASE", "OUTPUT_DIRECTORY"),
                        help="Override one case's input directory and its own worker provenance; repeat for distinct cases")
    parser.add_argument("--out", type=Path, default=HERE / "analysis")
    parser.add_argument("--state-min", type=float, default=1e-10)
    parser.add_argument("--state-max", type=float, default=1e-3)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    if not 0 < args.state_min < args.state_max:
        parser.error("State window must satisfy 0 < minimum < maximum")
    overrides = {}
    known_cases = {spec[0] for spec in CASES}
    for case, directory in args.case_dir:
        if case not in known_cases:
            parser.error(f"Unknown case {case!r}; choose one of {', '.join(sorted(known_cases))}")
        if case in overrides:
            parser.error(f"Duplicate --case-dir override for {case!r}")
        overrides[case] = Path(directory)
    results, histories = [], []
    for i, spec in enumerate(CASES):
        directory = overrides.get(spec[0], args.baseline_dir if i == 0 else args.new_dir)
        result, data = analyze_case(spec, directory, args.state_min, args.state_max)
        results.append(result)
        histories.append(data)
    baseline_rate = results[0].get("growth", {}).get("state_deviation", {}).get("rate")
    baseline_speed = results[0].get("throughput", {}).get("median_active_cell_steps_per_second")
    for result in results:
        rate = result.get("growth", {}).get("state_deviation", {}).get("rate")
        speed = result.get("throughput", {}).get("median_active_cell_steps_per_second")
        result["comparison_to_original_n16"] = {
            "state_growth_rate_ratio": rate / baseline_rate if finite(rate) and baseline_rate else None,
            "node_throughput_ratio": speed / baseline_speed if finite(speed) and baseline_speed else None,
            "state_growth_rate_divided_by_n": rate / result["n"] if finite(rate) else None,
            "note": "Amplitude-window fits can use different coordinate times; ratios are not convergence orders."}
    if args.require_all and any(r["outcome"] in ("pending", "history_only_unreported") or
                               r.get("history_status") != "present" or
                               not r["original_binary_evidence_verified"] for r in results):
        raise ValueError("Final analysis requires all reports/histories and verified worker source/binary/artifact bindings")
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "analysis_script": digest(Path(__file__)),
               "protocol": digest(HERE / "protocol.md"),
               "expected_source_manifest_sha256": SOURCE_SHA, "expected_binary_sha256": BINARY_SHA,
               "state_fit_amplitude_window": [args.state_min, args.state_max], "cases": results,
               "exact_common_diagnostic_samples": common_samples(results, histories),
               "common_samples_method": "intersection of available recorded coordinate times within 1e-10; no interpolation"}
    (args.out / "analysis.json").write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")
    write_table(results, args.out / "summary.md", args.state_min, args.state_max)
    plot(results, histories, args.out, args.state_min, args.state_max)
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
