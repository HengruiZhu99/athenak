#!/usr/bin/env python3
"""Bounded, read-only V1/V2/V3/V4 Minkowski duration comparison.

Requires NumPy/Matplotlib. No checkpoint is opened as an evolved state. Optional
payload verification hashes bytes only. Outputs go into a NEW directory; input
evidence is never rewritten. Without --require-v4, missing V4 stays pending.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
AURORA = CAMPAIGN.parent
ROOT = next(p for p in HERE.parents if (p / "src/z4c").is_dir())
SPECS = [
    ("original", "Original", "stability-v1",
     "3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f",
     "6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef"),
    ("v2", "V2: final tensor projection", "stability-projected-minkowski-v1",
     "8d0eba426c84cdf9997864cf7471a44914cc590b9a4d4ea2efe022b944287c93",
     "362093fc8c998576c7d88a14afdc2c5ebcdcb6a68f408b7fdaf92ad62b7b1486"),
    ("v3", "V3: cross-face angular jets", "stability-angular-minkowski-v1",
     "86b602e7e68fc93e6a76e16c9bd5c4aea9fcef64363670c5f95c6b0adf30fc13",
     "a557b2090b8c41c6999969a008f1731e743eea21caa3fa257af386ab15b1847d"),
    ("v4", "V4: angular dissipation, epsilon=2", "stability-dissipative-minkowski-v1",
     "0e7967f1f33559ed79b9f97d6416dacc296f0f14663d35c682177a9ac07db60f",
     "fc8d8074351a01ade6fd05dcf460555c935cf4ffe66c9bda31cc517679500fff"),
]
PLAN_SHA = "daf10ec4f22cfd961c27582895fb9d1e5cfb205f7d3f71458c6882185f8207f2"
ADMISSION_SHA = "32cc0fed05319e204f41edea3b62f614138f07a801079e6348536a07650713a1"
FIELDS = ["state_deviation", "H_L2", "M_L2", "Z_L2", "Theta_L2",
          "determinant_residual", "tracefree_residual", "null_surface", "null_ghost"]
SCIENCE = {"min_chi", "min_lapse", "max_state_deviation", "H_L2", "M_L2",
           "H_exterior_L2", "M_exterior_L2", "max_null_surface", "max_null_ghost",
           "max_determinant_residual", "max_tracefree_residual", "max_cfl",
           "scri_dissipation_cfl_max"}
COLORS = ["#333333", "#D55E00", "#0072B2", "#009E73"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = path.resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=sha(path), bytes=path.stat().st_size)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def check_value(check, report):
    value = report
    for part in check["path"]:
        value = value[part]
    op, target = check["op"], check.get("value")
    finite = isinstance(value, (float, int)) and np.isfinite(value)
    if op == "finite": passed = finite
    elif op == "eq": passed = value == target
    elif op == "le": passed = finite and value <= target
    elif op == "ge": passed = finite and value >= target
    elif op == "lt": passed = finite and value < target
    elif op == "gt": passed = finite and value > target
    else: raise ValueError("Unsupported check operator: " + op)
    return dict(check=check, observed=value, passed=bool(passed),
                category="science" if check["path"][0] in SCIENCE else "completion_or_configuration")


def read_history(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and all(None not in row and None not in row.values() for row in rows),
            "Empty or partial CSV: " + str(path))
    require(set(["t", "steps", "state_deviation", "H_L2", "M_L2", "null_surface", "null_ghost"]) <= set(rows[0]),
            "Missing scientific history fields: " + str(path))
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    for key in ["t", "steps"]:
        require(np.all(np.isfinite(data[key])) and np.all(np.diff(data[key]) > 0),
                "Nonfinite/nonincreasing " + key)
    require(np.all(data["steps"] == np.floor(data["steps"])), "Noninteger step count")
    return data


def growth(data):
    """Longest contiguous recorded interval in [1e-10,1e-3]; earliest tie."""
    y = data["state_deviation"]
    indexes = np.flatnonzero(np.isfinite(y) & (y >= 1e-10) & (y <= 1e-3))
    groups = np.split(indexes, np.flatnonzero(np.diff(indexes) != 1) + 1)
    indexes = max(groups, key=len)
    t, y = data["t"][indexes], y[indexes]
    result = dict(status="insufficient_points", points=len(t), minimum_points=4,
                  sample_indexes=indexes, times=t, amplitudes=y, rate=None, r_squared=None,
                  selection="longest contiguous recorded state_deviation interval in [1e-10,1e-3]; earliest tie",
                  method="unweighted OLS log(state_deviation)=intercept+rate*coordinate_time")
    if len(t): result["time_range"] = [t[0], t[-1]]
    if len(t) < 4: return clean(result)
    centered = t - t.mean()
    z = np.log(y)
    variance = np.dot(centered, centered)
    rate = np.dot(centered, z - z.mean()) / variance
    residual = np.sum((z - (z.mean() + rate * centered)) ** 2)
    total = np.sum((z - z.mean()) ** 2)
    result.update(status="fitted", rate=rate, intercept=z.mean() - rate*t.mean(),
                  r_squared=1-residual/total if total > 0 else None,
                  descriptive_rate_standard_error=np.sqrt(residual/(len(t)-2)/variance),
                  centered_time_sum_of_squares=variance,
                  log10_dynamic_range=np.log10(y.max()/y.min()),
                  monotone_nondecreasing=bool(np.all(np.diff(y) >= 0)),
                  caveat="Correlated diagnostic samples: OLS error is descriptive, not a confidence interval. Growth rates are not inferred from failure times.")
    return clean(result)


def verify_v4_admission(binding):
    preflight = AURORA / "preflight-dissipative-v1"
    path = preflight / "admission.json"
    require(sha(path) == ADMISSION_SHA == binding["admission"]["sha256"], "V4 admission binding mismatch")
    admission = load(path)
    require(admission["passed"] and admission["source_manifest_sha256"] == SPECS[-1][3], "Wrong V4 admission")
    require(SPECS[-1][4] in admission["binaries"].values(), "Executable not preflight-admitted")
    config = dict(executable=next(k for k,v in binding["binaries"].items() if v == SPECS[-1][4]),
                  checkpoint_schema="HYP3D_MPI_V4", scri_dissipation_epsilon=2.0,
                  scri_dissipation_degree=7, checkpoint_header_bytes=88)
    require(config in admission["scri_dissipation_configs"], "Epsilon-two V4 configuration not admitted")
    references = [record(path)]
    for p, expected in [(preflight/"raw/worker-status.json", admission["worker_status_sha256"]),
                        (preflight/"verification-v2.json", admission["verification_sha256"]),
                        (preflight/"verify.py", admission["verifier_sha256"]),
                        (preflight/"compile-evidence/source-v11-manifest.json", SPECS[-1][3])]:
        require(sha(p) == expected, "V4 admission evidence mismatch: " + str(p))
        references.append(record(p))
    build_path = preflight/"compile-evidence/v11/build-status.json"
    build = load(build_path)
    cache = preflight/"compile-evidence/CMakeCache.txt"
    require(build["status"] == "passed" and build["returncode"] == 0 and
            build["source_manifest_sha256"] == SPECS[-1][3] and
            build["binaries"].get(config["executable"]) == SPECS[-1][4] and
            build["cache_sha256"] == sha(cache), "V4 compile/source/cache mismatch")
    references += [record(build_path), record(cache)]
    for support in admission["supporting_reports"].values():
        path = preflight/"raw/output"/Path(support["path"]).name
        require(sha(path) == support["sha256"], "Admitted supporting report mismatch")
        references.append(record(path))
    return references


def history_gates(history):
    """Apply V4 trajectory limits to every recorded sample; no interpolation."""
    limits = [("state_deviation", "le", 1e-8), ("H_L2", "le", 1e-7),
              ("M_L2", "le", 1e-7), ("null_surface", "le", 1e-12),
              ("null_ghost", "le", 1e-12), ("determinant_residual", "le", 1e-12),
              ("tracefree_residual", "le", 1e-12), ("min_chi", "ge", 1e-12),
              ("min_lapse", "ge", 1e-12), ("cfl_max", "le", .5)]
    result=[]
    for field, op, limit in limits:
        item=dict(field=field, op=op, limit=limit, status="unavailable")
        if field in history:
            values=history[field]
            valid=np.isfinite(values)
            passed=valid & ((values <= limit) if op == "le" else (values >= limit))
            failed=np.flatnonzero(~passed)
            item.update(status="passed_recorded_samples" if np.all(passed) else "failed",
                        samples=len(values), nonfinite_samples=int(np.count_nonzero(~valid)),
                        first_failed_recorded_time=history["t"][failed[0]] if len(failed) else None,
                        extreme=np.max(values) if op == "le" else np.min(values))
        result.append(clean(item))
    return result


def terminal_and_affinity(directory, worker):
    records=worker["affinity"]["records"]
    require(worker["affinity"]["passed"] and len(records) == 12 and
            {r["rank"] for r in records} == set(range(12)) and
            len({r["host"] for r in records}) == 1 and
            all(r["mask"] == f'{r["rank"]//2}.{r["rank"]%2}' and r["hierarchy"] == "COMPOSITE" for r in records),
            "One-node twelve-tile affinity mismatch")
    affinity=directory/"output/affinity.log"
    if affinity.exists():
        saved=[json.loads(line.split("=",1)[1]) for line in affinity.read_text().splitlines() if line.startswith("AFFINITY_JSON=")]
        require(saved == records, "Affinity log/worker mismatch")
    result=dict(one_node_twelve_tiles_verified=True, terminal_verified=False)
    path=directory/"collected.json"
    if not path.exists(): path=directory/"terminal.json"
    release=directory/"allocation-released.json"
    if not path.exists() or not release.exists(): return result
    terminal=load(path);released=load(release)
    require(terminal["job_id"] == released["job_id"] == worker["job_id"], "Terminal/release job mismatch")
    scheduler=terminal.get("scheduler", {})
    state=scheduler.get("job_state",terminal.get("scheduler_state"))
    exit_status=scheduler.get("Exit_status",terminal.get("scheduler_exit_status"))
    require(state == "F" and isinstance(exit_status,int) and released.get("released"), "Allocation not terminal/released")
    result.update(terminal_verified=True, scheduler_state=state, scheduler_exit_status=exit_status,
                  allocation_released=True, records=[record(path),record(release)])
    return result


def verify_case(spec, directory, verify_payloads=False):
    key, label, _, source, binary = spec
    result = dict(case=key, label=label, outcome="pending", evidence_verified=False,
                  source_manifest_sha256=source, binary_sha256=binary)
    needed = [directory/"worker-status.json", directory/"binding.json", directory/"plan.json",
              directory/"output/minkowski.report.json", directory/"output/minkowski.history.csv"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        require(key == "v4", "Missing historical control: " + str(missing))
        result["missing"] = missing
        return result, None
    worker, binding, plan, report = [load(p) for p in needed[:4]]
    case = [c for c in worker["cases"] if c["name"] == "minkowski"]
    command = [c for c in plan["commands"] if c["name"] == "minkowski"]
    require(len(case) == len(command) == 1, "Ambiguous case identity")
    case, command = case[0], command[0]
    require(worker["source_manifest_sha256"] == binding["source_manifest_sha256"] == source,
            "Source mismatch: " + key)
    require(worker["binaries"] == binding["binaries"] and
            worker["binaries"].get(command["executable"]) == binary, "Executable mismatch: " + key)
    require(worker["plan_sha256"] == binding["plan_sha256"] == sha(needed[2]), "Plan binding mismatch")
    require(worker["resources"] == binding["resources"] == plan["resources"], "Resource binding mismatch")
    require(worker["resources"]["nodes"] == 1 and worker["resources"]["ranks"] == 12, "Wrong allocation")
    if (directory/"run_debug.pbs").exists():
        require(sha(directory/"run_debug.pbs") == binding["pbs_sha256"], "PBS binding mismatch")
    require([c["check"] for c in case["checks"]] == command["checks"], "Missing/reordered checks")
    checks = [check_value(c, report) for c in command["checks"]]
    for observed, saved in zip(checks, case["checks"]):
        require(observed["passed"] == saved["passed"] and clean(observed["observed"]) == clean(saved["observed"]),
                "Worker/report gate replay mismatch")
    require(case["report_status"] == report["status"], "Report/native status mismatch")
    evidence = [record(p) for p in needed[:3]]
    artifacts = []
    for name, expected in case["artifacts"].items():
        p = directory/"output"/Path(name).name
        small = p.suffix in [".csv", ".json", ".log"]
        item = dict(remote_path=name, **expected, bytes_verified=False)
        if verify_payloads:
            require(p.exists(), "Requested full payload verification has missing artifact: " + str(p))
        if p.exists() and (small or verify_payloads):
            actual = record(p)
            require(all(actual[k] == expected[k] for k in ["sha256", "bytes"]), "Artifact mismatch: " + str(p))
            item.update(local=actual, bytes_verified=True)
        artifacts.append(item)
    for p in needed[3:]:
        require(any(Path(a["remote_path"]).name == p.name and a["bytes_verified"] for a in artifacts),
                "Scientific input not bound to worker: " + str(p))
    require(all(report[k] == v for k,v in dict(n=16, cells=19*16**3, kind="minkowski", backend="SYCL", mpi_size=12, dt=.0005).items()),
            "Unexpected duration configuration")
    history = read_history(needed[4])
    require(history["t"][-1] == report["diagnostic_time"] <= report["t"], "Fresh diagnostic time mismatch")
    require(history["steps"][-1] <= report["steps"], "Diagnostic after accepted endpoint")
    require(bool(report["diagnostics_current"]) == (report["diagnostic_time"] == report["t"]), "Current/stale mismatch")
    for field in ["H_L2", "M_L2"]:
        require(history[field][-1] == report[field], "Last diagnostic value mismatch: " + field)
    if key == "v4":
        require(sha(needed[2]) == PLAN_SHA, "V4 plan differs from frozen protocol")
        result["admission_references"] = verify_v4_admission(binding)
        require(report["scri_dissipation_epsilon"] == 2 and report["scri_dissipation_degree"] == 7 and
                report["checkpoint_schema"] == "HYP3D_MPI_V4", "Wrong V4 configuration")
    elif (directory/"admission.json").exists():
        path = directory/"admission.json"
        require(sha(path) == binding["admission"]["sha256"], "Historical admission binding mismatch")
        adm = load(path)
        require(adm["passed"] and adm["source_manifest_sha256"] == source and binary in adm["binaries"].values(),
                "Historical admission source/binary mismatch")
        evidence.append(record(path))
    science_passed = all(c["passed"] for c in checks if c["category"] == "science")
    all_passed = all(c["passed"] for c in checks)
    trajectory=history_gates(history)
    trajectory_passed=all(c["status"] == "passed_recorded_samples" for c in trajectory)
    execution_ok=(case.get("returncode") == 0 and case.get("timed_out") is False and case.get("storage_error") is None)
    duration_passed=all_passed and execution_ok and (key != "v4" or trajectory_passed)
    if report["status"] == "failed": outcome = "numerical_failure"
    elif report["status"] == "bounded": outcome = "bounded_science_pass" if science_passed else "bounded_science_failure"
    elif report["status"] == "completed": outcome = "completed_pass" if duration_passed else "completed_gate_failure"
    else: outcome = "unclassified_" + report["status"]
    result.update(outcome=outcome, evidence_verified=True, job_id=worker.get("job_id"),
                  native_status=report["status"], worker_case_status=case["status"],
                  error=report.get("error"), returncode=case.get("returncode"),
                  timed_out=case.get("timed_out"), storage_error=case.get("storage_error"),
                  accepted_endpoint=report["t"], accepted_steps=report["steps"],
                  last_fresh_diagnostic=history["t"][-1], diagnostics_current=report["diagnostics_current"],
                  checkpoint_admitted=report.get("checkpoint_admitted"),
                  checks=checks, all_declared_checks_passed=all_passed, science_checks_passed=science_passed,
                  execution_succeeded=execution_ok, all_v4_recorded_trajectory_limits_passed=trajectory_passed,
                  inputs=evidence, artifacts=artifacts, growth=growth(history),
                  v4_limits_on_recorded_history=trajectory,
                  terminal=terminal_and_affinity(directory,worker),
                  last_fresh={k:history[k][-1] for k in FIELDS if k in history},
                  history_max={k:np.max(history[k]) for k in FIELDS if k in history},
                  unavailable_history_fields=[k for k in FIELDS if k not in history],
                  reported_running_bounds={k:report[k] for k in SCIENCE if k in report})
    return clean(result), history


def plot(out, cases, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=True)
    panels = [("State deviation", ["state_deviation"], 1e-8),
              ("Hamiltonian RMS", ["H_L2"], 1e-7),
              ("Momentum RMS", ["M_L2"], 1e-7),
              ("Algebraic residuals", ["determinant_residual", "tracefree_residual"], 1e-12),
              ("Null closure", ["null_surface", "null_ghost"], 1e-12),
              ("Other constraint RMS", ["Z_L2", "Theta_L2"], None)]
    for ax, (title, fields, threshold) in zip(axes.flat, panels):
        for case, data, color in zip(cases, histories, COLORS):
            if data is None: continue
            for j, field in enumerate(fields):
                if field not in data: continue
                y = np.where(np.isfinite(data[field]) & (data[field] > 0), data[field], np.nan)
                ax.semilogy(data["t"], y, color=color, linestyle="-" if j == 0 else "--",
                            marker="o" if j == 0 else None, markersize=3, linewidth=1.5)
            if title == "State deviation" and case["growth"]["status"] == "fitted":
                f = case["growth"]
                t = np.array(f["times"])
                ax.semilogy(t, np.exp(f["intercept"] + f["rate"]*t), color=color, linewidth=3, alpha=.23)
        if threshold:
            ax.axhline(threshold, color="#888888", linestyle=":", linewidth=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, which="major", alpha=.2)
        ax.set_xlim(0, 1)
    axes[1,0].text(.02, .98, "solid: |det g - 1|; dashed: |tr A|\nOriginal: not recorded", transform=axes[1,0].transAxes,
                   va="top", fontsize=8.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    null_values=np.concatenate([data[field][np.isfinite(data[field]) & (data[field] > 0)]
                                for data in histories if data is not None for field in ["null_surface","null_ghost"]])
    axes[1,1].set_ylim(min(3e-16,np.min(null_values)*.4) if len(null_values) else 3e-16,
                       max(5e-11,np.max(null_values)*50) if len(null_values) else 5e-11)
    axes[1,1].text(.02, .98, "solid: surface; dashed: ghost", transform=axes[1,1].transAxes,
                   va="top", fontsize=8.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    axes[1,2].text(.02, .98, "solid: Z; dashed: Theta", transform=axes[1,2].transAxes,
                   va="top", fontsize=8.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    lo,hi=axes[1,2].get_ylim()
    axes[1,2].set_ylim(lo,hi*20)
    for ax in axes[1]: ax.set_xlabel("Coordinate time")
    handles = [Line2D([0], [0], color=c, linewidth=2, label=s["label"] + (" (pending)" if h is None else ""))
               for s,h,c in zip(cases,histories,COLORS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=2, frameon=False)
    pending = any(h is None for h in histories)
    fig.suptitle("Minkowski n16, dt=0.0005 — " + ("V4 outcome pending" if pending else "recorded duration controls"), y=.99)
    endpoints = []
    for case in cases:
        if case["outcome"] == "pending": continue
        endpoints.append(f'{case["case"]}: {case["outcome"].replace("_"," ")}; last diagnostic {case["last_fresh_diagnostic"]:.4g}, accepted endpoint {case["accepted_endpoint"]:.4g}')
    fig.text(.08,.045,"\n".join(endpoints), fontsize=8)
    fig.text(.08,.012,"Dotted lines: declared limits. Curves stop at recorded diagnostics; no extension to later failure endpoints. Null precision alone is not stability.", fontsize=8)
    fig.subplots_adjust(top=.83,bottom=.20,hspace=.3,wspace=.30)
    fig.savefig(out/"comparison.png", dpi=190)
    fig.savefig(out/"comparison.pdf")
    plt.close(fig)


def self_test():
    data = {"t":np.arange(8.), "state_deviation":np.array([1e-11,1e-9,1e-8,np.nan,1e-7,1e-6,1e-5,1e-4])}
    f = growth(data)
    require(f["sample_indexes"] == [4,5,6,7] and abs(f["rate"]-np.log(10)) < 1e-14, "Contiguous fit test")
    data["state_deviation"][-1] = 1.
    require(growth(data)["status"] == "insufficient_points", "Three-point fit rejection")
    data["state_deviation"][:] = 1e-13
    require(growth(data)["rate"] is None and growth(data)["points"] == 0, "No-window is not zero growth")
    require(not check_value(dict(path=["x"],op="le",value=1),dict(x=None))["passed"], "Missing metric is not passing")
    print("Analysis controls passed: contiguous selection, four-point minimum, missing-rate and missing-metric semantics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-dir", type=Path, default=CAMPAIGN)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-v4", action="store_true")
    parser.add_argument("--verify-payloads", action="store_true", help="Hash available worker-bound checkpoints/failed stages without loading or admitting them")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.out is not None and not args.out.exists(), "--out must name a new directory")
    require(sha(CAMPAIGN/"plan.json") == PLAN_SHA, "Frozen portable V4 plan changed")
    cases, histories = [], []
    for spec in SPECS:
        directory = args.v4_dir if spec[0] == "v4" else AURORA/spec[2]
        case, history = verify_case(spec, directory, args.verify_payloads and spec[0] == "v4")
        cases.append(case)
        histories.append(history)
    require(not args.require_v4 or cases[-1]["evidence_verified"], "V4 terminal evidence pending or unverified")
    require(not args.require_v4 or cases[-1]["terminal"]["terminal_verified"], "V4 terminal/release evidence pending")
    args.out.mkdir(parents=True)
    summary = dict(scope="Bounded n16 Minkowski duration comparison; no 100M, puncture, convergence or AMR admission",
                   analysis=record(Path(__file__)), protocol=record(CAMPAIGN/"protocol.md"),
                   amplitude_window=[1e-10,1e-3], minimum_fit_points=4,
                   norms="Normalized compactified Cartesian coordinate-volume RMS; momentum is invariant conformal M_L2, not radial Mr or physical M.",
                   precision_caveat="Initial H residual is near floating-point cancellation of differentiated nonzero geometric terms. No H growth rate is fitted or interpreted near roundoff.",
                   missing_data="Missing or nonfinite samples stay null/NaN; nonpositive values cannot appear on log axes and are omitted without floors. Original algebraic histories are unavailable.",
                   cases=cases)
    write(args.out/"comparison.json", summary)
    columns=["case","outcome","accepted_endpoint","accepted_steps","last_fresh_diagnostic","diagnostics_current","growth_points","growth_start","growth_end","growth_rate","growth_r_squared","last_state","last_H","last_M"]
    with (args.out/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n");writer.writeheader()
        for case in cases:
            row={k:case.get(k) for k in columns};fit=case.get("growth",{});last=case.get("last_fresh",{})
            row.update(growth_points=fit.get("points"),growth_start=(fit.get("time_range") or [None,None])[0],
                       growth_end=(fit.get("time_range") or [None,None])[1],growth_rate=fit.get("rate"),
                       growth_r_squared=fit.get("r_squared"),last_state=last.get("state_deviation"),last_H=last.get("H_L2"),last_M=last.get("M_L2"))
            writer.writerow(row)
    with (args.out/"recorded-diagnostics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["case","t","steps",*FIELDS],lineterminator="\n");writer.writeheader()
        for case,data in zip(cases,histories):
            if data is None: continue
            for j in range(len(data["t"])):
                writer.writerow(dict(case=case["case"],t=data["t"][j],steps=int(data["steps"][j]),
                                     **{k:clean(data[k][j]) if k in data else None for k in FIELDS}))
    plot(args.out,cases,histories)
    print(json.dumps({c["case"]:dict(outcome=c["outcome"],rate=c.get("growth",{}).get("rate")) for c in cases},indent=2))


if __name__ == "__main__":
    main()
