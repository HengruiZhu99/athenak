#!/usr/bin/env python3
"""DRAFT: reuse the frozen V4 analysis for original/V3/V4 versus future V5.

No evolution or checkpoint decoding. A supplied V5 campaign must be terminal,
worker-bound and covered by an explicitly pinned preflight admission. Without
V5 arguments it remains pending. Do not mistake native admission for science.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "src/z4c").is_dir())
AURORA = ROOT / "docs/hyperbolic-shift-3d/reproduction/aurora"
BASE = AURORA / "stability-dissipative-minkowski-v1/analysis/analyze.py"
BASE_SHA = "685f65de88ea4065dde8f213f8151c518f4c4aa63411e7f55067e10ef18a666d"
spec = importlib.util.spec_from_file_location("frozen_v4_analysis", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
np = base.np
base.require(base.sha(BASE) == BASE_SHA, "Frozen analysis changed")

TIMER_FIELDS = ["seconds", "rhs_seconds", "exchange_seconds", "closure_seconds", "diagnostic_seconds"]
TIMING_CAVEAT = (
    "Cumulative timing buckets overlap and are independently MPI_MAX reduced; "
    "they cannot be summed or subtracted as disjoint wall time. Closure includes "
    "surface gathers also charged to exchange; V4/V5 add the 21-field residual "
    "gather. V5 additionally includes provisional local halo filling and five "
    "native numerator samples per row. Diagnostics contain a boundary refresh. "
    "Per-accepted-step cumulative values amortize initialization, diagnostics "
    "and failed-stage work, and are not isolated kernel timings. Full-step "
    "history rates include diagnostics; these separate allocations do not "
    "establish a causal GPU throughput improvement."
)


def timers(report, history):
    steps = report["steps"]
    cumulative = {k: report.get(k) for k in TIMER_FIELDS}
    for k, value in cumulative.items():
        base.require(value is None or (type(value) in (int, float) and np.isfinite(value) and value >= 0),
                     "Invalid timing: " + k)
    dt = np.diff(history["t"])[1:]
    ds = np.diff(history["steps"])[1:]
    dw = np.diff(history["wall_seconds"])[1:] if "wall_seconds" in history else np.array([])
    valid = np.isfinite(dw) & (dw > 0) & (ds > 0) if len(dw) else np.array([], dtype=bool)
    return base.clean(dict(cumulative=cumulative,
        cumulative_per_accepted_step={k: v/steps if v is not None and steps else None for k,v in cumulative.items()},
        accepted_steps=steps, full_step_interval_count=int(np.count_nonzero(valid)),
        discarded_initial_interval=True,
        median_history_seconds_per_step=float(np.median(dw[valid]/ds[valid])) if np.any(valid) else None,
        median_history_coordinate_advance_per_second=float(np.median(dt[valid]/dw[valid])) if np.any(valid) else None,
        median_diagnostic_coordinate_interval=float(np.median(np.diff(history["t"]))) if len(history["t"])>1 else None,
        interpretation=TIMING_CAVEAT))


def v5_case(directory, identity_path, admission_path):
    identity = base.load(identity_path)
    for key in ["source_manifest_sha256", "binary_sha256", "plan_sha256", "admission_sha256"]:
        digest = identity.get(key)
        base.require(isinstance(digest, str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),
                     "Missing reviewed identity: " + key)
    base.require(identity.get("source_commit") == "9d72318c14a2099b5aea118cca62e9fbd8773350", "Wrong V5 source commit")
    base.require(base.sha(directory/"plan.json") == identity["plan_sha256"], "Unreviewed V5 plan")
    base.require(base.sha(admission_path) == identity["admission_sha256"], "Unpinned V5 admission")
    admission, binding, plan = map(base.load, [admission_path, directory/"binding.json", directory/"plan.json"])
    base.require(binding["admission"]["sha256"] == identity["admission_sha256"], "Bound admission differs")
    base.require(plan["reviewed"] is True, "V5 plan remains draft")
    command = next(c for c in plan["commands"] if c["name"]=="minkowski")
    executable = command["executable"]
    configuration = dict(executable=executable, checkpoint_schema="HYP3D_MPI_V5",
        scri_dissipation_epsilon=2, scri_dissipation_degree=7, checkpoint_header_bytes=104,
        scri_normal_derivative="sampled_native_numerator6", scri_normal_radial_nodes=6,
        scri_normal_bulk_derivative_order=4)
    base.require(admission["passed"] is True and admission["source_manifest_sha256"]==identity["source_manifest_sha256"]
        and admission["binaries"].get(executable)==identity["binary_sha256"]
        and configuration in admission["normal_derivative_configs"], "V5 configuration was not preflight admitted")
    case_spec = ("v5", "V5: sampled native normal numerator", directory.name,
                 identity["source_manifest_sha256"], identity["binary_sha256"])
    case, history = base.verify_case(case_spec, directory)
    report = base.load(directory/"output/minkowski.report.json")
    base.require(all(report.get(k)==v for k,v in configuration.items() if k!="executable"), "Wrong V5 reported operator")
    base.require(report["algebraic_projection"]=="final_stage_standard" and
        report["scri_angular_stencil"]=="cross_face_centered5" and
        report["scri_dissipation_scheme"]=="degree7_weighted_negative_square", "V5 other discretization differs")
    args = command["args"]
    base.require(args[5:7]==["-","-"] and float(args[4])==.0005 and float(args[8])==.05
        and float(args[12])==2 and report["start_time"]==0 and report["target_time"]==float(args[3]),
        "V5 is not the declared fresh trajectory")
    base.require(case["terminal"]["terminal_verified"], "V5 allocation not terminal/released")
    history_passed = case["all_v4_recorded_trajectory_limits_passed"]
    scientific_pass = (report["status"]=="completed" and case["all_declared_checks_passed"]
        and case["execution_succeeded"] and history_passed)
    case.update(scientifically_admitted_for_this_bounded_test=scientific_pass,
        native_checkpoint_admitted=report["checkpoint_admitted"], target_time=report["target_time"],
        expected_steps=[c["value"] for c in command["checks"] if c["path"]==["steps"] and c["op"]=="eq"],
        preflight_admission=base.record(admission_path), reviewed_identity=base.record(identity_path),
        timing=timers(report,history))
    if report["status"]=="completed": case["outcome"]="completed_science_pass" if scientific_pass else "completed_science_failure"
    return case, history


def plot(out, cases, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=["#333333", "#0072B2", "#009E73", "#CC79A7"]
    fig, axes=plt.subplots(2,2,figsize=(10,7.2),sharex=True)
    panels=[("State deviation",["state_deviation"],1e-8),
            ("Reported formulation Hamiltonian RMS",["H_L2"],1e-7),
            ("Invariant conformal momentum RMS",["M_L2"],1e-7),
            ("Null closure: surface / ghost",["null_surface","null_ghost"],1e-12)]
    for ax,(title,fields,limit) in zip(axes.flat,panels):
        for case,data,color in zip(cases,histories,colors):
            if data is None: continue
            for j,field in enumerate(fields):
                values=np.where(np.isfinite(data[field]) & (data[field]>0),data[field],np.nan)
                ax.semilogy(data["t"],values,color=color,ls="-" if j==0 else "--",marker="o" if j==0 else None,ms=3,label=case["label"] if j==0 else None)
            if fields==["state_deviation"] and case["growth"]["status"]=="fitted":
                fit=case["growth"];t=np.asarray(fit["times"])
                ax.semilogy(t,np.exp(fit["intercept"]+fit["rate"]*t),color=color,lw=4,alpha=.25)
        ax.axhline(limit,color="gray",ls=":",lw=1);ax.set_title(title,fontsize=10)
        ax.grid(alpha=.2);ax.set_xlim(left=0)
    for ax in axes[1]:ax.set_xlabel("Coordinate time")
    axes[0,0].legend(fontsize=8,loc="upper left")
    pending=histories[-1] is None
    fig.suptitle("Minkowski n16, dt=.0005 — "+("V5 pending" if pending else "recorded duration controls"))
    fig.text(.08,.025,"Curves stop at recorded diagnostics; missing values stay missing. Null precision alone is not stability.",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(out/"comparison.png",dpi=180);fig.savefig(out/"comparison.pdf");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-dir",type=Path);parser.add_argument("--identity",type=Path)
    parser.add_argument("--admission",type=Path);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    base.require(not args.out.exists(),"Output directory must be new")
    base.require(bool(args.v5_dir)==bool(args.identity)==bool(args.admission),"Supply all three V5 evidence inputs, or none")
    cases=[];histories=[]
    for case_spec in [base.SPECS[0],base.SPECS[2],base.SPECS[3]]:
        directory=(AURORA/case_spec[2]) if case_spec[0]!="v4" else base.CAMPAIGN/"analysis/raw-v4"
        case,history=base.verify_case(case_spec,directory)
        report=base.load(directory/"output/minkowski.report.json")
        case.update(scientifically_admitted_for_this_bounded_test=False,
                    native_checkpoint_admitted=report["checkpoint_admitted"], timing=timers(report,history))
        cases.append(case);histories.append(history)
    if args.v5_dir:case,history=v5_case(args.v5_dir,args.identity,args.admission)
    else:case,history=dict(case="v5",label="V5: sampled native normal numerator",outcome="pending",evidence_verified=False),None
    cases.append(case);histories.append(history);args.out.mkdir(parents=True)
    base.write(args.out/"comparison.json",dict(analysis=base.record(Path(__file__)),reused_analysis=base.record(BASE),
        scope="Bounded same-grid Minkowski controls; no puncture, 100M, AMR or convergence admission",
        fit_selection="Unchanged frozen growth() function: longest contiguous amplitude window1e-10..1e-3, four samples minimum",
        accuracy="Sampled complete closure has roughly third-order measured manufactured rates; no fourth-order accuracy claim",
        norms="Reported formulation H and invariant conformal M, normalized compactified coordinate-volume RMS; no H fit near roundoff",
        timing_caveat=TIMING_CAVEAT,cases=cases))
    columns=["case","outcome","accepted_endpoint","last_fresh_diagnostic","accepted_steps","growth_rate","growth_points","growth_start","growth_end","science_pass",*TIMER_FIELDS,"closure_amortized_seconds_per_step","median_history_seconds_per_step"]
    with (args.out/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for case in cases:
            row={k:case.get(k) for k in columns};fit=case.get("growth",{});timing=case.get("timing",{})
            row.update(growth_rate=fit.get("rate"),growth_points=fit.get("points"),
                growth_start=(fit.get("time_range")or[None,None])[0],growth_end=(fit.get("time_range")or[None,None])[1],
                science_pass=case.get("scientifically_admitted_for_this_bounded_test"),
                closure_amortized_seconds_per_step=timing.get("cumulative_per_accepted_step",{}).get("closure_seconds"),
                median_history_seconds_per_step=timing.get("median_history_seconds_per_step"),**timing.get("cumulative",{}))
            writer.writerow(row)
    plot(args.out,cases,histories)


if __name__=="__main__":main()
