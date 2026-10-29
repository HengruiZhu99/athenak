#!/usr/bin/env python3
"""Read-only V1/V2/V3 comparison and exact frozen ghost-interpolation replay.

Requires NumPy, Matplotlib, retained raw V3 payloads and a local Serial Kokkos
build. Output must be a new directory. No evolution, checkpoint admission,
clamping, projection or restart is performed on any scientific input.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FROZEN = "ec0dd5afcf9d489e06c1fdf92e802a6fb8c14679"
SOURCE = "86b602e7e68fc93e6a76e16c9bd5c4aea9fcef64363670c5f95c6b0adf30fc13"
BINARY = "a557b2090b8c41c6999969a008f1731e743eea21caa3fa257af386ab15b1847d"
HEADER = struct.Struct("<16s4i3Q2d")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


a = module("history_analysis", HERE.parent/"resolution-control-v1/analyze.py")
s = module("surface_analysis", HERE.parent/"stability-projected-minkowski-v1/analyze.py")
i = module("interpolation_analysis", HERE.parent/"scri-positivity-audit-v1/run_audit.py")
sha, write = s.sha, s.write


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(raw):
    worker = json.loads((HERE/"worker-status.json").read_text())
    binding = json.loads((HERE/"binding.json").read_text())
    plan = json.loads((HERE/"plan.json").read_text())
    report = json.loads((HERE/"output/minkowski.report.json").read_text())
    require(len(worker["cases"]) == len(plan["commands"]) == 1, "unexpected case count")
    case, command = worker["cases"][0], plan["commands"][0]
    require(case["name"] == command["name"] == "minkowski", "case identity mismatch")
    require(worker["source_manifest_sha256"] == binding["source_manifest_sha256"] == SOURCE,
            "source binding mismatch")
    require(worker["binaries"] == binding["binaries"] and set(worker["binaries"].values()) == {BINARY},
            "binary binding mismatch")
    require(binding["plan_sha256"] == worker["plan_sha256"] == sha(HERE/"plan.json"), "plan mismatch")
    require(binding["pbs_sha256"] == sha(HERE/"run_debug.pbs"), "PBS mismatch")
    require(binding["resources"] == worker["resources"] == plan["resources"], "resources mismatch")
    require(binding["admission"]["sha256"] == sha(HERE/"admission.json"), "admission hash mismatch")
    admission = json.loads((HERE/"admission.json").read_text())
    require(admission["passed"] and admission["source_manifest_sha256"] == SOURCE and
            BINARY in admission["binaries"].values(), "wrong preflight admission")
    preflight = HERE.parent/"preflight-angular-v1"
    references = []
    for path, expected in [
            (preflight/"worker-status.json", admission["worker_status_sha256"]),
            (preflight/"verification-at-admission.json", admission["verification_sha256"]),
            (preflight/"compile-evidence/source-v10-manifest.json", SOURCE)]:
        require(sha(path) == expected, "preflight evidence hash mismatch: "+str(path))
        references.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    for support in admission["supporting_reports"].values():
        support_path = preflight/"output"/Path(support["path"]).name
        if support_path.name == "communication.report.json":
            support_path = preflight/"communication-analysis/communication.report.json.gz"
            actual = hashlib.sha256(gzip.decompress(support_path.read_bytes())).hexdigest()
        else:
            actual = sha(support_path)
        require(actual == support["sha256"], "admitted supporting report mismatch")
    build_path = preflight/"compile-evidence/build-status.json"
    build = json.loads(build_path.read_text())
    cache_path = preflight/"compile-evidence/CMakeCache.txt"
    require(build["status"] == "passed" and build["returncode"] == 0 and
            build["source_manifest_sha256"] == SOURCE and
            build["binaries"].get(command["executable"]) == BINARY and
            build["cache_sha256"] == sha(cache_path), "build/source/cache identity mismatch")
    for path in [build_path, cache_path]:
        references.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    require(case["status"] == case["report_status"] == worker["status"] == report["status"] == "failed",
            "failure status mismatch")
    require(case["returncode"] == 2 and case["timed_out"] is False and case["storage_error"] is None,
            "unexpected infrastructure outcome")
    require([c["check"] for c in case["checks"]] == command["checks"], "missing/reordered gate")
    for check in case["checks"]:
        value = report
        for part in check["check"]["path"]:
            value = value[part]
        op, target = check["check"]["op"], check["check"].get("value")
        if op == "finite": passed = np.isfinite(value)
        elif op == "eq": passed = value == target
        elif op == "le": passed = np.isfinite(value) and value <= target
        elif op == "ge": passed = np.isfinite(value) and value >= target
        else: raise ValueError("unsupported gate: "+op)
        require(bool(passed) == check["passed"] and value == check["observed"], "gate replay mismatch")
    require(set(command["artifacts"]) == set(case["artifacts"]), "artifact set mismatch")
    evidence = []
    for remote, expected in case["artifacts"].items():
        path = raw/"output"/Path(remote).name
        actual = dict(sha256=sha(path), bytes=path.stat().st_size)
        require(actual == expected, "raw artifact hash/size mismatch: "+str(path))
        retained = HERE/"output"/path.name
        if retained.exists(): require(sha(retained) == actual["sha256"], "package/raw mismatch")
        evidence.append(dict(remote_path=remote, local_path=str(path), **actual,
                             worker_bound=True, retained_in_package=retained.exists()))
    for name in ["plan.json", "binding.json", "worker-status.json", "run_debug.pbs", "allocation-released.json"]:
        require(sha(raw/name) == sha(HERE/name), "package metadata mismatch: "+name)
    require(sha(raw/"output/affinity.log") == sha(HERE/"output/affinity.log"), "affinity raw mismatch")
    affinity = [json.loads(line.split("=", 1)[1]) for line in (HERE/"output/affinity.log").read_text().splitlines()
                if line.startswith("AFFINITY_JSON=")]
    require(affinity == worker["affinity"]["records"] and worker["affinity"]["passed"], "affinity mismatch")
    require(len(affinity) == 12 and {r["rank"] for r in affinity} == set(range(12)) and
            len({r["host"] for r in affinity}) == 1 and
            all(r["mask"] == f'{r["rank"]//2}.{r["rank"]%2}' and r["hierarchy"] == "COMPOSITE"
                for r in affinity), "rank/tile mapping mismatch")
    terminal = json.loads((HERE/"terminal.json").read_text())
    released = json.loads((HERE/"allocation-released.json").read_text())
    collected = json.loads((raw/"collected.json").read_text())
    require(all(collected["scheduler"][k] == v for k,v in terminal["scheduler"].items()) and
            collected["job_id"] == terminal["job_id"] and collected["collected"] == terminal["collected"],
            "whitelisted terminal facts mismatch raw collection")
    require(terminal["job_id"] == released["job_id"] == worker["job_id"] and
            terminal["scheduler"]["job_state"] == "F" and terminal["scheduler"]["Exit_status"] == 1,
            "terminal failure/release mismatch")
    return report, evidence, dict(checks_recomputed=len(case["checks"]),
        failed_checks=[c["check"] for c in case["checks"] if not c["passed"]],
        all_recorded_checks_reproduced=True, terminal_exit_status=1, allocation_released=True,
        one_node_twelve_tiles_verified=True, source_manifest_sha256=SOURCE, binary_sha256=BINARY,
        plan_pbs_admission_verified=True, preflight_references=references)


def replay(out, build, n, stage, saved, failed):
    provenance = dict(frozen_numerical_commit=FROZEN, source_comparison=[],
        audit_cpp_sha256=sha(i.HERE/"scri_interpolation_audit.cpp"))
    for relative in i.HEADERS:
        frozen = subprocess.check_output(["git", "show", f"{FROZEN}:{relative}"], cwd=ROOT)
        require(frozen == (ROOT/relative).read_bytes(), "audit header changed: "+relative)
        provenance["source_comparison"].append(dict(path=relative, sha256=sha(ROOT/relative),
                                                    identical_to_frozen=True))
    flags_path = build/"tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_mpi.dir/flags.make"
    flags = dict((key, shlex.split(value)) for key, value in
                 (line.split(" = ", 1) for line in flags_path.read_text().splitlines() if " = " in line))
    provenance["flags_file"] = dict(path=str(flags_path), sha256=sha(flags_path))
    with tempfile.TemporaryDirectory(prefix="v3-scri-readonly-") as folder:
        tmp = Path(folder)
        exe = tmp/"audit"
        command = (["/usr/bin/c++"] + flags["CXX_DEFINES"] + flags["CXX_INCLUDES"] + flags["CXX_FLAGS"]
                   + [i.HERE/"scri_interpolation_audit.cpp", "-o", exe])
        command += [build/f"kokkos/{part}/src/libkokkos{part}.a" for part in ["containers", "algorithms", "core", "simd"]]
        write(out/"audit-build.json", i.run(command, timeout=120))
        controls = []
        for name in ["positive_identity", "negative_chi", "indefinite_metric", "nonfinite_auxiliary"]:
            size = 6
            data = np.zeros((6*size*size, 31), dtype="<f8"); data[:,0] = 1
            if name == "negative_chi": data[:,0] = -1
            elif name == "indefinite_metric": data[:,1] = -2
            elif name == "nonfinite_auxiliary": data[:,30] = np.nan
            payload, result = tmp/(name+".bin"), tmp/(name+".json")
            with payload.open("wb") as f:
                f.seek(8*31*19*size**3); data.tofile(f)
            execution = i.run([exe, size, payload, result])
            result_data = json.loads(result.read_text())
            expected = {k:0 for k in result_data["rejection_counts"]}
            if name != "positive_identity":
                key = dict(negative_chi="nonpositive_chi", indefinite_metric="metric_invalid",
                           nonfinite_auxiliary="nonfinite_field")[name]
                expected[key] = result_data["unique_global_boundary_queries"]
            require(result_data["rejection_counts"] == expected, "audit control failed: "+name)
            controls.append(dict(name=name, passed=True, counts=expected, seconds=execution["seconds"]))
        write(out/"audit-controls.json", controls)
        write(out/"audit-execution.json", i.run([exe, n, failed, out/"interpolation-audit.json"]))
    data = json.loads((out/"interpolation-audit.json").read_text())
    details = []
    for query in data["rejected"]:
        value = i.interpolate(stage, query)
        require(np.allclose(value, query["interpolated_fields"], rtol=0, atol=2e-13),
                "independent interpolation replay mismatch")
        previous = i.interpolate(saved, query)
        eigen = np.linalg.eigvalsh(i.metric(previous))
        weights = np.asarray(query["weights"])
        details.append(dict(query_id=query["query_id"], boundary_xyz=query["boundary_xyz"],
            donor_block=query["donor_block"], donor_face=query["donor_face"], donor_local=query["donor_local"],
            target_ghosts_block_i_j_k=query["targets"], interpolated_check=query["check"],
            eigenvalues=np.linalg.eigvalsh(np.asarray(query["check"]["metric"])),
            minimum_donor_eigenvalue=np.linalg.eigvalsh(i.metric(stage[query["donor_surface_ids"]])).min(),
            donor_chi_range=[query["donor_field_min"][0], query["donor_field_max"][0]],
            minimum_donor_determinant=query["donor_min_determinant"],
            weights=dict(sum=weights.sum(), l1=abs(weights).sum(), minimum=weights.min(), maximum=weights.max()),
            rollback_same_query=dict(chi=previous[0], determinant=np.linalg.det(i.metric(previous)), eigenvalues=eigen)))
    write(out/"interpolation-summary.json", dict(
        scope="Immediate rejection localization, not identification of the first growing mode",
        host_replay="Frozen stencil and Geometry on CPU binary64; not a bitwise SYCL replay",
        query_count_scope="Global exact-coordinate deduplication; not sum of rank-local invalid counters",
        q_zero_for_invalid_metric="Geometry returns before constructing its inverse; q=0 is consequent, not independent",
        nodal_surface=data["nodal_surface"], unique_global_boundary_queries=data["unique_global_boundary_queries"],
        rejected_exact_coordinate_queries=data["rejected_queries"],
        distinct_failed_directions_at_12_decimals=len({tuple(round(x,12) for x in q["boundary_xyz"]) for q in details}),
        rejection_counts=data["rejection_counts"], interpolant_ranges=data["interpolants"], rejected=details))
    write(out/"audit-provenance.json", provenance)


def compare(out):
    campaigns = ["stability-v1", "stability-projected-minkowski-v1", HERE.name]
    histories, comparisons, inputs = [], [], []
    for label, campaign in zip(["original_v1", "projected_v2", "angular_v3"], campaigns):
        base = HERE.parent/campaign
        worker = json.loads((base/"worker-status.json").read_text())
        case = worker["cases"][0]
        report = json.loads((base/"output/minkowski.report.json").read_text())
        for suffix in [".history.csv", ".report.json"]:
            path = base/"output"/("minkowski"+suffix)
            record = a.binding(path, worker)
            require(record["worker_bound"], "unbound comparison input")
            inputs.append(record)
        h = a.read_history(base/"output/minkowski.history.csv"); histories.append(h)
        ix = a.selected_interval(h, 1e-10, 1e-3)
        common = np.flatnonzero(np.isclose(h["t"], .45, rtol=0, atol=1e-12))
        require(len(common) == 1, "missing unique common comparison time")
        require(report["n"] == 16 and report["dt"] == .0005 and report["start_time"] == 0,
                "comparison parameters mismatch")
        comparisons.append(dict(case=label, source_manifest_sha256=worker["source_manifest_sha256"],
            binaries=worker["binaries"], native_status=report["status"], native_error=report["error"],
            accepted_time=report["t"], steps=report["steps"], target_time=report["target_time"],
            diagnostic_time=report["diagnostic_time"], diagnostics_current=report["diagnostics_current"],
            last_history_values={k:v[-1] for k,v in h.items()},
            common_time_values={k:v[common[0]] for k,v in h.items()},
            amplitude_window=[1e-10,1e-3],
            growth={f:a.fit_growth(h,ix,f) for f in ["state_deviation","H_L2","M_L2","Z_L2"]},
            throughput=a.throughput(h,19*16**3)))
    write(out/"comparison.json", dict(cases=comparisons, inputs=inputs,
        caveats=["Same state-amplitude window, explicit per-case fit times; unweighted OLS ln(amplitude) versus coordinate time.",
                 "H can remain near cancellation floor in the selected early window; its fitted slope is not clean mode evidence.",
                 "Historical builds differ beyond one bitwise operator toggle; V2 also cached sampling. These data do not identify a unique causal operator.",
                 "Correlated samples and changing leading-mode mixture make fitted rates descriptive, not a convergence or causal proof."]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    colors = ["#333333", "#0072B2", "#D55E00"]
    labels = ["V1 original", "V2 projected", "V3 cross-face"]
    for h,c,label,comparison in zip(histories,colors,labels,comparisons):
        positive = lambda key: np.where(h[key]>0,h[key],np.nan)
        axes[0,0].semilogy(h["t"],positive("state_deviation"),"o-",color=c,label=label,markersize=3)
        fit = comparison["growth"]["state_deviation"]
        fit_times = np.array(fit["time_range"])
        axes[0,0].semilogy(fit_times,np.exp(fit["intercept"]+fit["rate"]*fit_times),":",color=c,lw=2)
        axes[0,1].semilogy(h["t"],positive("H_L2"),color=c,label=label+" H")
        axes[0,1].semilogy(h["t"],positive("M_L2"),"--",color=c,label=label+" M")
        if "determinant_residual" in h:
            axes[1,0].semilogy(h["t"],positive("determinant_residual"),color=c,label=label+" det")
            axes[1,0].semilogy(h["t"],positive("tracefree_residual"),"--",color=c,label=label+" trace")
        axes[1,1].plot(h["t"],h["null_surface"],"o-",color=c,label=label+" surface",markersize=3)
        axes[1,1].plot(h["t"],h["null_ghost"],"--",color=c,label=label+" ghost")
    axes[0,0].axhspan(1e-10,1e-3,color="#aaaaaa",alpha=.15)
    axes[0,0].axhline(1e-8,color="#888888",ls="--",lw=.8)
    axes[0,0].set_title("State deviation; shaded fit-amplitude window")
    axes[0,0].set_ylabel("Maximum volume-field deviation")
    axes[0,1].set_title("Constraint norms; early H near cancellation floor")
    axes[0,1].set_ylabel("Coordinate-volume RMS")
    axes[1,0].set_title("Volume algebraic residuals; V1 did not record them")
    axes[1,0].set_ylabel("Maximum residual")
    axes[1,1].set_title("Null closure at successful recorded evaluations")
    axes[1,1].set_ylabel("Relative null residual")
    axes[1,1].set_ylim(0,1.8e-15)
    for ax in axes.flat:
        ax.set_xlabel("Coordinate time"); ax.set_xlim(0,.53); ax.grid(True,alpha=.2)
        ax.legend(fontsize=8,loc="best",ncol=2 if ax in [axes[0,1],axes[1,1]] else 1)
    fig.suptitle("Minkowski n16, dt=0.0005: all three duration attempts failed\n"
                 "Last diagnostics: V1/V2 0.45, V3 0.50; accepted rollback times: 0.4895 / 0.4850 / 0.5205",fontsize=12)
    fig.savefig(out/"original-projected-angular.png",dpi=180)
    fig.savefig(out/"original-projected-angular.pdf"); plt.close(fig)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir",type=Path,default=ROOT/"output/hyperbolic-core-shell/aurora"/HERE.name)
    parser.add_argument("--build-dir",type=Path,default=ROOT/"build-hyperbolic-mpi")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    raw = args.raw_dir.resolve()
    report,evidence,verification = verify(raw)
    n=report["n"]; nv=31*19*n**3; ns=31*6*n*n
    checkpoint=raw/"output/minkowski.checkpoint"; failed=raw/"output/minkowski.failed-stage.bin"
    with checkpoint.open("rb") as f:
        magic,size,nf,kind,admitted,cells,surface_size,steps,t,dt = HEADER.unpack(f.read(HEADER.size))
    require((magic.rstrip(b"\0"),size,nf,kind,admitted,cells,surface_size) ==
            (b"HYP3D_MPI_V3",n,31,0,0,19*n**3,ns), "checkpoint header mismatch")
    require(checkpoint.stat().st_size == 72+16*(nv+ns) and failed.stat().st_size == 8*(nv+ns),
            "payload size mismatch")
    require(t == report["t"] and steps == report["steps"] and dt == report["dt"] and
            report["checkpoint_admitted"] is False, "checkpoint/report mismatch")
    saved=np.memmap(checkpoint,dtype="<f8",mode="r",offset=72+16*nv,shape=(6*n*n,31))
    stage=np.memmap(failed,dtype="<f8",mode="r",offset=8*nv,shape=(6*n*n,31))
    row=int(np.argmin(np.linalg.eigvalsh(s.tensors(stage))[:,0]))
    write(out/"surface-audit.json",dict(checkpoint_schema="HYP3D_MPI_V3",admitted=0,accepted_time=t,
        steps=steps,dt=dt,attempted_interval=[t,t+dt],failed_stage_abscissa="not recorded",
        selected_row="row with minimum eigenvalue of the captured failed-stage surface metric",
        accepted_rollback=s.surface_audit(saved,row),failed_stage=s.surface_audit(stage,row)))
    replay(out,args.build_dir.resolve(),n,stage,saved,failed)
    comparisons=compare(out)
    for item in evidence:
        item["unchanged_after_read"] = sha(Path(item["local_path"])) == item["sha256"]
        require(item["unchanged_after_read"], "scientific input changed during analysis")
    helpers=[Path(__file__),Path(a.__file__),Path(s.__file__),Path(i.__file__)]
    write(out/"package-verification.json",dict(**verification,artifacts=evidence,
        helpers=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in helpers],
        scope="Read only; failed scientific gates retained. No checkpoint admission, evolution, clamping or restart."))
    print(json.dumps(dict(output=str(out),checks=verification["checks_recomputed"],
        artifacts=len(evidence),state_rates=[c["growth"]["state_deviation"]["rate"] for c in comparisons])))


if __name__ == "__main__":
    main()
