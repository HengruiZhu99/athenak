#!/usr/bin/env python3
"""Read-only history comparison and surface-tensor audit of projected V2 failure.

NumPy and Matplotlib required. Raw checkpoints/captures are never admitted,
rewritten or restarted. --output must name a new analysis directory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = "8d0eba426c84cdf9997864cf7471a44914cc590b9a4d4ea2efe022b944287c93"
BINARY = "362093fc8c998576c7d88a14afdc2c5ebcdcb6a68f408b7fdaf92ad62b7b1486"
HEADER = struct.Struct("<16s4i3Q2d")
HELPER = HERE.parent/"resolution-control-v1/analyze.py"
spec = importlib.util.spec_from_file_location("resolution_analysis", HELPER)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(a.clean(data), indent=2, allow_nan=False)+"\n")


def tensors(values, offset=1, add_identity=True):
    g = np.zeros(values.shape[:-1]+(3, 3))
    for f, (i, j) in enumerate([(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)], offset):
        g[...,i,j] = g[...,j,i] = values[...,f]+(add_identity and i==j)
    return g


def surface_audit(surface, row):
    g = tensors(surface)
    # Same determinant and leading-principal-minor test as native Geometry.
    m1 = g[:,0,0]
    m2 = g[:,0,0]*g[:,1,1]-g[:,0,1]*g[:,1,0]
    det = (g[:,0,0]*(g[:,1,1]*g[:,2,2]-g[:,1,2]*g[:,2,1])
           -g[:,0,1]*(g[:,1,0]*g[:,2,2]-g[:,1,2]*g[:,2,0])
           +g[:,0,2]*(g[:,1,0]*g[:,2,1]-g[:,1,1]*g[:,2,0]))
    valid = np.isfinite(det)&(det>0)&(m1>0)&(m2>0)
    eigen = np.linalg.eigvalsh(g)
    info = dict(surface_points=len(surface), nonfinite_field_count=int(np.count_nonzero(~np.isfinite(surface))),
                minimum_chi=float(surface[:,0].min()), invalid_metric_rows=np.flatnonzero(~valid),
                minimum_eigenvalue=float(eigen.min()), minimum_eigenvalue_row=int(np.argmin(eigen[:,0])),
                determinant_range=[float(det.min()),float(det.max())],
                row=dict(index=row, chi=surface[row,0], metric=g[row], eigenvalues=eigen[row],
                         leading_minor1=m1[row],leading_minor2=m2[row],determinant=det[row],
                         all31_fields=surface[row], native_metric_valid=bool(valid[row])))
    if np.all(valid):
        inverse = np.linalg.inv(g)
        A = tensors(surface, 7, False)
        info["maximum_abs_tracefree_residual"] = float(abs(np.einsum("pij,pij->p",inverse,A)).max())
        info["maximum_abs_determinant_residual"] = float(abs(det-1).max())
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir",type=Path,default=ROOT/"output/hyperbolic-core-shell/aurora/stability-projected-minkowski-v1")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    worker=json.loads((HERE/"worker-status.json").read_text()); case=worker["cases"][0]
    binding=json.loads((HERE/"binding.json").read_text());report=json.loads((HERE/"output/minkowski.report.json").read_text())
    if worker["source_manifest_sha256"]!=SOURCE or set(worker["binaries"].values())!={BINARY}:
        raise ValueError("unexpected projected source/binary")
    assert binding["plan_sha256"]==sha(HERE/"plan.json")==worker["plan_sha256"]
    assert binding["source_manifest_sha256"]==SOURCE and binding["binaries"]==worker["binaries"]
    assert binding["pbs_sha256"]==sha(HERE/"run_debug.pbs")
    assert binding["admission"]["sha256"]==sha(HERE/"admission.json")
    admission=json.loads((HERE/"admission.json").read_text())
    assert admission["passed"] and admission["source_manifest_sha256"]==SOURCE
    assert BINARY in admission["binaries"].values()
    evidence=[]
    for remote,expected in case["artifacts"].items():
        raw=args.raw_dir.resolve()/"output"/Path(remote).name
        actual=dict(sha256=sha(raw),bytes=raw.stat().st_size)
        assert actual==expected,(raw,actual,expected)
        retained=HERE/"output"/raw.name
        if retained.exists():assert sha(retained)==actual["sha256"]
        evidence.append(dict(remote_path=remote,local_path=str(raw),**actual,worker_bound=True,
                             retained_in_package=retained.exists()))
    for check in case["checks"]:
        value=report
        for component in check["check"]["path"]:value=value[component]
        op=check["check"]["op"];target=check["check"].get("value")
        if op=="finite":passed=np.isfinite(value)
        elif op=="eq":passed=value==target
        elif op=="le":passed=np.isfinite(value) and value<=target
        elif op=="ge":passed=np.isfinite(value) and value>=target
        else:raise ValueError(op)
        assert bool(passed)==check["passed"] and value==check["observed"]
    n=report["n"];nv=31*19*n**3;ns=31*6*n*n
    checkpoint=args.raw_dir.resolve()/"output/minkowski.checkpoint"
    failed=args.raw_dir.resolve()/"output/minkowski.failed-stage.bin"
    with checkpoint.open("rb") as f:
        magic,size,nf,kind,admitted,cells,surface_size,steps,t,dt=HEADER.unpack(f.read(HEADER.size))
    assert (magic.rstrip(b"\0"),size,nf,kind,admitted,cells,surface_size)==(b"HYP3D_MPI_V2",n,31,0,0,19*n**3,ns)
    assert checkpoint.stat().st_size==72+16*(nv+ns) and failed.stat().st_size==8*(nv+ns)
    assert t==report["t"] and steps==report["steps"] and report["checkpoint_admitted"] is False
    saved=np.memmap(checkpoint,dtype="<f8",mode="r",offset=72+16*nv,shape=(6*n*n,31))
    stage=np.memmap(failed,dtype="<f8",mode="r",offset=8*nv,shape=(6*n*n,31))
    row=1520;face=row//(n*n);j=row%(n*n)%n;k=row%(n*n)//n
    u=2*(j+.5)/n-1;v=2*(k+.5)/n-1;scale=1/np.sqrt(1+u*u+v*v)
    x=np.zeros(3);axis=face//2;x[axis]=(-1 if face%2 else 1)*scale;x[(axis+1)%3]=u*scale;x[(axis+2)%3]=v*scale
    audit=dict(checkpoint_schema="HYP3D_MPI_V2",admitted=0,accepted_time=t,steps=steps,dt=dt,
               attempted_interval=[t,t+dt],failed_stage_abscissa="not recorded",
               location=dict(global_row=row,outer_block=13+face,face=face,face_name="-z",j=j,k=k,
                             local_angular=[(j+.5)/n,(k+.5)/n],boundary_xyz=x),
               accepted_rollback=surface_audit(saved,row),failed_stage=surface_audit(stage,row))
    write(out/"surface-audit.json",audit)
    baseline_dir=HERE.parent/"stability-v1/output"
    bw,_=a.load_worker(baseline_dir)
    original_report=json.loads((baseline_dir/"minkowski.report.json").read_text())
    baseline_bindings=[a.binding(baseline_dir/("minkowski"+suffix),bw) for suffix in [".history.csv",".report.json"]]
    assert all(b["worker_bound"] for b in baseline_bindings)
    histories=[a.read_history(baseline_dir/"minkowski.history.csv"),a.read_history(HERE/"output/minkowski.history.csv")]
    comparisons=[]
    for label,h,r in zip(["original_v1","projected_v2"],histories,[original_report,report]):
        ix=a.selected_interval(h,1e-10,1e-3)
        comparisons.append(dict(case=label,source_manifest_sha256=bw["source_manifest_sha256"] if label=="original_v1" else SOURCE,
            native_status=r["status"],native_error=r["error"],accepted_time=r["t"],target_time=r["target_time"],
            diagnostic_time=r["diagnostic_time"],diagnostics_current=r["diagnostics_current"],
            last_history_values={key:vals[-1] for key,vals in h.items()},
            amplitude_window=[1e-10,1e-3],growth={f:a.fit_growth(h,ix,f) for f in ["state_deviation","H_L2","M_L2","Z_L2"]},
            throughput=a.throughput(h,19*n**3)))
    write(out/"comparison.json",dict(cases=comparisons,
        state_rate_fractional_change=comparisons[1]["growth"]["state_deviation"]["rate"]/comparisons[0]["growth"]["state_deviation"]["rate"]-1,
        caveat="Final-stage algebraic projection plus cached-jet sampling: not a bitwise single-change comparison. Similar growth rates do not identify a unique causal operator.",
        baseline_inputs=baseline_bindings,analysis_helper=dict(path=str(HELPER.relative_to(ROOT)),sha256=sha(HELPER))))
    for e in evidence:
        e["unchanged_after_read"]=sha(Path(e["local_path"]))==e["sha256"]
        assert e["unchanged_after_read"]
    write(out/"package-verification.json",dict(artifacts=evidence,checks_recomputed=len(case["checks"]),
        all_checks_match_worker=True,plan_pbs_admission_source_binary_verified=True,
        protocol_sha256=sha(HERE/"protocol.md"),analyze_script_sha256=sha(Path(__file__)),
        summary="Read only; no checkpoint admission, evolution, clamping or restart."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.3),layout="constrained")
    colors=["#222222","#0072B2"];labels=["Original V1","Final-stage projection V2"]
    for h,c,label in zip(histories,colors,labels):
        axes[0,0].semilogy(h["t"],np.where(h["state_deviation"]>0,h["state_deviation"],np.nan),"o-",color=c,label=label,markersize=3)
        axes[0,1].semilogy(h["t"],np.where(h["H_L2"]>0,h["H_L2"],np.nan),color=c,label=label+" H")
        axes[0,1].semilogy(h["t"],np.where(h["M_L2"]>0,h["M_L2"],np.nan),"--",color=c,label=label+" M")
        axes[1,1].plot(h["t"],h["null_surface"],"o-",color=c,label=label+" surface",markersize=3)
        axes[1,1].plot(h["t"],h["null_ghost"],"--",color=c,label=label+" ghost")
    axes[0,0].axhspan(1e-10,1e-3,color="#cccccc",alpha=.25)
    axes[0,0].axhline(1e-8,color="#999999",ls=":",lw=1)
    axes[0,0].set_title("State deviation; shaded fit-amplitude window")
    axes[0,0].set_ylabel("Maximum stored-field deviation")
    axes[0,0].legend(fontsize=8,loc="lower right")
    axes[0,1].set_title("Constraint norms; early H near cancellation floor")
    axes[0,1].set_ylabel("Coordinate-volume RMS")
    axes[0,1].legend(fontsize=8,loc="upper left")
    h=histories[1]
    for key,label,style in [("determinant_residual","max |det(g)−1|","-"),("tracefree_residual","max |tr(g⁻¹A)|","--")]:
        axes[1,0].semilogy(h["t"],np.where(h[key]>0,h[key],np.nan),style,label=label,color="#0072B2")
    axes[1,0].set_title("Projected volume algebraic residuals")
    axes[1,0].set_ylabel("Maximum residual")
    axes[1,0].legend(fontsize=8,loc="lower right")
    axes[1,1].set_title("Null closure at recorded successful evaluations")
    axes[1,1].set_ylabel("Relative null residual")
    axes[1,1].set_ylim(0,1.8e-15)
    axes[1,1].legend(fontsize=8,loc="lower right")
    for ax in axes.flat:
        ax.set_xlabel("Coordinate time")
        ax.set_xlim(0,.5)
        ax.grid(True,alpha=.2)
    fig.suptitle("Minkowski n16, dt=0.0005: projection does not suppress the growing mode\nHistories end at t=0.45; last accepted times: 0.4895 (V1), 0.4850 (V2)",fontsize=12)
    fig.savefig(out/"original-vs-projected.png",dpi=180)
    fig.savefig(out/"original-vs-projected.pdf")
    plt.close(fig)
    print(json.dumps(dict(output=str(out),state_rates=[c["growth"]["state_deviation"]["rate"] for c in comparisons],
                          invalid_failed_surface_rows=audit["failed_stage"]["invalid_metric_rows"].tolist())))


if __name__=="__main__":main()
