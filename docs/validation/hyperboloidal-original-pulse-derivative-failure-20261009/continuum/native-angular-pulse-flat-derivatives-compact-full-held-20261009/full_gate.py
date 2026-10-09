"""HELD full analytic derivatives using reviewed compact-root acceleration.

No candidate/scientific module has been imported or executed in preparation.
Original full scientific settings and analytic control math stay unchanged.
Only native graph constructors select the reviewed compact root subclass.
Progress wrappers add standard-library logging after each completed root.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import mpmath as mp
import derivative_core as core
import compact_root
from derivative_core import (vc,ZERO,strings,write,sha,fixed_boost,
                             integrate_ray,initial_oracle,jet_blocks,trace_error)

HERE=Path(__file__).resolve().parent
NAMES=["full_gate.py","compact_root.py","derivative_core.py","analytic_jets.py","values_context.py","PLAN.md","derivative-recipe.json"]
PROGRESS=None


class RootProgress:
    def __init__(self,out):
        self.out=out
        self.start=time.monotonic()
        self.last=self.start
        self.counts=Counter()
        self.methods=Counter()
        self.label="startup"
        self.rows=0

    def tick(self,kind,method):
        self.counts[kind]+=1
        self.methods[method]+=1
        if time.monotonic()-self.last>=30:
            self.emit("root_progress")

    def summary(self):
        return {"completed_root_calls":sum(self.counts.values()),"by_kind":dict(self.counts),
                "by_method":dict(self.methods),"current_label":self.label,
                "seconds_since_progress_context":time.monotonic()-self.start,
                "scope":"Root-call progress, not integrand or gate acceptance; source/jet failures can follow a completed root"}

    def emit(self,kind):
        row={"kind":kind,**self.summary()}
        with (self.out/"root-progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        self.rows+=1
        self.last=time.monotonic()
        print("root-progress",row["completed_root_calls"],self.label,row["seconds_since_progress_context"],flush=True)


class ProgressNativeGraph(compact_root.AcceleratedNativeGraph):
    def root(self,T,X,k,initial=False):
        ell=super().root(T,X,k,initial)
        if PROGRESS is None:
            raise RuntimeError("missing declared full-gate progress context")
        PROGRESS.tick("initial" if initial else "native",self.last_root["method"])
        return ell


class ProgressControlGraph(core.ControlGraph):
    def root(self,T,X,k,initial=False):
        ell=super().root(T,X,k,initial)
        if PROGRESS is None:
            raise RuntimeError("missing declared full-gate progress context")
        PROGRESS.tick("control","unchanged_analytic_control")
        return ell


def run(recipe,out):
    rows,controls,initials,checks=[],[],[],[]
    def check(name,error,tolerance):
        checks.append({"name":name,"error":vc.number(error),"tolerance":tolerance,"passed":bool(error<=mp.mpf(tolerance))})
    def add_metrics(prefix,m):
        for key in ["root","null","first_graph","second_graph","factor_quotient","K_jet","D_source_jet","Lorentz_matrix"]:
            tol=recipe["tolerances"]["root"] if key=="root" else recipe["tolerances"]["local_identities"]
            check(prefix+"/"+key,m[key],tol)
        if not m["minimum_D"]>0:
            raise ArithmeticError("nonpositive quadrature ray denominator")
    for dps in recipe["precisions"]:
        mp.mp.dps=dps
        for level in recipe["levels"]:
            for case in recipe["events"]:
                graph=ProgressNativeGraph(recipe,level["height_order"],case["amplitudes"])
                T,X,omega,gamma,boost,x=graph.event(case)
                for mode in recipe["native_boosts"]:
                    PROGRESS.label="native/%s/%s/%s/%s"%(dps,level["name"],case["name"],mode)
                    start=time.monotonic()
                    B=fixed_boost(X,gamma,boost,mode)
                    result,metrics=integrate_ray(graph,T,X,B,level["polar_order"],level["azimuth_order"])
                    row={"name":case["name"],"dps":dps,"level":level["name"],"boost":mode,"T":vc.number(T),"X":strings(X),"Omega_event":vc.number(omega),"boost_matrix":[strings(r) for r in B],"jet":[strings(r) for r in result],"phi_values":strings([r[0]/omega for r in result]),"metrics":{k:vc.number(v) for k,v in metrics.items()},"seconds":time.monotonic()-start,"scope":"fixed reference evaluation event; inertial derivatives of u; no inverse/native target"}
                    rows.append(row)
                    add_metrics("native/%s/%s/%s/%s"%(dps,level["name"],case["name"],mode),metrics)
                    row["wave_trace_scaled_error"]=vc.number(trace_error(result))
                    if level["name"]==recipe["final_level"]:
                        check("wave/native/%s/%s/%s/%s"%(dps,level["name"],case["name"],mode),trace_error(result),recipe["tolerances"]["wave_trace"])
                    write(out/"partial-native-jets.json",rows)
                    print("native-complete",dps,level["name"],case["name"],mode,time.monotonic()-start,flush=True)
            # Closed-form controls retain all four gradients and ten Hessians.
            for ctrl in recipe["controls"]:
                graph=ProgressControlGraph(ctrl["kind"],ctrl.get("degree",0))
                X=list(map(mp.mpf,ctrl["X"]))
                T=graph.height(X)+mp.mpf(ctrl["tau"])
                exact=[j.flat() for j in graph.exact(T,X)]
                for mode in recipe["control_boosts"]:
                    PROGRESS.label="control/%s/%s/%s/%s"%(dps,level["name"],ctrl["name"],mode)
                    B=fixed_boost(X,mp.mpf(1),ZERO(),mode)
                    result,metrics=integrate_ray(graph,T,X,B,level["polar_order"],level["azimuth_order"])
                    controls.append({"name":ctrl["name"],"dps":dps,"level":level["name"],"boost":mode,"jet":[strings(r) for r in result],"exact":[strings(r) for r in exact],"metrics":{k:vc.number(v) for k,v in metrics.items()}})
                    add_metrics("control/%s/%s/%s/%s"%(dps,level["name"],ctrl["name"],mode),metrics)
                    if level["name"]==recipe["final_level"]:
                        for block,numer in jet_blocks(result).items():
                            check("exact/%s/%s/%s/%s"%(dps,ctrl["name"],mode,block),vc.scaled(numer,jet_blocks(exact)[block]),recipe["tolerances"]["convergence"])
                    controls[-1]["wave_trace_scaled_error"]=vc.number(trace_error(result))
                    if level["name"]==recipe["final_level"]:
                        check("wave/control/%s/%s/%s/%s"%(dps,level["name"],ctrl["name"],mode),trace_error(result),recipe["tolerances"]["wave_trace"])
                    write(out/"partial-controls.json",controls)
                    print("control-complete",dps,level["name"],ctrl["name"],mode,flush=True)
        final=next(level for level in recipe["levels"] if level["name"]==recipe["final_level"])
        # Exact lambda=0 boundary limit, without a finite-difference derivative.
        for point in recipe["initial_points"]:
            graph=ProgressNativeGraph(recipe,final["height_order"],recipe["native_amplitudes"])
            case={"xyz":point,"tau_reference":"0"}
            T,X,omega,gamma,boost,x=graph.event(case)
            expected=initial_oracle(graph,X)
            for mode in recipe["native_boosts"]:
                B=fixed_boost(X,gamma,boost,mode)
                PROGRESS.label="initial/%s/%s/%s"%(dps,point,mode)
                result,metrics=integrate_ray(graph,T,X,B,final["polar_order"],final["azimuth_order"],initial=True)
                initials.append({"xyz":point,"dps":dps,"boost":mode,"initial_jet":[strings(r) for r in result],"independent_graph_wave_oracle":[strings(r) for r in expected],"metrics":{k:vc.number(v) for k,v in metrics.items()}})
                for block,numer in jet_blocks(result).items():
                    check("initial-limit/%s/%s/%s/%s"%(dps,point,mode,block),vc.scaled(numer,jet_blocks(expected)[block]),recipe["tolerances"]["convergence"])
                add_metrics("initial/%s/%s/%s"%(dps,point,mode),metrics)
                write(out/"partial-initial-limits.json",initials)
        # Values cross-binding invokes the copied independent coarea values
        # implementation only at identical declared events; no old receipt run.
        layer=vc.Layer(recipe,final["height_order"])
        for case in recipe["events"]:
            value,phi,meta=vc.coarea(layer,case,128,128)
            for mode in recipe["native_boosts"]:
                row=next(r for r in rows if r["dps"]==dps and r["level"]==recipe["final_level"] and r["name"]==case["name"] and r["boost"]==mode)
                got=[mp.mpf(r[0]) for r in row["jet"]]
                check("coarea-value/%s/%s/%s/u"%(dps,case["name"],mode),vc.scaled(got,value),recipe["tolerances"]["convergence"])
                check("coarea-value/%s/%s/%s/phi"%(dps,case["name"],mode),vc.scaled(list(map(mp.mpf,row["phi_values"])),phi),recipe["tolerances"]["convergence"])
    mp.mp.dps=max(recipe["precisions"])
    lookup={(r["dps"],r["level"],r["name"],r["boost"]):r for r in rows}
    for case in recipe["events"]:
        name=case["name"]
        for mode in recipe["native_boosts"]:
            for dps in recipe["precisions"]:
                final=lookup[dps,recipe["final_level"],name,mode]
                baseblocks=jet_blocks([[mp.mpf(v) for v in r] for r in final["jet"]])
                for other in recipe["comparison_levels"]:
                    row=lookup[dps,other,name,mode]
                    compare=jet_blocks([[mp.mpf(v) for v in r] for r in row["jet"]])
                    for block in baseblocks:
                        check("convergence/%s/%s/%s/%s/%s"%(dps,name,mode,other,block),vc.scaled(baseblocks[block],compare[block]),recipe["tolerances"]["convergence"])
                    check("convergence/%s/%s/%s/%s/phi"%(dps,name,mode,other),vc.scaled(list(map(mp.mpf,final["phi_values"])),list(map(mp.mpf,row["phi_values"]))),recipe["tolerances"]["convergence"])
            lo=lookup[recipe["precisions"][0],recipe["final_level"],name,mode]
            hi=lookup[recipe["precisions"][1],recipe["final_level"],name,mode]
            for block,a in jet_blocks([[mp.mpf(v) for v in r] for r in lo["jet"]]).items():
                b=jet_blocks([[mp.mpf(v) for v in r] for r in hi["jet"]])[block]
                check("precision/%s/%s/%s"%(name,mode,block),vc.scaled(a,b),recipe["tolerances"]["precision"])
            check("precision/%s/%s/phi"%(name,mode),vc.scaled(list(map(mp.mpf,lo["phi_values"])),list(map(mp.mpf,hi["phi_values"]))),recipe["tolerances"]["precision"])
            if case["amplitudes"]==["0","0"]:
                check("zero/%s/%s"%(name,mode),max(abs(mp.mpf(v)) for r in hi["jet"] for v in r),"0")
        # Independent constant Lorentz quadratures must produce the same
        # inertial component derivatives, not transformed output components.
        for dps in recipe["precisions"]:
            ra,rb=[lookup[dps,recipe["final_level"],name,m] for m in recipe["native_boosts"]]
            for block,a in jet_blocks([[mp.mpf(v) for v in r] for r in ra["jet"]]).items():
                b=jet_blocks([[mp.mpf(v) for v in r] for r in rb["jet"]])[block]
                check("boost/%s/%s/%s"%(dps,name,block),vc.scaled(a,b),recipe["tolerances"]["convergence"])
    for fn,data in [("native-jets.json",rows),("controls.json",controls),("initial-limits.json",initials),("checks.json",checks)]:
        write(out/fn,data)
    return {"passed_analytic_scalar_derivative_gate":all(c["passed"] for c in checks),"checks":len(checks),"failed_checks":[c for c in checks if not c["passed"]],"native_rows":len(rows),"control_rows":len(controls),"initial_rows":len(initials),"scope":"Finite reference-event scalar values, four inertial gradients and ten inertial Hessians. No inverse/native-target coverage, Jacobian/caustic verdict, PDE or native evolution, Einstein stability or black-hole gauge admission."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--recipe",required=True)
    parser.add_argument("--authorization",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    receipt={"kind":"Full analytic fixed-Lorentz-ray derivative attempt with compact root","accepted_native":False,"inverse_map_attempted":False}
    before={}
    try:
        recipepath,authpath=Path(args.recipe).resolve(),Path(args.authorization).resolve()
        if recipepath!=(HERE/"derivative-recipe.json").resolve():
            raise PermissionError("consumed recipe must be the pinned local derivative-recipe.json")
        recipe_bytes=recipepath.read_bytes()
        recipe=json.loads(recipe_bytes)
        auth=json.loads(authpath.read_text())
        required={str(HERE/name):sha(HERE/name) for name in NAMES}
        consumed_recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest()
        if required[str(HERE/"derivative-recipe.json")]!=consumed_recipe_sha256:
            raise RuntimeError("consumed recipe changed during admission")
        if auth.get("analytic_scalar_derivative_execution_admitted") is not True or auth.get("source_pins")!=required:
            raise PermissionError("missing exact root analytic-derivative release")
        if str(out)!=auth.get("fresh_output_path"):
            raise PermissionError("unreleased output path")
        for key,value in recipe["dependency_pins"].items():
            if sha(key)!=value:
                raise RuntimeError("dependency pin mismatch: "+key)
        value_receipt=json.loads(Path(recipe["completed_values_receipt"]).read_text())
        if value_receipt.get("passed_scalar_values_only") is not True or value_receipt.get("sources_unchanged") is not True:
            raise PermissionError("completed values PASS prerequisite absent")
        compact_receipt=json.loads(Path(recipe["completed_compact_pilot_receipt"]).read_text())
        if compact_receipt.get("passed_compact_root_comparison_gate") is not True or compact_receipt.get("passed_tiny_timing_identity_gate") is not True or compact_receipt.get("sources_unchanged") is not True or compact_receipt.get("ray_rows")!=480 or compact_receipt.get("group_rows")!=24 or compact_receipt.get("checks")!=38344 or compact_receipt.get("failed_checks")!=[]:
            raise PermissionError("completed compact480 comparison PASS prerequisite absent")
        original_recipe=json.loads(Path(recipe["original_full_recipe"]).read_text())
        if any(recipe[key]!=original_recipe[key] for key in recipe["unchanged_scientific_setting_keys"]):
            raise PermissionError("original full scientific settings changed")
        for key,expected in recipe["required_environment"].items():
            if os.environ.get(key)!=expected:
                raise RuntimeError("unreviewed environment: "+key)
        runtime=Path(sys.executable).resolve()
        if str(runtime)!=recipe["python_runtime_path"] or sha(runtime)!=recipe["python_runtime_sha256"] or sys.version_info[:2]!=(3,9) or mp.__version__!="1.3.0":
            raise RuntimeError("unreviewed Python/mpmath runtime")
        package={str(f):sha(f) for f in sorted(Path(mp.__file__).resolve().parent.rglob("*.py"))}
        if package!=recipe["mpmath_python_pins"]:
            raise RuntimeError("mpmath source inventory mismatch")
        before={**required,**recipe["dependency_pins"],**package,str(runtime):sha(runtime),str(authpath):sha(authpath)}
        receipt.update({"source_before":before,"consumed_recipe":{"path":str(recipepath),"sha256":consumed_recipe_sha256},"authorization":str(authpath),"command":sys.argv,"environment":{k:os.environ.get(k) for k in recipe["required_environment"]},"python":platform.python_version(),"mpmath":mp.__version__})
        write(out/"before.json",receipt)
        global PROGRESS
        PROGRESS=RootProgress(out)
        PROGRESS.emit("full_gate_start")
        result=run(recipe,out)
        expected_counts={"native":189440,"control":189440,"initial":114688}
        if dict(PROGRESS.counts)!=expected_counts:
            raise RuntimeError("full differentiated root-call inventory differs from declared493568")
        receipt.update(result)
        if not result["passed_analytic_scalar_derivative_gate"]:
            raise ArithmeticError("fixed analytic-derivative gate failed; no retry or tolerance change")
    except BaseException as exc:
        receipt.update({"passed_analytic_scalar_derivative_gate":False,"error":str(exc),"exception_type":type(exc).__name__,"traceback":traceback.format_exc()})
        raise
    finally:
        def protected_after(key):
            try:
                return sha(key)
            except BaseException as exc:
                return "READ_ERROR:"+repr(exc)
        receipt["source_after"]={key:protected_after(key) for key in before}
        receipt["sources_unchanged"]=receipt["source_after"]==before
        if not receipt["sources_unchanged"]:
            receipt["passed_analytic_scalar_derivative_gate"]=False
        if PROGRESS is not None:
            PROGRESS.emit("full_gate_stop")
            receipt["root_progress"]=PROGRESS.summary()
        receipt["seconds"]=time.monotonic()-start
        receipt["output_pins"]={str(f):sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!="receipt.json"}
        write(out/"receipt.json",receipt)
        if not receipt["sources_unchanged"]:
            raise RuntimeError("protected source/runtime drift")


if __name__=="__main__":
    main()
