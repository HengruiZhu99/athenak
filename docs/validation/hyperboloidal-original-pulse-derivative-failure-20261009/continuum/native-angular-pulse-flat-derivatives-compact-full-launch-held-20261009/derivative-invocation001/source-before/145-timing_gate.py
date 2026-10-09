"""HELD tiny local-ray identity/timing pilot, not the full derivative gate.

This fresh source is not imported or executed during preparation. Future
execution requires exact root admission. No angular integral accuracy,
inverse map, kernel/native query or PDE evolution is claimed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import mpmath as mp
import derivative_core as core


HERE=Path(__file__).resolve().parent
NAMES=["timing_gate.py","derivative_core.py","analytic_jets.py","values_context.py","PLAN.md","timing-recipe.json"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def run(recipe,out):
    rays,checks,groups=[],[],[]
    def check(name,error,tolerance):
        checks.append({"name":name,"error":core.vc.number(error),"tolerance":tolerance,"passed":bool(error<=mp.mpf(tolerance))})
    for dps in recipe["precisions"]:
        mp.mp.dps=dps
        for level in recipe["levels"]:
            for case in recipe["events"]:
                graph=core.NativeGraph(recipe,level["height_order"],recipe["native_amplitudes"])
                T,X,omega,gamma,boost,x=graph.event(case)
                for mode in recipe["boosts"]:
                    begin=time.monotonic()
                    B=core.fixed_boost(X,gamma,boost,mode)
                    lerror=max(abs(mp.fsum((-1 if l==0 else 1)*B[l][a]*B[l][b] for l in range(4))-(-1 if a==b==0 else int(a==b))) for a in range(4) for b in range(4))/max(1,*[abs(v)**2 for row in B for v in row])
                    prefix="%s/%s/%s/%s"%(dps,level["name"],case["name"],mode)
                    check(prefix+"/Lorentz_matrix",lerror,recipe["tolerances"]["local_identities"])
                    nodes,weights=core.vc.gauss(level["polar_nodes"],dps)
                    for im,mu in enumerate(nodes):
                        for ia in range(level["azimuth_nodes"]):
                            az=2*mp.pi*ia/level["azimuth_nodes"]
                            tr=mp.sqrt((1-mu)*(1+mu))
                            unit=[mp.mpf(1),tr*mp.cos(az),tr*mp.sin(az),mu]
                            k=[core.vc.dot(row,unit) for row in B]
                            start=time.monotonic()
                            result,metrics=core.ray_integrand(graph,T,X,k,initial=case["initial"])
                            seconds=time.monotonic()-start
                            flat=[j.flat() for j in result]
                            if not all(mp.isfinite(v) for row in flat for v in row) or not metrics["minimum_D"]>0:
                                raise ArithmeticError("nonfinite jet or nonpositive denominator in timing pilot")
                            name=prefix+"/%s/%s"%(im,ia)
                            for key in ["root","null","first_graph","second_graph","factor_quotient","K_jet","D_source_jet"]:
                                tol=recipe["tolerances"]["root"] if key=="root" else recipe["tolerances"]["local_identities"]
                                check(name+"/"+key,metrics[key],tol)
                            symmetry=max(abs(j.h[a][b]-j.h[b][a])/max(1,abs(j.h[a][b]),abs(j.h[b][a])) for j in result for a in range(4) for b in range(4))
                            check(name+"/Hessian_symmetry",symmetry,recipe["tolerances"]["local_identities"])
                            if case["initial"]:
                                check(name+"/initial_zero_value",max(abs(j.v) for j in result),"0")
                            rays.append({"dps":dps,"level":level["name"],"event":case["name"],"boost":mode,"polar_index":im,"azimuth_index":ia,"k":core.strings(k),"jet":[core.strings(row) for row in flat],"metrics":{key:core.vc.number(v) for key,v in metrics.items()},"seconds":seconds,"scope":"single fixed ray integrand jet; not an angularly integrated scalar derivative"})
                    groups.append({"dps":dps,"level":level["name"],"event":case["name"],"boost":mode,"ray_count":level["polar_nodes"]*level["azimuth_nodes"],"seconds":time.monotonic()-begin})
                    write(out/"partial-rays.json",rays)
                    write(out/"partial-groups.json",groups)
                    print("tiny-group-complete",prefix,groups[-1]["ray_count"],groups[-1]["seconds"],flush=True)
    mp.mp.dps=max(recipe["precisions"])
    key=lambda r:(r["level"],r["event"],r["boost"],r["polar_index"],r["azimuth_index"])
    low={key(r):r for r in rays if r["dps"]==recipe["precisions"][0]}
    high={key(r):r for r in rays if r["dps"]==recipe["precisions"][1]}
    if low.keys()!=high.keys():
        raise RuntimeError("timing precision ray-grid mismatch")
    for k in sorted(low):
        a=core.jet_blocks([[mp.mpf(v) for v in row] for row in low[k]["jet"]])
        b=core.jet_blocks([[mp.mpf(v) for v in row] for row in high[k]["jet"]])
        for block in a:
            check("fixed-ray-precision/%s/%s"%(k,block),core.vc.scaled(a[block],b[block]),recipe["tolerances"]["precision"])
    for name,obj in [("rays.json",rays),("groups.json",groups),("checks.json",checks)]:
        write(out/name,obj)
    return {"passed_tiny_timing_identity_gate":all(c["passed"] for c in checks),"ray_rows":len(rays),"group_rows":len(groups),"checks":len(checks),"failed_checks":[c for c in checks if not c["passed"]],"scope":"Measured finite local-ray timing and analytic identity consistency only. No quadrature convergence or completed full derivative gate, inverse map, target-time/Jacobian claim, native query or evolution."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--recipe",required=True)
    parser.add_argument("--authorization",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    receipt={"kind":"Separate tiny local-ray timing/identity pilot","accepted_native":False,"inverse_map_attempted":False,"full_derivative_gate_passed":False}
    before={}
    try:
        recipepath,authpath=Path(args.recipe).resolve(),Path(args.authorization).resolve()
        if recipepath!=(HERE/"timing-recipe.json").resolve():
            raise PermissionError("timing recipe must resolve to pinned local file")
        raw=recipepath.read_bytes()
        recipe,auth=json.loads(raw),json.loads(authpath.read_text())
        required={str(HERE/name):sha(HERE/name) for name in NAMES}
        digest=hashlib.sha256(raw).hexdigest()
        if required[str(HERE/"timing-recipe.json")]!=digest:
            raise RuntimeError("consumed timing recipe changed during admission")
        if auth.get("tiny_timing_identity_execution_admitted") is not True or auth.get("source_pins")!=required or auth.get("fresh_output_path")!=str(out):
            raise PermissionError("missing exact root tiny timing release")
        for path,expected in recipe["dependency_pins"].items():
            if sha(path)!=expected:
                raise RuntimeError("dependency mismatch: "+path)
        runtime=Path(sys.executable).resolve()
        if str(runtime)!=recipe["python_runtime_path"] or sha(runtime)!=recipe["python_runtime_sha256"] or sys.version_info[:2]!=(3,9) or mp.__version__!="1.3.0":
            raise RuntimeError("unreviewed timing runtime")
        package={str(p):sha(p) for p in sorted(Path(mp.__file__).resolve().parent.rglob("*.py"))}
        if package!=recipe["mpmath_python_pins"]:
            raise RuntimeError("timing mpmath source inventory mismatch")
        for key,expected in recipe["required_environment"].items():
            if os.environ.get(key)!=expected:
                raise RuntimeError("unreviewed timing environment: "+key)
        before={**required,**recipe["dependency_pins"],**package,str(runtime):sha(runtime),str(authpath):sha(authpath)}
        receipt.update({"source_before":before,"consumed_recipe":{"path":str(recipepath),"sha256":digest},"command":sys.argv,"environment":{key:os.environ.get(key) for key in recipe["required_environment"]},"python":sys.version,"mpmath":mp.__version__})
        write(out/"before.json",receipt)
        result=run(recipe,out)
        receipt.update(result)
        if not result["passed_tiny_timing_identity_gate"]:
            raise ArithmeticError("tiny timing identity gate failed; no retry or threshold change")
    except BaseException as exc:
        receipt.update({"passed_tiny_timing_identity_gate":False,"exception_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        raise
    finally:
        def after(path):
            try:
                return sha(path)
            except BaseException as exc:
                return "READ_ERROR:"+repr(exc)
        receipt["source_after"]={path:after(path) for path in before}
        receipt["sources_unchanged"]=receipt["source_after"]==before
        if not receipt["sources_unchanged"]:
            receipt["passed_tiny_timing_identity_gate"]=False
        receipt["seconds"]=time.monotonic()-start
        receipt["output_pins"]={str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="receipt.json"}
        write(out/"receipt.json",receipt)
        if not receipt["sources_unchanged"]:
            raise RuntimeError("timing source/dependency/runtime drift")


if __name__=="__main__":
    main()
