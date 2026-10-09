"""HELD standard-library one-shot launcher. No scientific imports.

Root must authorize these exact bytes and launch-plan.json. The child runs
at most once; no retry, source change, gate relaxation or inverse-map work.
This source has not been executed during preparation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent/"native-angular-pulse-flat-derivatives-v2-held-20261009"
INDEX = CANDIDATE/"source-index.json"
INDEX_SHA256 = "5e30fe0fa20ab5fb1bd75a73a768a4fa29a2bc0ce5e03d4309f7a5e0ea55408b"
INVOCATION = HERE/"derivative-invocation001"
OUTPUT = HERE/"derivative-attempt001"
PLAN = HERE/"launch-plan.json"
PROSE_PLAN = HERE/"PLAN.md"
RECIPE = CANDIDATE/"derivative-recipe.json"
CHILD = CANDIDATE/"flat_ivp_derivatives.py"
CHILD_NAMES = ["flat_ivp_derivatives.py", "analytic_jets.py", "values_context.py", "PLAN.md", "derivative-recipe.json"]
EXPECTED = {"native_rows":100,"control_rows":100,"initial_rows":28,"checks":2360}
COST = {"angular_nodes_sum_per_event_and_precision":9472,
        "native_events":5,"native_boosts":2,"precisions":2,
        "native_ray_integrand_evaluations":189440,
        "control_events":5,"control_boosts":2,
        "control_ray_integrand_evaluations":189440,
        "initial_points":7,"initial_boosts":2,"initial_nodes":4096,
        "initial_ray_integrand_evaluations":114688,
        "total_differentiated_ray_integrand_evaluations":493568,
        "independent_coarea_cross_bindings":10,
        "measured_runtime_available":False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def hash_or_error(path):
    try:
        return sha(path)
    except BaseException as exc:
        return "READ_ERROR:"+type(exc).__name__+":"+str(exc)


def directory_pins(path):
    if not Path(path).exists():
        return {}
    return {str(p):sha(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def checked_json(path,expected_hash=None):
    if expected_hash is not None and sha(path)!=expected_hash:
        raise RuntimeError("hash mismatch: "+str(path))
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--authorization",required=True)
    args=parser.parse_args()
    authpath=Path(args.authorization).resolve()
    # Refuse an already-used invocation without touching its existing bytes.
    # An external root runner must capture this rejection as a fresh failure.
    INVOCATION.mkdir(parents=False,exist_ok=False)
    started=time.monotonic()
    child_started=False
    child_returncode=None
    child_output_before_read=None
    protected={str(Path(__file__).resolve()):hash_or_error(__file__),str(PLAN):hash_or_error(PLAN),str(PROSE_PLAN):hash_or_error(PROSE_PLAN),str(INDEX):hash_or_error(INDEX),str(authpath):hash_or_error(authpath)}
    before={path:hash_or_error(path) for path in protected}
    receipt={"kind":"One-shot analytic derivative outer invocation",
             "scientific_gate_accepted":False,"accepted_native":False,
             "inverse_map_attempted":False,"invocation":str(INVOCATION),
             "sole_child_output":str(OUTPUT),"authorization":str(authpath),
             "outer_argv":sys.argv,"outer_python":sys.executable,
             "outer_python_version":platform.python_version(),
             "expected_child_counts":EXPECTED,"fixed_cost_breakdown":COST,
             "runtime_is_not_measured_in_source_only_plan":True,
             "inherited_environment":{key:os.environ.get(key) for key in ["PYTHONDONTWRITEBYTECODE","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","PYTHONPATH","PYTHONHOME","PYTHONWARNINGS"]}}
    write(INVOCATION/"startup.json",receipt)
    (INVOCATION/"stdout.log").write_bytes(b"")
    (INVOCATION/"stderr.log").write_bytes(b"")
    code=1
    try:
        if OUTPUT.exists():
            raise FileExistsError("sole child output already exists; never overwrite or resume")
        auth=checked_json(authpath)
        plan=checked_json(PLAN)
        outerpins={str(Path(__file__).resolve()):sha(__file__),str(PLAN):sha(PLAN),str(PROSE_PLAN):sha(PROSE_PLAN)}
        if auth.get("one_shot_outer_derivative_launch_admitted") is not True or auth.get("outer_source_pins")!=outerpins:
            raise PermissionError("missing exact root one-shot outer release")
        if auth.get("candidate_source_index")!={"path":str(INDEX),"sha256":INDEX_SHA256}:
            raise PermissionError("wrong candidate source-index authorization")
        if auth.get("fresh_invocation_path")!=str(INVOCATION) or auth.get("fresh_output_path")!=str(OUTPUT):
            raise PermissionError("wrong sole invocation/output path")
        if auth.get("authorization_path")!=str(authpath):
            raise PermissionError("authorization path mismatch")
        if auth.get("fixed_cost_breakdown")!=COST or auth.get("expected_child_counts")!=EXPECTED:
            raise PermissionError("unreviewed count/cost declaration")
        if plan.get("candidate_source_index")!={"path":str(INDEX),"sha256":INDEX_SHA256} or plan.get("fixed_cost_breakdown")!=COST or plan.get("expected_child_counts")!=EXPECTED:
            raise RuntimeError("held launch-plan schema mismatch")
        if plan.get("fresh_invocation_path")!=str(INVOCATION) or plan.get("fresh_output_path")!=str(OUTPUT):
            raise RuntimeError("held sole paths mismatch")
        index=checked_json(INDEX,INDEX_SHA256)
        indexed={row["path"]:row["sha256"] for row in index["files"]}
        if len(indexed)!=8 or len(index["files"])!=8:
            raise RuntimeError("candidate source-index count mismatch")
        for path,digest in indexed.items():
            if sha(path)!=digest:
                raise RuntimeError("indexed candidate changed: "+path)
        recipe=checked_json(RECIPE,indexed[str(RECIPE)])
        childpins={str(CANDIDATE/name):indexed[str(CANDIDATE/name)] for name in CHILD_NAMES}
        if auth.get("analytic_scalar_derivative_execution_admitted") is not True or auth.get("source_pins")!=childpins:
            raise PermissionError("missing exact root child derivative release")
        if recipe.get("anticipated_rows")!={"native":100,"closed_form_controls":100,"initial_limits":28}:
            raise RuntimeError("unreviewed scientific recipe row counts")
        runtime=Path(recipe["python_runtime_path"])
        if sha(runtime)!=recipe["python_runtime_sha256"]:
            raise RuntimeError("resolved Python executable changed")
        python=Path(plan["child_python_invocation_path"])
        if python.resolve()!=runtime.resolve() or Path(sys.executable).resolve()!=runtime.resolve():
            raise RuntimeError("outer/child Python executable resolution mismatch")
        if plan["child_python_runtime"]!={"path":str(runtime),"sha256":recipe["python_runtime_sha256"]}:
            raise RuntimeError("unreviewed child runtime plan")
        # Exact source inventories are pinned by the already accepted index
        # and recipe. No scientific package or candidate module is imported.
        protected={**outerpins,str(INDEX):INDEX_SHA256,**indexed,
                   **recipe["dependency_pins"],**recipe["mpmath_python_pins"],
                   str(runtime):recipe["python_runtime_sha256"],str(authpath):sha(authpath)}
        if len(protected)!=123:
            raise RuntimeError("unexpected protected-path inventory count")
        before={path:hash_or_error(path) for path in protected}
        mismatches={path:{"expected":digest,"actual":before[path]} for path,digest in protected.items() if before[path]!=digest}
        write(INVOCATION/"source-before.json",{"expected":protected,"actual":before,"mismatches":mismatches})
        if mismatches:
            raise RuntimeError("protected source/dependency/runtime mismatch")
        sourcecopy=INVOCATION/"source-before"
        sourcecopy.mkdir(exist_ok=False)
        copies=[]
        for number,path in enumerate(sorted(protected)):
            if Path(path)==runtime:
                copies.append({"path":path,"sha256":protected[path],"kind":"executable_metadata_only"})
                continue
            target=sourcecopy/("%03d-"%number+Path(path).name)
            shutil.copyfile(path,target)
            if sha(target)!=protected[path]:
                raise RuntimeError("source-copy hash mismatch: "+path)
            copies.append({"path":path,"copy":str(target),"sha256":protected[path],"kind":"byte_exact_source_copy"})
        write(INVOCATION/"source-copy-inventory.json",copies)
        env=os.environ.copy()
        env.update({"PYTHONDONTWRITEBYTECODE":"1","OPENBLAS_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1"})
        for key in ["PYTHONPATH","PYTHONHOME","PYTHONWARNINGS"]:
            env.pop(key,None)
        command=[str(python),"-B",str(CHILD),"--recipe",str(RECIPE),"--authorization",str(authpath),"--output",str(OUTPUT)]
        expected_command=[str(authpath) if part=="{authorization}" else part for part in plan["child_command_template"]]
        if command!=expected_command:
            raise RuntimeError("held child command mismatch")
        receipt.update({"command":command,"cwd":str(HERE),"environment":{key:env.get(key) for key in ["PYTHONDONTWRITEBYTECODE","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","PYTHONPATH","PYTHONHOME","PYTHONWARNINGS"]},"source_before":before,"protected_path_count":len(protected),"source_copy_count":sum(row["kind"]=="byte_exact_source_copy" for row in copies)})
        write(INVOCATION/"launch.json",receipt)
        with (INVOCATION/"stdout.log").open("wb") as stdout,(INVOCATION/"stderr.log").open("wb") as stderr:
            child_started=True
            proc=subprocess.run(command,cwd=str(HERE),env=env,stdout=stdout,stderr=stderr,check=False)
            child_returncode=proc.returncode
        receipt["child_returncode"]=child_returncode
        if child_returncode!=0:
            raise RuntimeError("child failed; complete stdout/stderr and partial scientific outputs retained")
        child_output_before_read=directory_pins(OUTPUT)
        childreceipt=checked_json(OUTPUT/"receipt.json")
        receipt["child_output_before_read"]=child_output_before_read
        for path,digest in childreceipt.get("output_pins",{}).items():
            if sha(path)!=digest:
                raise RuntimeError("child saved-output hash mismatch: "+path)
        receipt["child_receipt"]={"path":str(OUTPUT/"receipt.json"),"sha256":sha(OUTPUT/"receipt.json")}
        if childreceipt.get("passed_analytic_scalar_derivative_gate") is not True or childreceipt.get("sources_unchanged") is not True:
            raise RuntimeError("child scientific/provenance gate did not pass")
        for key,value in EXPECTED.items():
            if childreceipt.get(key)!=value:
                raise RuntimeError("unexpected child count: "+key)
        if childreceipt.get("failed_checks")!=[] or childreceipt.get("accepted_native") is not False or childreceipt.get("inverse_map_attempted") is not False:
            raise RuntimeError("child scope or failed-check declaration mismatch")
        receipt["scientific_gate_accepted"]=True
        code=0
    except BaseException as exc:
        receipt.update({"scientific_gate_accepted":False,"exception_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        code=1
    finally:
        receipt["child_started"]=child_started
        receipt["child_returncode"]=child_returncode
        receipt["source_before"]=before
        after={path:hash_or_error(path) for path in protected}
        receipt["source_after"]=after
        receipt["sources_unchanged"]=after==before and all(after[path]==digest for path,digest in protected.items())
        if not receipt["sources_unchanged"]:
            receipt["scientific_gate_accepted"]=False
            receipt["provenance_error"]="source/dependency/runtime/authorization drift or prelaunch mismatch"
            code=1
        receipt["child_output_pins"]=directory_pins(OUTPUT)
        if child_output_before_read is not None:
            receipt["child_outputs_unchanged_during_readback"]=receipt["child_output_pins"]==child_output_before_read
            if not receipt["child_outputs_unchanged_during_readback"]:
                receipt["scientific_gate_accepted"]=False
                receipt["output_provenance_error"]="child outputs changed during outer readback"
                code=1
        receipt["invocation_output_pins"]={str(p):sha(p) for p in sorted(INVOCATION.iterdir()) if p.is_file() and p.name!="receipt.json"}
        receipt["seconds"]=time.monotonic()-started
        receipt["outer_returncode"]=code
        write(INVOCATION/"receipt.json",receipt)
    return code


if __name__=="__main__":
    raise SystemExit(main())
