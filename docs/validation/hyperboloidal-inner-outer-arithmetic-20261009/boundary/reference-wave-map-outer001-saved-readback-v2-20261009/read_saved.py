#!/usr/bin/env python3
"""Read completed saved JSON only; never import/evaluate scientific source."""
import collections
import decimal
import hashlib
import itertools
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
EXPECTED = {"coefficient":1452,"coefficient-dual":160,"core-witness":144,
            "dual":2520,"invalid-coefficient":16,"invalid-source":6,
            "nonrepresentable":18,"principal":6384,"reference":336,"source":4704}
MAX_LIMITS = {"MP-precision-contrast":"1e-180","MP-precision-dual":"1e-65",
    "MP-precision-ordinary":"1e-65","actual22-gauge-dual":"2e-10",
    "coefficient-dual":"2e-10","coefficient-value":"2e-14",
    "connection-ADM":"2e-11","connection-embedding":"2e-11",
    "core-witness-relative":"2e-10","core-witness-zero":"0",
    "independent-core":"2e-10","proposal-principal-coefficients":"2e-11",
    "source-dual-parts":"2e-10","source-dual-rhs":"2e-10",
    "source-parts":"2e-10","source-rhs":"2e-10"}

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(pathlib.Path(path).read_text())
def write(path,value):
    with pathlib.Path(path).open("x") as out:
        json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write("\n")
def check_pin(row):
    p=pathlib.Path(row["path"])
    assert p.stat().st_size==row["bytes"] and sha(p)==row["sha256"],str(p)
def dec(value):
    out=decimal.Decimal(str(value));assert out.is_finite();return out
def bits(value):
    assert isinstance(value,(float,int));return float(value).hex()
def signature(value):
    if isinstance(value,float):return ("binary64",value.hex())
    if isinstance(value,list):return tuple(signature(v) for v in value)
    if isinstance(value,dict):return tuple((k,signature(v)) for k,v in sorted(value.items()))
    return value
def finite_pair(pair):
    assert len(pair)==2
    for value in pair:dec(value)
def near(value,reference):
    # Exact exported-binary64 interval predicate only, not a source product.
    n,d=float(value).as_integer_ratio();r,s=float(reference).as_integer_ratio()
    return n>0 and r>0 and 2*n*s>=r*d and n*s<=2*r*d
def line_rows(path):
    with pathlib.Path(path).open() as stream:
        for number,line in enumerate(stream,1):
            assert line.strip(),(str(path),number,"blank saved row")
            yield number,json.loads(line,parse_int=lambda token: -0.0 if token=="-0" else int(token))

def build_readback(case,recipe):
    child=read(case["child_receipt"]["path"]);outer=read(case["root_receipt"]["path"])
    assert child["completed"] and child["passed"] and child["returncode"]==0
    assert child["source_inputs_unchanged"]
    assert outer["completed"] and outer["accepted_local_gate"] and outer["returncode"]==0 and outer["inputs_unchanged"]
    assert outer["child_receipt_sha256"]==case["child_receipt"]["sha256"]
    assert child["source_index_sha256"]==recipe["outer_source_index_sha256"]
    assert child["recipe_sha256"]==recipe["outer_recipe_sha256"]
    assert child["executable_before"]==child["executable_after"]
    assert all(command["returncode"]==0 for command in child["commands"])
    report=read(child["oracle_report"]["path"])
    assert report["passed"] and not report["failures"] and report["source_only_inputs_unchanged"]
    assert report["counts"]==EXPECTED
    assert set(report["maxima"])==set(MAX_LIMITS)
    for name,value in report["maxima"].items():assert dec(value)<=dec(MAX_LIMITS[name]),name
    assert len(report["FD_sequences"])==2520
    fdmax=[decimal.Decimal(0) for unused in range(3)]
    floor=convergent=0
    for row in report["FD_sequences"]:
        errors=[dec(x) for x in row["errors"]];assert len(errors)==3
        assert errors[-1]<=dec("5e-7")
        is_floor=max(errors)<=dec("5e-9");is_convergent=errors[0]>=2*errors[-1]
        assert is_floor or is_convergent
        floor+=int(is_floor);convergent+=int(is_convergent)
        fdmax=[max(a,b) for a,b in zip(fdmax,errors)]
    counts=collections.Counter();routes=collections.Counter();by_family=collections.Counter()
    comp_old_bits=collections.Counter();dual_bits=collections.Counter()
    coefficient_calls=reference_zero=geometry_bits=0
    mode_counts={};examples={}
    geometry_indices=recipe["geometry_raw22_indices"]
    for mode in recipe["modes"]:
        mode_count=0
        for line,row in line_rows(pathlib.Path(case["attempt_path"])/(mode+".jsonl")):
            kind=row["kind"];counts[kind]+=1;mode_count+=1
            if kind.startswith("invalid-"):
                assert row["rejected"];continue
            if kind.startswith("coefficient"):
                assert row["valid"];k=dec(row["k"][0]);assert 0<=k<=1
                if row["W"]==1:assert bits(row["k"][0])==bits(.5)
                continue
            if kind=="nonrepresentable":continue
            assert row["valid"] and row["assembled"]
            for pair in row["parts"]+row["rhs"]:finite_pair(pair)
            assert len(row["parts"])==8 and len(row["rhs"])==4
            if kind=="reference":
                assert all(pair[0]==0 for pair in row["parts"]);reference_zero+=1
            if kind=="dual":
                assert len(row["actual22"])==len(row["baseline22"])==22
                for index in geometry_indices:
                    for component in range(2):
                        assert bits(row["actual22"][index][component])==bits(row["baseline22"][index][component])
                        geometry_bits+=1
            W=row["W"]
            if W==1:
                assert row["arithmetic"]["coefficient_calls"]==0;coefficient_calls+=1
            if "baseline_parts" not in row:continue
            assert len(row["baseline_parts"])==8
            actual_near=near(row["input"]["alpha"][0],row["reference"]["alpha"][0]) and near(row["input"]["chi"][0],row["reference"]["chi"][0])
            assert row["legacy_near"]==actual_near
            primal_equal=row["valid"]==row["baseline_valid"] and all(bits(x[0])==bits(y[0]) for x,y in zip(row["parts"],row["baseline_parts"]))
            full_equal=row["valid"]==row["baseline_valid"] and all(bits(x[c])==bits(y[c]) for x,y in zip(row["parts"],row["baseline_parts"]) for c in range(2))
            assert row["outer_value_bitwise"]==primal_equal
            region="W1" if W==1 else "W0" if W==0 else "0<W<1"
            route="near" if actual_near else "far"
            routes[region+"/"+route+"/rows"]+=1
            routes[region+"/"+route+"/primal_equal"]+=int(primal_equal)
            routes[region+"/"+route+"/full_dual_equal"]+=int(full_equal)
            if region=="W1" and route=="near":assert primal_equal and full_equal
            if region=="W1" and route=="far":
                family=row.get("family",kind)
                by_family[family+"/rows"]+=1
                by_family[family+"/primal_equal"]+=int(primal_equal)
                by_family[family+"/full_dual_equal"]+=int(full_equal)
                for index,(x,y) in enumerate(zip(row["parts"],row["baseline_parts"])):
                    comp_old_bits[str(index)+"/primal_changed"]+=int(bits(x[0])!=bits(y[0]))
                    dual_bits[str(index)+"/tangent_changed"]+=int(bits(x[1])!=bits(y[1]))
                if not primal_equal and family not in examples:
                    examples[family]={"mode":mode,"line":line,"curvature":row["curvature"],
                        "radius":row["radius"][0],"direction":row["direction"],"G0":row["G0"],
                        "scope":"first saved far old-bit inequality, not a new target calculation"}
        mode_counts[mode]=mode_count
    assert dict(counts)==EXPECTED and sum(counts.values())==15740
    assert reference_zero==336 and geometry_bits==90720
    return {"build":case["build"],"passed":True,"counts":dict(counts),"mode_counts":mode_counts,
        "oracle_maxima_saved":report["maxima"],"FD_saved":{"sequences":2520,"levels":3,
            "maximum_errors_per_level":[str(v) for v in fdmax],"floor_pass_rows":floor,"convergence_pass_rows":convergent},
        "reference_zero_part_rows":reference_zero,"actual22_geometry_bit_checks":geometry_bits,
        "W1_zero_inner_coefficient_call_rows":coefficient_calls,
        "joint_field_route_counts":dict(sorted(routes.items())),"W1_far_family_old_bit_metadata":dict(sorted(by_family.items())),
        "W1_far_part_primal_changed_counts":dict(sorted(comp_old_bits.items())),
        "W1_far_part_tangent_changed_counts":dict(sorted(dual_bits.items())),
        "first_far_old_bit_inequality_per_family":examples,"executable":child["executable_after"],
        "actual_wall_seconds_root":outer["seconds"],"child_receipt":case["child_receipt"],
        "oracle_report":child["oracle_report"],"root_receipt":case["root_receipt"],
        "limitations":["saved result audit, no MP target/residual recomputation",
            "aggregate MP-dual scope checked at weakest ordinary threshold; original per-row oracle retained",
            "W<1 outputs are compound inner source, W1 far intentionally has new arithmetic identity",
            "ordinary duals do not qualify separately held far high-contrast complete-dual supplement"]}

def compare_builds(cases,recipe):
    result={}
    for mode in recipe["modes"]:
        paths=[pathlib.Path(case["attempt_path"])/(mode+".jsonl") for case in cases]
        compared=changed=parts_changed=rhs_changed=baseline_changed=actual22_changed=0
        for left,right in itertools.zip_longest(line_rows(paths[0]),line_rows(paths[1])):
            assert left is not None and right is not None
            number,a=left;other,b=right;assert number==other
            for key in ("kind","curvature","G0","direction","family","column","field","case","witness"):
                assert signature(a.get(key))==signature(b.get(key)),(mode,number,key)
            compared+=1;changed+=int(signature(a)!=signature(b))
            parts_changed+=int(signature(a.get("parts"))!=signature(b.get("parts")))
            rhs_changed+=int(signature(a.get("rhs"))!=signature(b.get("rhs")))
            baseline_changed+=int(signature(a.get("baseline_parts"))!=signature(b.get("baseline_parts")))
            actual22_changed+=int(signature(a.get("actual22"))!=signature(b.get("actual22")))
        result[mode]={"rows":compared,"complete_parsed_bit_changed_rows":changed,
            "parts_changed_rows":parts_changed,"rhs_changed_rows":rhs_changed,
            "LegacyGauge_parts_changed_rows":baseline_changed,"actual22_changed_rows":actual22_changed,
            "payload_bytes_equal":sha(paths[0])==sha(paths[1])}
    return result

def main():
    assert len(sys.argv)==2
    recipe_path=pathlib.Path(sys.argv[1]).resolve();recipe=read(recipe_path)
    source_files=read(HERE/"source-index.json")["files"]
    for row in source_files:check_pin(row)
    pins=read(recipe["input_pins"])
    for row in pins:check_pin(row)
    attempt=pathlib.Path(recipe["output_attempt"]);attempt.mkdir(exist_ok=False)
    start=time.time();completed=False;error=None
    try:
        old=read(recipe["source003_failed_receipt"]["path"])
        assert old["completed"] is False and old["returncode"]==1 and not old["passed"] and old["source_inputs_unchanged"]
        assert read(recipe["source003_failed_report"]["path"])["passed"] is False
        summaries=[build_readback(case,recipe) for case in recipe["cases"]]
        comparison=compare_builds(recipe["cases"],recipe)
        for row in pins+source_files:check_pin(row)
        summary={"passed":True,"scope":"stdlib saved-output audit only; no target/source/scientific recomputation",
            "builds":summaries,"Release_Debug_saved_bit_comparison":comparison,
            "prior_source003_overall_FAIL_preserved":True,"inputs_unchanged":True,
            "far_complete_dual_supplement_executed":False,"global_or_native_admission":False}
        write(attempt/"summary.json",summary);completed=True
        print(json.dumps({"passed":True,"builds":len(summaries),"saved_records_each":15740,
                          "summary_path":str(attempt/"summary.json")},sort_keys=True))
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        unchanged=True
        try:
            for row in pins+source_files:check_pin(row)
        except Exception as exc:
            unchanged=False
            error=error or {"type":type(exc).__name__,"message":str(exc)}
        write(attempt/"receipt.json",{"completed":completed,"returncode":0 if completed and unchanged else 1,
            "passed":completed and unchanged,"inputs_unchanged":unchanged,"error":error,
            "wall_seconds":time.time()-start,"source_only_saved_readback":True,
            "scientific_imports_or_queries":False,"target_recomputation":False})

if __name__=="__main__":main()
