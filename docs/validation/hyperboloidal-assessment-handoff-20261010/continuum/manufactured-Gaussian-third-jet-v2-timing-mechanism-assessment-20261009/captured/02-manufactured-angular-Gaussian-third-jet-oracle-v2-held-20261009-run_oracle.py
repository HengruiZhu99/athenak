"""HELD one-shot analytic oracle. Only stdlib executes before exact admission."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def verify(pins):
    for name,digest in pins.items():
        if sha(name)!=digest:
            raise RuntimeError("protected input drift: "+name)


def registry(recipe,mp):
    def rat(value):
        f=Fraction(value)
        return mp.mpf(f.numerator)/f.denominator
    directions=[]
    for value in recipe["angular_p"]:
        p=rat(value)
        n1=mp.sqrt((1+mp.sqrt(1-4*p*p))/2)
        directions.append(("p="+value,[n1,p/n1,mp.mpf(0)]))
    directions += [("e2",[mp.mpf(0),mp.mpf(1),mp.mpf(0)]),("e3",[mp.mpf(0),mp.mpf(0),mp.mpf(1)]),
                   ("111",[1/mp.sqrt(3)]*3),("2,-3,6",[mp.mpf(2)/7,-mp.mpf(3)/7,mp.mpf(6)/7])]
    for eps in recipe["epsilon"]:
        for rlabel in recipe["radii"]:
            r=rat(rlabel)
            for tlabel in recipe["times"]:
                for dlabel,n in ([("origin",[mp.mpf(0)]*3)] if r==0 else directions):
                    key="eps"+eps+"/r"+rlabel+"/t"+tlabel+"/"+dlabel
                    yield key,[rat(tlabel)]+[r*v for v in n],rat("7/20"),rat(eps),False,Fraction(rlabel)
    q=rat("3/4")/mp.sqrt(2)
    yield "negative/sigma1/2/eps3/4/r3/4/t1/10/p1/2",[rat("1/10"),q,q,mp.mpf(0)],rat("1/2"),rat("3/4"),True,Fraction("3/4")


def scientific(stage,recipe,out):
    sys.path.insert(0,recipe["mpmath_parent"])
    import mpmath as mp
    if Path(mp.__file__).resolve()!=Path(recipe["mpmath_init"]).resolve():
        raise RuntimeError("unexpected mpmath import")
    sys.path.insert(0,str(HERE))
    import values_context,oracle,diagnostics,units,taylor3,reference3,gaussian3,geometry
    modules=(values_context,oracle,diagnostics,units,taylor3,reference3,gaussian3,geometry)
    if any(Path(module.__file__).resolve()!=HERE/(module.__name__+".py") for module in modules):
        raise RuntimeError("unexpected analytic module origin")
    def stringify(value):
        if isinstance(value,dict):return {str(k):stringify(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [stringify(v) for v in value]
        if isinstance(value,(str,bool,int)) or value is None:return value
        return diagnostics.number(value)
    failed=[]
    failed_count=0
    def check_rows(rows,tolerance):
        nonlocal failed_count
        for row in rows:
            if row["admission_gate"] and mp.mpf(row["scaled"])>mp.mpf(tolerance):
                failed_count+=1
                if len(failed)<recipe['failure_summary_cap']:
                    failed.append(dict(name=row["name"],branch=row["branch"],absolute=row["absolute"],
                                       term_sum=row["term_sum"],scaled=row["scaled"]))
    if stage=="units":
        mp.mp.dps=recipe["unit_digits"]
        layer=values_context.Layer(recipe["layer"],recipe["unit_height_order"])
        rows=units.unit_checks(layer,recipe["unit_roots"])
        check_rows(rows,recipe["identity_tolerance"])
        save(out/"unit-checks.json",rows)
        return dict(passed=failed_count==0,records=0,checks=len(rows),failed=failed,failed_total=failed_count)
    comparisons={}
    record_count=identity_count=precision_count=0
    contexts=[]
    begin=time.monotonic()
    with (out/"oracle.jsonl").open("x") as sink,(out/"precision-checks.jsonl").open("x") as precision_sink:
        for level in recipe["levels"]:
            mp.mp.dps=level["digits"]
            layer=values_context.Layer(recipe["layer"],recipe["height_order"])
            height_low=values_context.Layer(recipe["layer"],recipe["height_comparison_order"])
            height_rows=[]
            for label in recipe["radii"]:
                f=Fraction(label);r=mp.mpf(f.numerator)/f.denominator
                height_rows.append(diagnostics.comparison("height/"+label,layer.height(r),height_low.height(r),True,"scalar_context_only"))
            check_rows(height_rows,recipe["height_tolerance"])
            contexts.append(dict(digits=level["digits"],outer_height_constant=diagnostics.number(layer.outer_constant),height_checks=height_rows))
            events=list(registry(recipe,mp))
            if len(events)!=recipe["records_per_level"]:
                raise RuntimeError("fixed complete analytic registry count drift")
            if stage=="timing":
                by_key={entry[0]:entry for entry in events}
                events=[by_key[key] for key in recipe["timing_keys"]]
            for key,event,sigma,epsilon,negative,nominal_radius in events:
                print(json.dumps(dict(event="begin",key=key,digits=level["digits"],elapsed=time.monotonic()-begin)),flush=True)
                started=time.monotonic()
                data=oracle.construct(event,layer,sigma,epsilon,level["roots"])
                if data["refused"]:
                    if not (negative and data["J"]>0 and data["D"]<0):
                        raise ArithmeticError("unexpected ADM refusal or missing negative-control signs")
                    record=dict(key=key,digits=level["digits"],refused=True,J=diagnostics.number(data["J"]),D=diagnostics.number(data["D"]),root=stringify(data["root"]),
                                no_ADM_square_root=True,admission="negative_control_only")
                else:
                    if negative:raise ArithmeticError("negative slicing control unexpectedly admitted")
                    rows,aux=diagnostics.all_checks(data,nominal_radius,Fraction(recipe["graph_radius"]))
                    check_rows(rows,recipe["identity_tolerance"])
                    identity_count+=len(rows)
                    record=diagnostics.export_record(data,rows,aux)
                    record.update(key=key,digits=level["digits"],root=stringify(data["root"]))
                    component={"J":record["J"],"D":record["D"]}
                    for k,field in enumerate(record["fields"]):
                        for item in field["ordinary"]:
                            component["field%d/"%k+str(item["multiindex"])]=item["value"]
                    for k,value in enumerate(record["exact_time_rates"]):component["rate%d"%k]=value
                    for item in record["Omega"]["ordinary"]:component["Omega/"+str(item["multiindex"])]=item["value"]
                    if len(component)!=188:raise RuntimeError("compact component schema/count drift")
                    if level==recipe["levels"][0]:
                        comparisons[key]=component
                    else:
                        if set(component)!=set(comparisons[key]):raise RuntimeError("precision component schema drift")
                        for label,value in component.items():
                            row=diagnostics.comparison(key+"/"+label,mp.mpf(comparisons[key][label]),mp.mpf(value),True,"110_vs_150_compact_component")
                            precision_sink.write(json.dumps(row,allow_nan=False)+"\n")
                            check_rows([row],recipe["component_tolerance"])
                            precision_count+=1
                record["nominal_radius"]=str(nominal_radius)
                record["finite_graph_admission"]=not negative and nominal_radius<=Fraction(recipe["graph_radius"])
                record["elapsed_seconds"]=time.monotonic()-started
                sink.write(json.dumps(record,allow_nan=False)+"\n");sink.flush()
                record_count+=1
                print(json.dumps(dict(event="complete",key=key,digits=level["digits"],records=record_count,failed=failed_count,elapsed=time.monotonic()-begin)),flush=True)
                if time.monotonic()-begin>recipe["stage_seconds"][stage]:raise TimeoutError("declared oracle resource cap")
                if sink.tell()+precision_sink.tell()>recipe["payload_byte_cap"]:raise RuntimeError("declared payload byte cap")
    if stage=="full" and (record_count,identity_count,precision_count)!=(5010,3330320,470752):
        raise RuntimeError("full fixed counts drift")
    if stage=="timing" and record_count!=2*len(recipe["timing_keys"]):raise RuntimeError("timing registry count drift")
    save(out/"height-context.json",contexts)
    return dict(passed=failed_count==0,records=record_count,identity_checks=identity_count,
                precision_checks=precision_count,failed=failed,failed_total=failed_count,
                failed_summary_truncated=(failed_count>len(failed)),
                all_failure_rows_retained_in='oracle.jsonl and precision-checks.jsonl; height-context.json',
                full_registry=(stage=="full"),
                no_native_queries=True,no_inverse_coverage_or_BH_adoption=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("units","timing","full"),required=True)
    parser.add_argument("--recipe",required=True);parser.add_argument("--authorization",required=True)
    parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();before={};pins={};prerequisite_pins={};accepted=False
    receipt=dict(stage=args.stage,argv=sys.argv,scientific_imports_started=False,
                 completed=False,passed=False,scientific_stage_authorized=False)
    try:
        if sys.flags.optimize!=0:raise RuntimeError("optimized Python is not admitted")
        if sys.flags.isolated!=1 or sys.flags.dont_write_bytecode!=1:
            raise RuntimeError("the exact isolated -I -B execution route is required")
        if Path(args.recipe).resolve()!=HERE/"recipe.json":raise RuntimeError("consumed recipe must be the indexed local recipe")
        if sha(args.authorization)!=args.authorization_sha256:
            raise RuntimeError('consumed authorization hash mismatch')
        recipe,auth=load(args.recipe),load(args.authorization)
        index=load(HERE/"source-index.json")
        if auth.get("Gaussian_third_jet_oracle_stage_authorized")!=args.stage:raise RuntimeError("missing exact stage release")
        required={"source_index_sha256":sha(HERE/"source-index.json"),"recipe_sha256":sha(args.recipe),"driver_sha256":sha(__file__)}
        if any(auth.get(k)!=v for k,v in required.items()):raise RuntimeError("release/source binding fails")
        if Path(auth["output"]).resolve()!=out:raise RuntimeError("output is not the exact single-use release path")
        if args.stage in ("timing","full"):
            unit=load(auth["unit_receipt"]["path"])
            if sha(auth["unit_receipt"]["path"])!=auth["unit_receipt"]["sha256"] or not(unit.get("completed") and unit.get("passed") and unit.get("sources_unchanged")):
                raise RuntimeError("successful exact local unit prerequisite is required")
            if unit.get("source_index_sha256")!=required["source_index_sha256"]:raise RuntimeError("unit prerequisite is from another candidate")
            if unit.get('stage')!='units':raise RuntimeError('unit prerequisite stage mismatch')
            unit_path=Path(auth['unit_receipt']['path']).resolve()
            prerequisite_pins[str(unit_path)]=auth['unit_receipt']['sha256']
            prerequisite_pins.update({str(unit_path.parent/rel):h for rel,h in unit['output_hashes'].items()})
        if args.stage=="full":
            timing=load(auth["timing_receipt"]["path"])
            if sha(auth["timing_receipt"]["path"])!=auth["timing_receipt"]["sha256"] or not(timing.get("completed") and timing.get("passed") and timing.get("sources_unchanged")):
                raise RuntimeError("successful exact timing prerequisite is required")
            if timing.get("source_index_sha256")!=required["source_index_sha256"]:raise RuntimeError("timing prerequisite is from another candidate")
            if timing.get('stage')!='timing':raise RuntimeError('timing prerequisite stage mismatch')
            timing_path=Path(auth['timing_receipt']['path']).resolve()
            prerequisite_pins[str(timing_path)]=auth['timing_receipt']['sha256']
            prerequisite_pins.update({str(timing_path.parent/rel):h for rel,h in timing['output_hashes'].items()})
            review_pin=auth["measured_timing_review"]
            review_path=Path(review_pin["path"]).resolve()
            if sha(review_path)!=review_pin["sha256"]:
                raise RuntimeError("measured timing review pin mismatch")
            measured_review=load(review_path)
            review_required={"source_index_sha256":required["source_index_sha256"],
                             "timing_receipt_sha256":auth["timing_receipt"]["sha256"],
                             "timing_result_sha256":timing["output_hashes"]["result.json"]}
            if any(measured_review.get(k)!=v for k,v in review_required.items()):
                raise RuntimeError("measured timing review source/receipt/result binding mismatch")
            if not(measured_review.get("full_stage_source_review_passed") is True and
                   measured_review.get("full_stage_cost_admission") is True):
                raise RuntimeError("explicit source and measured-cost full-stage approval required")
            prerequisite_pins[str(review_path)]=review_pin["sha256"]
        if Path(sys.executable).resolve()!=Path(recipe["python_runtime_path"]).resolve():raise RuntimeError("unexpected Python runtime")
        for name,value in recipe["environment"].items():
            if os.environ.get(name)!=value:raise RuntimeError("fixed environment mismatch: "+name)
        if any(name in os.environ for name in ("PYTHONHOME","PYTHONPATH","PYTHONWARNINGS","PYTHONSTARTUP","PYTHONUSERBASE")):
            raise RuntimeError("injected Python environment is not admitted")
        pins={**recipe["protected_inputs"],**index["files"],**auth["review_pins"],**prerequisite_pins}
        pins[str(HERE/"source-index.json")]=required["source_index_sha256"]
        pins[str(Path(args.authorization).resolve())]=sha(args.authorization)
        verify(pins);before={p:sha(p) for p in pins}
        save(out/"source-before.json",before)
        receipt.update(required);receipt["scientific_stage_authorized"]=True;receipt["scientific_imports_started"]=True
        result=scientific(args.stage,recipe,out)
        save(out/"result.json",result)
        receipt.update(completed=True,passed=result["passed"])
        accepted=result["passed"]
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        (out/"failure.txt").write_text(traceback.format_exc())
    finally:
        after={}
        for p in pins:
            try:after[p]=sha(p)
            except OSError as error:after[p]=dict(error=str(error))
        receipt["sources_unchanged"]=bool(before) and before==after
        if not receipt["sources_unchanged"]:
            receipt["passed"]=False
            accepted=False
        receipt["elapsed_seconds"]=time.monotonic()-started
        save(out/"source-after.json",after)
        receipt["output_hashes"]={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.name!="receipt.json"}
        save(out/"receipt.json",receipt)
    return 0 if accepted and receipt["sources_unchanged"] else 1


if __name__=="__main__":
    raise SystemExit(main())
