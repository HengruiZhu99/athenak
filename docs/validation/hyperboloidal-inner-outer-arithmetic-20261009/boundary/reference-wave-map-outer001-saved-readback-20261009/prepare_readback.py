#!/usr/bin/env python3
"""Capture immutable saved-data/source pins only; no scientific imports."""
import ast
import hashlib
import json
import pathlib
import sys
import time

HERE=pathlib.Path(__file__).resolve().parent
OUTER=HERE.parent/"reference-wave-map-outer-arithmetic-source001-held-20261009"
ROOT=HERE.parent.parent/"outer-reference-wave-map-arithmetic-source001-root-release-20261009"
REVIEW=HERE.parent.parent/"continuum/reference-wave-map-outer-arithmetic-independent-review-20261009"

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()
def pin(path):
    p=pathlib.Path(path).absolute()
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}
def read(path):return json.loads(pathlib.Path(path).read_text())
def write(path,value):
    with pathlib.Path(path).open("x") as out:
        json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write("\n")
def check(row):
    assert pin(row["path"])=={k:row[k] for k in ("path","sha256","bytes")},row["path"]

def main():
    start=time.time();assert not (HERE/"source-index.json").exists()
    tree=ast.parse((HERE/"read_saved.py").read_text())
    imports={alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imports=={"collections","decimal","hashlib","itertools","json","pathlib","sys","time"}
    assert not any(isinstance(node,ast.ImportFrom) for node in ast.walk(tree))
    outer_index=pin(OUTER/"source-index.json")
    assert outer_index["sha256"]=="9da21a96d243441c95f276d1c076d3a4e561fc20921858be33f911cb3464f27e"
    original_recipe=read(OUTER/"recipe.json")
    protected={}
    def add(row):
        assert not (row["path"] in protected and protected[row["path"]]!=row),row["path"]
        protected[row["path"]]={k:row[k] for k in ("path","sha256","bytes")}
    for row in read(OUTER/"input-pins.json")+read(OUTER/"source-index.json")["files"]:add(row)
    add(outer_index)
    add(original_recipe["source003_failed_receipt"])
    add(original_recipe["source003_failed_report"])
    for row in read(REVIEW/"index.json")["files"]:add(row)
    add(pin(REVIEW/"index.json"))
    for name in ("authorization.json","release.json","source-review001.json","launch.py","prepare_release001.py"):
        add(pin(ROOT/name))
    cases=[]
    for build,childsha,rootname in [
        ("release","e79fb5801e440319ccde5bd52dc372e587d4958382d8e51aa6ab538fea613401","release-invocation001"),
        ("debug","9aca006638cdb6c2900411020a6b7504a5e778652130ea7642e3e92dba2843a8","debug-invocation001")]:
        attempt=OUTER/"attempts"/original_recipe["attempt_names"][build]
        cp=pin(attempt/"receipt.json");assert cp["sha256"]==childsha
        child=read(cp["path"]);rp=pin(ROOT/rootname/"receipt.json");root=read(rp["path"])
        assert child["completed"] and child["passed"] and child["returncode"]==0 and child["source_inputs_unchanged"]
        assert root["completed"] and root["accepted_local_gate"] and root["returncode"]==0 and root["inputs_unchanged"]
        assert root["child_receipt_sha256"]==childsha
        add(cp);add(rp)
        for row in child["output_inventory"]+root["output_inventory"]:add(row)
        for row in read(attempt/"dependencies.json"):add(row)
        for path in sorted((ROOT/rootname).rglob("*")):
            if path.is_file():add(pin(path))
        cases.append({"build":build,"attempt_path":str(attempt),"child_receipt":cp,"root_receipt":rp})
    for name in ("PLAN.md","read_saved.py","prepare_readback.py"):
        add(pin(HERE/name))
    add(pin(pathlib.Path(sys.executable)))
    rows=sorted(protected.values(),key=lambda row:row["path"])
    for row in rows:check(row)
    write(HERE/"input-pins.json",rows)
    recipe={"scope":"stdlib completed saved Release+Debug readback only; no oracle/target/source queries",
        "execution_scope_authorized_by_parent":True,"cases":cases,"modes":original_recipe["modes"],
        "outer_source_index_sha256":outer_index["sha256"],"outer_recipe_sha256":sha(OUTER/"recipe.json"),
        "geometry_raw22_indices":original_recipe["geometry_raw22_indices"],
        "expected_record_counts":original_recipe["expected_record_counts"],
        "source003_failed_receipt":original_recipe["source003_failed_receipt"],
        "source003_failed_report":original_recipe["source003_failed_report"],
        "input_pins":str(HERE/"input-pins.json"),"output_attempt":str(HERE/"attempt001"),
        "original_payloads_metadata_only":True,"no_target_recomputation":True}
    write(HERE/"recipe.json",recipe)
    for row in rows:check(row)
    write(HERE/"preparation-receipt.json",{"completed":True,"returncode":0,"inputs_unchanged":True,
        "protected_input_count":len(rows),"stdlib_source_sanity_passed":True,
        "wall_seconds":time.time()-start,"scientific_imports_or_queries":False})
    files=[pin(path) for path in sorted(HERE.iterdir()) if path.is_file()]
    write(HERE/"source-index.json",{"files":files,"source_only":True,"scope":"saved-output readback source/pins",
        "protected_input_count":len(rows),"scientific_execution":False})
    print(json.dumps({"prepared":True,"protected_inputs":len(rows),"source_index_sha256":sha(HERE/"source-index.json")}))

if __name__=="__main__":main()
