#!/usr/bin/env python3
"""Retain small V4 outcome records after analyze.py verified all raw payloads.

This reads the V4 checkpoint header only to confirm its unadmitted status;
it does not load a state, alter an admission bit, restart or evolve anything.
Raw scheduler environments and large binary payloads are not copied.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import struct

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v4_analysis",HERE/"analyze.py")
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--analysis",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    a.require(not args.out.exists(),"Output directory must be new")
    result=a.load(args.analysis)
    case=next(c for c in result["cases"] if c["case"] == "v4")
    a.require(case["evidence_verified"] and case["terminal"]["terminal_verified"] and
              all(x["bytes_verified"] for x in case["artifacts"]),"Full verified V4 analysis required")
    for artifact in case["artifacts"]:
        path=args.raw/"output"/Path(artifact["remote_path"]).name
        a.require(a.sha(path) == artifact["sha256"] and path.stat().st_size == artifact["bytes"],
                  "Raw artifact changed after analysis: "+str(path))
    source_paths=["plan.json","binding.json","worker-status.json","run_debug.pbs","allocation-released.json",
                  "output/minkowski.report.json","output/minkowski.history.csv","output/minkowski.log","output/affinity.log"]
    args.out.mkdir(parents=True)
    retained=[]
    for name in source_paths:
        source=args.raw/name
        target=args.out/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        a.require(a.sha(source) == a.sha(target),"Copy mismatch")
        retained.append(dict(path=name,sha256=a.sha(target),bytes=target.stat().st_size))
    collected=a.load(args.raw/"collected.json")
    scheduler=collected["scheduler"]
    terminal=dict(job_id=collected["job_id"],collected=collected["collected"],
                  scheduler={k:scheduler[k] for k in ["job_state","Exit_status","stime","obittime"] if k in scheduler},
                  collected_record=a.record(args.raw/"collected.json"),
                  scope="Whitelisted terminal facts only; scheduler environments, owner/email and account balances omitted.")
    a.write(args.out/"terminal.json",terminal)
    checkpoint=args.raw/"output/minkowski.checkpoint"
    header=struct.Struct("<16s4i3Q3d2i")
    with checkpoint.open("rb") as f: values=header.unpack(f.read(header.size))
    keys=["magic","n","fields","kind","admitted","cells","surface_scalars","steps","time","dt","epsilon","degree","reserved"]
    record=dict(zip(keys,values));record["magic"]=record["magic"].split(b"\0",1)[0].decode("ascii")
    report=a.load(args.raw/"output/minkowski.report.json")
    expected=dict(magic="HYP3D_MPI_V4",n=16,fields=31,kind=0,admitted=0,cells=19*16**3,
                  surface_scalars=31*6*16**2,steps=report["steps"],time=report["t"],dt=.0005,
                  epsilon=2.,degree=7,reserved=0)
    a.require(record == expected and report["checkpoint_admitted"] is False and report["status"] == "failed",
              "Failed checkpoint/report header mismatch")
    a.require(checkpoint.stat().st_size == header.size+16*(31*record["cells"]+record["surface_scalars"]),
              "V4 checkpoint byte count mismatch")
    a.write(args.out/"checkpoint-header-audit.json",dict(header_bytes=header.size,header=record,
        input=a.record(checkpoint),checkpoint_admitted=False,resume_allowed=False,
        interpretation="Accepted rollback state at failure, explicitly unadmitted. Header inspection only; no state loading, restart or admission change."))
    a.write(args.out/"inventory.json",dict(retained_originals=retained,
        full_artifact_verification=case["artifacts"],analysis=a.record(args.analysis),
        packaging_source=a.record(Path(__file__)),omitted=["minkowski.checkpoint","minkowski.failed-stage.bin","raw scheduler records"]))
    print(json.dumps(dict(retained_small_originals=len(retained),terminal_fields_whitelisted=True,
                          header_bytes=header.size,admitted=record["admitted"],time=record["time"]),indent=2))


if __name__ == "__main__":
    main()
