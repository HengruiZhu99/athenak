#!/usr/bin/env python3
"""Bounded CPU mapped-original-formalism controls with immutable reference hashes.

Completion of the angular control is a smoke result. It is not stability
acceptance; the old failure's much smaller minimum Omega is reported separately.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--amplitude',type=float,default=0)
    parser.add_argument('--bulk',type=int,default=16)
    parser.add_argument('--shell',type=int,default=12)
    parser.add_argument('--tlim',type=float,default=.01)
    parser.add_argument('--cfl',type=float,default=.025)
    parser.add_argument('--degree',type=int,choices=[2,3,4,5],default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    directory=args.output.resolve(); directory.mkdir(parents=True,exist_ok=False)
    binary=directory/'athena_frozen'; shutil.copy2(args.binary,binary)
    source=root/'inputs/z4c/cubed_sphere_conformal_probe.athinput'
    shutil.copy2(source,directory/'input.athinput')
    files=[*root.glob('src/**/*.cpp'),*root.glob('src/**/*.hpp'),root/'CMakeLists.txt',
           root/'src/CMakeLists.txt',source,Path(__file__).resolve()]
    receipt={'binary_sha256':sha(binary),
             'source_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(set(files))},
             'scope':'Original wave-map/geometric formalism; finite-Omega mapped CPU probe only.',
             'old_failed_minimum_Omega':.0021701388888885099}
    command=[str(binary),'-i',str(directory/'input.athinput'),'-d',str(directory),
             'mesh/nx1=%d'%args.bulk,'mesh/nx2=%d'%args.bulk,'mesh/nx3=%d'%args.bulk,
             'cubed_sphere/nr=%d'%args.shell,'cubed_sphere/nangular=%d'%args.shell,
             'cubed_sphere/conformal_amplitude=%.17g'%args.amplitude,
             'cubed_sphere/conformal_probe_tlim=%.17g'%args.tlim,
             'cubed_sphere/conformal_cfl=%.17g'%args.cfl,
             'cubed_sphere/conformal_outer_degree=%d'%args.degree]
    receipt['command']=command; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); start=time.monotonic()
    with (directory/'run.log').open('w') as log:
        try:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            receipt['exit_code']=proc.returncode
        except subprocess.TimeoutExpired:
            receipt['timeout']=True
    receipt['elapsed_seconds']=time.monotonic()-start
    rows=[]
    path=directory/'cubed_sphere_conformal.csv'
    if path.exists():
        rows=list(csv.DictReader(path.open()))
        receipt['last_rows']=rows[-2:]
        receipt['all_rows_finite']=all(math.isfinite(float(v)) for r in rows
                                      for k,v in r.items() if k!='region')
    refs=[directory/('cubed_sphere_reference_'+s+'.txt') for s in ['before','after']]
    receipt['reference_payloads']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size}
                                   for p in refs if p.exists()}
    receipt['reference_unchanged']=all(p.exists() for p in refs) and sha(refs[0])==sha(refs[1])
    receipt['stationary_acceptance']=args.amplitude==0 and bool(rows) and all(
        abs(float(r['max_deviation']))<1e-10 and abs(float(r['H_linf']))<1e-8 and
        abs(float(r['M_conformal_l2']))<1e-8 for r in rows)
    (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert receipt.get('exit_code')==0 and receipt['reference_unchanged'] and receipt['all_rows_finite'],receipt
    assert all(abs(float(r['time'])-args.tlim)<1e-14 for r in rows[-2:]),receipt
    if args.amplitude==0:
        assert receipt['stationary_acceptance'],receipt
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    main()
