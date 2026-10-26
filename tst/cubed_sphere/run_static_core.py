#!/usr/bin/env python3
"""Short regular-Z4c central mesh-refinement control at fixed shell N=24.

Uses native static octree refinement. Dynamic refine/coarsen coverage is in
check_evolution.py; this case holds the refined central region through t=2.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def make_input(source):
    text=(source/'inputs/z4c/cubed_sphere.athinput').read_text()
    assert text.count('refinement = none')==1
    return text.replace('refinement = none','refinement = static\nnum_levels = 2')+'''
<refined_region1>
level = 1
x1min = -0.5
x1max = 0.5
x2min = -0.5
x2max = 0.5
x3min = -0.5
x3max = 0.5
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary',type=Path)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--timeout',type=int,default=240)
    args=parser.parse_args()
    assert 0<args.timeout<=450
    source=args.source.resolve(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    binary=output/'athena_frozen'; shutil.copy2(args.binary,binary)
    input_file=output/'input.athinput'; input_file.write_text(make_input(source))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=[*source.glob('src/**/*.cpp'),*source.glob('src/**/*.hpp'),source/'CMakeLists.txt',
           source/'src/CMakeLists.txt',Path(__file__).resolve()]
    receipt={'binary_sha256':sha(binary),'input_sha256':sha(input_file),
             'source_sha256':{str(p.relative_to(source)):sha(p) for p in sorted(set(files))},
             'scope':'Static central octree refinement, B32 root/effective64 center, fixed S24.'}
    command=[str(binary),'-i',str(input_file),'-d',str(output),
             'cubed_sphere/nr=24','cubed_sphere/nangular=24',
             'problem/initial_data=angular_pulse','problem/amplitude=0.2',
             'cubed_sphere/outer_boundary=sommerfeld','time/tlim=2']
    receipt['command']=command; start=time.monotonic()
    env=dict(os.environ,ZE_AFFINITY_MASK='0.0',ONEAPI_DEVICE_SELECTOR='level_zero:gpu')
    with (output/'run.log').open('w') as log:
        try:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=args.timeout)
            receipt['exit_code']=proc.returncode
        except subprocess.TimeoutExpired:
            receipt['timeout']=True
    receipt['elapsed_seconds']=time.monotonic()-start
    path=output/'cubed_sphere_errors.csv'
    if path.exists():
        receipt['norms']={r['region']:{k:float(v) for k,v in r.items() if k!='region'}
                          for r in csv.DictReader(path.open())}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert receipt.get('exit_code')==0,receipt
    assert receipt['norms']['bulk']['blocks']==120,receipt
    assert all(math.isfinite(v) for row in receipt['norms'].values() for v in row.values()),receipt
    assert all(row['time']==2 for row in receipt['norms'].values()),receipt
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    main()
