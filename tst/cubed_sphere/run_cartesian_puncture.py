#!/usr/bin/env python3
"""Bounded fresh native cut-sphere wormhole controls with frozen executables."""
import argparse
import hashlib
import ast
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('binary',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--tlim',type=float,default=.001)
    ap.add_argument('--steps',type=int,default=2);ap.add_argument('--degree',type=int,default=3)
    ap.add_argument('--mass',type=float,default=.5);ap.add_argument('--timeout',type=int,default=450)
    ap.add_argument('--binary-source-receipt',type=Path)
    ap.add_argument('--pole-cfl',type=float,default=.04)
    args=ap.parse_args();root=Path(__file__).resolve().parents[2]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'athena_frozen';shutil.copy2(args.binary.resolve(),binary)
    inp=out/'input.athinput';shutil.copy2(root/'inputs/z4c/hyperboloidal_layer_wormhole.athinput',inp)
    command=[str(binary),'-i',str(inp),'-d',str(out),f'problem/mass={args.mass:.17g}',
        f'time/tlim={args.tlim:.17g}',f'time/nlim={args.steps}',
        f'z4c/hyperboloidal_ghost_degree={args.degree}',
        f'z4c/hyperboloidal_pole_cfl={args.pole_cfl:.17g}',
        'output1/dt=1','output2/dt=1','output3/dt=0.001','output4/dt=1','output5/dt=1']
    files=[*root.glob('src/**/*.cpp'),*root.glob('src/**/*.hpp'),root/'CMakeLists.txt',
           root/'src/CMakeLists.txt',Path(__file__).resolve()]
    receipt={'binary_sha256':sha(binary),'input_sha256':sha(inp),'command':command,
        'source_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(set(files))},
        'mass':args.mass,'target_time':args.tlim,'requested_steps':args.steps,'outer_degree':args.degree,
        'reference':{'kind':'Minkowski','R':1,'a':1,'r0':.35,'r1':.75},
        'gauge':'native physical-trace layer lapse, preferred_source=false',
        'scope':'One uniform Cartesian block, old cut-sphere ray ghosts, no puncture excision.'}
    if args.binary_source_receipt:
        old=json.loads(args.binary_source_receipt.read_text())
        assert old['binary_sha256']==receipt['binary_sha256']
        receipt['binary_source_sha256']=old['source_sha256']
        receipt['binary_source_receipt']=str(args.binary_source_receipt.resolve())
    shutil.copy2(Path(__file__),out/'runner.py')
    (out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:receipt['exit_code']=proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            receipt['timeout']=True;os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
    receipt['elapsed_seconds']=time.monotonic()-start
    paths=sorted(out.glob('*.hst'));hist=np.atleast_2d(np.loadtxt(paths[0])) if paths else np.empty((0,0))
    receipt['history_rows']=hist.tolist();receipt['history_all_finite']=bool(hist.size) and bool(np.isfinite(hist).all())
    receipt['outcome']='timeout' if receipt.get('timeout') else ('complete' if receipt.get('exit_code')==0 else 'failed')
    # Reuse the repository's binary reader without importing its unrelated
    # HDF5 writer dependency. This function needs only NumPy and os.
    reader_source=root/'vis/python/bin_convert.py'
    node=next(n for n in ast.parse(reader_source.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='read_binary')
    reader={'np':np,'os':os}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(reader_source),'exec'),reader)
    receipt['binary_reader_sha256']=sha(reader_source)
    snapshots={}
    for kind in ['z4c','adm','con']:
        for p in sorted((out/'bin').glob(f'*.{kind}.*.bin')):
            d={k:np.asarray(v) for k,v in reader['read_binary'](str(p))['mb_data'].items()}
            mask=d['z4c_active'].astype(bool)
            snapshots[p.name]={'active_cells':int(mask.sum()),
                'all_active_finite':all(bool(np.isfinite(v[mask]).all()) for v in d.values())}
            if kind=='z4c':
                snapshots[p.name].update({'min_alpha':float(d['z4c_alpha'][mask].min()),
                    'min_chi':float(d['z4c_chi'][mask].min())})
    receipt['snapshot_checks']=snapshots
    text=(out/'run.log').read_text(errors='replace')
    receipt['terminal_messages']=[l for l in text.splitlines() if 'what()' in l or 'FATAL ERROR' in l or 'uncaught exception' in l]
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'outcome':receipt['outcome'],'elapsed_seconds':receipt['elapsed_seconds'],
        'last_history':receipt['history_rows'][-1:] ,'terminal_messages':receipt['terminal_messages']}))


if __name__=='__main__':main()
