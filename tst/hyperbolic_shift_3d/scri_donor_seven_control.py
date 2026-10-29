#!/usr/bin/env python3
"""Bounded coupled action control for the optional seven-node radial closure.

No evolution or stability admission. Tests both observed V4 and V5 directions
and requires full-array V5 parity with a preserved pre-change library.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import scri_candidate_replay as candidate
import scri_directional_replay as prior
import scri_v5_direction as v5


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if not args.worker:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--library',str(args.library.resolve()),
            '--reference',str(args.reference.resolve()),'--v4-checkpoint',str(args.v4_checkpoint.resolve()),
            '--v5-checkpoint',str(args.v5_checkpoint.resolve()),'--output',str(args.output/'worker')]
        start=time.monotonic()
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=385)
            execution=dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:execution=dict(command=cmd,status='bounded',seconds=time.monotonic()-start)
        prior.write(args.output/'execution.json',execution)
        if execution.get('returncode',1):raise RuntimeError('Bounded control failed; see execution.json')
        return
    started=time.monotonic();deadline=started+350
    manifests={}
    for name,library in [('new',args.library),('reference',args.reference)]:
        manifest=json.loads((library.parent/'source.json').read_text())
        if prior.sha(library)!=manifest['library_sha256']:raise ValueError('Library binding differs')
        manifests[name]=dict(library_sha256=prior.sha(library),source_manifest_sha256=prior.sha(library.parent/'source.json'))
    inputs=[('V4_t030',*prior.checkpoint(args.v4_checkpoint)),('V5_t060',*v5.checkpoint(args.v5_checkpoint))]
    old=new=reference=None
    try:
        old=candidate.Probe(args.library,16,1,deadline);new=candidate.Probe(args.library,16,2,deadline)
        reference=candidate.Probe(args.reference,16,1,deadline)
        background=new.evaluate(np.zeros(new.size));cases=[];parity=0
        for name,direction,metadata in inputs:
            direction/=prior.norm(direction);rows=[];saved={}
            for amplitude in (1e-4,1e-5,1e-6):
                scale=amplitude/np.max(np.abs(direction));a=[];b=[]
                for sign in (1,-1):
                    x=sign*scale*direction;baseline=old.evaluate(x);frozen=reference.evaluate(x)
                    if not candidate.exact(baseline,frozen):raise ValueError('Default V5 changed from frozen full RHS')
                    parity+=1;a.append(baseline);b.append(new.evaluate(x))
                ad=(a[0]-a[1])/(2*scale);bd=(b[0]-b[1])/(2*scale);saved[amplitude]=bd
                rows.append(dict(amplitude=amplitude,V5=candidate.extra_metrics(direction,ad,16),
                                 donor_seven=candidate.extra_metrics(direction,bd,16)))
            for row in rows:row['relative_change_from_1e_5']=prior.norm(saved[row['amplitude']]-saved[1e-5])/prior.norm(saved[1e-5])
            cases.append(dict(name=name,checkpoint=metadata,rows=rows))
        prior.write(args.output/'action.json',dict(status='completed',cases=cases,
            scope='Tangent generator at Minkowski on two observed directions; no eigenvalue or stability assertion',
            baseline_full_array_bitwise_parity_evaluations=parity,background_rhs_max=float(np.max(np.abs(background))),
            candidate_diagnostics=dict(zip(candidate.NAMES,new.stats.tolist())),libraries=manifests,
            script_sha256=prior.sha(__file__),imported_scripts={str(Path(m.__file__)):prior.sha(m.__file__) for m in (prior,candidate,v5)},
            seconds=time.monotonic()-started))
    finally:
        for p in (old,new,reference):
            if p is not None:p.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','reference','v4-checkpoint','v5-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker',action='store_true');run(p.parse_args())
