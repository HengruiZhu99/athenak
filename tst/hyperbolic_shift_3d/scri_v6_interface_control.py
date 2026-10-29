#!/usr/bin/env python3
"""Read-only logical-halo control on the actual V6 disturbance and RHS span."""
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
import scri_v6_direction as direction
import scri_candidate_replay as candidate
import scri_directional_replay as prior


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if not args.worker:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--library',str(args.library.resolve()),
                 '--frozen-library',str(args.frozen_library.resolve()),'--checkpoint',str(args.checkpoint.resolve()),'--output',str(args.output/'worker'),'--variant',str(args.variant)]
        start=time.monotonic()
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=385)
            result=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            result=dict(command=command,status='bounded',seconds=time.monotonic()-start)
        prior.write(args.output/'execution.json',result)
        if result.get('returncode',1):raise RuntimeError('Diagnostic failed: see execution.json')
        return
    start=time.monotonic()
    for lib in (args.library,args.frozen_library):
        manifest=json.loads((lib.parent/'source.json').read_text())
        if prior.sha(lib)!=manifest['library_sha256']:raise ValueError('Library binding differs')
        for name,digest in manifest['sources'].items():
            if prior.sha(lib.parent/'source-tree'/name)!=digest:raise ValueError('Source snapshot differs')
    v,meta=direction.checkpoint(args.checkpoint.resolve());v/=prior.norm(v)
    probes=[];parity=0
    try:
        for lib,variant in [(args.library,2),(args.frozen_library,2),(args.library,args.variant)]:
            probes.append(candidate.Probe(lib,16,variant,start+350))
        base,frozen,logical=probes
        def actions(x,amplitude):
            nonlocal parity
            scale=amplitude/float(np.max(np.abs(x)));a=[];b=[]
            for sign in (1,-1):
                z=sign*scale*x;y=base.evaluate(z);ref=frozen.evaluate(z)
                if not candidate.exact(y,ref):raise ValueError('V6 changed from frozen full array')
                parity+=1;a.append(y);b.append(logical.evaluate(z))
            return (a[0]-a[1])/(2*scale),(b[0]-b[1])/(2*scale)
        rows=[]
        for amplitude in (1e-5,1e-6):
            jv,lv=actions(v,amplitude);q=jv-prior.dot(v,jv)*v;q/=prior.norm(q)
            jq,lq=actions(q,amplitude);row=dict(amplitude=amplitude)
            for name,a,b in [('V6',jv,jq),('logical_halos' if args.variant==3 else 'logical_inner_transition',lv,lq)]:
                matrix=np.array([[prior.dot(v,a),prior.dot(v,b)],[prior.dot(q,a),prior.dot(q,b)]])
                row[name]=dict(projected_matrix=matrix.tolist(),
                    column_relative_residuals=[prior.norm(z-v*matrix[0,i]-q*matrix[1,i])/prior.norm(z) for i,z in enumerate((a,b))],
                    projected_eigenvalues=[dict(real=float(z.real),imag=float(z.imag)) for z in np.linalg.eigvals(matrix)],
                    direction_metrics=candidate.extra_metrics(v,a,16))
            row['relative_action_difference']=prior.norm(lv-jv)/prior.norm(jv);rows.append(row)
        prior.write(args.output/'interface-control.json',dict(status='completed',scope='Same observed V6 two-vector span, alternate halo interpolation only; no evolution or stability proof',
            checkpoint=meta,variant=args.variant,rows=rows,full_array_frozen_parity_evaluations=parity,
            library_sha256=prior.sha(args.library),source_manifest_sha256=prior.sha(args.library.parent/'source.json'),
            frozen_library_sha256=prior.sha(args.frozen_library),script_sha256=prior.sha(__file__),
            imported_script_sha256={str(Path(m.__file__).relative_to(direction.ROOT)):prior.sha(m.__file__) for m in (direction,prior,candidate)},
            seconds=time.monotonic()-start))
    finally:
        for p in probes:p.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--frozen-library',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--variant',type=int,choices=(3,4),default=3);run(parser.parse_args())
