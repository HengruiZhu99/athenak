#!/usr/bin/env python3
"""Bounded full-RHS diagnostic of the observed V6 direction; no evolution."""
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
from scri_v5_direction import groups
ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/hyperbolic-shift-3d/reproduction/aurora'


def checkpoint(path):
    decoder_path=EVIDENCE/'preflight-donor-seven-v1/verification/verify.py'
    admission_path=EVIDENCE/'preflight-donor-seven-v1/admission.json'
    admission=json.loads(admission_path.read_text())
    identity=json.loads((EVIDENCE/'stability-donor-seven-minkowski-v1/analysis/identity.json').read_text())
    if prior.sha(decoder_path)!=admission['verifier_sha256'] or prior.sha(admission_path)!=identity['admission_sha256']:
        raise ValueError('V6 decoder/admission provenance differs')
    decoder=prior.module('v6_independent_decoder',decoder_path)
    meta=decoder.checkpoint(path)
    if (meta['n'],meta['kind'],meta['time'],meta['dt'],meta['steps'],meta['scri_dissipation_epsilon'])!=(16,0,1,.0005,2001,2):
        raise ValueError('Not the declared V6 duration checkpoint')
    if any(meta[k]!=v for k,v in decoder.NORMAL_FIELDS.items()):
        raise ValueError('Wrong V6 radial closure')
    worker_path=path.parent.parent/'worker-status.json'
    worker=json.loads(worker_path.read_text())
    bound=[(c['name'],k) for c in worker['cases'] for k,v in c['artifacts'].items()
           if Path(k).name==path.name and v['sha256']==meta['sha256'] and v['bytes']==meta['bytes']]
    if len(bound)!=1 or worker['source_manifest_sha256']!=identity['source_manifest_sha256'] or worker['plan_sha256']!=identity['plan_sha256']:
        raise ValueError('V6 checkpoint/worker/source binding mismatch')
    with path.open('rb') as stream:
        stream.seek(meta['header_bytes'])
        arrays = [np.fromfile(stream, dtype='<f8', count=c) for c in
                  (31*meta['cells'], 31*meta['cells'], meta['surface_size'], meta['surface_size'])]
    volume, surface = arrays[0].reshape(-1,31), arrays[2].reshape(-1,31)
    def extract(raw, fields):
        out = np.zeros((len(raw), fields)); out[:,0] = raw[:,0]-1
        for source, dest in ((1,1), (7,6)):
            xx,xy,xz,yy,yz,zz = raw[:,source:source+6].T
            out[:,dest] = (xx-yy)/np.sqrt(2)
            out[:,dest+1] = (xx+yy-2*zz)/np.sqrt(6)
            out[:,dest+2:dest+5] = np.column_stack((xy,xz,yz))*np.sqrt(2)
        out[:,11:13] = raw[:,13:15]; out[:,13:16] = raw[:,15:18]; out[:,16:19] = raw[:,19:22]
        if fields == 26:
            out[:,19] = raw[:,18]; out[:,20:23] = raw[:,22:25]; out[:,23:26] = raw[:,25:28]
        return out
    direction = np.concatenate((extract(volume,26).ravel(), extract(surface,19).ravel()))
    discarded = sum(float(np.sum((a[:,1]+a[:,4]+a[:,6])**2+(a[:,7]+a[:,10]+a[:,12])**2))/3
                    for a in (volume,surface))
    meta.update(worker_sha256=prior.sha(worker_path), bound_artifact=bound,
        scientifically_admitted=False, tangent_norm=prior.norm(direction),
        trace_discard_relative=float(np.sqrt(discarded/(prior.dot(direction,direction)+discarded))))
    return direction, meta



def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if not args.worker:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--library',str(args.library.resolve()),
                 '--frozen-library',str(args.frozen_library.resolve()),'--checkpoint',str(args.checkpoint.resolve()),
                 '--output',str(args.output/'worker')]
        start=time.monotonic()
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=385)
            result=dict(command=command,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            result=dict(command=command,status='bounded',seconds=time.monotonic()-start)
        prior.write(args.output/'execution.json',result)
        if result.get('returncode',1): raise RuntimeError('Diagnostic failed: see execution.json')
        return
    started=time.monotonic(); deadline=started+350
    for library in (args.library,args.frozen_library):
        manifest=json.loads((library.parent/'source.json').read_text())
        if prior.sha(library)!=manifest['library_sha256']: raise ValueError('Library binding differs')
        for name,digest in manifest['sources'].items():
            if prior.sha(library.parent/'source-tree'/name)!=digest: raise ValueError('Source snapshot differs: '+name)
    v,metadata=checkpoint(args.checkpoint.resolve()); v/=prior.norm(v)
    probes=[]
    try:
        v6=candidate.Probe(args.library,16,2,deadline);probes.append(v6)
        frozen=candidate.Probe(args.frozen_library,16,2,deadline);probes.append(frozen)
        v5=candidate.Probe(args.library,16,1,deadline);probes.append(v5)
        if any(p.size!=len(v) for p in probes): raise ValueError('Coordinate size differs')
        rows=[];saved={};parity=0
        for amplitude in (1e-4,1e-5,1e-6):
            scale=amplitude/float(np.max(np.abs(v)));aa=[];bb=[]
            for sign in (1,-1):
                x=sign*scale*v;a=v6.evaluate(x);ref=frozen.evaluate(x);b=v5.evaluate(x)
                if not candidate.exact(a,ref): raise ValueError('V6 differs from frozen full array')
                parity+=1;aa.append(a);bb.append(b)
            a=(aa[0]-aa[1])/(2*scale);b=(bb[0]-bb[1])/(2*scale);saved[amplitude]=(a,b)
            rows.append(dict(amplitude=amplitude,V6=candidate.extra_metrics(v,a,16),
                             V5=candidate.extra_metrics(v,b,16),fields=groups(v,a)))
        for row in rows:
            a,b=saved[row['amplitude']];ra,rb=saved[1e-5]
            row.update(relative_V6_change=prior.norm(a-ra)/prior.norm(ra),relative_V5_change=prior.norm(b-rb)/prior.norm(rb))
        result=dict(status='completed',scope='Observed V6 direction; tangent RHS at Minkowski, not an eigenvalue or stability proof',
                    norm='Unweighted point-coordinate Euclidean; Frobenius tracefree tensors, not physical energy',
                    checkpoint=metadata,rows=rows,full_array_frozen_parity_evaluations=parity,
                    native_diagnostics={name:dict(zip(candidate.NAMES,p.stats.tolist())) for name,p in [('V6',v6),('V5',v5)]},
                    library_sha256=prior.sha(args.library),source_manifest_sha256=prior.sha(args.library.parent/'source.json'),
                    frozen_library_sha256=prior.sha(args.frozen_library),script_sha256=prior.sha(__file__),
                    imported_script_sha256={str(Path(m.__file__).relative_to(ROOT)):prior.sha(m.__file__) for m in (prior,candidate,sys.modules[groups.__module__])},
                    seconds=time.monotonic()-started)
        prior.write(args.output/'action.json',result)
    finally:
        for p in probes:p.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--frozen-library',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');run(parser.parse_args())
