#!/usr/bin/env python3
"""Bounded read-only action on the observed V5 direction; no evolution.

Keeps the historical V3/V4 decoder and evidence unchanged. Requires the exact
worker-bound V5 duration checkpoint and independently verified V5 header.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scri_candidate_replay as candidate
import scri_directional_replay as prior

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/hyperbolic-shift-3d/reproduction/aurora'


def checkpoint(path):
    decoder_path = EVIDENCE/'preflight-sampled-normal-v1/verification/verify.py'
    admission = json.loads((EVIDENCE/'preflight-sampled-normal-v1/admission.json').read_text())
    if prior.sha(decoder_path) != admission['verifier_sha256']:
        raise ValueError('V5 decoder provenance differs')
    decoder = prior.module('v5_independent_decoder', decoder_path)
    meta = decoder.checkpoint(path)
    if (meta['n'], meta['kind'], meta['time'], meta['dt'], meta['steps'], meta['scri_dissipation_epsilon']) != (16, 0, .6, .0005, 1201, 2):
        raise ValueError('Not the declared V5 duration checkpoint')
    worker_path = path.parent.parent/'worker-status.json'
    worker = json.loads(worker_path.read_text())
    bound = [(c['name'], k) for c in worker['cases'] for k, v in c['artifacts'].items()
             if Path(k).name == path.name and v['sha256'] == meta['sha256'] and v['bytes'] == meta['bytes']]
    identity = json.loads((EVIDENCE/'stability-sampled-normal-minkowski-v1/analysis/identity.json').read_text())
    if len(bound) != 1 or worker['source_manifest_sha256'] != identity['source_manifest_sha256'] or worker['plan_sha256'] != identity['plan_sha256']:
        raise ValueError('V5 checkpoint/worker/source binding mismatch')
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


def groups(direction, action):
    nv = 26*19*16**3
    result = {}
    fields = {'chi':slice(0,1), 'metric':slice(1,6), 'A':slice(6,11),
              'trace':slice(11,12), 'Theta':slice(12,13), 'Lambda':slice(13,16), 'B':slice(16,19)}
    for region, x, y in [('volume',direction[:nv].reshape(-1,26),action[:nv].reshape(-1,26)),
                         ('surface',direction[nv:].reshape(-1,19),action[nv:].reshape(-1,19))]:
        columns = dict(fields)
        if region == 'volume': columns.update(lapse=slice(19,20), beta=slice(20,23), D=slice(23,26))
        total = prior.dot(direction,direction)
        for name, sl in columns.items():
            a,b = x[:,sl],y[:,sl]; den=prior.dot(a,a)
            result[region+'_'+name] = dict(norm_squared_fraction=den/total,
                projected_rate=prior.dot(a,b)/den if den else None,
                action_norm=prior.norm(b))
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if not args.worker:
        command = [sys.executable,str(Path(__file__).resolve()),'--worker',
            '--library',str(args.library.resolve()),'--frozen-library',str(args.frozen_library.resolve()),
            '--checkpoint',str(args.checkpoint.resolve()),'--output',str(args.output/'worker')]
        start=time.monotonic()
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=385)
            result=dict(command=command,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr,
                        seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            result=dict(command=command,status='bounded',seconds=time.monotonic()-start)
        prior.write(args.output/'execution.json',result)
        if result.get('returncode',1): raise RuntimeError('Bounded diagnostic failed: see execution.json')
        return
    started=time.monotonic(); deadline=started+350
    current_manifest=json.loads((args.library.parent/'source.json').read_text())
    frozen_manifest=json.loads((args.frozen_library.parent/'source.json').read_text())
    if prior.sha(args.library)!=current_manifest['library_sha256'] or prior.sha(args.frozen_library)!=frozen_manifest['library_sha256']:
        raise ValueError('Diagnostic library binding differs')
    # Exact compiled source identities, not the build commit alone, bind this run.
    for name,digest in current_manifest['sources'].items():
        if prior.sha(args.library.parent/'source-tree'/name)!=digest:
            raise ValueError('Candidate source snapshot differs: '+name)
    v,metadata=checkpoint(args.checkpoint.resolve()); v/=prior.norm(v)
    ad=sampled=frozen=None
    try:
        ad=candidate.Probe(args.library,16,False,deadline)
        sampled=candidate.Probe(args.library,16,True,deadline)
        frozen=prior.Probe(args.frozen_library,16,2,.0005,350)
        if not ad.size==sampled.size==frozen.size==len(v):raise ValueError('Coordinate size differs')
        rows=[]; saved={}; full_parity=0
        for amplitude in (1e-4,1e-5,1e-6):
            scale=amplitude/float(np.max(np.abs(v))); aa=[];bb=[]
            for sign in (1,-1):
                x=sign*scale*v; a=ad.evaluate(x); ref=frozen.evaluate(x,True); b=sampled.evaluate(x)
                if not candidate.exact(a,ref):raise ValueError('Default AD differs from frozen full array')
                full_parity+=1;aa.append(a);bb.append(b)
            a=(aa[0]-aa[1])/(2*scale);b=(bb[0]-bb[1])/(2*scale);saved[amplitude]=(a,b)
            rows.append(dict(amplitude=amplitude,AD=candidate.extra_metrics(v,a,16),
                V5=candidate.extra_metrics(v,b,16),fields=groups(v,b)))
        for row in rows:
            a,b=saved[row['amplitude']]; ra,rb=saved[1e-5]
            row.update(relative_AD_change=prior.norm(a-ra)/prior.norm(ra),relative_V5_change=prior.norm(b-rb)/prior.norm(rb))
        # Decompose the observed direction with native old terms and replace only
        # the normal term. Independently compare its sum with full candidate RHS.
        scale=1e-5/float(np.max(np.abs(v))); parts=[]; stats=[]
        for sign in (1,-1):
            _,terms,check=frozen.surface_parts(sign*scale*v)
            sampled_normal,_,_,check2=frozen.sampled_normal(sign*scale*v)
            terms[1]=sampled_normal
            parts.append(terms);stats.append([check.tolist(),check2.tolist()])
        terms=(parts[0]-parts[1])/(2*scale); nv=ad.volume_size
        surface=saved[1e-5][1][nv:]; summed=terms.sum(axis=0)
        reconstruction=prior.norm(summed-surface)/prior.norm(surface)
        if reconstruction>1e-5:raise ValueError('Replaced native term sum differs from complete V5 surface action')
        names=['regular','sampled_normal','weak_numerator','regularity','angular_Q','B_flux']
        den=prior.dot(v[nv:],v[nv:])
        components={name:dict(projected_contribution=prior.dot(v[nv:],term)/den,action_norm=prior.norm(term))
                    for name,term in zip(names,terms)}
        result=dict(status='completed',scope='Observed V5 direction; tangent generator at Minkowski, not an eigenvalue or stability proof',
            norm='Unweighted point-coordinate Euclidean; Frobenius tracefree tensors, not physical energy',
            checkpoint=metadata,rows=rows,surface_terms=components,surface_reconstruction_relative_error=reconstruction,
            term_diagnostics=stats,default_full_array_frozen_parity_evaluations=full_parity,
            native_diagnostics=dict(AD=dict(zip(candidate.NAMES,ad.stats.tolist())),V5=dict(zip(candidate.NAMES,sampled.stats.tolist()))),
            library_sha256=prior.sha(args.library),source_manifest_sha256=prior.sha(args.library.parent/'source.json'),
            frozen_library_sha256=prior.sha(args.frozen_library),script_sha256=prior.sha(__file__),
            imported_script_sha256={str(Path(m.__file__).relative_to(ROOT)):prior.sha(m.__file__) for m in (prior,candidate)},
            seconds=time.monotonic()-started)
        prior.write(args.output/'action.json',result)
    finally:
        for probe in (ad,sampled,frozen):
            if probe is not None:probe.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--frozen-library',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');run(parser.parse_args())
