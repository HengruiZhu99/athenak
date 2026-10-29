#!/usr/bin/env python3
"""Read-only V7 endpoint localization and two-vector action control.

The checkpoint is never resumed. Projected eigenvalues are not a full-spectrum
or long-time stability result. Each child diagnostic has a385s wall limit.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
import scri_candidate_replay as candidate
import scri_directional_replay as prior
ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/hyperbolic-shift-3d/reproduction/aurora'
FROZEN='9022f51f26e66acafdd3190d0c33f0c1ea53a661'

def checkpoint(path,identity_path):
    decoder_path=EVIDENCE/'preflight-inner-transition-v1/verification/verify.py'
    admission_path=EVIDENCE/'preflight-inner-transition-v1/admission.json'
    admission=json.loads(admission_path.read_text())
    identity=json.loads(identity_path.read_text())
    if prior.sha(decoder_path)!=admission['verifier_sha256'] or prior.sha(admission_path)!=identity['admission_sha256']:
        raise ValueError('V7 decoder/admission provenance differs')
    decoder=prior.module('v7_independent_decoder',decoder_path)
    if identity['source_commit']!=FROZEN or identity['source_manifest_sha256']!=admission['source_manifest_sha256']:
        raise ValueError('V7 identity does not match preflight source')
    meta=decoder.checkpoint(path)
    if (meta['n'],meta['kind'],meta['time'],meta['dt'],meta['steps'],meta['scri_dissipation_epsilon'])!=(16,0,2,.0005,4001,2):
        raise ValueError('Not the declared V7 duration checkpoint')
    if any(meta[k]!=v for k,v in decoder.NORMAL_FIELDS.items()):
        raise ValueError('Wrong V7 radial closure')
    worker_path=path.parent.parent/'worker-status.json'
    worker=json.loads(worker_path.read_text())
    bound=[(c['name'],k) for c in worker['cases'] for k,v in c['artifacts'].items()
           if Path(k).name==path.name and v['sha256']==meta['sha256'] and v['bytes']==meta['bytes']]
    if len(bound)!=1 or worker['source_manifest_sha256']!=identity['source_manifest_sha256'] or worker['plan_sha256']!=identity['plan_sha256']:
        raise ValueError('V7 checkpoint/worker/source binding mismatch')
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
            '--checkpoint',str(args.checkpoint.resolve()),'--identity',str(args.identity.resolve()),'--output',str(args.output/'worker')]
        start=time.monotonic()
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=385)
            record=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            record=dict(command=command,status='bounded',seconds=time.monotonic()-start)
        prior.write(args.output/'execution.json',record)
        if record.get('returncode',1):raise RuntimeError('Diagnostic failed: see execution.json')
        return
    start=time.monotonic();manifest=json.loads((args.library.parent/'source.json').read_text())
    if prior.sha(args.library)!=manifest['library_sha256']:raise ValueError('Library binding differs')
    # This retained library predates the V7 evolution executable, but all45
    # captured CPU sources must match the frozen V7 source exactly.
    for name,digest in manifest['sources'].items():
        if prior.sha(args.library.parent/'source-tree'/name)!=digest:raise ValueError('Snapshot changed: '+name)
        blob=subprocess.check_output(['git','show',FROZEN+':'+name],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Library source differs from V7: '+name)
    v,meta=checkpoint(args.checkpoint.resolve(),args.identity);v/=prior.norm(v)
    n=16;nv=26*19*n**3;volume=v[:nv].reshape(19,n,n,n,26);w=np.sum(volume**2,axis=-1)
    regions={}
    for lo,hi,name in [(0,1,'Cartesian'),(1,7,'transition_shell'),(7,13,'middle_shell'),(13,19,'outer_shell')]:
        a=w[lo:hi];regions[name]=dict(norm_squared_fraction=float(a.sum()),i_row_fractions=a.sum(axis=(0,1,2)).tolist())
    edges=np.zeros((n,n,n),dtype=bool);edges[:2,:,:]=True;edges[-2:,:,:]=True;edges[:,:2,:]=True;edges[:,-2:,:]=True
    fields={'chi':slice(0,1),'metric':slice(1,6),'A':slice(6,11),'trace':slice(11,12),'Theta':slice(12,13),
        'Lambda':slice(13,16),'B':slice(16,19),'lapse':slice(19,20),'beta':slice(20,23),'D':slice(23,26)}
    localization=dict(regions=regions,block_norm_squared_fractions=w.sum(axis=(1,2,3)).tolist(),
        volume_field_norm_squared_fractions={k:float(np.sum(volume[...,sl]**2)) for k,sl in fields.items()},
        surface_norm_squared_fraction=prior.dot(v[nv:],v[nv:]),
        transition_angular_outer_two_rows_fraction=float(w[1:7,edges].sum()),
        peak_block_k_j_i=[int(i) for i in np.unravel_index(np.argmax(w),w.shape)])
    probes=[]
    try:
        v7=candidate.Probe(args.library,16,4,start+350);probes.append(v7)
        v6=candidate.Probe(args.library,16,2,start+350);probes.append(v6)
        def action(probe,x,amplitude):
            scale=amplitude/float(np.max(np.abs(x)))
            return (probe.evaluate(scale*x)-probe.evaluate(-scale*x))/(2*scale)
        rows=[]
        for amplitude in (1e-5,1e-6):
            jv=action(v7,v,amplitude);q=jv-prior.dot(v,jv)*v;q/=prior.norm(q)
            jq=action(v7,q,amplitude);av=action(v6,v,amplitude);aq=action(v6,q,amplitude)
            row=dict(amplitude=amplitude)
            for name,a,b in [('V7',jv,jq),('V6_on_same_plane',av,aq)]:
                matrix=np.array([[prior.dot(v,a),prior.dot(v,b)],[prior.dot(q,a),prior.dot(q,b)]])
                row[name]=dict(projected_matrix=matrix.tolist(),
                    column_relative_residuals=[prior.norm(z-v*matrix[0,i]-q*matrix[1,i])/prior.norm(z) for i,z in enumerate((a,b))],
                    projected_eigenvalues=[dict(real=float(z.real),imag=float(z.imag)) for z in np.linalg.eigvals(matrix)],
                    direction_metrics=candidate.extra_metrics(v,a,16))
            row['relative_action_difference']=prior.norm(av-jv)/prior.norm(jv);rows.append(row)
        prior.write(args.output/'localization.json',dict(status='completed',
            scope='Observed V7 endpoint localization and tangent RHS at Minkowski; no checkpoint evolution and no verified eigenvalue or stability proof',
            norm='Unweighted point-coordinate Euclidean, not physical energy',checkpoint=meta,localization=localization,rows=rows,
            library_sha256=prior.sha(args.library),source_manifest_sha256=prior.sha(args.library.parent/'source.json'),
            frozen_V7_commit=FROZEN,exact_frozen_source_blobs_verified=len(manifest['sources']),script_sha256=prior.sha(__file__),
            imported_script_sha256={str(Path(m.__file__).relative_to(ROOT)):prior.sha(m.__file__) for m in (prior,candidate)},
            calls=dict(V7=v7.calls,V6=v6.calls),seconds=time.monotonic()-start))
    finally:
        for probe in probes:probe.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');run(p.parse_args())
