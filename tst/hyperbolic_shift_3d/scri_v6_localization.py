#!/usr/bin/env python3
"""Locate the observed V6 disturbance and test its two-dimensional RHS span.

Read-only diagnostic, not an eigenvalue proof or a modified evolution.
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
import scri_v6_direction as direction
import scri_candidate_replay as candidate
import scri_directional_replay as prior


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if not args.worker:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--library',str(args.library.resolve()),
                 '--checkpoint',str(args.checkpoint.resolve()),'--output',str(args.output/'worker')]
        start=time.monotonic()
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=385)
            result=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            result=dict(command=command,status='bounded',seconds=time.monotonic()-start)
        prior.write(args.output/'execution.json',result)
        if result.get('returncode',1):raise RuntimeError('Diagnostic failed: see execution.json')
        return
    start=time.monotonic();manifest=json.loads((args.library.parent/'source.json').read_text())
    if prior.sha(args.library)!=manifest['library_sha256']:raise ValueError('Library binding differs')
    for name,digest in manifest['sources'].items():
        if prior.sha(args.library.parent/'source-tree'/name)!=digest:raise ValueError('Source snapshot differs')
    v,meta=direction.checkpoint(args.checkpoint.resolve());v/=prior.norm(v)
    # Native active order is block,k,j,i,field; block 0 is Cartesian, then
    # six faces in each of three radial layers (CompositeMesh/CompositeDifference).
    n=16;nv=26*19*n**3;volume=v[:nv].reshape(19,n,n,n,26);w=np.sum(volume**2,axis=-1)
    regions={}
    for lo,hi,name in [(0,1,'Cartesian'),(1,7,'transition_shell'),(7,13,'middle_shell'),(13,19,'outer_shell')]:
        a=w[lo:hi];regions[name]=dict(norm_squared_fraction=float(a.sum()),i_row_fractions=a.sum(axis=(0,1,2)).tolist())
    transition=w[1:7]
    angular_edge=np.zeros((n,n,n),dtype=bool);angular_edge[:2,:,:]=True;angular_edge[-2:,:,:]=True
    angular_edge[:,:2,:]=True;angular_edge[:,-2:,:]=True
    localization=dict(regions=regions,block_norm_squared_fractions=w.sum(axis=(1,2,3)).tolist(),
        surface_norm_squared_fraction=prior.dot(v[nv:],v[nv:]),
        transition_angular_outer_two_rows_fraction=float(transition[:,angular_edge].sum()),
        peak_block_k_j_i=[int(i) for i in np.unravel_index(np.argmax(w),w.shape)])
    probe=candidate.Probe(args.library,16,2,start+350)
    try:
        def action(x,amplitude):
            scale=amplitude/float(np.max(np.abs(x)))
            return (probe.evaluate(scale*x)-probe.evaluate(-scale*x))/(2*scale)
        rows=[]
        for amplitude in (1e-5,1e-6):
            jv=action(v,amplitude);q=jv-prior.dot(v,jv)*v;q/=prior.norm(q)
            jq=action(q,amplitude)
            matrix=np.array([[prior.dot(v,jv),prior.dot(v,jq)],[prior.dot(q,jv),prior.dot(q,jq)]])
            residuals=[prior.norm(a-v*matrix[0,i]-q*matrix[1,i])/prior.norm(a) for i,a in enumerate((jv,jq))]
            eigenvalues=np.linalg.eigvals(matrix)
            rows.append(dict(amplitude=amplitude,projected_matrix=matrix.tolist(),
                column_relative_residuals=residuals,
                projected_eigenvalues=[dict(real=float(z.real),imag=float(z.imag)) for z in eigenvalues]))
        result=dict(status='completed',scope='Localization and two-vector projected tangent RHS; projected eigenvalues require small residuals and are not full-spectrum stability results',
            norm='Unweighted point-coordinate Euclidean, not physical energy',checkpoint=meta,localization=localization,rows=rows,
            library_sha256=prior.sha(args.library),source_manifest_sha256=prior.sha(args.library.parent/'source.json'),script_sha256=prior.sha(__file__),
            imported_script_sha256={str(Path(m.__file__).relative_to(direction.ROOT)):prior.sha(m.__file__) for m in (direction,prior,candidate)},
            ordering_source_sha256={name:prior.sha(direction.ROOT/name) for name in ('src/mesh/cubed_sphere/composite.hpp','src/mesh/cubed_sphere/composite_difference.hpp')},
            calls=probe.calls,seconds=time.monotonic()-start)
        prior.write(args.output/'localization.json',result)
    finally:probe.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true');run(parser.parse_args())
