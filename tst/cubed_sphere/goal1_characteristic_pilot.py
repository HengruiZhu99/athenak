#!/usr/bin/env python3
"""Fill oblique/native-state and approach-to-scri symbol coverage gaps.

Imports the existing 20-mode algebraic/constraint/gauge/TT reviewer as a
subprocess; no native evolution is repeated. Actual stored states are tagged
separately from synthetic null-compatible states. Exact-scri PDE remains open.
"""
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
protocol=dict(version=1,scope='Full actual frozen native 20-mode principal symbol; no lower-source/endpoint stability claim',
    extraction_steps=[1.,.5],algebraic_leak_budget=1e-11,constraint_gauge_leak_budget=1e-10,
    polynomial_budget=1e-12,spectrum='physical/gauge/constraint sectors and positive symmetrizer',
    raw_signed_speeds=True,exact_scri_endpoint_admitted=False,
    normal_sign_tolerance='64*binary64 epsilon*max(1,abs(beta.normal),light_speed), diagnostic coefficient-roundoff allowance only')
pr=a.output/'characteristic-pilot-protocol-v1.json'
if pr.exists():assert json.loads(pr.read_text())==protocol
else:pr.write_text(json.dumps(protocol,indent=2)+'\n')
base=Path('docs/validation/cubed-sphere-balanced-20261013/finite-radius-long/pulse-t02-controls')
source=base/'pulse-boundary-speeds-t02.logical-samples'
directions=np.array([[1,0,0],[0,1,0],[0,0,1],[.8,-.48,.36],[1,1,1],[-1,2,-3]],dtype=float)
directions/=np.linalg.norm(directions,axis=1)[:,None]
requests=[];metadata=[]
for index,line in enumerate(source.read_text().splitlines()):
    if not line.strip():continue
    words=line.split();assert len(words)==35
    normal=np.array(list(map(float,words[-3:])))
    for d in directions:
        requests.append(' '.join(words[:32])+ ' '+' '.join(repr(float(v)) for v in d))
        metadata.append(dict(kind='evolved_native_t0.2',source=str(source),source_row=index,
            covector=d.tolist(),boundary_normal=False))
    # Boundary-normal spectra already exist and are reused, not re-extracted.
for curvature in [.5,1.]:
    for radius in [.99,.999,.9999,.99999]:
        n=np.array([.8,-.48,.36]);xyz=radius*n;omega=(1-radius*radius)/(2*curvature)
        alpha=(1+radius*radius)/(2*curvature);beta=-xyz/curvature
        for variant in [0,1]:
            values=[1.,1.,0.,0.,1.,0.,1.,-3/curvature,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                alpha*(1+variant*.01*omega**2),*(beta+variant*.01*omega**2*n),0.,0.,0.]
            assert len(values)==25
            for d in [n,*directions]:
                requests.append(' '.join(repr(float(v)) if j else '0' for j,v in enumerate(
                    [0,curvature,.05 if curvature==.5 else .35,
                    .95 if curvature==.5 else .75,*xyz,*values,*d])))
                metadata.append(dict(kind='synthetic_reference' if not variant else 'synthetic_Omega2_gauge',
                    radius=radius,curvature=curvature,covector=d.tolist(),boundary_normal=bool(np.all(d==n))))
inputs=a.output/'oblique-symbol.samples';inputs.write_text('\n'.join(requests)+'\n')
raw=a.output/'oblique-symbol.txt'
with raw.open('w') as f:
    run=subprocess.run([str(a.binary.resolve()),str(inputs),'--logical-normal'],stdout=f,stderr=subprocess.PIPE,text=True)
(a.output/'oblique-extraction.stderr').write_text(run.stderr);assert run.returncode==0
review=a.output/'oblique-review.json'
command=[sys.executable,'tst/cubed_sphere/review_radial_principal_symbol.py',str(raw),str(review),'--evolved']
run=subprocess.run(command,capture_output=True,text=True)
(a.output/'oblique-review.log').write_text(run.stdout+run.stderr)
if run.returncode:
    (a.output/'characteristic-pilot-failed.json').write_text(json.dumps(dict(command=command,exit_code=run.returncode,metadata=metadata),indent=2)+'\n')
    print(run.stderr[-2000:]);sys.exit(run.returncode)
r=json.loads(review.read_text());assert len(r['records'])==len(metadata)
for meta,record in zip(metadata,r['records']):record['pilot_case']=meta
normals=[]
for row in r['records']:
    if row['pilot_case']['boundary_normal']:
        scale=max(1.,*map(abs,row['physical_speeds']));tol=64*np.finfo(float).eps*scale
        normals.append(dict(case=row['pilot_case'],speeds=row['physical_speeds'],roundoff_allowance=tol,
            incoming_resolved=bool(row['physical_speeds'][0]<-tol)))
summary=dict(passed=True,samples=len(metadata),evolved_oblique=sum(m['kind']=='evolved_native_t0.2' for m in metadata),
    synthetic_samples=sum(m['kind']!='evolved_native_t0.2' for m in metadata),
    max_algebraic_leak=max(x['algebraic_subspace_leak'] for x in r['records']),
    max_constraint_leak=max(x['constraint_kernel_leak'] for x in r['records']),
    max_gauge_leak=max(x['gauge_kernel_leak'] for x in r['records']),
    max_polynomial_defect=max(x['two_branch_polynomial_relative_residual'] for x in r['records']),
    max_symmetrizer_condition=max(x['symmetrizer_condition'] for x in r['records']),
    near_scri_normals=normals,exact_scri_endpoint_admitted=False,gauge_change_required_by_this_test=False,
    reviewer_source_sha256=hashlib.sha256(Path(command[1]).read_bytes()).hexdigest(),
    input_source_sha256={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
    binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest())
(a.output/'oblique-review-classified.json').write_text(json.dumps(r,indent=2)+'\n')
(a.output/'characteristic-pilot-review.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='near_scri_normals'},indent=2))
