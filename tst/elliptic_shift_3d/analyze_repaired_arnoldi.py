#!/usr/bin/env python3
"""Bounded fresh witness for the repaired operator's Arnoldi numerical range."""
import argparse,gzip,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();out=a.evidence.resolve()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((out/'manifest.json').read_text());binary=out/'probe'
assert sha(binary)==m['binary_sha256']
hpath=out/'repaired.H.json';data=json.loads(hpath.read_text());H=np.array(data['hessenberg']);S=H[:-1]
eigen,vectors=np.linalg.eigh((S+S.T)/2);singular=np.linalg.svd(H,compute_uv=False)
coeff=out/'negative.coefficients.dat';np.savetxt(coeff,vectors[:,0],fmt='%.17g')
cmd=[str(binary),'composed','witness',str(out/'repaired'),str(coeff)]
record={'scope':'Fresh full-action numerical-range witness; no coercivity or spectral-gap certification',
        'command':cmd,'max_seconds':60,'binary_sha256':sha(binary),'H_sha256':sha(hpath),
        'basis_sha256':sha(out/'repaired.Q.bin'),'coefficients_sha256':sha(coeff),
        'script_sha256':sha(Path(__file__)), 'restricted_smin':float(singular[-1]),
        'restricted_smax':float(singular[0]),'restricted_condition':float(singular[0]/singular[-1]),
        'projected_range_min':float(eigen[0]),'orthogonality_max':data['orthogonality_max']}
start=time.monotonic()
with (out/'witness.stdout').open('w') as so,(out/'witness.stderr').open('w') as se:
    run=subprocess.run(cmd,stdout=so,stderr=se,timeout=60,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
record.update(exit_code=run.returncode,seconds=time.monotonic()-start)
assert run.returncode==0
csv=out/'repaired.witness.csv';d=np.genfromtxt(csv,delimiter=',',names=True)
r=np.sqrt(d['x']**2+d['y']**2+d['z']**2);block=d['block'].astype(int);cart=block<8;layer=np.where(cart,-1,(block-8)//6)
ratio=r/np.maximum.reduce([abs(d['x']),abs(d['y']),abs(d['z'])]);radii=np.array([.45,.55,.60,.65,.70,.75,1.])
lower=np.where(layer==0,.25*ratio,radii[np.maximum(layer-1,0)]);upper=radii[np.maximum(layer,0)]
w=np.where(cart,(.25/6)**3,4/6**3*r*r*(upper-lower)/ratio**3);w/=sum(w)
v=np.stack([d['v'+str(i)] for i in range(3)],axis=1);z=np.stack([d['Pv'+str(i)] for i in range(3)],axis=1);az=np.stack([d['APv'+str(i)] for i in range(3)],axis=1)
def dot(a,b):return math.fsum(float(wi)*math.fsum(float(x)*float(y) for x,y in zip(ai,bi)) for wi,ai,bi in zip(w,a,b))
record.update(input_norm=math.sqrt(dot(v,v)),fresh_preconditioned_rayleigh=dot(v,az),physical_rayleigh=dot(z,az)/dot(z,z),physical_energy_normalized_volume=dot(z,az))
assert abs(record['fresh_preconditioned_rayleigh']-eigen[0])<1e-10
record['physical_negative_witness']=record['physical_rayleigh']<0
record['actual_action_norm']=math.sqrt(dot(az,az))
record['unrestarted_curves']=[]
for j in [16,32,48,64,80]:
    unit=np.zeros(j+1);unit[0]=1;y=np.linalg.lstsq(H[:j+1,:j],unit,rcond=None)[0]
    product=np.array([math.fsum(float(x)*float(z) for x,z in zip(row,y)) for row in H[:j+1,:j]])
    record['unrestarted_curves'].append({'columns':j,'residual':float(np.linalg.norm(unit-product))})
compressed=csv.with_suffix('.csv.gz');compressed.write_bytes(gzip.compress(csv.read_bytes(),mtime=0))
record['raw_csv_sha256']=sha(csv);record['compressed_csv_sha256']=sha(compressed)
assert sha(binary)==record['binary_sha256']
(out/'witness-assessment.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['command','unrestarted_curves']},indent=2))
