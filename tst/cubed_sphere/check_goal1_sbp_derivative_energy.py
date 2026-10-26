#!/usr/bin/env python3
"""Tensor-sector derivative-energy audit of the exported native radial KO."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();out=a.evidence
assert (out/'derivative-dissipation-protocol-v1.json').exists()
sbp=Path('docs/validation/cubed-sphere-balanced-20261013/outer-closure/finite-omega-sbp-exploration.json')
s=json.loads(sbp.read_text());rows=np.array(s['left_boundary_rows']);records=[]
for line in (out/'operator-native.jsonl').read_text().splitlines():
    if not line.startswith('{'):continue
    case=json.loads(line)
    if case['type']!='KO':continue
    n=case['nodes'];h=np.array(case['norm_volume']);L=np.array(case['matrix']);D=np.zeros((n,n));D[:8,:11]=rows*n;D[-8:,-11:]=-rows[::-1,::-1]*n
    for i in range(8,n-8):
        for j,c in [(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)]:D[i,i+j]=c*n
    for i in range(n):D[i,i]-=np.sum(D[i])
    H=np.diag(h);G=H+np.einsum('ji,j,jk->ik',D,h,D)
    EL2=h[:,None]*L+L.T*h[None,:]
    E=np.einsum('ij,jk->ik',G,L)+np.einsum('ij,jk->ik',L.T,G)
    R=np.linalg.cholesky(G);left=np.linalg.solve(R,E);whitened=np.linalg.solve(R,left.T).T
    eig,V=np.linalg.eigh((whitened+whitened.T)/2);direction=np.linalg.solve(R.T,V[:,-1]);energy=float(np.einsum('i,ij,j->',direction,G,direction));rate=float(np.einsum('i,ij,j->',direction,E,direction))
    gradient=np.einsum('ij,j->i',D,direction);velocity=np.einsum('ij,j->i',L,direction);gradient_velocity=np.einsum('ij,j->i',D,velocity)
    direct_energy=float(np.sum(h*(direction*direction+gradient*gradient)))
    direct_rate=float(2*np.sum(h*(direction*velocity+gradient*gradient_velocity)))
    crosscheck=max(abs(direct_energy-energy),abs(direct_rate-rate))/max(1.,abs(energy),abs(rate))
    scale=max(1.,np.linalg.norm(E));normalized=float(np.linalg.eigvalsh((E+E.T)/2)[-1]/scale)
    records.append({'nodes':n,'stretch':case['stretch'],'component_L2_max_normalized_energy_eigenvalue':float(np.linalg.eigvalsh(EL2)[-1]/max(1.,np.linalg.norm(EL2))),
        'derivative_energy_logarithmic_rate':float(eig[-1]),'witness_energy':energy,'witness_energy_rate':rate,
        'direct_witness_energy':direct_energy,'direct_witness_energy_rate':direct_rate,'direct_energy_crosscheck_relative':crosscheck,
        'witness_q':direction.tolist(),'derivative_energy_max_normalized_eigenvalue':normalized,
        'passed':bool(normalized<=1e-10)})
report={'scope':__doc__,'all_declared_derivative_energy_gates_pass':all(r['passed'] for r in records),'records':records,
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'native_operator_payload_sha256':hashlib.sha256((out/'operator-native.jsonl').read_bytes()).hexdigest(),
    'interpretation':'A positive dissipative-stage derivative-energy rate refutes contraction in this declared norm; it does not establish a growing eigenmode or instability of the complete native solver.'}
(out/'derivative-dissipation-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['records','scope']},indent=2));print([(r['nodes'],r['stretch'],r['derivative_energy_logarithmic_rate'],r['passed']) for r in records])
