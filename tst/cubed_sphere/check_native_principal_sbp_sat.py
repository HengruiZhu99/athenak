#!/usr/bin/env python3
"""Energy control for frozen native principal symbols; no native evolution claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
np.seterr(all="raise")
def mm(x,y):return np.einsum("ij,jk->ik",x,y)

p = argparse.ArgumentParser()
p.add_argument('--symbols', nargs='+', type=Path, required=True)
p.add_argument('--sbp', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sbp = json.loads(a.sbp.read_text())
weights = np.array(sbp['norm_boundary_weights'])
rows = np.array(sbp['left_boundary_rows'])
trace = np.array(sbp['left_trace'])

def symbols(path):
    lines = iter(path.read_text().splitlines())
    for line in lines:
        if not line.startswith('case '):
            continue
        _, family, radius, label, omega, lo, hi = line.split()
        inverse = np.array(list(map(float,next(lines).split()[1:]))).reshape(3,3)
        normal = next(lines)
        assert normal.startswith('normal ')
        assert next(lines) == 'constraints'
        constraints = np.array([list(map(float,next(lines).split())) for _ in range(8)])
        assert next(lines) == 'matrix 1'
        matrix = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert next(lines) == 'matrix 0.5'
        other = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert np.linalg.norm(matrix-other)/max(1,np.linalg.norm(matrix)) < 1e-11
        tr = np.array([inverse[0,0],2*inverse[0,1],2*inverse[0,2],inverse[1,1],2*inverse[1,2],inverse[2,2]])
        cq = np.zeros((1,11));cq[0,1:7] = tr
        cp = np.zeros((1,11));cp[0,:6] = tr
        q = np.linalg.svd(cq,full_matrices=True)[2][1:].T
        v = np.linalg.svd(cp,full_matrices=True)[2][1:].T
        basis = np.zeros((22,20));basis[:11,:10] = q;basis[11:,10:] = v
        yield dict(source=str(path),family=int(family),radius=float(radius),sample=label,omega=float(omega),
                   lo=float(lo),hi=float(hi),matrix=mm(mm(basis.T,matrix),basis),constraints=mm(constraints,basis))

def spatial(n):
    m = len(weights);h = np.ones(n);h[:m] = weights;h[-m:] = weights[::-1]
    d = np.zeros((n,n));d[:m,:m+2] = rows;d[-m:,-m-2:] = -rows[::-1,::-1]
    for k in range(m,n-m):
        for off,c in [(-2,1/12),(-1,-2/3),(1,2/3),(2,-1/12)]:d[k,k+off] = c
    left = np.zeros(n);left[:m] = trace;right = left[::-1]
    bl,br = np.outer(left,left),np.outer(right,right)
    residual = np.linalg.norm(h[:,None]*d+d.T*h[None,:]-(br-bl))
    assert min(h)>0 and residual<1e-10 and abs(sum(h)-n)<1e-10
    return h,d,bl,br,residual

records = []
for path in a.symbols:
    for s in symbols(path):
        M = s.pop('matrix');C = s.pop('constraints');S = -M;eye = np.eye(20)
        lo,hi = s['lo'],s['hi'];minus = (hi*eye-S)/(hi-lo);plus = (S-lo*eye)/(hi-lo)
        W = mm(minus.T,minus)+mm(plus.T,plus)
        Sm,Sp = lo*minus,hi*plus
        flux = mm(W,S);absflux = mm(W,Sp-Sm)
        symerr = np.linalg.norm(flux-flux.T)/max(1,np.linalg.norm(flux))
        assert lo<0<hi and np.linalg.eigvalsh(W).min()>0 and symerr<1e-10
        beta = -(lo+hi)/2;c = (hi-lo)/2
        Aq,B = M[:10,:10],M[:10,10:]
        T = np.block([[np.eye(10),np.zeros((10,10))],[Aq-beta*np.eye(10),B]])
        canonical = np.linalg.solve(T.T,mm(T,M).T).T
        target = np.block([[beta*np.eye(10),np.eye(10)],[c*c*np.eye(10),beta*np.eye(10)]])
        canonical_error = np.linalg.norm(canonical-target)/max(1,np.linalg.norm(target))
        C /= np.linalg.norm(C,axis=1)[:,None]
        rank = int(np.linalg.matrix_rank(mm(C,minus),tol=1e-9))
        tests = []
        for n in [24,48]:
            h,d,bl,br,sbperr = spatial(n)
            # This is Htotal*L. Forming it directly avoids an ill-conditioned
            # inverse energy norm while testing the full 20-mode identity.
            HL = -np.kron(h[:,None]*d,flux)-np.kron(bl,mm(W,Sp))+np.kron(br,mm(W,Sm))
            E = HL+HL.T
            expected = -np.kron(bl+br,absflux)
            scale = max(1,np.linalg.norm(expected))
            error = np.linalg.norm(E-expected)/scale
            largest = np.linalg.eigvalsh(E).max()/scale
            unclosed = HL-np.kron(br,mm(W,Sm))
            positive = np.linalg.eigvalsh(unclosed+unclosed.T).max()/scale
            tests.append(dict(nodes=n,sbp_error=sbperr,relative_energy_identity_error=float(error),
                              normalized_max_energy_eigenvalue=float(largest),
                              unclosed_normalized_max_energy_eigenvalue=float(positive),
                              passed=bool(error<=1e-10 and largest<=1e-10 and positive>1e-12)))
        s.update(symmetrizer_min_eigenvalue=float(np.linalg.eigvalsh(W).min()),
                 symmetrizer_condition=float(np.linalg.cond(W)),flux_symmetry_error=float(symerr),
                 canonical_B_condition=float(np.linalg.cond(B)),canonical_transform_condition=float(np.linalg.cond(T)),
                 canonical_relative_error=float(canonical_error),incoming_constraint_projection_rank=rank,
                 tests=tests,passed=bool(canonical_error<=1e-8 and rank==4 and all(x['passed'] for x in tests)))
        records.append(s)
out = dict(passed=all(x['passed'] for x in records),records=records,
           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           input_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in a.symbols+[a.sbp]},
           scope='Frozen implemented20-mode principal symbols, linear homogeneous incoming SAT only. '
                 'Not native second-order derivative/KO closure, nonlinear constraint boundary data, '
                 'variable-coefficient/patch stability or physical evolution validation.')
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(dict(passed=out['passed'],states=len(records),canonical_errors=[x['canonical_relative_error'] for x in records]))
