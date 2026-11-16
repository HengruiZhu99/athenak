#!/usr/bin/env python3
"""Reconstruct and independently check the regular-singular boundary pencil."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def position(f):
    return f <= 6 or f == 18 or 22 <= f <= 24


def sample(binary, root, name, radii, kappa=.25, h=.005, eps=1e-5, bg='flat'):
    prefix = root/name
    command = [str(binary), ','.join(format(r, '.17g') for r in radii),
               str(kappa), str(eps), bg, str(h), '.5', str(prefix)]
    subprocess.run(command, check=True)
    meta = json.loads(Path(str(prefix)+'.json').read_text())
    data = np.fromfile(Path(str(prefix)+'.bin')).reshape(meta['shape'])
    if not np.isfinite(data).all():
        raise ValueError('nonfinite Jacobian')
    Path(str(prefix)+'.receipt.json').write_text(json.dumps(dict(
        command=command, executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(Path(str(prefix)+'.bin').read_bytes()).hexdigest()), indent=2))
    return meta, data


def indicial(jac, rho, fields, exponent):
    # q_f = rho^(s+w_f) v_f. Position fields, P,Theta have w=0;
    # A,Lambda,B,D have w=-1. d_r=-d_rho.
    weights = np.array([0 if position(f) or f in [13, 14] else -1 for f in fields])
    jet = np.zeros((260, 26))
    for i, w in enumerate(weights):
        jet[10*i, i] = rho**float(w)
        jet[10*i+1, i] = -(exponent+w)*rho**float(w-1)
        jet[10*i+4, i] = (exponent+w)*(exponent+w-1)*rho**float(w-2)
    return rho**(1-weights[:, None])*np.einsum('ij,jk->ik', jac[:26], jet)


def principal(jac, omega, fields):
    scales = np.array([omega if f in [13, 14] else 1 for f in fields])
    out = np.zeros((26, 26))
    for i, fi in enumerate(fields):
        for j, fj in enumerate(fields):
            kind = (1 if position(fj) else 0) if position(fi) else (4 if position(fj) else 1)
            out[i, j] = jac[i, 10*j+kind]*scales[j]/scales[i]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sympy-path', type=Path)
    args = parser.parse_args()
    if args.sympy_path:
        sys.path.insert(0, str(args.sympy_path))
    import sympy as sp
    root = args.output.resolve();root.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    rho = np.array([.001, .0005, .00025, .000125])
    meta, kernels = sample(binary, root, 'indicial025', 1-rho, eps=1e-3)
    _, halfkernels = sample(binary, root, 'indicial025-half', 1-rho, eps=5e-4)
    kernels = (4*halfkernels-kernels)/3
    fields = meta['fields'];s, k = sp.symbols('s k')
    fitted = [];rational_errors = []
    for exponent in [0, 1, 2]:
        values = np.array([indicial(jac, r, fields, exponent) for jac, r in zip(kernels, rho)])
        limit = np.polynomial.polynomial.polyfit(rho, values.reshape(4, -1), 3)[0].reshape(26, 26)
        rational = sp.Matrix(26, 26, lambda i, j: sp.Rational(Fraction(float(limit[i, j])).limit_denominator(120)))
        rational_errors.append(float(np.max(abs(limit-np.array(rational, float)))))
        fitted.append(rational)
    M = fitted[0]+s*(fitted[1]-fitted[0])+s*(s-1)*(fitted[2]-2*fitted[1]+fitted[0])/2
    lapse = fields.index(18)
    assert sp.simplify(M[lapse, lapse]-(s-sp.Rational(3, 2))) == 0
    M[lapse, lapse] = s-1-2*k
    scalar_fields = [0, 1, 7, 13, 14, 15, 18, 19, 22, 25]
    idx = [fields.index(f) for f in scalar_fields]
    embedding = sp.zeros(26, 10)
    for j, i in enumerate(idx):embedding[i, j] = 1
    embedding[fields.index(4), 1] = -sp.Rational(1, 2)
    embedding[fields.index(10), 2] = -sp.Rational(1, 2)
    scalar = (M*embedding)[idx, :]
    assert sp.simplify(M*embedding-embedding*scalar) == sp.zeros(26, 10)
    determinant = sp.factor(scalar.det())
    expected = sp.Rational(8, 15)*s*(s-3)*(s-1)**2*(31*s-300)*(k*s*s+5*k*s-4*k-8)
    assert sp.simplify(determinant-expected) == 0
    branch_k = 8/(s*s+5*s-4)
    vectors = scalar.subs(k, branch_k).nullspace();assert len(vectors) == 1
    vector = sp.simplify(vectors[0]/vectors[0][6])
    assert sp.simplify(scalar.subs(k, branch_k)*vector) == sp.zeros(10, 1)
    assert sp.simplify(vector[4]) == 0
    assert sp.simplify(vector[9]-vector[5]) == 0
    assert sp.simplify(vector[5]+s*vector[1]) == 0  # leading connection constraint
    # Independent radii, FD perturbations, noninteger s and damping values.
    checks = [];richardson_checks=[]
    for damping in [.25, .375, .5, .75, 1.]:
        extrapolated={}
        for eps in [1e-3, 5e-4]:
            rr = np.array([.0008, .0004, .0002, .0001])
            _, kk = sample(binary, root, f'check-k{damping}-e{eps}', 1-rr, damping, eps=eps)
            for exponent in [-.4, .7, 2.3, (-5+np.sqrt(41+32/damping))/2]:
                vals = np.array([indicial(j, r, fields, exponent) for j, r in zip(kk, rr)])
                extrap = np.polynomial.polynomial.polyfit(rr, vals.reshape(4, -1), 3)[0].reshape(26, 26)
                target = np.array(M.subs({s:exponent, k:damping}), float)
                extrapolated[eps,exponent]=extrap
                checks.append(dict(kappa=damping, fdstep=eps, exponent=float(exponent),
                                   max_limit_error=float(np.max(abs(extrap-target)))))
        for exponent in [-.4,.7,2.3,(-5+np.sqrt(41+32/damping))/2]:
            extrap=(4*extrapolated[5e-4,exponent]-extrapolated[1e-3,exponent])/3
            target=np.array(M.subs({s:exponent,k:damping}),float)
            error=float(np.max(abs(extrap-target)))
            assert error<1e-6,error
            richardson_checks.append(dict(kappa=damping,exponent=float(exponent),max_limit_error=error))
    wormhole_checks=[]
    for damping in [.25,.5,1.]:
        rr=np.array([.0008,.0004,.0002,.0001])
        _,ka=sample(binary,root,f'wormhole-k{damping}',1-rr,damping,eps=1e-3,bg='wormhole')
        _,kb=sample(binary,root,f'wormhole-half-k{damping}',1-rr,damping,eps=5e-4,bg='wormhole')
        exponent=(-5+np.sqrt(41+32/damping))/2
        vals=np.array([indicial(j,r,fields,exponent) for j,r in zip((4*kb-ka)/3,rr)])
        extrap=np.polynomial.polynomial.polyfit(rr,vals.reshape(4,-1),3)[0].reshape(26,26)
        error=float(np.max(abs(extrap-np.array(M.subs({s:exponent,k:damping}),float))))
        assert error<1e-5,error
        wormhole_checks.append(dict(kappa=damping,exponent=float(exponent),max_limit_error=error))
    # Check all 26 normal principal fields, then reconstruct exact rational limit.
    pp = np.array([principal(j, z['omega'], fields) for j,z in zip(kernels, meta['points'])])
    plim = np.polynomial.polynomial.polyfit(rho, pp.reshape(4, -1), 3)[0].reshape(26, 26)
    P = sp.Matrix(26, 26, lambda i,j:sp.Rational(Fraction(float(plim[i,j])).limit_denominator(120)))
    eigenspaces=[]
    for eigenvalue, multiplicity in P.eigenvals().items():
        nullity=26-(P-eigenvalue*sp.eye(26)).rank()
        eigenspaces.append(dict(eigenvalue=str(eigenvalue), multiplicity=multiplicity, nullity=nullity))
        assert nullity==multiplicity
    # Principal time-differentiated shift-value conditions on incoming modes.
    B = P[[fields.index(f) for f in [22,23,24]], :]
    B[0,:] += (P[fields.index(0),:]-P[fields.index(1),:])/2
    incoming=[]
    for eigenvalue in P.eigenvals():
        if float(eigenvalue)>1e-8:incoming.extend((P-eigenvalue*sp.eye(26)).nullspace())
    R = sp.Matrix.hstack(*incoming)
    boundary_rank=(B*R).rank();assert R.cols==3 and boundary_rank==3
    result=dict(caveat='Rational leading matrices reconstructed numerically, then checked at independent radii/FD steps. This is not an all-frequency Kreiss or energy estimate.',
                fields=fields, weights=[0 if position(f) or f in [13,14] else -1 for f in fields],
                reconstruction_errors=rational_errors, scalar_fields=scalar_fields,
                scalar_matrix=str(scalar), scalar_determinant=str(determinant),
                mode_vector_alpha_normalized=[str(v) for v in vector],
                mode_exponent='(-5 + sqrt(41 + 32/kappa))/2', checks=checks,
                max_independent_limit_error=max(c['max_limit_error'] for c in checks),
                richardson_checks=richardson_checks,wormhole_checks=wormhole_checks,
                max_richardson_limit_error=max(c['max_limit_error'] for c in richardson_checks),
                principal_rational_error=float(np.max(abs(plim-np.array(P,float)))),
                principal_eigenspaces=eigenspaces, incoming_count=R.cols,
                normal_shift_boundary_rank=boundary_rank,
                principal_matrix=str(P), boundary_incoming_matrix=str(B*R))
    (root/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({q:result[q] for q in ['scalar_determinant','max_independent_limit_error','principal_eigenspaces','normal_shift_boundary_rank']},indent=2))


if __name__=='__main__':main()
