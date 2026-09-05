"""Static operator comparison of pinned VC transfers and current PC-GH CC transfers.

This is matrix/polynomial analysis, not a CUDA evolution or a full AMR interface
test. Periodic whole-grid prolongation isolates the actual one-dimensional
tensor factors; shared-node reconciliation, physical boundaries, finite-block
restriction switches, and RK feedback are outside its scope.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parents[2]
VC = '6bfa5c11c3b10775294f5f2d95a196d84d6718cb'
CC = 'be93dfd0'


def read(ref, path):
    return subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=ROOT,text=True)


def weights(vc, cc):
    midpoint = {}
    for q in [4,6,8]:
        block = vc.split(f'struct MidpointRule<{q}>')[1].split('\n};')[0]
        matches = re.findall(r'p == (\d+) \?\s*([+-]?\d+\.\d+) / (\d+\.\d+)',block)
        assert len(matches) == q
        midpoint[q] = [s.Rational(a)/s.Rational(b) for _,a,b in matches]
        for power in range(q):
            assert sum(w*s.Integer(i-q//2+1)**power for i,w in enumerate(midpoint[q])) == s.Rational(1,2)**power
    cell = {}
    for ng in [2,4]:
        entries = re.search(rf'const Real wght{ng}\[{ng+1}\] = \{{([^}}]+)\}}',cc)[1]
        cell[ng] = [s.Rational(x.strip()) for x in entries.split(',')]
        for power in range(ng+1):
            assert sum(w*s.Integer(i-ng//2)**power for i,w in enumerate(cell[ng])) == (-s.Rational(1,4))**power
    return midpoint,cell


def prolong(n, kind, order, midpoint, cell):
    p = np.zeros((2*n,n))
    if kind == 'vertex':
        w = np.asarray(midpoint[order+2],float)
        offsets = np.arange(len(w))-len(w)//2+1
        for j in range(n):
            p[2*j,j] = 1
            p[2*j+1,(j+offsets)%n] = w
    elif kind == 'cell_elevated_control':
        # An analytic control, not an implemented production transfer: retain
        # CC geometry but match the vertex branch's interpolation order.
        q=order+2
        offsets=np.arange(-q//2,q//2)
        exact=s.finite_diff_weights(0,list(map(int,offsets)),s.Rational(-1,4))[0][-1]
        for power in range(q):
            assert sum(w*s.Integer(i)**power for i,w in zip(offsets,exact)) == (-s.Rational(1,4))**power
            assert sum(w*s.Integer(-i)**power for i,w in zip(offsets,exact)) == s.Rational(1,4)**power
        w=np.asarray(exact,float)
        for j in range(n):
            p[2*j,(j+offsets)%n]=w
            p[2*j+1,(j-offsets[::-1])%n]=w[::-1]
    else:
        w = np.asarray(cell[2 if order==2 else 4],float)
        offsets = np.arange(len(w))-len(w)//2
        for j in range(n):
            p[2*j,(j+offsets)%n] = w
            p[2*j+1,(j+offsets)%n] = w[::-1]
    return p


def derivative(n, order, degree):
    offsets = np.arange(-order//2,order//2+1)
    weights = s.finite_diff_weights(degree,list(map(int,offsets)),0)[degree][-1]
    matrix = np.zeros((n,n))
    for j in range(n):
        matrix[j,(j+offsets)%n] = np.asarray(weights,float)*n**degree
    return matrix


def multiply(a,b):
    # Explicit contraction avoids the workstation BLAS matmul path, which
    # emitted spurious floating-point exceptions on these small finite arrays.
    return np.einsum('ij,jk->ik' if b.ndim==2 else 'ij,j->i',a,b,optimize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sources = {'vc-vertex_amr.hpp':read(VC,'src/mesh/vertex_amr.hpp'),
        'cc-mesh_refinement.cpp':read(CC,'src/mesh/mesh_refinement.cpp'),
        'cc-prolongation.hpp':read(CC,'src/mesh/prolongation.hpp'),
        'cc-finite_diff.hpp':read(CC,'src/utils/finite_diff.hpp')}
    assert read(VC,'src/utils/finite_diff.hpp') == sources['cc-finite_diff.hpp']
    for name,text in sources.items(): (args.output/name).write_text(text)
    midpoint,cell=weights(sources['vc-vertex_amr.hpp'],sources['cc-mesh_refinement.cpp'])
    rows=[]
    for order in [2,4,6]:
        for n in [16,32,64,128]:
            dc,df=derivative(n,order,1),derivative(2*n,order,1)
            d2f=derivative(2*n,order,2)
            for kind,phase in [('cell',.5),('vertex',0.),('cell_elevated_control',.5)]:
                p=prolong(n,kind,order,midpoint,cell)
                xc=(np.arange(n)+phase)/n
                xf=(np.arange(2*n)+phase)/(2*n)
                uc=np.sin(2*np.pi*xc);uf=np.sin(2*np.pi*xf)
                state=multiply(p,uc)-uf
                commutator=multiply(df,p)-multiply(p,dc)
                # One-sided consumption at a stationary x=.5 coarse/fine seam:
                # coarse interpolation supplies the fine left ghosts; the right
                # active values are exact. No physical boundary is involved.
                ghost_error=multiply(p,np.sin(2*np.pi*xc+.37))-np.sin(2*np.pi*xf+.37)
                ghost_error[xf>=.5]=0
                consumers=(xf>=.5)&(xf<=.5+order/(4*n))
                seam_first=multiply(df,ghost_error)
                seam_second=multiply(d2f,ghost_error)
                assert np.isfinite(commutator).all() and np.isfinite(seam_second).all()
                rows.append(dict(order=order,coarse_n=n,centering=kind,
                    interpolation_order=(order+2 if kind!='cell' else (3 if order==2 else 5)),
                    state_linf=float(abs(state).max()),
                    transfer_first_derivative_linf=float(abs(multiply(df,state)).max()),
                    transfer_second_derivative_linf=float(abs(multiply(d2f,state)).max()),
                    seam_first_derivative_linf=float(abs(seam_first[consumers]).max()),
                    seam_second_derivative_linf=float(abs(seam_second[consumers]).max()),
                    reduction_commutator_linf=float(abs(multiply(commutator,uc)).max())))
    norms=[]
    n=32
    for order in [2,4,6]:
        dc,df=derivative(n,order,1),derivative(2*n,order,1)
        for kind in ['cell','vertex','cell_elevated_control']:
            p=prolong(n,kind,order,midpoint,cell)
            norm=float(np.linalg.svd((multiply(df,p)-multiply(p,dc))/n,compute_uv=False)[0]/np.sqrt(2))
            item=dict(order=order,centering=kind,
                      H_times_commutator_norm_in_uniform_quadrature=norm)
            if kind=='vertex':
                assert np.array_equal(p[::2,:],np.eye(n))
                item['restriction_prolongation_identity']='bitwise exact'
            norms.append(item)
    # An exact counterexample, with the production q4 midpoint interpolation.
    # Cubic data are reproduced exactly, but centered FD2 has different errors
    # on spacings H and H/2. Hence RP=I does not imply derivative commutation.
    x,H=s.symbols('x H',real=True,nonzero=True)
    f=x**3
    dc_f=s.expand((f.subs(x,x+H)-f.subs(x,x-H))/(2*H))
    df_f=s.expand((f.subs(x,x+H/2)-f.subs(x,x-H/2))/H)
    counterexample=s.simplify(df_f-dc_f)
    assert counterexample == -3*H**2/4
    report=dict(created=datetime.now(timezone.utc).isoformat(),vc_commit=VC,
        cc_commit=subprocess.check_output(['git','rev-parse',CC],cwd=ROOT,text=True).strip(),
        source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
        scope=__doc__,rows=rows,operator_norms=norms,
        exact_vertex_FD2_cubic_commutator=str(counterexample),
        fd_source_identical_between_branches=True)
    assert all(np.isfinite(v) for row in rows for v in row.values() if isinstance(v,float))
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: source-extracted weights reproduce their claimed polynomial spaces')
    print('PASS: finite-difference source is identical between the pinned branches')
    print('PASS: VC restriction/prolongation is exactly identity')
    print('COUNTEREXAMPLE: vertex FD2, q4, g=x^3: (Df P-P Dc)g =',counterexample)
    for order in [2,4,6]:
        for kind in ['cell','vertex','cell_elevated_control']:
            series=[r for r in rows if r['order']==order and r['centering']==kind]
            a,b=series[-2:]
            keys=['transfer_first_derivative_linf','transfer_second_derivative_linf','reduction_commutator_linf']
            print(order,kind,'N128 errors',{k:b[k] for k in keys},
                  '64->128 orders',{k:float(np.log2(a[k]/b[k])) for k in keys})
            a,b=series[1:3]
            keys=['seam_first_derivative_linf','seam_second_derivative_linf']
            print(order,kind,'SEAM N64 errors',{k:b[k] for k in keys},
                  '32->64 orders',{k:float(np.log2(a[k]/b[k])) for k in keys})
    print('DIMENSIONLESS all-mode commutator norms:',norms)


if __name__=='__main__': main()
