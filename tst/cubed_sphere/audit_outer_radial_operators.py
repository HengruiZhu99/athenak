#!/usr/bin/env python3
"""Exact ghost elimination for NGHOST=3, at angular interior nodes.

Coordinates are logical radial indices, last active node zero. Not a proof
for patch-corner interpolation, floating point evaluation, or SBP stability.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

D1={-2:F(1,12),-1:F(-2,3),1:F(2,3),2:F(-1,12)}
D2={-2:F(-1,12),-1:F(4,3),0:F(-5,2),1:F(4,3),2:F(-1,12)}
DL={-3:F(-1,12),-2:F(1,2),-1:F(-3,2),0:F(5,6),1:F(1,4)}
DR={-k:-v for k,v in DL.items()}

def extrapolate(x, degree):
    weights={}
    for a in range(-degree,1):
        w=F(1)
        for b in range(-degree,1):
            if b!=a: w*=F(x-b,a-b)
        weights[a]=w
    return weights

def eliminate(stencil,depth,degree):
    out={}
    for delta,c in stencil.items():
        x=-depth+delta
        for j,w in (extrapolate(x,degree) if x>0 else {x:F(1)}).items():
            out[j]=out.get(j,F(0))+c*w
    return {j:v for j,v in sorted(out.items()) if v}

def moment(weights,depth,k):
    return sum(v*F(j+depth)**k for j,v in weights.items())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args(); rows=[]
    for degree in range(2,6):
        for depth in range(3):
            for name,stencil,derivative in [('D1',D1,1),('D2',D2,2),('advection_beta_negative',DL,1),('advection_beta_positive',DR,1)]:
                w=eliminate(stencil,depth,degree)
                exact=0
                for k in range(12):
                    expected=F(1 if derivative==1 else 2) if k==derivative else F(0)
                    if moment(w,depth,k)!=expected:break
                    exact=k
                # A direct interpolating derivative on the same support is
                # equivalent iff all moments through number-of-nodes-1 match.
                rows.append(dict(degree=degree,depth=depth,operator=name,
                    weights={str(j):str(v) for j,v in w.items()},
                    polynomial_exact_degree=exact,formal_order=exact+1-derivative,
                    equivalent_to_interpolating_derivative=exact>=len(w)-1,
                    coefficient_l1=float(sum(abs(v) for v in w.values()))))
                assert all(j<=0 for j in w)
                assert exact>=min(degree,4)
    sources=['src/utils/finite_diff.hpp','src/mesh/cubed_sphere/conformal_probe.cpp',
             'src/z4c/hyperboloidal/athenak_bridge.hpp','src/mesh/cubed_sphere/derivatives.hpp']
    data=dict(scope='Exact logical radial elimination at angular interior nodes; h^-d omitted',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},rows=rows,
        mixed='D_r D_A is tensor product of eliminated radial D1 and centered angular D1 only where radial ghost interpolation is angular nodal identity. Mapped Hessian also contains map Hessian times D1.',
        dissipation='Probe radial KO already truncates third-difference lines to active points; algebraically negative D3 transpose D3 in the unweighted constant-coefficient radial problem. No mapped multidimensional energy proof implied.',
        limitations=['No SBP identity established','Patch seams/corners require direct three-dimensional plan audit','Analytic background derivatives are added separately','Floating-point anchored ghost evaluation is not bitwise equivalent to collapsed weights'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    for r in rows:
        if r['depth']==0: print(r['degree'],r['operator'],r['weights'],'order',r['formal_order'],'interpolating',r['equivalent_to_interpolating_derivative'])

if __name__=='__main__':main()
