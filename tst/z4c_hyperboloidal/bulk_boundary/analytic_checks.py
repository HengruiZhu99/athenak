#!/usr/bin/env python3
"""Exact checks independent of finite-difference rational reconstruction.

The six equations come from the two metric evolution equations, Hamiltonian
and radial momentum constraints, and the lapse/shift gauge equations. Their
leading coefficients are derived explicitly in README.md.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--sympy-path',type=Path)
    args=p.parse_args()
    if args.sympy_path:sys.path.insert(0,str(args.sympy_path))
    import sympy as S
    s,k,R=S.symbols('s k R');C,G,A,P,a,b=S.symbols('C G A P a b')
    equations=[
        s*C+S.Rational(2,3)*P-2*a+(S.Rational(2,3)*s-2)*b,
        s*G-2*A-S.Rational(4,3)*s*b,
        (3-s)*A+S.Rational(2,3)*s*P,
        (s-2)*(s-3)*G+2*(s-3)*(s+1)*C-4*P,
        (s-R)*b-S.Rational(3,2)*s*G,
        -P+(s-1-2*k)*a+b]
    matrix=S.linear_eq_to_matrix(equations,[C,G,A,P,a,b])[0]
    factor=k*s*s+k*(R+1)*s-R*(k+2)
    determinant=S.factor(matrix.det())
    assert S.simplify(determinant+8*s*(s-3)*factor)==0
    integer_four=S.solve(factor.subs(s,4),R)[0]
    assert S.simplify(integer_four-20*k/(2-3*k))==0
    d=s*s+5*s-4
    mode=S.Matrix([2*(s*s-3*s+20)/(3*d),4*(-s*s+3*s+4)/(3*d),
                   2*s*(s*s+5*s+4)/(3*d),(s**3+2*s*s-11*s-12)/d,
                   1,2*s*(-s-1)/d])
    assert S.simplify(matrix.subs({R:4,k:8/d})*mode)==S.zeros(6,1)
    # Principal incoming spatial-gauge system: D=dt+dx and mu=3/2.
    # Energy after expanding D^2 has positive spatial quadratic form since mu>1.
    # The boundary conditions are xi_normal,t=0 and xi_tangent,x=-xi_tangent,t.
    mu=S.symbols('mu',positive=True)
    vx,vy,vz,xx,xy,xz,div=S.symbols('vx vy vz xx xy xz div',real=True)
    flux=vx*vx+vy*vy+vz*vz-(mu-1)*(vx*xx+vy*xy+vz*xz)-mu*vx*div/3
    boundary_flux=S.factor(flux.subs({vx:0,xy:-vy,xz:-vz}))
    assert S.simplify(boundary_flux-mu*(vy*vy+vz*vz))==0
    result=dict(six_equations=[str(e) for e in equations],variables=['C','G','A','P','a','b'],
                determinant=str(determinant),general_indicial_factor=str(factor),
                restoring_for_exponent_four=str(integer_four),
                normalized_mode=[str(v) for v in mode],
                principal_shift_energy_flux=str(boundary_flux),
                scope='Exact leading constraint-compatible radial system and homogeneous incoming spatial-gauge principal boundary energy. Not a full nonlinear or regular-singular energy estimate.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
