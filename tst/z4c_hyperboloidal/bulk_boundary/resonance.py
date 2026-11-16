#!/usr/bin/env python3
"""Check the first radial Frobenius recurrence, including spherical jet terms."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from fractions import Fraction
from analyze import sample


def spherical_jets(r,s,fields):
    rho=1-r;fs=[0,1,7,13,14,15,18,19,22,25]
    J=np.zeros((260,10));ix=lambda f:fields.index(f)
    def put(f,c,kind,value):J[10*ix(f)+kind,c]+=value
    for c,f in enumerate(fs):
        power=s if f in [0,1,13,14,18,22] else s-1
        v=rho**power;d=-power*rho**(power-1);dd=power*(power-1)*rho**(power-2)
        if f in [0,13,14,18]:
            for kind,val in [(0,v),(1,d),(4,dd),(7,d/r),(9,d/r)]:put(f,c,kind,val)
        elif f in [1,7]:
            # rr=v, tt=-v/2; only independent Cartesian components are supplied.
            for field,mult in [(f,1),(f+3,-.5)]:
                for kind,val in [(0,v),(1,d),(4,dd)]:put(field,c,kind,mult*val)
            for field,kind,val in [(f+1,2,1.5*v/r),(f+2,3,1.5*v/r),
                                    (f,7,d/r-3*v/r**2),(f,9,d/r-3*v/r**2),
                                    (f+3,7,-.5*d/r+3*v/r**2),(f+3,9,-.5*d/r),
                                    (f+1,5,1.5*(d/r-v/r**2)),(f+2,6,1.5*(d/r-v/r**2)),
                                    (f+4,8,1.5*v/r**2)]:put(field,c,kind,val)
        else:
            for field,kind,val in [(f,0,v),(f,1,d),(f,4,dd),
                                    (f+1,2,v/r),(f+2,3,v/r),
                                    (f,7,d/r-v/r**2),(f,9,d/r-v/r**2),
                                    (f+1,5,d/r-v/r**2),(f+2,6,d/r-v/r**2)]:put(field,c,kind,val)
    return J


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('output',type=Path);p.add_argument('--sympy-path',type=Path);p.add_argument('--radius-scale',type=float,default=1);p.add_argument('--fd-step',type=float,default=.001);p.add_argument('--background',choices=['flat','wormhole'],default='flat');p.add_argument('--name',default='resonance');args=p.parse_args()
    if args.sympy_path:sys.path.insert(0,str(args.sympy_path))
    import sympy as S
    root=args.output;rho=args.radius_scale*np.array([.004,.002,.001,.0005])
    meta,ka=sample(args.binary.resolve(),root,args.name+'-recurrence',1-rho,eps=args.fd_step,bg=args.background)
    _,kb=sample(args.binary.resolve(),root,args.name+'-recurrence-half',1-rho,eps=args.fd_step/2,bg=args.background)
    fields=meta['fields'];ids=[fields.index(f) for f in [0,1,7,13,14,15,18,19,22,25]]
    weights=np.array([0,0,-1,0,0,-1,0,-1,0,-1])
    matrices=[]
    for rr,J in zip(rho,(4*kb-ka)/3):
        B=np.einsum('ij,jk->ik',J[:26],spherical_jets(1-rr,3,fields))[ids]
        matrices.append(rr**(-2-weights[:,None])*B)
    fit=np.polynomial.polynomial.polyfit(rho,np.array(matrices).reshape(4,-1),3)
    m1=fit[1].reshape(10,10)
    exact=S.Matrix(10,10,lambda i,j:S.Rational(Fraction(float(m1[i,j])).limit_denominator(1200)))
    s,k=S.symbols('s k');analysis=json.loads((root/'analysis.json').read_text())
    M=S.sympify(analysis['scalar_matrix'],locals={'s':s,'k':k}).subs(k,S.Rational(1,4))
    mass=M.subs(s,3).nullspace()[0];mass=S.simplify(mass/mass[1])
    gauge=M.subs(s,4).nullspace()[0];gauge=S.simplify(gauge/gauge[6])
    left=M.subs(s,4).T.nullspace()[0];tau=S.symbols('tau')
    leading_error=float(np.max(abs(fit[0].reshape(10,10)-np.array(M.subs(s,3),float))))
    assert leading_error<1e-4,leading_error
    numerator=S.factor((left.T*(tau*S.eye(10)-exact)*mass)[0])
    denominator=S.factor((left.T*M.diff(s).subs(s,4)*gauge)[0])
    numeric_log=float((-np.array(left.T,float)@m1@np.array(mass,float))[0,0]/float(denominator))
    assert abs(numeric_log-120/13)<.001,numeric_log
    result=dict(first_recurrence_matrix=str(exact),background=args.background,radius_scale=args.radius_scale,fd_step=args.fd_step,numeric_static_log_coefficient=numeric_log,exact_static_log_coefficient='120/13',
                rational_error=float(np.max(abs(m1-np.array(exact,float)))),leading_error=leading_error,
                mass_mode=[str(v) for v in mass],gauge_mode=[str(v) for v in gauge],
                compatibility_numerator=str(numerator),log_coefficient=str(S.factor(numerator/denominator)),
                caveat='First formal recurrence only. The static coefficient is independently checked by resonance_exact.py; the time-dependent coefficient is not promoted to a result.')
    (root/(args.name+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
