#!/usr/bin/env python3
"""Re-extract near-scri mass from saved binary64 states with Decimal arithmetic.

Uses exact rational Lagrange weights for the same seven-node stencil. This
isolates postprocessing roundoff; it cannot remove error already in the state.
"""
import argparse,json,hashlib
from pathlib import Path
from fractions import Fraction
from decimal import Decimal,localcontext
import numpy as np


def decimal_mass(r, q, i, precision=80):
    with localcontext() as context:
        context.prec=precision
        D=lambda value:Decimal.from_float(float(value))
        F=lambda value:Decimal(value.numerator)/Decimal(value.denominator)
        n=len(r);start=min(max(i-3,0),n-7);indices=list(range(start,start+7))
        xs=[Fraction(float(r[j])) for j in indices];x=Fraction(float(r[i]))
        weights=[]
        for j,xj in enumerate(xs):
            if xj==x:
                weight=sum((1/(x-xk) for xk in xs if xk!=x),Fraction(0))
            else:
                num=Fraction(1);den=Fraction(1)
                for k,xk in enumerate(xs):
                    if k!=j:den*=xj-xk
                    if k!=j and xk!=x:num*=x-xk
                weight=num/den
            weights.append(weight)
        slope=sum((F(w)*(D(q[j,0])-D(q[i,0])) for j,w in zip(indices,weights)),Decimal(0))
        x=D(r[i]);O=(1-x*x)/2;A=(1+x*x)/2
        if not 0<x<1:raise ValueError('Finite-radius mass only')
        cr=x**4/(x**4+O**4)
        crp=(4*x**3*O**4+4*x**5*O**3)/(x**4+O**4)**2
        ex=D(q[i,0]).exp();chi=cr*ex;cp=ex*(crp+cr*slope)
        c=cr-1+cr*(ex-1);w=x*x/(x*x+O**4);gap=O**4/(x*x+O**4)
        Arr=w*D(q[i,2]);dk=3*gap+w*(D(q[i,3])+2*D(q[i,4]));K=dk-3
        numerator=dk*(K-3)/9-c+O*(-K*Arr/3+A*cp/x)+O*O*(Arr*Arr/4-cp*cp/(4*chi))
        canceled=x**3*numerator/(2*chi*chi.sqrt()*O**3)
        R=x/(O*chi.sqrt());normal=R*(K/3-O*Arr/2);spatial=A/O-x*cp/(2*chi)
        geometric=R*(1+normal*normal-spatial*spatial)/2
        if abs(canceled-geometric)>Decimal(10)**(-precision+20):
            raise AssertionError('Independent geometric mass formula disagrees')
        return canceled


def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
    for path in a.profiles:
        with np.load(path,allow_pickle=False) as data:
            r=data['r'];q=data['q'];mass=data['diagnostic'][:,10]
            indices=sorted(set(list(range(len(r)-5,len(r)-1))+[int(np.argmin(abs(r-(1-1/1024))))]))
            rows=[]
            for i in indices:
                high=decimal_mass(r,q,i,80);control=decimal_mass(r,q,i,60)
                if abs(high-control)>Decimal('1e-40'):raise AssertionError('Precision check failed')
                rows.append(dict(index=i,r=float(r[i]),saved_mass=float(mass[i]),decimal80_mass=str(high),
                    extraction_difference=float(high-Decimal.from_float(float(mass[i]))),
                    mass_error=float(high-Decimal(1)/3)))
            records.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n=len(r),time=float(data['t']),points=rows))
    result=dict(records=records,scope='Same saved states and seven-node derivative stencil; exact rational stencil weights and 60/80-digit Decimal reconstruction. Geometric and canceled mass formulas agree. This isolates extraction arithmetic only, not roundoff in the evolution or spatial truncation. All failed invariant/constraint diagnostics remain reported.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
