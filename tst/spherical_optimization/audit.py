#!/usr/bin/env python3
"""Gate reports: symbols, independent initial constraints, auxiliary trace."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import Evolution

def auxiliary_audit(e,q):
    rate=e.rhs(q);base=e.last.copy();eps=1e-6/max(1.,np.max(abs(rate)))
    e.rhs(q+eps*rate);plus=e.last['dependent_D']
    e.rhs(q-eps*rate);minus=e.last['dependent_D']
    derivative=(plus-minus)/(2*eps)
    return dict(dependent_trace_rate=derivative,bulk_auxiliary_trace_rate=base['auxiliary_trace_pde'],
                differential_defect=derivative-base['auxiliary_trace_pde'],probe_epsilon=eps)

def eigenspace_audit(P):
    eigen=np.linalg.eigvals(P);groups=[]
    for value in sorted(eigen.real):
        if not groups or abs(value-np.mean(groups[-1]))>1e-7*max(1.,max(abs(eigen))):groups.append([value])
        else:groups[-1].append(value)
    basis=[];defects=[];residual=0.
    for group in groups:
        value=np.mean(group);U,sv,Vh=np.linalg.svd(P-value*np.eye(len(P)))
        nullity=int(sum(sv<1e-8*max(1.,sv[0])))
        m=len(group);vectors=Vh[-m:].T;basis.append(vectors)
        residual=max(residual,float(np.linalg.norm(P@vectors-value*vectors)))
        if nullity<m:defects.append(dict(eigenvalue=float(value),multiplicity=m,nullity=nullity))
    return dict(condition=float(np.linalg.cond(np.concatenate(basis,axis=1))),defects=defects,residual=residual)

def coincidences(k):
    from scipy.optimize import brentq
    def gap(r):
        q,_,_=k.initial(np.array([r]));v=k.gauge_speeds(r,q[0]);return v[0]-v[1]
    r=np.linspace(.003,.999,301);g=[gap(x) for x in r];roots=[]
    for i in range(len(r)-1):
        if g[i]*g[i+1]<0:
            x=brentq(gap,r[i],r[i+1],xtol=1e-14);q,_,_=k.initial(np.array([x]))
            roots.append(dict(r=x,radial=eigenspace_audit(k.symbol(x,q[0])),transverse=eigenspace_audit(k.symbol(x,q[0],(0.,1.,0.)))))
    return roots

def symbol_audit(k,r,q):
    rows=[]
    for i in range(len(r)):
        if r[i]>=1:continue
        for label,n in [('radial',(1.,0.,0.)),('transverse',(0.,1.,0.)),('oblique',(.6,.8,0.))]:
            P=k.symbol(r[i],q[i],n);lam,V=np.linalg.eig(P)
            space=eigenspace_audit(P)
            rows.append([r[i],{'radial':0,'transverse':1,'oblique':2}[label],max(abs(lam.real)),max(abs(lam.imag)),np.linalg.cond(V),space['condition'],len(space['defects']),space['residual']])
    return np.array(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);result={}
    for mode,name in [(0,'historical'),(1,'central20'),(2,'central20-separated')]:
        k=Kernel(a.library,min(mode,1),separated=(mode==2));info={'configuration':k.config.tolist(),'initial':[],'coincidences':coincidences(k)}
        for n in (64,128,256):
            e=Evolution(k,n);q=e.initial;report=e.report(q,0);aux=auxiliary_audit(e,q)
            # Independent physical-geometry initial constraints (isotropic metric).
            w=e.omega[:,0]*np.sqrt(q[:,0]);dw,ddw=e.op.derivatives(w,1)
            kr=e.omega[:,0]*q[:,3]+q[:,5]/3;kt=e.omega[:,0]*q[:,4]+q[:,5]/3
            dkr,_=e.op.derivatives(kr,1);dK,_=e.op.derivatives(q[:,5],1)
            H=4*w*(ddw+2*dw/e.r)-6*dw*dw+q[:,5]**2-kr**2-2*kt**2
            M=w*(dkr+2*(1/e.r-np.divide(dw,w,out=np.zeros_like(w),where=w!=0))*(kr-kt)-dK)
            cut=(e.r>=.65)&(e.r<=.995);weight=e.r[cut]**2*e.dr[cut]
            independent_H=np.sqrt(np.average(H[cut]**2,weights=weight));independent_M=np.sqrt(np.average(M[cut]**2,weights=weight))
            info['initial'].append(dict(n=n,report=report,auxiliary=aux,independent_H=float(independent_H),independent_M=float(independent_M)))
        r=np.linspace(.003, .999,129);q,src,om=k.initial(r);symbols=symbol_audit(k,r,q)
        np.savetxt(a.output/f'{name}-symbols.csv',symbols,delimiter=',',header='r,direction,max_abs_speed,max_imag,raw_eigenvector_condition,orthonormal_eigenspace_condition,defective_clusters,eigenspace_residual',comments='')
        info['symbol']=dict(max_speed=float(max(symbols[:,2])),max_imag=float(max(symbols[:,3])),max_raw_eigenvector_condition=float(max(symbols[:,4])),max_eigenspace_condition=float(max(symbols[:,5])),defective_sampled_clusters=int(sum(symbols[:,6])),
            caveat='Numerical repeated-eigenvalue conditioning is basis dependent; interior samples are not a uniform endpoint or completeness proof.')
        result[name]=info;k.close()
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
