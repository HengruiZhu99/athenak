#!/usr/bin/env python3
"""Frozen-face, full-field outer strip: native normal closure, Fourier angles.

This is not the global cubed-sphere/seam operator. Angular derivatives are
continuous Fourier derivatives in local angular coordinates at the +x face.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as ss
from analyze import sample


def weights(nodes, at, derivative):
    offsets=np.asarray(nodes)-at
    rhs=np.zeros(len(nodes));rhs[derivative]=float(math.factorial(derivative))
    return np.linalg.solve(np.array([offsets**k for k in range(len(nodes))]),rhs)


def assemble(meta, kernels, eta, lift=True, upwind=True):
    fields=meta['fields'];nf=26;count=len(meta['points']);n=count-1
    r=np.array([p['r'] for p in meta['points']]);h=meta['spacing'];size=count*nf
    ix=lambda f:fields.index(f)
    # First four inner nodes fixed; this artificial inner closure is a limitation.
    free=[i*nf+f for i in range(4,count) for f in range(nf)
          if i<n or fields[f] not in [18,22,23,24,25,26,27]]
    P=ss.eye(size,format='lil',dtype=complex)
    P[:4*nf,:]=0
    for f in [18,22,23,24,25,26,27]:P[n*nf+ix(f),:]=0
    P[n*nf+ix(22),n*nf+ix(0)]=-.5
    P[n*nf+ix(22),n*nf+ix(1)]=.5
    ex=weights(np.arange(n-5,n),n,0)
    for a in range(3):
        P[n*nf+ix(25+a),n*nf+ix(15+a)]=1
        for q,w in zip(range(n-5,n),ex):
            P[n*nf+ix(25+a),q*nf+ix(25+a)]=w
            P[n*nf+ix(25+a),q*nf+ix(15+a)]=-w
    E=P.tocsr()[:,free]
    d1=np.zeros((count,count));d2=d1.copy();biased=d1.copy()
    for i in range(4,count):
        nodes=np.arange(n-6,n+1) if i>=n-3 else np.arange(i-2,i+3)
        d1[i,nodes]=weights(nodes,i,1)/h;d2[i,nodes]=weights(nodes,i,2)/(h*h)
        if i<n-3:
            nodes=np.array([i+1,i,i-1,i-2,i-3])
            biased[i,nodes]=weights(nodes,i,1)/h
    jets=[]
    for i in range(count):
        J=ss.lil_matrix((260,size),dtype=complex)
        if i<4:jets.append(J.tocsr());continue
        for f,field in enumerate(fields):
            v=np.zeros(count);v[i]=1;dx=d1[i].copy();dxx=d2[i].copy()
            if field==0:
                # Production origin-r^2 residual convention; scri weight is one.
                dx=(r[i]**2*d1[i]+2*r[i]*v)/(r*r)
                dxx=(r[i]**2*d2[i]+4*r[i]*d1[i]+2*v)/(r*r)
            rows=[v,dx,1j*eta*v/r[i],0*v,dxx,
                  1j*eta*(dx/r[i]-v/r[i]**2),0*v,
                  dx/r[i]-eta*eta*v/r[i]**2,0*v,dx/r[i]]
            for kind,row in enumerate(rows):
                for q in np.flatnonzero(row):J[10*f+kind,q*nf+f]=row[q]
        jets.append(J.tocsr())
    edge=kernels[-1];response=np.column_stack([edge[26:29,10*ix(22+a)+1] for a in range(3)])
    correction=ss.csr_matrix(-np.linalg.solve(response,edge[26:29]@jets[-1]))
    H=np.polynomial.Polynomial.fromroots(-np.arange(7))/720
    blocks=[];normal_residual=None;clock=None
    for i,J in enumerate(jets):
        if lift and i>=n-2:
            z=i-n;value=h*H(z);first=H.deriv()(z);second=H.deriv(2)(z)/h
            L=np.zeros((260,3),complex)
            rows=[value,first,1j*eta*value/r[i],0,second,
                  1j*eta*(first/r[i]-value/r[i]**2),0,
                  first/r[i]-eta*eta*value/r[i]**2,0,first/r[i]]
            for a in range(3):L[10*ix(22+a):10*ix(22+a)+10,a]=rows
            J=J+ss.csr_matrix(L)@correction
        K=ss.csr_matrix(kernels[i,:26])@J
        if upwind and 4<=i<n-3:
            transport=ss.lil_matrix((26,size),dtype=complex)
            beta=meta['points'][i]['values'][ix(22)]-r[i]
            for f,field in enumerate(fields):
                row=beta*(biased[i]-d1[i])
                if field==0:row=row*r[i]**2/(r*r)
                for q in np.flatnonzero(row):transport[f,q*nf+f]=row[q]
            K=K+transport.tocsr()
        blocks.append(K)
        if i==n:
            normal_residual=float(np.max(abs((edge[26:29]@J)@E)))
            clock=np.asarray((edge[29:30]@J)@E).ravel()
    operator=(ss.vstack(blocks).tocsr()@E)[free,:].toarray()
    return operator,E,free,clock,normal_residual,fields,r,response


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--n',type=int,default=24);p.add_argument('--width',type=float,default=.06)
    p.add_argument('--kappa',type=float,default=.25);p.add_argument('--eta',type=float,default=0)
    p.add_argument('--fdstep',type=float,default=1e-5)
    p.add_argument('--background',choices=['flat','wormhole'],default='flat')
    p.add_argument('--no-lift',action='store_true');args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    h=args.width/args.n;r=1-h*np.arange(args.n,-1,-1)
    meta,K=sample(args.binary.resolve(),out,'kernels',r,args.kappa,h,args.fdstep,args.background)
    L,E,free,clock,residual,fields,r,response=assemble(meta,K,args.eta,not args.no_lift)
    eigen,vectors=la.eig(L,check_finite=True)
    order=np.argsort(eigen.real)[::-1];rows=[]
    for j in order[:12]:
        mode=(E@vectors[:,j]).reshape(-1,26)
        energy=np.sum(abs(mode)**2,axis=1);energy/=energy.sum()
        rows.append(dict(real=float(eigen[j].real),imag=float(eigen[j].imag),
                         last_six_value_fraction=float(energy[-7:].sum()),
                         radial_centroid=float(energy@r),
                         relative_residual=float(np.linalg.norm(np.einsum('ij,j->i',L,vectors[:,j])-eigen[j]*vectors[:,j])/max(1,np.linalg.norm(L)*np.linalg.norm(vectors[:,j])))))
    dt=.0938884074712349*h;z=dt*eigen;rk=1+z+z*z/2+z**3/6+z**4/24
    result=dict(n=args.n,h=h,width=args.width,kappa=args.kappa,eta=args.eta,
                background=args.background,lift=not args.no_lift,ndof=L.shape[0],
                normal_solve_condition=float(np.linalg.cond(response)),
                normal_compatibility_matrix_residual=residual,
                max_real_eigenvalue=float(max(eigen.real)),max_rk4_modulus=float(max(abs(rk))),
                eigenvalue_leaders=rows,
                caveat='Frozen +x face, continuous tangential Fourier modes, artificial fixed inner four nodes; no angular grid, seams, or uniform energy estimate.')
    np.savez_compressed(out/'operator.npz',operator=L,eigenvalues=eigen,clock=clock,r=r,fields=fields)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
