#!/usr/bin/env python3
"""Check the spatial-coordinate gauge reduction against the full principal kernel."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from analyze import position


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    root=a.output;meta=json.loads((root/'indicial025.json').read_text())
    K=np.fromfile(root/'indicial025.bin').reshape(meta['shape'])
    Kh=np.fromfile(root/'indicial025-half.bin').reshape(meta['shape']);K=(4*Kh-K)/3
    fields=meta['fields'];ix=lambda f:fields.index(f)
    rho=1-np.array([z['r'] for z in meta['points']]);coeff=[]
    for J,z in zip(K,meta['points']):
        scale=np.array([z['omega'] if f in [13,14] else 1 for f in fields])
        C=np.zeros((10,26,26))
        for i,fi in enumerate(fields):
            for j,fj in enumerate(fields):
                kinds=([1,2,3] if position(fj) else [0]) if position(fi) else ([4,5,6,7,8,9] if position(fj) else [1,2,3])
                for kind in kinds:C[kind,i,j]=J[i,10*j+kind]*scale[j]/scale[i]
        coeff.append(C)
    lim=np.polynomial.polynomial.polyfit(rho,np.array(coeff).reshape(4,-1),3)[0].reshape(10,26,26)
    C=np.array([float(Fraction(float(x)).limit_denominator(120)) for x in lim.ravel()]).reshape(lim.shape)
    rational_error=float(np.max(abs(lim-C)));assert rational_error<1e-6
    def symbol(tau,lam,eta):
        w=[1,lam,1j*eta,0,lam*lam,lam*1j*eta,0,-eta*eta,0,0]
        return tau*np.eye(26)-np.einsum('k,kij->ij',w,C)
    def coordinate_mode(tau,lam,eta,xi):
        d=np.array([lam,1j*eta,0]);div=np.dot(d,xi);v=np.zeros(26,complex)
        v[ix(0)]=-2*div/3
        for f,i,j in [(1,0,0),(2,0,1),(3,0,2),(4,1,1),(5,1,2)]:
            v[ix(f)]=d[i]*xi[j]+d[j]*xi[i]-(2*div/3 if i==j else 0)
        connection=(lam*lam-eta*eta)*xi+d*div/3
        for j in range(3):
            v[ix(15+j)]=connection[j];v[ix(25+j)]=connection[j]
            v[ix(22+j)]=(tau+lam)*xi[j]
        return v
    tau=.2+.3j;lam=.4+.1j;eta=.7;xi=np.array([.3+.1j,.2-.4j,.5+.2j])
    v=coordinate_mode(tau,lam,eta,xi);d=np.array([lam,1j*eta,0])
    expected=np.zeros(26,complex)
    for j in range(3):expected[ix(22+j)]=((tau+lam)**2*xi-1.5*((lam*lam-eta*eta)*xi+d*np.dot(d,xi)/3))[j]
    identity_error=float(np.max(abs(np.einsum('ij,j->i',symbol(tau,lam,eta),v)-expected)))
    assert identity_error<1e-10
    B=np.zeros((3,26));B[0,ix(22)]=1;B[0,ix(0)]=.5;B[0,ix(1)]=-.5
    B[1,ix(23)]=1;B[2,ix(24)]=1
    checks=[]
    for sigma in [.1,.01,.001,.0001]:
        for theta in np.linspace(0,np.pi/2,41):
            tau=sigma+1j*np.sqrt(1-sigma*sigma)*np.cos(theta)
            eta=np.sqrt(1-sigma*sigma)*np.sin(theta)
            normal=[]
            for c in [np.sqrt(2),np.sqrt(1.5)]:
                normal.append((tau+c*np.sqrt(tau*tau+(c*c-1)*eta*eta+0j))/(c*c-1))
            l,t=normal
            modes=[coordinate_mode(tau,l,eta,np.array([l,1j*eta,0])),
                   coordinate_mode(tau,t,eta,np.array([1j*eta,-t,0])),
                   coordinate_mode(tau,t,eta,np.array([0,0,1]))]
            R=np.column_stack([v/np.linalg.norm(v) for v in modes])
            residual=max(float(np.linalg.norm(np.einsum('ij,j->i',symbol(tau,z,eta),v))/np.linalg.norm(v)) for z,v in zip([l,t,t],modes))
            assert residual<1e-9
            smallest=float(np.linalg.svd(B@R,compute_uv=False)[-1])
            checks.append(dict(sigma=sigma,eta=float(eta),omega=float(tau.imag),smallest_boundary_singular_value=smallest,kernel_residual=residual))
    result=dict(principal_coefficient_rational_error=rational_error,coordinate_mode_identity_error=identity_error,
                samples=len(checks),max_kernel_residual=max(z['kernel_residual'] for z in checks),
                min_boundary_singular_value=min(z['smallest_boundary_singular_value'] for z in checks),checks=checks,
                caveat='Finite frequency-direction samples support the analytic principal shift energy identity; no uniform regular-singular full-system bound is inferred.')
    (root/'principal-gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
