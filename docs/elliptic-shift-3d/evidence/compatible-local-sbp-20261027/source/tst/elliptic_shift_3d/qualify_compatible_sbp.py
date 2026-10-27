#!/usr/bin/env python3
"""Falsifiable accuracy/energy controls for a standalone replacement derivative pair."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from compatible_sbp import local,chain,face,mortar,mm,basis


def observed(errors,counts):
    return [float(np.log(errors[i-1]/errors[i])/np.log(counts[i]/counts[i-1])) for i in range(1,len(errors))]


def tensor_control(n,p):
    a=chain(n,p,[1.],periodic=True);D=a['Dminus'];E=a['Dplus'];x=a['x'];w=a['H']
    X=np.meshgrid(x,x,x,indexing='ij');W=w[:,None,None]*w[None,:,None]*w[None,None,:]
    g=np.array([[1.3,.13,-.07],[.13,.9,.08],[-.07,.08,1.1]]);gi=np.linalg.inv(g);W*=np.sqrt(np.linalg.det(g))
    def derivative(value,axis,op):
        moved=np.moveaxis(value,axis,0)
        return np.moveaxis(mm(op,moved.reshape(n,-1)).reshape(moved.shape),0,axis)
    def rates(b):
        grad=np.stack([derivative(b,j,D) for j in range(3)],axis=-2)
        div=np.einsum('...ii->...',grad,optimize=False)
        h=np.einsum('kj,...ik->...ij',g,grad,optimize=False)+np.einsum('ik,...jk->...ij',g,grad,optimize=False)-(2/3)*g*div[...,None,None]
        return h
    def variation(h):
        dh=np.stack([derivative(h,j,E) for j in range(3)],axis=-3)
        return np.einsum('jk,il,...jlk->...i',gi,gi,dh,optimize=False)-.5*np.einsum('jk,il,...ljk->...i',gi,gi,dh,optimize=False)
    def connection(metric):
        inv=np.linalg.inv(metric);dg=np.stack([derivative(metric,j,E) for j in range(3)],axis=-3)
        return np.einsum('...jk,...il,...jlk->...i',inv,inv,dg,optimize=False)-.5*np.einsum('...jk,...il,...ljk->...i',inv,inv,dg,optimize=False)
    def dot(b,c):return float(np.sum(W*np.einsum('...i,ij,...j->...',b,g,c,optimize=False)))
    cases={};random=np.random.default_rng(271127)
    smooth=np.stack([np.sin(2*np.pi*X[0])*np.cos(2*np.pi*X[1]),.3*np.sin(2*np.pi*X[2]),.2*np.cos(2*np.pi*(X[0]+X[2]))],axis=-1)
    checker=np.zeros_like(smooth);checker[...,0]=(-1.)**np.indices((n,n,n)).sum(axis=0)
    for name,b in [('smooth',smooth),('checker',checker),('random',random.normal(size=smooth.shape))]:
        h=rates(b);Ab=-variation(h)
        strain=float(.5*np.sum(W*np.einsum('ik,jl,...ij,...kl->...',gi,gi,h,h,optimize=False)))
        energy=dot(b,Ab);c=random.normal(size=b.shape);Ac=-variation(rates(c))
        cases[name]={'energy':energy,'strain':strain,'energy_relative_defect':abs(energy-strain)/max(1,abs(strain)),'rayleigh':energy/dot(b,b),'bilinear_symmetry_relative':abs(dot(b,Ac)-dot(Ab,c))/max(1,abs(dot(b,Ac)),abs(dot(Ab,c)))}
    h=rates(smooth);exact=variation(h);metric=np.broadcast_to(g,h.shape).copy();fd=[]
    for eps in [1e-3,1e-4,1e-5,1e-6]:
        plus=metric+eps*h;minus=metric-eps*h
        if min(np.min(np.linalg.det(plus)),np.min(np.linalg.det(minus)))<=0:raise ValueError('finite perturbation lost SPD')
        difference=(connection(plus)-connection(minus))/(2*eps)-exact
        fd.append({'epsilon':eps,'scaled_rms':float(np.sqrt(np.mean(difference*difference))/max(1,np.sqrt(np.mean(exact*exact)))),'maximum':float(np.max(abs(difference)))})
    return {'n':n,'cells':n**3,'degree':p,'metric':g.tolist(),'cases':cases,'frechet':fd,'scope':'Periodic Cartesian tensor kernel; constant non-diagonal metric only. No mapped shells, scri, variable-metric or evolution claim.'}


def run():
    report={'scope':'Standalone component progress toward full variable-metric 3D gauge; no operator adoption','local':[],'chains':[],'mortars':[],'periodic':[],'gates':{}}
    for p,r in [(2,5),(3,8),(4,11)]:
        ns=[2*r,4*r,8*r];local_errors=[];chain_errors=[];transfer_errors=[]
        for n in ns:
            a=local(n,p);H=a['H'];x=a['x'];B=np.outer(a['trace'][1],a['trace'][1])-np.outer(a['trace'][0],a['trace'][0])
            error=mm(a['Dminus'],np.exp(x))-np.exp(x);e=float(np.sqrt(sum(H*error*error)));local_errors.append(e)
            row={'n':n,'degree':p,'closure_width':r,'minimum_scaled_weight':float(n*min(H)),'sbp_max':float(np.max(abs(H[:,None]*a['Dplus']+a['Dminus'].T*H[None,:]-B))),'polynomial_max':float(np.max(abs(mm(a['Dminus'],a['V'])-a['dV']))),'smooth_rms':e,'smooth_max':float(max(abs(error)))}
            report['local'].append(row)
            c=chain(n,p,[.25,.20,.10,.05,.05,.05,.05,.25]);err=mm(c['Dminus'],np.exp(c['x']))-np.exp(c['x']);ce=float(np.sqrt(sum(c['H']*err*err)));chain_errors.append(ce)
            report['chains'].append({'n':n,'degree':p,'smooth_rms':ce,'smooth_max':float(max(abs(err)))})
            left=face([(1,n)],p);right=face([(.5,n),(.5,n)],p);ILR,IRL=mortar(left,right)
            forward=mm(ILR,np.exp(right['x']))-np.exp(left['x']);reverse=mm(IRL,np.exp(left['x']))-np.exp(right['x'])
            te=max(float(np.sqrt(sum(left['H']*forward**2))),float(np.sqrt(sum(right['H']*reverse**2))));transfer_errors.append(te)
            report['mortars'].append({'n':n,'degree':p,'split_children':2,'adjoint_max':float(np.max(abs(left['H'][:,None]*ILR-IRL.T*right['H'][None,:]))),'polynomial_max':float(max(np.max(abs(mm(ILR,right['V'])-left['V'])),np.max(abs(mm(IRL,left['V'])-right['V'])))),'smooth_rms_max_direction':te,'smooth_point_max':float(max(max(abs(forward)),max(abs(reverse)))),'forward_L1':float(np.max(np.sum(abs(ILR),axis=1))),'reverse_L1':float(np.max(np.sum(abs(IRL),axis=1))))})
        report.setdefault('orders',{})[str(p)]={'local':observed(local_errors,ns),'shell_width_chain':observed(chain_errors,ns),'split_mortar':observed(transfer_errors,ns)}
        for biased in [False,True]:
            n=2*r;c=chain(n,p,[.25]*4,periodic=True,biased=biased);H=c['H'];coefficient=1+.3*np.sin(2*np.pi*c['x'])
            A=-mm(c['Dplus']*coefficient[None,:],c['Dminus']);weighted=np.sqrt(H)[:,None]*A/np.sqrt(H)[None,:];eig=np.linalg.eigvalsh((weighted+weighted.T)/2)
            q=np.random.default_rng(17).normal(size=len(H));v=mm(c['Dminus'],q);energy=float(sum(H*q*mm(A,q)));strain=float(sum(H*coefficient*v*v))
            report['periodic'].append({'n':n,'degree':p,'biased_pair':biased,'minimum_eigenvalue':float(eig[0]),'near_null_count':int(sum(abs(eig)<1e-7)),'smallest_nonnull':float(eig[np.flatnonzero(eig>1e-7)[0]]),'largest_eigenvalue':float(eig[-1]),'energy_relative_error':abs(energy-strain)/max(1,abs(strain))})
    report['tensor']=tensor_control(10,2)
    report['gates']={
       'local_identity_and_polynomials':all(r['sbp_max']<1e-10 and r['polynomial_max']<1e-8 and r['minimum_scaled_weight']>.05 for r in report['local']),
       'mortar_identity_polynomials_amplification':all(r['adjoint_max']<1e-10 and r['polynomial_max']<1e-9 and max(r['forward_L1'],r['reverse_L1'])<16 for r in report['mortars']),
       'smooth_refinement':all(min(o['local'][-1],o['shell_width_chain'][-1])>int(p)+.1 and o['split_mortar'][-1]>int(p)+.1 for p,o in report['orders'].items()),
       'periodic_energy_no_checker_kernel':all(r['near_null_count']==1 and r['minimum_eigenvalue']>-1e-7 and r['energy_relative_error']<2e-12 for r in report['periodic'] if r['biased_pair']),
       'tensor_energy':all(r['energy_relative_defect']<2e-12 and r['bilinear_symmetry_relative']<2e-12 and r['rayleigh']>0 for r in report['tensor']['cases'].values()),
       'tensor_frechet':min(r['scaled_rms'] for r in report['tensor']['frechet'])<1e-7}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    np.seterr(all='raise');start=time.monotonic();r=run();r['seconds']=time.monotonic()-start
    r['python_sources_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('compatible_sbp.py')]}
    (a.output/'assessment.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ['orders','gates','tensor']},indent=2));raise SystemExit(0 if all(r['gates'].values()) else 1)
