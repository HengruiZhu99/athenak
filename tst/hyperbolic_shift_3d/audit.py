"""Audit compiled Cartesian kernels; this is NOT an evolution framework."""
import argparse
import ctypes as C
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_1d'))
from boundary import vector_shift, vector_rate_identity
from principal import coefficient, symbol as donor_symbol
from gamma_driver import parameters
from constraint_damping import correction
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from telegraph import window
from elliptic import connection
from kernel import Kernel

fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')

def eigenspaces(P,expected):
    groups=[]
    for z in sorted(expected):
        if groups and abs(z-groups[-1][0])<1.e-8:groups[-1][1]+=1
        else:groups.append([float(z),1])
    spaces=[];vectors=[]
    for z,mult in groups:
        _,s,vh=np.linalg.svd(P-z*np.eye(len(P)))
        nullity=int(sum(s<2.e-9))
        spaces.append(dict(value=z,algebraic=mult,geometric=nullity))
        vectors.extend(vh[-nullity:])
    complete=all(d['algebraic']==d['geometric'] for d in spaces)
    return dict(complete=complete,eigenspaces=spaces,
        basis_condition=float(np.linalg.cond(np.array(vectors).T)) if complete else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    lib=C.CDLL(str(a.library.resolve()))
    lib.hyperbolic_symbol.argtypes=[C.c_double,C.c_double,fp,fp]+[C.c_double]*6+[fp]
    lib.hyperbolic_radial_gauge.argtypes=[C.c_double,fp,fp,fp]+[C.c_double]*3+[fp]
    lib.hyperbolic_scri.argtypes=[C.c_double,C.c_double,fp,fp,fp,C.c_double,fp,fp,C.c_double,fp,fp]
    lib.hyperbolic_metric_max.argtypes=[fp];lib.hyperbolic_metric_max.restype=C.c_double
    rng=np.random.default_rng(20261028)
    donor_kernel=Kernel(a.library)
    radial_errors=[]
    for r in np.r_[np.linspace(.001,.999,101),.56,.8,.9,1.]:
        q=rng.normal(0,.01,8);d=rng.normal(0,.01,8);dd=rng.normal(0,.01,8)
        B,dB,D=rng.normal(0,.01,3);out=np.zeros(11)
        assert lib.hyperbolic_radial_gauge(r,q,d,dd,B,dB,D,out)==0
        O=(1-r*r)/2;A=(1+r*r)/2;alpha=A+q[6];chi=1+q[0];g=1+q[1]
        w,wp=window(np.array([r]),width=.24,inner_profile='af-regular');w=float(w[0]);wp=float(wp[0])
        eta=1.0001*alpha/(alpha+1.e-4*A);F=alpha**2+4*O**2*alpha
        sep=coefficient(alpha,chi,g,F,eta,w,.005,.02)
        mu,xi=parameters(r,sep,4.,4.)
        beta=-r+q[7];da=r+d[6]
        R,S,_=donor_kernel.parts(np.array([r]),q[None,:],d[None,:],dd[None,:],
            np.array([beta]),np.array([-1+d[7]]),np.array([dd[7]]))
        expected=np.array([R[0,6]+eta*chi*(w*dB+(wp+2*w/r)*B),
            S[0,6]-2*(1+2*(1-r*r)**4)*alpha*q[6]
                +16*O**4*r*r*q[6]/A,
            beta*(-1+d[7])-r+mu*D,-xi*q[7],
            beta*dB+(.005*w*d[6]-B)/.02,mu,w,wp])
        delta=correction(np.array([r]),np.array([alpha]),np.array([q[4]]),
            np.array([q[5]-connection(r,g,d[1])]),np.array([w]),8.)[0]
        expected=np.r_[expected,delta[3],delta[4]+3*beta*r*q[4],delta[5]]
        radial_errors.append(float(max(abs(out-expected))))
    assert max(radial_errors)<1.e-11,max(radial_errors)

    boundary=[]
    for sample in range(200):
        m=rng.normal(size=(3,3));g=m@m.T+np.eye(3)
        gt=rng.normal(size=(3,3));gt=(gt+gt.T)/2
        n=rng.normal(size=3);trial=rng.normal(size=3);other=rng.normal(size=3)
        alpha=float(np.exp(rng.uniform(-3,1)));at=float(rng.normal());mu=1.5
        D=rng.normal(size=3);oldD=D.copy();out=np.zeros(4)
        assert lib.hyperbolic_scri(0.,alpha,g,n,trial,at,gt,other,mu,D,out)==0
        s=n/np.sqrt(n@np.linalg.inv(g)@n)
        sup=np.linalg.solve(g,s)
        beta=vector_shift(alpha,g,n,trial)
        required=vector_rate_identity(alpha,g,n,beta,at,gt)
        assert np.max(abs(out[:3]-beta))<1.e-12
        assert abs(s@(other+mu*D)-required)<1.e-12
        assert np.linalg.norm((np.eye(3)-np.outer(sup,s))@(D-oldD))<1.e-12
        dt=1.e-5
        fd=(vector_shift(alpha+dt*at,g+dt*gt,n,trial)-vector_shift(alpha-dt*at,g-dt*gt,n,trial))/(2*dt)
        boundary.append(dict(null=abs(s@out[:3]+alpha),rate=abs(s@fd-required)))
    assert max(x['rate'] for x in boundary)<2.e-8
    assert lib.hyperbolic_scri(1.e-14,alpha,g,n,trial,at,gt,other,mu,D,out)!=0
    bad=np.diag([-1.,-1.,1.]);assert lib.hyperbolic_scri(0.,1.,bad,n,trial,at,gt,other,mu,D,out)!=0

    def symbol(alpha,chi,g,n,F,eta,w,mu):
        P=np.zeros((26,26));n=np.ascontiguousarray(n,dtype=float)
        g=np.ascontiguousarray(g,dtype=float)
        assert lib.hyperbolic_symbol(alpha,chi,g,n,F,eta,w,.005,.02,mu,P)==0
        q=n@np.linalg.inv(g)@n;e=n@n
        cp=alpha*np.sqrt(chi*q);cl=np.sqrt(chi*F*q+eta*chi*.25*w*w*e)
        cs=np.sqrt(4*mu*q/3);ct=np.sqrt(mu*q)
        expected=np.array([0.]*6+[-cp,cp]*6+[-cl,cl,-cs,cs]+[-ct,ct]*2)
        values=np.linalg.eigvals(P)
        error=float(max(abs(np.sort_complex(values)-np.sort_complex(expected))))
        # Sort real parts separately: tiny imaginary roundoff otherwise permutes
        # degenerate eigenvalues in lexicographic complex sorting.
        error=max(float(max(abs(np.sort(values.real)-np.sort(expected)))),float(max(abs(values.imag))))
        return P,expected,error,eigenspaces(P,expected)

    samples=[];modified_samples=[]
    radial_symbol_errors=[]
    for alpha,chi,gscalar in [(.8,1.2,.9),(1.,1.,1.),(.2,.3,1.4)]:
        F=alpha**2+.64*alpha;eta=.9;w=.7;mu=4.
        P,_,_,_=symbol(alpha,chi,np.diag([gscalar,gscalar**-.5,gscalar**-.5]),
            [1.,0.,0.],F,eta,w,mu)
        indices=[0,1,6,11,12,13,16,17,20,23]
        lift=np.zeros((26,10));lift[indices,np.arange(10)]=1
        lift[4,1]=-.5*gscalar**-1.5;lift[9,2]=-.5*gscalar**-1.5
        expected=donor_symbol(alpha,chi,gscalar,F,eta,w,.005,.02,mu)
        radial_symbol_errors.append(float(np.max(abs((P@lift)[indices]-expected))))
    assert max(radial_symbol_errors)<1.e-11,radial_symbol_errors
    for case in range(100):
        m=rng.normal(size=(3,3));g=m@m.T+np.eye(3);g/=np.linalg.det(g)**(1/3)
        n=rng.normal(size=3);n/=np.linalg.norm(n)
        assert abs(lib.hyperbolic_metric_max(g)-np.linalg.eigvalsh(g)[-1])<1.e-12
        r=float(rng.uniform(0,1));alpha=(1+r*r)/2;chi=1.
        w=float(window(np.array([r]),width=.24,inner_profile='af-regular')[0][0])
        eta=1.;F=alpha**2+(1-r*r)**2*alpha
        mu=float(parameters(r,1.5*chi*F,4.,4.)[0])
        P,expected,err,audit=symbol(alpha,chi,g,n,F,eta,w,mu)
        assert err<1.e-7 and audit['complete'],(case,err,audit)
        samples.append(dict(error=err,condition=audit['basis_condition']))
        mu_bound=1.5*chi*(F+eta*.25*w*w*np.linalg.eigvalsh(g)[-1])
        _,_,err_mod,audit_mod=symbol(alpha,chi,g,n,F,eta,w,mu_bound)
        assert err_mod<1.e-7 and audit_mod['complete'],(case,err_mod,audit_mod)
        modified_samples.append(dict(error=err_mod,condition=audit_mod['basis_condition']))

    sector_collisions={}
    for label,F,w,mu in [('transverse_shift_light',.75,1.,.25),
                        ('lapse_light',.25,0.,4.),
                        ('transverse_shift_lapse',.75,1.,1.)]:
        _,_,err_test,test=symbol(.5,1.,np.eye(3),[1.,0.,0.],F,1.,w,mu)
        assert err_test<1.e-7 and test['complete'],(label,err_test,test)
        sector_collisions[label]=test

    P,expected,err,scri=symbol(1.,1.,np.eye(3),[1.,0.,0.],1.,1.,0.,1.5)
    assert scri['complete'] and err<1.e-7
    coordinate=1-expected
    counts=dict(incoming=int(sum(coordinate< -1.e-8)),outgoing=int(sum(coordinate>1.e-8)),tangent=int(sum(abs(coordinate)<1.e-8)))
    assert counts==dict(incoming=3,outgoing=16,tangent=7)

    # A positive, determinant-one spherical metric with radial margin >0 but
    # a defective TANGENTIAL lapse/longitudinal-shift speed coincidence.
    r=.4;alpha=.5;chi=1.;A=(1+r*r)/2;eta=1.0001*alpha/(alpha+1.e-4*A)
    F=alpha**2+(1-r*r)**2*alpha;T=(16/3-F)/(.25*eta)
    g=np.diag([1/T**2,T,T]);n=np.array([0.,1.,0.])
    P,expected,err,collision=symbol(alpha,chi,g,n,F,eta,1.,4.)
    assert not collision['complete']
    defective=[v for v in collision['eigenspaces'] if v['algebraic']!=v['geometric']]
    assert len(defective)==2 and all(v['algebraic']==2 and v['geometric']==1 for v in defective),defective
    np.savez(a.output/'oblique-collision.npz',P=P,metric=g,normal=n,expected=expected)
    # Explicitly labeled modification: change mu enough to separate ALL
    # directions using lambda_max(g). This does NOT retain constant bulk four.
    mu_modified=1.5*chi*(F+.25*eta*np.linalg.eigvalsh(g)[-1])
    _,_,moderr,modified=symbol(alpha,chi,g,n,F,eta,1.,mu_modified)
    assert modified['complete'] and moderr<1.e-7
    record=dict(donor='c0ae7472a7aecc92cc72312c33d4cdc417a26291',
        radial_point_max_error=max(radial_errors),radial_cases=len(radial_errors),
        radial_native_symbol_errors=radial_symbol_errors,
        boundary_cases=len(boundary),max_null=max(x['null'] for x in boundary),
        max_moving_normal_fd_error=max(x['rate'] for x in boundary),
        positive_metric_oblique_samples=samples,sector_collision_controls=sector_collisions,
        scri=scri,scri_counts=counts,
        obstruction=dict(r=r,alpha=alpha,chi=chi,F=F,eta=eta,mu=4.,metric=g.tolist(),
            direction=n.tolist(),radial_gap_times_g=16/3-chi*(F+g[0,0]*eta*.25),audit=collision),
        modified_control=dict(mu=mu_modified,audit=modified,samples=modified_samples,adopted=False,
            reason='Changes the required constant bulk coefficient; point control only.'),
        scope='Native 3D point kernels and interior frozen symbol. No task integration, evolution, boundary well-posedness, or uniform puncture estimate.')
    (a.output/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['radial_point_max_error','max_null','scri_counts','scope']},indent=2))

if __name__=='__main__':main()
