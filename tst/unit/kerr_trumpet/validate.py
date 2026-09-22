#!/usr/bin/env python3
"""Standalone stationary Kerr-trumpet geometry checks, not an evolution test.

Requires a C++17 compiler and NumPy. Outputs are written only to --output.
Physical ADM contractions and value-only finite differences are independent of
provider conformal helpers. The generic contractions are shared in form with
the Liu geometry test, but this test has no Liu lapse or radial-map semantics.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,time
import numpy as np

class Provider:
    def __init__(self,library):
        self.lib=ctypes.CDLL(str(library));self.fn=self.lib.trumpet_eval
        self.fn.argtypes=[ctypes.c_double,ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64,shape=(3,),flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64,shape=(387,),flags='C_CONTIGUOUS')]
        self.fn.restype=ctypes.c_int
    def __call__(self,x,spin=.9,gauge=0,mass=1.):
        if gauge!=0:raise ValueError('Only the stationary trumpet gauge is implemented')
        out=np.empty(387);status=self.fn(mass,spin,np.ascontiguousarray(x,dtype=float),out)
        if status:raise ValueError(status)
        j=out[:299].reshape(23,13);f=out[299:].reshape(22,4)
        def second(start,count,shape):
            q=j[start:start+count];return q[:,0].reshape(shape),q[:,1:4].T.reshape((3,)+shape),q[:,4:].T.reshape((3,3)+shape)
        def first(start,count,shape):
            q=f[start:start+count];return q[:,0].reshape(shape),q[:,1:].T.reshape((3,)+shape)
        return dict(g=second(0,9,(3,3)),alpha=second(9,1,()),beta=second(10,3,(3,)),chi=second(13,1,()),gt=second(14,9,(3,3)),
            K=first(0,9,(3,3)),At=first(9,9,(3,3)),traceK=first(18,1,()),Gamma=first(19,3,(3,))[0],dGamma=first(19,3,(3,))[1],raw=out)

def spherical_reference(x,a,mass=1.):
    """DBM paper Eq12 metric/shift, transformed with an independent Jacobian."""
    x=np.array(x);r=np.linalg.norm(x);sn=np.hypot(x[0],x[1])/r;C=x[2]/r
    assert sn>0
    R=r+mass;c=np.sqrt(mass*mass-a*a);S=R*R+a*a*C*C;A=R*R+a*a
    xi1=S*A+2*a*a*mass*R*sn*sn;xi2=A*A-a*a*r*r*sn*sn
    gs=np.array([[S/r**2,0,-a*c*sn**2/r],[0,S,0],[-a*c*sn**2/r,0,sn**2*xi1/S]])
    n=x/r;J=np.array([n,(C*n-np.array([0,0,1]))/(r*sn),np.array([-x[1],x[0],0])/(r*r*sn*sn)])
    alpha=r*np.sqrt(S/xi2);bs=np.array([A*c*r/xi2,0,-a*(A-r*r)/xi2]);beta=np.linalg.solve(J,bs)
    derivative=2*R*np.sqrt(S/xi2)+A*R/np.sqrt(S*xi2)-A*np.sqrt(S)*(4*R*A-2*a*a*r*sn*sn)/(2*xi2**1.5)
    tr=c/S*derivative
    return J.T@gs@J,alpha,beta,tr

def adm(g,dg,ddg,K,dK,alpha,da,dda,beta,db):
    """Physical Cartesian ADM contractions, independent of conformal helper."""
    gi=np.linalg.inv(g);dgi=-np.einsum('ia,pab,bj->pij',gi,dg,gi)
    low=np.empty((3,3,3));dlow=np.empty((3,3,3,3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                low[k,i,j]=(dg[i,k,j]+dg[j,k,i]-dg[k,i,j])/2
                for p in range(3):dlow[p,k,i,j]=(ddg[p,i,k,j]+ddg[p,j,k,i]-ddg[p,k,i,j])/2
    conn=np.einsum('kl,lij->kij',gi,low)
    dconn=np.einsum('pkl,lij->pkij',dgi,low)+np.einsum('kl,plij->pkij',gi,dlow)
    ric=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            ric[i,j]=sum(dconn[k,k,i,j]-dconn[j,k,i,k] for k in range(3))
            ric[i,j]+=sum(conn[k,i,j]*conn[l,k,l]-conn[l,i,k]*conn[k,j,l] for k in range(3) for l in range(3))
    mix=gi@K;tr=np.trace(mix);dmix=np.einsum('pjk,ki->pji',dgi,K)+np.einsum('jk,pki->pji',gi,dK)
    mom=np.array([sum(dmix[j,j,i] for j in range(3))-np.trace(dmix[i])+
                  sum(conn[j,j,l]*mix[l,i]-conn[l,j,i]*mix[j,l] for j in range(3) for l in range(3)) for i in range(3)])
    ham=np.sum(gi*ric)+tr*tr-np.trace(mix@mix)
    lieg=np.einsum('k,kij->ij',beta,dg)+np.einsum('ik,kj->ij',db,g)+np.einsum('jk,ik->ij',db,g)
    lieK=np.einsum('k,kij->ij',beta,dK)+np.einsum('ik,kj->ij',db,K)+np.einsum('jk,ik->ij',db,K)
    rhs_g=lieg-2*alpha*K
    rhs_K=-dda+np.einsum('kij,k->ij',conn,da)+alpha*(ric+tr*K-2*K@gi@K)+lieK
    norm=lambda t:float(np.sqrt(max(np.einsum('ia,jb,ij,ab->',gi,gi,t,t),0)))
    return dict(H=float(ham),M_norm=float(np.sqrt(max(mom@gi@mom,0))),K_trace=float(tr),
                metric_rhs_norm=norm(rhs_g),curvature_rhs_norm=norm(rhs_K),
                rhs_metric=rhs_g,rhs_curvature=rhs_K)

def analytic_constraints(q):
    return adm(*q['g'],q['K'][0],q['K'][1],*q['alpha'],q['beta'][0],q['beta'][1])

def finite_difference(provider,x,spin,step,gauge=0):
    """Independent sixth-order Cartesian value-only derivatives; mixed D1D1."""
    x=np.array(x);cache={}
    def get(shift):
        key=tuple(shift)
        if key not in cache:cache[key]=provider(x+step*np.array(shift),spin,gauge)
        return cache[key]
    zero=get((0,0,0));d={k:np.zeros((3,)+zero[k][0].shape) for k in ['g','K','alpha','beta']};dd={k:np.zeros((3,3)+zero[k][0].shape) for k in ['g','alpha']}
    w1={-3:-1/60,-2:3/20,-1:-3/4,1:3/4,2:-3/20,3:1/60}
    w2={-3:1/90,-2:-3/20,-1:3/2,0:-49/18,1:3/2,2:-3/20,3:1/90}
    dgtinv=np.zeros((3,3,3));dGamma=np.zeros((3,3))
    for a in range(3):
        for k,w in w1.items():
            v=[0]*3;v[a]=k;q=get(v)
            for name in d:d[name][a]+=w*q[name][0]/step
            dgtinv[a]+=w*np.linalg.inv(q['gt'][0])/step
            dGamma[a]+=w*q['Gamma']/step
        for k,w in w2.items():
            v=[0]*3;v[a]=k;q=get(v)
            for name in dd:dd[name][a,a]+=w*q[name][0]/step**2
        for b in range(a):
            for i,wi in w1.items():
                for j,wj in w1.items():
                    v=[0]*3;v[a]=i;v[b]=j;q=get(v)
                    for name in dd:dd[name][a,b]+=wi*wj*q[name][0]/step**2
            for name in dd:dd[name][b,a]=dd[name][a,b]
    result=adm(zero['g'][0],d['g'],dd['g'],zero['K'][0],d['K'],zero['alpha'][0],d['alpha'],dd['alpha'],zero['beta'][0],d['beta'])
    result['Gamma_error']=float(np.max(abs(-np.einsum('jij->i',dgtinv)-zero['Gamma'])))
    result['Gamma_gradient_error']=float(np.max(abs(dGamma-zero['dGamma'])))
    result['K_first_error']=float(np.max(abs(d['K']-zero['K'][1]))/max(np.max(abs(zero['K'][1])),1))
    result['lapse_first_error']=float(np.max(abs(d['alpha']-zero['alpha'][1]))/max(np.max(abs(zero['alpha'][1])),1))
    result['lapse_second_error']=float(np.max(abs(dd['alpha']-zero['alpha'][2]))/max(np.max(abs(zero['alpha'][2])),1))
    result['shift_first_error']=float(np.max(abs(d['beta']-zero['beta'][1]))/max(np.max(abs(zero['beta'][1])),1))
    result['metric_first_error']=float(np.max(abs(d['g']-zero['g'][1]))/max(np.max(abs(zero['g'][1])),1))
    result['metric_second_error']=float(np.max(abs(dd['g']-zero['g'][2]))/max(np.max(abs(zero['g'][2])),1))
    return {k:v for k,v in result.items() if not k.startswith('rhs_')}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cxx',default=os.environ.get('CXX','c++'));args=p.parse_args()
    start=time.monotonic();dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[3];library=dest/'trumpet_provider.so'
    command=[args.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC','-I'+str(source/'src'),str(Path(__file__).with_name('driver.cpp')),'-o',str(library)]
    subprocess.run(command,check=True);provider=Provider(library)
    directions=[np.array([.3,.4,np.sqrt(.75)]),np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,0,-1.]),np.array([1e-8,0,np.sqrt(1-1e-16)])]
    points=[]
    for spin in [0.,.5,.9,-.9]:
        for radius in [.03125,.0625,.125,.25,np.sqrt(1-spin*spin),1.,2.,8.,64.,2048.]:
            for direction in directions:
                x=radius*direction;q=provider(x,spin);z=analytic_constraints(q)
                g=q['g'][0];gi=np.linalg.inv(g);gt=q['gt'][0];gtinv=np.linalg.inv(gt)
                C=np.array([sum(gtinv[i,l]*gtinv[j,k]*(q['gt'][1][j,l,k]+q['gt'][1][k,l,j]-q['gt'][1][l,j,k])/2 for j in range(3) for k in range(3) for l in range(3)) for i in range(3)])
                r=np.linalg.norm(x);R=r+1;S=R*R+spin*spin*(x[2]/r)**2;X=(R*R+spin*spin)**2-spin*spin*(x[0]**2+x[1]**2)
                row=dict(spin=spin,radius=radius,xyz=x.tolist(),alpha=float(q['alpha'][0]),
                    minimum_metric_eigenvalue=float(np.linalg.eigvalsh(g).min()),
                    conformal_det_error=float(abs(np.linalg.det(gt)-1)),A_trace=float(np.sum(gtinv*q['At'][0])),
                    Gamma_connection_error=float(np.max(abs(C-q['Gamma']))),det_relative_error=float(abs(np.linalg.det(g)/(S*X/r**6)-1)),
                    diagnostics={k:v for k,v in z.items() if not k.startswith('rhs_')})
                if np.hypot(*x[:2])>1e-6*r:
                    gr,al,be,tr=spherical_reference(x,spin)
                    row['spherical_metric_error']=float(np.linalg.norm(gr-g)/np.linalg.norm(g))
                    row['spherical_alpha_error']=float(abs(al-q['alpha'][0]))
                    row['spherical_shift_error']=float(np.linalg.norm(be-q['beta'][0]))
                    row['spherical_K_trace_error']=float(abs(tr-q['traceK'][0]))
                if spin==0:
                    n=x/r;refg=(R/r)**2*np.eye(3);refK=(np.eye(3)-2*np.outer(n,n))/r**2
                    row['Schwarzschild_error']=max(float(np.linalg.norm(g-refg)/np.linalg.norm(refg)),float(np.linalg.norm(q['K'][0]-refK)/np.linalg.norm(refK)),float(abs(q['alpha'][0]-r/R)),float(np.linalg.norm(q['beta'][0]-x/R**2)))
                points.append(row)
    horizons=[]
    for spin in [0.,.5,.9,-.9]:
        for angle in np.linspace(0,np.pi,17):
            n=np.array([np.sin(angle),0,np.cos(angle)]);x=np.sqrt(1-spin*spin)*n;q=provider(x,spin)
            speed=-q['beta'][0]@n+q['alpha'][0]*np.sqrt(n@np.linalg.inv(q['g'][0])@n)
            horizons.append(float(abs(speed)))
    convergence=[]
    for spin,radius,direction in [(.9,.125,directions[0]),(.9,.435889894,directions[0]),(.9,2.,directions[0]),(.9,.435889894,directions[2]),(0.,1.,directions[0])]:
        x=radius*direction
        rows=[dict(step_over_radius=h,**finite_difference(provider,x,spin,radius*h)) for h in [.08,.04,.02,.01]]
        for name in ['H','M_norm','metric_rhs_norm','curvature_rhs_norm','Gamma_error','Gamma_gradient_error','metric_first_error','metric_second_error','K_first_error','lapse_first_error','lapse_second_error','shift_first_error']:
            for i in range(1,len(rows)):
                rows[i][name+'_order']=float(np.log2(abs(rows[i-1][name])/max(abs(rows[i][name]),1e-300))) if rows[i-1][name] else None
        convergence.append(dict(spin=spin,xyz=x.tolist(),radius=radius,rows=rows))
    scaling=[];q1=provider([.19,.27,.31],.9);q2=provider([.38,.54,.62],1.8,mass=2.)
    for name,dimension in [('g',0),('K',-1),('alpha',0),('beta',0),('chi',0),('gt',0),('At',-1),('traceK',-1)]:
        for degree in range(len(q1[name])):
            expected=q1[name][degree]*2**(dimension-degree)
            scaling.append(float(np.max(abs(q2[name][degree]-expected))/max(np.max(abs(expected)),1)))
    parity=[];P=np.diag([1.,-1.,1.]);x=np.array([.19,.27,.31]);qp=provider(x,.9);qm=provider(P@x,-.9)
    for name in ['g','K','gt','At']:
        parity.append(float(np.max(abs(qp[name][0]-P@qm[name][0]@P))/max(np.max(abs(qp[name][0])),1)))
    parity.append(float(np.max(abs(qp['beta'][0]-P@qm['beta'][0]))))
    mu,weights=np.polynomial.legendre.leggauss(32);charges=[]
    for radius in [10.,20.,40.,80.,160.,320.]:
        mass_integral=angular=0.
        for C,weight in zip(mu,weights):
            n=np.array([np.sqrt(1-C*C),0,C]);x=radius*n;q=provider(x,.9);g,dg,_=q['g'];gi=np.linalg.inv(g);K=q['K'][0]
            flux=np.array([sum(dg[j,i,j]-dg[i,j,j] for j in range(3)) for i in range(3)])
            mass_integral+=weight*radius**2*np.dot(flux,n)/8
            angular+=weight*radius**2*np.sqrt(np.linalg.det(g))*np.array([-x[1],x[0],0])@(K-np.trace(gi@K)*g)@gi@n/4
        charges.append(dict(radius=radius,ADM_mass_surface_integral=float(mass_integral),angular_momentum=float(angular)))
    mass_limit=float(np.polyfit([1/q['radius'] for q in charges[-4:]],[q['ADM_mass_surface_integral'] for q in charges[-4:]],3)[-1])
    bad=[]
    for mass,spin,x in [(1,1,[1,0,0]),(1,-1,[1,0,0]),(1,1.01,[1,0,0]),(0,0,[1,0,0]),(1,.9,[0,0,0]),(1,.9,[np.nan,0,1]),(np.inf,.9,[1,0,0])]:
        try:provider(x,spin,mass=mass)
        except ValueError as e:bad.append(int(e.args[0]))
        else:raise AssertionError('Unsupported geometry accepted')
    keys=['H','M_norm','metric_rhs_norm','curvature_rhs_norm']
    extrema={name:max(abs(row['diagnostics'][name]) for row in points) for name in keys}
    for name in ['conformal_det_error','A_trace','Gamma_connection_error','det_relative_error','spherical_metric_error','spherical_alpha_error','spherical_shift_error','spherical_K_trace_error','Schwarzschild_error']:
        extrema[name]=max(abs(row.get(name,0)) for row in points)
    extrema.update(minimum_metric_eigenvalue=min(row['minimum_metric_eigenvalue'] for row in points),min_alpha=min(row['alpha'] for row in points),max_alpha=max(row['alpha'] for row in points),max_horizon_outgoing_speed=max(horizons),mass_scaling_error=max(scaling),spin_reflection_error=max(parity))
    # Test sixth-order truncation on spinning off-axis samples. At spin zero or
    # an exact symmetry axis some fields are analytic zeros, so roundoff-level
    # residuals have no meaningful fitted convergence order.
    # The finest H stencil subtracts metric terms and can be limited by
    # cancellation near the puncture. Require sixth order before that floor,
    # continued decrease, and an absolute fine-grid error bound separately.
    convergence_passed=all(case['rows'][-2][name+'_order']>5.7 for case in convergence[:3] for name in ['H','M_norm','curvature_rhs_norm','metric_first_error','metric_second_error','K_first_error','lapse_first_error','lapse_second_error','shift_first_error'])
    convergence_passed &= all(abs(case['rows'][-1]['H'])<2e-10 and abs(case['rows'][-1]['H'])<abs(case['rows'][-2]['H'])/8 for case in convergence[:3])
    convergence_passed &= all(case['rows'][-1][name]<2e-8 for case in convergence[:3] for name in ['metric_first_error','metric_second_error','K_first_error','lapse_first_error','lapse_second_error','shift_first_error'])
    passed=convergence_passed and extrema['H']<1e-9 and extrema['M_norm']<1e-9 and extrema['metric_rhs_norm']<1e-11 and extrema['curvature_rhs_norm']<1e-9
    passed &= extrema['minimum_metric_eigenvalue']>0 and extrema['min_alpha']>0 and extrema['max_alpha']<1
    passed &= all(extrema[k]<1e-11 for k in ['conformal_det_error','A_trace','Gamma_connection_error','det_relative_error','spherical_metric_error','spherical_alpha_error','spherical_shift_error','spherical_K_trace_error','Schwarzschild_error','max_horizon_outgoing_speed','mass_scaling_error','spin_reflection_error'])
    passed &= max(abs(q['angular_momentum']-.9) for q in charges)<1e-10 and abs(mass_limit-1)<1e-6
    report=dict(scope='Standalone compiled double-precision geometry/derivative checks; not evolution, GPU/MPI/AMR or perturbation stability.',source='https://arxiv.org/pdf/1409.1887v2',spin_candidate=.9,R0_over_M=1.,checks_passed=bool(passed),convergence_checks_passed=bool(convergence_passed),compile_command=command,wall_seconds=time.monotonic()-start,summary=extrema,points=points,convergence=convergence,invalid_statuses=bad,outer_charge_integrals=charges,ADM_mass_extrapolated_cubic_in_inverse_radius=mass_limit,
        hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source/'src/coordinates/kerr_trumpet.hpp',Path(__file__).with_name('driver.cpp'),Path(__file__),library]})
    (dest/'geometry-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['points','convergence','hashes']},indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
