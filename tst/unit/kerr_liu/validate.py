#!/usr/bin/env python3
"""Compile/check Liu ADM jets independently; no evolution or external jobs.

Requires a C++17 compiler and Python NumPy. All generated evidence goes to
--output (required); no built artifact is placed in the source tree.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,sys,time
import numpy as np

class Provider:
    def __init__(self,library):
        self.lib=ctypes.CDLL(str(library));self.fn=self.lib.liu_eval
        self.fn.argtypes=[ctypes.c_double,ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64,shape=(3,),flags='C_CONTIGUOUS'),ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64,shape=(571,),flags='C_CONTIGUOUS')]
        self.fn.restype=ctypes.c_int
    def __call__(self,x,spin=.9,gauge=0,mass=1.):
        out=np.empty(571);status=self.fn(mass,spin,np.ascontiguousarray(x,dtype=float),gauge,out)
        if status:raise ValueError(status)
        j=out[:559].reshape(43,13)
        def field(start,count,shape):
            q=j[start:start+count];return q[:,0].reshape(shape),q[:,1:4].T.reshape((3,)+shape),q[:,4:].T.reshape((3,3)+shape)
        return dict(g=field(0,9,(3,3)),K=field(9,9,(3,3)),alpha=field(18,1,()),
                    beta=field(19,3,(3,)),chi=field(22,1,()),gt=field(23,9,(3,3)),
                    At=field(32,9,(3,3)),traceK=field(41,1,()),signed=field(42,1,()),
                    Gamma=out[559:].reshape(3,4)[:,0].copy(),dGamma=out[559:].reshape(3,4)[:,1:].T.copy(),raw=out)

def spherical_reference(x,spin,mass=1.):
    """Paper's spherical components transformed by a separate coordinate Jacobian."""
    x=np.array(x);r=np.linalg.norm(x);rho=np.hypot(x[0],x[1]);assert rho>0
    C=x[2]/r;sn=rho/r;rp=mass+np.sqrt(mass*mass-spin*spin);rm=2*mass-rp;c=rp/4
    R=(r+c)**2/r;Sigma=R*R+spin*spin*C*C;Delta=(R-rp)*(R-rm)
    bigA=(R*R+spin*spin)**2-Delta*spin*spin*sn*sn
    gs=np.diag([Sigma*(r+c)**2/(r**3*(R-rm)),Sigma,bigA*sn*sn/Sigma])
    P=3*R**4+2*spin*spin*R*R-spin**4-spin*spin*(R*R-spin*spin)*sn*sn
    kr=mass*spin*sn*sn*P/(Sigma*np.sqrt(bigA*Sigma))*(1+c/r)/np.sqrt(r*(R-rm))
    kt=-2*mass*spin**3*R*C*sn**3/(Sigma*np.sqrt(bigA*Sigma))*(r-c)*np.sqrt((R-rm)/r)
    ks=np.array([[0,0,kr],[0,0,kt],[kr,kt,0.]])
    n=x/r;J=np.array([n,(C*n-np.array([0,0,1]))/(r*sn),np.array([-x[1],x[0],0])/rho**2])
    return J.T@gs@J,J.T@ks@J

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
    result['metric_first_error']=float(np.max(abs(d['g']-zero['g'][1]))/max(np.max(abs(zero['g'][1])),1))
    result['metric_second_error']=float(np.max(abs(dd['g']-zero['g'][2]))/max(np.max(abs(zero['g'][2])),1))
    return {k:v for k,v in result.items() if not k.startswith('rhs_')}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cxx',default=os.environ.get('CXX','c++'));args=p.parse_args()
    start=time.monotonic();dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[3];library=dest/'liu_provider.so'
    command=[args.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC','-I'+str(source/'src'),str(Path(__file__).with_name('driver.cpp')),'-o',str(library)]
    subprocess.run(command,check=True);provider=Provider(library)
    points=[];directions=[np.array([.3,.4,np.sqrt(.75)]),np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,0,-1.]),np.array([1e-8,0,np.sqrt(1-1e-16)])]
    for spin in [0.,.5,.9]:
        c=(1+np.sqrt(1-spin*spin))/4
        for factor in [.1,.5,.99,1.,1.01,2.,10.]:
            for direction in directions:
                x=c*factor*direction;gauges={};q=provider(x,spin)
                g=q['g'][0];gi=np.linalg.inv(g)
                for mode,name in [(0,'precollapsed_zero_shift'),(1,'signed_stationary')]:
                    f=provider(x,spin,mode);z=analytic_constraints(f)
                    gauges[name]={k:v for k,v in z.items() if not k.startswith('rhs_')};gauges[name]['alpha']=float(f['alpha'][0])
                gt=q['gt'][0];gtinv=np.linalg.inv(gt)
                C=np.array([sum(gtinv[i,l]*gtinv[j,k]*(q['gt'][1][j,l,k]+q['gt'][1][k,l,j]-q['gt'][1][l,j,k])/2 for j in range(3) for k in range(3) for l in range(3)) for i in range(3)])
                # Gamma=-d_j gt^{ij} equals contracted Christoffel when det(gt)=1.
                errorGamma=float(np.max(abs(C-q['Gamma'])))
                row=dict(spin=spin,radius=float(np.linalg.norm(x)),r_over_throat=factor,xyz=x.tolist(),
                         minimum_metric_eigenvalue=float(np.linalg.eigvalsh(g).min()),
                         conformal_det_error=float(abs(np.linalg.det(gt)-1)),
                         A_trace=float(np.sum(gtinv*q['At'][0])),Gamma_connection_error=errorGamma,gauges=gauges)
                if np.hypot(*x[:2])>1e-6*c:
                    gr,kr=spherical_reference(x,spin)
                    row['spherical_metric_relative_error']=float(np.linalg.norm(gr-g)/np.linalg.norm(g))
                    row['spherical_K_relative_error']=float(np.linalg.norm(kr-q['K'][0])/max(np.linalg.norm(kr),1))
                if spin==0:
                    psi=1+.5/np.linalg.norm(x)
                    row['schwarzschild_metric_relative_error']=float(np.max(abs(g/psi**4-np.eye(3))))
                    row['schwarzschild_chi_relative_error']=float(abs(q['chi'][0]*psi**4-1))
                points.append(row)
    convergence=[]
    for spin,factor,direction in [(.5,.5,directions[0]),(.5,1.,directions[0]),(.9,.5,directions[0]),(.9,1.,directions[0]),(.9,2.,directions[0]),(.9,1.,directions[2])]:
        c=(1+np.sqrt(1-spin*spin))/4;x=c*factor*direction
        rows=[dict(step_over_radius=h,**finite_difference(provider,x,spin,c*factor*h)) for h in [.08,.04,.02,.01]]
        for name in ['H','M_norm','Gamma_error','Gamma_gradient_error','metric_first_error','metric_second_error']:
            for i in range(1,len(rows)):
                rows[i][name+'_order']=float(np.log2(abs(rows[i-1][name])/max(abs(rows[i][name]),1e-300))) if rows[i-1][name] else None
        convergence.append(dict(spin=spin,xyz=x.tolist(),r_over_throat=factor,rows=rows))
    parity=[]
    for spin in [.5,.9]:
        q=provider([.19,.27,.31],spin,1);n=provider([.19,.27,.31],-spin,1)
        parity.append(dict(spin=spin,metric=float(np.max(abs(q['g'][0]-n['g'][0]))),K=float(np.max(abs(q['K'][0]+n['K'][0]))),shift=float(np.max(abs(q['beta'][0]+n['beta'][0])))))
    # Dimensionful mass/spin rescaling and asymptotic charges are independent
    # normalization checks; finite-radius ADM mass is not expected exactly1.
    q1=provider([.19,.27,.31],.9);q2=provider([.38,.54,.62],1.8,mass=2.)
    scale_errors=[]
    for key,dimension in [('g',0),('K',-1),('alpha',0),('beta',0),('chi',0),('gt',0),('At',-1)]:
        for degree in range(3):
            expected=q1[key][degree]*2**(dimension-degree)
            scale_errors.append(float(np.max(abs(q2[key][degree]-expected))/max(np.max(abs(expected)),1)))
    mu,weights=np.polynomial.legendre.leggauss(24);charges=[]
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
    for mass,spin,x,gauge in [(1,1,[1,0,0],0),(1,1.01,[1,0,0],0),(0,0,[1,0,0],0),(1,.9,[0,0,0],0),(1,.9,[np.nan,0,1],0),(1,.9,[1,0,0],7)]:
        try:provider(x,spin,gauge,mass)
        except ValueError as e:bad.append(int(e.args[0]))
        else:raise AssertionError('Unsupported geometry accepted')
    extrema={name:max(abs(row['gauges']['signed_stationary'][key]) for row in points) for name,key in [('max_analytic_H','H'),('max_analytic_M_norm','M_norm'),('max_signed_metric_rhs_norm','metric_rhs_norm'),('max_signed_curvature_rhs_norm','curvature_rhs_norm')]}
    extrema.update(max_conformal_det_error=max(x['conformal_det_error'] for x in points),max_Gamma_connection_error=max(x['Gamma_connection_error'] for x in points),minimum_metric_eigenvalue=min(x['minimum_metric_eigenvalue'] for x in points),
                   max_spherical_metric_relative_error=max(x.get('spherical_metric_relative_error',0) for x in points),
                   max_spherical_K_relative_error=max(x.get('spherical_K_relative_error',0) for x in points),
                   max_Schwarzschild_relative_error=max(max(x.get('schwarzschild_metric_relative_error',0),x.get('schwarzschild_chi_relative_error',0)) for x in points),
                   max_A_trace=max(abs(x['A_trace']) for x in points))
    # Axis momentum is identically zero and already at roundoff: do not claim a
    # convergence order for a symmetry-zero quantity. The final refinement pair
    # tests the asymptotic regime, after coarse cancellation in axis H.
    convergence_passed=all(p['rows'][-1][name+'_order']>5.4 for p in convergence for name in ['H','Gamma_error','Gamma_gradient_error','metric_first_error','metric_second_error'])
    convergence_passed &= all(p['rows'][-1]['M_norm_order']>5.4 for p in convergence if np.hypot(*p['xyz'][:2])>0)
    passed=extrema['max_analytic_H']<1e-9 and extrema['max_analytic_M_norm']<1e-10 and extrema['max_signed_metric_rhs_norm']<1e-10 and extrema['max_signed_curvature_rhs_norm']<1e-9 and extrema['max_conformal_det_error']<1e-12 and extrema['max_Gamma_connection_error']<1e-10 and extrema['minimum_metric_eigenvalue']>0
    passed &= convergence_passed and extrema['max_spherical_metric_relative_error']<1e-12 and extrema['max_spherical_K_relative_error']<1e-12 and extrema['max_Schwarzschild_relative_error']<1e-12 and extrema['max_A_trace']<1e-12
    passed &= all(p['metric']==p['K']==p['shift']==0 for p in parity)
    passed &= max(scale_errors)<1e-12 and max(abs(q['angular_momentum']-.9) for q in charges)<1e-12 and abs(mass_limit-1)<1e-6
    report=dict(scope='Standalone compiled double-precision geometry/derivative checks; no evolution, GPU/MPI/AMR or stability claim.',source='https://arxiv.org/pdf/1001.4077',spin_candidate=.9,
                checks_passed=bool(passed),convergence_checks_passed=bool(convergence_passed),compile_command=command,wall_seconds=time.monotonic()-start,summary=extrema,points=points,convergence=convergence,spin_parity=parity,invalid_statuses=bad,
                mass_rescaling_max_error=max(scale_errors),outer_charge_integrals=charges,ADM_mass_extrapolated_cubic_in_inverse_radius=mass_limit,
                hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source/'src/coordinates/kerr_liu.hpp',Path(__file__).with_name('driver.cpp'),Path(__file__),library]})
    (dest/'geometry-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['points','convergence','hashes']},indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
