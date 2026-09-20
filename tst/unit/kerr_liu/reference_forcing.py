#!/usr/bin/env python3
"""Point audit of continuum constraint forcing from a frozen Liu reference.

Uses the compiled geometry provider. No evolution is run. The source is frozen
in conformal Z4c variables (chi,gtilde,Khat,Atilde,Gamma), not in ADM variables.
"""
from pathlib import Path
import argparse,json
import numpy as np
from validate import Provider,analytic_constraints

def scalar_product(a,b):
    av,ad,add=a;bv,bd,bdd=b;shape=(1,)*np.ndim(bv)
    ad=ad.reshape((3,)+shape);add=add.reshape((3,3)+shape)
    return av*bv,ad*bv+av*bd,add*bv+av*bdd+ad[:,None]*bd[None,:]+ad[None,:]*bd[:,None]
def scale(a,x):return tuple(x*v for v in a)
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def inverse_scalar(x):
    v,d,dd=x;return 1/v,-d/v**2,2*d[:,None]*d[None,:]/v**3-dd/v**2

def constraints(g,dg,ddg,K,dK):
    """Analytic algebra; complex inputs retain the exact first variation."""
    gi=np.linalg.inv(g);dgi=-np.einsum('ia,pab,bj->pij',gi,dg,gi)
    low=np.empty((3,3,3),dtype=g.dtype);dlow=np.empty((3,3,3,3),dtype=g.dtype)
    for k in range(3):
      for i in range(3):
       for j in range(3):
        low[k,i,j]=(dg[i,k,j]+dg[j,k,i]-dg[k,i,j])/2
        for p in range(3):dlow[p,k,i,j]=(ddg[p,i,k,j]+ddg[p,j,k,i]-ddg[p,k,i,j])/2
    G=np.einsum('kl,lij->kij',gi,low);dG=np.einsum('pkl,lij->pkij',dgi,low)+np.einsum('kl,plij->pkij',gi,dlow)
    Ric=np.zeros((3,3),dtype=g.dtype)
    for i in range(3):
      for j in range(3):
        Ric[i,j]=sum(dG[k,k,i,j]-dG[j,k,i,k] for k in range(3))+sum(G[k,i,j]*G[l,k,l]-G[l,i,k]*G[k,j,l] for k in range(3) for l in range(3))
    T=gi@K;dT=np.einsum('pjk,ki->pji',dgi,K)+np.einsum('jk,pki->pji',gi,dK)
    H=np.sum(gi*Ric)+np.trace(T)**2-np.trace(T@T)
    M=np.array([sum(dT[j,j,i] for j in range(3))-np.trace(dT[i])+sum(G[j,j,l]*T[l,i]-G[l,j,i]*T[j,l] for j in range(3) for l in range(3)) for i in range(3)])
    return H,M

def gamma_variation(q,Fgt):
    gi=np.linalg.inv(q['gt'][0]);dgi=-np.einsum('ia,pab,bj->pij',gi,q['gt'][1],gi)
    # delta[-d_j(gt^{-1})^{ij}] = d_j[(gt^{-1}) Fgt (gt^{-1})]^{ij}.
    dB=np.einsum('pik,kl,lj->pij',dgi,Fgt[0],gi)+np.einsum('ik,pkl,lj->pij',gi,Fgt[1],gi)+np.einsum('ik,kl,plj->pij',gi,Fgt[0],dgi)
    return np.einsum('jij->i',dB)

def frozen_source(provider,x,step_fraction):
    q=provider(x,.9,0);Fgt=scale(scalar_product(q['chi'],scalar_product(q['alpha'],q['K'])),-2)
    def momentum_sources(x):
        p=provider(x,.9,0);rhs=analytic_constraints(p);gi=np.linalg.inv(p['g'][0])
        FK=float(np.trace(gi@rhs['rhs_curvature'])-np.trace(gi@rhs['rhs_metric']@gi@p['K'][0]))
        FA=p['chi'][0]*(rhs['rhs_curvature']-p['g'][0]*FK/3)
        return FK,FA
    fk,fa=momentum_sources(x);dfk=np.zeros(3);dfa=np.zeros((3,3,3));step=np.linalg.norm(x)*step_fraction
    weights={-3:-1/60,-2:3/20,-1:-3/4,1:3/4,2:-3/20,3:1/60}
    for axis in range(3):
        for k,w in weights.items():
            xx=np.array(x);xx[axis]+=k*step;dk,da=momentum_sources(xx);dfk[axis]+=w*dk/step;dfa[axis]+=w*da/step
    return dict(Fgt=Fgt,Fk=(fk,dfk,np.zeros((3,3))),FA=(fa,dfa,np.zeros((3,3,3,3))),FGamma=gamma_variation(q,Fgt))

def constraint_source(provider,x,mass,F,project=False):
    # This is a constraint-satisfying mass family with fixed dimensionless a/M=.9.
    q=provider(x,.9*mass,0,mass);invchi=inverse_scalar(q['chi'])
    # Fchi=0, Theta=0 and K=0 on every member of this exact maximal family.
    dg=scalar_product(invchi,F['Fgt'])
    FA=F['FA']
    if project:
        gt,dgt,_=q['gt'];At,dAt,_=q['At'];gi=np.linalg.inv(gt)
        dgi=-np.einsum('ia,pab,bj->pij',gi,dgt,gi)
        Aup=gi@At@gi
        dAup=np.einsum('pik,kl,lj->pij',dgi,At,gi)+np.einsum('ik,pkl,lj->pij',gi,dAt,gi)+np.einsum('ik,kl,plj->pij',gi,At,dgi)
        defect=np.sum(gi*FA[0])-np.sum(Aup*F['Fgt'][0])
        ddefect=np.einsum('pij,ij->p',dgi,FA[0])+np.einsum('ij,pij->p',gi,FA[1])-np.einsum('pij,ij->p',dAup,F['Fgt'][0])-np.einsum('ij,pij->p',Aup,F['Fgt'][1])
        # For this axisymmetric mass family Fgt is azimuthal off-diagonal while
        # gt is polar-block diagonal, so metric tangent projection is exactly
        # inactive. A's trace projection generally is active away from bg.
        assert abs(np.sum(gi*F['Fgt'][0]))<1e-10
        FA=(FA[0]-gt*defect/3,FA[1]-(dgt*defect+ddefect[:,None,None]*gt)/3,FA[2])
    dK=scalar_product(invchi,add(FA,scale(scalar_product(F['Fk'],q['gt']),1/3)))
    eps=1e-20;g=[v+1j*eps*f for v,f in zip(q['g'],dg)];K=q['K'][0]+1j*eps*dK[0];DK=q['K'][1]+1j*eps*dK[1]
    H,M=constraints(*g,K,DK)
    Q=F['FGamma']-gamma_variation(q,F['Fgt'])
    base=constraints(*q['g'],q['K'][0],q['K'][1])
    return np.r_[H.imag/eps,M.imag/eps,Q],dict(H=float(base[0]),M=list(map(float,base[1])))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve();provider=Provider(root/'liu_provider.so')
    c=(1+np.sqrt(1-.9**2))/4;n=np.array([.3,.4,np.sqrt(.75)]);rows=[]
    for factor in [.5,1.,2.,10.]:
      for project in [False,True]:
        x=c*factor*n;checks=[]
        for step in [.004,.002]:
            F=frozen_source(provider,x,step);at_bg,_=constraint_source(provider,x,1.,F,project)
            for eps in [1e-3,3e-4,1e-4]:
                cp,bp=constraint_source(provider,x,1+eps,F,project);cm,bm=constraint_source(provider,x,1-eps,F,project)
                # Physical evolution preserves constraints; the subtracted source
                # changes their linear propagation by -C''[delta U,F_background].
                derivative=-(cp-cm)/(2*eps);gi=np.linalg.inv(provider(x,.9)['g'][0]);ch=provider(x,.9)['chi'][0]
                checks.append(dict(source_derivative_step_fraction=step,mass_step=eps,
                    background_constraint_tangent_max=float(np.max(abs(at_bg))),
                    mass_family_max_abs_H=max(abs(bp['H']),abs(bm['H'])),
                    H_source_per_delta_mass=float(derivative[0]),
                    M_source_covector_per_delta_mass=derivative[1:4].tolist(),
                    M_source_physical_norm_per_delta_mass=float(np.sqrt(derivative[1:4]@gi@derivative[1:4])),
                    Q_source_vector_per_delta_mass=derivative[4:].tolist(),
                    Q_source_Euclidean_norm_per_delta_mass=float(np.linalg.norm(derivative[4:]))))
        # The two finest independent mass steps and source-gradient stencils agree.
        finest=checks[-1];previous=checks[-2];coarse=checks[2]
        keys=['H_source_per_delta_mass','M_source_physical_norm_per_delta_mass','Q_source_Euclidean_norm_per_delta_mass']
        absolute={k:max(abs(finest[k]-previous[k]),abs(finest[k]-coarse[k])) for k in keys}
        errors={k:absolute[k]/abs(finest[k]) if abs(finest[k])>1e-8 else None for k in keys}
        assert all(absolute[k]<1e-8 if errors[k] is None else errors[k]<2e-5 for k in keys),(factor,errors)
        assert max(q['background_constraint_tangent_max'] for q in checks)<1e-8
        rows.append(dict(r_over_throat=factor,algebraic_tangent_projection=project,xyz=x.tolist(),checks=checks,max_absolute_calibration_errors=absolute,max_relative_calibration_errors=errors))
    out=dict(scope='Continuum point test, not a grid evolution; fixed source in conformal Z4c geometric variables, positive precollapsed lapse and zero shift.',
             formula='Without projection: delta C_t(modified)-delta C_t(physical) = -C_double_prime[delta_U,F(U_background)]. With algebraic projection P(U), differentiate -C_prime[U] P(U) F(U_background), including P_prime.',
             variation='d/dM of exact Liu family at M=1 with a/M=.9 and fixed Cartesian coordinates',
             caveat='Background is an exact algebraic fixed point after subtraction, but neighboring physical constraint-satisfying mass data acquire M/Q sources. H is symmetry-zero in this direction. This is a property to measure in the authorized reference-forced numerical model, not a reason to conflate it with physical vacuum evolution.',
             checks_passed=True,points=rows)
    (root/'reference-forcing-results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'checks_passed':True,'points':[{'r_over_throat':q['r_over_throat'],'algebraic_tangent_projection':q['algebraic_tangent_projection'],**{k:v for k,v in q['checks'][-1].items() if 'source' in k},'calibration':q['max_relative_calibration_errors']} for q in rows]},indent=2))

if __name__=='__main__':main()
