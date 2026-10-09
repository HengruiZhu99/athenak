"""Common-rho collocation, exact dense mass, scalar wave energy model only."""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
import numpy as np
import scipy
from scipy.special import eval_jacobi, roots_jacobi

warnings.filterwarnings('error',category=RuntimeWarning)
P=Path(__file__).resolve().parent
plan=json.loads((P/'plan.json').read_text())
T=plan['tolerances'];started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mm(a,b):return np.einsum('ik,kj->ij',a,b,optimize=False)
def mv(a,b):return np.einsum('ij,j->i',a,b,optimize=False)
def dot(a,b):return float(np.einsum('i,i->',a,b,optimize=False))
def error(res,*terms):return {'absolute':float(np.linalg.norm(res)),'scaled':float(np.linalg.norm(res)/max(1,sum(np.linalg.norm(t) for t in terms))),'absolute_max':float(np.max(np.abs(res)))}
def quadrature(q,beta,B):
    x,w=roots_jacobi(q,0,beta)
    return (x+1)*B/2,w*(B/2)**(beta+1)/2
def modal(x,N,L,B):
    beta=L+.5;z=2*np.asarray(x)/B-1
    norm=np.sqrt(2*(2*np.arange(N)+beta+1)/B**(beta+1))
    v=np.column_stack([norm[n]*eval_jacobi(n,0,beta,z) for n in range(N)])
    d=np.zeros_like(v);dd=np.zeros_like(v)
    for n in range(1,N):d[:,n]=norm[n]*(n+beta+1)/B*eval_jacobi(n-1,1,beta+1,z)
    for n in range(2,N):dd[:,n]=norm[n]*(n+beta+1)*(n+beta+2)/B**2*eval_jacobi(n-2,2,beta+2,z)
    return v,d,dd
def barycentric(x):
    diff=x[:,None]-x[None,:];np.fill_diagonal(diff,1)
    w=1/np.prod(diff,axis=1);w/=np.max(np.abs(w))
    D=w[None,:]/w[:,None]/diff;np.fill_diagonal(D,0)
    np.fill_diagonal(D,-D.sum(axis=1))
    return w,D
def evaluate_lagrange(x,w,y):
    diff=np.asarray(y)[:,None]-x[None,:]
    assert not (diff==0).any(), 'quadrature unexpectedly shares a collocation node'
    a=w[None,:]/diff;return a/a.sum(axis=1)[:,None]
def gram(v,w):return np.einsum('qi,q,qj->ij',v,w,v,optimize=False)
def congruence(V,A):return np.einsum('ki,kl,lj->ij',V,A,V,optimize=False)
def integrate(N,L,rb,q,x,w,D):
    B=rb*rb;qm,wm=quadrature(q,L+.5,B);qs,ws=quadrature(q,L-.5,B)
    lm=evaluate_lagrange(x,w,qm);ls=evaluate_lagrange(x,w,qs);lds=mm(ls,D)
    M=gram(lm,wm)
    Q=L*ls+2*qs[:,None]*lds
    S=gram(Q,ws)+L*(L+1)*gram(ls,ws)
    vm,dm,ddm=modal(qm,N,L,B);vs,ds,dds=modal(qs,N,L,B)
    Mm=gram(vm,wm)
    Qm=L*vs+2*qs[:,None]*ds
    Sm=gram(Qm,ws)+L*(L+1)*gram(vs,ws)
    lap=4*qm[:,None]*ddm+(4*L+6)*dm
    Am=np.einsum('qi,q,qj->ij',vm,wm,lap,optimize=False)
    return M,S,Mm,Sm,Am

results=[];matrices={};checks_all=[]
rng=np.random.default_rng(plan['random_polynomial_seed'])
negative=[]
for rb in plan['rb_values']:
    B=rb*rb
    for N in plan['N_values']:
        x,common=quadrature(N,.5,B);w,D=barycentric(x)
        t=evaluate_lagrange(x,w,[B])[0];td=mv(D.T,t)
        for L in plan['L_values']:
            M,S,Mm,Sm,Am=integrate(N,L,rb,2*N+8,x,w,D)
            M2,S2,Mm2,Sm2,Am2=integrate(N,L,rb,3*N+17,x,w,D)
            V,Vd,Vdd=modal(x,N,L,B)
            vb,db,ddb=modal([B],N,L,B);vb=vb[0];db=db[0]
            DL=4*x[:,None]*mm(D,D)+(4*L+6)*D
            # Direct polynomial modal action: independent of IBP/S construction.
            DLm=np.linalg.solve(Mm,Am)
            DLcoll=np.linalg.solve(V,mm(DL,V))
            Db=np.linalg.solve(V.T,Vd.T).T
            boundary=rb**(2*L+1)*np.outer(t,L*t+2*B*td)
            boundarym=rb**(2*L+1)*np.outer(vb,L*vb+2*B*db)
            MD=mm(M,DL);MDm=mm(Mm,DLm)
            nodal_ibp=error(MD+S-boundary,MD,S,boundary)
            modal_ibp=error(MDm+Sm-boundarym,MDm,Sm,boundarym)
            weak=np.linalg.solve(Mm,-Sm+boundarym)
            action=error(DLm-weak,DLm,weak)
            collocation=error(DLcoll-DLm,DLcoll,DLm)
            bary=error(D-Db,D,Db)
            mass_identity=error(Mm-np.eye(N),Mm,np.eye(N))
            congruence_mass=error(congruence(V,M)-Mm,Mm)
            congruence_stiffness=error(congruence(V,S)-Sm,Sm)
            over=max(error(a-b,a,b)['scaled'] for a,b in [(M,M2),(S,S2),(Mm,Mm2),(Sm,Sm2),(Am,Am2)])
            # Same scalar wave SAT as requested, transformed by polynomial congruence.
            beta=rb**(2*L+2);q=(L/rb)*vb+2*rb*db
            lift=np.linalg.solve(Mm,beta*vb)
            J=np.zeros((2*N,2*N));J[:N,N:]=np.eye(N);J[N:,:N]=DLm-np.outer(lift,q);J[N:,N:]=-np.outer(lift,vb)
            H=np.zeros_like(J);H[:N,:N]=Sm;H[N:,N:]=Mm
            HJ=mm(H,J);loss=np.zeros_like(J);loss[N:,N:]=2*beta*np.outer(vb,vb)
            wave=error(HJ+HJ.T+loss,HJ,HJ.T,loss)
            energy=[];rate_error=0
            for _ in range(plan['random_energy_samples_per_case']):
                c=rng.normal(size=N)/(1+np.arange(N))**2;p=rng.normal(size=N)/(1+np.arange(N))**2;z=np.r_[c,p]
                E=.5*(dot(c,mv(Sm,c))+dot(p,mv(Mm,p)))
                rate=dot(mv(H,z),mv(J,z));pb=dot(vb,p);expected=-beta*pb*pb
                scale=max(1,abs(rate),abs(expected),np.linalg.norm(mv(H,z))*np.linalg.norm(mv(J,z)))
                err=abs(rate-expected)/scale;rate_error=max(rate_error,err)
                energy.append({'E':E,'actual_Edot':rate,'expected_outflow_Edot':expected,'absolute_error':abs(rate-expected),'scaled_error':err})
            constant={'applicable':L==0,'energy_is_seminorm_for_L0':L==0}
            if L==0:
                coeff=np.zeros(N);coeff[0]=1/V[0,0];z=np.r_[coeff,np.zeros(N)]
                constant.update(E=.5*dot(coeff,mv(Sm,coeff)),generator_max=float(np.max(np.abs(mv(J,z)))),DL_constant_max=float(np.max(np.abs(mv(DLm,coeff)))))
            # Origin flux of an exactly regular polynomial W=1+rho+rho².
            def flux(r):
                rho=r*r;v=1+rho+rho*rho;d=1+2*rho
                return r**(2*L+1)*v*(L*v+2*rho*d)
            origin={'at_zero':flux(0.),'samples':[{'r':r,'flux':flux(r)} for r in (.01,.001,.0001)],'leading_power':3 if L==0 else 2*L+1}
            naive=np.diag(common*x**L)
            naive_err=float(np.linalg.norm(M-naive)/np.linalg.norm(M));negative.append(naive_err)
            diag=np.sqrt(np.diag(M));scaled=M/diag[:,None]/diag[None,:]
            # Test positive modal energy through a Cholesky Gram, leaving the L0 zero column intact.
            chol=np.linalg.cholesky(Sm if L else Sm[1:,1:])
            checks={'barycentric_modal_D':bary['scaled']<=T['common_node_barycentric_vs_modal_D_scaled'],
                    'independent_overintegration':over<=T['independent_overintegration_scaled'],
                    'nodal_IBP':nodal_ibp['scaled']<=T['nodal_IBP_scaled'],
                    'modal_IBP':modal_ibp['scaled']<=T['modal_IBP_scaled'],
                    'strong_weak_polynomial_action':action['scaled']<=T['strong_weak_polynomial_action_scaled'],
                    'common_collocation_modal_action':collocation['scaled']<=T['strong_weak_polynomial_action_scaled'],
                    'wave_matrix_energy':wave['scaled']<=T['wave_energy_matrix_scaled'],
                    'sampled_wave_rate':rate_error<=T['sampled_wave_energy_rate_scaled'],
                    'modal_mass_identity':mass_identity['scaled']<=T['modal_mass_identity_scaled'],
                    'zero_origin_flux':origin['at_zero']==0,
                    'L0_static_constant':L!=0 or max(abs(constant['E']),constant['generator_max'],constant['DL_constant_max'])<=T['L0_static_constant_absolute'],
                    'nonnegative_energy':all(v['E']>=0 for v in energy)}
            row={'N':N,'L':L,'rb':rb,'common_rho_min':float(x.min()),'common_rho_max':float(x.max()),
                 'raw_mass_condition':float(np.linalg.cond(M)),'diagonally_scaled_mass_condition':float(np.linalg.cond(scaled)),
                 'modal_mass_condition':float(np.linalg.cond(Mm)),'common_node_modal_V_condition':float(np.linalg.cond(V)),
                 'nodal_IBP':nodal_ibp,'modal_IBP':modal_ibp,'strong_weak_action':action,'common_collocation_action':collocation,'barycentric_modal_D':bary,
                 'independent_overintegration_scaled':over,'modal_mass_identity':mass_identity,'mass_congruence':congruence_mass,'stiffness_congruence':congruence_stiffness,
                 'wave_energy_matrix':wave,'sampled_energy_rates':energy,'sampled_rate_error_max':rate_error,'L0_constant':constant,'origin_flux':origin,
                 'naive_diagonal_mass_relative_error':naive_err,'checks':checks,'passed':all(checks.values())}
            results.append(row);checks_all.extend(checks.values())
            key='rb%.17g_N%d_L%d'%(rb,N,L)
            for name,value in [('rho',x),('M',M),('S',S),('D',D),('DL',DL),('t',t),('V',V),('Mm',Mm),('Sm',Sm),('DLm',DLm)]:matrices[key+'_'+name]=value
assert max(negative)>=T['naive_diagonal_common_weight_mass_negative_control_min_relative']
np.savez_compressed(P/'model-matrices.npz',**matrices)
summary={'cases':len(results),'passed_all_math_model_gates':all(checks_all),'maximum_raw_mass_condition':max(r['raw_mass_condition'] for r in results),'maximum_scaled_mass_condition':max(r['diagonally_scaled_mass_condition'] for r in results),'maximum_modal_mass_condition':max(r['modal_mass_condition'] for r in results),'maximum_common_V_condition':max(r['common_node_modal_V_condition'] for r in results),'maximum_nodal_IBP_scaled':max(r['nodal_IBP']['scaled'] for r in results),'maximum_modal_IBP_scaled':max(r['modal_IBP']['scaled'] for r in results),'maximum_wave_energy_matrix_scaled':max(r['wave_energy_matrix']['scaled'] for r in results),'maximum_sampled_energy_rate_scaled':max(r['sampled_rate_error_max'] for r in results),'maximum_strong_weak_scaled':max(r['strong_weak_action']['scaled'] for r in results),'maximum_collocation_action_scaled':max(r['common_collocation_action']['scaled'] for r in results),'naive_diagonal_mass_error_max':max(negative),'all_L0_static_constants_zero':all(r['L']!=0 or (r['L0_constant']['E']==0 and r['L0_constant']['generator_max']==0) for r in results)}
report={'scope':plan['scope'],'plan_sha256':sha(P/'plan.json'),'source_sha256':sha(__file__),'runtime_seconds':time.monotonic()-started,'environment':{'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS')},'summary':summary,'results':results,'no_actual_Z4c_matrix_eigensolve_or_evolution':True,'L0_energy_is_seminorm_with_static_constant_mode':True,'matrix_npz_sha256':sha(P/'model-matrices.npz')}
(P/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))
assert summary['passed_all_math_model_gates']
