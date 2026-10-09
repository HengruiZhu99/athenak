"""Actual source/principal/pole identities and finite-Omega primitive screen."""
from pathlib import Path
import json
import numpy as np
import sympy as sp

P = Path(__file__).resolve().parent


def matrix(x):
    a=np.asarray(x)
    assert np.isfinite(a).all()
    return a[:,:,0]+1j*a[:,:,1]


def key(x):
    return tuple(x[n] for n in ("a","perturb","r","Omega","k","oblique"))


def main():
    np.seterr(all="raise")
    alpha,ahat,beta,bhat,G,c=sp.symbols('alpha ahat beta bhat G c',real=True)
    delta=-c*((beta-bhat)+beta*(alpha-ahat)/ahat)*G
    assert sp.simplify(delta+c*(alpha*beta/ahat-bhat)*G)==0
    assert sp.simplify(sp.diff(delta,alpha)+c*beta*G/ahat)==0
    assert sp.simplify(sp.diff(delta,beta)+c*alpha*G/ahat)==0
    assert sp.simplify(sp.diff(delta,alpha,beta)+c*G/ahat)==0
    nonlinear=json.loads((P/'nonlinear.json').read_text())
    assert nonlinear==json.loads((P/'nonlinear-debug.json').read_text())
    assert nonlinear['rows']==4004
    for name in ('reference_source_max','outer_offconstraint_source_max','exact_Cauchy_core_source_max',
                 'all_shift_and_pole_difference_max','nonphysical_lapse_source_max','outer_cutoff_first_three_jets_max'):
        assert nonlinear[name]==0
    assert max(nonlinear['nonlinear_identity_max'],nonlinear['value_jacobian_max'],nonlinear['Wzero_log_relative_identity_max'])<1e-12
    principal=json.loads((P/'principal.log').read_text())
    assert principal['passed_kernel_cases']==360 and principal['max_kernel_symbol_error']<1e-12
    rows=json.loads((P/'full20.json').read_text())
    assert len(rows)==1900
    groups={};roots=[];rk=[]
    for row in rows:
        L=matrix(row['L']);groups.setdefault(key(row),{})[row['candidate']]=row
        assert row['reference_fixedpoint']<2e-9
        weights=np.array([1.]*12+[1/max(row['k'],1.)]*8)
        B=weights[:,None]*L/weights[None,:]
        eig,V=np.linalg.eig(B)
        backward=max(float(abs(np.einsum('ij,j->i',B,V[:,i])-eig[i]*V[:,i]).max()/
                           ((np.max(np.sum(abs(B),axis=1))+abs(eig[i]))*abs(V[:,i]).max())) for i in range(20))
        assert backward<1e-12
        roots.append({n:row[n] for n in ('a','candidate','perturb','r','Omega','k','oblique')}|
                     {'max_real':float(eig.real.max()),'backward_error':backward})
        if row['a']==.5 and row['perturb']==0:
            for omega in (.0142578125,.0038368055555556,.003251953125):
                if abs(row['Omega']-omega)>1e-10:continue
                z=.03*omega*eig[eig.real<=0]
                excess=max(0.,float(abs(1+z+z*z/2+z*z*z/6).max())-1)
                assert excess<1e-12
                rk.append({'candidate':row['candidate'],'Omega':omega,'k':row['k'],'oblique':row['oblique'],'excess':excess})
    identity=0;outer=0;core=0
    for pair in groups.values():
        assert set(pair)=={0,1}
        row=pair[1];deltaL=matrix(row['L'])-matrix(pair[0]['L'])
        expected=np.zeros((20,20))
        grad=np.asarray(row['dalpha_ref']);beta_live=np.asarray(row['beta_live'])
        expected[0,0]=-row['c']*np.dot(beta_live,grad)/row['alpha_ref']
        expected[0,4:7]=-row['c']*row['alpha_live']/row['alpha_ref']*grad
        identity=max(identity,float(abs(deltaL-expected).max()/(1+abs(expected).max())))
        if row['r']>=.85:outer=max(outer,float(abs(deltaL).max()))
        if row['r']<=.05:core=max(core,float(abs(deltaL).max()))
    assert identity<1e-12 and outer==core==0 and len(rk)==132
    # A previous immutable gate supplies an unchanged independent base C0 matrix.
    previous=P.parent/'damping-profile-control/full20.json'
    old=json.loads(previous.read_text());common=[]
    for row in old:
        if row['profile']!=0:continue
        pair=groups[key(row)]
        common.append(float(abs(matrix(row['L'])-matrix(pair[0]['L'])).max()))
    assert len(common)==380 and max(common)==0
    poles=json.loads((P/'poles.json').read_text());assert len(poles)==8
    for a in (.5,.75,1.,2.):
        both=[x for x in poles if x['a']==a]
        assert len(both)==2 and both[0]['P']==both[1]['P']
    worst=[]
    for candidate in (0,1):
        for pert in (0,.01):
            cases=[x for x in roots if x['a']==.5 and x['candidate']==candidate and x['perturb']==pert]
            worst.append(max(cases,key=lambda x:x['max_real']))
    summary={'status':'PASS','scope':'Lower-order regular lapse source, finite-Omega actual-kernel admission only',
      'source_identity':str(sp.factor(delta)),'value_jacobian_alpha':str(sp.diff(delta,alpha)),
      'value_jacobian_beta':str(sp.diff(delta,beta)),'nonlinear_mixed_hessian':str(sp.diff(delta,alpha,beta)),
      'geometry_layer_radii':[.05,.95],'gauge_layer_radii':[.45,.85],
      'nonlinear':nonlinear,'principal':principal,'actual_full20_rows':len(rows),
      'only_regular_lapse_value_row_changes_error':identity,'outer_matrix_difference_exact':outer,
      'exact_Cauchy_core_matrix_difference_exact':core,'independent_base_matrix_rows_bitwise_equal':len(common),
      'four_curvatures_scri_pole_matrices_bitwise_identical':True,'negative_root_RK3_cases':len(rk),
      'wavenumber_labels':'pi*N/2.2 are global tangent-grid frequencies; pi*N/2.1 are actual native-grid frequencies',
      'worst_sampled_reference_and_perturbed_roots':worst,'roots':roots,
      'max_reference_full20_rhs':max(x['reference_fixedpoint'] for x in rows),
      'primitive_positive_root_is_gauge_claim':False,'native_global_or_scri_stability_accepted':False,
      'future_BH_scope':'Exact Omega1 Cauchy core and its eta_inner=v conditional obstruction unchanged; no BH formation or global matching proof'}
    (P/'check-report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('PASS lower-order gauge source/principal/pole identities; no stability acceptance')
    print('rows',len(rows),'deltaL error',identity,'unchanged baseline rows',len(common))
    print('worst roots',worst)


if __name__=='__main__':main()
