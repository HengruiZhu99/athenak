"""Independent nonlinear Jacobian and comparative actual full20 screen."""
from pathlib import Path
import json
import numpy as np
import sympy as sy
import mpmath as mp

P=Path(__file__).resolve().parent


def matrix(x):
    a=np.asarray(x);assert np.isfinite(a).all()
    return a[...,0]+1j*a[...,1]


def key(row):
    return tuple(row[k] for k in ('a','perturb','r','Omega','k','oblique'))


def main():
    alpha,B,b,c,O=sy.symbols('alpha B b c Omega',real=True)
    F=3*c*(alpha+2*c)*(alpha*B-b)/O
    assert sy.simplify(sy.diff(F,alpha)-3*c*((2*alpha+2*c)*B-b)/O)==0
    assert sy.diff(F,alpha,alpha)==6*c*B/O
    assert sy.diff(F,alpha,b)==-3*c/O
    symbolic_alpha_jacobian=sy.diff(F,alpha)
    nonlinear=json.loads((P/'nonlinear.json').read_text())
    assert nonlinear==json.loads((P/'nonlinear-debug.json').read_text())
    assert nonlinear['rows']==4004 and nonlinear['crossbranch_rows']==112
    for name in ('reference_source_max','outer_source_difference_max','core_source_difference_max',
                 'shift_pole_difference_max','nonphysical_source_max'):
        assert nonlinear[name]==0
    for name in ('nonlinear_identity_max','value_jacobian_max','Wzero_original_Q_identity_max','crossbranch_identity_relative_max','combined_crossbranch_jacobian_relative_max'):
        assert nonlinear[name]<1e-12
    assert nonlinear['a.5_sampled_support_Omega_min']>=.2775
    tiny=nonlinear['tiny_lapse'];mp.mp.dps=100
    expected=3*(mp.mpf(tiny['alpha'])+2)*mp.mpf(tiny['alpha'])*mp.mpf(tiny['beta_ref_dot_dOmega'])/(mp.mpf(tiny['alpha_ref'])*mp.mpf(tiny['Omega']))
    tiny_error=abs(mp.mpf(tiny['trace_addition'])/expected-1)
    assert tiny_error<mp.mpf('1e-14') and tiny['trace_addition']>0
    assert tiny['combined_regular_alpha']==tiny['trace_addition']
    principals={name:json.loads((P/(name+'.log')).read_text()) for name in ('principal-trace','principal-combined')}
    assert all(x['passed_kernel_cases']==360 and x['max_kernel_symbol_error']<1e-12 for x in principals.values())
    rows=json.loads((P/'full20.json').read_text());assert len(rows)==3800
    groups={};roots=[]
    for row in rows:
        L=matrix(row['L']);groups.setdefault(key(row),{})[row['candidate']]=row
        assert row['reference_fixedpoint']<2e-9
        weight=np.r_[np.ones(12),np.full(8,1/max(row['k'],1))]
        A=weight[:,None]*L/weight[None,:]
        ev,V=np.linalg.eig(A);norm=np.max(np.sum(abs(A),axis=1))
        backward=max(float(abs(A@V[:,i]-ev[i]*V[:,i]).max()/((norm+abs(ev[i]))*abs(V[:,i]).max())) for i in range(20))
        assert backward<1e-12
        roots.append({k:row[k] for k in ('a','candidate','perturb','r','Omega','k','oblique')}|{'max_real':float(ev.real.max()),'backward_error':backward})
    identity=0;outer=0;core=0
    for group in groups.values():
        assert set(group)=={0,1,2,3}
        for mode in (1,2,3):
            row=group[mode];expected=np.zeros((20,20))
            h,alpha,c,O=row['alpha_ref'],row['alpha_live'],row['c'],row['Omega']
            live=np.asarray(row['beta_live']);gradient=np.asarray(row['dalpha_ref']);Om=np.asarray(row['domega']);ref=np.asarray(row['beta_reference'])
            if mode&1:
                expected[0,0]-=c*np.dot(live,gradient)/h
                expected[0,4:7]-=c*alpha*gradient/h
            if mode&2 and c>0:
                B=np.dot(ref,Om)/h;b=np.dot(live,Om)
                expected[0,0]+=3*c*((2*alpha+2*c)*B-b)/O
                expected[0,4:7]-=3*c*(alpha+2*c)*Om/O
            difference=matrix(row['L'])-matrix(group[0]['L'])
            identity=max(identity,float(abs(difference-expected).max()/(1+abs(expected).max())))
            assert np.count_nonzero(difference[1:])==0
            if row['r']>=.85:outer=max(outer,float(abs(difference).max()))
            if row['r']<=.05:core=max(core,float(abs(difference).max()))
    assert identity<1e-12 and outer==core==0
    previous=json.loads((P.parent/'inner-lapse-advection-control/full20.json').read_text())
    assert len(previous)==1900
    assert all(row['L']==groups[key(row)][row['candidate']]['L'] for row in previous)
    poles=json.loads((P/'poles.json').read_text());assert len(poles)==16
    for a in (.5,.75,1.,2.):
        group=[row for row in poles if row['a']==a]
        assert len(group)==4 and all(row['P']==group[0]['P'] for row in group)
    worst=[max((row for row in roots if row['a']==.5 and row['candidate']==mode and row['perturb']==pert),key=lambda row:row['max_real']) for mode in (0,1,2,3) for pert in (0,.01)]
    report={'status':'PASS','scope':'FiniteOmega lowerorder inner trace/combined gauge identity and comparative actualfull20 screen; no native/global/stability/BH admission',
      'nonlinear':nonlinear,'principals':principals,'symbolic_trace_source':str(F),
      'alpha_jacobian':str(symbolic_alpha_jacobian),'alpha_alpha_hessian':'6cB/Omega','alpha_beta_hessian':'-3cOmega_i/Omega',
      'tiny_lapse_100digit_relative_error':mp.nstr(tiny_error,30),
      'actual_full20_rows':len(rows),'matched_four_mode_groups':len(groups),
      'predicted_alpha_only_delta_error':identity,'outer_matrix_exact_difference':outer,'core_matrix_exact_difference':core,
      'prior_baseline_and_advection_matrices_bitwise_equal':len(previous),'four_all_curvature_poles_bitwise_equal':True,
      'max_reference_RHS':max(row['reference_fixedpoint'] for row in rows),'worst_a.5_reference_and_perturbed_roots':worst,'roots':roots,
      'mode_labels':{'0':'physicalP baseline','1':'rejected legacy advection control','2':'trace-only robust D','3':'combined direct regular advection+trace'},
      'limitations':['Positive primitive roots retained and not classified as gauge/constraint modes.','No invariant lapse positivity, uniform puncture/scri hyperbolicity, energy or wormhole-to-trumpet gate.','Direct equivalent evaluation avoids old additive advection cancellation; finite supportOmega bound is for S1,a.5 current geometry.']}
    (P/'check-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS trace/combined finiteOmega gate; no stability claim; matrix error',identity)
    print('tiny lapse relative error',mp.nstr(tiny_error,12))
    for row in worst:print(row)


if __name__=='__main__':main()
