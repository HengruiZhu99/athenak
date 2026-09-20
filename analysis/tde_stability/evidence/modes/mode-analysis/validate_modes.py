#!/usr/bin/env python3
import json
import numpy as np
from mode_operator import *
label='arnoldi_s40_m32_eps0.001'
report=json.loads((ROOT/(label+'-results.json')).read_text())
op=Operator('mode_validation')
results=[]
for n,mode in enumerate(report['modes'][:2]):
    v=np.fromfile(mode['file']).reshape(SHAPE)
    ev=mode['real'];res={'index':n,'eigenvalue_40steps':ev,'gamma':np.log(ev)/(40*DT),'efold_M':40*DT/np.log(ev),'profile':profile(v)}
    rs={e:op.response(v,40,e,label=f'mode{n}_eps{e:g}') for e in [3e-3,1e-3,3e-4]}
    res['amplitude_convergence']={str(e):discrepancy(rs[e],rs[1e-3]) for e in [3e-3,3e-4]}
    res['eigen_residual_40steps']=discrepancy(ev*v,rs[1e-3])
    r1=op.response(v,1,1e-3,label=f'mode{n}_one_step')
    mu1=ev**(1/40)
    res['eigenvalue_one_step_expected']=mu1
    res['eigen_residual_one_step']=discrepancy(mu1*v,r1)
    for name,slice_ in [('full', (...,)),('active',ACTIVE)]:
        vv=v[slice_];rr=r1[slice_];q=np.vdot(vv,rr)/np.vdot(vv,vv)
        res[f'one_step_rayleigh_{name}']=float(q)
        res[f'one_step_gamma_{name}']=float(np.log(q)/DT)
    # One unmodified, uninterrupted evolution of each candidate, initialized at
    # peak1e-4, checks growth and shape outside the central-difference wrapper.
    eps=1e-4;steps=100
    direct=op.advance(eps*v,steps,label=f'mode{n}_nonlinear100steps')/eps
    expected=v*ev**(steps/40)
    res['nonlinear100steps']=discrepancy(expected,direct)
    # Capture the actual first-stage volume, KO and CPBC RHS and projection.
    operations='volume_rhs,post_ko_rhs,post_boundary_rhs,post_rk,post_recast'
    for sign in [1,-1]:
        lab=f'mode{n}_stage_sign{sign}'
        op.advance(sign*1e-3*v,1,label=lab,overrides=('z4c/debug_balance=true',f'z4c/debug_snapshot_operations={operations}',))
        folder=op.directory/lab
        # Keep first-stage signed snapshots and metadata, remove duplicate BG
        # arrays and later stages; final complete map output is retained.
        for p in folder.glob('z4c_snapshot_*'):
            if '.background.' in p.name or '_stage2.' in p.name or '_stage3.' in p.name:p.unlink()
    snaps={}
    for term in operations.split(','):
        pat=f'z4c_snapshot_{term}_rank0_cycle0_stage1.bin'
        plus=np.fromfile(op.directory/f'mode{n}_stage_sign1'/pat).reshape(SHAPE)
        minus=np.fromfile(op.directory/f'mode{n}_stage_sign-1'/pat).reshape(SHAPE)
        snaps[term]=(plus-minus)/2e-3
    va=v[ACTIVE];norm=np.vdot(va,va)
    terms={'volume':snaps['volume_rhs'],'KO':snaps['post_ko_rhs']-snaps['volume_rhs'],'boundary':snaps['post_boundary_rhs']-snaps['post_ko_rhs'],'final_rhs':snaps['post_boundary_rhs']}
    res['stage1_active_growth_budget']={name:float(np.vdot(va,rhs[ACTIVE])/norm) for name,rhs in terms.items()}
    res['stage1_active_rhs_profiles']={name:profile(rhs) for name,rhs in terms.items()}
    res['stage1_projection_change']=discrepancy(snaps['post_recast'][ACTIVE],snaps['post_rk'][ACTIVE])
    results.append(res)
    print(json.dumps({k:x for k,x in res.items() if k not in ['profile','stage1_active_rhs_profiles']},indent=2),flush=True)
    (ROOT/'validated-modes.json').write_text(json.dumps(results,indent=2)+'\n')
