#!/usr/bin/env python3
"""Physical-field convergence and independent controls with verified ancestry.

Reads saved states only. Does not infer accepted-step coverage from old reports.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from numerics import weights
from kernel import PARITY
from trajectory import read_trajectory


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(folder):
    folder=Path(folder);m,rows,stages,segments=read_trajectory(folder)
    if m['status']!='finished':raise ValueError(f'Incomplete trajectory: {folder}')
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as z:
        r=z['r'].copy();q=z['q_physical' if 'q_physical' in z else 'q'].copy();t=float(z['t'])
    if not np.isfinite(q).all():raise ValueError('Nonfinite checkpoint')
    return dict(folder=str(folder.resolve()),manifest=m,rows=rows,stages=stages,segments=segments,
                r=r,q=q,t=t,checkpoint_sha256=digest(folder/'checkpoint.npz'))


def require_same_time(zs):
    if np.ptp([z['t'] for z in zs])>1e-11:raise ValueError('Cannot compare different physical times')


def interpolate(z,r):
    # Ninth-point polynomial interpolation, with true regular-center tensor
    # parity. The compared runs all have the same half-cell center and scri.
    q=z['q'];fine=z['r'];width=9
    if fine[0]<=0:raise ValueError('This control analyzer expects half-cell regular-center grids')
    parity=np.r_[PARITY,np.ones(max(0,q.shape[1]-len(PARITY)))];ext=np.r_[-fine[:width][::-1],fine]
    values=np.r_[q[:width][::-1]*parity,q];out=[]
    for x in r:
        start=min(max(np.searchsorted(ext,x)-width//2,0),len(ext)-width)
        out.append(weights(ext[start:start+width],x)@values[start:start+width])
    return np.asarray(out)


def norm(delta,r,mask):
    fields=np.r_[np.arange(7),np.arange(8,delta.shape[1])]
    return float(np.sqrt(np.sum(r[mask,None]**2*delta[mask][:,fields]**2)/np.sum(r[mask]**2)))


def fit_order(errors,hs):
    # Unequal actual grid spacing, not a log2 approximation.
    if min(errors)<=0:return None
    target=errors[0]/errors[1];a=np.log(hs[0]/hs[1]);b=np.log(hs[1]/hs[2])
    def ratio(p):return a/b if abs(p)<1e-8 else np.expm1(p*a)/(-np.expm1(-p*b))
    left,right=-15.,15.
    if not ratio(left)<target<ratio(right):return None
    for _ in range(80):
        mid=(left+right)/2
        if ratio(mid)<target:left=mid
        else:right=mid
    return float((left+right)/2)


def spatial(zs):
    require_same_time(zs);hs=[float(np.diff(z['r'])[0]) for z in zs];result={}
    for region in ['bulk','endpoint']:
        errors=[];per_field=[]
        for lo,hi in zip(zs[:-1],zs[1:]):
            r=lo['r'];mask=r<.8 if region=='bulk' else r>.9;delta=lo['q']-interpolate(hi,r)
            errors.append(norm(delta,r,mask))
            per_field.append(np.sqrt(np.sum(r[mask,None]**2*delta[mask]**2,axis=0)/np.sum(r[mask]**2)).tolist())
        result[region]=dict(differences=errors,order=fit_order(errors,hs),per_field_differences=per_field,
                            per_field_orders=[fit_order([e[i] for e in per_field],hs) for i in range(zs[0]['q'].shape[1])])
    return dict(h=hs,regions=result)


def controls(zs,time_controls,quad_controls,acceptance,coarse_time=None):
    base=zs[1];r=base['r'];require_same_time(zs+time_controls+quad_controls)
    if any(not np.array_equal(z['r'],r) for z in time_controls+quad_controls):raise ValueError('Control grid mismatch')
    result={};times=([coarse_time] if coarse_time is not None else [])+[base]+time_controls
    cfl=[z['manifest']['arguments']['cfl'] for z in times]
    if not np.allclose(np.array(cfl[:-1])/np.array(cfl[1:]),2):raise ValueError('Temporal controls must halve CFL')
    space=spatial(zs)
    for region in ['bulk','endpoint']:
        mask=r<.8 if region=='bulk' else r>.9;errors=[norm(x['q']-y['q'],r,mask) for x,y in zip(times[:-1],times[1:])]
        # The unscaled standard-to-half difference is conservative once a
        # decaying RK4 asymptotic regime is independently demonstrated.
        temporal_orders=[None if min(x,y)<=0 else float(np.log2(x/y)) for x,y in zip(errors[:-1],errors[1:])]
        standard_error=errors[1 if coarse_time is not None else 0]
        denominator=space['regions'][region]['differences'][1]
        quadrature=[]
        for z in quad_controls:
            error=norm(z['q']-base['q'],r,mask)
            opts=z['manifest']['arguments'];ode=opts.get('elliptic_method')=='discrete-ode'
            quadrature.append(dict(parameter='elliptic_refinements' if ode else 'quadrature_width',
                value=opts['elliptic_refinements' if ode else 'quadrature_width'],difference=error,
                spatial_fraction=error/denominator,pass_fraction=error/denominator<acceptance['elliptic_error_fraction_of_finest_spatial_error_max']))
        result[region]=dict(temporal=dict(cfl=cfl,differences=errors,orders=temporal_orders,
            standard_to_half_fraction=standard_error/denominator,pass_fraction=standard_error/denominator<.1,
            resolved_RK4_order=any(order is not None and 3.5<=order<=4.5 for order in temporal_orders),
            interpretation='Each order uses successive differences at the three listed CFL values. Small-step bulk differences may reach the roundoff floor; retain their measured order.'),quadrature=quadrature)
    return space,result


def constraint_convergence(zs):
    hs=np.array([np.diff(z['r'])[0] for z in zs]);result={}
    for field in ['H','M','Theta','Z']:
        for region in ['rms','bulk','endpoint']:
            key='max_accepted_'+field+'_'+region
            if not all(key in z['stages'] for z in zs):continue
            values=np.array([z['stages'][key] for z in zs]);orders=np.log(values[:-1]/values[1:])/np.log(hs[:-1]/hs[1:])
            result[field+'_'+region]=dict(maxima=values.tolist(),pair_orders=orders.tolist())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resolutions',type=int,nargs=3,default=[512,1024,2048])
    p.add_argument('--acceptance',type=Path,default=Path(__file__).resolve().parents[2]/'docs/elliptic-shift-1d/acceptance.json')
    p.add_argument('--dissipation-input',type=Path);p.add_argument('--coarse-time-controls',action='store_true')
    p.add_argument('--finest-quadrature-input','--finest-elliptic-input',dest='finest_quadrature_input',type=Path)
    p.add_argument('--elliptic-solver-controls',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    acceptance=json.loads(a.acceptance.read_text())['preliminary_evolution_gates']
    zs=[load(a.input/f'spatial-n{n}') for n in a.resolutions]
    time=[load(a.input/name) for name in ['time-cfl0.025','time-cfl0.0125']]
    quad=[load(a.input/(f'elliptic-refinement{n}-n{a.resolutions[1]}' if a.elliptic_solver_controls else f'quadrature-q{n}')) for n in ([0,4] if a.elliptic_solver_controls else [9,11])]
    coarse=load(a.input/'time-cfl0.1') if a.coarse_time_controls else None
    space,independent=controls(zs,time,quad,acceptance,coarse);constraints=constraint_convergence(zs)
    runs={}
    for z in zs+time+quad+([coarse] if coarse is not None else []):
        runs[Path(z['folder']).name]={k:z[k] for k in ['folder','t','checkpoint_sha256','segments','stages']}
        runs[Path(z['folder']).name].update(arguments=z['manifest']['arguments'],binding=z['manifest']['binding'])
    if a.coarse_time_controls and (a.input/'time-cfl0.2/manifest.json').exists():
        m=json.loads((a.input/'time-cfl0.2/manifest.json').read_text())
        runs['time-cfl0.2']=dict(status=m['status'],time=m.get('time'),failure=m.get('failure'),binding=m['binding'],note='Failed cases are retained and excluded from same-time convergence arithmetic.')
    dissipation={}
    if a.dissipation_input:
        # A previous source-bound matrix is labeled separately. It cannot be
        # used to fabricate missing all-step instrumentation in the new one.
        for strength in [.25,.5,1.]:
            dz=zs if strength==.5 else [load(a.dissipation_input/f'KO{strength}-n{n}') for n in a.resolutions]
            for z,standard in zip(dz,zs):
                if z['manifest']['binding']!=standard['manifest']['binding']:raise ValueError('Dissipation control source/library differs from its baseline')
            dissipation[str(strength)]=dict(spatial=spatial(dz),runs=[dict(folder=z['folder'],checkpoint_sha256=z['checkpoint_sha256'],segments=z['segments'],stages=z['stages']) for z in dz])
    gates=dict(accepted_bulk_constraints=bool(constraints and all(min(constraints[f+'_bulk']['pair_orders'])>=acceptance['gauge_pulse_constraint_bulk_order_min'] for f in ['H','M'])),
        physical_endpoint_order=space['regions']['endpoint']['order'] is not None and space['regions']['endpoint']['order']>=acceptance['endpoint_error_order_min'],
        timestep_error_fractions=all(z['temporal']['pass_fraction'] for z in independent.values()),
        resolved_temporal_order=all(z['temporal']['resolved_RK4_order'] for z in independent.values()),
        elliptic_error_fractions=all(v['pass_fraction'] for z in independent.values() for v in z['quadrature']))
    finest={}
    if a.finest_quadrature_input:
        base=zs[-1];qcontrols=[load(a.finest_quadrature_input/(f'elliptic-refinement{q}-n{a.resolutions[-1]}' if a.elliptic_solver_controls else f'quadrature-q{q}-n{a.resolutions[-1]}')) for q in ([0,4] if a.elliptic_solver_controls else [9,11])]
        require_same_time([base]+qcontrols)
        if any(not np.array_equal(z['r'],base['r']) for z in qcontrols):raise ValueError('Finest quadrature grid mismatch')
        for region in ['bulk','endpoint']:
            order=space['regions'][region]['order'];difference=space['regions'][region]['differences'][-1]
            spatial_error=difference/np.expm1(order*np.log(space['h'][-2]/space['h'][-1]))
            r=base['r'];mask=r<.8 if region=='bulk' else r>.9
            finest[region]=dict(Richardson_finest_spatial_error=spatial_error,raw_middle_to_finest_difference=difference,
                quadrature=[dict(parameter='elliptic_refinements' if a.elliptic_solver_controls else 'quadrature_width',
                    value=z['manifest']['arguments']['elliptic_refinements' if a.elliptic_solver_controls else 'quadrature_width'],difference=norm(z['q']-base['q'],r,mask),
                    fraction=norm(z['q']-base['q'],r,mask)/spatial_error,
                    pass_fraction=bool(norm(z['q']-base['q'],r,mask)/spatial_error<acceptance['elliptic_error_fraction_of_finest_spatial_error_max'])) for z in qcontrols])
        gates['finest_elliptic_error_fractions']=all(q['pass_fraction'] for region in finest.values() for q in region['quadrature'])
        for z in qcontrols:runs[Path(z['folder']).name]={k:z[k] for k in ['folder','t','checkpoint_sha256','segments','stages']}
    if a.elliptic_solver_controls:
        gates['all_stage_discrete_ODE_residual']=all(z['stages'].get('max_elliptic_discrete_residual',float('inf'))<=1e-11 for z in zs+time+quad)
    result=dict(scope='Finite-amplitude positive-variable gauge pulse at the exact control time. Boundary safety and duration assessed separately; no full-goal admission.',
        physical_fields='chi-1, g-1, Ar, Khat+3, Theta, Lambda, alpha-Aref; shift placeholder excluded',
        comparison_norm='r^2 weighted RMS of physical fields; bulk r<.8 and endpoint r>.9; degree-8 parity interpolation',
        spatial=space,accepted_constraint_convergence=constraints,independent_controls=independent,finest_elliptic_controls=finest,gates=gates,runs=runs,dissipation=dissipation,
        elliptic_control_kind='Direct banded arithmetic refinement; distinct from the failed sampled-source quadrature discretization study.' if a.elliptic_solver_controls else 'Independent quadrature discretization degree',
        provenance=dict(analysis_sha256=digest(__file__),trajectory_sha256=digest(Path(__file__).with_name('trajectory.py')),acceptance_sha256=digest(a.acceptance)))
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for field in ['H','M']:
        axes[0].loglog(a.resolutions,constraints[field+'_bulk']['maxima'],'o-',label=field+' accepted')
        axes[1].loglog(a.resolutions,[z['stages']['max_'+field+'_rms'] for z in zs],'o-',label=field+' all RK stages')
        axes[1].loglog(a.resolutions,constraints[field+'_rms']['maxima'],'x--',label=field+' accepted')
    for z in zs:
        axes[2].semilogy([r['t'] for r in z['rows']],[r['H_rms'] for r in z['rows']],label=str(len(z['r'])))
    for ax,title in zip(axes,['Peak accepted bulk constraints','RMS: stage and accepted peaks','Accepted Hamiltonian history']):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_xlabel('N');axes[1].set_xlabel('N');axes[2].set_xlabel('Coordinate time')
    fig.savefig(a.output/'refinement.png',dpi=160);fig.savefig(a.output/'refinement.pdf');plt.close(fig)
    print(json.dumps({k:result[k] for k in ['spatial','accepted_constraint_convergence','independent_controls','gates']}))


if __name__=='__main__':main()
