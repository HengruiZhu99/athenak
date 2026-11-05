#!/usr/bin/env python3
"""Controlled serial cost frontier with exterior and separate boundary screens."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from physical_constraints import evaluate
from kernel import Kernel
from evolution import make_evolution


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--library',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--hardware',type=Path)
    a=ap.parse_args()
    plan=json.loads((a.runs/'plan.json').read_text())
    completion=json.loads((a.runs/'completion.json').read_text())
    assert len(completion['results'])==len(plan['plan'])
    assert hashlib.sha256(Path(a.library).read_bytes()).hexdigest()==plan['binary_sha256']
    reference=plan['plan'][0]['configuration']
    allowed_changes={'n','output','fixed_dt','experimental_timestep','penalty'}
    for item in plan['plan']:
        cfg=item['configuration']
        assert set(cfg)==set(reference)
        assert all(cfg[key]==reference[key] for key in cfg if key not in allowed_changes),'Unattributed configuration change'
    a.output.mkdir(parents=True,exist_ok=False)
    costs={x['name']:x for x in completion['results']}
    common=np.geomspace(25,100,301);inputs={str(a.runs/name):hashlib.sha256((a.runs/name).read_bytes()).hexdigest() for name in ('plan.json','completion.json')};groups={};runs=[]
    def read(p):
        inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text()) if p.suffix=='.json' else np.load(p)
    hardware=read(a.hardware) if a.hardware else None
    for item in plan['plan']:
        folder=a.runs/item['name'];process=costs[item['name']]
        row=dict(name=item['name'],case=item['case'],family=item['family'],n=item['n'],repeat=item['repeat'],process_seconds=process['process_seconds'],status=process['status'])
        if (folder/'result.json').exists():
            result=read(folder/'result.json');manifest=read(folder/'manifest.json');cfg=manifest['config']
            assert cfg==item['configuration'] and manifest['binary_sha256']==plan['binary_sha256']
            assert {k:manifest['sources'][k] for k in plan['execution_sources']}==plan['execution_sources']
            assert cfg['duration']==10 and not cfg['floors'] and not cfg['probes'] and cfg['perturbation']=='none'
            row['result']=result
            if result['status']=='completed_unqualified':
                assert process['returncode']==0 and result['floor_calls']==0
                initial=read(folder/'initial.npz');last=read(sorted(folder.glob('profile-*.npz'))[-1]);r=initial['r'];om=initial['omega']
                assert float(last['t_M'])==10
                phys=evaluate(r,last['q'],last['d'],last['dd'],om);phys[:,:2]/=9;phys[:,2:]/=3
                mask=r[:-1]>initial['configuration'][3]/6
                R=3*r[:-1][mask]/(om[:-1,0][mask]*np.sqrt(initial['q'][:-1,0][mask]))
                assert np.all(np.diff(R)>0) and R[0]<25 and R[-1]>100
                pchip=PchipInterpolator(R,phys[mask])(common)
                linear=np.column_stack([np.interp(common,R,phys[mask,f]) for f in range(4)])
                rms=np.sqrt(np.mean(pchip**2,axis=0));lrms=np.sqrt(np.mean(linear**2,axis=0))
                row.update(physical_rms=rms.tolist(),linear_physical_rms=lrms.tolist(),interpolation_disagreement_rms=np.sqrt(np.mean((pchip-linear)**2,axis=0)).tolist(),indicator=float(max(rms)),screening_indicator=float(max(max(rms),max(lrms))),final_q_sha256=hashlib.sha256(last['q'].tobytes()).hexdigest())
        runs.append(row);groups.setdefault(item['case'],[]).append(row)
    summary=[]
    for case,rows in groups.items():
        ok=[x for x in rows if x['status']=='completed_unqualified'];s=dict(case=case,family=rows[0]['family'],n=rows[0]['n'],repeats=len(rows),completed=len(ok))
        if ok:
            assert len({x['final_q_sha256'] for x in ok})==1,'Repeated final states differ'
            s['repeated_states_bitwise_equal']=True
            for key in ('stepping_seconds','setup_seconds','rhs_seconds','rhs_calls','steps','C_char_max_accepted','dt_over_M','min_dr_over_M','steps_per_M','initial_speed','worst_evaluated_speed'):
                v=[x['result'][key] for x in ok];s[key]=dict(min=min(v),median=float(np.median(v)),max=max(v))
            v=[x['process_seconds'] for x in ok];s['process_seconds']=dict(min=min(v),median=float(np.median(v)),max=max(v))
            for key in ('indicator','screening_indicator','physical_rms','linear_physical_rms','interpolation_disagreement_rms'):s[key]=ok[0][key]
            folder=a.runs/ok[0]['name'];cfg=read(folder/'manifest.json')['config'];last=read(sorted(folder.glob('profile-*.npz'))[-1])
            k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);state=last['q'][:,e.independent];rate=e.rhs(state);base=e.last.copy();probes=[]
            for scale in (1e-5,3e-6,1e-6):
                eps=scale/max(1.,np.max(abs(rate)));e.rhs(state+eps*rate);plus=e.last['dependent_D'];e.rhs(state-eps*rate);minus=e.last['dependent_D']
                probes.append(dict(epsilon=float(eps),defect=float((plus-minus)/(2*eps)-base['auxiliary_trace_pde'])))
            s.update(clock=float(base['clock']),D_defect_max=float(max(abs(x['defect']) for x in probes)),D_probes=probes,pole=float(max(abs(base['pole']))),trace=float(max(abs(base['trace']))));k.close()
        summary.append(s)
    complete=[s for s in summary if s['completed']==s['repeats']]
    families=list(dict.fromkeys(x['family'] for x in summary))
    def select(E,clock=None,D=None):
        selected={}
        for family in families:
            eligible=[s for s in complete if s['family']==family and s['screening_indicator']<=E and (clock is None or abs(s['clock'])<=clock) and (D is None or s['D_defect_max']<=D)]
            best=min(eligible,key=lambda s:s['stepping_seconds']['median']) if eligible else None
            selected[family]={k:best[k] for k in ('case','n','indicator','screening_indicator','clock','D_defect_max','stepping_seconds','process_seconds','rhs_calls')} if best else None
        base=selected[families[0]]
        ratios={f:base['stepping_seconds']['median']/v['stepping_seconds']['median'] if base and v else None for f,v in selected.items()}
        return dict(E=E,clock=clock,D=D,selected=selected,baseline_over_family_cost_ratio=ratios)
    targets=[select(t) for t in (.003,.001,.0001)]
    boxes=[dict(name=name,**select(E,c,D)) for name,E,c,D in [('loose',.003,.1,.1),('medium',.001,.01,.01),('tight',.0001,.001,.001)]]
    frontier=[s['case'] for s in complete if not any(t['screening_indicator']<=s['screening_indicator'] and t['stepping_seconds']['median']<=s['stepping_seconds']['median'] and (t['screening_indicator']<s['screening_indicator'] or t['stepping_seconds']['median']<s['stepping_seconds']['median']) for t in complete)]
    out=dict(domain='Reference R/M25–100;301 log samples;10M;M=1/3',platform=plan['platform'],thread_environment=plan['thread_environment'],hardware=hardware,runs=runs,summary=summary,exterior_only_frontier=frontier,exterior_targets=targets,boundary_aware_targets=boxes,execution_sources=plan['execution_sources'],binary_sha256=plan['binary_sha256'],inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Local CPU medians/min-max with identical diagnostics. Both interpolation indicators must pass. Constraints are not a solution-error bound. Boundary targets have fixed laboratory units and are separate screens. Finite-duration pulse and convergence qualification is documented separately. No all-time, boosted, 3D or GPU performance claim.')
    (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for family in families:
        v=sorted([s for s in complete if s['family']==family],key=lambda s:s['n']);x=[s['stepping_seconds']['median'] for s in v];xerr=[[s['stepping_seconds']['median']-s['stepping_seconds']['min'] for s in v],[s['stepping_seconds']['max']-s['stepping_seconds']['median'] for s in v]]
        for ax,key in zip(axes,('screening_indicator','clock','D_defect_max')):
            y=[abs(s[key]) for s in v];ax.errorbar(x,y,xerr=xerr,fmt='o-',capsize=3,label=family)
            for xx,yy,s in zip(x,y,v):ax.annotate(str(s['n']),(xx,yy),xytext=(4,5),textcoords='offset points',fontsize=7)
    for ax,label in zip(axes,('Exterior constraint indicator (screen)','Absolute clock residual','Maximum differential D defect')):
        ax.set(xscale='log',yscale='log',xlabel='10M CPU stepping seconds (median; min–max)',ylabel=label);ax.grid(which='both',alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(a.output/'accuracy-cost.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(exterior_targets=targets,boundary_aware_targets=boxes),indent=2))

if __name__=='__main__':main()
