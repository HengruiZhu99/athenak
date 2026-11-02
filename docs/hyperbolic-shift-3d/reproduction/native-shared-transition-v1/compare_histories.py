#!/usr/bin/env python3
"""Compare verified native N16 histories with SG1; no submission or restart.

Endpoint radial reconstruction is a separate operation with the matching native
diagnostic executable. This script never substitutes histories for that audit.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
import control
import native_transition_campaign as campaign
spec=importlib.util.spec_from_file_location('sg1_comparison',HERE.parent/'short-gauge-v1/compare_resolutions.py')
sg1=importlib.util.module_from_spec(spec);spec.loader.exec_module(sg1)
require=sg1.require
read=sg1.read
digest=sg1.digest
LABELS={'SG1':'SG1 external initializer','A':'A native, original collar',
        'B':'B native, shared restoration','C':'C native, original + KO10',
        'D':'D native, original + KO10 + first6','E':'E native, shared + KO10 + first6'}
COLORS={'SG1':'#222222','A':'#b55315','B':'#1769aa','C':'#228844','D':'#8055aa','E':'#cc3344'}


def load_native(raw,variant,plans,campaign_module=None):
    """Use local mirrors, without requiring the remote executable to be copied.

Verify every recorded artifact and frozen identity. Healthy checkpoint payloads
are admitted only by the exact schema/configuration inspector. Failed payloads
are preserved evidence, never parsed as an accepted evolution endpoint.
"""
    campaign=campaign_module or globals()['campaign']
    raw=Path(raw).resolve();science=campaign.science(raw)
    require(variant in campaign.VARIANTS and plans,'Unknown/empty case')
    require(digest(raw/campaign.MANIFEST)==science['source_manifest_sha256'],'Source manifest identity changed')
    require(digest(raw/campaign.SCIENCE)==read(raw/Path(campaign.SCIENCE).parent/'provenance.json')['files'][Path(campaign.SCIENCE).name]['sha256'],
            'Frozen science provenance changed')
    histories=[];reports=[];records=[];previous=None
    for segment,relative in enumerate(plans,1):
        path=control.local_path(raw,relative);plan=read(path);binding=read(path.with_name('binding.json'))
        worker=read(path.with_name('worker-status.json'));collected=read(path.with_name('collected.json'))
        submission=read(path.with_name('submission.json'))
        require(plan['variant']==variant and plan['purpose']=='evolution' and plan['segment']==segment,'Wrong case/stage/order')
        campaign.validate_native_plan(raw,plan)
        require(worker['job_id']==collected['job_id']==submission['job_id'],'Job identities differ')
        require(collected['returncode']==0 and collected['scheduler']['job_state'] in ['F','X'],'Scheduler outcome is not terminal')
        require(worker['status'] in ['passed','checkpointed','failed'],'Worker is not terminal')
        require(digest(path)==worker['plan_sha256']==binding['plan_sha256']==submission['plan_sha256'],'Plan binding changed')
        expected={campaign.binary(variant):science['binaries'][campaign.binary(variant)]}
        for record in (binding,worker):
            require(record['source_manifest_sha256']==science['source_manifest_sha256'] and record['binaries']==expected,'Scientific binding changed')
        require(binding['inputs'][campaign.SCIENCE]==digest(raw/campaign.SCIENCE),'Input science changed')
        require(len(worker['cases'])==1,'Expected one puncture evolution')
        case=worker['cases'][0];command=plan['commands'][0]
        require(case['name']=='puncture' and case['status']==worker['status'] and not case['timed_out'] and not case['storage_error'] and not case.get('report_error'),
                'Infrastructure failure needs diagnosis before scientific comparison')
        if worker['status']!='failed':
            require(collected['scheduler']['Exit_status']==(0 if worker['status']=='passed' else 3),'Scheduler disposition differs')
        required=[command['report'],command['args'][1]+'.history.csv',command['args'][1]+'.checkpoint']
        require(all(name in case['artifacts'] for name in required),'Missing required artifact')
        for name,identity in case['artifacts'].items():
            artifact=control.local_path(raw,name)
            require(artifact.stat().st_size==identity['bytes'] and digest(artifact)==identity['sha256'],'Artifact changed: '+name)
        report=read(raw/command['report'])
        # Static scientific settings must match even for a numerical failure.
        for check in campaign.required_checks(variant,'evolution','puncture',segment>1):
            if check['op']=='eq' and check['path'] not in [['status'],['checkpoint_admitted']]:
                require(control.evaluate_check(report,check)['passed'],'Scientific report differs: '+str(check['path']))
        require(report['status'] in ['completed','checkpointed','failed'],'No native numerical disposition')
        require((worker['status']=='failed')==(report['status']=='failed'),'Native/worker disposition differs')
        start=plan['start_verification']
        if previous is None:
            require(start['kind']=='native_fresh' and start['time']==0,'Case must start from native t=0')
        else:
            require(previous['status']=='checkpointed' and previous['checkpoint_admitted'],'Never continue a terminal/failed state')
            require(start['time']==previous['t'] and start['steps']==previous['steps'] and start['sha256']==records[-1]['checkpoint_sha256'],
                    'Restart continuity changed')
        data=np.atleast_1d(np.genfromtxt(raw/(command['args'][1]+'.history.csv'),delimiter=',',names=True))
        require(len(data)>0 and np.all(np.isfinite(data['t'])) and np.all(np.diff(data['t'])>=0),'Invalid history time ordering')
        require(data['t'][0]>=start['time']-1e-12 and data['t'][-1]<=report['t']+1e-10,'History outside accepted interval')
        checkpoint=command['args'][1]+'.checkpoint';sha=case['artifacts'][checkpoint]['sha256']
        if report['status']!='failed':
            cp=campaign.inspect_checkpoint(raw/checkpoint,variant,sha)
            require(report['checkpoint_admitted'] and cp['time']==report['t'] and cp['steps']==report['steps'],'Checkpoint admission/time mismatch')
            require((report['status']=='completed')==(cp['time']==campaign.END),'Incorrect completed time')
        else:
            require(not report['checkpoint_admitted'],'Failed checkpoint incorrectly admitted')
        records.append(dict(plan=relative,job_id=worker['job_id'],status=worker['status'],checkpoint_sha256=sha,artifacts=case['artifacts']))
        histories.append(data);reports.append(report);previous=report
    data,repeats=sg1.merge_histories(histories)
    return dict(history=data,reports=reports,records=records,restart_diagnostic_repeats=repeats)


def render(cases,output,title="Native N16 controls versus preserved SG1"):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    summary=dict(scope='Measured histories; own-candidate exterior cutoffs. No missing-value interpolation or endpoint radial reconstruction.',cases={})
    for name,case in cases.items():
        r=case['reports'][-1];data=case['history']
        boundary={};missing={}
        for key in ['max_null_surface','max_null_ghost','max_pole','max_clock']:
            values=[report.get(key) for report in case['reports']]
            missing[key]=sum(not isinstance(x,(float,int)) or not math.isfinite(x) for x in values)
            boundary[key]=None if missing[key] else max(values)
        summary['cases'][name]=dict(label=LABELS[name],status=r['status'],last_accepted_t_over_M=r['t_over_M'],
            last_diagnostic_t_over_M=float(data['t_over_M'][-1]),accepted_horizon_rows=int(np.sum(data['horizon_found']==1)),
            rows=len(data),final_accepted_mass_error=(3*r['horizon_irreducible_mass']-1) if r['horizon_found'] and r['horizon_irreducible_mass'] is not None else None,
            counters=sg1.counters(case['reports']),segments=case['records'],restart_diagnostic_repeats=case['restart_diagnostic_repeats'],
            boundary_report_maxima=boundary,missing_boundary_report_values=missing)
    specifications=[('histories.png',[
        ('H_exterior_L2','Exterior Hamiltonian RMS'),('M_exterior_L2','Exterior momentum RMS'),('Mr_exterior_L2','Exterior radial momentum RMS'),
        ('horizon_irreducible_mass','Accepted horizon mass / M − 1'),('null_surface','Scri null residual (maximum)'),('pole_max','Pole residual (maximum)')]),
        ('full-and-boundary.png',[
        ('H_L2','Full-domain Hamiltonian RMS'),('M_L2','Full-domain momentum RMS'),('Mr_L2','Full-domain radial momentum RMS'),
        ('null_ghost','Ghost null residual (maximum)'),('clock_max','Clock residual (maximum)'),('min_lapse','Minimum lapse')])]
    for filename,fields in specifications:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for name,case in cases.items():
            data=case['history']
            for ax,(key,label) in zip(axes.flat,fields):
                values=data[key]
                if key=='horizon_irreducible_mass':values=np.where(data['horizon_found']==1,3*values-1,np.nan)
                ax.plot(data['t_over_M'],values,'.-',ms=2.5,lw=1.2,color=COLORS[name],label=name)
                ax.set(xlabel='t / M',ylabel=label);ax.grid(alpha=.2)
                if key in ['null_surface','null_ghost','pole_max','clock_max']:ax.set_yscale('log')
        axes[0,0].legend(ncol=3);fig.suptitle(title)
        fig.tight_layout(rect=(0,.09,1,.96))
        fig.text(.5,.025,'Exterior norms use each run’s candidate horizon. Accepted mass gaps are retained.\nSame dt and floors; each named case isolates the documented initializer/gauge/operator change.',ha='center',fontsize=10)
        fig.savefig(output/filename,dpi=160);plt.close(fig)
    (output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--case',action='append',required=True,help='A=relative-plan01,relative-plan02 (repeat for other cases)')
    p.add_argument('--case-helper',action='append',default=[],help='C=/absolute/frozen/native_transition_campaign.py (repeat per case)')
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--baseline-plan',default='campaigns/n16-short-gauge-fresh10m-v1-segment01/plan.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from frozen_campaign import load
    helpers={}
    for value in a.case_helper:
        name,path=value.split('=',1);require(name not in helpers,'Duplicate case helper')
        helpers[name]=load(path)
    require(set(helpers).issubset({x.split('=',1)[0] for x in a.case}),'Helper has no requested case')
    cases={'SG1':sg1.load_campaign(a.baseline_root.resolve(),[a.baseline_plan],16)}
    for value in a.case:
        name,paths=value.split('=',1);require(name not in cases,'Duplicate case')
        cases[name]=load_native(a.root,name,paths.split(','),helpers.get(name))
    render(cases,a.output)


if __name__=='__main__':main()
