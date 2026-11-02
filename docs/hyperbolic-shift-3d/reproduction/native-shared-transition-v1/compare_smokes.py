#!/usr/bin/env python3
"""Compare admitted two-step SERIAL and SYCL smoke physical fields, not accuracy."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
import control
import native_transition_campaign as c


def compare(serial_prefix,gpu_root,gpu_plan,campaign_module=None):
    c=campaign_module or globals()['c']
    plan_path=control.local_path(gpu_root,gpu_plan);plan=json.loads(plan_path.read_text())
    assert plan['purpose']=='smoke'
    v=plan['variant'];worker=json.loads(plan_path.with_name('worker-status.json').read_text())
    verified=json.loads(plan_path.with_name('checkpoint-verified.json').read_text())
    assert worker['status']=='passed' and verified['verified'] and verified['variant']==v
    assert len(plan['commands'])==len(worker['cases'])==len(verified['cases'])==2
    assert [x['kind'] for x in plan['commands']]==['minkowski','puncture']
    assert worker['plan_sha256']==control.digest(plan_path)
    science=c.science(gpu_root)
    assert worker['source_manifest_sha256']==science['source_manifest_sha256']
    assert worker['binaries']=={c.binary(v):science['binaries'][c.binary(v)]}
    results=[]
    for command,case in zip(plan['commands'],worker['cases']):
        kind=command['kind']
        serial=Path(str(serial_prefix).format(kind=kind) if '{kind}' in str(serial_prefix) else str(serial_prefix)+'-'+kind)
        gpu=gpu_root/command['args'][1]
        reports=[json.loads(Path(str(p)+'.report.json').read_text()) for p in [serial,gpu]]
        assert reports[0]['backend']=='Serial' and reports[0]['mpi_size']==1
        assert reports[1]['backend']=='SYCL' and reports[1]['mpi_size']==12
        for report in reports:
            for check in c.required_checks(v,'smoke',kind):
                if check['path'] in [['backend'],['mpi_size']]:continue
                assert control.evaluate_check(report,check)['passed'],check
            assert report['t']==.001 and report['steps']==2
        checkpoints=[Path(str(p)+'.checkpoint') for p in [serial,gpu]]
        for p in checkpoints:c.inspect_checkpoint(p,v,control.digest(p),kind,.001)
        for name in [command['report'],command['args'][1]+'.checkpoint']:
            expected=case['artifacts'][name];p=gpu_root/name
            assert p.stat().st_size==expected['bytes'] and control.digest(p)==expected['sha256']
        admitted=next(x for x in verified['cases'] if x['kind']==kind)
        assert admitted['completed'] and admitted['sha256']==control.digest(checkpoints[1])
        arrays=[np.fromfile(p,dtype='<f8',offset=168) for p in checkpoints]
        nc=c.cells();ns=6*16*16;fields={}
        # Compare physical states separately from compensated-RK bookkeeping.
        for label,start,count in [('volume',0,nc),('volume_compensation',31*nc,nc),
                                  ('surface',62*nc,ns),('surface_compensation',62*nc+31*ns,ns)]:
            a,b=[x[start:start+31*count].reshape(count,31) for x in arrays]
            absolute=np.max(np.abs(a-b),axis=0)
            scale=1+np.maximum(np.max(np.abs(a),axis=0),np.max(np.abs(b),axis=0))
            fields[label]=dict(max_absolute_by_field=absolute.tolist(),scaled_max_by_field=(absolute/scale).tolist(),
                largest_scaled_difference=float(np.max(absolute/scale)))
        physical=max(fields[k]['largest_scaled_difference'] for k in ['volume','surface'])
        results.append(dict(variant=v,kind=kind,serial_checkpoint_sha256=control.digest(checkpoints[0]),
            gpu_checkpoint_sha256=control.digest(checkpoints[1]),fields=fields,physical_parity_within_1e_10=physical<=1e-10))
    return dict(scope='Two-step physical-state backend/partition comparison, scaled by 1+max field magnitude. Not evolution accuracy or convergence.',cases=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--serial-prefix',type=Path,required=True)
    p.add_argument('--gpu-root',type=Path,required=True);p.add_argument('--gpu-plan',required=True)
    p.add_argument('--campaign-helper',type=Path,help='Matching frozen helper; default preserves original science')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing comparison output')
    from frozen_campaign import load
    helper=load(a.campaign_helper) if a.campaign_helper else None
    r=compare(a.serial_prefix.resolve(),a.gpu_root.resolve(),a.gpu_plan,helper)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    if not all(x['physical_parity_within_1e_10'] for x in r['cases']):raise RuntimeError('Investigate measured backend/partition disagreement')


if __name__=='__main__':main()
