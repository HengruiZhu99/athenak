#!/usr/bin/env python3
"""Read-only V6/V7 duration comparison with unchanged scientific limits."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/z4c').is_dir())
AURORA=ROOT/'docs/hyperbolic-shift-3d/reproduction/aurora'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

old=module('v5_comparison',AURORA/'stability-sampled-normal-minkowski-v1/analysis/compare.py')
base=old.base
def audit(raw,identity_path,admission_path,version,target,label):
    folder='preflight-donor-seven-v1' if version==6 else 'preflight-inner-transition-v1'
    decoder=module(f'v{version}_decoder',AURORA/folder/'verification/verify.py')
    identity=base.load(identity_path);admission=base.load(admission_path)
    base.require(identity['source_commit']==decoder.FROZEN,'Wrong frozen V6 source')
    base.require(base.sha(raw/'plan.json')==identity['plan_sha256'],'Plan differs from reviewed protocol')
    base.require(base.sha(admission_path)==identity['admission_sha256'],'Admission differs')
    base.require(base.sha(Path(decoder.__file__))==admission['verifier_sha256'],'Decoder differs from verified preflight')
    base.require(admission['passed'] is True and admission['phase']=='preflight'
        and admission['source_manifest_sha256']==identity['source_manifest_sha256'],'Invalid preflight admission')
    binding=base.load(raw/'binding.json');plan=base.load(raw/'plan.json')
    base.require(binding['admission']['sha256']==identity['admission_sha256'],'Bound admission differs')
    command=plan['commands'][0];exe=command['executable']
    base.require(len(plan['commands'])==1 and command['name']=='minkowski','Unexpected case coverage')
    base.require(admission['binaries'][exe]==identity['binary_sha256'],'Unadmitted binary')
    config=dict(executable=exe,checkpoint_schema=f'HYP3D_MPI_V{version}',checkpoint_header_bytes=104,
        scri_dissipation_epsilon=2.,scri_dissipation_degree=7,**decoder.NORMAL_FIELDS)
    base.require(config in admission['normal_derivative_configs'],'Unadmitted radial configuration')
    case,history=base.verify_case((f'v{version}',label,raw.name,
        identity['source_manifest_sha256'],identity['binary_sha256']),raw,verify_payloads=True)
    report=base.load(raw/'output/minkowski.report.json')
    cp=decoder.checkpoint(raw/'output/minkowski.checkpoint')
    for ck,rk in [('magic','checkpoint_schema'),('time','t'),('steps','steps'),('dt','dt'),
                  ('admitted','checkpoint_admitted'),('header_bytes','checkpoint_header_bytes'),
                  ('scri_dissipation_epsilon','scri_dissipation_epsilon'),('scri_dissipation_degree','scri_dissipation_degree'),
                  *[(k,k) for k in decoder.NORMAL_FIELDS]]:
        base.require(cp[ck]==report[rk],'Checkpoint/report differs: '+ck)
    base.require(all(report[k]==v for k,v in config.items() if k!='executable'),'Wrong radial configuration')
    base.require(command['args'][5:7]==['-','-'] and report['start_time']==0
        and float(command['args'][3])==report['target_time']==target and float(command['args'][8])==.05,
        'Not the declared fresh target-time trajectory')
    base.require(case['terminal']['terminal_verified'],'Allocation not terminal/released')
    passed=(report['status']=='completed' and report['t']==target and case['execution_succeeded']
        and case['terminal']['scheduler_exit_status']==0 and base.load(raw/'worker-status.json')['status']=='passed'
        and case['all_declared_checks_passed'] and case['all_v4_recorded_trajectory_limits_passed'])
    case.update(scientifically_admitted_for_this_bounded_test=passed,checkpoint=cp,
        target_time=target,preflight_admission=base.record(admission_path),reviewed_identity=base.record(identity_path),
        timing=old.timers(report,history),scope='Only the declared n16 Minkowski duration; no100M or puncture/convergence admission')
    keep=base.np.ones(len(history['t']),dtype=bool)
    keep[1:]=base.np.diff(history['t'])>report['dt']*1e-6
    case['endpoint_sensitivity']=dict(rule='Drop later sample when adjacent time gap <= dt*1e-6',
        removed_indices=base.np.flatnonzero(~keep),fit=base.growth({k:v[keep] for k,v in history.items()}))
    return case,history,report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();base.require(not args.out.exists(),'Output must be new')
    previous=AURORA/'stability-donor-seven-minkowski-v1'
    raw6=ROOT/'output/hyperbolic-core-shell/aurora/stability-donor-seven-minkowski-v1'
    v6,h6,r6=audit(raw6,previous/'analysis/identity.json',previous/'admission.json',6,1,'V6: seven-node radial closure')
    v7,h7,r7=audit(args.raw,args.identity,args.admission,7,2,'V7: logical inner-transition halos')
    args.out.mkdir(parents=True)
    base.write(args.out/'comparison.json',dict(evidence_verified=True,
        scientifically_admitted_for_this_bounded_test=v7['scientifically_admitted_for_this_bounded_test'],
        cases=[v6,v7],analysis=base.record(Path(__file__)),
        fit_selection='Unchanged amplitude window1e-10..1e-3, at least four samples; no fit below the window',
        warning='Bounded test acceptance is not a long-time stability, puncture, convergence or100M result'))
    old.plot(args.out,[v6,v7],[h6,h7])
    print(json.dumps(dict(evidence_verified=True,passed_this_duration=v7['scientifically_admitted_for_this_bounded_test'],
        native_status=r7['status'],reached=r7['t'],state=r7['max_state_deviation'],H=r7['H_L2'],M=r7['M_L2'])))

if __name__=='__main__':main()
