#!/usr/bin/env python3
"""Verify a pinned V9 duration campaign, preserving failed scientific outcomes."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def need(value,message):
    if not value:raise ValueError(message)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','admission','decoder','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);a=p.parse_args()
    need(not a.out.exists(),'Use a fresh verification output')
    adm=read(a.admission);need(adm['passed'] and sha(a.decoder)==adm['verifier_sha256'],'Admission/decoder mismatch')
    spec=importlib.util.spec_from_file_location('pinned_v9',a.decoder);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    raw=a.raw;plan=read(raw/'plan.json');worker=read(raw/'worker-status.json');bind=read(raw/'binding.json')
    sub=read(raw/'submission.json');col=read(raw/'collected.json');rel=read(raw/'allocation-released.json')
    need(sha(raw/'plan.json')==a.plan_sha256==worker['plan_sha256']==bind['plan_sha256']==sub['plan_sha256'],'Plan identity mismatch')
    need(bind['admission']['sha256']==sha(a.admission),'Admission binding mismatch')
    need(worker['source_manifest_sha256']==sub['source_manifest_sha256']==bind['source_manifest_sha256']==adm['source_manifest_sha256'],'Source mismatch')
    need(col['scheduler']['job_state'] in ['F','X'],'Job not terminal')
    need(rel['job_id']==sub['job_id']==col['job_id']==worker['job_id'],'Job identity mismatch');v.verify_release(rel)
    need(len(worker['cases'])==len(plan['commands'])==1,'Wrong case count')
    case=worker['cases'][0];command=plan['commands'][0];exe=command['executable']
    need(worker['binaries'][exe]==bind['binaries'][exe]==adm['binaries'][exe],'Binary mismatch')
    artifacts=[]
    for remote,item in case['artifacts'].items():
        path=raw/'output'/Path(remote).name
        need(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Artifact differs: '+remote)
        artifacts.append(dict(path=remote,**item))
    report=read(raw/'output'/Path(command['report']).name)
    cp=v.checkpoint(raw/'output'/(command['name']+'.checkpoint'))
    for ck,rk in [('magic','checkpoint_schema'),('time','t'),('dt','dt'),('steps','steps'),('admitted','checkpoint_admitted'),('header_bytes','checkpoint_header_bytes'),('cells','cells'),('nominal_cells','nominal_cells'),*[(k,k) for k in v.NORMAL_FIELDS]]:
        need(cp[ck]==report[rk],'Checkpoint/report mismatch: '+ck)
    need(cp['n']==16 and cp['kind']=={'minkowski':0,'puncture':2}[command['kind']] and cp['scri_dissipation_epsilon']==2,'Wrong physical configuration')
    checks=[]
    for rule in command['checks']:
        value,passed=v.check_value(rule,report);checks.append(dict(check=rule,observed=value,passed=passed))
    need(len(checks)==len(case['checks']) and all(x==y for x,y in zip(checks,case['checks'])),'Worker scientific gate replay differs')
    result=dict(verification_passed=True,scientific_checks_passed=all(x['passed'] for x in checks) and worker['status']=='passed' and col['scheduler'].get('Exit_status')==0,
        job_id=sub['job_id'],scheduler_state=col['scheduler']['job_state'],scheduler_exit_status=col['scheduler'].get('Exit_status'),worker_status=worker['status'],report=report,checkpoint=cp,checks=checks,artifacts=artifacts,
        source_manifest_sha256=worker['source_manifest_sha256'],binary_sha256=worker['binaries'][exe],plan_sha256=a.plan_sha256,script_sha256=sha(__file__))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['verification_passed','scientific_checks_passed','job_id','worker_status']}))
if __name__=='__main__':main()
