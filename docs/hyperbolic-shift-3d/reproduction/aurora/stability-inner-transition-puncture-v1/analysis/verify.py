#!/usr/bin/env python3
"""Independently check the frozen V7 puncture duration, including failed gates."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
ROOT=Path.cwd()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def require(x,msg):
 if not x:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 require(not a.out.exists(),'Use fresh output');a.out.mkdir(parents=True)
 admission_path=ROOT/'docs/hyperbolic-shift-3d/reproduction/aurora/preflight-inner-transition-v1/admission.json';admission=load(admission_path)
 decoder_path=admission_path.parent/'verification/verify.py';require(sha(decoder_path)==admission['verifier_sha256'],'Decoder hash differs')
 spec=importlib.util.spec_from_file_location('independent_v7_decoder',decoder_path);decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
 raw=a.raw;worker=load(raw/'worker-status.json');plan=load(raw/'plan.json');binding=load(raw/'binding.json');submission=load(raw/'submission.json');collected=load(raw/'collected.json')
 require(sha(raw/'plan.json')==worker['plan_sha256']==submission['plan_sha256']=='ef60d24bb3caea9d38ccd65309224489b501464b4ad20c8c1d71c6ba06f51469','Wrong plan')
 require(worker['source_manifest_sha256']==admission['source_manifest_sha256']==submission['source_manifest_sha256']=='da9304ba17ad5a1336aef4468dc88ce375622104bac2c69cc1d3cc534abcb2a4','Wrong source')
 require(binding['admission']['sha256']==sha(admission_path),'Admission binding differs')
 require(collected['scheduler']['job_state'] in ['F','X'],'Scheduler not terminal')
 require(load(raw/'allocation-released.json')['job_id']==submission['job_id']==collected['job_id']==worker['job_id'],'Job identity differs')
 require(len(worker['cases'])==len(plan['commands'])==1,'Case coverage differs');case=worker['cases'][0];command=plan['commands'][0]
 require(case['name']==command['name']=='puncture','Wrong case');exe=command['executable']
 require(worker['binaries'][exe]==admission['binaries'][exe]==binding['binaries'][exe],'Wrong binary')
 require(command['args'][2:7]==['puncture','1','.0005','inputs/puncture-n16.txt','-'],'Not the matched fresh puncture control')
 require(binding['inputs']=={'inputs/puncture-n16.txt':'c8ffa46d9007ee8b643db41980cfa4a08f9ab3f2f1c2383978a002f067f9db31'},'Wrong isotropic initial data')
 artifacts=[]
 for path,record in case['artifacts'].items():
  local=raw/'output'/Path(path).name
  require(local.stat().st_size==record['bytes'] and sha(local)==record['sha256'],'Artifact mismatch: '+path);artifacts.append(dict(path=path,**record))
 report=load(raw/'output/puncture.report.json');require(report['interpatch_interpolation']=='logical_inner_transition','Wrong interface policy');cp=decoder.checkpoint(raw/'output/puncture.checkpoint')
 for ck,rk in [('magic','checkpoint_schema'),('time','t'),('dt','dt'),('steps','steps'),('admitted','checkpoint_admitted'),('header_bytes','checkpoint_header_bytes'),*[(k,k) for k in decoder.NORMAL_FIELDS]]:
  require(cp[ck]==report[rk],'Report/checkpoint differs: '+ck)
 require(cp['n']==16 and cp['kind']==2 and cp['dt']==.0005 and cp['scri_dissipation_epsilon']==2,'Wrong configuration')
 checks=[]
 for rule in command['checks']:
  value=report
  for key in rule['path']:value=value[key]
  op=rule['op'];finite=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
  passed=value==rule.get('value') if op=='eq' else finite and (op=='finite' or (op=='le' and value<=rule['value']) or (op=='ge' and value>=rule['value']))
  checks.append(dict(check=rule,observed=value,passed=bool(passed)))
 require(len(checks)==len(case['checks']) and all(x['check']==y['check'] and x['passed']==y['passed'] for x,y in zip(checks,case['checks'])),'Worker gate replay differs')
 result=dict(verification_passed=True,scientific_checks_passed=all(x['passed'] for x in checks) and worker['status']=='passed' and collected['scheduler'].get('Exit_status')==0,job_id=submission['job_id'],scheduler_state=collected['scheduler']['job_state'],scheduler_exit_status=collected['scheduler'].get('Exit_status'),worker_status=worker['status'],report=report,checkpoint=cp,checks=checks,artifacts=artifacts,source_manifest_sha256=worker['source_manifest_sha256'],plan_sha256=worker['plan_sha256'],binary_sha256=worker['binaries'][exe],script_sha256=sha(__file__))
 (a.out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ['verification_passed','scientific_checks_passed','job_id','scheduler_state','scheduler_exit_status','worker_status']},indent=2))
if __name__=='__main__':main()
