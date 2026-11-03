"""Read-only analysis of this completed run's explicit penalty-policy failure.

Does not change the frozen verifier, worker result, binding, or restart policy.
A native-admitted finite10M state may be diagnosed, but the campaign remains failed.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
from frozen_campaign import load

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def load_completed():
 raw=ROOT/'output/restored-six-shells/aurora';relative='campaigns/n16-restored-six-shells-v1-a-evolution-segment01/plan.json';path=raw/relative
 c=load(raw/'launcher-restored-six-shells-v1/native_transition_campaign.py');ctl=c.control
 plan=read(path);binding=read(path.with_name('binding.json'));worker=read(path.with_name('worker-status.json'));sub=read(path.with_name('submission.json'));col=read(path.with_name('collected.json'))
 science=c.science(raw);prov=read(raw/'launcher-restored-six-shells-v1/provenance.json')
 for name,v in prov['files'].items():
  p=raw/'launcher-restored-six-shells-v1'/name;assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
 assert digest(raw/c.MANIFEST)==science['source_manifest_sha256']==prov['source_manifest_sha256']
 c.validate_native_plan(raw,plan)
 assert plan['segment']==1 and plan['variant']=='A' and plan['purpose']=='evolution'
 assert sub['job_id']==worker['job_id']==col['job_id'] and sub['plan']==relative
 assert col['returncode']==0 and col['scheduler']['job_state']=='F' and col['scheduler']['Exit_status']==1
 assert digest(path)==binding['plan_sha256']==worker['plan_sha256']==sub['plan_sha256']
 binaries={c.binary('A'):science['binaries'][c.binary('A')]}
 for item in [binding,worker]:
  assert item['binaries']==binaries and item['source_manifest_sha256']==science['source_manifest_sha256']
 assert sub['source_manifest_sha256']==science['source_manifest_sha256']
 assert binding['admission']==c.validate_science_binding(raw,plan,science['source_manifest_sha256'],binaries)
 assert binding['inputs']=={c.SCIENCE:digest(raw/c.SCIENCE)}
 assert worker['status']=='failed' and len(worker['cases'])==1
 case=worker['cases'][0];cmd=plan['commands'][0]
 assert case['status']=='failed' and case['returncode']==0 and not case['timed_out'] and not case['storage_error'] and not case.get('report_error')
 report=read(raw/cmd['report']);assert report['status']=='completed' and report['error']=='' and report['checkpoint_admitted'] and report['t_over_M']==10 and report['steps']==6667
 results=[ctl.evaluate_check(report,x) for x in cmd['checks']]
 assert results==case['checks']
 failures=[x for x in results if not x['passed']]
 assert len(failures)==1 and failures[0]['check']==dict(path=['scri_penalty_scale_min'],op='eq',value=1)
 assert abs(report['scri_penalty_scale_min']-.5*(1-1e-12)/.768)<1e-14
 artifacts=case['artifacts'];assert len(artifacts)==6
 for name,v in artifacts.items():
  p=raw/name;assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256'],name
 cp=cmd['args'][1]+'.checkpoint';inspected=c.inspect_checkpoint(raw/cp,'A',artifacts[cp]['sha256'])
 assert inspected['time']==report['t']==c.END and inspected['steps']==report['steps'] and inspected['dt']==report['dt']
 history=np.atleast_1d(np.genfromtxt(raw/(cmd['args'][1]+'.history.csv'),delimiter=',',names=True))
 assert np.isfinite(history['t']).all() and np.all(np.diff(history['t'])>0) and history['t'][0]==0 and history['t'][-1]==c.END
 audit=dict(native_completed=True,campaign_accepted=False,restart_allowed=False,reason='Frozen acceptance check rejects scri penalty limiter activation; preserve rejection. Native admission supports read-only endpoint diagnostics, not a matched-parameter refinement claim.',job_id=sub['job_id'],scheduler_exit=1,worker_status='failed',native_returncode=0,failed_checks=failures,checkpoint=inspected,artifacts=artifacts,source_manifest_sha256=science['source_manifest_sha256'],binary_sha256=binaries[c.binary('A')],plan_sha256=digest(path),native_elapsed_seconds=case['elapsed_seconds'])
 return dict(history=history,reports=[report],records=[dict(plan=relative,job_id=sub['job_id'],status='completed_but_penalty_policy_failed',checkpoint_sha256=inspected['sha256'],artifacts=artifacts)],restart_diagnostic_repeats=0),audit

if __name__=='__main__':
 case,audit=load_completed();(HERE/'evolution-disposition.json').write_text(json.dumps(audit,indent=2)+'\n')
 (HERE/'evolution-local-verification.json').write_text(json.dumps(dict(verified=True,scope='All worker artifacts locally hash/length verified; campaign acceptance remains failed.',artifacts=audit['artifacts']),indent=2)+'\n')
 print(json.dumps({k:v for k,v in audit.items() if k!='artifacts'},indent=2))
