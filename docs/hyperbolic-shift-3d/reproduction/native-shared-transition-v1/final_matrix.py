#!/usr/bin/env python3
"""Read preserved matrix evidence, including E's explicitly adjudicated dual failure.

Does not admit a checkpoint, repair worker metadata, restart, or submit any job.
"""
import json,struct,hashlib,sys,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
import compare_histories as h
from frozen_campaign import load
RAW=ROOT/'output/native-shared-transition/aurora'
FROZEN=load(RAW/'launcher-native-transition-ko-dependencies-v1/native_transition_campaign.py')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_e():
 rel='campaigns/n16-native-transition-ko-dependencies-v1-e-evolution-segment01/plan.json'
 p=RAW/rel;d=p.parent;plan=read(p);worker=read(d/'worker-status.json');collected=read(d/'collected.json');binding=read(d/'binding.json');submission=read(d/'submission.json');inventory=read(d/'postmortem-artifacts.json')['artifacts']
 FROZEN.validate_native_plan(RAW,plan)
 assert plan['variant']=='E' and plan['segment']==1 and plan['purpose']=='evolution'
 assert sha(p)==worker['plan_sha256']==binding['plan_sha256']==submission['plan_sha256']=='68a673727fadc4341ab816d965fbabbb6b7c8798022add9494f6ed138b03abd3'
 assert worker['job_id']==collected['job_id']==submission['job_id']=='9032187.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov'
 assert collected['scheduler']['job_state']=='F' and collected['scheduler']['Exit_status']==1
 science=FROZEN.science(RAW);expected={FROZEN.binary('E'):science['binaries'][FROZEN.binary('E')]}
 assert sha(RAW/FROZEN.MANIFEST)==science['source_manifest_sha256']
 for record in [worker,binding]:
  assert record['binaries']==expected and record['source_manifest_sha256']==science['source_manifest_sha256']
 assert binding['inputs'][FROZEN.SCIENCE]==sha(RAW/FROZEN.SCIENCE)
 assert sha(RAW/FROZEN.SCIENCE)==read((RAW/FROZEN.SCIENCE).parent/'provenance.json')['files']['native-transition-science.json']['sha256']
 # No silent relaxation of ordinary loader: this exact historical E failure is
 # checked against a separate post-terminal inventory, retaining original errors.
 for name,v in inventory.items():
  f=RAW/name
  # collect/release legitimately refreshed the local/remote scheduler envelope;
  # local original collected.json stays the original inventoried version.
  assert f.stat().st_size==v['bytes'] and sha(f)==v['sha256'],name
 assert worker['status']=='failed' and len(worker['cases'])==1
 case=worker['cases'][0];assert case['returncode']==2 and not case['timed_out'] and case['storage_error']
 assert 'cap/reserve exceeded' in case['storage_error'] and 'cap/reserve exceeded' in worker['reason']
 report=read(d/'output/puncture.report.json');assert report['status']=='failed' and not report['checkpoint_admitted'] and report['checkpoint_written']
 assert report['error']=='device RHS failure at active cell 57352 reason 2'
 for check in FROZEN.required_checks('E','evolution','puncture',False):
  if check['op']=='eq' and check['path'] not in [['status'],['checkpoint_admitted']]:assert h.control.evaluate_check(report,check)['passed'],check
 cp=d/'output/puncture.checkpoint';b=cp.read_bytes();header=struct.unpack('<16s4i3Q2d',b[:72])
 assert header[:7]==(b'HYP3D_MPI_NS2'.ljust(16,b'\0'),16,31,2,0,FROZEN.cells(),6*16*16*31)
 assert b[72:168]==FROZEN.CONFIG and len(b)==79725224
 assert header[7:]==(report['steps'],report['t'],.0005) and report['steps']==6260
 capture=d/'output/puncture.failed-stage.bin';u=np.fromfile(capture,dtype='<f8');assert len(u)==31*(report['cells']+6*16*16)
 v=u[:31*report['cells']].reshape(-1,31);surface=u[31*report['cells']:].reshape(-1,31)
 def eigen(a):
  g=np.zeros((len(a),3,3))
  for k,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=a[:,k+1]+(i==j)
  return np.linalg.eigvalsh(g)
 assert np.isfinite(u).all();ev=eigen(v);se=eigen(surface);bad=np.where(ev[:,0]<=0)[0];assert len(bad)==8 and bad[0]==57352
 coord=ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv';coords=np.genfromtxt(coord,delimiter=',',names=True);assert sha(coord)=='c3b0ae43c916677a3f717ee7754e4b1f6ff970a9925c8666de50b1bff2c44b07'
 locations=[]
 for i in bad:
  r=float(coords['r'][i]);locations.append(dict(cell=int(i),x=float(coords['x'][i]),y=float(coords['y'][i]),z=float(coords['z'][i]),r=r,chi=float(v[i,0]),conformal_lapse=float(v[i,18]+(1+r*r)/2),metric_eigenvalues=ev[i].tolist(),determinant=float(np.prod(ev[i]))))
 data=np.atleast_1d(np.genfromtxt(d/'output/puncture.history.csv',delimiter=',',names=True));assert np.isfinite(data['t']).all() and np.all(np.diff(data['t'])>0) and data['t'][0]==0 and data['t'][-1]<=report['t']
 log=(d/'output/puncture.log').read_text();assert 'status=failed t=3.13 error=device RHS failure at active cell 57352 reason 2' in log and 'rank 0 exited with code 2' in log
 disposition=dict(scope='Postmortem only: native numerical failure independently demonstrated, concurrent infrastructure reserve failure retained; no healthy admission or restart.',worker_infrastructure_error=case['storage_error'],worker_final_error=worker['reason'],worker_sent_sigterm_on_storage_error=True,signal_vs_native_failure_order='Exact ordering not recorded; finite indefinite failed candidate and complete native failure report independently demonstrate numerical failure. Do not claim storage guard absent or pure scheduler numerical disposition.',inventory_sha256=sha(d/'postmortem-artifacts.json'),last_accepted_t_over_M=report['t_over_M'],last_diagnostic_t_over_M=float(data['t_over_M'][-1]),rejected_stage_abscissa='not recorded',bad_cells=locations,volume_nonfinite_values=0,surface_min_metric_eigenvalue=float(se.min()),checkpoint_admitted=False,checkpoint_sha256=sha(cp),failed_stage_sha256=sha(capture),last_candidate_cutoff=float(data['horizon_trial_radius'][-1]) if 'horizon_trial_radius' in data.dtype.names else report['horizon_trial_radius'],last_horizon_accepted=bool(data['horizon_found'][-1]),candidate_norm_qualification='Last rejected candidate cutoff0.740995 lies outside failure radius0.609375. Small candidate-exterior norms exclude the failing shell and are not a matched exterior accuracy improvement.')
 (HERE/'aurora/e-failure/disposition.json').write_text(json.dumps(disposition,indent=2)+'\n')
 records=[dict(plan=rel,job_id=worker['job_id'],status='failed',checkpoint_sha256=sha(cp),artifacts=inventory,worker_storage_error=case['storage_error'],analysis_disposition='Explicit dual-failure postmortem; original worker metadata unchanged')]
 return dict(history=data,reports=[report],records=records,restart_diagnostic_repeats=0)
def main():
 cases={'SG1':h.sg1.load_campaign(ROOT/'output/hyperbolic-core-shell/short-gauge-v1/aurora',['campaigns/n16-short-gauge-fresh10m-v1-segment01/plan.json'],16)}
 for c,namespace in [('A','v2'),('B','v1'),('C','ko-dependencies-v1'),('D','ko-dependencies-v1')]:
  cases[c]=h.load_native(RAW,c,[f'campaigns/n16-native-transition-{namespace}-{c.lower()}-evolution-segment01/plan.json'],FROZEN if c in 'CD' else None)
 cases['E']=load_e();h.render(cases,args.output)
if __name__=='__main__':main()
