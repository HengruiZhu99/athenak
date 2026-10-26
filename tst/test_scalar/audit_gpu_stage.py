#!/usr/bin/env python3
"""Audit a collected source-bound GPU matrix against its admitted CPU cases."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot,history

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--cpu-review',type=Path,required=True);p.add_argument('--physics-review',type=Path);p.add_argument('--additional-audit',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing GPU audit is immutable')
 admission=json.loads(a.admission.read_text());cpu=json.loads(a.cpu_review.read_text());remote=json.loads((a.gpu.parent/'remote-manifest.json').read_text());collection=json.loads((a.gpu.parent/'collection-receipt.json').read_text());receipt=json.loads((a.gpu/'gpu-receipt.json').read_text());expected=json.loads(a.source_manifest.read_text());ns=admission['resolutions'];refs=[Path(d) for d in cpu['runs']];checks=[];integrity=[];backends=[]
 if not admission['passed'] or len(ns)!=3 or cpu['resolutions']!=ns:raise RuntimeError('A completed matching CPU scientific admission is required')
 source_match=remote['source_files']==expected and not remote['source_mismatches'] and all(expected.get('source/src/test_scalar/'+name)==digest for name,digest in admission['delivery_source_sha256'].items()) and bool(admission.get('dependency_source_sha256')) and all(expected.get('source/'+name)==digest for name,digest in admission['dependency_source_sha256'].items())
 admission_match=remote['admission_sha256']==sha(a.admission)==receipt['scientific_admission_sha256'];exe_match=(a.gpu/'executable.sha256').read_text().split()[0]==sha(a.gpu/'test_scalar')
 if len(receipt['records'])!=3:raise RuntimeError('GPU matrix is incomplete; preserve checkpoints and resume missing prefixes')
 for n,reference,case,record in zip(ns,refs,admission['gpu_cases'],receipt['records']):
  run=a.gpu/f'{admission["stage"]}-n{n}';frozen=json.loads((run/'frozen-integrity.json').read_text());params=json.loads((run/'parameters.json').read_text());cp=json.loads((reference/'parameters.json').read_text());mismatch=[name for name,digest in record['files'].items() if not (run/name).is_file() or sha(run/name)!=digest];cmd=record['command'];args=cmd[cmd.index('--n'):];argument_match=args[:-4]==case['arguments'] and args[-4]=='--wall-seconds' and args[-2]=='--output';parameter_match=params==cp
  integrity.append(dict(n=n,passed=record['returncode']==0 and (run/'complete.txt').exists() and frozen['unchanged'] and not mismatch and argument_match and parameter_match,mismatches=mismatch,arguments_match=argument_match,parameters_match=parameter_match,frozen=frozen))
  backend=(run/'backend.txt').read_text();backends.append(dict(n=n,SYCL='Default Device: SYCL' in backend,Intel_GPU='Is GPU: 1' in backend and 'Intel' in backend))
  initial=snapshot(reference,0);scale=max(float(np.max(abs(initial[:,4:9]))),1e-12)
  for t in [0,2,6,12]+([100] if admission['stage']!='minkowski' else []):
   x,y=snapshot(run,t),snapshot(reference,t);coordinate=float(np.max(abs(x[:,:4]-y[:,:4])/(1+abs(y[:,:4]))));delta=float(np.max(abs(x[:,4:]-y[:,4:]))/scale);checks.append(dict(n=n,time=t,joint_initial_scale=scale,coordinate_weight_relative=coordinate,scaled_max_field_derivative_difference=delta,passed=coordinate<1e-11 and delta<1e-9))
 convergence=[];physics=None
 if admission['stage']=='minkowski':
  hs=[history(a.gpu/f'minkowski-n{n}') for n in ns]
  for region in ['all','bulk','shell']:
   for t in [2,6,12]:
    for field in ['psi','v','Dpsi','reduction']:
     for norm in ['L2','Linf']:
      errors=[float(h[t,region][field+'_'+norm]) for h in hs];orders=[math.log(errors[k]/errors[k+1])/math.log(ns[k+1]/ns[k]) for k in [0,1]];convergence.append(dict(region=region,time=t,field=field,norm=norm,orders=orders,passed=errors[0]>errors[1]>errors[2] and min(orders)>=1.5))
  initial=snapshot(a.gpu/f'minkowski-n{ns[-1]}',0);scale=float(np.linalg.norm(initial[:,6:9],axis=1).max());maximum=max(float(r['reduction_Linf']) for (t,region),r in hs[-1].items() if region=='all');physics=dict(passed=all(r['passed'] for r in convergence) and maximum/scale<=.02,normalized_reduction_max=maximum/scale)
 else:
  if not a.physics_review:raise RuntimeError('Direct independent field/QNM GPU analysis is required for black holes')
  pr=json.loads(a.physics_review.read_text());same_runs=[Path(d).resolve() for d in pr['runs']]==[(a.gpu/f'{admission["stage"]}-n{n}').resolve() for n in ns];oracle_binding=pr['oracle']['files']==cpu['oracle']['files'] and pr.get('reference_sampler','pchip')==cpu.get('reference_sampler','pchip') and pr.get('interpolator_sha256')==cpu.get('interpolator_sha256');physics=dict(path=str(a.physics_review),sha256=sha(a.physics_review),oracle_binding=oracle_binding,passed=same_runs and oracle_binding and pr['complete'] and pr['field_checks_passed'] and pr['finest_accuracy_passed'] and pr['reduction'][-1]['passed'])
 additional=[dict(path=str(path),sha256=sha(path),passed=json.loads(path.read_text()).get('passed',False)) for path in a.additional_audit]
 passed=collection['passed'] and source_match and admission_match and exe_match and physics['passed'] and all(c['passed'] for c in checks+integrity+additional) and all(c['SYCL'] and c['Intel_GPU'] for c in backends)
 result=dict(stage=admission['stage']+'-gpu',passed=passed,source_manifest_match=source_match,cpu_admission_match=admission_match,executable_match=exe_match,comparisons=checks,integrity=integrity,backend=backends,convergence=convergence,physics=physics,additional_audits=additional,scope='Identical source-bound equations and arguments; direct three-grid science and CPU pointwise agreement. Inherit only the matching CPU timestep/inner/outer controls. This does not complete metric evolution.',source_sha256=sha(Path(__file__)))
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(passed=passed,source_match=source_match,admission_match=admission_match,physics=physics),indent=2))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
