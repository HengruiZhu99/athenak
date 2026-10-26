#!/usr/bin/env python3
"""Audit source-bound GPU trajectories completed across exact checkpoints."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot,parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);p.add_argument('--cpu-review',type=Path,required=True);p.add_argument('--physics-review',type=Path,required=True);p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--collections',nargs='+',type=Path,required=True);p.add_argument('--prepared',nargs='+',type=Path,required=True);p.add_argument('--additional-audit',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing GPU continuation audit is immutable')
proof=read(a.admission);cpu=read(a.cpu_review);physics=read(a.physics_review);stage=proof['stage'];ns=proof['resolutions'];collections=[];integrity=[];comparisons=[];binary=None
if not proof['passed'] or cpu['resolutions']!=ns or sha(a.cpu_review)!=proof['primary_review_sha256'] or len(a.collections)!=len(a.prepared):raise RuntimeError('Matching complete CPU admission and GPU collection/source pairs required')
for directory,prepared in zip(a.collections,a.prepared):
 receipt=read(directory/'collection-receipt.json');manifest=read(directory/'remote-manifest.json');expected=read(prepared/'portable-source-manifest.json');job=manifest['scheduler']['Jobs'][receipt['job']];digest=sha(directory/'result/test_scalar');binary=binary or digest;source_ok=manifest['source_files']==expected and not manifest['source_mismatches'] and all(expected.get('source/src/test_scalar/'+name)==value for name,value in proof['delivery_source_sha256'].items()) and all(expected.get('source/'+name)==value for name,value in proof['dependency_source_sha256'].items());limits=job['job_state']=='F' and job['Exit_status']==0 and job['queue']=='debug' and job['Account_Name']=='CompactBinaryMerger' and job['Resource_List']['nodect']==1 and job['Resource_List']['walltime']=='01:00:00'
 collections.append(dict(directory=str(directory),receipt_sha256=sha(directory/'collection-receipt.json'),passed=receipt['passed'] and source_ok and limits and manifest['admission_sha256']==sha(a.admission) and digest==binary and (directory/'result/executable.sha256').read_text().split()[0]==digest,source_match=source_ok,limits_match=limits))
for n,run,reference in zip(ns,a.runs,map(Path,cpu['runs'])):
 meta=read(reference/'parameters.json');current=run;previous_frozen=None;lineage=[];seen=set();fingerprints=['physical_before','reference_before','physical_derivatives_before','reference_derivatives_before']
 while current is not None:
  if str(current.resolve()) in seen:raise RuntimeError('GPU checkpoint ancestry cycle')
  seen.add(str(current.resolve()));restart=(current/'restart-invocation.json').exists();inv=read(current/('restart-invocation.json' if restart else 'invocation.json'));receipt=read(current/('restart-receipt.json' if restart else 'receipt.json'));g=read(current/'frozen-integrity.json');changed=[name for name,digest in receipt['files'].items() if not (current/name).is_file() or sha(current/name)!=digest];backend=(current/'backend.txt').read_text();device='Default Device: SYCL' in backend and 'Is GPU: 1' in backend and 'Intel' in backend;geometry_same=previous_frozen is None or all(g[k]==previous_frozen[k] for k in fingerprints);native_binary=inv['binary_sha256'] if restart else inv['binary'];lineage.append(dict(run=str(current),passed=receipt['returncode']==0 and not changed and g['unchanged'] and geometry_same and native_binary==binary and read(current/'parameters.json')==meta and device,changed_files=changed,actual_SYCL_Intel_GPU=device,frozen=g));previous_frozen=g;current=parent(current)
 complete=(run/'complete.txt').exists() and 'time=100 ' in (run/'complete.txt').read_text();integrity.append(dict(n=n,passed=complete and all(row['passed'] for row in lineage),lineage=lineage))
 initial=snapshot(reference,0);scale=max(float(np.max(abs(initial[:,4:9]))),1e-12)
 for t in [0,2,6,12,100]:
  x,y=snapshot(run,t),snapshot(reference,t)
  if x.shape!=y.shape:raise RuntimeError('GPU and CPU field layouts differ')
  coordinate=float(np.max(abs(x[:,:4]-y[:,:4])/(1+abs(y[:,:4]))));delta=float(np.max(abs(x[:,4:]-y[:,4:]))/scale);comparisons.append(dict(n=n,time=t,joint_initial_scale=scale,coordinate_weight_relative=coordinate,scaled_max_field_derivative_difference=delta,passed=coordinate<1e-11 and delta<1e-9))
physics_bound=physics['runs']==[str(run.resolve()) for run in a.runs] and physics['oracle']==cpu['oracle'] and physics.get('reference_sampler','pchip')==cpu.get('reference_sampler','pchip') and physics.get('interpolator_sha256')==cpu.get('interpolator_sha256') and physics['stage']==stage and physics['resolutions']==ns and physics['complete'] and physics['field_checks_passed'] and physics['finest_accuracy_passed'] and physics['reduction'][-1]['passed'];additional=[dict(path=str(path),sha256=sha(path),passed=read(path)['passed']) for path in a.additional_audit];passed=physics_bound and all(row['passed'] for row in collections+integrity+comparisons+additional)
result=dict(stage=stage+'-gpu',passed=passed,cpu_admission_sha256=sha(a.admission),source_manifest_match=all(row['source_match'] for row in collections),collections=collections,integrity=integrity,comparisons=comparisons,physics=dict(path=str(a.physics_review),sha256=sha(a.physics_review),passed=physics_bound),additional_audits=additional,source_sha256=sha(Path(__file__)),scope='All collected scheduler/source/executable/native-payload checks; exact same-binary checkpoint ancestry and all four frozen fingerprints; actual SYCL Intel GPU; direct independent three-grid field/derivative/reduction/waveform/free scalar-QNM gates and strict CPU comparison. Inherit only the matching admitted CPU controls. Scalar success does not complete metric evolution.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=passed,maximum_CPU_GPU_difference=max(row['scaled_max_field_derivative_difference'] for row in comparisons)),indent=2))
if not passed:raise SystemExit(1)
