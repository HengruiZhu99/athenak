#!/usr/bin/env python3
"""Audit collected original CPU-admitted Minkowski GPU results; never submits jobs."""
import argparse,json,hashlib,csv,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing GPU audit is immutable')
 receipt=json.loads((a.gpu/'gpu-receipt.json').read_text());checks=[];integrity=[];backend=[];histories=[]
 remote=json.loads((a.gpu.parent/'remote-manifest.json').read_text());expected_source=json.loads((a.evidence/'aurora/portable-source-manifest.json').read_text());source_match=remote['source_files']==expected_source and not remote['source_mismatches'];admission_match=remote['admission_sha256']==sha(a.evidence/'aurora/cpu-admission.json')
 cpu=[a.evidence/'cpu-v2/minkowski-n12',a.evidence/'cpu-v3/minkowski-n18',a.evidence/'cpu-v3/minkowski-n24']
 expected_exe=(a.gpu/'executable.sha256').read_text().split()[0];executable_match=expected_exe==sha(a.gpu/'test_scalar')
 for n,reference,record in zip([12,18,24],cpu,receipt['records']):
  run=a.gpu/f'minkowski-n{n}';mismatches=[name for name,digest in record['files'].items() if not (run/name).is_file() or sha(run/name)!=digest]
  command=record['command'];args=command[command.index('--n'):];argument_match=record['n']==n and len(args)==6 and args[:4]==['--n',str(n),'--tlim','12'] and args[4]=='--output'
  frozen=json.loads((run/'frozen-integrity.json').read_text());integrity.append(dict(n=n,returncode=record['returncode'],complete=(run/'complete.txt').exists(),mismatches=mismatches,arguments_match=argument_match,frozen=frozen,passed=not mismatches and argument_match and record['returncode']==0 and (run/'complete.txt').exists() and frozen['unchanged']))
  configuration=(run/'backend.txt').read_text();backend.append(dict(n=n,SYCL='SYCL' in configuration,Intel_GPU=('GPU' in configuration and ('Intel' in configuration or 'INTEL_PVC' in configuration))))
  histories.append({(float(r['time']),r['region']):r for r in csv.DictReader((run/'history.csv').open())})
  initial=np.fromfile(reference/'fields-t0.000.f64').reshape(-1,12);scale=np.max(abs(initial[:,4:9]),axis=0)
  for t in [2,6,12]:
   g=np.fromfile(run/f'fields-t{t:.3f}.f64').reshape(-1,12);c=np.fromfile(reference/f'fields-t{t:.3f}.f64').reshape(-1,12)
   coordinate=float(np.max(abs(g[:,:4]-c[:,:4])/(1+abs(c[:,:4]))));scaled=float(np.max(abs(g[:,4:9]-c[:,4:9])/np.maximum(scale,1e-12)));derivative_scale=float(np.max(np.linalg.norm(initial[:,6:9],axis=1)));derivative=float(np.max(abs(g[:,9:12]-c[:,9:12]))/derivative_scale)
   checks.append(dict(n=n,time=t,coordinate_weight_relative=coordinate,scaled_max_field_difference=scaled,scaled_max_derivative_difference=derivative,passed=coordinate<1e-11 and scaled<1e-9 and derivative<1e-9))
 convergence=[]
 for region in ['all','bulk','shell']:
  for t in [2,6,12]:
   for field in ['psi','v','Dpsi','reduction']:
    for norm in ['L2','Linf']:
     errors=[float(h[t,region][field+'_'+norm]) for h in histories];orders=[math.log(errors[i]/errors[i+1])/math.log([1.5,4/3][i]) for i in [0,1]];convergence.append(dict(time=t,region=region,field=field,norm=norm,orders=orders,passed=errors[0]>errors[1]>errors[2] and min(orders)>=1.5))
 passed=len(receipt['records'])==3 and executable_match and source_match and admission_match and all(c['passed'] for c in checks+convergence+integrity) and all(c['SYCL'] and c['Intel_GPU'] for c in backend)
 result=dict(stage='original-minkowski-gpu',passed=passed,source_manifest_match=source_match,cpu_admission_hash_match=admission_match,cpu_comparisons=checks,convergence=convergence,integrity=integrity,backend=backend,executable_sha256=expected_exe,executable_hash_match=executable_match,scope='GPU three-grid analytic convergence plus pointwise CPU agreement. CPU timestep and boundary controls are inherited only for the identical original admitted equations and arguments; no Schwarzschild/Kerr admission.',source_sha256=sha(Path(__file__)))
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(passed=passed,backend=backend,executable_hash_match=executable_match),indent=2))
if __name__=='__main__':main()
