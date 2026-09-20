#!/usr/bin/env python3
"""Read-only package guards. This helper cannot submit or arm a job."""
import argparse,hashlib,json,re
from pathlib import Path
p=Path(__file__).resolve().parent

def check(require_ready=False):
 manifest=p/'package-manifest.json';m=json.loads(manifest.read_text())
 for name,digest in m['files'].items():
  path=p/name
  if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Package hash mismatch: '+name)
 for name in ['zero.athinput','pulse.athinput']:
  params={};section=None
  for line in (p/name).read_text().splitlines():
   line=line.split('#',1)[0].strip()
   if line.startswith('<'):section=line[1:-1];params[section]={}
   elif '='in line:
    k,v=map(str.strip,line.split('=',1))
    if k in params[section]:raise ValueError('Duplicate input key '+k)
    params[section][k]=v
  expectations={'mesh/nghost':'4','meshblock/nx1':'16','meshblock/nx2':'16','meshblock/nx3':'16','mesh_refinement/refinement':'static','time/tlim':'1000','time/nlim':'3'if name.startswith('zero')else'-1','z4c/damp_kappa1':'0','z4c/shift_eta':'0.02','z4c/residual_lapse_damping':'0.01','z4c/characteristic_bc_source':'zero_rate','problem/outer_sponge_start_radius':'8','problem/outer_sponge_ramp_width':'20','problem/outer_sponge_damping_time':'20','output2/single_file_per_rank':'true','problem/vacuum_gauge_pulse_amplitude':'0'if name.startswith('zero')else'1e-08'}
  for key,value in expectations.items():
   sec,item=key.split('/')
   if params[sec][item]!=value:raise ValueError('Unexpected input '+name+': '+key)
 digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
 if require_ready:
  ready=p/'READY.json'
  if not ready.is_file():raise ValueError('PREPARED ONLY: READY.json is absent; access, queue and package must be reviewed before submission')
  r=json.loads(ready.read_text())
  if r.get('package_sha256')!=digest or r.get('campaign')!='strongfield_smr232' or r.get('reviewed_after_access_renewal')is not True:raise ValueError('Invalid or stale readiness record')
 return {'passed':True,'prepared_only':not(p/'READY.json').exists(),'package_sha256':digest,'expected_ranks':24,'expected_blocks':232}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--require-ready',action='store_true');args=a.parse_args()
 try:r=check(args.require_ready)
 except Exception as e:print(str(e));raise SystemExit(1)
 print(json.dumps(r,indent=2))
