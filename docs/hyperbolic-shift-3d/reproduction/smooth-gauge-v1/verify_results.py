#!/usr/bin/env python3
"""Validate collected worker artifacts and both admitted GW1 3M checkpoints."""
from pathlib import Path
import argparse,json,hashlib,struct
from array import array
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
worker=json.loads((root/'worker-status.json').read_text());binding=json.loads((root/'binding.json').read_text());plan=json.loads((root/'plan.json').read_text());collected=json.loads((root/'collected.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert worker['status']=='passed' and len(worker['cases'])==2
assert collected['scheduler']['job_state'] in ['F','X']
assert binding['plan_sha256']==sha(root/'plan.json')==worker['plan_sha256']
assert worker['source_manifest_sha256']==binding['source_manifest_sha256']=='7308e7e5c3f54d960c0b4bae9b6cf92127a9aa2d66d7f46a5c88c0baaad64fdb'
assert worker['binaries']==binding['binaries']=={'build-sycl-mpi-smooth-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_smooth_gauge':'53eda0563a1b48846ac7372422906c35569f6ed685b75bb1dc980ccea83567f1'}
verified={};checkpoints=[]
for n,case,command in zip([16,24],worker['cases'],plan['commands']):
 assert case['name']==f'n{n}' and case['status']=='passed' and not case['timed_out'] and not case['storage_error']
 for name,v in case['artifacts'].items():
  relative=Path(name).relative_to(Path(plan['output_dir']).parent);local=root/relative
  assert local.stat().st_size==v['bytes'] and sha(local)==v['sha256'];verified[name]=v
 r=json.loads((root/f'output/n{n}.report.json').read_text());assert r['status']=='completed' and r['checkpoint_admitted'] and r['t']==1 and r['t_over_M']==3
 assert r['gauge_blend']=='smoothstep9_w_056_080' and r['complete_candidate_exterior_enabled']
 assert r['n']==n and r['dt']==.0005*16/n and r['backend']=='SYCL' and r['mpi_size']==12
 assert r['positive_variables']=='log_chi_log_lapse_rk_and_ko' and r['max_null_surface']<1e-10 and r['max_null_ghost']<1e-10
 data=(root/f'output/n{n}.checkpoint').read_bytes();h=struct.unpack('<16s4i3Q2d',data[:72]);nc=12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2
 assert h[:7]==(b'HYP3D_MPI_GW1'.ljust(16,b'\0'),n,31,2,1,nc,6*n*n*31)
 assert h[7:]==(r['steps'],1.,.0005*16/n) and len(data)==152+16*(31*nc+h[6])
 expected=bytes.fromhex('0000000000000040070000000000000003000000070000000400000003000000040000000200000003000000030000000ad7a3703d0ac73f7b14ae47e17ac43fbbbdd7d9df7cdb3d0100000000000000')
 assert data[72:152]==expected
 values=array('d');values.frombytes(data[152:]);import math
 assert all(map(math.isfinite,values)) and all(values[i]>0 for i in range(0,31*nc,31))
 checkpoints.append(dict(n=n,sha256=sha(root/f'output/n{n}.checkpoint'),bytes=len(data),steps=r['steps'],t_over_M=3))
result=dict(verified=True,artifacts=verified,checkpoints=checkpoints)
(root/'local-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
