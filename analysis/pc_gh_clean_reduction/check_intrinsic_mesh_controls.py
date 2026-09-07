#!/usr/bin/env python3
"""Reject unsupported intrinsic mesh paths and verify 50-field output identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--restart',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
t=a.input.read_text();records=[]

def run(name,text,expect=None,extra=()):
 d=a.output/name;d.mkdir();f=d/'used.athinput';f.write_text(text)
 cmd=[str(a.binary.resolve()),'-i',str(f.resolve()),*extra];start=time.time()
 with (d/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 log=(d/'run.log').read_text();ok=(r.returncode==0) if expect is None else (r.returncode!=0 and expect in log)
 records.append(dict(name=name,command=cmd,returncode=r.returncode,expected_message=expect,status='PASS' if ok else 'FAIL',wall_seconds=time.time()-start))
 (a.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n');assert ok,(name,log)
 return d

for key in ['project_gauge_constraints','project_reduction_constraints']:
 run(key,t.replace('<pc_gh>',f'<pc_gh>\n{key} = true'),'both projections off')
run('unsupported-tracker-metadata',t.replace('<pc_gh>','<pc_gh>\nrestart_tracker_state = true'),'tracker restart state is not supported')
run('outflow',t.replace('ix1_bc = periodic','ix1_bc = outflow').replace('ox1_bc = periodic','ox1_bc = outflow'),'uniform periodic boundaries')
run('legacy-pgen',t.replace('intrinsic_smooth','pc_gh_minkowski'),'initial data must be')
run('legacy-option',t.replace('<pc_gh>','<pc_gh>\ngauge = harmonic'),'unsupported option gauge')
run('negative-ko',t.replace('dissipation = 0.3','dissipation = -0.3'),'rates must be finite')
run('unknown-rate',t.replace('reduction_profile = constant','reduction_profile = invalid').replace('reduction_profile = lapse_scaled','reduction_profile = invalid'),'reduction_profile must be')
run('rk4',t.replace('integrator = rk3','integrator = rk4'),'requires rk3')
run('history',t+'\n<output2>\nfile_type = hst\ndt = 0.1\n','history diagnostics are not integrated')
run('legacy-constraints',t+'\n<output2>\nfile_type = bin\nvariable = pcgh_con\ndt = 0.1\n','only complete intrinsic state')
bad=run('invalid-state',t.replace('<problem>','<problem>\namplitude = 3'),'invalid state/RHS or hyperbolicity domain')
assert list(bad.glob('intrinsic-first-bad-rank*.txt'))
run('restart-to-legacy',t, 'incompatible PC-GH restart', ['-r',str(a.restart.resolve()),'pc_gh/formulation=legacy'])
d=run('state-output',t+'\n<output2>\nfile_type = bin\nvariable = pcgh\ndt = 0.1\n')
f=sorted((d/'bin').glob('*.bin'))[-1];raw=f.read_bytes();header=raw[:raw.index(b'header offset=')].decode();assert 'number of variables=50' in header
names=header.split('variables:')[1].split();assert len(names)==50 and len(set(names))==50 and all(n.startswith('pcghi_') for n in names)
summary=dict(status='PASS',runs=len(records),binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),output_names=names,scope='unsupported-path and domain rejection, first-bad stencil retention, intrinsic-to-legacy restart rejection, actual state output header')
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
