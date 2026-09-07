#!/usr/bin/env python3
"""Production restart metadata guards and nontrivial legacy continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from run_legacy_equivalence import input_text
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
binary=str(a.binary.resolve());records=[]
def run(name,args,expect=None):
 d=a.output/name;d.mkdir();cmd=[binary,*args];start=time.time()
 with (d/'run.log').open('w') as f:r=subprocess.run(cmd,cwd=d,stdout=f,stderr=subprocess.STDOUT,timeout=120)
 log=(d/'run.log').read_text();ok=r.returncode==0 if expect is None else r.returncode!=0 and expect in log
 records.append(dict(name=name,command=cmd,returncode=r.returncode,expected_message=expect,status='PASS' if ok else 'FAIL',wall_seconds=time.time()-start))
 (a.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n');assert ok,(name,log[-3000:]);return d

def initial(name,steps):
 text=input_text(6,2,'collision_factorized',steps).replace('tlim = 0.0001',f'tlim = {steps*1e-4}').replace('boundedness_output = false','boundedness_output = false\nresearch_dt_ceiling = 0.0001')
 text+='\n<output1>\nfile_type = rst\ndt = 0.0001\n'
 f=a.output/(name+'.athinput');f.write_text(text);return run(name,['-i',str(f.resolve())])
full=initial('full',2);split=initial('split',1)
restart=sorted((split/'rst').glob('*.rst'))[-1];raw=restart.read_bytes();end=raw.index(b'<par_end>')+len(b'<par_end>');header=raw[:end].decode();tail=raw[end:]
assert re.search(r'restart_layout\s*=\s*legacy_pcgh55',header);assert re.search(r'restart_layout_fields\s*=\s*55',header)
continued=run('continued',['-r',str(restart.resolve()),'time/tlim=0.0002','time/nlim=2'])
# This fixture has one 16x16x1 block including ghosts, PC-GH as its sole payload.
def payload(folder):
 path=sorted((folder/'rst').glob('*.rst'))[-1];b=path.read_bytes();return np.frombuffer(b[-55*16*16*8:],dtype=np.float64).copy(),path
reference,refpath=payload(full);actual,actpath=payload(continued);assert np.isfinite(actual).all();err=float(np.max(abs(actual-reference)/(1+abs(reference))));(a.output/'continuity.json').write_text(json.dumps(dict(status='PASS' if err<=2e-12 else 'FAIL',max_error=err,tolerance=2e-12),indent=2)+'\n');assert err<=2e-12

def altered(name,transform):
 f=a.output/(name+'.rst');f.write_bytes(transform(header).encode()+tail);return str(f.resolve())
def replace(key,value):return lambda h:re.sub(r'(?m)^(\s*'+key+r'\s*=\s*)[^\n]*',lambda m:m.group(1)+str(value),h)
untagged=altered('untagged',lambda h:re.sub(r'(?m)^\s*restart_layout(?:_version|_fields)?\s*=.*\n','',h))
run('untagged-rejected',['-r',untagged,'-n'],'untagged PC-GH restart')
run('wrong-declaration',['-r',untagged,'-n','pc_gh/restart_untagged_layout=intrinsic_pcgh50'],'untagged PC-GH restart')
resumed=run('untagged-declared',['-r',untagged,'pc_gh/restart_untagged_layout=legacy_pcgh55','time/tlim=0.0002','time/nlim=2'])
u,upath=payload(resumed);uerr=float(np.max(abs(u-reference)/(1+abs(reference))));assert uerr<=2e-12
for key,value in [('restart_layout','older_55'),('restart_layout_version',2),('restart_layout_fields',50)]:
 f=altered('bad-'+key,replace(key,value));run('reject-'+key,['-r',f,'-n'],'unknown or inconsistent')
 # Trying to repair a corrupt header by command-line overwrite must fail first.
 expected={'restart_layout':'legacy_pcgh55','restart_layout_version':'1','restart_layout_fields':'55'}[key]
 run('reject-relabel-'+key,['-r',f,'-n',f'pc_gh/{key}={expected}'],'unknown or inconsistent')
partial=altered('partial',lambda h:re.sub(r'(?m)^\s*restart_layout_version\s*=.*\n','',h));run('reject-partial',['-r',partial,'-n'],'incomplete PC-GH restart')
run('reject-formulation',['-r',str(restart.resolve()),'-n','pc_gh/formulation=intrinsic_clean'],'incompatible PC-GH restart')
run('reject-cli-metadata',['-r',str(restart.resolve()),'-n','pc_gh/restart_layout_fields=50'],'cannot be overridden')
f=a.output/'override.athinput';f.write_text('<pc_gh>\nrestart_layout = intrinsic_pcgh50\nrestart_layout_version = 1\nrestart_layout_fields = 50\n')
run('reject-input-metadata',['-r',str(restart.resolve()),'-i',str(f.resolve()),'-n'],'cannot be overridden')
run('reject-untagged-intrinsic',['-r',untagged,'-n','pc_gh/restart_untagged_layout=legacy_pcgh55','pc_gh/formulation=intrinsic_clean'],'incompatible PC-GH restart')
run('tagged-inspection',['-r',str(restart.resolve()),'-n'])
run('reject-intrinsic-fresh',['-i',str((a.output/'full.athinput').resolve()),'pc_gh/formulation=intrinsic_clean'],'is not enabled in mesh tasks yet')
summary=dict(status='PASS',runs=len(records),tagged_resume_error=err,declared_untagged_resume_error=uerr,compared_fields=55,compared_cells_with_ghosts=256,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),restarts={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [restart,refpath,actpath,upath]},scope='actual serial legacy restart writer/reader and pre-override layout guard; no intrinsic evolution')
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
