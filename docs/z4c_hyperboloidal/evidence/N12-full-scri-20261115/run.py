#!/usr/bin/env python3
"""Qualified full-overlap N12 scri run, with mandatory GPU admission on first launch."""
from pathlib import Path
import argparse, hashlib, json, math, os, re, struct, subprocess, time
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--wrapper',required=True);p.add_argument('--binding',required=True);p.add_argument('--restart',type=Path);a=p.parse_args()
case=a.case.resolve();job=os.environ['PBS_JOBID'].split('.')[0];launch=case/('launch-'+job)
start=time.monotonic();dt=.00046944203735617445;target=math.floor(20/dt)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def header(path):
 with path.open('rb') as f:
  text=bytearray()
  while not text.endswith(b'<par_end>\n'):
   b=f.read(1)
   if not b:raise ValueError('Truncated restart')
   text+=b
  fixed=f.read(struct.calcsize('=ii9d38i2di'));n=struct.unpack_from('=i',fixed)[0]
  t,step,cycle=struct.unpack_from('=2di',fixed,len(fixed)-20)
  atlas=f.read(struct.unpack('=Q',f.read(8))[0]);tree=f.read(n*20);f.read(n*4)
  meta=struct.unpack('=8Q',f.read(64));assert meta[0]==0x4859504552525354 and not meta[7]
  identity=f.read(meta[2]).decode();stride=struct.unpack('=Q',f.read(8))[0];offset=f.tell()
 assert path.stat().st_size==offset+n*stride
 return dict(cycle=cycle,time=t,time_M0=3*t,dt=step,blocks=n,identity=identity,offset=offset,
             geometry_sha256=hashlib.sha256(atlas+tree).hexdigest(),stride=stride)
def compare(left,right):
 x,y=header(left),header(right)
 for key in ['cycle','time','dt','blocks','identity','geometry_sha256','stride']:assert x[key]==y[key],key
 digest=hashlib.sha256();size=0
 with left.open('rb') as f,right.open('rb') as g:
  f.seek(x['offset']);g.seek(y['offset'])
  while True:
   b=f.read(8*1024*1024);c=g.read(8*1024*1024);assert b==c,'GPU accepted state changed'
   if not b:break
   size+=len(b);digest.update(b)
 return dict(bitwise_equal=True,payload_bytes=size,payload_sha256=digest.hexdigest())
def setval(text,block,key,value):
 pattern=rf'(<{re.escape(block)}>\n)(.*?)(?=\n<|\Z)';m=re.search(pattern,text,re.S);assert m,block
 body=m[2];entry=rf'(?m)^{re.escape(key)}\s*=.*$';line=f'{key} = {value}'
 body=re.sub(entry,lambda _:line,body) if re.search(entry,body) else line+'\n'+body
 return text[:m.start(2)]+body+text[m.end(2):]
def last(work):return max((work/'rst').glob('*.rst'),key=lambda f:header(f)['cycle'])
receipt=json.loads((case/'build-final/build-receipt.json').read_text());exe=case/'build-final/athena';assert sha(exe)==receipt['executable_sha256']
old=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111')
reference=old/'full-overlap-perf20-20261114/athena-cache-only';assert sha(reference)=='de633cda33dd17fc312214a0d7e2253512bf4ea0f2ba19911b71fe2fb94a36e3'
base=(case/'input.athinput').read_text();assert 'overlap_mode = agreement_full' in base
assert float(re.search(r'(?m)^dt\s*=\s*(.+)$',base)[1])==dt
manifest=Path(re.search(r'(?m)^binary_directory\s*=\s*(.+)$',base)[1].strip())/'manifest.json'
config=json.loads((case/'configuration.json').read_text());assert sha(manifest)==config['initial_manifest_sha256']
env=[line.split() for line in (launch/'rank-environment.txt').read_text().splitlines()]
cpus=[set(range(int(part.split('-')[0]),int(part.split('-')[1])+1)) for part in a.binding.split(':')]
assert len(env)==24 and len({r[0] for r in env})==2
for host in {r[0] for r in env}:
 rows=[r for r in env if r[0]==host];assert len(rows)==12 and len({r[2] for r in rows})==12 and {int(r[1]) for r in rows}==set(range(12))
assert all(r[3:8]==['1','1','specialized','limited','128'] and set(map(int,r[8].split(',')))==cpus[int(r[1])] for r in env)
scri='\n<output20>\nfile_type = scri\nid = scri\ndt = .01\nnodal = true\nmodes = true\nlmax = 8\nunit_description = geometric code units; M0=1/3; native time t/M0=3t\n'
results={}
def run(name,binary,limit,restart=None,record=True,wall=600):
 work=case/name;work.mkdir(exist_ok=False)
 text=setval(setval(base,'time','nlim',limit),'time','tlim',20)
 if limit<target:text=setval(text,'output2','dt',100)
 if record:text+=scri.replace('dt = .01','dcycle = 1') if limit<target else scri
 inp=work/'input.athinput';inp.write_text(text)
 args=(['-r',str(restart)] if restart else [])+['-i',str(inp),'-d',str(work),'-t',f'{wall//3600:02}:{wall//60%60:02}:{wall%60:02}']
 cmd=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+a.binding,a.wrapper,str(binary),*args]
 provenance=dict(command=cmd,executable_sha256=sha(binary),input_sha256=sha(inp),runner_sha256=sha(Path(__file__)),initial_manifest_sha256=sha(manifest),job=job)
 if restart:provenance['restart']={'path':str(restart),'sha256':sha(restart),**header(restart)}
 (work/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');begin=time.monotonic()
 with (work/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT,timeout=wall+180)
 (work/'exit-code.txt').write_text(str(r.returncode)+'\n');r.check_returncode()
 rst=last(work);h=header(rst);assert h['dt']==dt and h['blocks']==730 and 'agreement=full-unmasked-' in h['identity'] and h['cycle']<=limit
 assert abs(h['time']-h['cycle']*dt)<1e-8
 if record:
  assert list(work.glob('*.r00000000.scri')),'No scri records'
 result=dict(last=h,last_checkpoint=str(rst),last_checkpoint_sha256=sha(rst),complete=h['cycle']==limit,wall_seconds=time.monotonic()-begin)
 (work/'segment-result.json').write_text(json.dumps(result,indent=2)+'\n');print(name,json.dumps(result),flush=True)
 return work
if not (case/'gpu-admission.json').exists():
 assert not a.restart, 'Restart requires prior GPU admission'
 for label,restart,begin in [('fresh',None,0),('late',old/'science/N12-full-overlap-30M-20261114/segment-9089673/rst/headon.00026.rst',18462)]:
  if restart:assert header(restart)['cycle']==begin
  ref=run(f'admission-{job}-{label}-reference',reference,begin+8,restart,False)
  cand=run(f'admission-{job}-{label}-candidate',exe,begin+8,restart,True)
  assert header(last(ref))['cycle']==begin+8
  results[label]=compare(last(ref),last(cand))
 (case/'gpu-admission.json').write_text(json.dumps(dict(executable_sha256=sha(exe),comparisons=results,job=job),indent=2)+'\n')
else:
 admission=json.loads((case/'gpu-admission.json').read_text());assert admission['executable_sha256']==sha(exe)
 assert all(admission['comparisons'][key]['bitwise_equal'] for key in ['fresh','late'])
 if a.restart:
  assert a.restart.resolve().is_relative_to(case) and header(a.restart)['cycle']<target
  previous=json.loads((a.restart.parent.parent/'segment-result.json').read_text());assert str(a.restart)==previous['last_checkpoint'] and sha(a.restart)==previous['last_checkpoint_sha256']

wall=int(3180-(time.monotonic()-start));assert wall>=300,'Admission used allocation; submit science separately'
run('segment-'+job,exe,target,a.restart,True,wall)
