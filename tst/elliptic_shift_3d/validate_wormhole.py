#!/usr/bin/env python3
"""Bounded initial-slice and short 3D wormhole diagnostic, including failures."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--isotropic-reference',type=int,choices=[0,1,2],default=0);a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=True);binary=str(a.binary.resolve())
sources=sorted(set(Path('src').glob('**/*.hpp'))|set(Path('tst/elliptic_shift_3d').glob('*.cpp'))|set(Path('tst/elliptic_shift_3d').glob('*.py')))
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
tag=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
binary_hash=hashlib.sha256(Path(binary).read_bytes()).hexdigest();cases=[]
def run(name,cmd):
 start=time.monotonic();result=dict(name=name,command=cmd,timeout_seconds=180)
 try:
  r=subprocess.run(cmd,text=True,capture_output=True,timeout=180);stdout=r.stdout;stderr=r.stderr;result['exit_code']=r.returncode
 except subprocess.TimeoutExpired as e:
  stdout=e.stdout or '';stderr=e.stderr or '';result['exit_code']=124
  if isinstance(stdout,bytes):stdout=stdout.decode()
  if isinstance(stderr,bytes):stderr=stderr.decode()
 result['seconds']=time.monotonic()-start
 (out/(name+'.stdout')).write_text(stdout);(out/(name+'.stderr')).write_text(stderr)
 try:result['result']=json.loads(stdout)
 except ValueError:pass
 cases.append(result);(out/'manifest.json').write_text(json.dumps(dict(source_tag=tag,source_hashes=hashes,
  binary_sha256=binary_hash,cases=cases),indent=2)+'\n');print(name,result['exit_code'],round(result['seconds'],2),flush=True)
 return result['exit_code']
for n in [6,8,12]:
 coordinates=out/f'coordinates-n{n}.csv';initial=out/f'initial-n{n}.dat'
 base=[binary,str(n),'0','.0001','0','0','2','1e-10',str(coordinates),'','', '4','1','4','8','0','','2']
 if run(f'coordinates-n{n}',base):continue
 if run(f'initialize-n{n}',['python3','tst/elliptic_shift_3d/wormhole_initial.py',str(coordinates),str(initial)]):continue
 for steps in [0,2]:
  name=f'wormhole-n{n}-steps{steps}';cmd=base.copy();cmd[4]=str(steps);cmd[8]=str(out/(name+'.csv'))
  cmd[10]=tag;cmd[15]='1';cmd[16]=str(initial)
  if steps and n==6:cmd[9]=str(out/'wormhole-n6.bin')
  if a.isotropic_reference:cmd.append(str(a.isotropic_reference))
  run(name,cmd)
# Independent timestep and elliptic controls at the middle spatial resolution.
for name,dt,steps,tol,angular in [('dt-half','.00005',4,'1e-10',0),('tight-tolerance','.0001',2,'1e-12',0),('angular','.0001',2,'1e-10',1)]:
 cmd=[binary,'8','0.000001' if angular else '0',dt,str(steps),str(angular),'2',tol,str(out/(name+'.csv')),'',tag,'4','1','4','8','1',str(out/'initial-n8.dat'),'2']
 if a.isotropic_reference:cmd.append(str(a.isotropic_reference))
 run(name,cmd)
manifest=json.loads((out/'manifest.json').read_text());manifest['sources_unchanged']=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in hashes.items())
manifest['binary_unchanged']=hashlib.sha256(Path(binary).read_bytes()).hexdigest()==binary_hash
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
