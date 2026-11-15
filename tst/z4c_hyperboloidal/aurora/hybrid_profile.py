#!/usr/bin/env python3
"""Three short profiled admissions, verified against unprofiled accepted payloads."""
import argparse,hashlib,json,os,struct,subprocess
from pathlib import Path
from hybrid_science_segment import header,sha

def offset(path):
 with path.open('rb') as f:
  text=bytearray()
  while not text.endswith(b'<par_end>\n'):text+=f.read(1)
  f.read(struct.calcsize('=ii9d38i2di'))
  f.seek(struct.unpack('=Q',f.read(8))[0]+730*24,1)
  meta=struct.unpack('=8Q',f.read(64));f.seek(meta[2]+8,1)
  return f.tell()

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
 p.add_argument('--wrapper',required=True);p.add_argument('--binding',required=True)
 p.add_argument('--reference-run',type=Path)
 a=p.parse_args();source=a.reference_run or a.root/'runs/paired-9076258';results=[]
 for index,mode in enumerate(['legacy','masked','agreement']):
  reference=source/('%02d-%s-8'%(index,mode));assert (reference/'exit-code.txt').read_text().strip()=='0'
  d=a.run/mode;d.mkdir();inp=d/'input.athinput';inp.write_bytes((reference/'input.athinput').read_bytes())
  prov=json.loads((reference/'provenance.json').read_text());exe=Path(prov['command'][7]);assert exe.name=='athena'
  assert sha(exe)==prov['executable_sha256']
  command=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+a.binding,a.wrapper,'/usr/bin/python3.11',str(a.root/'hybrid_profile_rank.py'),str(d),str(exe),'-i',str(inp),'-d',str(d)]
  (d/'provenance.json').write_text(json.dumps({'command':command,'reference':prov,'input_sha256':sha(inp),'rank_launcher_sha256':sha(a.root/'hybrid_profile_rank.py')},indent=2)+'\n')
  with (d/'run.log').open('w') as f:r=subprocess.run(command,cwd=d,stdout=f,stderr=subprocess.STDOUT,timeout=700)
  (d/'exit-code.txt').write_text(str(r.returncode)+'\n');r.check_returncode()
  old=max((reference/'rst').glob('*.rst'),key=lambda f:header(f)['cycle'])
  new=max((d/'rst').glob('*.rst'),key=lambda f:header(f)['cycle'])
  assert header(old)==header(new) and header(new)['cycle']==8
  digest=hashlib.sha256();size=0
  with old.open('rb') as x,new.open('rb') as y:
   x.seek(offset(old));y.seek(offset(new))
   while True:
    b=x.read(8*1024*1024);c=y.read(8*1024*1024)
    assert b==c,'profile changed accepted checkpoint payload'
    if not b:break
    digest.update(b);size+=len(b)
  report={'mode':mode,'accepted_payload_bitwise_equal':True,'payload_sha256':digest.hexdigest(),'payload_bytes':size,'profiled_ranks':[0,8,22,23]}
  results.append(report);(a.run/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
