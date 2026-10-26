#!/usr/bin/env python3
"""CPU/GPU admission comparison. Tolerances predeclared before GPU execution."""
import argparse,csv,hashlib,json,struct
from pathlib import Path
import numpy as np

def checkpoint(root,name,receipt):
 p=root/name/'probe_checkpoint.bin';payload=receipt['checkpoint_payload']
 assert p.stat().st_size==payload['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==payload['sha256'],(name,'payload hash')
 with p.open('rb') as f:
  assert f.read(16) in [b'ATHENA_CSP_CK01!',b'ATHENA_CSP_CK02!']
  tag=f.read(64).decode();sig=struct.unpack('<17d',f.read(136));fingerprint,time,cycle=struct.unpack('<Qdq',f.read(24));packs=[]
  for _ in range(2):
   shape=struct.unpack('<5Q',f.read(40));packs.append(np.memmap(p,mode='r',dtype='<f8',offset=f.tell(),shape=shape));f.seek(int(np.prod(shape))*8,1)
  assert f.tell()+8==p.stat().st_size
 assert tag==receipt['extra_parameters']['cubed_sphere/conformal_checkpoint_tag']
 assert cycle==4 and sig[-1]==8
 return sig,time,cycle,packs

def norms(root,name,time,cycle):
 result={}
 for filename in ['cubed_sphere_conformal.csv','cubed_sphere_conformal_regions.csv']:
  rows=list(csv.DictReader((root/name/filename).open()))
  for row in rows:
   if float(row['time'])!=time or int(row['cycle'])!=cycle:continue
   key=filename+'/'+row['region']+'/'+row.get('subset','all')
   assert key not in result
   result[key]={n:float(row[n]) for n in ['H_l2','M_conformal_l2','Z_conformal_l2']}
 assert any(k.endswith('shell/fixed_outer_collar') for k in result)
 return result

def compare(cpu,gpu):
 a=json.loads((cpu/'summary.json').read_text());b=json.loads((gpu/'summary.json').read_text())
 assert a['grids']==b['grids'] and a['phase']==b['phase']
 for p,digest in a['source_sha256'].items():
  if p.startswith(('src/','inputs/')) or p=='CMakeLists.txt':assert b['source_sha256'][p]==digest,(p,'source mismatch')
 assert a['cases'].keys()==b['cases'].keys();out={};passed=True
 for name,ar in a['cases'].items():
  br=b['cases'][name]
  for r in [ar,br]:
   assert r['exit_code']==0 and r['reached_step_target'] and r['all_rows_finite'] and r['reference_unchanged']
   refs=r['reference_payloads'];assert refs['cubed_sphere_reference_before.bin']['sha256']==refs['cubed_sphere_reference_after.bin']['sha256']
  sa,ta,ca,pa=checkpoint(cpu,name,ar);sb,tb,cb,pb=checkpoint(gpu,name,br)
  assert sa==sb and ta==tb and ca==cb,(name,'signature/time mismatch')
  fields=[]
  for region,(x,y) in enumerate(zip(pa,pb)):
   assert x.shape==y.shape
   for n in range(25):
    u=np.asarray(x[:,n,4:-4,4:-4,4:-4]);v=np.asarray(y[:,n,4:-4,4:-4,4:-4]);assert np.isfinite(u).all() and np.isfinite(v).all()
    error=float(np.max(abs(u-v)));scale=float(max(np.max(abs(u)),np.max(abs(v))));limit=1e-10+1e-8*scale;passed &= error<=limit
    fields.append(dict(region=['bulk','shell'][region],component=n,error=error,tolerance=limit,passed=error<=limit))
  na=norms(cpu,name,ta,ca);nb=norms(gpu,name,tb,cb);assert na.keys()==nb.keys();nchecks=[]
  for region in na:
   for norm in na[region]:
    x=na[region][norm];y=nb[region][norm];assert np.isfinite([x,y]).all()
    error=abs(x-y);limit=1e-10+1e-6*max(abs(x),abs(y));passed &= error<=limit
    nchecks.append(dict(region=region,norm=norm,cpu=x,gpu=y,error=error,tolerance=limit,passed=error<=limit))
  out[name]=dict(time=ta,cycle=ca,fields=fields,norms=nchecks)
 return dict(passed=bool(passed),cpu_binary=a['binary_sha256'],gpu_binary=b['binary_sha256'],cases=out,scope='Four-step cross-device admission, no spatial/temporal convergence or long-time stability claim')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('gpu',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 result=compare(a.cpu,a.gpu);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('passed',result['passed']);raise SystemExit(0 if result['passed'] else 1)
