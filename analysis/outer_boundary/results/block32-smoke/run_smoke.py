"""Short eight32^3-block zero/oblique CPU MPI check; no stability clearance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import time

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--exe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--launcher',default='mpiexec')
parser.add_argument('--analyze-only',action='store_true')
args=parser.parse_args()
sys.path.insert(0,str(args.repo.resolve()/'tst/regression'))
from z4c_boundary_oblique import input_text,set_parameter
from z4c_background_restart import cohort,validate_run


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def input_case(case):
 text=input_text(args.repo.resolve())
 for a in (1,2,3):
  text=set_parameter(text,'mesh',f'nx{a}',64)
  text=set_parameter(text,'meshblock',f'nx{a}',32)
 for section,key,value in [('z4c','debug_balance','false'),
                           ('z4c','debug_snapshot_operations','none'),
                           ('output3','file_type','rst'),
                           ('output3','dt',100),
                           ('output3','single_file_per_rank','true')]:
  text=set_parameter(text,section,key,value)
 # Output3 is now a restart; remove the old binary-variable selector.
 text=re.sub(r'(?m)^variable = z4c_residual\n','',text)
 if case=='zero':
  text=set_parameter(text,'problem','characteristic_test_family','none')
  text=set_parameter(text,'problem','characteristic_test_amplitude',0)
 return text


def inspect(run,ranks,zero):
 validate_run(run)
 assert not list(run.glob('z4c_snapshot*'))
 assert not list(run.glob('z4c_balance*'))
 assert not any(s in (run/'run.log').read_text() for s in
                ('Z4C_INVALID_STATE','An error occurred during the primitive solve:',
                 'MPI_Abort','NANS_IN_CONS'))
 first,records=cohort(run,ranks,3)
 assert records[0]['total']==8 and sum(len(r['state'])for r in records)==8
 raw=first.read_bytes();end=raw.index(b'<par_end>\n')+len(b'<par_end>\n')
 ng,nx,ny,nz=struct.unpack_from('<19i',raw,end+8+72+76)[:4]
 assert (ng,nx,ny,nz)==(4,32,32,32)
 locations_start=end+8+72+2*76+20
 locs=[struct.unpack_from('<4i',raw,locations_start+16*i)for i in range(8)]
 assert len(set(locs))==8 and all(loc[3]==records[0]['level']for loc in locs)
 payload_start=locations_start+20*8+16+8
 mins=dict.fromkeys(('alpha','chi','gxx','minor2','detg'),np.inf)
 blocks=[];files=[];count=0;nonzero=0;max_residual=0;max_skew=0
 global_gid=0
 for rank,record in enumerate(records):
  path=run/'rst'/f'rank_{rank:08d}'/first.name
  buf=path.read_bytes()
  assert np.isfinite(np.frombuffer(buf[payload_start:],dtype='<f8')).all()
  files.append({'rank':rank,'name':str(path.relative_to(args.output)),
                'bytes':len(buf),'sha256':hashlib.sha256(buf).hexdigest()})
  for state in record['state']:
   u=np.frombuffer(state,dtype='<f8').reshape(25,40,40,40)
   assert np.isfinite(u).all()
   count+=u.size;nonzero+=int(np.count_nonzero(u));max_residual=max(max_residual,float(abs(u).max()))
   # Exact fixture background is Minkowski; this is not a trumpet decoder.
   xx,xy,xz,yy,yz,zz=1+u[1],u[2],u[3],1+u[4],u[5],1+u[6]
   values={'alpha':1+u[18],'chi':1+u[0],'gxx':xx,
           'minor2':xx*yy-xy*xy,
           'detg':xx*yy*zz+2*xy*xz*yz-xx*yz*yz-yy*xz*xz-zz*xy*xy}
   for key,val in values.items():
    assert np.isfinite(val).all() and np.all(val>0),(key,global_gid)
    mins[key]=min(mins[key],float(val.min()))
   loc=locs[global_gid]
   for tangent in (1,2):
    point=[20,20,20];point[0]=4 if loc[0]==0 else 35
    point[tangent]=35 if loc[tangent]==0 else 4
    i,j,k=point
    g=np.array([[xx[k,j,i],xy[k,j,i],xz[k,j,i]],
                [xy[k,j,i],yy[k,j,i],yz[k,j,i]],
                [xz[k,j,i],yz[k,j,i],zz[k,j,i]]])
    gi=np.linalg.inv(g);normal=gi[:,0]/np.sqrt(gi[0,0])
    max_skew=max(max_skew,float(abs(normal[tangent])))
   blocks.append({'gid':global_gid,'logical_location':loc,
      'residual_sha256':hashlib.sha256(state.tobytes()).hexdigest()})
   global_gid+=1
 assert count==8*25*40**3
 if zero:assert nonzero==0 and max_skew==0
 else:assert nonzero>0 and max_residual>1e-3 and max_skew>1e-5
 return {'cycle':3,'time_M':records[0]['time'],'dt_M':records[0]['dt'],
         'ranks':ranks,'blocks':8,'block_cells':[32,32,32],
         'residual_values_including_ghosts':count,'nonzero_residual_values':nonzero,
         'max_abs_residual':max_residual,'minimum_full_metric_including_ghosts':mins,
         'max_abs_raised_tangential_normal_at_internal_edge':max_skew,
         'headers_match_across_ranks':True,'all_checkpoint_payload_finite':True,
         'blocks_by_gid':blocks,'files':files,'passed':True}


args.output.mkdir(parents=True,exist_ok=True)
binary_hash=sha(args.exe)
result={'scope':__doc__,'binary_sha256':binary_hash,
        'mesh':'64^3 active cells on[-1,1]^3; eight32^3 blocks; no AMR',
        'scheme':'Exact oblique Minkowski fixture; G1; RK3; ng4; original zero_rate; linearghosts',
        'diagnostics':'Signed stage dumps disabled; invalid state/CPBC checks retained; per-rank restarts',
        'cases':{}}
started=time.monotonic()
for kind in ('zero','pulse'):
 reference=None
 for ranks in (1,4):
  run=args.output/f'{kind}_r{ranks}';config=input_case(kind)
  if not args.analyze_only:
   run.mkdir(exist_ok=False);(run/'input.athinput').write_text(config)
   t=time.monotonic()
   with (run/'run.log').open('w')as log:
    subprocess.run(shlex.split(args.launcher)+['-n',str(ranks),str(args.exe.resolve()),'-i','input.athinput'],
                   cwd=run,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
   (run/'elapsed.json').write_text(json.dumps({'launch_through_exit_seconds':time.monotonic()-t})+'\n')
  assert (run/'input.athinput').read_text()==config
  checked=inspect(run,ranks,kind=='zero')
  checked['input_sha256']=sha(run/'input.athinput')
  checked.update(json.loads((run/'elapsed.json').read_text()))
  if reference is None:reference=checked
  else:
   assert checked['blocks_by_gid']==reference['blocks_by_gid'],'MPI repartition changed complete residual payload'
   assert checked['time_M']==reference['time_M'] and checked['dt_M']==reference['dt_M']
  checked['same_layout_MPI_residual_bitwise_equal']=True
  result['cases'][run.name]=checked
  result['output_bytes_so_far']=sum(p.stat().st_size for p in args.output.rglob('*')if p.is_file())
  assert result['output_bytes_so_far']<4_000_000_000
  (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
  print(run.name,'PASS','t',checked['time_M'],'nonzero',checked['nonzero_residual_values'],
        'wall',checked['launch_through_exit_seconds'],'bytes',result['output_bytes_so_far'],flush=True)
assert sha(args.exe)==binary_hash,'Executable changed during testing'
result['pass']=True;result['total_test_seconds']=time.monotonic()-started
result['script_sha256']=sha(Path(__file__))
(args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
