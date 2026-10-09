"""Unique-path degree-control receipt with pinned degree-independent cached inputs."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
P=Path(__file__).resolve().parent;old=P.parent/'total-j-finite-rb-control-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--N',type=int,required=True);p.add_argument('--J',type=int,default=0);p.add_argument('--Q',type=int,default=64);p.add_argument('--theta',type=int,default=12);p.add_argument('--phi',type=int,default=24);p.add_argument('--name',required=True);a=p.parse_args()
assert a.N in (12,16) and a.J in (0,1,2) and a.Q in (32,64)
assert (a.theta,a.phi)==(12,24),'Primary cached degree controls first'
d=P/a.name;assert not d.exists();d.mkdir()
base=old/f'J{a.J}-N8-rb.98-segmentedQ{a.Q}-a12x24-refinement001'
r=json.loads((base/'report.json').read_text());assert sha(base/'operator.npz')==r['operator_sha256']
for name in ('source','input'):(d/name).symlink_to((base/name).resolve(),target_is_directory=True)
for name in ('reference.txt','reference-receipt.json','reference.stderr'):shutil.copyfile(base/name,d/name)
cmd=[sys.executable,str(P/'assemble_degree.py'),'--J',str(a.J),'--N',str(a.N),'--rb','.98','--Q',str(a.Q),'--theta',str(a.theta),'--phi',str(a.phi),'--name',a.name]
rec={'command':cmd,'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch_HEAD':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
     'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','PYTHONPATH')},'wrapper_sha256':sha(__file__),'source_sha256':sha(P/'assemble_degree.py'),
     'executable_sha256':sha(P/'radial-bridge-release'),'cached_N8_input_parent':str(base),'cached_matrix_sha256':r['operator_sha256'],
     'cached_receipts':{name:sha(base/name) for name in ('source/receipt.json','input/receipt.json','reference-receipt.json')},
     'scope':'same point queries independent of degree; new polynomial trial/operator; no spectra or propagation'}
(d/'run_degree-launch-source.py').write_bytes(Path(__file__).read_bytes());(d/'launch-receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
start=time.monotonic()
with (d/'run.stdout').open('wb') as fo,(d/'run.stderr').open('wb') as fe:run=subprocess.run(cmd,stdout=fo,stderr=fe)
rec.update({'exit_code':run.returncode,'seconds':time.monotonic()-start,'stdout_sha256':sha(d/'run.stdout'),'stderr_sha256':sha(d/'run.stderr'),
            'source_unchanged':sha(P/'assemble_degree.py')==rec['source_sha256'],'executable_unchanged':sha(P/'radial-bridge-release')==rec['executable_sha256']})
(d/'run-receipt.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');print(json.dumps(rec,indent=2));sys.exit(run.returncode)
