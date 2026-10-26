#!/usr/bin/env python3
"""Continue a scalar checkpoint with unchanged saved numerical/geometry inputs."""
import argparse,hashlib,json,re,subprocess,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tlim',type=float,required=True);p.add_argument('--wall-seconds',type=float,default=0);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing restart attempt is immutable')
parameters=json.loads((a.parent/'parameters.json').read_text());geometry=json.loads((a.parent/'frozen-integrity.json').read_text());assert geometry['unchanged'];receipt=a.parent/('complete.txt' if (a.parent/'complete.txt').exists() else 'checkpointed.txt');start=float(re.search(r'time=(\S+)',receipt.read_text())[1]);assert a.tlim>start
options={'n':'n','M':'mass','a_over_M':'spin','cap':'cap','excision':'excision','cfl':'cfl','ko':'ko','reduction_damping':'reduction-damping','radial_stretch':'radial-stretch','outer_sbp':'outer-sbp','boundary':'boundary','data':'data','smooth_physical_layer':'smooth-layer','characteristic_transfer':'characteristic-transfer','compatible_reduction':'compatible-reduction','normal_upwind':'normal-upwind'}
args=[]
for key,opt in options.items():
 if key in parameters:args+=['--'+opt,str(int(parameters[key])) if isinstance(parameters[key],bool) else str(parameters[key])]
checkpoint=a.parent/'checkpoint.f64';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
parent_receipt_path=a.parent/('receipt.json' if (a.parent/'receipt.json').exists() else 'restart-receipt.json')
parent_receipt=json.loads(parent_receipt_path.read_text())
if parent_receipt['returncode']!=0:raise RuntimeError('Cannot resume a failed attempt')
if parent_receipt.get('files',{}).get('checkpoint.f64')!=digest:raise RuntimeError('Parent checkpoint does not match its saved SHA256 receipt')
binary_digest=hashlib.sha256(a.exe.read_bytes()).hexdigest()
parent_binary=(json.loads((a.parent/'invocation.json').read_text())['binary'] if (a.parent/'invocation.json').exists() else parent_receipt['binary_sha256'])
if parent_binary!=binary_digest:raise RuntimeError('Continuation requires the same preserved executable')
a.output.mkdir(parents=True)
command=[str(a.exe.resolve()),*args,'--restart',str(checkpoint.resolve()),'--start-time',str(start),'--restart-physical-hash',geometry['physical_before'],'--restart-reference-hash',geometry['reference_before'],'--tlim',str(a.tlim),'--wall-seconds',str(a.wall_seconds),'--output',str(a.output.resolve())]
environment={'OMP_NUM_THREADS':os.environ.get('SCALAR_CPU_THREADS','6'),'OMP_PROC_BIND':os.environ.get('SCALAR_CPU_PROC_BIND','false')}
if 'SCALAR_CPU_PLACES' in os.environ:environment['OMP_PLACES']=os.environ['SCALAR_CPU_PLACES']
record=dict(parent=str(a.parent.resolve()),checkpoint_sha256=digest,binary_sha256=binary_digest,command=command,threads=environment['OMP_NUM_THREADS'],environment=environment);(a.output/'restart-invocation.json').write_text(json.dumps(record,indent=2)+'\n')
with (a.output/'run.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**environment})
record.update(returncode=r.returncode,complete=(a.output/'complete.txt').exists(),checkpointed=(a.output/'checkpointed.txt').exists(),files={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()});(a.output/'restart-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
if r.returncode:raise RuntimeError('Restart failed; attempt preserved')
