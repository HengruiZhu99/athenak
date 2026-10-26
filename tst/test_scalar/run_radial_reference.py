#!/usr/bin/env python3
"""Run one new source-bound independent radial oracle; preserve every output."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--n',type=int,required=True);a=p.parse_args();E=a.evidence.resolve();O=E/'radial-oracle/smooth';table=O/f'normal-coefficients-n{a.n}.txt';wave=O/f'waveform-upwind-n{a.n}.csv';exe=E/'radial-oracle/radial_upwind-checkpoint-v2';source=E/'radial-oracle/radial_upwind-checkpoint-v2.cpp';scripts=Path(__file__).resolve().parent
if wave.exists() or Path(str(wave)+'.invocation.json').exists():raise RuntimeError('Existing independent trajectory is immutable; never duplicate')
while not table.with_suffix('.json').exists():time.sleep(5)
meta=json.loads(table.with_suffix('.json').read_text());preflight=json.loads((O/'checkpoint-smoke-n1200-v2-review.json').read_text())
if meta['n']!=a.n or meta['inner']!=.04 or not meta['smooth_physical_layer'] or meta['table_sha256']!=sha(table) or not preflight['passed'] or preflight['binary_sha256']!=sha(exe) or preflight['source_sha256']!=sha(source) or sha(source)!=sha(scripts/'radial_upwind.cpp'):raise RuntimeError('Preserved independent scalar source/binary/table preflight failed')
command=[str(exe),str(table),str(wave),'100'];identity=dict(command=command,binary_sha256=sha(exe),source_sha256=sha(source),table_sha256=sha(table));inv=Path(str(wave)+'.invocation.json');inv.write_text(json.dumps(identity,indent=2)+'\n');start=time.time()
with Path(str(wave)+'.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
checkpoint=Path(str(wave)+'-checkpoint.f64');cm=json.loads(Path(str(wave)+'-checkpoint.json').read_text()) if checkpoint.exists() else {};values=np.fromfile(checkpoint) if checkpoint.exists() else np.array([]);complete=r.returncode==0 and cm.get('time')==100 and cm.get('n')==a.n and len(values)==3*a.n and np.isfinite(values).all() and sha(table)==identity['table_sha256'];record=dict(**identity,returncode=r.returncode,elapsed_seconds=time.time()-start,complete=bool(complete),files={p.name:sha(p) for p in O.glob(wave.name+'*') if p.is_file()});receipt=Path(str(wave)+'.receipt.json');receipt.write_text(json.dumps(record,indent=2)+'\n')
if not complete:raise RuntimeError('Independent radial reference failed; all attempted evidence preserved')
subprocess.run([sys.executable,str(scripts/'fit_ringdown.py'),str(wave),'--spin','0','--expected',str(E/'spin-zero-qnm.json'),'--output',str(O/f'qnm-n{a.n}-review.json')],check=True)
print(json.dumps(dict(complete=complete,n=a.n,elapsed_seconds=record['elapsed_seconds']),indent=2),flush=True)
