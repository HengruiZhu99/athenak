from pathlib import Path
import json,hashlib
import numpy as np
from check_smr_restart import read,OUT
HERE=Path(__file__).resolve().parent
names=['smr-policy-uninterrupted2','production-cadence','production-walltime-resume']
parsed=[read(n) for n in names];records=[]
for name,(meta,topology,ages,arrays) in zip(names,parsed):
 base=parsed[0];assert meta['config']==base[0]['config'] and topology==base[1] and ages==base[2]
 assert meta['identity']==base[0]['identity'] and meta['time']==base[0]['time']
 exact=[bool(np.array_equal(a,b)) for a,b in zip(arrays,base[3])];assert all(exact)
 records.append(dict(name=name,checkpoint=meta,all_eight_arrays_identical=exact))
wall=json.load(open(OUT/'production-walltime.report.json'));assert wall['status']=='checkpointed' and wall['checkpoint_admitted'] and wall['steps']==1
resume=json.load(open(OUT/'production-walltime-resume.report.json'));assert resume['status']=='completed' and resume['checkpoint_admitted'] and resume['steps']==2
h=[json.loads(l) for l in (OUT/'production-cadence.history.jsonl').read_text().splitlines()]
assert len(h)==2 and [q['step'] for q in h]==[0,2]
assert all(len(q['fixed'])==3 and all(np.isfinite([f['H'],f['M'],f['Mr']]).all() for f in q['fixed']) for q in h)
old=json.load(open(OUT/'smr-policy-uninterrupted2.rank0.performance.json'));new=json.load(open(OUT/'production-cadence.rank0.performance.json'))
result=dict(passed=True,records=records,healthy_walltime=wall,resumed=resume,diagnostic_rows=[0,2],fixed_cuts=[.1,.3,.6],
 old_serial_mpi2=old,new_serial_mpi2=new,
 qualification='Two stationary N12 steps; rank-changing restart and reduced diagnostic cadence leave evolution arrays bit-identical. Finder seed changes; walltime-based restart shifts the available boundary row. CPU timings are not GPU scaling.')
(HERE/'production-controls-verification.json').write_text(json.dumps(result,indent=2)+'\n')
for n in ['production-cadence','production-walltime','production-walltime-resume']:
 for suffix in ['.report.json','.history.jsonl','.rank0.performance.json']:
  (HERE/(n+suffix)).write_bytes((OUT/(n+suffix)).read_bytes())
print('passed',[(r['name'],r['checkpoint']['sha256']) for r in records])
