from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;D=P/'immutable-Q-sigma3-raw22-intrinsic20-frozen-Fourier-20261009'
assert not D.exists()
receipt=json.loads((P/'receipt.json').read_text());assert receipt['passed_local_gate']
files=[]
for p in P.rglob('*'):
 if not p.is_file() or any(x.startswith('immutable-')for x in p.relative_to(P).parts):continue
 # The pilot is preserved under history; redundant root pilot outputs need no second copy.
 if p.parent==P and (p.name.startswith('pilot-') or p.name in ['metadata-pilot.json','matrices-pilot.bin']):continue
 files.append(p)
D.mkdir()
for p in files:
 q=D/p.relative_to(P);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
rows={str(p.relative_to(D)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in D.rglob('*')if p.is_file()}
index={'purpose':'Local actual C0/Q sigma3 frozen continuum Fourier screen, no native/global admission','files':rows,'file_count':len(rows),'bytes':sum(d['bytes']for d in rows.values()),'original_source_inputs':receipt['source_before'],'exploration_pilot_launch_HEAD':receipt['exploration_pilot_launch_HEAD'],'final_runner_launch_HEAD':receipt['final_runner_launch_HEAD'],'compiled_production_implementation':receipt['compiled_production_implementation']}
(D/'index.json').write_text(json.dumps(index,indent=2)+'\n')
for name,d in rows.items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==d['sha256']
print(json.dumps({'path':str(D),'index_sha256':hashlib.sha256((D/'index.json').read_bytes()).hexdigest(),'files':len(rows),'bytes':index['bytes']},indent=2))
