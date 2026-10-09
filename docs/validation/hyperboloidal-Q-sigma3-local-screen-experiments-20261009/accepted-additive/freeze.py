from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;D=P/'immutable-Q-sigma3-deterministic-reanalysis-20261009';assert not D.exists()
r=json.loads((P/'receipt.json').read_text());assert r['passed_warning_free_additive_reanalysis']
D.mkdir()
for p in P.iterdir():
 if p.is_file():shutil.copy2(p,D/p.name)
rows={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in D.iterdir()if p.is_file()}
j={'purpose':'Warning-free additive deterministic reanalysis; original 46-file C++/BLAS-warning freeze unchanged','files':rows,'file_count':len(rows),'bytes':sum(d['bytes']for d in rows.values()),'original_frozen_index_sha256':r['original_frozen_index_sha256'],'source_inputs':r['source_before']}
(D/'index.json').write_text(json.dumps(j,indent=2)+'\n')
for n,d in rows.items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==d['sha256']
print(json.dumps({'index_sha256':hashlib.sha256((D/'index.json').read_bytes()).hexdigest(),'files':len(rows),'bytes':j['bytes'],'path':str(D)},indent=2))
