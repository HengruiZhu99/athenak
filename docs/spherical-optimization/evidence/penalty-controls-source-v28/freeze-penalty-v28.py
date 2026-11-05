import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
root=Path.cwd();e=root/'docs/spherical-optimization/evidence';dest=e/'penalty-controls-source-v28';dest.mkdir(exist_ok=False)
paths=sorted([*root.glob('src/z4c/hyperboloidal/*.hpp'),*root.glob('tst/spherical_optimization/*.py'),*root.glob('tst/spherical_optimization/*.cpp'),root/'tst/spherical_optimization/CMakeLists.txt',root/'tst/elliptic_shift_1d/numerics.py',*root.glob('docs/spherical-optimization/*.md'),e/'cost-specification-v28.json',e/'serial-cost-hardware-v28.json'])
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with tarfile.open(dest/'source.tar.gz','w:gz') as t:
 for p in paths:t.add(p,arcname=str(p.relative_to(root)))
for name in ['penalty-refinement-v28.py','penalty05-controls-v28.py','analyze-penalty-v28.py','make-cost-spec-v28.py','freeze-penalty-v28.py']:
 p=Path('/tmp')/name
 if p.exists():shutil.copy2(p,dest/name)
(dest/'requirements-frozen.txt').write_text(subprocess.check_output([str(root/'build-spherical/python/bin/python'),'-m','pip','freeze'],text=True))
rows=[]
for p in sorted(e.glob('*v28/**/result.json')):
 folder=p.parent;cfg=json.loads((folder/'manifest.json').read_text())['config'];r=json.loads(p.read_text())
 rows.append(dict(run=str(folder.relative_to(root)),points=cfg['n'],target_M=cfg['duration'],accepted_M=r['time_M'],perturbation=cfg['perturbation'],outer_lapse_damping=cfg['outer_lapse_damping'],penalty=cfg['penalty'],fixed_dt=cfg['fixed_dt'],status=r['status'],failure=r['failure'],floor_calls=r['floor_calls'],result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(dest/'run-index.json').write_text(json.dumps(dict(runs=rows,total=len(rows),completed=sum(x['status']=='completed_unqualified' for x in rows),failed=sum(x['status']=='failed' for x in rows)),indent=2)+'\n')
meta=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=hashes,command_builders={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.py')},binary_sha256=hashlib.sha256((root/'build-spherical-v26/libcurrent_spherical.dylib').read_bytes()).hexdigest(),qualification='Final phase source snapshot; commit is the preceding checkpoint and content hashes identify uncommitted phase changes. No compiled-kernel change. Execution sources were frozen throughout the serial cost batch. Raw evolutions preserve their own contemporaneous source archives; final analysis source is included here.')
(dest/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(runs=len(rows),completed=sum(x['status']=='completed_unqualified' for x in rows),failed=sum(x['status']=='failed' for x in rows))))
