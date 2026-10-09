#!/usr/bin/env python3
"""Freeze exact certificate source/proof capsule with external NPZ metadata."""
from pathlib import Path
import hashlib,json,shutil,subprocess
HERE=Path(__file__).resolve().parent;OUT=HERE/'immutable-exact-rounded-finite-certificate-20261009';R=HERE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={str(p.relative_to(HERE)):p for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
    and not any(s.startswith('immutable-') for s in p.relative_to(HERE).parts) and p.name not in ('freeze.stdout','freeze.stderr','frozen-readback.stdout','frozen-readback.stderr')}
external=[];inputs={}
for n in (8,12,16):
    receipt=json.loads((HERE/f'N{n}-certificate001/receipt.json').read_text())
    assert receipt['error'] is None and receipt['sources_unchanged'] is True
    assert receipt['certificate_computation_completed'] is True and receipt['certified_positive_eigenvalues_at_least']==2
    for path,expected in receipt['source_before'].items():
        assert sha(path)==expected
        if path in inputs:assert inputs[path]==expected
        inputs[path]=expected
already={str(p.resolve()) for p in sources.values()}
for path,expected in sorted(inputs.items()):
    p=Path(path)
    if p.suffix in ('.npz','.npy'):
        external.append({'path':str(p.resolve()),'sha256':expected,'bytes':p.stat().st_size,'role':'external_large_payload'});continue
    if str(p.resolve()) not in already:sources['input-context/'+expected[:16]+'/'+p.name]=p
prior=[R/'continuum/finite-rb-limited-matrix-growth-20261009/J0-N8-rb98-growth001/receipt.json',
       R/'continuum/finite-rb-projection-defect/immutable-J0-projection-constraint-defect-20261009/index.json',
       R/'continuum/finite-rb-successful-point-readback-20261009/immutable-saved-point-readbacks-20261009/index.json']
for p in prior:sources['prior-context/'+sha(p)[:16]+'/'+p.name]=p
before={key:sha(p) for key,p in sources.items()};OUT.mkdir(exist_ok=False);files=[]
for key,p in sorted(sources.items()):
    target=OUT/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha(target)==before[key]
    role='large_payload' if target.name=='exact-binary64-input.json' or target.suffix in ('.npz','.npy','.jsonl') or target.stat().st_size>1048576 else 'source_or_receipt'
    files.append({'path':key,'sha256':before[key],'bytes':target.stat().st_size,'role':role,'origin':str(p.resolve())})
assert before=={key:sha(p) for key,p in sources.items()}
(OUT/'external-inputs.json').write_text(json.dumps({'NPZ_NPY_metadata_only':True,'files':external},indent=2)+'\n')
p=OUT/'external-inputs.json';files.append({'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'role':'source_or_receipt','origin':'generated from exact admitted input hashes'})
index={'scope':'Exact rounded finite J0 matrices N8/12/16 each have two certified positive-real eigenvalues; no continuum/subsidiary/native theorem.',
 'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'public_runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'external_NPZ_metadata_only':True,'hex_inputs_always_large_payload':True,'files':files}
(OUT/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(OUT),'index_sha256':sha(OUT/'index.json'),'files':len(files),'bytes':sum(v['bytes'] for v in files),
 'large_payload_files':sum(v['role']=='large_payload' for v in files),'external_NPZ_inputs':len(external)},indent=2))
