#!/usr/bin/env python3
"""Snapshot completed saved-data point readbacks and exact input context."""
from pathlib import Path
import hashlib,json,shutil,subprocess
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'finite-rb-saved-point-readback-20261009'
OUT=HERE/'immutable-saved-point-readbacks-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={}
for prefix,label in [(OLD,'failed-and-degree'),(HERE,'successful')]:
    for p in sorted(prefix.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or any(part.startswith('immutable-') for part in p.relative_to(prefix).parts):continue
        if p==OUT or p.name in ('freeze.stdout','freeze.stderr'):continue
        sources[label+'/'+str(p.relative_to(prefix))]=p
receipts=[OLD/n/'receipt.json' for n in ('failed-growth-readback001','N12-projected-readback001','N16-projected-readback001')]
receipts += [HERE/f'N{n}-readback001/receipt.json' for n in (8,12,16)]
inputs={}
for p in receipts:
    d=json.loads(p.read_text());assert d['error'] is None and d['sources_unchanged'] is True
    assert d['source_before']==d['source_after']
    for path,expected in d['source_before'].items():
        assert sha(path)==expected
        if path in inputs:assert inputs[path]==expected
        inputs[path]=expected
already={str(p.resolve()) for p in sources.values()}
for path,expected in inputs.items():
    if str(Path(path).resolve()) in already:continue
    sources['input-context/'+expected[:16]+'/'+Path(path).name]=Path(path)
before={key:sha(path) for key,path in sources.items()}
OUT.mkdir(exist_ok=False);files=[]
for key,path in sorted(sources.items()):
    target=OUT/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    assert sha(target)==before[key]
    role='large_payload' if target.suffix in ('.npz','.npy','.jsonl') or target.stat().st_size>1048576 else 'source_or_receipt'
    files.append({'path':key,'sha256':before[key],'bytes':target.stat().st_size,'role':role,'origin':str(path.resolve())})
assert before=={key:sha(path) for key,path in sources.items()}
index={'scope':'Saved physical8 point diagnostics of original failed growth payload, N12/N16 projected fields, and successful N8/N12/N16 seven-time finite-matrix payloads; no new query, spectrum or evolution.',
 'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'public_runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'original_SciPy_expm_attempt_remains_failed':True,'both_ordinary_FD_attempts_remain_failed':True,
 'general_nongauge_continuum_comparator_unresolved':True,'NPZ_NPY_JSONL_always_large_payload':True,'files':files}
(OUT/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(OUT),'index_sha256':sha(OUT/'index.json'),'files':len(files),'bytes':sum(v['bytes'] for v in files),'large_payload_files':sum(v['role']=='large_payload' for v in files)},indent=2))
