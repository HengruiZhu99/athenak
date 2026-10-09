"""One-shot compact byte-preserving collection; original immutable capsules stay intact."""
from pathlib import Path
import hashlib,json,math,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
DEST=ROOT/'docs/validation/hyperboloidal-reference-wave-map-local-experiments-20261009'
FILES={};OMITTED={};PENDING={};EXTERNAL={}
sha=lambda b:hashlib.sha256(b).hexdigest()
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def queue(source,target,forced=False,spec=None):
    assert target not in FILES and target not in OMITTED
    assert not Path(target).is_absolute()and '..'not in Path(target).parts
    data=source.read_bytes();e={'source':str(source.relative_to(ROOT)),'bytes':len(data),'sha256':sha(data)}
    if spec:assert e['bytes']==spec['bytes']and e['sha256']==spec['sha256'],source
    if source.suffix=='.json':finite(json.loads(data))
    binary=data[:4]in{b'\x7fELF',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xce',b'\xca\xfe\xba\xbe'}or data[:8]==b'!<arch>\n'
    omit=forced or binary or len(data)>1048576 or source.suffix in{'.npz','.npy','.o','.a','.bin','.rst','.raw','.jsonl'}
    if omit:OMITTED[target]=e
    else:FILES[target]=e;PENDING[target]=data
def external(path,h,size=None):
    p=Path(path);p=p if p.is_absolute()else ROOT/p;b=p.read_bytes();assert sha(b)==h,p
    if size is not None:assert len(b)==size,p
    EXTERNAL[str(p)]={'sha256':h,'bytes':len(b)}
def frozen(path,label,pin):
    p=ROOT/path;assert sha((p/'index.json').read_bytes())==pin
    idx=json.loads((p/'index.json').read_text());finite(idx)
    for e in idx['files']:queue(p/e['path'],label+'/'+e['path'],e.get('role')=='large_payload',e)
    for key in ['large_payloads_metadata_only','large_or_binary_records']:
        for e in idx.get(key,[]):external(e.get('original_path',e.get('origin')),e['sha256'],e['bytes'])
    for path,h in idx.get('external_source_inputs',{}).items():external(path,h)
    for mode,entries in idx.get('external_compiler_dependencies',{}).items():
        for path,h in entries.items():external(path,h)
    queue(p/'index.json',label+'/index.json')
def tree(path,label):
    p=ROOT/path
    for s in sorted(p.rglob('*')):
        if s.is_file()and '__pycache__'not in s.parts:
            assert not s.is_symlink();queue(s,label+'/'+str(s.relative_to(p)))
assert not DEST.exists()
frozen('build-layer-research/boundary/einstein-coordinate-gauge-local-20261009/immutable-Einstein-coordinate-local-attempts-20261009','reference-and-failed-coordinate','9f4ec9f106ec95f27d912a33b29e7dae7056025bf5b9d1f8d92bdf57809095df')
frozen('build-layer-research/continuum/immutable-independent-higher-reference-jets-20261009','reference-independent','7c556fcf4bc82671ebf0a8de7751a316363d541b7dd77e0e9a7ac86f7e94f8eb')
frozen('build-layer-research/continuum/reference-wave-map-gauge-20261009/immutable-local-reference-wave-map-20261009','wave-map-local','25f09ff04e1f7067147b9bb75ff916750f989beb4366477b1db8748fde74dd31')
tree('build-layer-research/boundary/einstein-coordinate-gauge-held-20261009','coordinate-derivation')
tree('build-layer-research/continuum/einstein-coordinate-source-independent-review-20261009','coordinate-derivation-review')
tree('build-layer-research/continuum/reference-wave-map-independent-review-20261009','wave-map-derivation-review')
tree('build-layer-research/reference-CPP-composition-root-review-20261009','CPP-composition-review')
tree('build-layer-research/wave-map-local-root-review-20261009','wave-map-root-review')
queue(HERE/'README.md','README.md');queue(HERE/'verify_archive.py','verify_archive.py');queue(Path(__file__),'collect_stage.py')
catalog={'scope':'Private physical Minkowski reference wave-map gauge local algebra/dual gates PASS; complete higher reference jets PASS; original broad Einstein-coordinate geometry controls FAILED. No principal/operator/continuum/native/nonlinear evolution or exact-scri/BH acceptance.',
 'collection_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'files':FILES,'omitted_large_payloads':OMITTED,'rehashed_external_inputs':EXTERNAL,
 'policy':'Byte-preserved original freezes; all arrays/binaries/archives and >1MiB payloads metadata only. No collection rerun into an existing destination.'}
finite(catalog);data=(json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode();DEST.mkdir(parents=True)
for name,b in PENDING.items():
    p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);assert p.read_bytes()==b
(DEST/'catalog.json').write_bytes(data)
print(json.dumps({'files':len(FILES),'bytes':sum(e['bytes']for e in FILES.values()),'omitted':len(OMITTED),'external':len(EXTERNAL),'catalog_sha256':sha(data)}))
