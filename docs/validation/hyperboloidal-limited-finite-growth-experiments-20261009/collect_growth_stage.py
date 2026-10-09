"""One-shot byte-preserving compact collection of the negative finite control."""
from pathlib import Path
import hashlib, json, math, subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
DEST=ROOT/'docs/validation/hyperboloidal-limited-finite-growth-experiments-20261009'
FILES={}; OMITTED={}; PENDING={}; EXTERNAL={}
sha=lambda b:hashlib.sha256(b).hexdigest()
def finite(x):
    if isinstance(x,float): assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values(): finite(v)
    elif isinstance(x,list):
        for v in x: finite(v)
def inspect_arrays(source):
    result={}
    if source.suffix=='.npz':
        with np.load(source,allow_pickle=False) as data:
            for k in data.files:
                v=data[k]
                if np.issubdtype(v.dtype,np.number): assert np.isfinite(v).all(), (source,k)
                result[k]={'shape':list(v.shape),'dtype':str(v.dtype)}
    return result
def queue(source,target,omit=False,expected=None):
    assert target not in FILES and target not in OMITTED
    assert not Path(target).is_absolute() and '..' not in Path(target).parts
    data=source.read_bytes(); spec={'source':str(source.relative_to(ROOT)),'sha256':sha(data),'bytes':len(data)}
    if expected: assert spec['sha256']==expected['sha256'] and spec['bytes']==expected['bytes'], source
    magic=data[:4] in {b'\x7fELF',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xce',b'\xca\xfe\xba\xbe'}
    if omit or magic or data[:8]==b'!<arch>\n' or source.suffix in {'.npz','.npy','.bin','.o','.a','.rst','.raw','.jsonl'} or len(data)>1048576:
        spec['arrays']=inspect_arrays(source); OMITTED[target]=spec; return
    if source.suffix=='.json':
        if target=='original-held-growth-source/synthetic-preflight-001/history/001-bundled-import-probe/stdout.json':
            assert not data; spec['empty_diagnostic_log']=True
        else: finite(json.loads(data))
    FILES[target]=spec; PENDING[target]=data
def external(item):
    source=Path(item.get('origin',item['path']))
    if not source.is_absolute(): source=ROOT/source
    data=source.read_bytes(); assert sha(data)==item['sha256'] and len(data)==item['bytes'],source
    EXTERNAL[str(source.relative_to(ROOT))]={'sha256':sha(data),'bytes':len(data),'arrays':inspect_arrays(source)}
def frozen(path,label,pin):
    source=ROOT/path; assert sha((source/'index.json').read_bytes())==pin
    index=json.loads((source/'index.json').read_text()); finite(index)
    for item in index['files']:
        queue(source/item['path'],label+'/'+item['path'],item.get('role')=='large_payload',item)
    queue(source/'index.json',label+'/index.json')
    for key in ('external_large_files','large_external_records','external_input_records'):
        for item in index.get(key,[]): external(item)
def tree(path,label):
    for source in sorted((ROOT/path).rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            assert not source.is_symlink(),source
            queue(source,label+'/'+str(source.relative_to(ROOT/path)))

assert not DEST.exists()
frozen('build-layer-research/boundary/total-j-finite-rb-degree-control-20261009/immutable-J0-finite-rb-degree-control-20261009','degree-J0','056942564fdbd6f36510a63cddde38c0b2a0b6b811ad02942a665883e125fdb7')
frozen('build-layer-research/boundary/total-j-finite-rb-degree-control-20261009/immutable-J1-J2-finite-rb-degree-control-20261009','degree-J1-J2','46ba57a7d8d0eb3be3f703826ab8036384512876e3a3651e5be4db4d640a5dfb')
frozen('build-layer-research/continuum/immutable-finite-rb-degree-independent-readback-20261009','degree-independent','0b2707726cca86034c34d8da653975f77ca83df75c605ca1856046bb73be7c84')
frozen('build-layer-research/continuum/finite-rb-growth-control-20261009/immutable-growth-helper-review-synthetic-20261009','original-held-growth-source','0d753253f577211bfab2c56dc6eed3bcaeada81810636a4fbb63d9d6e69839cb')
frozen('build-layer-research/continuum/immutable-expm-diagnosis-pade13-synthetic-20261009','expm-diagnosis-synthetic','9b0718dae154af5580c214e77e72f0612def9ece50013a31bfbdb89899d659c5')
frozen('build-layer-research/continuum/immutable-J0-N8-t6-expm-high-precision-20261009','actual-high-precision','1884f8808e2f189fceb0476a61ea1291458a3e02f8863ef6089755af0226f386')
frozen('build-layer-research/continuum/finite-rb-successful-point-readback-20261009/immutable-saved-point-readbacks-20261009','saved-point-readbacks','8490e226fe9fe18c83898bdb4620133e3a3743202ef5bf3c0f3ec4c07b14bf56')
tree('build-layer-research/continuum/finite-rb-J0-degree-independent-root-review-20261009','degree-root-review')
tree('build-layer-research/continuum/finite-rb-limited-matrix-growth-20261009/independent-review','limited-scope-review')
tree('build-layer-research/continuum/finite-rb-limited-matrix-growth-pade13-20261009','growth')
tree('build-layer-research/continuum/finite-rb-growth-receipts-independent-review-20261009','actual-growth-review')
queue(ROOT/'build-layer-research/continuum/finite-rb-successful-point-readback-20261009/root-frozen-readback-001.stdout','root-full-point-readback.stdout')
tree('build-layer-research/limited-growth-stage-20261009/history','collector-history')
queue(HERE/'README.md','README.md'); queue(HERE/'verify_growth_archive.py','verify_growth_archive.py'); queue(Path(__file__),'collect_growth_stage.py')
catalog={'scope':'Negative actual finite energy-Galerkin control: J0/N8,N12,N16 rb=.98 spectrum estimates/guarded propagation and saved physical constraint readbacks; degree J1/J2 consistency only. General nongauge continuum comparator unresolved; no continuum/native/nonlinear/scri/BH acceptance.',
 'collection_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':FILES,'omitted_large_payloads':OMITTED,
 'rehashed_external_inputs':EXTERNAL,'large_data_policy':'All NPZ/NPY/JSONL, executable/object/archive magic, tagged payloads and >1MiB metadata only. All original freezes remain byte-for-byte unchanged.',
 'exact_mass_review_source_limit':'Root exact rational convolution was executed as an inline tool command; its saved receipt is retained. No pre-execution saved source hash is claimed for that auxiliary review.'}
finite(catalog); data=(json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode()
DEST.mkdir(parents=True)
for name,value in PENDING.items():
    out=DEST/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(value);assert out.read_bytes()==value
(DEST/'catalog.json').write_bytes(data)
print(json.dumps({'files':len(FILES),'bytes':sum(v['bytes'] for v in FILES.values()),'omitted':len(OMITTED),'external':len(EXTERNAL),'catalog_sha256':sha(data)}))
