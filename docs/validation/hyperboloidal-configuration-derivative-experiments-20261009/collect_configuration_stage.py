"""One-shot collection of stable as-built configuration controls; no reruns."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
P=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
HELD=P.parent/'total-j-finite-rb-control-held-20261009'
DEST=ROOT/'docs/validation/hyperboloidal-configuration-derivative-experiments-20261009'
FILES={};OMITTED={};PENDING={}
sha=lambda data:hashlib.sha256(data).hexdigest()


def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)


def queue(source,target):
    data=source.read_bytes()
    assert target not in FILES and target not in OMITTED
    assert not Path(target).is_absolute() and '..' not in Path(target).parts
    spec={'source':str(source.relative_to(ROOT)),'sha256':sha(data),'bytes':len(data)}
    if source.suffix=='.json':finite(json.loads(data))
    magic=data[:4] in {b'\x7fELF',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xce',b'\xca\xfe\xba\xbe'}
    if magic or data[:8]==b'!<arch>\n' or source.suffix in {'.npz','.npy','.bin','.o','.a','.rst','.raw'} or len(data)>1048576:
        OMITTED[target]=spec;return
    FILES[target]=spec;PENDING[target]=data


def tree(folder,target):
    assert folder.is_dir()
    for source in sorted(folder.rglob('*')):
        if source.is_file():queue(source,target+'/'+str(source.relative_to(folder)))


assert not DEST.exists()
assert (HERE/'independent-draft-review.json').is_file()
assert (HERE/'archive-README-final.md').is_file()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert head=='2e0aa3b0d5ade2fef4d86807a4569fef61f6b202'
deps={};dep_sources={};old_binaries=[];pointers={}
for mode,attempt,gate in [('release','release-004','source-gate-release-002'),('debug','debug-002','source-gate-debug-001')]:
    bpath=P/'build-attempts'/attempt/'receipt.json'
    build=json.loads(bpath.read_text());run=json.loads((P/gate/'receipt.json').read_text())
    assert build['exit_code']==run['exit_code']==0
    assert build['sources_before']==build['sources_after']
    assert build['executable_sha256']==run['executable_sha256']
    assert build['sources_before']['radial_bridge.cpp']==run['source_sha256']
    for name,digest in build['sources_before'].items():
        source=P/'build-attempts'/attempt/name
        if not source.exists():source=P/name
        assert sha(source.read_bytes())==digest
    for name,digest in (build['compiler_dependency_hashes']|build['link_archive_hashes']).items():
        source=Path(name)
        if source.parent==P and (P/'build-attempts'/attempt/source.name).is_file():
            source=P/'build-attempts'/attempt/source.name
        assert sha(source.read_bytes())==digest,name
        if name in deps:assert deps[name]==digest
        deps[name]=digest
        dep_sources[name]=source
    assert (P/gate/'stderr').stat().st_size==0
    assert sha((P/gate/'stdout.json').read_bytes())==run['stdout_sha256']
    pointer={'attempt':str(P/'build-attempts'/attempt),'receipt_sha256':sha(bpath.read_bytes()),'executable_sha256':build['executable_sha256']}
    data=(json.dumps(pointer,indent=2)+'\n').encode()
    assert sha(data)==run['build_latest_sha256']
    recovered=HERE/('reconstructed-build-'+mode+'-latest.json')
    assert recovered.read_bytes()==data
    pointers[mode]={'reconstructed_sha256':sha(data),'matches_original_gate_pointer':True}
    old_binaries.append({'mode':mode,'historical_receipt_sha256':build['executable_sha256'],'local_old_binary_retained':False,'fresh_binary_readback_performed':False})

assert (P/'source-gate-release-002/stdout.json').read_bytes()==(P/'source-gate-debug-001/stdout.json').read_bytes()
for name,digest in deps.items():
    original=Path(name);source=dep_sources[name]
    if original.is_relative_to(ROOT):
        relative=original.relative_to(ROOT)
        # Project/custom and generated configuration inputs; tracked Kokkos
        # headers/libraries and platform headers remain receipt metadata.
        if relative.parts[0]!='kokkos' and source.suffix not in {'.a','.o'}:
            queue(source,'as-built-inputs/'+str(relative))
for attempt in ['release-001','release-002','release-003','release-004','debug-001','debug-002']:
    tree(P/'build-attempts'/attempt,'build-attempts/'+attempt)
for gate in ['source-gate-release-001','source-gate-release-002','source-gate-debug-001']:
    tree(P/gate,gate)
tree(P/'history/001-source-only-auto-type-scrutiny','history/001-source-only-auto-type-scrutiny')
tree(P/'independent-chunk-reviews/002-configuration-normalization','reviews/independent-chunk002')
queue(P/'independent-chunk-reviews/history/001-source-pin-race.json','reviews/history/001-source-pin-race.json')
queue(P/'root-chunk002-review.json','reviews/root-chunk002-review.json')
for name in ['RECIPE.md','FINAL-ADDENDUM.md','source-pins.json','preparation-receipt.json','final-addendum-receipt.json','independent-advisory-review.json']:
    queue(HELD/name,'held-design/'+name)
for name in ['audit-draft.md','archive-README.md','independent-draft-review.json','independent_review.py','independent-review.stdout','independent-review.stderr','root-provenance-review.json','review_build_inputs.py','reconstructed-build-release-latest.json','reconstructed-build-debug-latest.json']:
    queue(HERE/name,'reviews/'+name)
queue(HERE/'archive-README-final.md','README.md')
queue(Path(__file__),'collect_configuration_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py','verify_archive.py')
catalog={'scope':'Actual linearized C0 configuration-source radial derivative and reference/frame U/V map checkpoint only; no radial operator/rates/spectrum/propagation/pulse/BH admission.','collection_head':head,'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':FILES,'omitted_large_payloads':OMITTED,'accepted_executable_preservation_limitation':old_binaries,'reconstructed_pointer_readback':pointers,'dependency_and_link_unique_readbacks':len(deps)}
finite(catalog);data=(json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode()
DEST.mkdir(parents=True)
for target,value in PENDING.items():
    out=DEST/target;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(value);assert out.read_bytes()==value
(DEST/'catalog.json').write_bytes(data)
print(json.dumps({'files':len(FILES),'bytes':sum(v['bytes'] for v in FILES.values()),'catalog_sha256':sha(data),'dependencies_read_back':len(deps)}))
