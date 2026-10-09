"""Read-only compact native-preflight evidence verification; no scientific calls."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

WORK = Path('/Users/hz0693/research/hyperboloidal')
ARCHIVE = WORK / 'docs/validation/hyperboloidal-reference-wave-map-native-preflights-20261009'
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    x = json.loads(Path(p).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    def check(v):
        if isinstance(v, dict):
            for a in v.values(): check(a)
        elif isinstance(v, list):
            for a in v: check(a)
        elif isinstance(v, float): assert math.isfinite(v)
    check(x)
    return x

catalog = load(ARCHIVE / 'catalog.json')
assert sha(ARCHIVE / 'catalog.json') == 'fc9de674ff19ce3bd9e6cf51b4ccd489f5ff5820748aa09e0b7f1a8ccd372349'
expected = set(catalog['files']) | {'catalog.json', 'dependency-metadata.json', 'production-source-identity.json', 'archive-README.md', 'collection-receipt.json'}
paths = sorted(p for p in ARCHIVE.rglob('*') if p.is_file())
assert {str(p.relative_to(ARCHIVE)) for p in paths} == expected
assert len(paths) == 740 and sum(p.stat().st_size for p in paths) == 24095462
magic = [b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe', b'!<arch>\n', b'\x93NUMPY', b'PK\x03\x04']
json_files = 0
for p in paths:
    b = p.read_bytes()
    assert len(b) <= 1024*1024 and not any(b.startswith(v) for v in magic), str(p)
    assert p.suffix.lower() not in ['.npz','.npy','.jsonl','.rst','.bin','.o','.a','.so','.dylib','.pyc']
    b.decode('utf-8')
    rel = str(p.relative_to(ARCHIVE))
    if rel in catalog['files']:
        m = catalog['files'][rel]
        assert len(b) == m['bytes'] and sha(p) == m['sha256']
    if p.suffix == '.json': load(p); json_files += 1
assert json_files == 304
assert len(catalog['omitted_large_payloads']) == 278

failure = WORK/'docs/validation/hyperboloidal-reference-wave-map-native-t2-failure-20261009'
fc = load(failure/'catalog.json')
fr = load(failure/'preservation-receipt.json')
assert sha(failure/'catalog.json') == fr['catalog_sha256']
assert fr['passed_failure_preservation'] and fr['original_native_returncode'] == -6
assert fr['original_process_passed'] is False and fr['all_originals_rehashed_before_after']
assert len(fc['copied']) == 10 and len(fc['omitted_payloads']) == 221
assert {str(p.relative_to(failure)) for p in failure.rglob('*') if p.is_file()} == set(fc['copied']) | {'README.md','catalog.json','preservation-receipt.json'}
for name,item in list(fc['copied'].items()) + list(fc['omitted_payloads'].items()):
    original=Path(item['source'])
    assert original.stat().st_size == item['bytes'] and sha(original) == item['sha256']
    if name in fc['copied']:
        p=failure/name;assert sha(p)==item['sha256'] and p.stat().st_size<=1024*1024
        b=p.read_bytes();b.decode('utf-8');assert not any(b.startswith(v) for v in magic)
    else: assert not (failure/name).exists()
failed_launch=load(failure/'launch-receipt.json')
assert failed_launch['returncode']==-6 and failed_launch['sources_before_after_equal']
assert failed_launch['passed_native_process_and_provenance'] is False

summary = load(ARCHIVE / 'native-preparation/snapshot-preflight-summary001/summary.json')
assert summary['passed_all_eight_short_reference_snapshot_gates']
assert len(summary['cases']) == 8 and sum(c['saved_arrays'] for c in summary['cases']) == 64
all_case_results = []
for c in summary['cases']:
    name = c['case']
    n = c['saved_arrays']
    owner = load(ARCHIVE / ('native-preparation/snapshot-attempts/'+name+'-001/analysis/receipt.json'))
    indep = load(ARCHIVE / ('root-native-preflights/field-audits/'+name+'-001/receipt.json'))
    rows = load(ARCHIVE / ('root-native-preflights/field-audits/'+name+'-001/rows.json'))
    launch = load(ARCHIVE / ('root-native-preflights/batch001/'+name+'/launch-receipt.json'))
    assert launch['returncode'] == 0 and launch['error'] is None
    assert launch['passed_native_process_and_provenance'] and launch['sources_before_after_equal']
    assert owner['passed_saved_snapshot_finite_and_diagnostic_gates'] and owner['protected_inputs_before_after_equal']
    assert indep['passed'] and indep['pins_before_after_equal'] and indep['arrays'] == n == len(rows)
    assert c['t0_actual_initializer_max_error'] == 0
    assert c['max_history_native_kernel_scaled_RMS_error'] <= 2e-11
    assert c['min_saved_alpha'] > 0 and c['min_saved_chi'] > 0 and c['min_saved_conformal_metric_eigenvalue'] > 0
    ref = 'reference' in name
    assert c['max_saved_det_error'] <= (1e-11 if ref else 1e-10)
    assert c['max_saved_trace_error'] <= (1e-11 if ref else 1e-10)
    if ref:
        assert c['max_saved_reference_full25_deviation'] <= 1e-10
        assert max(c['max_saved_H_Mcon_Zcon_Theta_physical']) <= 1e-9
    assert owner['saved_arrays'] == n
    all_case_results.append({'case':name,'arrays':n,'owner_receipt':sha(ARCHIVE/('native-preparation/snapshot-attempts/'+name+'-001/analysis/receipt.json')),'independent_receipt':sha(ARCHIVE/('root-native-preflights/field-audits/'+name+'-001/receipt.json')),'launch_receipt':sha(ARCHIVE/('root-native-preflights/batch001/'+name+'/launch-receipt.json'))})
prod = subprocess.run(['git','diff','--quiet','27c19d20696ea6dd4704032c51dfd026218f64f2','--','src','CMakeLists.txt'],cwd=WORK)
assert prod.returncode == 0
docs = [WORK/'docs'/s for s in ['hyperboloidal-reference-wave-map-native-audit.md','hyperboloidal-wave-map-consistency-audit.md','hyperboloidal-layer.md']]
pins = {str(p.relative_to(WORK)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths+docs+sorted(p for p in failure.rglob('*') if p.is_file())}
pins['review_source'] = {'sha256':sha(__file__), 'bytes':Path(__file__).stat().st_size}
(OUT/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
(OUT/'root_verify.py').write_bytes(Path(__file__).read_bytes())
receipt = {'passed_compact_saved_evidence_review':True,'archive_files':len(paths),'archive_bytes':24095462,'finite_json':json_files,'saved_arrays':64,'cases':all_case_results,'catalog_sha256':sha(ARCHIVE/'catalog.json'),'failed_N16_capsule_verified':True,'failure_catalog_sha256':sha(failure/'catalog.json'),'production_equal_27c19':True,'source_sha256':sha(__file__),'pins_sha256':sha(OUT/'pins.json'),'new_scientific_queries':0,'new_native_steps':0,'scope':'Saved receipts, archive bytes/policy, failed process provenance and unchanged production only; no array replay or stability acceptance.'}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
