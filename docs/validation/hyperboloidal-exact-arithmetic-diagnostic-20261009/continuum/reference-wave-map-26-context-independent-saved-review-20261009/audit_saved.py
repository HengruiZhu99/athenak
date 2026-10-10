"""Standard-library compact saved-report/provenance audit; no scientific inputs decoded."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = ROOT / 'build-layer-research/boundary/reference-wave-map-far-dual-metric-flux-diagnostic-held-20261009'
RUN = OWNER / 'attempts/diagnostic001'
RELEASE = ROOT / 'build-layer-research/far-dual-metric-flux-diagnostic-root-release-20261009'
PRIOR = ROOT / 'build-layer-research/continuum/reference-wave-map-26-context-independent-source-review-20261009'
EXPECTED = {
    OWNER / 'source-index.json': '3da906bcddd42dafcee590a775ee7c142cf78767784c08fae7deb924134ff6a5',
    RUN / 'receipt.json': '9ad15627de14b911c5765d5b2efff8ad63482e09d7926383e76bfccceb1cd64e',
    RUN / 'oracle-report.json': '62401d80bb454aa9e16b7d14d0ec4c7b36700605324cc3a9eb023c622f57af48',
    PRIOR / 'index.json': '1d18ba7384418823c3d0cabecc9f74dc06a3597e197b93403542b35ae94e7e27',
    PRIOR / 'receipt.json': '4298cd1a7b7b4115311a38204658b23dffbf0c376364623462b6cffd37c457a9',
    RELEASE / 'authorization.json': '777fd1619a2ea9efd1173325684d99bbaf7ceea796c946d68dcbb93a52f8f675',
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    if path.suffix in ('.jsonl', '.npz', '.npy') or path.stat().st_size > 1048576:
        raise RuntimeError('forbidden payload decoding: ' + str(path))
    return json.loads(path.read_text())

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')

def require(condition, label):
    if not condition:
        raise RuntimeError(label)

def meta(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def records(obj):
    if isinstance(obj, list):
        return obj
    return [{'path': p, 'sha256': h} for p, h in obj.items()]

def ratio(obj):
    require(set(obj) == {'numerator', 'denominator'}, 'ratio schema')
    n, d = int(obj['numerator']), int(obj['denominator'])
    require(n >= 0 and d > 0, 'ratio nonnegative / positive denominator')
    return n, d

def main():
    require(sys.flags.optimize == 0, 'optimization disabled')
    require(not (HERE / 'receipt.json').exists(), 'single use')
    start = time.monotonic()
    protected = dict(EXPECTED)
    for p, h in EXPECTED.items():
        require(sha(p) == h, 'fixed pin ' + str(p))
    source_index = load(OWNER / 'source-index.json')
    source_files = source_index['files']
    require(source_index['file_count'] == len(source_files) == 10, 'source count')
    for r in source_files:
        protected[Path(r['path'])] = r['sha256']
    for name in ('input-pins.json',):
        for r in records(load(OWNER / name)):
            protected[Path(r['path'])] = r['sha256']
    for p in (RUN / 'pins-before.json', RUN / 'pins-before-query.json', RUN / 'oracle-pins-before.json', RELEASE / 'diagnostic-invocation001/pins-before.json', RUN / 'dependencies.json'):
        protected[p] = sha(p)
        for r in records(load(p)):
            q = Path(r['path'])
            h = r['sha256']
            require(q not in protected or protected[q] == h, 'consistent overlapping pin ' + str(q))
            protected[q] = h
    for r in load(PRIOR / 'index.json')['files']:
        protected[Path(r['path'])] = r['sha256']
    compact_paths = list(OWNER.glob('*')) + list(RUN.iterdir()) + list(RELEASE.glob('*')) + list((RELEASE / 'diagnostic-invocation001').iterdir())
    compact_paths = sorted(set(p for p in compact_paths if p.is_file()))
    for p in compact_paths:
        protected[p] = protected.get(p, sha(p))
    before = {}
    for p, h in sorted(protected.items()):
        got = sha(p)
        require(got == h, 'before pin ' + str(p))
        before[str(p)] = got
    write(HERE / 'protected-before.json', before)
    child = load(RUN / 'receipt.json')
    outer = load(RELEASE / 'diagnostic-invocation001/receipt.json')
    inv = load(RELEASE / 'diagnostic-invocation001/invocation.json')
    auth = load(RELEASE / 'authorization.json')
    report = load(RUN / 'oracle-report.json')
    for key in ('completed', 'diagnostic_completed', 'inputs_unchanged'):
        require(child[key] is True, 'child flag ' + key)
    require(child['returncode'] == 0 and child['original_far_Release_passed'] is False, 'child classification')
    for key in ('completed', 'accepted_observational_diagnostic', 'inputs_unchanged'):
        require(outer[key] is True, 'outer flag ' + key)
    require(outer['returncode'] == 0 and outer['original_far_Release_passed'] is False, 'outer classification')
    require(outer['child_receipt_sha256'] == EXPECTED[RUN / 'receipt.json'], 'outer child binding')
    require(auth['metric_flux_diagnostic_local_admitted'] is True, 'root admission')
    require(auth['source_index_sha256'] == EXPECTED[OWNER / 'source-index.json'], 'admitted source index')
    require(inv['command'][1:3] == ['-I', '-B'], 'isolated bytecode-off runner')
    require(inv['command'][-1] == EXPECTED[RELEASE / 'authorization.json'], 'consumed authorization')
    require(inv['environment'] == {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'PYTHONOPTIMIZE': '0'}, 'fixed invocation environment')
    require(len(child['commands']) == 4, 'four child commands')
    require([c['name'] for c in child['commands']] == ['compiler-version', 'compile', 'diagnostic', 'oracle'], 'child command ordering')
    for c in child['commands']:
        require(c['returncode'] == 0, 'child command exit ' + c['name'])
        require(c['stderr']['bytes'] == 0, 'empty stderr ' + c['name'])
    require(child['commands'][-1]['command'][1:3] == ['-I', '-B'], 'isolated oracle')
    for key in ('diagnostic_completed', 'exact_inverse_derivative_identity_passed', 'inputs_unchanged', 'original_bits_equal'):
        require(report[key] is True, 'report flag ' + key)
    require(report['original_far_Release_passed'] is False, 'report original failure retained')
    require((report['rows'], report['helper_calls'], report['original_failed_labels_checked'], report['original_saved_zero_targets']) == (26, 52, 63, 8), 'reported fixed coverage')
    require(len(report['stage_maxima']) == 20, 'ten stages x two components')
    for key, m in report['stage_maxima'].items():
        for field in ('absolute', 'scaled', 'expression_term_scaled'):
            ratio(m[field])
        require(isinstance(m['target_exact_zero'], bool), 'recorded zero target schema')
    inverse = report['stage_maxima']['inverse-stage/dual1']
    require(ratio(inverse['absolute']) == (1060101518999515, 65077583243356051315733322268672), 'inverse absolute maximum')
    effect = report['stage_maxima']['inverse-effect-exact-downstream/dual1']
    rest = report['stage_maxima']['remaining-arithmetic-after-native-inverse/dual1']
    require(ratio(effect['scaled']) == (1, 1) and effect['scaled_base'] == 11 and effect['scaled_label'] == '3' and effect['target_exact_zero'], 'inverse downstream selected maximum')
    require(ratio(rest['scaled']) == (1, 1) and rest['scaled_base'] == 3 and rest['scaled_label'] == '1' and not rest['target_exact_zero'], 'remaining arithmetic selected maximum')
    inv_records = outer['output_inventory'] + child['outputs'] + [child['executable']]
    for r in inv_records:
        p = Path(r['path'])
        require(p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], 'recorded output ' + str(p))
    copied, omitted = [], []
    dest = HERE / 'copies'
    dest.mkdir()
    for i, p in enumerate(compact_paths):
        entry = meta(p)
        with p.open('rb') as f:
            magic = f.read(4)
        binary = magic in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe', b'\x7fELF')
        if p.suffix in ('.npz', '.npy', '.jsonl') or p.stat().st_size > 1048576 or binary:
            entry['policy'] = 'large_payload_metadata_only'
            omitted.append(entry)
        else:
            q = dest / ('%03d-%s' % (i, p.name))
            shutil.copyfile(p, q)
            require(sha(q) == entry['sha256'], 'copy equality')
            entry['copy'] = str(q)
            copied.append(entry)
    write(HERE / 'inventory.json', {'copies': copied, 'omitted': omitted})
    write(HERE / 'compact-summary.json', {'reported_rows': 26, 'reported_helper_calls': 52, 'reported_original_failed_labels': 63, 'reported_exact_zero_labels': 8, 'stage_maxima': report['stage_maxima'], 'maxima_recomputed': False, 'jsonl_decoded': False, 'targets_recomputed': False})
    for p, h in protected.items():
        require(sha(p) == h, 'after pin ' + str(p))
    receipt = {'passed': True, 'status': 'PASS_SAVED_COMPACT_REPORT_AND_PROVENANCE', 'seconds': time.monotonic() - start, 'protected_unique_files': len(protected), 'compiler_dependencies': len(load(RUN / 'dependencies.json')), 'copied_files': len(copied), 'metadata_only_files': len(omitted), 'original_far_Release_passed': False, 'source_and_outputs_unchanged': True, 'jsonl_decoded': False, 'target_or_spectrum_recomputation': False, 'candidate_imports_or_calls': False, 'maxima_scope': 'Compact saved summary only; no independent entry-level maxima recomputation.', 'original_source_review_index_sha256': EXPECTED[PRIOR / 'index.json']}
    write(HERE / 'receipt.json', receipt)
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name != 'index.json')
    write(HERE / 'index.json', {'status': 'immutable saved-only independent review', 'file_count': len(files), 'files': [meta(p) for p in files]})
    print(json.dumps(receipt, sort_keys=True))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        (HERE / 'failure.txt').write_text(traceback.format_exc())
        raise
