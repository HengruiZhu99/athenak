"""Saved-only compact archive verification; no scientific source/API execution."""
from pathlib import Path
import argparse
import hashlib
import json
import math

BANNED = {'.npz', '.npy', '.jsonl', '.o', '.a', '.so', '.dylib', '.bin', '.rst',
          '.raw', '.zip', '.tar', '.gz', '.bz2', '.xz', '.pdf', '.png', '.jpg',
          '.jpeg', '.gif', '.webp'}
MAGIC = (b'\x7fELF', b'MZ', b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe',
         b'\xfe\xed\xfa\xcf', b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe',
         b'\xbe\xba\xfe\xca', b'!<arch>\n', b'PK\x03\x04', b'\x93NUMPY',
         b'\x1f\x8b', b'BZh', b'\xfd7zXZ', b'%PDF-', b'\x89PNG', b'\xff\xd8\xff')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for block in iter(lambda: s.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def finite(x):
    if isinstance(x, float):
        assert math.isfinite(x)
    elif isinstance(x, dict):
        for v in x.values():
            finite(v)
    elif isinstance(x, list):
        for v in x:
            finite(v)


def load(path):
    v = json.loads(Path(path).read_text())
    finite(v)
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('--rehash-externals', action='store_true',
                        help='also rehash explicitly retained original payloads/dependencies')
    args = parser.parse_args()
    p = Path(args.archive)
    catalog = load(p/'catalog.json')
    json_count = 1
    for name, spec in catalog['files'].items():
        q = p/name
        assert q.is_file() and not q.is_symlink()
        assert q.stat().st_size == spec['bytes'] <= 1048576
        assert sha(q) == spec['sha256'], q
        b = q.read_bytes()
        assert q.suffix.lower() not in BANNED
        assert not b.startswith(MAGIC) and b'\0' not in b and b[257:262] != b'ustar'
        b.decode('utf-8')
        if q.suffix == '.json':
            load(q)
            json_count += 1
    actual_paths = {str(q.relative_to(p)) for q in p.rglob('*') if q.is_file()}
    assert actual_paths == set(catalog['files']) | {'catalog.json'}
    # Original indexes remain byte-identical within their semantic hierarchy.
    for capsule in catalog['capsules']:
        assert sha(p/capsule['label']/'index.json') == capsule['sha256']
    principal_index = load(p/'principal-core/index.json')
    attempt = p/'principal-core'/principal_index['accepted_attempt']
    r = load(attempt/'receipt.json')
    assert r['passed_AB'] and r['sources_unchanged']
    assert r['release_debug_numeric_JSON_equal'] and not r['C_queries_or_evolution_run']
    for mode in ['release', 'debug']:
        a = load(attempt/('check-principal-'+mode+'.json'))
        assert a['passed'] and a['actual_principal_cases'] == 792
        assert a['exact_projector_ranks'] == [10, 10]
        core = load(attempt/('core-'+mode+'.json'))
        assert core['core_cases'] == 204 and core['witnesses'] == 17
    assert load(p/'independent-principal-core/index.json')['passed']
    assert load(p/'bounded-coordinate/index.json')['passed_stable_equivalent_identity_and_local_scientific_gates']
    oracle_index = load(p/'nonlinear-oracle/index.json')
    assert oracle_index['scientific_cases'] == 96 and oracle_index['unique_point_amplitude_cases'] == 48
    assert not oracle_index['actual_helper_kernel_query'] and not oracle_index['operator_eigen_or_propagation']
    rhs_index = load(p/'actual-rhs/index.json')
    a = p/'actual-rhs'/rhs_index['accepted_attempt']
    r = load(a/'receipt.json')
    assert r['passed'] and r['release_debug_equal'] and r['source_before'] == r['source_after']
    assert len(r['source_before']) == 381 and len(r['commands']) == 5
    assert all(c['exit_code'] == 0 and c['stderr_sha256'] == hashlib.sha256(b'').hexdigest()
               for c in r['commands'])
    assert (a/'run-release.stdout').read_bytes() == (a/'run-debug.stdout').read_bytes()
    expected = load(p/'actual-rhs/prepared-input001/expected.json')
    thresholds = {'rhs22_scaled': 5e-9, 'connection_scaled': 5e-9, 'source_scaled': 5e-9,
                  'physical_constraints_absolute': 5e-9, 'input_normals_absolute': 5e-11,
                  'rate_normals_scaled': 5e-11, 'omega_difference_absolute': 2e-10}
    for mode in ['release', 'debug']:
        report = load(a/('analysis-'+mode+'/receipt.json'))
        assert report['passed'] and report['cases'] == 48 and report['thresholds'] == thresholds
        assert all(report['maxima'][k] <= t for k, t in thresholds.items())
        rows = [json.loads(s) for s in (a/('run-'+mode+'.stdout')).read_text().splitlines()]
        assert len(rows) == len(expected) == 48
        for row, exp in zip(rows, expected):
            finite(row)
            assert row['point'] == exp['point']
            for key, n in {'actual_rhs22': 22, 'physical_constraints8': 8,
                           'input_normals2': 2, 'rate_normals2': 2,
                           'submitted_minus_native_omega13': 13,
                           'scaled_source4': 4, 'scaled_reference_connection36': 36}.items():
                assert len(row[key]) == n
    assert load(p/'independent-results/saved-result-readback.json')['passed']
    assert load(p/'independent-binder/saved-input-readback.json')['passed']
    assert (p/'actual-rhs/freeze.stdout').read_bytes() == b''
    assert (p/'addenda/completed-owner-freeze.stdout').stat().st_size > 0
    external_count = 0
    if args.rehash_externals:
        for name, spec in catalog['external_originals_rehashed'].items():
            q = Path(name)
            assert q.stat().st_size == spec['bytes'] and sha(q) == spec['sha256'], q
            external_count += 1
    print(json.dumps({'passed': True, 'copied_files_excluding_catalog': len(catalog['files']),
                      'copied_bytes_excluding_catalog': sum(s['bytes'] for s in catalog['files'].values()),
                      'finite_JSON_including_catalog': json_count,
                      'catalog_sha256': sha(p/'catalog.json'),
                      'metadata_only_payloads': len(catalog['omitted_large_payloads']),
                      'external_hashes_rechecked': external_count,
                      'all48case_saved_thresholds_pass': True,
                      'scope': 'saved hashes, finite JSON, receipt thresholds and shape checks only; no scientific execution'}, indent=2))


if __name__ == '__main__':
    main()
