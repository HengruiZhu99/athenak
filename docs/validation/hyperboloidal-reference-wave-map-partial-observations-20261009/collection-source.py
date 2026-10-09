"""One-shot preservation of finalized partial diagnostics and pencil notes."""
from pathlib import Path
import hashlib
import json

ROOT = Path('/Users/hz0693/research/hyperboloidal')
HERE = Path(__file__).resolve().parent
DEST = ROOT / 'docs/validation/hyperboloidal-reference-wave-map-partial-observations-20261009'
TREES = {
    'partial-N16': 'reference-wave-map-partial-diagnostic-held-20261009',
    'partial-N24': 'reference-wave-map-partial-N24-held-20261009',
    'independent-fields-N16': 'wave-map-native-partial-fields-root-20261009',
    'independent-fields-N24': 'wave-map-native-N24-partial-fields-root-20261009',
    'partial-plots-N16': 'wave-map-native-partial-plot-root-20261009',
    'conformal-gauge-assessment': 'continuum/conformal-reference-wave-map-feasibility-20261009',
    'independent-stationary-assessment': 'continuum/conformal-reference-stationary-BH-pencil-20261009',
    'independent-joint-source-assessment': 'continuum/preferred-Box-mass-log-source-pencil-20261009',
}
EXEC_MAGIC = {b'\x7fELF', b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe',
              b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe',
              b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'}
OMIT_SUFFIX = {'.npy', '.npz', '.jsonl', '.rst', '.bin', '.o', '.a', '.so', '.dylib', '.pyc'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path):
    path = Path(path)
    return {'source': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def main():
    assert not DEST.exists(), 'Never freeze again into an existing archive'
    originals = {}
    external = {}
    for label, suffix in TREES.items():
        prefix = ROOT / 'build-layer-research' / suffix
        for path in sorted(prefix.rglob('*')):
            if path.is_file():
                originals[label + '/' + str(path.relative_to(prefix))] = metadata(path)
        for path in prefix.rglob('*-before.json'):
            value = load(path)
            if isinstance(value, dict) and value and all(isinstance(k, str) and k.startswith('/')
                                                        and isinstance(v, str) and len(v) == 64
                                                        for k, v in value.items()):
                for name, digest in value.items():
                    assert name not in external or external[name] == digest, name
                    external[name] = digest
        inventory = prefix / 'source-inventory.json'
        if inventory.exists():
            for name, digest in load(inventory)['source_pins'].items():
                assert name not in external or external[name] == digest, name
                external[name] = digest
        if label.startswith('independent-') and label.endswith('-assessment'):
            for item in load(prefix / 'receipt.json')['sources'].values():
                name, digest = str(ROOT / item['path']), item['sha256']
                assert name not in external or external[name] == digest, name
                external[name] = digest
    # Link exact root releases without copying any live native run tree.
    for label, suffix in {
        'root-release-N16.json': 'wave-map-native-partial-root-release-20261009/authorization.json',
        'root-release-N24.json': 'wave-map-native-N24-partial-root-release-20261009/authorization.json',
    }.items():
        originals[label] = metadata(ROOT / 'build-layer-research' / suffix)
    originals['collection-source.py'] = metadata(__file__)
    for name, digest in external.items():
        assert sha(name) == digest, name
    # Partial protocol success is deliberately distinct from native acceptance.
    for suffix, cases in [
        (TREES['partial-N16'], ['wave-map-N16-large-t2', 'c0-N16-large-t2']),
        (TREES['partial-N24'], ['wave-map-N24-large-t2']),
    ]:
        for case in cases:
            p = ROOT / 'build-layer-research' / suffix / 'attempts' / (case + '-001')
            value = load(p / 'receipt.json')
            assert value['observer_completed'] and value['protected_before_after_equal']
            assert value['accepted_native_run'] is False and value['native_returncode'] == -6
    DEST.mkdir(parents=True)
    copied, omitted = {}, {}
    for rel, item in originals.items():
        path = Path(item['source'])
        with path.open('rb') as stream:
            magic = stream.read(4)
        reason = None
        if item['bytes'] > 1024 * 1024:
            reason = 'larger_than_1MiB'
        elif path.suffix.lower() in OMIT_SUFFIX:
            reason = 'array_or_compiled_payload'
        elif magic in EXEC_MAGIC or magic[:2] == b'MZ':
            reason = 'executable_magic'
        elif 'matplotlib-config' in path.parts:
            reason = 'derived_plot_font_cache'
        if reason:
            omitted[rel] = dict(item, omission_reason=reason)
            continue
        content = path.read_bytes()
        if path.suffix.lower() not in {'.png', '.pdf'}:
            content.decode('utf-8')
        q = DEST / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(content)
        assert q.stat().st_size == item['bytes'] and sha(q) == item['sha256']
        copied[rel] = item
    for item in originals.values():
        assert metadata(item['source']) == item, item['source']
    for name, digest in external.items():
        assert sha(name) == digest, name
    catalog = {'scope': 'Three failed native runs: saved partial observations, independent '
                        'field readbacks, scalar plots and separately labeled pencil-only gauge assessments. '
                        'No native acceptance or causal stability claim.',
               'copied': copied, 'omitted_payloads': omitted,
               'new_native_steps': 0, 'new_scientific_queries': 0}
    dump(DEST / 'catalog.json', catalog)
    assert (DEST / 'catalog.json').stat().st_size <= 1024 * 1024
    dump(DEST / 'external-protected-identities.json', external)
    assert (DEST / 'external-protected-identities.json').stat().st_size <= 1024 * 1024
    receipt = {'passed_observation_preservation': True, 'copied_files': len(copied),
               'omitted_files': len(omitted), 'original_files': len(originals),
               'external_protected_identities_rehashed_before_after': len(external),
               'all_originals_rehashed_before_after': True,
               'accepted_native_run': False, 'new_native_steps': 0, 'new_scientific_queries': 0,
               'catalog_sha256': sha(DEST / 'catalog.json'),
               'external_identities_sha256': sha(DEST / 'external-protected-identities.json'),
               'source_sha256': sha(__file__)}
    dump(DEST / 'preservation-receipt.json', receipt)
    (DEST / 'README.md').write_text('''# Failed-run partial observations and independent pencil assessments

Wave-map N16, matched C0 N16 and wave-map N24 are failed original t2 runs.
The 55, 80 and 32 saved arrays are earlier snapshots, not reconstructed abort
stages. Diagnostic protocol completion never changes native acceptance.
Original RST/BIN arrays and compiled artifacts remain at their pinned local
origins; this archive copies scalar observation JSON, source, inputs, exact
logs/receipts and N16 scientific plots. Protected external identities are
rehash-verified before and after collection. No live native tree is copied.

Independent root field extrema agree exactly with all 167 owner observations.
The H/M/Z/Theta columns compare original native history with the native probe;
they are not an independent differentiated constraint calculation. C0 value-only
wave-gauge pole columns evaluate a different gauge on C0 fields and do not
represent its evolved gauge poles. Saved algebraic normals do not describe
unrecorded intermediate RK stages or establish a cause of failure.

The conformal-reference and preferred-source notes are pencil-only, conditional
stationary spherical Einstein analyses. They do not implement or admit another
gauge. Their Schwarzschild mass-log obstruction and leading joint-source terms
are distinct from finite-time Minkowski pulse failures and do not explain them.
Higher angular/off-constraint/scri closure and the later wormhole-to-trumpet
evolution with a Minkowski hyperboloidal reference remain unresolved.

Never regenerate this archive in place. New observations or corrections require
a fresh destination. All omitted arrays/compiled/large/cache payloads are
identified by original path, byte count and SHA256; none is silently truncated.
''')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
