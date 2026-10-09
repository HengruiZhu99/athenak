"""Archive completed spatial-norm and coupled-wave experiments by exact hash."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/validation/hyperboloidal-spatial-norm-wave-experiments-20261009'
SMALL = {'.py', '.cpp', '.hpp', '.md', '.json', '.jsonl', '.log', '.stdout',
         '.stderr', '.txt', '.png', '.athinput', '.cmake', '.hst', '.patch'}
CATALOG = {
    'date': '2026-10-09',
    'capture_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                            text=True).strip(),
    'compiled_production_commit': '27c19d20696ea6dd4704032c51dfd026218f64f2',
    'scope': 'Archival experiments only; no production gauge/boundary change.',
    'status': 'Goal active. Finite-pulse and wormhole-to-trumpet acceptance unpassed.',
    'files': {}, 'local_only': {}, 'experiments': {}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(source, relative, expected=None):
    value = digest(source)
    if expected is not None:
        assert value == expected, source
    item = {'source_path': str(source.relative_to(ROOT)), 'sha256': value,
            'bytes': source.stat().st_size}
    if source.suffix not in SMALL or source.stat().st_size > 2_000_000:
        CATALOG['local_only'][relative] = item
        return
    target = OUT/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    item['archive_path'] = str(target.relative_to(ROOT))
    CATALOG['files'][relative] = item


wave = ROOT/'build-layer-research/boundary/coupled-wave-isolate'
manifest_path = wave/'artifact-manifest.json'
assert digest(manifest_path) == 'd7dc989002d4a952c6849735a91845ec0ccfe8e3be1b40bd54e8ad6c27d59ea6'
manifest = json.loads(manifest_path.read_text())
for name, item in manifest['retained_files'].items():
    retain(wave/name, 'coupled-wave/'+name, item['sha256'])
retain(manifest_path, 'coupled-wave/artifact-manifest.json')
CATALOG['experiments']['coupled_wave'] = {
    'summary': 'coupled-wave/summary.json',
    'scope': 'Conformal two-field wave, actual 3D mixed native stencils; no Z4c theorem.',
    'result': 'Tested spectra and t6 exact dipole controls decay; no production remedy.'}


bh = ROOT/'build-layer-research/detached-wormhole/spatialnorm-gate'
index_path = bh/'frozen-index.json'
assert digest(index_path) == '6d1ab0ddc47852bb9b0b422c2a9b4a5e7180c2569ca4aeae7ba12a3dceddb08b'
index = json.loads(index_path.read_text())
for name, item in index['files'].items():
    source = ROOT/name
    retain(source, 'black-hole-initial/'+str(source.relative_to(bh)), item['sha256'])
retain(index_path, 'black-hole-initial/frozen-index.json')
CATALOG['experiments']['black_hole_initial'] = {
    'summary': 'black-hole-initial/report.md',
    'scope': 'Detached BH initial jets with Minkowski hyperboloidal reference.',
    'result': 'Five tests pass; original second jets fail and corrected jets pass initially.'}


branch = ROOT/'build-layer-research/spatial-norm-zerojet'
proof = json.loads((branch/'result.json').read_text())
for name, expected in proof['source_sha256'].items():
    assert digest(ROOT/name) == expected, name
retain(branch/'proof.py', 'necessary-value-branch/proof.py')
retain(branch/'result.json', 'necessary-value-branch/result.json')
review = bh/'value_branch_review.md'
retain(review, 'necessary-value-branch/value_branch_review.md')
CATALOG['experiments']['necessary_value_branch'] = {
    'summary': 'necessary-value-branch/result.json',
    'scope': 'Positive null finite-Q, Theta0=0, finite gauge pole values, 1<=rho<=5/2.',
    'result': 'Unique Minkowski lapse/shift/spatial normal norm; evolution unproved.'}


general = ROOT/'build-layer-research/detached-wormhole/general-spatialnorm-gate'
general_index = general/'frozen-index.json'
assert digest(general_index) == '824fb1cea8d5a03c9d2c18fce29aa25717fd768e4a2e4f891cf12f67c385f1ad'
for name, item in json.loads(general_index.read_text())['files'].items():
    source = ROOT/name
    retain(source, 'black-hole-general/'+str(source.relative_to(general)), item['sha256'])
retain(general_index, 'black-hole-general/frozen-index.json')
CATALOG['experiments']['black_hole_general'] = {
    'summary': 'black-hole-general/report.md',
    'scope': 'General S,a,M,rho and dimensionful regular gauge rates; eight initial data sets.',
    'result': 'Reusable second shift jet; initial compatibility only, not BH evolution.'}


family = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
native = family/'immutable-spatial-norm-20261009'
native_index = native/'immutable-manifest.json'
assert digest(native_index) == 'ee856041d61a47175ffe1ed2af909730f5323f7e1304129fce0801b3cd430be2'
native_manifest = json.loads(native_index.read_text())
for name, item in native_manifest['archived_files'].items():
    retain(native/name, 'spatial-norm-native/'+name, item['sha256'])
retain(native_index, 'spatial-norm-native/immutable-manifest.json')
for name, item in native_manifest['large_artifact_hashes'].items():
    source = Path(name)
    assert digest(source) == item['sha256'], source
    relative = 'spatial-norm-native/local-only/'+str(source.relative_to(ROOT))
    CATALOG['local_only'][relative] = dict(item, source_path=str(source.relative_to(ROOT)))
for name in ['poles.json', 'report.json']:
    retain(family/name, 'spatial-norm-native/'+name,
           native_manifest['large_artifact_hashes'][str(family/name)]['sha256'])
supplement = family/'immutable-restart-supplement-20261009.json'
retain(supplement, 'spatial-norm-native/immutable-restart-supplement-20261009.json',
       'fdfee950b73184af8839c2b27b2671b2c9e376df50447958314d8f51377fb700')
CATALOG['experiments']['spatial_norm_native'] = {
    'summary': 'spatial-norm-native/native-summary.json',
    'result': 'N24 t2 completes but constraints grow; rejected stabilization candidate.',
    'runtime_integrated': False}


control = ROOT/'build-layer-research/spatial-norm-native-controls'
audit = json.loads((control/'N36-audit.json').read_text())
for name, expected in audit['audit_source_sha256'].items():
    assert digest(ROOT/name) == expected, name
for name, item in audit['all_retained_files'].items():
    source = ROOT/name
    retain(source, 'native-N36/'+str(source.relative_to(control)), item['sha256'])
for name in ['audit.py', 'N36-audit.json', 'N36-launch.json', 'N36.athinput']:
    retain(control/name, 'native-N36/'+name)
CATALOG['experiments']['native_N36'] = {
    'summary': 'native-N36/N36-audit.json',
    'scope': 'Same gauge N36 t.2; actual dt.000115104167 versus N24 .000427734375.',
    'result': 'Early errors reduced with combined space/time refinement; no acceptance.'}


OUT.mkdir(parents=True, exist_ok=True)
retain(Path(__file__).resolve(), 'collection-source.py')
(OUT/'catalog.json').write_text(json.dumps(CATALOG, indent=2, allow_nan=False) + '\n')
print('Archived', len(CATALOG['files']), 'files,',
      sum(v['bytes'] for v in CATALOG['files'].values()), 'bytes; local-only',
      len(CATALOG['local_only']))
