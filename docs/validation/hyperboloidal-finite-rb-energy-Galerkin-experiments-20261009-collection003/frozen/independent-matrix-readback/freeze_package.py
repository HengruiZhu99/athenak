"""Freeze independent small evidence; external matrix payloads by hash only."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
DEST = HERE.parent/'immutable-finite-rb-independent-matrix-readback-20261009'
assert not DEST.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': sha(path)}


source_targets = {
    'assemble_segmented.py': P/'J0-N8-rb.98-segmentedQ64-a12x24-refinement001/assemble_segmented.py',
    'assemble_blas.py': P/'J0-N8-rb.98-segmentedQ64-a12x24-BLAS-equivalence001/assemble_blas.py',
    'energy_coefficients.py': P/'J0-N8-rb.98-segmentedQ64-a12x24-refinement001/energy_coefficients.py',
    'canceled_basis_complex.py': P/'J0-N8-rb.98-segmentedQ64-a12x24-refinement001/canceled_basis_complex.py',
    'check_blas_equivalence.py': P/'check_blas_equivalence.py',
    'BLAS-equivalence-report.json': P/'BLAS-equivalence-report.json',
    'BLAS-equivalence-plan.json': P/'BLAS-equivalence-plan.json',
    'J1-segmented-quadrature-comparison.json': P/'J1-segmented-quadrature-comparison.json',
    'J1-angular-comparison.json': P/'J1-angular-comparison.json',
    'J2-segmented-quadrature-comparison.json': P/'J2-segmented-quadrature-comparison.json',
    'J2-angular-comparison.json': P/'J2-angular-comparison.json'}
copied = HERE/'reviewed-owner-sources'
assert not copied.exists()
copied.mkdir()
copy_manifest = []
for name, source in source_targets.items():
    target = copied/name
    shutil.copyfile(source, target)
    assert sha(target) == sha(source)
    copy_manifest.append({'copy': name, 'source': record(source)})
(copied/'copy-manifest.json').write_text(json.dumps(copy_manifest, indent=2)+'\n')

external = {}


def visit(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'}.issubset(value) and str(value['path']).endswith('.npz'):
            path = Path(value['path'])
            assert path.exists() and path.stat().st_size == value['bytes']
            assert sha(path) == value['sha256']
            external[str(path.resolve())] = record(path)
        for child in value.values():
            visit(child)
    elif isinstance(value, list):
        for child in value:
            visit(child)


for path in HERE.rglob('*.json'):
    visit(json.loads(path.read_text()))
external_path = HERE/'external-matrix-metadata.json'
assert not external_path.exists()
external_path.write_text(json.dumps({'payload_policy': 'Metadata only; no external scientific arrays copied.',
                                     'matrices': list(external.values())}, indent=2)+'\n')
DEST.mkdir()
files, large = [], []
for source in sorted(HERE.rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
        continue
    rel = source.relative_to(HERE)
    entry = {'path': str(rel), 'origin': str(source.resolve()),
             'bytes': source.stat().st_size, 'sha256': sha(source)}
    if source.stat().st_size > 1048576:
        large.append(entry)
        continue
    target = DEST/rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(target) == entry['sha256']
    files.append(entry)
index = {'kind': 'Immutable independent finite-rb saved-matrix readback',
         'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'production_source_commit': '27c19d20696ea6dd4704032c51dfd026218f64f2',
         'source_directory': str(HERE.resolve()), 'files': files,
         'file_count': len(files), 'total_bytes': sum(f['bytes'] for f in files),
         'large_metadata_only': large, 'external_matrix_count': len(external),
         'scientific_kernel_or_assembler_run': False,
         'generator_eigensolve_or_propagation': False,
         'status': 'PASS_saved_matrix_readback_with_global_rule_failures_preserved',
         'manufactured_scope': 'One saved mixed vector per J only; per-channel polynomial replay remains separate and pending.'}
index_path = DEST/'index.json'
index_path.write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
assert all(sha(DEST/f['path']) == f['sha256'] for f in files)
print(json.dumps({'index': record(index_path), 'files': len(files),
                  'bytes': index['total_bytes'], 'large_metadata_count': len(large),
                  'external_matrix_count': len(external)}, indent=2))
