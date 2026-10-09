"""Freeze local oracle checkpoint once; large decimal payloads by hash."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = HERE.parent/'immutable-nonlinear-Minkowski-wave-map-oracle-20261009'
DEST.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files, large = [], []


def capture(path, relative):
    row = dict(path=str(relative), origin=str(path), bytes=path.stat().st_size,
               sha256=sha(path))
    if path.stat().st_size > 1024*1024:
        large.append(row)
    else:
        target = DEST/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(target) == row['sha256']
        files.append(row)
    if path.suffix == '.json':
        json.dumps(json.loads(path.read_text()), allow_nan=False)


for path in sorted(HERE.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        capture(path, path.relative_to(HERE))
plan = json.loads((HERE/'plan.json').read_text())
admission = json.loads((HERE/'execution-admission.json').read_text())
for i, row in enumerate(plan['inputs']+admission['external_inputs']):
    source = ROOT/row['path']
    assert sha(source) == row['sha256']
    capture(source, Path('provenance')/('%02d-%s' % (i, source.name)))
index = dict(kind='Immutable independent nonlinear Minkowski wave-map oracle',
             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             production_source_commit='27c19d20696ea6dd4704032c51dfd026218f64f2',
             source_directory=str(HERE), files=files, large_records=large,
             file_count=len(files), copied_bytes=sum(row['bytes'] for row in files),
             finite_json_count=sum(row['path'].endswith('.json') for row in files),
             scientific_receipt_sha256=sha(HERE/'attempt002/receipt.json'),
             scientific_cases=96, unique_point_amplitude_cases=48,
             all_consumed_jets_and_time_rates='See SCHEMA.md; physical metric2/inverse embedding3/Z4c fields ordinary spacetime jets.',
             preserved_failures=['attempt001 embedding radial-vector adapter failure',
                                 'readback-failure001 Path hash-guard failure'],
             actual_helper_kernel_query=False, operator_eigen_or_propagation=False,
             scope='Finite sampled exact-flat local source oracle; no PDE/native/scri/BH stability or adoption.')
(DEST/'index.json').write_text(json.dumps(index, indent=2)+'\n')
print(json.dumps(dict(path=str(DEST/'index.json'), sha256=sha(DEST/'index.json'),
                      files=len(files), bytes=index['copied_bytes'],
                      finite_json=index['finite_json_count'], large_records=len(large)), indent=2))
