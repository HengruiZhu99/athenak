"""Copy this local checkpoint once; large/binary inputs remain hash metadata."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = HERE.parent/'immutable-independent-higher-reference-jets-20261009'
DEST.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files, large = [], []


def capture(path, rel, binary=False):
    record = dict(path=str(rel), origin=str(path), bytes=path.stat().st_size,
                  sha256=sha(path))
    if binary or path.stat().st_size > 1024*1024:
        large.append(record)
    else:
        target = DEST/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(target) == record['sha256']
        files.append(record)


for path in sorted(HERE.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        capture(path, path.relative_to(HERE))
recipe = json.loads((HERE/'recipe.json').read_text())
owner = ROOT/'build-layer-research/boundary/einstein-coordinate-gauge-local-20261009'
for record in recipe['inputs']:
    source = ROOT/record['path']
    assert sha(source) == record['sha256']
    capture(source, Path('owner-inputs')/source.relative_to(owner),
            binary=source.name == 'probe')
for record in files:
    if record['path'].endswith('.json'):
        data = json.loads((DEST/record['path']).read_text())
        json.dumps(data, allow_nan=False)
index = dict(kind='Immutable independent higher reference-jet oracle checkpoint',
             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             production_source_commit='27c19d20696ea6dd4704032c51dfd026218f64f2',
             source_directory=str(HERE), files=files, large_or_binary_records=large,
             file_count=len(files), copied_bytes=sum(row['bytes'] for row in files),
             finite_json_count=sum(row['path'].endswith('.json') for row in files),
             failed_history='radial-attempt001 remains FAILED: initial derivative-magnitude endpoint classifier rejected four midpoint cancellation zeros; source/recipe/result bytes retained.',
             scope='Fixed S1/a.5 reference radial oracle and separate independent Cartesian composition only; no C++ coordinate lift/PDE/kernel/propagation/BH acceptance.')
(DEST/'index.json').write_text(json.dumps(index, indent=2)+'\n')
print(json.dumps(dict(index=str(DEST/'index.json'), sha256=sha(DEST/'index.json'),
                      files=len(files), bytes=index['copied_bytes'],
                      finite_json=index['finite_json_count'],
                      large_or_binary=len(large)), indent=2))
