"""Pack the completed 300M pilot evidence without mutable continuation data."""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent
selected = {}
for pattern in ('*.md', '*.py', '*results.json', '*regression.json',
                '*source-results.json', '*runs.json', 'manifest.json',
                'peer-*.json', '*.patch', 'pilot-comparison.png',
                'point_source_driver.cpp'):
    for path in root.glob(pattern):
        if path.name.startswith('continue_'):
            continue
        selected[path.name] = path
for name in ('covariant_alpha01', 'covariant_const01', 'covariant_const03',
             'covariant_const10'):
    folder = root/name
    for pattern in ('*.hst', 'input.athinput', 'run.log', 'manifest.json',
                    'exit_code.txt', 'checkpoint-validity.json'):
        for path in folder.glob(pattern):
            selected[str(path.relative_to(root))] = path
for name in ('baseline', 'scaled_01', 'scaled_03'):
    folder = root.parent/'lapse-damping'/name
    for pattern in ('*.hst', 'input.athinput', 'manifest.json', 'exit_code.txt'):
        for path in folder.glob(pattern):
            selected[f'reference_z4c/{name}/{path.name}'] = path
for name in ('README.md', 'manifest.json', 'hook-validation.json',
             'candidate-validated-modes.json', 'sigma1-directional-responses.json',
             'covariant-mode-comparison.png'):
    path = root/'modes'/name
    if path.exists():
        selected[f'modes/{name}'] = path
manifest = {
    'scope': 'Completed fresh 300M CPU pilots only; continuation excluded.',
    'limitations': [
        'Immutable binaries and full snapshots/checkpoints remain in the review directory.',
        'Checkpoint validation JSON records exact checkpoint hashes and validity.',
        'Reference raw histories are stored under reference_z4c; original source paths remain in JSON.',
        'This evidence archive does not establish perturbation, matter, MPI or AMR stability.'
    ],
    'files': {name: {'bytes': path.stat().st_size,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in sorted(selected.items())}
}
mpath = root/'pilot-evidence-manifest.json'
mpath.write_text(json.dumps(manifest, indent=2)+'\n')
selected[mpath.name] = mpath
target = root/'pilot-evidence-300M.tar.gz'
with tarfile.open(target, 'w:gz') as archive:
    for name, path in sorted(selected.items()):
        archive.add(path, arcname=f'covariant-sources/{name}', recursive=False)
with tarfile.open(target) as archive:
    for item in archive.getmembers():
        name = str(Path(item.name).relative_to('covariant-sources'))
        if name == mpath.name:
            continue
        data = archive.extractfile(item).read()
        assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256']
print(json.dumps({'archive': str(target), 'bytes': target.stat().st_size,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'files': len(selected), 'all_archived_hashes_verified': True}))
