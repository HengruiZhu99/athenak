"""Prepare untouched C0 templates only; intentionally does not bind or compile Q."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[2]
original = here.parent / 'full-tensor-propagator'
frozen = here.parent / 'full-tensor-global-final'
full22 = here / 'full22-candidate'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected_manifest = '4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2'
assert sha(frozen / 'manifest.json') == expected_manifest
manifest = json.loads((frozen / 'manifest.json').read_text())
for name, row in manifest['files'].items():
    p = frozen / name
    assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']

full22.mkdir(exist_ok=True)
copied = {}
groups = [
    (here, original / 'projected-v1', [
        'tangent_server.cpp', 'old-jv-source.cpp', 'native_injection.hpp',
        'spatial_norm_control.hpp', 'validate.py', 'spatialnorm-validation-vectors.npz']),
    (full22, original / 'full22-v2', [
        'projected_base.hpp', 'full22_server.cpp', 'old-jv-source.cpp',
        'native_injection.hpp', 'spatial_norm_control.hpp',
        'diagnostic_fields.cpp', 'diagnostic_constraint_norms.cpp',
        'krylov_propagate.py', 'expm_propagate.py', 'analyze_history.py',
        'analyze_fields.py']),
]
for destination, source, names in groups:
    for name in names:
        p, q = source / name, destination / name
        assert not q.exists(), f'refusing to overwrite {q}'
        archival = frozen / 'sources' / source.name / name
        if archival.exists():
            assert sha(p) == sha(archival)
        shutil.copy2(p, q)
        copied[str(q)] = {'source': str(p), 'sha256': sha(p),
                          'archived_copy_verified': archival.exists()}

# These command files are inert recipes. No process is spawned from them here.
for destination, source in [(here, original / 'projected-v1'),
                            (full22, original / 'full22-v2')]:
    command = json.loads((source / 'build-spatialnorm.json').read_text())
    prefix = str(original / 'full22-v2') if destination == full22 else str(original)
    command = [x.replace(prefix, str(destination)) for x in command]
    command[1:1] = ['-I' + str(here / 'overlay'), '-I' + str(full22)]
    assert command[command.index('-o') + 1] == str(destination / 'server-spatialnorm')
    (destination / 'HELD-build-spatialnorm.json').write_text(
        json.dumps(command, indent=2) + '\n')

receipt = {
    'status': 'HELD: source templates only; final immutable local gate and independent review pending',
    'launch_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'runtime_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
    'original_global_manifest_path': str(frozen / 'manifest.json'),
    'original_global_manifest_sha256': expected_manifest,
    'all_original_frozen_small_files_verified': True,
    'copied_source_identities': copied,
    'candidate_helper_bound': False,
    'scientific_compilation_authorized': False,
    'scientific_compilation_or_matrix_or_propagation_performed': False,
    'planned_scope': 'distinct conformal-Q/physical-P lapse blend plus preferred shift and sigma5 null feedback; C0 geometry and damping',
    'native_evolution_owned_by_root': True,
    'no_old_outputs_overwritten': True,
}
(here / 'SOURCE_PREPARATION_HOLD.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Prepared source-only templates. Candidate binding, compile, matrix and propagation remain HELD.')
