"""Freeze reviewed mathematical bytes, without rerunning or editing the gate."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import sympy

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
F = P/'immutable-Cartesian-total-J-basis-20261009'
assert not F.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
root_review = ROOT/'build-layer-research/continuum/total-j-root-basis-review.json'
boundary = ROOT/'build-layer-research/boundary/total-j-basis-independent-review-20261009'
assert sha(root_review) == '5315c569daee5d87210954bbd06db14891748e25969a3312a2ce157f1fe5952a'
assert sha(boundary/'receipt.json') == '40a3824a8b109408bd0587259071fbc2124165422f758f2807cd83d0319122f2'
assert sha(boundary/'REVIEW.md') == '8508c262e3b76bc8835643e932dd808b65125ecab10318b1d4f538c9374c0161'
for name, pin in json.loads(root_review.read_text())['reviewed_files'].items():
    assert sha(P/name) == pin
for row in json.loads((boundary/'receipt.json').read_text())['reviewed_sources']:
    assert sha(Path(row['path'])) == row['sha256']
receipt = json.loads((P/'receipt.json').read_text())
assert receipt['source_unchanged'] and all(r['returncode'] == 0 for r in receipt['commands'])
environment = {'python_version': sys.version, 'python_binary': str(Path(sys.executable).resolve()),
               'python_binary_sha256': sha(Path(sys.executable).resolve()),
               'sympy_version': sympy.__version__, 'sympy_module_sha256': sha(Path(sympy.__file__)),
               'compiler_wrapper_sha256': sha(Path('/usr/bin/c++'))}
(P/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
F.mkdir()
for p in P.iterdir():
    if p.is_file() and p.suffix in ['.py', '.cpp', '.hpp', '.md', '.json', '.stdout', '.stderr']:
        shutil.copyfile(p, F/p.name)
(F/'reviews').mkdir()
shutil.copyfile(root_review, F/'reviews/root-review.json')
shutil.copyfile(boundary/'receipt.json', F/'reviews/boundary-review-receipt.json')
shutil.copyfile(boundary/'REVIEW.md', F/'reviews/boundary-REVIEW.md')
entries = [{'path': str(p.relative_to(F)), 'sha256': sha(p), 'bytes': p.stat().st_size}
           for p in sorted(F.rglob('*')) if p.is_file()]
index = {'kind': 'reviewed-mathematical-Cartesian-total-J-basis-only',
         'freeze_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'files': entries, 'source_inputs': len(receipt['source_before']),
         'all_m_records': 69, 'm0_full_channel_counts': [8, 16, 20],
         'commands_all_zero': 6, 'release_debug_byte_equal': True,
         'review_pins': {'root': sha(root_review), 'boundary': sha(boundary/'receipt.json')},
         'omitted_binaries_by_hash': {name: {'sha256': sha(P/name), 'bytes': (P/name).stat().st_size}
                                      for name in ['probe-release', 'probe-debug']},
         'scope': 'Exact Cartesian harmonic polynomial/jet/conversion mathematics. No PDE kernel/operator, boundary, propagation, stability, or evolution admission.'}
(F/'index.json').write_text(json.dumps(index, indent=2)+'\n')
for row in entries:
    assert sha(F/row['path']) == row['sha256']
print(json.dumps({'index': str(F/'index.json'), 'sha256': sha(F/'index.json'),
                  'files': len(entries), 'bytes': sum(r['bytes'] for r in entries),
                  'passed_final_readback': True}, indent=2))
