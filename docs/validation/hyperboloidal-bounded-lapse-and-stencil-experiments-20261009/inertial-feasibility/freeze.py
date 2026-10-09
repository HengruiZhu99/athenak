"""Freeze small feasibility evidence; executables retained by metadata only."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
F = P/'immutable-killing-damping-feasibility-20261009'
F.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names = ('DERIVATION.md', 'normal_kernel.cpp', 'check_projection.py', 'run_audit.py',
         'freeze.py', 'receipt.json', 'commands-in-progress.json', 'run.log',
         'normal-release.json', 'normal-debug.json', 'projection.json', 'projection.log')
for n in names:
    (F/n).write_bytes((P/n).read_bytes())
dual = ROOT/'build-layer-research/continuum/discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp'
(F/'dependencies').mkdir()
(F/'dependencies/dual_helpers.hpp').write_bytes(dual.read_bytes())
files = {str(p.relative_to(F)): {'sha256': sha(p), 'bytes': p.stat().st_size}
         for p in F.rglob('*') if p.is_file()}
receipt = json.loads((F/'receipt.json').read_text())
assert receipt['status'] == 'PASS' and receipt['source_count'] == 369
assert len(receipt['commands']) == 5
assert all(q['returncode'] == 0 and q['stderr'] == '' for q in receipt['commands'])
assert (F/'normal-release.json').read_bytes() == (F/'normal-debug.json').read_bytes()
index = {'scope': 'Mathematical finiteOmega killing damping-vector feasibility only; not an alternative RHS/native admission',
         'files': files, 'file_count': len(files),
         'bytes': sum(v['bytes'] for v in files.values()),
         'large_external_by_hash_only': {
             str((P/n).relative_to(ROOT)): {'sha256': sha(P/n), 'bytes': (P/n).stat().st_size}
             for n in ('normal_kernel', 'normal_debug')},
         'receipt_sha256': sha(F/'receipt.json'), 'commands': 5,
         'actual_normal_damping_points_per_build': 1616, 'hundred_digit_limit_rows': 36,
         'recommendation': 'Do not admit direct swap under current explicit Omega timestep and raw field weights.',
         'no_evolution_or_production_change': True}
(F/'index.json').write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
for n, m in files.items():
    assert sha(F/n) == m['sha256'] and (F/n).stat().st_size == m['bytes']
print('PASS frozen', len(files), index['bytes'], sha(F/'index.json'))
