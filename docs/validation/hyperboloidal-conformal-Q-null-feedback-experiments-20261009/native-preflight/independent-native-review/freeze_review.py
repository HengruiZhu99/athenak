"""Run cheap read-only checks, capture reviewed source bytes and freeze once."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
NATIVE = ROOT/'build-layer-research/q-null-native'
PY = '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
D = P/'immutable-independent-Q-null-native-review-20261009'
assert not D.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
commands = []
for name in ('review_build', 'review_utility'):
    command = [PY, str(P/(name+'.py'))]
    start = time.monotonic()
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    for stream in ('stdout', 'stderr'):
        (P/(name+'.'+stream)).write_text(getattr(run, stream))
    commands.append({'command': command, 'returncode': run.returncode,
                     'seconds': time.monotonic()-start,
                     'stdout': name+'.stdout', 'stdout_sha256': sha(P/(name+'.stdout')),
                     'stderr': name+'.stderr', 'stderr_sha256': sha(P/(name+'.stderr'))})
    (P/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    assert run.returncode == 0 and run.stderr == ''
captured = ['build_native.py', 'audit_native.py', 'launch_preflight.py', 'native.athinput',
            'native-build/build-receipt.json', 'native-build/include/native_q_injection.hpp',
            'snapshot-audit/check_snapshot.cpp', 'snapshot-audit/run_snapshot_audit.py',
            'snapshot-audit/compile_utility.py', 'snapshot-audit/compile-command.json',
            'snapshot-audit/compile-receipt.json', 'snapshot-audit/snapshot-results.json']
captured += [str(p.relative_to(NATIVE)) for p in sorted((NATIVE/'snapshot-audit/first-enum-namespace-compile-failure').iterdir()) if p.is_file()]
metadata = {}
for name in captured:
    source, target = NATIVE/name, P/'as-reviewed'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert source.read_bytes() == target.read_bytes()
    metadata[str(source.relative_to(ROOT))] = {'sha256': sha(source), 'bytes': source.stat().st_size}
audits = {}
for stage in ('reference', 'short'):
    source = NATIVE/'physical-inner/audit'/(stage+'.json')
    audit = json.loads(source.read_text())
    assert audit['status'] == 'PASS' and len(audit['private_snapshots']) == 3
    assert audit['initial_active25_binary64_bitwise_equal']
    assert audit['coordinates_masks_and_grid_bitwise_equal']
    for row in audit['private_snapshots']:
        assert row['all_25_active_fields_finite'] and row['all_25_BIN_quantizations_bitwise_equal']
        assert row['alpha_min'] > 0 and row['chi_min'] > 0 and row['physical_metric_eigen_min'] > 0
    audits[stage] = {'path': str(source.relative_to(ROOT)), 'sha256': sha(source),
                      'final_time': audit['exact_final_comparison_time'],
                      'recorded_original_HST_diagnostics': audit['original_HST_diagnostics']}
(P/'external-inputs.json').write_text(json.dumps({'captured_source_paths': metadata,
                                                'root_native_state_audit_records_not_rerun': audits}, indent=2)+'\n')
receipt = {'status': 'PASS_READ_ONLY_NATIVE_WIRING_AND_UTILITY_REVIEW',
           'commands': commands, 'seconds': sum(q['seconds'] for q in commands),
           'no_native_binary_compile_or_propagation_executed': True,
           'scope': 'Build/audit/launcher wiring and values-only snapshot provenance/algebra; no stability/scri/puncture/BH acceptance.'}
(P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
files = sorted(p for p in P.rglob('*') if p.is_file())
D.mkdir()
entries = {}
for source in files:
    name = str(source.relative_to(P))
    target = D/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    entries[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
index = {'scope': receipt['scope'], 'files': entries, 'file_count': len(entries),
         'total_bytes': sum(q['bytes'] for q in entries.values()),
         'math_core_index_sha256': 'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96',
         'independent_math_review_index_sha256': 'a456408b97c35ddf34ffe0af9c52411a13d990c9fa95f83a987234de24a3b650'}
(D/'index.json').write_text(json.dumps(index, indent=2)+'\n')
for name, entry in entries.items():
    assert sha(D/name) == entry['sha256']
print(json.dumps({'index': str(D/'index.json'), 'sha256': sha(D/'index.json'),
                  'files': index['file_count'], 'bytes': index['total_bytes'],
                  'commands': len(commands), 'seconds': receipt['seconds']}, indent=2))
