"""Read-only six-snapshot values-only algebra/provenance review; no binary run."""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct

import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
NATIVE = ROOT/'build-layer-research/q-null-native'
UTILITY = NATIVE/'snapshot-audit'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = json.loads((UTILITY/'snapshot-results.json').read_text())
compile_receipt = json.loads((UTILITY/'compile-receipt.json').read_text())
assert result['status'] == 'PASS' and compile_receipt['returncode'] == 0
assert result['compile_command'] == compile_receipt['command']
assert result['compile_command'] == json.loads((UTILITY/'compile-command.json').read_text())
assert sha(UTILITY/'check_snapshot.cpp') == compile_receipt['source_sha256']
assert sha(UTILITY/'check_snapshot') == compile_receipt['executable_sha256'] == result['utility_executable_sha256']
assert (UTILITY/'compile.log').read_bytes() == b''
assert sha(UTILITY/'compile.log') == result['compile_log_sha256']
assert result['private_build_receipt_sha256'] == '8f7e60ef5d5df9dad31f8feba359b244b36859aa115d525d2f52b057a8813508'
for name, digest in compile_receipt['all_repository_dependencies_sha256'].items():
    assert sha(ROOT/name) == digest, name
for name, digest in compile_receipt['libraries_sha256'].items():
    assert sha(ROOT/name) == digest, name
for name, digest in result['source_sha256'].items():
    assert sha(ROOT/name) == digest, name
source = (UTILITY/'check_snapshot.cpp').read_text()
failed = UTILITY/'first-enum-namespace-compile-failure'
assert (failed/'check_snapshot.cpp').read_text().replace('Z4c::', 'z4c::Z4c::') == source
failure = json.loads((failed/'receipt.json').read_text())
assert failure['returncode'] != 0 and (failed/'compile.log').stat().st_size > 0
assert 'auto u=p.state;' in source
assert 'const auto five=qnf::Gauge(p,u,gauge,{.85,.95,5,true});' in source
assert 'const auto zero=qnf::Gauge(p,u,gauge,{.85,.95,0,true});' in source
assert 'actual/p.omega' in source
assert 'LoadMeshJet' not in source  # Deliberately not full live derivative jets.
readerpath = ROOT/'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py'
assert sha(readerpath) == '74c62c97afd113819438d8be42be84cc605902983bd974022ebb7e0747f3cb13'
spec = importlib.util.spec_from_file_location('independent_q_snapshot_reader', readerpath)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
assert len(result['rows']) == 6
errors = {n: 0. for n in ('manual_null_error', 'manual_beta_pole_error', 'single_pole_assembly_error')}
times = {'reference': [], 'short': []}
for row in result['rows']:
    assert row['returncode'] == 0 and row['stderr'] == ''
    rst, raw = ROOT/row['restart'], ROOT/row['raw_file']
    assert sha(rst) == row['restart_sha256'] and sha(raw) == row['raw_sha256']
    checkpoint = reader.read_rst(rst)
    assert checkpoint['time'] == row['time']
    content = raw.read_bytes()
    n = struct.unpack('<3i', content[:12])
    first = struct.unpack('<3d', content[12:36])
    spacing = struct.unpack('<3d', content[36:60])
    assert n == (30, 30, 30)
    assert np.array_equal(np.frombuffer(content[60:], dtype='<f8').reshape((25, 30, 30, 30)), checkpoint['u'])
    ng = checkpoint['mb_indcs']['ng']
    expected_spacing = [checkpoint['mesh_size'][f'dx{d}'] for d in (1, 2, 3)]
    expected_first = [checkpoint['mesh_size'][f'x{d}min']+(.5-ng)*expected_spacing[d-1] for d in (1, 2, 3)]
    assert tuple(expected_spacing) == spacing and tuple(expected_first) == first
    assert row['computed']['active_cells'] == 6152 and row['computed']['outer_cells'] == 2312
    for name in errors:
        errors[name] = max(errors[name], row['computed'][name])
        assert row['computed'][name] < 1e-10
    times[row['run']].append(row['time'])
assert times['reference'][0] == times['short'][0] == 0.
assert times['reference'][-1] == .05 and times['short'][-1] == .02
out = {'status': 'PASS_READ_ONLY_VALUES_ONLY_SNAPSHOT_REVIEW',
       'snapshot_receipt_sha256': sha(UTILITY/'snapshot-results.json'),
       'utility_source_sha256': sha(UTILITY/'check_snapshot.cpp'),
       'utility_compile_receipt_sha256': sha(UTILITY/'compile-receipt.json'),
       'utility_executable_sha256': result['utility_executable_sha256'],
       'dependency_count': len(compile_receipt['all_repository_dependencies_sha256']),
       'library_count': len(compile_receipt['libraries_sha256']),
       'six_full25_raw_arrays_equal_parsed_binary64_RST': True,
       'times': times, 'maximum_algebra_errors': errors,
       'failed_compile_preserved': {str(p.relative_to(ROOT)): sha(p) for p in sorted(failed.iterdir()) if p.is_file()},
       'compile_repair_only_namespace_qualifiers': True,
       'scope': 'Read-only algebra, original compiled provenance and six raw/RST byte64 consistency checks. Utility retains reference derivative jets and tests sigma5-minus0 algebraic increment; no rerun, complete live-source reconstruction, falloff, stability, native improvement or BH claim.'}
(P/'utility-review-result.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
