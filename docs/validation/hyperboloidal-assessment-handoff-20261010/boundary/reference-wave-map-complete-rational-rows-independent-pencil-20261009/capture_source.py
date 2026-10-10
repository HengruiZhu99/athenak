#!/usr/bin/env python3
"""One-shot stdlib metadata capture; no candidate import or target evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/hz0693/research/hyperboloidal')
RESEARCH = ROOT / 'build-layer-research'
INPUTS = [
    ('GaugeFar.hpp', 'boundary/reference-wave-map-outer-arithmetic-source001-held-20261009/inputs/reference_wave_map.hpp'),
    ('LegacyGauge-and-Assemble.hpp', 'boundary/reference-wave-map-outer-arithmetic-source001-held-20261009/inputs/reference_wave_map_legacy.hpp'),
    ('far-complete-dual-oracle.py', 'boundary/reference-wave-map-far-dual-source002-held-20261009/oracle.py'),
    ('rational-gradient-ASSESSMENT.md', 'continuum/reference-wave-map-rational-gradient-flux-pencil-20261009/ASSESSMENT.md'),
    ('rational-gradient-index.json', 'continuum/reference-wave-map-rational-gradient-flux-pencil-20261009/index.json'),
    ('exact-inverse-ASSESSMENT.md', 'continuum/exact-3x3-dual-inverse-pencil-20261009/ASSESSMENT.md'),
    ('exact-inverse-index.json', 'continuum/exact-3x3-dual-inverse-pencil-20261009/index.json'),
    ('literature-paused-capture.json', 'continuum/reference-wave-map-complete-rational-rows-pencil-20261009/capture.json'),
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def item(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    started = time.perf_counter()
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize:
        raise RuntimeError('Require isolated, unoptimized, bytecode-off Python')
    if os.environ.get('PYTHONOPTIMIZE') != '0':
        raise RuntimeError('Require explicit PYTHONOPTIMIZE=0')
    destination = HERE / 'inputs'
    if destination.exists() or (HERE / 'index.json').exists():
        raise RuntimeError('Fresh one-shot capture only')
    external = [(name, RESEARCH / rel) for name, rel in INPUTS]
    external.append(('production-Geometry.hpp', ROOT / 'src/z4c/hyperboloidal/conformal_constraints.hpp'))
    local = [HERE / name for name in ('PLAN.md', 'ASSESSMENT.md', 'capture_source.py')]
    before = [item(path) for path in local + [path for _, path in external]]
    if before[3]['sha256'] != 'f7a43a53481d3147c4394628de55651f8605fb6c0a8758b6b0a12116a5f9cfa8':
        raise RuntimeError('Unexpected frozen GaugeFar source')
    destination.mkdir()
    copied = []
    for name, source in external:
        target = destination / name
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise RuntimeError('Copy hash mismatch')
        copied.append({'source': str(source), 'copy': str(target.relative_to(HERE)),
                       'sha256': sha(target), 'bytes': target.stat().st_size})
    after = [item(path) for path in local + [path for _, path in external]]
    if before != after:
        raise RuntimeError('Source input changed during capture')
    save(HERE / 'source-pins.json', {'before': before, 'after': after,
                                   'inputs_unchanged': True, 'copied': copied})
    runtime = {'argv': sys.argv, 'executable': sys.executable,
               'resolved_executable': str(Path(sys.executable).resolve()),
               'executable_sha256': sha(Path(sys.executable).resolve()),
               'isolated': bool(sys.flags.isolated), 'optimize': sys.flags.optimize,
               'dont_write_bytecode': sys.dont_write_bytecode,
               'PYTHONOPTIMIZE': os.environ.get('PYTHONOPTIMIZE'),
               'historical_observed_HEAD': '284b4c21e09077ab86f0d0cbbbb5b3a11503cf58'}
    save(HERE / 'receipt.json', {
        'passed': True, 'source_math_pencil_only': True, 'inputs_unchanged': True,
        'candidate_execution': False, 'numerical_or_CAS_verification': False,
        'payload_decode': False, 'source_input_count': len(before),
        'source_copy_count': len(copied), 'runtime': runtime,
        'scope': 'Manual exact-row derivation and format-bound proof; metadata capture only',
        'elapsed_seconds': time.perf_counter() - started,
        'existing_far_dual_failure_upgraded': False,
        'backend_source_or_execution_accepted': False,
    })
    files = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file():
            record = item(path)
            record['path'] = str(path.relative_to(HERE))
            files.append(record)
    save(HERE / 'index.json', {'scope': 'SOURCE_ONLY_MANUAL_MATHEMATICAL_PENCIL',
                               'files': files, 'candidate_execution': False,
                               'inputs_unchanged': True})
    print(json.dumps({'passed': True, 'index_sha256': sha(HERE / 'index.json'),
                      'receipt_sha256': sha(HERE / 'receipt.json'),
                      'assessment_sha256': sha(HERE / 'ASSESSMENT.md'),
                      'file_records': len(files)}, allow_nan=False))


if __name__ == '__main__':
    main()
