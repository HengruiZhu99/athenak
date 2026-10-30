#!/usr/bin/python3.11
"""One fresh frozen-source build; bounded and storage guarded, never submits jobs."""
from pathlib import Path
import datetime
import json
import os
import sys

root = Path('/home/hzhu/hyperbolic-core-shell-20261028')
sys.path.insert(0, str(root / 'source-thin-probes-v1/tst/hyperbolic_shift_3d/aurora'))
import control
import storage
import worker

logs = root / 'build-logs/thin-probes-v1'
logs.mkdir(parents=True, exist_ok=True)
status_file = logs / 'build-status.json'
assert not status_file.exists()
assert not (root / 'build-sycl-mpi-thin-probes-v1').exists()
plan = {'storage': {'limit_bytes': 5000000000, 'planned_new_bytes': 350000000,
                    'atomic_checkpoint_reserve_bytes': 100000000},
        'commands': [{'kind': 'check'}]}
status = {'status': 'running', 'pid': os.getpid(),
          'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'storage_before': storage.require_capacity(root, plan, prospective=True),
          'source_manifest_sha256': control.verify_source(root, 'source-thin-probes-v1-manifest.json'),
          'build_script_sha256': control.digest(root / 'build-thin-probes-v1.sh')}
control.atomic(status_file, status)
try:
    result = worker.run_bounded(['bash', '-l', str(root / 'build-thin-probes-v1.sh')],
                                root, logs / 'runner.log', 3600, plan)
    status.update(result)
    if result['returncode'] or result['timed_out'] or result['storage_error']:
        raise RuntimeError('Build did not complete successfully')
    assert control.verify_source(root, 'source-thin-probes-v1-manifest.json') == status['source_manifest_sha256']
    targets = ['mpi_thin_trace_sparse', 'mpi_thin_trace', 'paired_trace_device', 'thin_snapshot_profile']
    status['binaries'] = {str(p.relative_to(root)): control.digest(p)
                         for p in [root / 'build-sycl-mpi-thin-probes-v1/tst/hyperbolic_shift_3d' /
                                   ('hyperbolic_shift_3d_' + target) for target in targets]}
    status['cache_sha256'] = control.digest(root / 'build-sycl-mpi-thin-probes-v1/CMakeCache.txt')
    status['status'] = 'passed'
except Exception as error:
    status.update(status='failed', error=f'{type(error).__name__}: {error}')
status['storage_after'] = storage.footprint(root)
status['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
control.atomic(status_file, status)
raise SystemExit(0 if status['status'] == 'passed' else 1)
