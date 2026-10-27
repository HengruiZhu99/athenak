#!/usr/bin/env python3
"""Bounded full tensor pilot for an explicitly checkpoint-bound halo radius."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import tarfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=.24)
    parser.add_argument('--puncture-scheme', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if not 0 < args.radius <= .25:
        parser.error('radius must lie in (0,.25]')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = Path.cwd()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = root/'build-shift-3d/elliptic_shift_3d_coupled_pilot'
    initial = root/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat'
    frozen = output/'pilot'
    shutil.copy2(binary, frozen)
    (output/'.gitignore').write_text('/pilot\n')
    depfile = root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/tst/elliptic_shift_3d/coupled.cpp.o.d'
    dependencies = depfile.read_text().replace('\\\n', ' ').split(':', 1)[1].split()
    sources = {Path(p).resolve() for p in dependencies}
    sources |= {root/'CMakeLists.txt', Path(__file__).resolve()}
    sources = sorted(p for p in sources if p.is_file() and root in p.parents)
    before = {str(p.relative_to(root)): digest(p) for p in sources}
    with tarfile.open(output/'compiled-source.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(root)))
    tag = digest(frozen)
    command = [str(frozen), '6', '0', '.0001', '2', '0', '7', '1e-10',
               str(output/'coupled.csv'), str(output/'coupled.checkpoint'), tag,
               '4', '1', '4', '8', '1', str(initial), '2', '0',
               '.45,.55,.60,.65,.70,.75,1', '7', '1', '2', '.24',
               '1', '1', '1', '3', '1', str(args.radius), str(args.puncture_scheme)]
    report = dict(command=command, maximum_seconds=180, workers=1,
                  active_cell_budget=50000, actual_active_cells=47088,
                  radius=args.radius, puncture_derivative_scheme=args.puncture_scheme,
                  binary_sha256=tag,
                  source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  compiled_dependencies=before, initial_sha256=digest(initial),
                  qualification='Full 3D research engine RK4, not native Driver tasks; unchanged scientific targets; all cells retained')
    manifest = output/'manifest.json'
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    started = time.monotonic()
    with (output/'pilot.stdout').open('w') as stdout, (output/'pilot.stderr').open('w') as stderr:
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr, start_new_session=True)
        report['owned_process_group'] = process.pid
        try:
            report['exit_code'] = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            report['exit_code'] = 124
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError('refusing cleanup outside the owned process group')
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            report['cleanup'] = 'only this owned process group terminated'
    report['seconds'] = time.monotonic()-started
    report['binary_stable'] = digest(frozen) == tag
    report['compiled_dependencies_stable'] = all(digest(root/name) == value for name, value in before.items())
    report['initial_stable'] = digest(initial) == report['initial_sha256']
    summaries = [json.loads(line) for line in (output/'pilot.stdout').read_text().splitlines() if line.startswith('{')]
    if summaries:
        result = summaries[-1]
        report['result'] = result
        report['completed_with_bitwise_restart'] = bool(result.get('restart_equal') and result.get('steps') == 2)
        report['original_vector_controls_pass'] = result['shift_residual_max'] <= 1e-10 and result['shift_residual_point_max'] <= 1e-7
        report['scri_pole_target'] = 1e-7
        report['scri_pole_pass'] = result['scri_tensor_pole'] <= 1e-7
    else:
        report['completed_with_bitwise_restart'] = False
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('compiled_dependencies', 'command')}, indent=2))


if __name__ == '__main__':
    main()
