#!/usr/bin/env python3
"""Bounded native V4 integration checks; no duration or convergence admission."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time

HEADER = struct.Struct('<16s4i3Q2d')
CONFIG = struct.Struct('<dii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix='dissipative-mpi-')
    output = args.output_dir or Path(temporary.name)
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), 'evidence directory must be empty'
    deadline = time.monotonic()+390
    records = []
    binary, control = args.binary.resolve(), args.control.resolve()

    def run(name, *, ranks=2, executable=binary, end=.001, dt=.00025,
            epsilon='2', restart='-', success=True, error=None):
        prefix = output/name
        command = [str(args.mpiexec), '-n', str(ranks), str(executable), '6', str(prefix),
                   'pulse', str(end), str(dt), '-', str(restart), '150', '.001', '.001', '6', '12']
        if epsilon is not None:
            command.append(str(epsilon))
        remaining = deadline-time.monotonic()
        assert remaining > 0, 'bounded integration test budget exhausted'
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=min(180, remaining))
        (output/f'{name}.log').write_text(result.stdout)
        records.append(dict(name=name, command=command, returncode=result.returncode))
        assert (result.returncode == 0) == success, (name, result.stdout)
        report_path = prefix.with_suffix('.report.json')
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if error:
            assert error in result.stdout, (name, result.stdout)
        if success:
            assert report['status'] == 'completed' and report['t'] == end
            assert report['checkpoint_admitted'] and report['diagnostics_current']
            assert report['mpi_size'] == ranks
            assert report['max_null_surface'] < 5e-13
            assert report['max_null_ghost'] < 5e-13
            assert report['max_determinant_residual'] < 5e-13
            assert report['max_tracefree_residual'] < 5e-13
            assert report['checkpoint_header_bytes'] == (88 if executable == binary else 72)
            if executable == binary:
                assert report['scri_dissipation_epsilon'] == float(epsilon)
                assert report['scri_dissipation_degree'] == 7
                assert report['scri_dissipation_cfl_max'] <= 2.7
        return prefix.with_suffix('.checkpoint'), report

    def checkpoint(path, v4=True):
        data = path.read_bytes()
        header = HEADER.unpack_from(data)
        assert header[0].rstrip(b'\0') == (b'HYP3D_MPI_V4' if v4 else b'HYP3D_MPI_V3')
        offset = 88 if v4 else 72
        assert len(data) == offset+16*(31*header[5]+header[6])
        if v4:
            config = CONFIG.unpack_from(data, 72)
            assert config[1:] == (7, 0)
        return header, data[offset:]

    old, _ = run('v3-control', executable=control, epsilon=None)
    zero, _ = run('v4-zero', epsilon='0')
    old_header, old_payload = checkpoint(old, False)
    zero_header, zero_payload = checkpoint(zero)
    assert old_header[1:] == zero_header[1:] and old_payload == zero_payload

    first, _ = run('v4-part-one', end=.0005)
    restarted, _ = run('v4-restarted', restart=first)
    direct, direct_report = run('v4-direct-one-rank', ranks=1)
    assert restarted.read_bytes() == direct.read_bytes(), 'partition/restart changed any checkpoint byte'
    header, payload = checkpoint(direct)
    assert header[7] == 4 and CONFIG.unpack_from(direct.read_bytes(), 72) == (2., 7, 0)

    run('epsilon-mismatch', restart=first, epsilon='1', success=False,
        error='checkpoint angular dissipation configuration mismatch')
    run('v3-rejected-by-v4', restart=old, success=False,
        error='checkpoint schema/configuration/admission mismatch')
    run('v4-rejected-by-v3', executable=control, epsilon=None, restart=first,
        success=False, error='checkpoint schema/configuration/admission mismatch')
    for name, epsilon in [('negative', '-1'), ('nan', 'nan'), ('infinite', 'inf'), ('suffix', '2junk')]:
        run('invalid-'+name, epsilon=epsilon, success=False, error='angular dissipation strength')
    rejected, report = run('cfl-rejected', epsilon='1000', success=False,
                          error='angular dissipation RK4 bound exceeded')
    assert report['status'] == 'failed' and report['t'] == 0 and not report['checkpoint_admitted']
    run('unadmitted-rejected', restart=rejected, epsilon='1000', success=False,
        error='checkpoint schema/configuration/admission mismatch')
    _, overflow_report = run('cfl-overflow-rejected', epsilon='1e308', dt=1,
                            success=False, error='angular dissipation RK4 bound exceeded')
    assert overflow_report['status'] == 'failed' and not overflow_report['checkpoint_admitted']
    assert overflow_report['scri_dissipation_cfl'] is None
    assert overflow_report['scri_dissipation_cfl_max'] is None

    summary = dict(status='passed', scope='native short-step integration and restart checks only',
                   zero_strength_matches_v3_payload_bitwise=True,
                   two_rank_restart_matches_one_rank_direct_bitwise=True,
                   backend=direct_report['backend'], commands=records,
                   binaries={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (binary, control)},
                   artifacts=[dict(path=p.name, bytes=p.stat().st_size,
                                   sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                              for p in sorted(output.iterdir()) if p.is_file()])
    (output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('commands', 'binaries', 'artifacts')}))


if __name__ == '__main__':
    main()
