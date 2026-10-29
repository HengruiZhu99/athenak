#!/usr/bin/env python3
"""Bounded experimental V5 native MPI/ownership/restart checks; no stability admission."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import time

HEADER = struct.Struct('<16s4i3Q2d')
DISSIPATION = struct.Struct('<dii')
NORMAL = struct.Struct('<4i')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--frozen-control', type=Path)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix='sampled-normal-mpi-')
    output = (args.output_dir or Path(temporary.name)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), 'evidence directory must be empty'
    deadline = time.monotonic()+390
    records = []
    binary, control = args.binary.resolve(), args.control.resolve()

    def run(name, *, ranks=2, executable=binary, end=.001, dt=.00025,
            epsilon='2', restart='-', kind='pulse', success=True, error=None):
        prefix = output/name
        command = [str(args.mpiexec), '-n', str(ranks), str(executable), '6', str(prefix),
                   kind, str(end), str(dt), '-', str(restart), '150', '.001', '.001', '6', '12']
        if epsilon is not None:
            command.append(str(epsilon))
        remaining = deadline-time.monotonic()
        assert remaining > 0, 'bounded integration test budget exhausted'
        started = time.monotonic()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=min(180, remaining))
        (output/f'{name}.log').write_text(result.stdout)
        records.append(dict(name=name, command=command, returncode=result.returncode,
                            seconds=time.monotonic()-started))
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
            assert report['scri_dissipation_epsilon'] == float(epsilon if epsilon is not None else 2)
            assert report['scri_dissipation_degree'] == 7
            assert report['scri_dissipation_cfl_max'] <= 2.7
            if executable == binary:
                assert report['checkpoint_schema'] == 'HYP3D_MPI_V5'
                assert report['checkpoint_header_bytes'] == 104
                assert report['scri_normal_derivative'] == 'sampled_native_numerator6'
                assert report['scri_normal_radial_nodes'] == 6
                assert report['scri_normal_bulk_derivative_order'] == 4
            else:
                assert report['checkpoint_schema'] == 'HYP3D_MPI_V4'
                assert report['checkpoint_header_bytes'] == 88
                assert 'scri_normal_derivative' not in report
        return prefix.with_suffix('.checkpoint'), report

    def checkpoint(path, version=5):
        data = path.read_bytes()
        header = HEADER.unpack_from(data)
        assert header[0].rstrip(b'\0') == f'HYP3D_MPI_V{version}'.encode()
        assert header[1:5] == (6, 31, 1, 1)
        offset = 104 if version == 5 else 88
        assert len(data) == offset+16*(31*header[5]+header[6])
        assert DISSIPATION.unpack_from(data, 72)[1:] == (7, 0)
        if version == 5:
            assert NORMAL.unpack_from(data, 88) == (1, 6, 4, 0)
        values = array('d'); values.frombytes(data[offset:])
        assert all(map(math.isfinite, values)), 'nonfinite checkpoint state/compensation'
        return header, data[offset:]

    # Surface rows split independently of bulk-cell ownership, already with two
    # ranks. Twelve-way ownership is a separate single-process component check;
    # actual twelve-rank MPI belongs to Aurora preflight, not this local test.
    # Complete checkpoint bytes include both RK compensation arrays and final D.
    direct, direct_report = run('v5-direct-one-rank', ranks=1, epsilon=None)
    two, _ = run('v5-direct-two-ranks', ranks=2)
    first, _ = run('v5-part-one', ranks=2, end=.0005)
    restarted, _ = run('v5-restarted', ranks=2, restart=first)
    assert direct.read_bytes() == two.read_bytes(), '2-rank mixed ownership changed checkpoint bytes'
    assert direct.read_bytes() == restarted.read_bytes(), '2+2 restart changed checkpoint bytes'
    header, payload = checkpoint(direct)
    assert header[7] == 4 and DISSIPATION.unpack_from(direct.read_bytes(), 72) == (2., 7, 0)

    zero, _ = run('v5-epsilon-zero', ranks=2, epsilon='0', end=.0005)
    checkpoint(zero)
    assert DISSIPATION.unpack_from(zero.read_bytes(), 72) == (0., 7, 0)
    _, mink = run('v5-minkowski', ranks=2, kind='minkowski', end=.0005)
    assert mink['max_state_deviation'] < 2e-10, 'short Minkowski identity lost'

    old, _ = run('v4-control', executable=control, end=.0005)
    checkpoint(old, 4)
    frozen_match = None
    if args.frozen_control:
        frozen, _ = run('v4-frozen-control', executable=args.frozen_control.resolve(), end=.0005)
        frozen_match = old.read_bytes() == frozen.read_bytes()
        assert frozen_match, 'default V4 checkpoint changed from frozen executable'

    run('v4-rejected-by-v5', restart=old, success=False,
        error='checkpoint schema/configuration/admission mismatch')
    run('v5-rejected-by-v4', executable=control, restart=first, success=False,
        error='checkpoint schema/configuration/admission mismatch')
    run('epsilon-mismatch', restart=first, epsilon='1', success=False,
        error='checkpoint angular dissipation configuration mismatch')
    for index, (label, bad) in enumerate([('scheme', 0), ('radial-nodes', 7), ('bulk-order', 2), ('reserved', 1)]):
        data = bytearray(first.read_bytes())
        struct.pack_into('<i', data, 88+4*index, bad)
        path = output/f'wrong-{label}.checkpoint'; path.write_bytes(data)
        run('rejected-'+label, restart=path, success=False,
            error='checkpoint sampled-normal configuration mismatch')
    truncated = output/'truncated-normal.checkpoint'
    truncated.write_bytes(first.read_bytes()[:100])
    run('truncated-rejected', restart=truncated, success=False,
        error='checkpoint sampled-normal configuration mismatch')
    unadmitted = bytearray(first.read_bytes()); struct.pack_into('<i', unadmitted, 28, 0)
    rejected = output/'unadmitted-copy.checkpoint'; rejected.write_bytes(unadmitted)
    run('unadmitted-rejected', restart=rejected, success=False,
        error='checkpoint schema/configuration/admission mismatch')

    binaries = [binary, control]+([args.frozen_control.resolve()] if args.frozen_control else [])
    summary = dict(status='passed', experimental=True,
                   scope='four-step native MPI ownership/restart component tests only; no duration or convergence admission',
                   one_vs_two_rank_checkpoint_bitwise=True,
                   two_rank_restart_vs_one_rank_direct_bitwise=True,
                   frozen_v4_checkpoint_bitwise=frozen_match,
                   normal_configuration_rejection_checks=5, cross_schema_rejections=2,
                   backend=direct_report['backend'], commands=records,
                   binaries={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
                   artifacts=[dict(path=p.name, bytes=p.stat().st_size,
                                   sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                              for p in sorted(output.iterdir()) if p.is_file()])
    (output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('commands', 'binaries', 'artifacts')}))


if __name__ == '__main__':
    main()
