#!/usr/bin/env python3
"""Compare continuous and split/restarted sixth-order residual vacuum controls.

Uses the double-precision MHD+Z4c checkpoint layout in outputs/restart.cpp.
Rank-file checkpoints require the same MPI partition on continuation.
"""
import argparse
from array import array
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys


def checkpoint(path):
    with path.open('rb') as stream:
        prefix = stream.read(262144)
        marker = b'<par_end>\n'
        stop = prefix.find(marker)
        assert stop >= 0, 'Missing parameter header'
        end = stop + len(marker)
        params, block = {}, None
        for line in prefix[:stop].decode().splitlines():
            line = line.split('#', 1)[0].strip()
            if line.startswith('<'):
                block = line[1:-1]
                params[block] = {}
            elif '=' in line:
                key, value = line.split('=', 1)
                params[block][key.strip()] = value.strip()
        assert {'mhd', 'z4c'} <= params.keys()
        assert not {'hydro', 'radiation', 'turbulence'} & params.keys()
        assert not any(k.startswith('co_') and k.endswith('_type') for k in params['z4c'])
        assert not any(k.startswith('dump_horizon_') and v == 'true'
                       for k, v in params['z4c'].items())
        stream.seek(end)
        total, level = struct.unpack('<ii', stream.read(8))
        stream.read(72 + 76)  # RegionSize and mesh RegionIndcs.
        indices = struct.unpack('<19i', stream.read(76))
        time, dt, cycle = struct.unpack('<ddi', stream.read(20))
        ng, nx, ny, nz = indices[:4]
        assert min(nx, ny, nz) > 1 and ng > 0
        stream.seek(20 * total + 16, 1)  # Locations/costs and Z4c output times.
        stride, = struct.unpack('<Q', stream.read(8))
        n1, n2, n3 = nx + 2*ng, ny + 2*ng, nz + 2*ng
        cells = n1*n2*n3
        nmhd = 5 + int(params['mhd'].get('nscalars', '0'))
        faces = (n1+1)*n2*n3 + n1*(n2+1)*n3 + n1*n2*(n3+1)
        offset = 8*(nmhd*cells + faces)
        assert stride == offset + 8*25*cells, 'Unsupported checkpoint payload'
        payload = stream.read()
        assert payload and len(payload) % stride == 0
        assert sys.byteorder == 'little', 'This reader expects a little-endian host'
        state = []
        for start in range(0, len(payload), stride):
            values = array('d')
            values.frombytes(payload[start+offset:start+stride])
            assert all(math.isfinite(x) for x in values)
            state.append(values)
        active = [((n*n3+k)*n2+j)*n1+i for n in range(25)
                  for k in range(ng, ng+nz) for j in range(ng, ng+ny)
                  for i in range(ng, ng+nx)]
        return {'time': time, 'dt': dt, 'cycle': cycle, 'total': total,
                'state': state, 'active': active, 'cells': cells, 'level': level}


def cohort(run, ranks, cycle):
    paths = sorted((run/'rst/rank_00000000').glob('*.rst'))
    matches = [p for p in paths if checkpoint(p)['cycle'] == cycle]
    assert matches, f'No checkpoint at cycle {cycle}'
    name = matches[-1].name
    files = [run/'rst'/f'rank_{r:08d}'/name for r in range(ranks)]
    assert set((run/'rst').glob('rank_*/'+name)) == set(files)
    records = [checkpoint(p) for p in files]
    assert len({(r['cycle'], r['time'], r['dt'], r['total']) for r in records}) == 1
    assert sum(len(r['state']) for r in records) == records[0]['total']
    return files[0], records


def launch(exe, launcher, ranks, run, arguments):
    run.mkdir(parents=True, exist_ok=False)
    with (run/'run.log').open('w') as log:
        subprocess.run([launcher, '-n', str(ranks), str(exe)] + arguments,
                       cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
    validate_run(run)
    return run


def validate_run(run):
    log = (run/'run.log').read_text()
    assert 'Terminating on cycle limit' in log and '### FATAL ERROR' not in log
    metric_checked = False
    for history in run.glob('*.hst'):
        lines = history.read_text().splitlines()
        labels = re.findall(r'\[\d+\]=([^\s]+)',
                            next(line for line in lines if '[1]=' in line))
        rows = [[float(v) for v in line.split()] for line in lines
                if line.strip() and not line.startswith('#')]
        assert rows and all(all(math.isfinite(v) for v in row) for row in rows)
        if 'bad-metric' in labels:
            metric_checked = True
            assert all(row[labels.index('bad-metric')] == 0 and
                       all(row[labels.index(name)] > 0
                           for name in ['alpha-min', 'chi-min', 'detg-min'])
                       for row in rows)
    assert metric_checked, 'Missing metric validity history'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ranks', type=int, nargs='+', default=[1, 4])
    p.add_argument('--launcher', default='mpiexec')
    a = p.parse_args()
    baseline = (Path(__file__).resolve().parents[1]/'inputs/z4c_ks_background.athinput').read_text()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for ranks in a.ranks:
        for case in ['vacuum', 'dipole', 'refined_vacuum', 'refined_dipole']:
            text = baseline.replace('debug_balance = true', 'debug_balance = false')
            refined, perturbed = case.startswith('refined_'), 'dipole' in case
            if refined:
                head, rest = text.split('<meshblock>', 1)
                for axis in range(1, 4):
                    head = re.sub(rf'^nx{axis}\s*=.*', f'nx{axis} = 32', head, flags=re.M)
                    head = re.sub(rf'^x{axis}min\s*=.*', f'x{axis}min = -8', head, flags=re.M)
                    head = re.sub(rf'^x{axis}max\s*=.*', f'x{axis}max = 8', head, flags=re.M)
                text = head + '<meshblock>' + rest
                text = text.replace('refinement = none', 'refinement = static')
                text = text.replace('max_nmb_per_rank = 8', 'max_nmb_per_rank = 128')
                text += ('\n<refined_region0>\nlevel = 1\n'
                         'x1min = -1\nx1max = 1\nx2min = -1\nx2max = 1\nx3min = -1\nx3max = 1\n')
            if perturbed:
                text = text.replace('<problem>', '<problem>\n'
                    'outer_sponge_test_theta_pulse_dipole_axis = 1\n'
                    'outer_sponge_test_theta_pulse_amplitude = 1e-8\n'
                    f'outer_sponge_test_theta_pulse_radius = {4 if refined else 3}\n'
                    'outer_sponge_test_theta_pulse_width = 0.3')
            text += '\n<output2>\nfile_type = rst\nsingle_file_per_rank = true\ndt = 100\n'
            base = output/f'{case}_r{ranks}'
            base.mkdir()
            deck = base/'input.athinput'
            deck.write_text(text)
            direct = launch(a.exe.resolve(), a.launcher, ranks, base/'continuous',
                            ['-i', str(deck), 'time/nlim=6'])
            first = launch(a.exe.resolve(), a.launcher, ranks, base/'first',
                           ['-i', str(deck), 'time/nlim=3'])
            saved, before = cohort(first, ranks, 3)
            restored = launch(a.exe.resolve(), a.launcher, ranks, base/'restored',
                              ['-r', str(saved), 'time/nlim=3'])
            _, after = cohort(restored, ranks, 3)
            for left, right in zip(before, after):
                for x, y in zip(left['state'], right['state']):
                    assert all(struct.pack('<d', x[i]) == struct.pack('<d', y[i])
                               for i in left['active']), 'Restart altered saved active state'
            resumed = launch(a.exe.resolve(), a.launcher, ranks, base/'resumed',
                             ['-r', str(saved), 'time/nlim=6'])
            _, ref = cohort(direct, ranks, 6)
            _, rst = cohort(resumed, ranks, 6)
            assert ref[0]['time'] == rst[0]['time'] and ref[0]['dt'] == rst[0]['dt']
            difference = magnitude = 0.0
            for left, right in zip(ref, rst):
                assert len(left['state']) == len(right['state'])
                for x, y in zip(left['state'], right['state']):
                    if not perturbed:
                        assert x.tobytes() == bytes(len(x)*8)
                        assert y.tobytes() == bytes(len(y)*8)
                    for index in left['active']:
                        difference = max(difference, abs(x[index]-y[index]))
                        magnitude = max(magnitude, abs(x[index]))
            assert difference < 1e-12, 'Restart changed the physical residual response'
            if perturbed:
                assert magnitude > 1e-10, 'Perturbation was erased'
            results[base.name] = {'time': ref[0]['time'], 'max_active_difference': difference,
                'max_active_residual': magnitude, 'exact_zero': not perturbed,
                'restored_active_state_bitwise_equal': True}
    for case in ['vacuum', 'dipole', 'refined_vacuum', 'refined_dipole']:
        assert all(results[f'{case}_r{r}'] == results[f'{case}_r{a.ranks[0]}']
                   for r in a.ranks), 'Restart response depends on MPI partition'
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
