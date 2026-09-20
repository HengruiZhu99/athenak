#!/usr/bin/env python3
"""Compact constraint seed: support, nonzero response, MPI parity and default parity.

Requires a z4c_tov_ks MPI binary. --reference-exe optionally verifies bitwise
compatibility of the default Gaussian with the pre-change executable. This
three-cycle test establishes centered-monopole initialization semantics, not
long-term stability. C-infinity refers to that centered monopole fixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis/outer_boundary/long_sponge/gpu'))
from check_minkowski_checkpoint import cohort, validate


def blocks(run, ranks, cycle):
    path, records = cohort(run, ranks, cycle)
    raw = path.read_bytes()
    end = raw.index(b'<par_end>\n') + len(b'<par_end>\n')
    start = end + 8 + 72 + 2*76 + 20
    locs = [struct.unpack_from('<4i', raw, start+16*i)
            for i in range(records[0]['total'])]
    data = [np.asarray(v).reshape(25, 24, 24, 24)
            for rec in records for v in rec['state']]
    assert len(data) == len(locs) == 8
    return list(zip(locs, data))


def physical_payloads(run, ranks, cycle):
    """Raw per-block MHD + face-B + Z4c bytes, excluding mutable metadata/text."""
    path, records = cohort(run, ranks, cycle)
    payloads = []
    for rank in range(ranks):
        raw = (run/'rst'/f'rank_{rank:08d}'/path.name).read_bytes()
        end = raw.index(b'<par_end>\n')+len(b'<par_end>\n')
        total, = struct.unpack_from('<i',raw,end)
        stride_offset = end+8+72+2*76+20+20*total+16
        stride, = struct.unpack_from('<Q',raw,stride_offset)
        start = stride_offset+8
        assert len(raw)-start == len(records[rank]['state'])*stride
        payloads.extend(raw[n:n+stride] for n in range(start,len(raw),stride))
    assert len(payloads)==8
    return payloads


def boundary_support_check(run, ranks):
    """All initial physical-face stencil belts, including corner/edge ghosts.

    ng=4 volume stencils followed by the active one-sided D2 boundary
    derivative can reach six active cells inward from the first active cell.
    Seven active layers plus all four physical ghost layers therefore cover
    the composed initial rate stencil as well as incoming state derivatives.
    This assertion is fixture-specific; it is not a future-time guarantee.
    """
    values_checked = 0
    for loc, u in blocks(run, ranks, 0):
        mask = np.zeros(u.shape[1:], dtype=bool)
        for axis in range(3):
            belt = [slice(None)]*3
            belt[2-axis] = slice(0, 11) if loc[axis] == 0 else slice(-11, None)
            mask[tuple(belt)] = True
        assert np.count_nonzero(u[:, mask]) == 0, 'Nonzero initial physical-face stencil/ghost belt'
        values_checked += 25*int(mask.sum())
    return {'all_initial_physical_stencil_and_ghost_values_zero': True,
            'values_checked': values_checked, 'physical_block_faces': 24,
            'active_layers_per_face': 7, 'ghost_layers_per_face': 4,
            'scope': 'initial checkpoint only; ng4 volume plus D2 boundary footprint; no later-time zero claim'}


def seed_check(run, ranks, amplitude):
    peak = 0.
    for loc, u in blocks(run, ranks, 0):
        active = u[:, 4:20, 4:20, 4:20]
        assert np.count_nonzero(np.delete(active, 17, axis=0)) == 0
        axes = [-2048 + (loc[a]*16 + np.arange(16)+.5)*128
                for a in range(3)]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
        q2 = (x*x+y*y+z*z)/512**2
        inside = q2 < 1
        expected = np.zeros_like(q2)
        expected[inside] = amplitude*np.exp(1-1/(1-q2[inside]))
        assert np.count_nonzero(active[17][~inside]) == 0
        assert np.max(abs(active[17]-expected)) <= abs(amplitude)*2e-14
        peak = max(peak, float(abs(active[17]).max()))
    assert (peak > 0) == (amplitude != 0)
    return peak


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--reference-exe', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launcher', default='mpiexec')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'analysis/outer_boundary/long_sponge/constraint-pulse/pulse_gate_loweta_coremask.athinput'
    base = source.read_text().split('<output3>')[0]
    mesh, rest = base.split('<meshblock>', 1)
    mesh = re.sub(r'^(nx[123])\s*=.*', r'\1 = 32', mesh, flags=re.M)
    block, rest = rest.split('<mesh_refinement>', 1)
    block = re.sub(r'^(nx[123])\s*=.*', r'\1 = 16', block, flags=re.M)
    base = mesh+'<meshblock>'+block+'<mesh_refinement>'+rest
    base = re.sub(r'^nlim\s*=.*', 'nlim = 3', base, flags=re.M)
    base = re.sub(r'^outer_sponge_test_theta_pulse_width\s*=.*',
                  'outer_sponge_test_theta_pulse_width = 512', base, flags=re.M)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    results, states, payloads = {}, {}, {}
    cases = [('compact_zero', 'compact', 0., 1),
             ('compact_r1', 'compact', 1e-6, 1),
             ('compact_r2', 'compact', 1e-6, 2),
             ('compact_double', 'compact', 2e-6, 2),
             ('gaussian_default', None, 1e-6, 2),
             ('gaussian_explicit', 'gaussian', 1e-6, 2)]
    if a.reference_exe:
        cases.append(('gaussian_reference', None, 1e-6, 2))
    for name, profile, amplitude, ranks in cases:
        run = a.output/name
        run.mkdir()
        text = re.sub(r'^outer_sponge_test_theta_pulse_amplitude\s*=.*',
                      f'outer_sponge_test_theta_pulse_amplitude = {amplitude}', base, flags=re.M)
        if profile:
            text = text.replace('<problem>', '<problem>\n'
                f'outer_sponge_test_theta_pulse_profile = {profile}')
        (run/'input.athinput').write_text(text)
        exe = a.reference_exe if name.endswith('reference') else a.exe
        cmd = [a.launcher, '-n', str(ranks), str(exe.resolve()), '-i', 'input.athinput']
        with (run/'run.log').open('w') as log:
            subprocess.run(cmd, cwd=run, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        result = validate(run, ranks, exact_zero=amplitude == 0)
        assert result['passed'] and result['cycle'] == 3
        final = blocks(run, ranks, 3)
        states[name] = [u for _, u in final]
        payloads[name] = physical_payloads(run,ranks,3)
        result['final_physical_payload_hashes'] = [hashlib.sha256(v).hexdigest() for v in payloads[name]]
        result['final_z4c_block_hashes'] = [hashlib.sha256(u.tobytes()).hexdigest()
                                           for _, u in final]
        if profile == 'compact':
            result['initial_peak'] = seed_check(run, ranks, amplitude)
            result['initial_boundary_support'] = boundary_support_check(run, ranks)
            response = max(float(abs(u[[0, 7, 18, 19, 20, 21]]).max()) for _, u in final)
            assert (response > 0) == (amplitude != 0)
            result['response_max'] = response
        results[name] = result
        print(name, 'passed', flush=True)
    for left, right in [('compact_r1', 'compact_r2'),
                        ('gaussian_default', 'gaussian_explicit')]:
        assert all(np.array_equal(x, y) for x, y in zip(states[left], states[right]))
        assert payloads[left] == payloads[right], 'MHD/B/Z4c payload differs'
    if a.reference_exe:
        assert all(np.array_equal(x, y) for x, y in
                   zip(states['gaussian_default'], states['gaussian_reference']))
        assert payloads['gaussian_default'] == payloads['gaussian_reference'], 'Old/new MHD/B/Z4c payload differs'
    ratio = results['compact_double']['response_max']/results['compact_r2']['response_max']
    assert abs(ratio-2) < 5e-4
    results['response_amplitude_ratio'] = ratio
    results['executable_sha256'] = hashlib.sha256(a.exe.read_bytes()).hexdigest()
    if a.reference_exe:
        results['reference_executable_sha256'] = hashlib.sha256(a.reference_exe.read_bytes()).hexdigest()
    reject = a.output/'reject_profile'
    reject.mkdir()
    (reject/'input.athinput').write_text(base.replace('<problem>', '<problem>\n'
        'outer_sponge_test_theta_pulse_profile = unknown'))
    r = subprocess.run([a.launcher, '-n', '1', str(a.exe.resolve()), '-i', 'input.athinput'],
                       cwd=reject, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    (reject/'run.log').write_text(r.stdout)
    assert r.returncode != 0 and 'must be gaussian or compact' in r.stdout
    results['invalid_profile_rejected'] = True
    (a.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Compact support, zero preservation, response, MPI and Gaussian parity passed.')


if __name__ == '__main__':
    main()
