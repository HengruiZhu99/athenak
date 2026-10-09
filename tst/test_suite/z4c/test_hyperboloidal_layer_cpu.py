"""Native layer fixed point, live gauges, rejection and restart tests."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from .test_hyperboloidal_native_cpu import fields
from .overhaul_utils import ROOT

EXE = Path(os.environ.get('ATHENA_OVERHAUL_EXE', './athena')).resolve()


def run_layer(directory, extra='', checkpoint=None, success=True):
    directory.mkdir(parents=True, exist_ok=True)
    text = (ROOT / 'inputs/z4c/hyperboloidal_layer.athinput').read_text()
    text += '\n' + extra
    input_file = directory / 'layer.athinput'
    input_file.write_text(text)
    command = [str(EXE), '-i', str(input_file)]
    if checkpoint:
        command += ['-r', str(checkpoint)]
    result = subprocess.run(
        command,
        cwd=directory,
        text=True,
        capture_output=True,
        timeout=120)
    (directory / 'run.log').write_text(result.stdout + result.stderr)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def test_native_layer_reference(tmp_path):
    run_layer(tmp_path)
    initial, final = fields(tmp_path, 'z4c', True), fields(tmp_path, 'z4c')
    mask = initial['z4c_active'].astype(bool)
    for key in initial:
        np.testing.assert_allclose(
            final[key][mask],
            initial[key][mask],
            rtol=1e-6,
            atol=1e-9)
    # The transition really has nontrivial curvature and connection.
    assert np.max(np.abs(initial['z4c_Axx'][mask])) > 0.1
    assert np.max(np.abs(initial['z4c_Gamx'][mask])) > 0.1
    history = np.atleast_2d(np.loadtxt(tmp_path / 'hyp.z4c.user.hst'))
    assert np.max(history[:, 2:6]) < 2e-10
    a = fields(tmp_path, 'adm')
    n = mask.shape[-1]
    xyz = -1.05 + (np.arange(n) - 2.5) * 2.1 / 24
    z, y, x = np.meshgrid(xyz, xyz, xyz, indexing='ij')
    r = np.sqrt(x * x + y * y + z * z)[None]
    w = np.zeros_like(r)
    inside = (r > .35) & (r < .75)
    s = (r[inside] - .35) / .4
    w[inside] = 1 / (1 + np.exp(1 / s - 1 / (1 - s)))
    w[r >= .75] = 1
    omega = 1 - w + w * (1 - r * r) / 2
    np.testing.assert_allclose(
        a['adm_alpha'][mask],
        final['z4c_alpha'][mask] /
        omega[mask],
        rtol=3e-7)
    assert np.isnan(a['adm_alpha'][~mask]).all()


@pytest.mark.parametrize('extra,diagnostic', [
    ('<z4c>\nhyperboloidal_layer_r0=0', '0<r0<r1<S'),
    ('<z4c>\nhyperboloidal_layer_r1=1', '0<r0<r1<S'),
    ('<z4c>\nhyperboloidal_gauge_q0=0', '0<q0<1'),
    ('<z4c>\nhyperboloidal_gauge_q0=1', '0<q0<1'),
    ('<z4c>\nhyperboloidal_gauge_r0=0.2', 'beyond the Cauchy interior'),
    ('<z4c>\nhyperboloidal_layer_shift_outer=-1', 'restoring rates'),
    ('<problem>\nmass=0.5', 'incompatible with the layer foliation'),
    ('<problem>\npulse_width=0', 'gauge pulse'),
])
def test_layer_rejections(tmp_path, extra, diagnostic):
    result = run_layer(tmp_path, extra, success=False)
    assert diagnostic in result.stdout + result.stderr


@pytest.mark.parametrize('angular', ['true', 'false'])
def test_live_layer_gauge_and_restart(tmp_path, angular):
    extra = f'<problem>\nlapse_pulse=0.02\nshift_pulse=0.005\npulse_angular={angular}\n'
    run_layer(tmp_path / 'live', extra)
    initial, final = fields(tmp_path / 'live', 'z4c',
                            True), fields(tmp_path / 'live', 'z4c')
    mask = initial['z4c_active'].astype(bool)
    assert np.max(np.abs(final['z4c_alpha'][mask] - initial['z4c_alpha'][mask])) > 1e-8
    assert np.max(np.abs(final['z4c_betax'][mask] - initial['z4c_betax'][mask])) > 1e-8
    for key in final:
        assert np.isfinite(final[key][mask]).all(), key
    checkpoint = sorted((tmp_path / 'live' / 'rst').glob('*.rst'))[-1]
    run_layer(tmp_path / 'restart', extra + '<time>\nnlim=3\n', checkpoint=checkpoint)
    run_layer(tmp_path / 'whole', extra + '<time>\nnlim=3\n')
    restarted, whole = fields(tmp_path /
                              'restart', 'z4c'), fields(tmp_path /
                                                        'whole', 'z4c')
    for key in restarted:
        np.testing.assert_allclose(
            restarted[key][mask],
            whole[key][mask],
            rtol=2e-6,
            atol=2e-7)
