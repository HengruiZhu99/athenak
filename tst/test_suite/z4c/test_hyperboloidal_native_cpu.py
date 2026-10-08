"""Native single-block conformal tasks, physical ADM mapping and restart."""
import importlib.util
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from .overhaul_utils import ROOT

spec = importlib.util.spec_from_file_location(
    'hyp_reader', ROOT / 'vis/python/bin_convert.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
profile_spec = importlib.util.spec_from_file_location(
    'hyp_radial', ROOT / 'tst/hyperboloidal/analyze_native_constraints.py')
radial = importlib.util.module_from_spec(profile_spec)
profile_spec.loader.exec_module(radial)
EXE = Path(os.environ.get('ATHENA_OVERHAUL_EXE', './athena')).resolve()
PATCH_EXE = Path(os.environ.get(
    'ATHENA_HYP_PATCH_EXE', './hyperboloidal_cartesian_tests')).resolve()


def run(directory, *overrides, checkpoint=None, success=True):
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(EXE)]
    if checkpoint is None:
        command += ['-i', str(ROOT / 'inputs/z4c/hyperboloidal.athinput')]
    else:
        command += ['-r', str(checkpoint)]
    result = subprocess.run(command + list(overrides), cwd=directory,
                            text=True, capture_output=True, timeout=120)
    (directory / ('restart.log' if checkpoint else 'run.log')).write_text(
        result.stdout + result.stderr)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def fields(directory, kind, first=False):
    paths = sorted((directory / 'bin').glob(f'*.{kind}.*.bin'))
    assert paths
    data = reader.read_binary(str(paths[0] if first else paths[-1]))
    return {k: np.asarray(v) for k, v in data['mb_data'].items()}


def check_adm(directory, first=False):
    q, a = fields(directory, 'z4c', first), fields(directory, 'adm', first)
    mask = q['z4c_active'].astype(bool)
    np.testing.assert_array_equal(mask, a['z4c_active'])
    n = mask.shape[-1]
    assert n == 30  # includes all three halo cells
    xyz = -1.05 + (np.arange(n) - 2.5) * (2.1 / 24)
    z, y, x = np.meshgrid(xyz, xyz, xyz, indexing='ij')
    omega = ((1 - x*x - y*y - z*z) / 2)[None]
    np.testing.assert_array_equal(mask, omega > 0)
    assert np.isfinite(q['z4c_alpha']).all()
    for name, value in a.items():
        if name == 'z4c_active':
            continue
        assert np.isfinite(value[mask]).all(), name
        assert np.isnan(value[~mask]).all(), name
    np.testing.assert_allclose(a['adm_alpha'][mask], q['z4c_alpha'][mask] / omega[mask],
                               rtol=2e-7)
    psi = 1 / (omega[mask]**2 * q['z4c_chi'][mask])
    np.testing.assert_allclose(a['adm_psi4'][mask], psi, rtol=3e-7)
    for component in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
        metric = psi * q['z4c_g' + component][mask]
        np.testing.assert_allclose(a['adm_g' + component][mask], metric,
                                   rtol=3e-7, atol=1e-10)
        first_term = psi * omega[mask] * q['z4c_A' + component][mask]
        second_term = metric * (q['z4c_Khat'][mask] + 2*q['z4c_Theta'][mask]) / 3
        curvature = first_term + second_term
        # Binary dumps round each input independently to float32; cancellation
        # in K requires an absolute error budget based on both terms.
        error = np.abs(a['adm_K' + component][mask] - curvature)
        assert np.all(error < 5e-7 * (np.abs(first_term) + np.abs(second_term)) + 1e-9)
    c = fields(directory, 'con', first)
    np.testing.assert_array_equal(c['z4c_active'].astype(bool), mask)
    for name, value in c.items():
        assert np.isfinite(value).all(), name
        if name != 'z4c_active':
            assert (value[~mask] == 0).all(), name


def check_radial_budget(directory, history_row):
    path = sorted((directory / 'bin').glob('*.con.*.bin'))[-1]
    profile = radial.analyze(path, [0, .25, .5, .75, .9, 1])
    assert profile['active_cells'] == history_row[-1]
    assert sum(row['cells'] for row in profile['radial_bins']) == profile['active_cells']
    for column, name in [(2, 'H'), (3, 'M'), (4, 'Z')]:
        np.testing.assert_allclose(profile['global'][name]['rms'], history_row[column],
                                   rtol=3e-7, atol=1e-12)
        fraction = sum(row[name]['squared_norm_fraction']
                       for row in profile['radial_bins'])
        np.testing.assert_allclose(fraction, 1 if profile['global'][name]['rms'] else 0,
                                   rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='cover all active nodes'):
        radial.analyze(path, [.5, 1])


@pytest.mark.parametrize('degree', [2, 3, 4])
@pytest.mark.parametrize('mass,pulse', [(0, 0), (0, 1e-4), (0.5, 0)])
def test_native_tasks_and_adm(tmp_path, mass, pulse, degree):
    run(tmp_path, 'time/nlim=3', f'problem/mass={mass}', f'problem/lapse_pulse={pulse}',
        f'z4c/hyperboloidal_ghost_degree={degree}')
    check_adm(tmp_path, first=True)
    check_adm(tmp_path)
    history = np.loadtxt(next(tmp_path.glob('*.hst')))
    final = history[-1]
    check_radial_budget(tmp_path, final)
    amplitude, profile = (mass, 'trumpet') if mass else (pulse, 'smooth')
    result = subprocess.run([str(PATCH_EXE), '24', f'{final[0]:.17g}',
                             str(amplitude), profile, '.04', str(degree)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(re.findall(r'(\w+)=([-+\deE.]+)', result.stdout.splitlines()[-1]))
    for column, name in [(2, 'H'), (3, 'M'), (4, 'Z'), (5, 'Theta')]:
        np.testing.assert_allclose(final[column], float(values[name]),
                                   rtol=2e-5, atol=1e-11)
    if mass == pulse == 0:
        assert np.max(np.abs(history[:, 2:8])) < 1e-11


@pytest.mark.parametrize('degree', [2, 3])
def test_native_trumpet_restart(tmp_path, degree):
    full, split = tmp_path / 'full', tmp_path / 'split'
    mass_option = f'z4c/hyperboloidal_mass_diagnostics={str(degree == 2).lower()}'
    gauge_option = f'z4c/hyperboloidal_shift_driver={1 if degree == 2 else 0.1}'
    run(full, 'problem/mass=0.5', 'time/nlim=3',
        f'z4c/hyperboloidal_ghost_degree={degree}', mass_option, gauge_option)
    run(split, 'problem/mass=0.5', 'time/nlim=1',
        f'z4c/hyperboloidal_ghost_degree={degree}', mass_option, gauge_option)
    checkpoint = sorted((split / 'rst').glob('*.rst'))[-1]
    run(split, 'time/nlim=3', checkpoint=checkpoint)
    for kind in ('z4c', 'adm', 'con'):
        a, b = fields(full, kind), fields(split, kind)
        assert a.keys() == b.keys()
        for name in a:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-11, atol=1e-12,
                                       equal_nan=True, err_msg=name)
    check_adm(split)
    np.testing.assert_allclose(np.loadtxt(next(full.glob('*.hst')))[-1],
                               np.loadtxt(next(split.glob('*.hst')))[-1],
                               rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('option', ['mesh/nx1=48', 'mesh/nghost=4',
                                    'time/integrator=rk4', 'z4c/floor_chi=true',
                                    'z4c/nrad_wave_extraction=1'])
def test_unsupported_native_configuration(tmp_path, option):
    result = run(tmp_path, option, success=False)
    assert 'hyperboloidal prototype requires' in result.stdout + result.stderr


@pytest.mark.parametrize('variable', ['z4c_Kretschmann', 'weyl_rpsi4'])
def test_unsupported_native_diagnostics(tmp_path, variable):
    result = run(tmp_path, 'output1/variable=' + variable, success=False)
    assert 'not conformal-mask aware' in result.stdout + result.stderr


def test_scalar_output_carries_mask(tmp_path):
    run(tmp_path, 'output1/variable=z4c_alpha', 'time/nlim=0')
    data = fields(tmp_path, 'z4c')
    assert set(data) == {'z4c_alpha', 'z4c_active'}


def test_nodes_exactly_on_scri_are_inactive(tmp_path):
    # Dyadic spacing makes r=1 exact, avoiding a tolerance-based classification.
    options = []
    for d in (1, 2, 3):
        options += [f'mesh/nx{d}=48', f'meshblock/nx{d}=48',
                    f'mesh/x{d}min=-1.53125', f'mesh/x{d}max=1.46875']
    run(tmp_path, *options)
    q, a = fields(tmp_path, 'z4c'), fields(tmp_path, 'adm')
    center = 27
    for d in range(3):
        for boundary in (11, 43):
            index = [0, center, center, center]
            index[d+1] = boundary
            assert q['z4c_active'][tuple(index)] == 0
            assert np.isnan(a['adm_alpha'][tuple(index)])
    history = np.loadtxt(next(tmp_path.glob('*.hst')))
    assert np.isfinite(history).all()
    assert np.max(np.abs(history[:, 2:8])) < 1e-10
    check_radial_budget(tmp_path, history[-1])


@pytest.mark.parametrize('degree', [1, 6])
def test_invalid_spherical_ghost_degree(tmp_path, degree):
    result = run(tmp_path, f'z4c/hyperboloidal_ghost_degree={degree}', success=False)
    assert 'invalid spherical ghost policy' in result.stdout + result.stderr


def test_fifth_degree_requires_interior_donors(tmp_path):
    result = run(tmp_path / 'coarse', 'z4c/hyperboloidal_ghost_degree=5', success=False)
    assert 'no interior normal-ray rectangles' in result.stdout + result.stderr
    fine = tmp_path / 'fine'
    options = [f'{block}/nx{d}=36' for block in ('mesh', 'meshblock') for d in (1, 2, 3)]
    run(fine, *options, 'z4c/hyperboloidal_ghost_degree=5', 'problem/lapse_pulse=0.0001',
        'time/nlim=3')
    final = np.loadtxt(next(fine.glob('*.hst')))[-1]
    result = subprocess.run([str(PATCH_EXE), '36', f'{final[0]:.17g}', '0.0001',
                             'smooth', '.04', '5'], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(re.findall(r'(\w+)=([-+\deE.]+)', result.stdout.splitlines()[-1]))
    for column, name in [(2, 'H'), (3, 'M'), (4, 'Z'), (5, 'Theta')]:
        np.testing.assert_allclose(final[column], float(values[name]),
                                   rtol=2e-5, atol=1e-11)


@pytest.mark.parametrize('mass', [0, 0.5])
def test_native_hawking_history(tmp_path, mass):
    enabled, disabled = tmp_path / 'enabled', tmp_path / 'disabled'
    run(enabled, f'problem/mass={mass}', 'z4c/hyperboloidal_mass_diagnostics=true')
    run(disabled, f'problem/mass={mass}')
    history = np.loadtxt(next(enabled.glob('*.hst')))
    assert history.shape[1] == 21
    assert np.isfinite(history).all()
    # The independent exact-data tests quantify the coarsest interpolation bias.
    np.testing.assert_allclose(history[:, [15, 17, 19]], mass, rtol=0,
                               atol=0.005 if mass else 1e-10)
    assert (np.diff(history[:, [16, 18, 20]], axis=1) > 0).all()
    for kind in ('z4c', 'adm', 'con'):
        a, b = fields(enabled, kind), fields(disabled, kind)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize('nmu', [3, 129])
def test_invalid_hawking_quadrature(tmp_path, nmu):
    result = run(tmp_path, f'z4c/hyperboloidal_mass_nmu={nmu}', success=False)
    assert 'invalid Hawking quadrature' in result.stdout + result.stderr


@pytest.mark.parametrize('parameter', ['slicing', 'shift_driver', 'lapse_damping',
                                       'shift_damping'])
@pytest.mark.parametrize('value', [-1, 'nan'])
def test_invalid_reference_gauge(tmp_path, parameter, value):
    result = run(tmp_path, f'z4c/hyperboloidal_{parameter}={value}', success=False)
    assert ('Reference gauge coefficients must be finite >=0'
            in result.stdout + result.stderr)


def test_changed_gauge_preserves_minkowski(tmp_path):
    run(tmp_path, 'time/nlim=3', 'z4c/hyperboloidal_slicing=0.5',
        'z4c/hyperboloidal_shift_driver=1', 'z4c/hyperboloidal_lapse_damping=3',
        'z4c/hyperboloidal_shift_damping=0')
    a, b = fields(tmp_path, 'z4c', first=True), fields(tmp_path, 'z4c')
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)
    history = np.loadtxt(next(tmp_path.glob('*.hst')))
    assert np.max(np.abs(history[:, 2:8])) < 1e-11


def test_shift_driver_changes_live_puncture(tmp_path):
    baseline, changed = tmp_path / 'baseline', tmp_path / 'changed'
    run(baseline, 'problem/mass=0.5', 'time/nlim=3')
    run(changed, 'problem/mass=0.5', 'time/nlim=3', 'z4c/hyperboloidal_shift_driver=1')
    a, b = fields(baseline, 'z4c'), fields(changed, 'z4c')
    mask = a['z4c_active'].astype(bool)
    assert np.max(np.abs(a['z4c_betax'][mask] - b['z4c_betax'][mask])) > 1e-10
    check_adm(changed)
