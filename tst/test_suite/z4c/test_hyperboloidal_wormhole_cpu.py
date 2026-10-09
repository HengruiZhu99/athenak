"""Native layer wormhole initialization and short live steps, not stability tests."""
from pathlib import Path

import numpy as np
import pytest

from .test_hyperboloidal_layer_cpu import run_layer
from .test_hyperboloidal_native_cpu import fields


CONTROLS = '''<z4c>
hyperboloidal_physical_trace_lapse=true
hyperboloidal_preferred_source=false
hyperboloidal_symmetric_ghosts=true
<problem>
lapse_pulse=0
shift_pulse=0
'''


def initial_geometry(shape, mass):
    n = shape[-1]
    xyz = -1.05 + (np.arange(n) - 2.5) * (2.1 / 24)
    z, y, x = np.meshgrid(xyz, xyz, xyz, indexing='ij')
    r = np.sqrt(x*x + y*y + z*z)[None]
    direction = [x[None]/r, y[None]/r, z[None]/r]
    w, wp = np.zeros_like(r), np.zeros_like(r)
    collar = (r > .35) & (r < .75)
    s = (r[collar] - .35) / .4
    t = (.75 - r[collar]) / .4
    g = -1/s + 1/t
    small_exp = np.exp(-np.abs(g))
    w[collar] = np.where(g <= 0, small_exp/(1+small_exp), 1/(1+small_exp))
    wp[collar] = small_exp/(1+small_exp)**2 * (1/s**2 + 1/t**2) / .4
    w[r >= .75] = 1
    omega = 1 - w + w*(1-r*r)/2
    op = wp*((1-r*r)/2 - 1) - w*r
    ell = omega - r*op
    b = r*w
    alpha_ref = np.hypot(omega, b)
    chi_ref = (alpha_ref/ell)**(2/3)
    gr = chi_ref*(ell/alpha_ref)**2
    gt = chi_ref
    m = mass*omega/(2*r)
    psi = 1 + m
    static_lapse = (1-m)/psi
    alpha = alpha_ref*((1-w)/psi**2 + w*static_lapse)
    chi = chi_ref/psi**4
    beta = -b*alpha_ref/ell * static_lapse/psi**2
    kr, kt, shear = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
    height = w > 0
    eta = r*omega*wp/ell
    kr[height] = -(w[height] + eta[height] +
                   2*w[height]*m[height]/(1-m[height]**2))/psi[height]**2
    kt[height] = -w[height]*static_lapse[height]/psi[height]**2
    shear[height] = -(r[height]*wp[height]/ell[height] +
                      w[height]*mass*(2-m[height]) /
                      (r[height]*(1-m[height]**2)))/psi[height]**2
    return dict(r=r, direction=direction, w=w, omega=omega, ell=ell, psi=psi,
                m=m, alpha=alpha, chi=chi, beta=beta, trace=kr+2*kt, shear=shear,
                gr=gr, gt=gt)


@pytest.mark.parametrize('mass', [.1, .2, .5])
def test_layer_wormhole_initial_geometry_and_live_steps(tmp_path, mass):
    reference = tmp_path / 'reference'
    live = tmp_path / 'wormhole'
    run_layer(reference, CONTROLS + '<time>\nnlim=0\n')
    run_layer(live, CONTROLS + f'<problem>\nmass={mass}\n')
    initial = fields(live, 'z4c', True)
    final = fields(live, 'z4c')
    adm = fields(live, 'adm', True)
    background = fields(reference, 'z4c', True)
    mask = initial['z4c_active'].astype(bool)
    model = initial_geometry(mask.shape, mass)
    assert np.min(initial['z4c_alpha'][mask]) > 0
    assert np.min(initial['z4c_chi'][mask]) > 0
    assert np.isnan(adm['adm_alpha'][~mask]).all()
    for key, expected in [('alpha', model['alpha']), ('chi', model['chi']),
                          ('Khat', model['trace'])]:
        np.testing.assert_allclose(initial['z4c_' + key][mask], expected[mask],
                                   rtol=3e-7, atol=1e-9)
    np.testing.assert_array_equal(initial['z4c_Theta'][mask], 0)
    for i, name in enumerate(('x', 'y', 'z')):
        np.testing.assert_allclose(initial['z4c_beta' + name][mask],
                                   (model['beta']*model['direction'][i])[mask],
                                   rtol=3e-7, atol=1e-10)
        np.testing.assert_array_equal(initial['z4c_Gam' + name][mask],
                                      background['z4c_Gam' + name][mask])
    radial_metric = np.zeros_like(model['r'])
    radial_curvature = np.zeros_like(model['r'])
    trace_metric = sum(adm['adm_g' + name] for name in ('xx', 'yy', 'zz'))
    trace_curvature = sum(adm['adm_K' + name] for name in ('xx', 'yy', 'zz'))
    for i, j, name in [(0, 0, 'xx'), (0, 1, 'xy'), (0, 2, 'xz'),
                       (1, 1, 'yy'), (1, 2, 'yz'), (2, 2, 'zz')]:
        nn = model['direction'][i]*model['direction'][j]
        delta = int(i == j)
        expected_a = model['shear']*(2*model['gr']*nn - model['gt']*(delta-nn))/3
        np.testing.assert_allclose(initial['z4c_A' + name][mask], expected_a[mask],
                                   rtol=6e-7, atol=1e-8)
        np.testing.assert_array_equal(initial['z4c_g' + name][mask],
                                      background['z4c_g' + name][mask])
        multiplicity = 1 if i == j else 2
        radial_metric += multiplicity*nn*adm['adm_g' + name]
        radial_curvature += multiplicity*nn*adm['adm_K' + name]
    core = mask & (model['r'] < .35)
    assert np.count_nonzero(core) > 100
    np.testing.assert_array_equal(initial['z4c_Khat'][core], 0)
    np.testing.assert_array_equal(initial['z4c_Axx'][core], 0)
    np.testing.assert_allclose(initial['z4c_alpha'][core], (1/model['psi']**2)[core],
                               rtol=2e-7)
    # Recover the invariant Schwarzschild mass from the dumped physical geometry.
    # The analytic areal-radius derivative follows directly from rho=R+M+M^2/(4R).
    tangent_metric = (trace_metric-radial_metric)/2
    tangent_k = (trace_curvature-radial_curvature)/(2*tangent_metric)
    areal = model['r']*np.sqrt(tangent_metric)
    areal_d = (1-model['m']**2)*model['ell']/model['omega']**2
    invariant = areal*(1+areal**2*tangent_k**2-areal_d**2/radial_metric)/2
    probe = mask & (model['r'] > .15) & (model['r'] < .7)
    np.testing.assert_allclose(invariant[probe], mass, rtol=2e-5, atol=2e-6)
    history = np.atleast_2d(np.loadtxt(live / 'hyp.z4c.user.hst'))
    assert np.max(np.abs(history[0, 2:6])) < 2e-10
    for key in final:
        assert np.isfinite(final[key][mask]).all(), key
    assert np.min(final['z4c_alpha'][mask]) > 0
    assert np.min(final['z4c_chi'][mask]) > 0
    # The actual geometry and gauge evolve; the initializer is not a BH RHS subtraction.
    assert np.max(np.abs(final['z4c_Khat'][core]-initial['z4c_Khat'][core])) > 1e-8
    assert np.max(np.abs(final['z4c_alpha'][mask]-initial['z4c_alpha'][mask])) > 1e-8


def test_layer_wormhole_short_restart(tmp_path):
    controls = CONTROLS + '<problem>\nmass=.2\n'
    live, restart, whole = (tmp_path / name for name in ('live', 'restart', 'whole'))
    run_layer(live, controls + '<time>\nnlim=1\n')
    checkpoint = sorted((live / 'rst').glob('*.rst'))[-1]
    run_layer(restart, controls + '<time>\nnlim=3\n', checkpoint=checkpoint)
    run_layer(whole, controls + '<time>\nnlim=3\n')
    for kind in ('z4c', 'adm', 'con'):
        resumed, uninterrupted = fields(restart, kind), fields(whole, kind)
        mask = resumed['z4c_active'].astype(bool)
        for name in resumed:
            np.testing.assert_allclose(resumed[name][mask], uninterrupted[name][mask],
                                       rtol=2e-6, atol=2e-7, err_msg=name)


def test_layer_wormhole_requires_even_grid(tmp_path):
    extra = CONTROLS + '''<problem>
mass=.2
<mesh>
nx1=25
nx2=25
nx3=25
<meshblock>
nx1=25
nx2=25
nx3=25
'''
    result = run_layer(tmp_path, extra, success=False)
    assert 'even cell counts excluding r=0' in result.stdout + result.stderr
