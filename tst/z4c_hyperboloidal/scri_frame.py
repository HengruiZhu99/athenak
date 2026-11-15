#!/usr/bin/env python3
"""Inertial extraction against a radiative metric in rotating, rescaled coordinates."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'vis/python'))
from scri_angular import scalar_basis, spin_basis
from scri_frame import FrameTransport, generator_geometry
from scri_geometry import ricci, derivative_weights
spec = importlib.util.spec_from_file_location('manufactured', Path(__file__).with_name('scri_geometry.py'))
manufactured = importlib.util.module_from_spec(spec); spec.loader.exec_module(manufactured)


def main():
    mu, weights = np.polynomial.legendre.leggauss(12); phi = np.arange(24)*2*np.pi/24
    mu = np.repeat(mu, 24); weights = np.repeat(weights, 24)*2*np.pi/24; phi = np.tile(phi, 12)
    x = np.column_stack((mu, np.sqrt(1-mu*mu)*np.cos(phi), np.sqrt(1-mu*mu)*np.sin(phi)))
    scalar, gradient, hessian = scalar_basis(x, 8)
    np.testing.assert_allclose(scalar.T@(weights[:, None]*scalar), np.eye(81), atol=4e-14)
    for ell in range(9):
        np.testing.assert_allclose(np.trace(hessian[:, ell*ell:(ell+1)**2], axis1=-2, axis2=-1),
                                   -ell*(ell+1)*scalar[:, ell*ell:(ell+1)**2], atol=5e-11)
    spin, labels = spin_basis(x, 8)
    np.testing.assert_allclose(spin.conj().T@(weights[:, None]*spin), np.eye(77), atol=3e-13)
    np.testing.assert_allclose(spin[:, labels.index((2, 0))], np.sqrt(15/(32*np.pi))*(1-mu*mu), atol=1e-14)
    np.testing.assert_allclose(spin[:, labels.index((2, 2))], np.sqrt(5/(64*np.pi))*(1+mu)**2*np.exp(2j*phi), atol=1e-14)
    print('PASS scalar/spin harmonics, independent low modes and angular Laplacian')
    times = np.linspace(0, .4, 41); rotation = .3
    for amplitude in (0, .02):
        transport = FrameTransport(x, weights, 4)
        transport.initial = transport.project@np.column_stack((-.13*x[:, 0]**2, x, np.zeros(len(x))))
        data = []; flows = []; sources = []; generator_times = []
        for t in times:
            g, dg, ddg, _, _, omega = manufactured.radiative_metric(t, x, amplitude=amplitude, gauge=True,
                                                                    rotation=rotation, return_omega=True)
            curvature, connection, inverse = ricci(g, dg, ddg)
            flow, source, generator_time, _ = generator_geometry(x, inverse, connection, omega.d, omega.dd)
            data.append((g, curvature, connection, omega))
            flows.append(flow); sources.append(source); generator_times.append(generator_time)
        flows, sources, generator_times = map(np.array, (flows, sources, generator_times))
        states = transport.evolve(times, flows, sources, generator_times)
        q = 20; t = times[q]
        dt = np.einsum('t,tkf->kf', derivative_weights(times[q-3:q+4], t, 1), states[q-3:q+4])
        ddt = np.einsum('t,tkf->kf', derivative_weights(times[q-3:q+4], t, 2), states[q-3:q+4])
        g, curvature, connection, omega = data[q]
        result = transport.observe(states[q], g, curvature, connection, flows[q], sources[q], generator_times[q],
                                   ddt, dt, omega.d)
        expected_direction = x.copy()
        expected_direction[:, 0] = np.cos(rotation*t)*x[:, 0]-np.sin(rotation*t)*x[:, 1]
        expected_direction[:, 1] = np.sin(rotation*t)*x[:, 0]+np.cos(rotation*t)*x[:, 1]
        expected_news = amplitude*1.7*np.cos(1.7*t)*(1-expected_direction[:, 0]**2)
        np.testing.assert_allclose(result['direction'], expected_direction, atol=2e-12)
        np.testing.assert_allclose(result['bondi_time'], t, atol=2e-12)
        np.testing.assert_allclose(result['news'], expected_news, atol=3e-10)
        for key in ('round_cut_max', 'label_norm_max', 'frame_source_fit_max', 'clock_transport_max'):
            assert result['diagnostics'][key] < 2e-10, result['diagnostics']
        print('PASS full frame transport and radiation', amplitude, float(np.max(abs(result['news']-expected_news))), result['diagnostics'])


if __name__ == '__main__':
    main()
