#!/usr/bin/env python3
"""Offline Bondi-news estimate from a fixed-topology native scri metric archive.

Requires a fresh initial round cut. Uses geometric conformal Ricci projection,
transported inertial angles and Bondi time, and reports frame/discretization
residuals. A single N12 result is not a resolution-converged waveform.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scri_geometry import spacetime_derivatives, derivative_weights, ricci
from scri_frame import FrameTransport, generator_geometry
from scri_angular import scalar_basis, spin_basis, weighted_projection


def stencil(times, index, width=7):
    begin = min(max(index-width//2, 0), len(times)-width)
    return slice(begin, begin+width), index-begin


def interpolate_time(nodes, values, target, width=6):
    """Local polynomial time interpolation on one fixed inertial generator."""
    if np.any(np.diff(nodes) <= 0):
        raise ValueError('Bondi time is not increasing along an inertial generator')
    first = np.clip(np.searchsorted(nodes, target)-width//2, 0, len(nodes)-width)
    index = first[:, None]+np.arange(width)
    x = nodes[index]; numerator = target[:, None, None]-x[:, None, :]
    denominator = x[:, :, None]-x[:, None, :]
    diagonal = np.arange(width)
    numerator = np.broadcast_to(numerator, denominator.shape).copy()
    numerator[:, diagonal, diagonal] = denominator[:, diagonal, diagonal] = 1
    weights = np.prod(numerator/denominator, axis=-1)
    return np.sum(weights*values[index], axis=-1)


def angular_grid(lmax):
    mu, weights = np.polynomial.legendre.leggauss(2*lmax+8)
    phi = np.arange(4*lmax+16)*2*np.pi/(4*lmax+16)
    weights = np.repeat(weights, len(phi))*2*np.pi/len(phi)
    mu = np.repeat(mu, len(phi)); phi = np.tile(phi, 2*lmax+8)
    return np.column_stack((mu, np.sqrt(1-mu*mu)*np.cos(phi), np.sqrt(1-mu*mu)*np.sin(phi))), weights


def analyze(cache, output, lmax=8, stride=1, mass=1/3):
    if lmax < 2 or stride < 1 or not np.isfinite(mass) or mass <= 0:
        raise ValueError('need lmax >= 2, stride >= 1 and positive finite mass')
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((cache/'records.json').read_text())
    indices = np.arange(0, len(records), stride)
    times = np.array([records[i]['time'] for i in indices])
    if len(times) < 10 or abs(times[0]) > 1e-14:
        raise ValueError('need at least ten records, starting at native time zero')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('native record times must be finite and strictly increasing')
    grid = np.load(cache/'grid.npz'); x, weights = grid['x'], grid['weights']
    archive = np.load(cache/'metric.npy', mmap_mode='r')
    first_metric = archive[0, ..., 0]
    tangent = np.eye(3)-x[:, :, None]*x[:, None, :]
    defect = tangent@(first_metric[:, 1:, 1:]-np.eye(3))@tangent
    initial_round_error = float(np.max(np.linalg.norm(defect, axis=(-2, -1))))
    if initial_round_error > 1e-9:
        raise ValueError('initial cut is not round; initial uniformization is required')
    transport = FrameTransport(x, weights, lmax)
    flows = np.empty((len(times), len(x), 3)); sources = np.empty((len(times), len(x)))
    generator_times = np.empty_like(sources)
    null_error = []; shear_error = []; regularity_error = []; scale_rate = []
    def geometry_at(q):
        window, center = stencil(times, q)
        metric, dg, ddg = spacetime_derivatives(archive[indices[window]], times[window], center)
        curvature, connection, inverse = ricci(metric, dg, ddg)
        if not np.isfinite(curvature).all():
            raise ValueError('nonfinite conformal curvature')
        return metric, curvature, connection, inverse
    for q in range(len(times)):
        metric, curvature, connection, inverse = geometry_at(q)
        flows[q], sources[q], generator_times[q], omega_hessian = generator_geometry(x, inverse, connection)
        null_error.append(float(np.max(abs(np.einsum('qi,qi->q', flows[q]*generator_times[q, :, None], -x)))))
        # Trace-free tangential Hessian of Omega measures the shear-free condition.
        gamma = metric[:, 1:, 1:]; mu = x[:, 0]; sn = np.sqrt(1-mu*mu)
        e = (x*mu[:, None]-np.array([1., 0, 0]))/sn[:, None]
        f = np.column_stack((np.zeros(len(x)), -x[:, 2], x[:, 1]))/sn[:, None]
        ee = np.einsum('qi,qij,qj->q', e, gamma, e); ef = np.einsum('qi,qij,qj->q', e, gamma, f)
        ff = np.einsum('qi,qij,qj->q', f, gamma, f)
        dyad = (e/np.sqrt(ee)[:, None]-1j*(f-(ef/ee)[:, None]*e)/np.sqrt(ff-ef*ef/ee)[:, None])/np.sqrt(2)
        shear_error.append(float(np.max(abs(np.einsum('qi,qj,qij->q', dyad, dyad, omega_hessian[:, 1:, 1:])))))
        box = 4*sources[q]*generator_times[q]
        trace_free_hessian = omega_hessian-metric*box[:, None, None]/4
        regularity_error.append(float(np.max(abs(trace_free_hessian))))
        # In a regular vacuum conformal completion, the angular expansion and
        # four-dimensional trace give the same preferred-frame scale rate.
        # Their difference predicts round-cut drift independently of transport.
        angular_trace = 2*np.einsum('qi,qj,qij->q', dyad, dyad.conj(), omega_hessian[:, 1:, 1:]).real
        rate = angular_trace/generator_times[q]-2*sources[q]
        scale_rate.append(float(np.sum(weights*rate)/np.sum(weights)))
    states = transport.evolve(times, flows, sources, generator_times)
    news_coefficients = []; clock_coefficients = []; diagnostics = []
    valid = range(3, len(times)-3)
    for q in valid:
        window, center = stencil(times, q)
        first = np.einsum('t,tkf->kf', derivative_weights(times[window], times[q], 1), states[window])
        second = np.einsum('t,tkf->kf', derivative_weights(times[window], times[q], 2), states[window])
        metric, curvature, connection, _ = geometry_at(q)
        observed = transport.observe(states[q], metric, curvature, connection, flows[q], sources[q],
                                     generator_times[q], second, first)
        spin, labels = spin_basis(observed['direction'], lmax)
        scalar, _, _ = scalar_basis(observed['direction'], lmax)
        news_fit = weighted_projection(spin, observed['angular_weights'])@observed['news']
        clock_fit = weighted_projection(scalar, observed['angular_weights'])@observed['bondi_time']
        residual = spin@news_fit-observed['news']
        d = observed['diagnostics']
        d.update(news_fit_rms=float(np.sqrt(np.sum(observed['angular_weights']*abs(residual)**2)/np.sum(observed['angular_weights']))),
                 news_native_rms=float(np.sqrt(np.sum(observed['angular_weights']*abs(observed['news'])**2)/np.sum(observed['angular_weights']))),
                 null_max=null_error[q], shear_max=shear_error[q],
                 conformal_hessian_tf_max=regularity_error[q],
                 round_scale_rate_mean=scale_rate[q])
        diagnostics.append(d); news_coefficients.append(news_fit); clock_coefficients.append(clock_fit)
        if q % 100 == 0:
            print('Analyzed', q, 'of', len(times), 'native t/M0', times[q]/mass, flush=True)
    points, quadrature = angular_grid(lmax)
    spin, labels = spin_basis(points, lmax); scalar, _, _ = scalar_basis(points, lmax)
    news_samples = np.array(news_coefficients)@spin.T
    clock_samples = np.array(clock_coefficients)@scalar.T
    if np.any(np.diff(clock_samples, axis=0) <= 0):
        raise ValueError('fitted Bondi clock is not monotone')
    begin = float(np.max(clock_samples[0])); end = float(np.min(clock_samples[-1]))
    if end <= begin:
        raise ValueError('no common Bondi-time interval across inertial generators')
    count = len(news_samples)
    common_time = np.linspace(begin, end, count)
    common_news = np.empty((count, len(points)), dtype=complex)
    for angle in range(len(points)):
        common_news[:, angle] = interpolate_time(clock_samples[:, angle], news_samples[:, angle], common_time)
    modes = common_news@(quadrature[:, None]*spin.conj())
    np.savez(output/'news.npz', bondi_time=common_time, bondi_time_over_M0=common_time/mass,
             news_modes=modes, ell_m=np.array(labels), native_times=times[3:-3],
             native_cut_news_coefficients=np.array(news_coefficients), bondi_clock_coefficients=np.array(clock_coefficients),
             frame_coefficients=states, frame_times=times)
    summary = dict(kind='preliminary Bondi-news estimate', normalization='N=d[r(h_plus-i*h_cross)]/du_B',
                   polarization_axis='Cartesian x', initial_frame='round initial cut; u_B=0; inertial angles=native angles',
                   mass_M0=mass, angular_lmax=lmax, temporal_stride=stride, samples=len(times),
                   native_time_range_M0=[float(times[0]/mass),float(times[-1]/mass)],
                   bondi_time_range_M0=[begin/mass,end/mass], initial_round_error=initial_round_error,
                   diagnostics_max={key:max(d[key] for d in diagnostics) for key in diagnostics[0]},
                   diagnostics_max_abs={key:max(abs(d[key]) for d in diagnostics) for key in diagnostics[0]},
                   resolution_convergence_established=False,
                   peak_mode_abs={str(pair):float(np.max(abs(modes[:,k]))) for k,pair in enumerate(labels)})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lmax',type=int,default=8)
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--mass',type=float,default=1/3,help='M0 in native code units')
    args=parser.parse_args()
    analyze(args.cache.resolve(),args.output.resolve(),args.lmax,args.stride,args.mass)
