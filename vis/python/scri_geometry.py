"""Conformal spacetime geometry from native endpoint jets and time differences.

This module performs tensor reconstruction only. It does not identify native
coordinates or a native polarization dyad with an inertial radiation frame.
Array conventions: dg[..., derivative, a, b], ddg[..., d, e, a, b].
"""
import math
import numpy as np
from scri_reader import constant, multiply, power, close_dependent


def endpoint_jets(state, epoch):
    """Restore archived accepted endpoint jets, using the redundant saved rows.

The recorder's reader can independently recompute both saved arrays from the
retained native support. Use that independent check when admitting an archive.
"""
    sampled = state['sampled_residual_jets']
    ref = epoch['surface_reference']
    x = epoch['surface_geometry'][:, :3]
    r2 = constant(np.sum(x*x, axis=-1)); r2[:, 1:4] = 2*x
    r2[:, [4, 8, 12]] = 2
    jets = sampled+ref
    jets[:, 0] = multiply(sampled[:, 0]+multiply(ref[:, 0], power(r2, -1)), r2)
    jets = close_dependent(jets, sampled)
    lift = state['normal_lift_jets']
    jets[:, 22:25] += lift[:, :3]
    if lift.shape[1] == 4:
        jets[:, 18] += lift[:, 3]
    return jets


def spacetime_metric(jets, x):
    """Penrose 4-metric and its Cartesian first/second spatial derivatives."""
    chi_inverse = power(jets[..., 0, :], -1)
    gamma = np.zeros(jets.shape[:-2]+(3, 3, 13))
    for slot, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), 1):
        gamma[..., i, j, :] = multiply(jets[..., slot, :]+constant(float(i == j)), chi_inverse)
        gamma[..., j, i, :] = gamma[..., i, j, :]
    lapse_reference = constant((1+np.sum(x*x, axis=-1))/2)
    lapse_reference[..., 1:4] = x
    lapse_reference[..., [4, 8, 12]] = 1
    alpha = jets[..., 18, :]+lapse_reference
    coords = constant(x)
    for i in range(3):
        coords[..., i, 1+i] = 1
    beta = jets[..., 22:25, :]-coords
    metric = np.zeros(jets.shape[:-2]+(4, 4, 13))
    metric[..., 1:, 1:, :] = gamma
    metric[..., 0, 0, :] = -multiply(alpha, alpha)
    for i in range(3):
        for j in range(3):
            metric[..., 0, i+1, :] += multiply(gamma[..., i, j, :], beta[..., j, :])
        metric[..., i+1, 0, :] = metric[..., 0, i+1, :]
        metric[..., 0, 0, :] += multiply(metric[..., 0, i+1, :], beta[..., i, :])
    return metric


def derivative_weights(nodes, center, order):
    """Polynomial finite differences on actual accepted (possibly uneven) times."""
    z = np.asarray(nodes)-center
    scale = np.max(abs(z))
    if not scale > 0 or order >= len(z):
        raise ValueError('insufficient distinct time nodes')
    rhs = np.zeros(len(z)); rhs[order] = math.factorial(order)
    return np.linalg.solve(np.vander(z/scale, N=len(z), increasing=True).T, rhs)/scale**order


def spacetime_derivatives(series, times, center):
    """Complete a 4-metric spatial-jet time stencil at its selected center."""
    g = series[center, ..., 0]
    dg = np.zeros(g.shape[:-2]+(4, 4, 4))
    ddg = np.zeros(g.shape[:-2]+(4, 4, 4, 4))
    spatial = series[center, ..., 1:4]
    dg[..., 1:, :, :] = np.moveaxis(spatial, -1, -3)
    dg[..., 0, :, :] = np.einsum('t,t...ab->...ab', derivative_weights(times, times[center], 1), series[..., 0])
    ddg[..., 0, 0, :, :] = np.einsum('t,t...ab->...ab', derivative_weights(times, times[center], 2), series[..., 0])
    mixed = np.einsum('t,t...abd->...dab', derivative_weights(times, times[center], 1), series[..., 1:4])
    ddg[..., 0, 1:, :, :] = mixed
    ddg[..., 1:, 0, :, :] = mixed
    ddg[..., 1:, 1:, :, :] = np.moveaxis(series[center, ..., 4:].reshape(g.shape+(3, 3)), (-2, -1), (-4, -3))
    return g, dg, ddg


def ricci(metric, dg, ddg):
    """4D Levi-Civita connection and Ricci tensor, signature (-+++)."""
    inverse = np.linalg.inv(metric)
    # h[d,b,c] = partial_b g_dc + partial_c g_db - partial_d g_bc.
    h = np.swapaxes(dg, -3, -2)+np.moveaxis(dg, -3, -1)-dg
    connection = .5*np.einsum('...ad,...dbc->...abc', inverse, h)
    dinverse = -np.einsum('...am,...emn,...nd->...ead', inverse, dg, inverse)
    dh = np.swapaxes(ddg, -3, -2)+np.moveaxis(ddg, -3, -1)-ddg
    dconnection = .5*(np.einsum('...ead,...dbc->...eabc', dinverse, h)
                         + np.einsum('...ad,...edbc->...eabc', inverse, dh))
    curvature = (np.einsum('...ccab->...ab', dconnection)
                 - np.einsum('...bcac->...ab', dconnection)
                 + np.einsum('...ccd,...dab->...ab', connection, connection)
                 - np.einsum('...cbd,...dac->...ab', connection, connection))
    return curvature, connection, inverse


def conformal_ricci_projection(ricci_tensor, connection, dyad, log_factor_gradient,
                               log_factor_hessian, factor):
    """R'_ab m'^a m'^b for g'=factor^2 g and m'=m/factor.

The omitted conformal transformation terms are proportional to g_ab, whose
contraction with a complex null dyad vanishes. This is a geometric projection;
its identification as news requires an admitted inertial conformal frame.
"""
    covariant_hessian = log_factor_hessian-np.einsum('...cab,...c->...ab', connection, log_factor_gradient)
    transformed = ricci_tensor-2*covariant_hessian+2*log_factor_gradient[..., :, None]*log_factor_gradient[..., None, :]
    return np.einsum('...a,...b,...ab->...', dyad, dyad, transformed)/factor**2
