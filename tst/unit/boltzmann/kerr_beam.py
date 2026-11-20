"""Independent gray Kerr beam oracle in Cartesian canonical phase space.

Trace each arrival direction backwards with H=alpha sqrt(gamma^ij p_i p_j)
- beta^i p_i. The intensity transported to unit arrival energy is I0/epsilon0^4.
The test uses the stationary quasi-isotropic radial coordinate of the Kerr pgen.
"""

import numpy as np


def metric(x):
    rx, ry, rz = x
    r2 = rx * rx + ry * ry + rz * rz
    cyl2 = rx * rx + ry * ry
    r = np.sqrt(r2)
    a, c, rminus = 0.6, 0.45, 0.2  # M=1, chi=.6, c=r_plus/4
    bl = r + 2 * c + c * c / r
    sigma = bl * bl + a * a * rz * rz / r2
    delta = bl * bl - 2 * bl + a * a
    cap = (bl * bl + a * a) ** 2 - delta * a * a * cyl2 / r2
    radial = sigma * (r + c) ** 2 / (r**3 * (bl - rminus))
    polar = sigma / r2
    azimuthal = cap / (sigma * r2)
    return np.sqrt(delta * sigma / cap), -2 * a * bl / cap, radial, polar, azimuthal


def hamiltonian(x, p, velocity=False):
    alpha, beta, gr, gt, gp = metric(x)
    rx, ry, _ = x
    lz = rx * p[1] - ry * p[0]
    ang = (1 / gp - 1 / gt) * lz / (rx * rx + ry * ry)
    rad = (1 / gr - 1 / gt) * np.sum(x * p, axis=0) / np.sum(x * x, axis=0)
    rot = np.stack((-ry, rx, np.zeros_like(rx)))
    up = p / gt + x * rad + rot * ang
    eps = np.sqrt(np.sum(up * p, axis=0))
    if velocity:
        return alpha * up / eps - beta * rot, eps
    return alpha * eps - beta * lz


def rhs(state):
    x, p = state[:3], state[3:]
    v, _ = hamiltonian(x, p, True)
    force = []
    # Complex-step derivatives avoid differencing nearby Hamiltonian values.
    for j in range(3):
        probe = x.astype(complex)
        probe[j] += 1.0e-20j
        force.append(-hamiltonian(probe, p).imag / 1.0e-20)
    return np.vstack((v, force))


def coframe(x):
    _, _, gr, gt, gp = metric(x)
    r = np.hypot(x[0], x[1])
    z = x[2]
    total = r * r + z * z
    c11 = np.sqrt(gt + (gr - gt) * z * z / total)
    c01 = (gr - gt) * r * z / (total * c11)
    c00 = np.sqrt(gt + (gr - gt) * r * r / total - c01 * c01)
    return c00, c01, c11, np.sqrt(gp)


def stress(r, z, t, step=0.08, angular=32):
    mu, weight = np.polynomial.legendre.leggauss(angular)
    psi = (np.arange(angular * 3) + 0.5) * 2 * np.pi / (angular * 3)
    mu, psi = np.meshgrid((mu + 1) / 2, psi, indexing="ij")
    n0 = (np.sqrt(1 - mu**2) * np.cos(psi)).ravel()
    n1 = (np.sqrt(1 - mu**2) * np.sin(psi)).ravel()
    n2 = mu.ravel()
    # The black hole is below the z domain at -12, so every point on the grid
    # is in a regular exterior patch, including the evolved axis vertices.
    x = np.array([np.full_like(n0, r), np.zeros_like(n0), np.full_like(n0, z + 12)])
    c00, c01, c11, c22 = coframe(x)
    p = np.array([c00 * n0 + c01 * n1, c22 * n2, c11 * n1])
    initial_pr = p[0].copy()
    state = np.vstack((x, p))
    steps = max(1, int(np.ceil(t / step)))
    dt = -t / steps
    for _ in range(steps):
        a = rhs(state)
        b = rhs(state + 0.5 * dt * a)
        c = rhs(state + 0.5 * dt * b)
        d = rhs(state + dt * c)
        state += dt * (a + 2 * b + 2 * c + d) / 6
    x, p = state[:3], state[3:]
    rs = np.hypot(x[0], x[1])
    c00, c01, c11, c22 = coframe(x)
    _, eps = hamiltonian(x, p, True)
    axial = p[2] / c11
    radial = ((p[0] * x[0] + p[1] * x[1]) / rs - c01 * axial) / (c00 * eps)
    swirl = (p[1] * x[0] - p[0] * x[1]) / (rs * c22 * eps)
    dot = 0.25 * swirl - np.sqrt(1 - 0.25**2) * radial
    intensity = (
        0.002 * np.exp(-0.5 * ((rs - 6) ** 2 + (x[2] - 12) ** 2) + (dot - 1) / 0.3**2) / eps**4
    )
    intensity *= (rs <= 16) & (abs(x[2] - 12) <= 8)

    def integral(v):
        return np.sum(v.reshape(mu.shape) * weight[:, None]) * np.pi / len(psi[0])

    return integral(intensity), integral(intensity * initial_pr)
