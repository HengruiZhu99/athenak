#!/usr/bin/env python3
"""Independent 1D g=1 oracle for the small regular-center 3D lapse pilot.

Uses the previously verified analytically canceled spherical equations, their
quadrature shift, actual-origin limits and raw-l'Hopital scri limit. No 3D code
or angular averaging enters this evolution. This is not the wormhole benchmark.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from numerics import RadialOperators
from kernel import PARITY
import g1_rhs
from telegraph import window


def evolve(n=513, dt=.000125, steps=16, amplitude=1e-6):
    r = np.linspace(0., 1., n)
    op = RadialOperators(r)
    op.elliptic_quadrature = 'inward'
    omega = (1-r*r)/2
    y = np.zeros((n, 9))
    y[:, 6] = amplitude*(1-r*r)**4*np.exp(-r*r/.09)
    w, wp = window(r, inner_profile='af-regular')

    def rhs(state):
        q = state[:, :8]
        d, dd = op.derivatives(q, PARITY)
        regular, pole, _, beta, *_ = g1_rhs.parts(op, q, d, dd, restoring=1., slicing=1.)
        out = np.zeros_like(state)
        out[:-1, :8] = regular[:-1]+pole[:-1]/omega[:-1, None]
        dpole, _ = op.derivatives(pole, PARITY)
        out[-1, :8] = regular[-1]-dpole[-1]
        out[-1, 6] = 0.  # prescribed lapse clock, independent geometric rates
        B = state[:, 8]
        dB, _ = op.derivatives(B, -1)
        ratio = np.empty_like(B)
        ratio[1:] = B[1:]/r[1:]
        ratio[0] = dB[0]
        out[:, 6] += (1+q[:, 0])*(w*dB+wp*B+2*w*ratio)
        out[:, 8] = beta*dB+(.005*w*d[:, 6]-B)/.02
        return out

    for _ in range(steps):
        k1 = rhs(y)
        k2 = rhs(y+dt*k1/2)
        k3 = rhs(y+dt*k2/2)
        k4 = rhs(y+dt*k3)
        y += dt*(k1+2*k2+2*k3+k4)/6
    d, dd = op.derivatives(y[:, :8], PARITY)
    regular, pole, diagnostics, beta, *_ = g1_rhs.parts(op, y[:, :8], d, dd, restoring=1.)
    return dict(r=r, alpha=(1+r*r)/2+y[:, 6], chi=1+y[:, 0], g_rr=np.ones(n),
                A_rr=y[:, 2], Khat=y[:, 3]-3, Theta=y[:, 4], Lambda_r=y[:, 5],
                beta_r=beta, H=diagnostics[:, 0], B=y[:, 8], state=y)


def interpolate(r, values, x, width=7):
    """Local degree-six interpolation; no radial binning/smoothing of the 3D data."""
    x = np.asarray(x)
    first = np.clip(np.searchsorted(r, x)-width//2, 0, len(r)-width)
    result = np.zeros_like(x)
    for i in range(width):
        weight = np.ones_like(x)
        for j in range(width):
            if i != j:
                weight *= (x-r[first+j])/(r[first+i]-r[first+j])
        result += weight*values[first+i]
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = evolve()
    np.savez_compressed(args.output, **data)
    print('independent radial reference written:', args.output)
