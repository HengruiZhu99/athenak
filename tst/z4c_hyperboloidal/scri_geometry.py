#!/usr/bin/env python3
"""Independent analytic curvature, radiative shear, and conformal-gauge checks."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'vis/python'))
import scri_geometry as geometry


class Jet:
    """Analytic spacetime second jets used only to manufacture test metrics."""
    def __init__(self, value, axis=None):
        self.v = np.asarray(value, dtype=float)
        self.d = np.zeros(self.v.shape+(4,))
        self.dd = np.zeros(self.v.shape+(4, 4))
        if axis is not None:
            self.d[..., axis] = 1
    @staticmethod
    def cast(value):
        return value if isinstance(value, Jet) else Jet(value)
    def __add__(self, other):
        other = Jet.cast(other); out = Jet(self.v+other.v)
        out.d = self.d+other.d; out.dd = self.dd+other.dd
        return out
    __radd__ = __add__
    def __neg__(self):
        return self*-1
    def __sub__(self, other):
        return self+-Jet.cast(other)
    def __rsub__(self, other):
        return Jet.cast(other)+-self
    def __mul__(self, other):
        other = Jet.cast(other); out = Jet(self.v*other.v)
        out.d = self.d*other.v[..., None]+self.v[..., None]*other.d
        out.dd = (self.dd*other.v[..., None, None]+self.v[..., None, None]*other.dd
                  + self.d[..., :, None]*other.d[..., None, :]+other.d[..., :, None]*self.d[..., None, :])
        return out
    __rmul__ = __mul__
    def function(self, value, first, second):
        out = Jet(value)
        out.d = first[..., None]*self.d
        out.dd = first[..., None, None]*self.dd+second[..., None, None]*self.d[..., :, None]*self.d[..., None, :]
        return out
    def __pow__(self, p):
        return self.function(self.v**p, p*self.v**(p-1), p*(p-1)*self.v**(p-2))
    def __truediv__(self, other):
        return self*Jet.cast(other)**-1
    def __rtruediv__(self, other):
        return Jet.cast(other)*self**-1
    def sin(self):
        return self.function(np.sin(self.v), np.cos(self.v), -np.sin(self.v))
    def cos(self):
        return self.function(np.cos(self.v), -np.sin(self.v), -np.cos(self.v))
    def exp(self):
        v = np.exp(self.v)
        return self.function(v, v, v)


def arrays(matrix, count):
    g = np.empty((count, 4, 4)); dg = np.empty((count, 4, 4, 4)); ddg = np.empty((count, 4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            q = Jet.cast(matrix[a][b]); g[:, a, b] = q.v
            dg[:, :, a, b] = q.d; ddg[:, :, :, a, b] = q.dd
    return g, dg, ddg


def radiative_metric(t, x, amplitude=.02, frequency=1.7, gauge=False, rotation=0, return_omega=False, time_gauge=0):
    # Bondi conformal metric: g_uu=-ell^2, g_uell=1,
    # g_AB=q_AB+ell*a(u)*sin(theta)^2*diag(1,-sin(theta)^2).
    # Pull back u=t+(1-r)/(1+r), ell=(1-r^2)/(2r), then multiply by r^2.
    # The background becomes the actual native hyperboloidal Minkowski metric.
    t = Jet(np.full(len(x), t), 0)
    xyz = [Jet(x[:, i], i+1) for i in range(3)]
    angle = rotation*t; cosine, sine = angle.cos(), angle.sin()
    original = xyz
    xyz = [cosine*original[0]-sine*original[1], sine*original[0]+cosine*original[1], original[2]]
    jac = [[Jet(0) for _ in range(4)] for _ in range(4)]
    r_original = sum(c*c for c in original)**.5
    mu_original = original[0]/r_original
    p2 = (3*mu_original*mu_original-1)/2
    physical_time = t+time_gauge*t.sin()*p2
    jac[0][0] = 1+time_gauge*t.cos()*p2
    for i in range(3):
        jac[0][i+1] = time_gauge*t.sin()*3*mu_original*(float(i == 0)-mu_original*original[i]/r_original)/r_original
    jac[3][3] = Jet(1)
    jac[1][0], jac[2][0] = -rotation*xyz[1], rotation*xyz[0]
    jac[1][1], jac[1][2], jac[2][1], jac[2][2] = cosine, -sine, sine, cosine
    r2 = sum(c*c for c in xyz); r = r2**.5; omega = (1-r2)/2
    n = [c/r for c in xyz]; mu = n[0]
    u = physical_time+(1-r)/(1+r); ell = omega/r
    wave = amplitude*(frequency*u).sin()
    v = [1-mu*n[0], -mu*n[1], -mu*n[2]]
    w = [Jet(0), n[2], -n[1]]
    matrix = [[Jet(0) for _ in range(4)] for _ in range(4)]
    matrix[0][0] = -omega*omega
    for i in range(3):
        matrix[0][i+1] = matrix[i+1][0] = -xyz[i]
        for j in range(3):
            matrix[i+1][j+1] = float(i == j)+ell*wave*(v[i]*v[j]-w[i]*w[j])
    log_undo = -(.17*t+.13*mu*mu) if gauge else Jet(np.zeros(len(x)))
    scale = (-log_undo).exp()
    if gauge:
        matrix = [[m*scale*scale for m in row] for row in matrix]
    pulled = [[sum(jac[c][a]*matrix[c][d]*jac[d][b] for c in range(4) for d in range(4))
               for b in range(4)] for a in range(4)]
    result = (*arrays(pulled, len(x)), log_undo, scale.v)
    return (*result, omega*scale) if return_omega else result


def main():
    theta = np.linspace(.1, 3, 19)
    q = Jet(theta, 2)
    product = [[0]*4 for _ in range(4)]
    for a, value in enumerate([-1, 1, 1, q.sin()**2]):
        product[a][a] = value
    g, dg, ddg = arrays(product, len(theta)); ricci, _, _ = geometry.ricci(g, dg, ddg)
    expected = np.zeros_like(g); expected[:, 2, 2] = 1; expected[:, 3, 3] = np.sin(theta)**2
    np.testing.assert_allclose(ricci, expected, atol=4e-14)
    time = Jet(np.linspace(-1, 1, 19), 0); h = .7; scale = (h*time).exp()
    flrw = [[0]*4 for _ in range(4)]
    for a, value in enumerate([-1, scale*scale, scale*scale, scale*scale]):
        flrw[a][a] = value
    g, dg, ddg = arrays(flrw, 19); ricci, _, _ = geometry.ricci(g, dg, ddg)
    np.testing.assert_allclose(ricci, 3*h*h*g, atol=2e-14)
    phi = np.linspace(.2, 5.5, len(theta))
    x = np.column_stack((np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)))
    etheta = np.column_stack((-np.sin(theta), np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi)))
    ephi = np.column_stack((np.zeros(len(theta)), -np.sin(phi), np.cos(phi)))
    dyad = np.column_stack((np.zeros(len(theta)), (etheta-1j*ephi)/np.sqrt(2)))
    t = .4
    for amplitude in (0, .02):
        expected = amplitude*1.7*np.cos(1.7*t)*np.sin(theta)**2
        for gauge in (False, True):
            g, dg, ddg, log_undo, scale = radiative_metric(t, x, amplitude=amplitude, gauge=gauge)
            ricci, connection, _ = geometry.ricci(g, dg, ddg)
            news = -geometry.conformal_ricci_projection(ricci, connection, dyad/scale[:, None], log_undo.d, log_undo.dd, 1/scale)
            np.testing.assert_allclose(news, expected, atol=3e-14)
            print('PASS manufactured Bondi shear derivative', amplitude, gauge, float(np.max(abs(news-expected))))
    # Actual nonuniform accepted times, including mixed spacetime derivatives.
    times = t+np.array([-.031, -.021, -.01, 0, .009, .0205, .0302])
    samples = []
    for at in times:
        g, dg, ddg, _, _ = radiative_metric(at, x)
        samples.append(np.concatenate((g[..., None], np.moveaxis(dg[:, 1:], 1, -1),
                           np.moveaxis(ddg[:, 1:, 1:], (1, 2), (-2, -1)).reshape(g.shape+(9,))), axis=-1))
    numerical = geometry.spacetime_derivatives(np.array(samples), times, 3)
    exact = radiative_metric(t, x)[:3]
    for actual, reference in zip(numerical, exact):
        np.testing.assert_allclose(actual, reference, atol=1e-10)
    print('PASS curvature conventions, conformal gauge, nonuniform temporal and mixed derivatives')


if __name__ == '__main__':
    main()
