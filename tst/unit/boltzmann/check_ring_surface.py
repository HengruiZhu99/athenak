"""Check outgoing expansion, area and axial spin on analytic Kerr-Schild horizons."""

import numpy as np
from numpy.polynomial import legendre as leg

from ring_surface import measure


class KerrSchild:
    dx = 0.02
    bound = 8.0

    def __init__(self, a):
        self.a = a

    def fields(self, points):
        x, y, z = points.T
        a = self.a
        b = x*x + y*y + z*z - a*a
        r = np.sqrt((b + np.hypot(b, 2*a*z))/2)
        l = np.column_stack(((r*x+a*y)/(r*r+a*a), (r*y-a*x)/(r*r+a*a), z/r))
        twice_h = 2*r**3/(r**4 + a*a*z*z)
        return (np.eye(3) + np.einsum("n,ni,nj->nij", twice_h, l, l),
                twice_h[:, None]*l, 1/np.sqrt(1+twice_h))

    def tensor(self, rho, z, start):
        points = np.column_stack((rho, np.zeros_like(rho), z))
        g, shift, lapse = self.fields(points)
        if start:
            # Stationarity gives 2 alpha K_ij = D_i beta_j + D_j beta_i.
            eps = 1.e-5
            dg, db = [], []
            for d in np.eye(3)*eps:
                gp, bp, _ = self.fields(points+d)
                gm, bm, _ = self.fields(points-d)
                dg.append((gp-gm)/(2*eps))
                db.append((bp-bm)/(2*eps))
            grad = np.stack(dg, axis=1)
            dshift = np.stack(db, axis=1)
            beta = np.einsum("nij,nj->ni", np.linalg.inv(g), shift)
            conn = grad + grad.swapaxes(1, 2) - grad.transpose(0, 2, 3, 1)
            g = (dshift + dshift.swapaxes(1, 2) - np.einsum("nk,nijk->nij", beta, conn))
            g /= 2*lapse[:, None, None]
        return g[:, [0, 2, 1]][:, :, [0, 2, 1]]


if __name__ == "__main__":
    x, w = leg.leggauss(128)
    theta = np.arccos(x)
    for a in [0.0, 0.95, -0.95]:
        rp = 1 + np.sqrt(1-a*a)
        h = 1/np.sqrt((1-x*x)/(rp*rp+a*a) + x*x/(rp*rp))
        coeff = leg.legfit(x, h, 40)
        _, plus, minus, da, integrand = measure(KerrSchild(a), coeff, theta)
        area = 2*np.pi*(w @ (da/np.sin(theta)))
        j = 0.25*(w @ (da*integrand/np.sin(theta)))
        print(a, np.max(abs(plus)), max(minus), area, j)
        assert np.max(abs(plus)) < 2e-7 and max(minus) < 0
        np.testing.assert_allclose(area, 8*np.pi*rp, rtol=1e-10)
        np.testing.assert_allclose(j, a, atol=1e-9)
