"""Diagnostic NumPy transcription of C++ point geometry/RHS; no evolution code imports this."""
import numpy as np
pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def tensor(six):
    t = np.empty(six.shape[:-1] + (3, 3), dtype=six.dtype)
    for (q, (a, b)) in enumerate(pairs):
        t[..., a, b] = t[..., b, a] = six[..., q]
    return t

def curvature(v, d, dd, details=False):
    g = tensor(v[..., 1:7])
    gu = np.linalg.inv(g)
    dg = tensor(d[..., 1:7])
    ddg = tensor(dd[..., 1:7])
    cdown = np.empty(g.shape[:-2] + (3, 3, 3), dtype=v.dtype)
    for c in range(3):
        for a in range(3):
            for b in range(3):
                cdown[..., c, a, b] = 0.5 * (dg[..., a, b, c] + dg[..., b, a, c] - dg[..., c, a, b])
    cup = np.einsum('...cd,...dab->...cab', gu, cdown)
    contracted = np.einsum('...ab,...cab->...c', gu, cup)
    ric = np.zeros_like(g)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                ric[..., a, b] += 0.5 * (g[..., c, a] * d[..., b, 14 + c] + g[..., c, b] * d[..., a, 14 + c] + contracted[..., c] * (cdown[..., a, b, c] + cdown[..., b, a, c]))
                for e in range(3):
                    ric[..., a, b] -= 0.5 * gu[..., c, e] * ddg[..., c, e, a, b]
                    for f in range(3):
                        ric[..., a, b] += gu[..., c, e] * (cup[..., f, c, a] * cdown[..., b, f, e] + cup[..., f, c, b] * cdown[..., a, f, e] + cup[..., f, a, e] * cdown[..., f, c, b])
    chi = v[..., 0]
    phi = -0.25 * d[..., 0] / chi[..., None]
    covphi = -0.25 * dd[..., 0] / chi[..., None, None] + 4 * phi[..., :, None] * phi[..., None, :] - np.einsum('...cab,...c->...ab', cup, phi)
    rp = 4 * phi[..., :, None] * phi[..., None, :] - 2 * covphi
    rp -= 2 * g * np.einsum('...cd,...cd->...', gu, covphi + 2 * phi[..., :, None] * phi[..., None, :])[..., None, None]
    scalar = chi * np.einsum('...ab,...ab->...', gu, ric + rp)
    if details:
        return {'R': scalar, 'Rij': ric + rp, 'g': g, 'gu': gu, 'cup': cup, 'contracted': contracted, 'phi': phi}
    return scalar
def point_rhs(v, d, dd, adv, kappa=0.1):
    geo = curvature(v, d, dd, details=True)
    (g, gu, cup, C, phi) = (geo[k] for k in ['g', 'gu', 'cup', 'contracted', 'phi'])
    A = tensor(v[..., 8:14])
    Auu = np.einsum('...ac,...cd,...db->...ab', gu, A, gu)
    AAs = np.einsum('...ab,...ab->...', Auu, A)
    AAt = np.einsum('...ac,...cd,...db->...ab', A, gu, A)
    alpha = v[..., 18]
    chi = v[..., 0]
    K = v[..., 7] + 2 * v[..., 17]
    theta = v[..., 17]
    da = d[..., 18]
    db = d[..., 19:22]
    div = np.trace(db, axis1=-2, axis2=-1)
    hess = dd[..., 18] - 2 * (phi[..., :, None] * da[..., None, :] + phi[..., None, :] * da[..., :, None])
    hess -= np.einsum('...cab,...c->...ab', cup, da)
    hess += 2 * g * np.einsum('...cd,...c,...d->...', gu, phi, da)[..., None, None]
    lap = chi * np.einsum('...ab,...ab->...', gu, hess)
    DA = -1.5 * np.einsum('...ab,...b->...a', Auu, d[..., 0]) / chi[..., None]
    DA -= np.einsum('...ab,...b->...a', gu, 2 * d[..., 7] + d[..., 17]) / 3
    DA += np.einsum('...abc,...bc->...a', cup, Auu)
    Lg = tensor(adv[..., 1:7]) - 2 * g * div[..., None, None] / 3
    LA = tensor(adv[..., 8:14]) - 2 * A * div[..., None, None] / 3
    Lg += np.einsum('...ac,...bc->...ab', db, g) + np.einsum('...bc,...ac->...ab', db, g)
    LA += np.einsum('...ac,...bc->...ab', db, A) + np.einsum('...bc,...ac->...ab', db, A)
    LG = adv[..., 14:17] + 2 * C * div[..., None] / 3 - np.einsum('...b,...ba->...a', C, db)
    graddiv = np.zeros_like(C)
    for a in range(3):
        for b in range(3):
            graddiv[..., a] += dd[..., a, b, 19 + b] / 3
    LG += np.einsum('...ab,...b->...a', gu, graddiv)
    for a in range(3):
        LG[..., a] += np.einsum('...bc,...bc->...', gu, dd[..., 19 + a])
    out = np.zeros_like(v)
    out[..., 0] = adv[..., 0] - 2 * chi * div / 3 + 2 * chi * alpha * K / 3
    out[..., 7] = -lap + alpha * (AAs + K * K / 3) + adv[..., 7] + kappa * alpha * theta
    out[..., 17] = adv[..., 17] + alpha * (0.5 * (geo['R'] + 2 * K * K / 3 - AAs) - 2 * kappa * theta)
    out[..., 14:17] = 2 * alpha[..., None] * DA + LG - 2 * kappa * alpha[..., None] * (v[..., 14:17] - C) - 2 * np.einsum('...ab,...b->...a', Auu, da)
    rg = -2 * alpha[..., None, None] * A + Lg
    ra = chi[..., None, None] * (-hess + alpha[..., None, None] * geo['Rij'])
    ra -= g * (-lap + alpha * geo['R'])[..., None, None] / 3
    ra += alpha[..., None, None] * (K[..., None, None] * A - 2 * AAt) + LA
    for (q, (a, b)) in enumerate(pairs):
        out[..., 1 + q] = rg[..., a, b]
        out[..., 8 + q] = ra[..., a, b]
    return out
