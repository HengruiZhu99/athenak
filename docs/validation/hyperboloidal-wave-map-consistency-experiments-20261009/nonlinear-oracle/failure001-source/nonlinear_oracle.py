"""Finite sampled exact-flat wave-map oracle. No scientific kernel imports."""
import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import mpmath as mp

from jet_algebra import Jet, det3, export, from_ordinary, inverse3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE/'reference_helpers'))
from radial_oracle import component, exact, radial, side_at
from cartesian_composition import scalar_cart, vector_cart, tensor_cart


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value, digits):
    if isinstance(value, dict):
        return {k: serial(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v, digits) for v in value]
    if isinstance(value, Jet):
        return export(value, digits)
    if isinstance(value, (str, bool, int)):
        return value
    return mp.nstr(value, digits, strip_zeros=False)


def ref(r, name):
    side = side_at(r)
    return component(r, side, name, False)+component(r, side, name, True)


def hprime(r):
    if r <= exact(.05):
        return mp.mpf(0)
    return ref(r, 'b')*ref(r, 'L')/(ref(r, 'alpha')*ref(r, 'omega')**2)


class Height:
    def __init__(self, panels):
        self.panels = [exact(x) for x in panels]
        self.cache = {}
        self.prefix = [mp.mpf(0)]
        for a, b in zip(self.panels[:-1], self.panels[1:]):
            self.prefix.append(self.prefix[-1]+mp.quadts(hprime, [a, b]))
        self.match = self.prefix[-1]
        r = exact(.95)
        self.constant = self.match-mp.sqrt((r/ref(r, 'omega'))**2+mp.mpf('.5')**2)

    def integral(self, r):
        if r <= self.panels[0]:
            return mp.mpf(0)
        key = mp.nstr(r, mp.mp.dps+5)
        if key not in self.cache:
            i = max(k for k, x in enumerate(self.panels) if x <= r)
            self.cache[key] = self.prefix[i]+(mp.quadts(hprime, [self.panels[i], r])
                                              if r > self.panels[i] else 0)
        return self.cache[key]

    def value(self, r):
        if r >= exact(.95):
            rr = r/ref(r, 'omega')
            return mp.sqrt(rr*rr+mp.mpf('.5')**2)+self.constant
        return self.integral(r)


def embedding(point, height):
    t, xyz = point[0], point[1:]
    r = mp.sqrt(sum(x*x for x in xyz))
    omega = ref(r, 'omega')
    value = [t+height.value(r)]+[x/omega for x in xyz]
    first = [[mp.mpf(0)]*4 for _ in range(4)]
    second = [[[mp.mpf(0)]*4 for _ in range(4)] for _ in range(4)]
    third = [[[[mp.mpf(0)]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    first[0][0] = 1
    if r == 0:
        for i in range(3):
            first[i+1][i+1] = 1
        omega_jet = Jet(1)
        return value, first, second, third, omega_jet
    hjet = [height.value(r), hprime(r), mp.diff(hprime, r), mp.diff(hprime, r, 2)]
    oj = [radial(r, 'omega', n) for n in range(4)]
    fjet = [1/oj[0], -oj[1]/oj[0]**2,
            2*oj[1]**2/oj[0]**3-oj[2]/oj[0]**2,
            -6*oj[1]**3/oj[0]**4+6*oj[1]*oj[2]/oj[0]**3-oj[3]/oj[0]**2]
    for axes_count in (1, 2, 3):
        for axes in itertools.product(range(3), repeat=axes_count):
            m = tuple(axes.count(i) for i in range(3))
            target = first if axes_count == 1 else second if axes_count == 2 else third
            for a in range(4):
                number = scalar_cart(hjet, xyz, m) if a == 0 else vector_cart(fjet, xyz, m, a-1)
                if axes_count == 1:
                    target[a][axes[0]+1] = number
                elif axes_count == 2:
                    target[a][axes[0]+1][axes[1]+1] = number
                else:
                    target[a][axes[0]+1][axes[1]+1][axes[2]+1] = number
    od = [mp.mpf(0)]+[scalar_cart(oj, xyz, tuple(int(k == i) for k in range(3))) for i in range(3)]
    odd = [[mp.mpf(0)]*4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            m = tuple(int(k == i)+int(k == j) for k in range(3))
            odd[i+1][j+1] = scalar_cart(oj, xyz, m)
    return value, first, second, third, from_ordinary(omega, od, odd)


def phi_rho(t, rho):
    sigma = mp.mpf('.35')
    q = (t+mp.mpf('.5'))/sigma
    return 4*q*mp.exp(-q*q-rho/sigma**2)*mp.hyp0f1(mp.mpf(3)/2, q*q*rho/sigma**2)


def phi_jets(x):
    t, xyz = x[0], x[1:]
    rho = sum(v*v for v in xyz)
    mixed = {(nt, nr): mp.diff(phi_rho, (t, rho), (nt, nr))
             for nt in range(4) for nr in range(4-nt)}
    def d(axes):
        nt = axes.count(0)
        spatial = [i-1 for i in axes if i]
        if not spatial:
            return mixed[nt, 0]
        if len(spatial) == 1:
            return 2*xyz[spatial[0]]*mixed[nt, 1]
        if len(spatial) == 2:
            i, j = spatial
            return 2*int(i == j)*mixed[nt, 1]+4*xyz[i]*xyz[j]*mixed[nt, 2]
        i, j, k = spatial
        return (4*(int(i == j)*xyz[k]+int(i == k)*xyz[j]+int(j == k)*xyz[i])*mixed[0, 2]
                +8*xyz[i]*xyz[j]*xyz[k]*mixed[0, 3])
    return (mixed[0, 0], [d([i]) for i in range(4)],
            [[d([i, j]) for j in range(4)] for i in range(4)],
            [[[d([i, j, k]) for k in range(4)] for j in range(4)] for i in range(4)], mixed)


def inverse_map(y, y1, y2, y3, epsilon):
    if epsilon == 0:
        x = list(y)
        iterations = 0
    else:
        lo, hi = y[3]-2*epsilon, y[3]+2*epsilon
        z = y[3]
        iterations = 0
        for iterations in range(1, 600):
            value = phi_rho(y[0], y[1]**2+y[2]**2+z*z)
            f = z+epsilon*value-y[3]
            if abs(f) <= mp.mpf('1e-70')*max(1, abs(y[3])):
                break
            if f > 0:
                hi = z
            else:
                lo = z
            derivative = 1+epsilon*mp.diff(lambda zz: phi_rho(y[0], y[1]**2+y[2]**2+zz*zz), z)
            proposal = z-f/derivative
            z = proposal if lo < proposal < hi else (lo+hi)/2
        else:
            raise RuntimeError('Prescribed inverse scalar solve did not converge')
        x = list(y[:3])+[z]
    phi, p1, p2, p3, mixed = phi_jets(x)
    determinant = 1+epsilon*p1[3]
    assert determinant > 0, 'Sample inverse-map orientation failure'
    ji = [[mp.mpf(int(a == b))-(epsilon*p1[b]/determinant if a == 3 else 0)
           for b in range(4)] for a in range(4)]
    x1 = [[sum(ji[a][b]*y1[b][i] for b in range(4)) for i in range(4)] for a in range(4)]
    x2 = [[[mp.mpf(0)]*4 for _ in range(4)] for _ in range(4)]
    x3 = [[[[mp.mpf(0)]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for i, j in itertools.product(range(4), repeat=2):
        correction = epsilon*sum(p2[b][c]*x1[b][i]*x1[c][j] for b, c in itertools.product(range(4), repeat=2))
        for a in range(4):
            x2[a][i][j] = sum(ji[a][b]*(y2[b][i][j]-(correction if b == 3 else 0)) for b in range(4))
    for i, j, k in itertools.product(range(4), repeat=3):
        correction = epsilon*(
            sum(p3[b][c][d]*x1[b][i]*x1[c][j]*x1[d][k]
                for b, c, d in itertools.product(range(4), repeat=3))
            +sum(p2[b][c]*(x2[b][i][j]*x1[c][k]+x2[b][i][k]*x1[c][j]+x2[b][j][k]*x1[c][i])
                 for b, c in itertools.product(range(4), repeat=2)))
        for a in range(4):
            x3[a][i][j][k] = sum(ji[a][b]*(y3[b][i][j][k]-(correction if b == 3 else 0)) for b in range(4))
    return x, x1, x2, x3, dict(phi=phi, first=p1, second=p2, third=p3,
                               mixed=mixed, determinant=determinant, iterations=iterations)


def metric(x1, x2, x3):
    sign = [-1, 1, 1, 1]
    g = [[None]*4 for _ in range(4)]
    for a, b in itertools.product(range(4), repeat=2):
        value = sum(sign[c]*x1[c][a]*x1[c][b] for c in range(4))
        first = [sum(sign[c]*(x2[c][a][i]*x1[c][b]+x1[c][a]*x2[c][b][i]) for c in range(4)) for i in range(4)]
        second = [[sum(sign[c]*(x3[c][a][i][j]*x1[c][b]+x2[c][a][i]*x2[c][b][j]
                                +x2[c][a][j]*x2[c][b][i]+x1[c][a]*x3[c][b][i][j])
                       for c in range(4)) for j in range(4)] for i in range(4)]
        g[a][b] = from_ordinary(value, first, second)
    return g


def geometry(g, axes):
    n = len(g)
    inv = mp.inverse(mp.matrix([[v.v for v in row] for row in g]))
    gamma = [[[mp.mpf(0)]*n for _ in range(n)] for _ in range(n)]
    dg = [[[[mp.mpf(0)]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a, b, c in itertools.product(range(n), repeat=3):
        gamma[a][b][c] = sum(inv[a, d]*(g[d][b].derivative(axes[c])+g[d][c].derivative(axes[b])
                                       -g[b][c].derivative(axes[d]))/2 for d in range(n))
    for e in range(n):
        dinv = [[-sum(inv[a, b]*g[b][c].derivative(axes[e])*inv[c, d]
                      for b, c in itertools.product(range(n), repeat=2)) for d in range(n)] for a in range(n)]
        for a, b, c in itertools.product(range(n), repeat=3):
            dg[e][a][b][c] = sum(
                dinv[a][d]*(g[d][b].derivative(axes[c])+g[d][c].derivative(axes[b])-g[b][c].derivative(axes[d]))/2
                +inv[a, d]*(g[d][b].derivative(axes[c], axes[e])+g[d][c].derivative(axes[b], axes[e])
                             -g[b][c].derivative(axes[d], axes[e]))/2 for d in range(n))
    riemann = [[[[mp.mpf(0)]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    scaled = mp.mpf(0)
    absolute = mp.mpf(0)
    for a, b, c, d in itertools.product(range(n), repeat=4):
        terms = [dg[c][a][d][b], -dg[d][a][c][b]]
        terms += [gamma[a][c][e]*gamma[e][d][b] for e in range(n)]
        terms += [-gamma[a][d][e]*gamma[e][c][b] for e in range(n)]
        value = sum(terms)
        riemann[a][b][c][d] = value
        absolute = max(absolute, abs(value))
        scaled = max(scaled, abs(value)/max(1, sum(abs(v) for v in terms)))
    ricci = [[sum(riemann[a][b][a][d] for a in range(n)) for d in range(n)] for b in range(n)]
    return inv, gamma, riemann, ricci, scaled, absolute


def storage(g, omega):
    gamma = [row[1:] for row in g[1:]]
    gi = inverse3(gamma)
    beta = [sum(gi[i][j]*g[0][j+1] for j in range(3)) for i in range(3)]
    lapse2 = -g[0][0]+sum(g[0][i+1]*beta[i] for i in range(3))
    assert lapse2.v > 0, 'Physical constant-t slices are not spacelike'
    lapse = lapse2**mp.mpf('.5')
    k = [[-(gamma[i][j].diff(0)
            -sum(beta[a]*gamma[i][j].diff(a+1)+gamma[a][j]*beta[a].diff(i+1)
                 +gamma[i][a]*beta[a].diff(j+1) for a in range(3)))/(2*lapse)
          for j in range(3)] for i in range(3)]
    trace = sum(gi[i][j]*k[i][j] for i, j in itertools.product(range(3), repeat=2))
    bargamma = [[omega**2*v for v in row] for row in gamma]
    determinant = det3(bargamma)
    assert determinant.v > 0
    chi = determinant**(-mp.mpf(1)/3)
    gt = [[chi*v for v in row] for row in bargamma]
    ti = inverse3(gt)
    aa = [[omega*chi*(k[i][j]-gamma[i][j]*trace/3) for j in range(3)] for i in range(3)]
    christoffel = [[[sum(ti[i][ell]*(gt[ell][j].diff(k+1)+gt[ell][k].diff(j+1)
                                    -gt[j][k].diff(ell+1))/2 for ell in range(3))
                     for k in range(3)] for j in range(3)] for i in range(3)]
    lam = [sum(ti[j][k]*christoffel[i][j][k] for j, k in itertools.product(range(3), repeat=2)) for i in range(3)]
    return dict(physical_lapse=lapse, physical_gamma=gamma, physical_gamma_inverse=gi,
                physical_K=k, beta=beta, alpha=omega*lapse, omega=omega,
                chi=chi, metric=gt, metric_inverse=ti, A=aa, P=trace, Lambda=lam,
                Theta=Jet(0, 1))


def case(point_definition, epsilon, height, digits):
    point = [exact(float.fromhex(point_definition['time_hex']))]
    point += [exact(float.fromhex(x)) for x in point_definition['cartesian_hex']]
    y, y1, y2, y3, omega = embedding(point, height)
    x, x1, x2, x3, inverse = inverse_map(y, y1, y2, y3, epsilon)
    g = metric(x1, x2, x3)
    refg = metric(y1, y2, y3)
    u = storage(g, omega)
    gi, gamma, riemann, ricci, curvature_scaled, curvature_abs = geometry(g, range(4))
    _, refgamma, _, _, _, _ = geometry(refg, range(4))
    inverse_y1 = mp.inverse(mp.matrix(y1))
    reference_connection = [[[sum(inverse_y1[a, c]*y2[c][b][d] for c in range(4))
                             for d in range(4)] for b in range(4)] for a in range(4)]
    checks = []
    def check(name, residual, terms, tolerance=mp.mpf('1e-55')):
        scale = max(1, sum(abs(v) for v in terms))
        error = abs(residual)/scale
        checks.append(dict(name=name, absolute=abs(residual), scale=scale, scaled=error,
                           tolerance=tolerance, passed=bool(error <= tolerance)))
    check('scalar_inverse', x[3]+epsilon*inverse['phi']-y[3],
          [x[3], epsilon*inverse['phi'], y[3]], mp.mpf('1e-65'))
    check('box_phi', -inverse['mixed'][2, 0]+6*inverse['mixed'][0, 1]
          +4*sum(v*v for v in x[1:])*inverse['mixed'][0, 2],
          [inverse['mixed'][2, 0], 6*inverse['mixed'][0, 1],
           4*sum(v*v for v in x[1:])*inverse['mixed'][0, 2]])
    radius = mp.sqrt(sum(v*v for v in x[1:]))
    sigma = mp.mpf('.35')
    f = lambda s: mp.exp(-((s+mp.mpf('.5'))/sigma)**2)
    original_phi = (-2*sigma*mp.diff(f, x[0]) if radius == 0 else
                    sigma*(f(x[0]-radius)-f(x[0]+radius))/radius)
    check('phi_regular_original_agreement', inverse['phi']-original_phi, [inverse['phi'], original_phi])
    check('flat_Riemann', curvature_scaled, [1])
    for a, b, c in itertools.product(range(4), repeat=3):
        check('reference_connection_%d_%d_%d' % (a, b, c),
              refgamma[a][b][c]-reference_connection[a][b][c],
              [refgamma[a][b][c], reference_connection[a][b][c]])
    for a in range(4):
        terms = [gi[b, c]*(gamma[a][b][c]-reference_connection[a][b][c])
                 for b, c in itertools.product(range(4), repeat=2)]
        check('wave_map_%d' % a, sum(terms), terms)
        terms = [gi[b, c]*(y2[a][b][c]-sum(gamma[d][b][c]*y1[a][d] for d in range(4)))
                 for b, c in itertools.product(range(4), repeat=2)]
        check('harmonic_Yhat_%d' % a, sum(terms), terms)
    # Full differentiated implicit identities, independently forward applied.
    p1, p2, p3 = inverse['first'], inverse['second'], inverse['third']
    for i in range(4):
        terms = [x1[3][i], epsilon*sum(p1[b]*x1[b][i] for b in range(4)), -y1[3][i]]
        check('implicit_first_%d' % i, sum(terms), terms)
    for i, j in itertools.product(range(4), repeat=2):
        terms = [x2[3][i][j], epsilon*sum(p1[b]*x2[b][i][j] for b in range(4)),
                 epsilon*sum(p2[b][c]*x1[b][i]*x1[c][j] for b, c in itertools.product(range(4), repeat=2)), -y2[3][i][j]]
        check('implicit_second_%d_%d' % (i, j), sum(terms), terms)
    for i, j, k in itertools.product(range(4), repeat=3):
        terms = [x3[3][i][j][k], epsilon*sum(p1[b]*x3[b][i][j][k] for b in range(4)),
                 epsilon*sum(p3[b][c][d]*x1[b][i]*x1[c][j]*x1[d][k] for b, c, d in itertools.product(range(4), repeat=3)),
                 epsilon*sum(p2[b][c]*(x2[b][i][j]*x1[c][k]+x2[b][i][k]*x1[c][j]+x2[b][j][k]*x1[c][i])
                             for b, c in itertools.product(range(4), repeat=2)), -y3[3][i][j][k]]
        check('implicit_third_%d_%d_%d' % (i, j, k), sum(terms), terms)
    # ADM metric reconstruction and stored algebraic/connection conditions.
    physical_gamma, physical_inverse, k = u['physical_gamma'], u['physical_gamma_inverse'], u['physical_K']
    beta, lapse, alpha, chi, ti = u['beta'], u['physical_lapse'], u['alpha'], u['chi'], u['metric_inverse']
    minors = [physical_gamma[0][0].v,
              (physical_gamma[0][0]*physical_gamma[1][1]-physical_gamma[0][1]*physical_gamma[1][0]).v,
              det3(physical_gamma).v]
    assert all(v > 0 for v in minors), 'Sample physical spatial metric is not SPD'
    reconstructed = -lapse*lapse+sum(physical_gamma[i][j]*beta[i]*beta[j] for i, j in itertools.product(range(3), repeat=2))
    for m in reconstructed.c:
        check('ADM_g00_%s' % (m,), reconstructed.c[m]-g[0][0].c[m], [reconstructed.c[m], g[0][0].c[m]])
    determinant_tilde = det3(u['metric'])-1
    for m, v in determinant_tilde.c.items():
        check('det_gtilde_%s' % (m,), v, [1])
    atrace = sum(ti[i][j]*u['A'][i][j] for i, j in itertools.product(range(3), repeat=2))
    for m, v in atrace.c.items():
        check('A_trace_%s' % (m,), v, [1]+[ti[i][j].v*u['A'][i][j].v for i, j in itertools.product(range(3), repeat=2)])
    for i in range(3):
        alternate = -sum(ti[i][j].diff(j+1) for j in range(3))
        for m in alternate.c:
            check('Lambda_identity_%d_%s' % (i, m), alternate.c[m]-u['Lambda'][i].c[m], [alternate.c[m], u['Lambda'][i].c[m]])
    # Physical ADM H/M constraints use the independently assembled spatial Ricci.
    _, spatial_connection, _, spatial_ricci, _, _ = geometry(physical_gamma, [1, 2, 3])
    curvature = sum(physical_inverse[i][j].v*spatial_ricci[i][j] for i, j in itertools.product(range(3), repeat=2))
    k2 = sum(physical_inverse[i][a].v*physical_inverse[j][b].v*k[i][j].v*k[a][b].v
             for i, j, a, b in itertools.product(range(3), repeat=4))
    check('physical_H', curvature+u['P'].v**2-k2, [curvature, u['P'].v**2, k2])
    kmixed = [[sum(physical_inverse[j][a]*k[a][i] for a in range(3)) for i in range(3)] for j in range(3)]
    for i in range(3):
        terms = [kmixed[j][i].derivative(j+1) for j in range(3)]
        terms += [spatial_connection[j][j][a]*kmixed[a][i].v for j, a in itertools.product(range(3), repeat=2)]
        terms += [-spatial_connection[a][j][i]*kmixed[j][a].v for j, a in itertools.product(range(3), repeat=2)]
        terms += [-u['P'].derivative(i+1)]
        check('physical_M_%d' % i, sum(terms), terms)
    # Independent finite-Omega conformal source and ADM gauge identities.
    barinv = gi/omega.v**2
    hhat = [sum(barinv[b, c]*reference_connection[a][b][c] for b, c in itertools.product(range(4), repeat=2)) for a in range(4)]
    fbar = [hhat[a]-2*sum(barinv[a, i+1]*omega.derivative(i+1)/omega.v for i in range(3)) for a in range(4)]
    refbar = [[omega**2*v for v in row] for row in refg]
    refbarinv, refbargamma, _, _, _, _ = geometry(refbar, range(4))
    s = sum(barinv[b, c]*refbar[b][c].v for b, c in itertools.product(range(4), repeat=2))
    for a in range(4):
        alternative = (sum(barinv[b, c]*refbargamma[a][b][c] for b, c in itertools.product(range(4), repeat=2))
                       -sum((4*barinv[a, i+1]-s*refbarinv[a, i+1])*omega.derivative(i+1)/omega.v for i in range(3)))
        check('conformal_source_%d' % a, fbar[a]-alternative, [fbar[a], alternative])
    wn = -sum(beta[i].v*omega.derivative(i+1) for i in range(3))/alpha.v
    q = (u['P'].v-3*wn)/omega.v
    lhs = alpha.derivative(0)-sum(beta[i].v*alpha.derivative(i+1) for i in range(3))
    rhs = -alpha.v**2*q-alpha.v**3*fbar[0]
    check('wave_map_alpha_identity', lhs-rhs, [lhs, rhs, alpha.v**2*q, alpha.v**3*fbar[0]])
    for i in range(3):
        lhs = beta[i].derivative(0)-sum(beta[j].v*beta[i].derivative(j+1) for j in range(3))
        rhs = alpha.v**2*(chi.v*u['Lambda'][i].v
                          +sum(ti[i][j].v*(chi.derivative(j+1)/2-chi.v*alpha.derivative(j+1)/alpha.v) for j in range(3))
                          -fbar[i+1]-beta[i].v*fbar[0])
        check('wave_map_beta_identity_%d' % i, lhs-rhs, [lhs, rhs])
    fields = {name: u[name] for name in ('omega', 'alpha', 'beta', 'chi', 'metric', 'A', 'P', 'Lambda', 'Theta')}
    if epsilon == 0:
        native_reference_checks(point_definition, point[1:], fields, check)
    result = dict(point=point, epsilon=epsilon, X=x, Yhat=y,
                  inverse_first=x1, inverse_second=x2, inverse_third=x3,
                  reference_first=y1, reference_second=y2, reference_third=y3,
                  physical_metric=g, fields=fields,
                  exact_time_derivatives={'alpha': alpha.derivative(0), 'beta': [b.derivative(0) for b in beta],
                                          'chi': chi.derivative(0), 'P': u['P'].derivative(0),
                                          'metric': [[v.derivative(0) for v in row] for row in u['metric']],
                                          'A': [[v.derivative(0) for v in row] for row in u['A']],
                                          'Lambda': [v.derivative(0) for v in u['Lambda']], 'Theta': 0},
                  scaled_reference_connection=[[[omega.v*reference_connection[a][i+1][j+1] for j in range(3)] for i in range(3)] for a in range(4)],
                  source_Fbar=fbar, inverse_determinant=inverse['determinant'], scalar_iterations=inverse['iterations'],
                  physical_spatial_principal_minors=minors, physical_lapse=lapse.v,
                  riemann_max_absolute=curvature_abs,
                  checks=checks, passed=all(row['passed'] for row in checks))
    return serial(result, digits)


def flattened(value):
    if isinstance(value, dict):
        for key in sorted(value):
            if key not in ('checks', 'passed', 'scalar_iterations', 'point_name'):
                for name, number in flattened(value[key]):
                    yield key+'/'+name, number
    elif isinstance(value, list):
        for i, part in enumerate(value):
            for name, number in flattened(part):
                yield str(i)+'/'+name, number
    elif isinstance(value, str):
        yield '', mp.mpf(value)


def native_reference_checks(definition, xyz, fields, check):
    frozen = ROOT/'build-layer-research/continuum/immutable-independent-higher-reference-jets-20261009'
    native = [[float(v) for v in line.split()] for line in
              (frozen/'owner-inputs/reference-gate-attempt001/reference-release.stdout').read_text().splitlines()]
    recipe = json.loads((frozen/'owner-inputs/reference-local-recipe.json').read_text())
    row = next(row for row in native if row[0].hex() == float(definition['prescribed_radius']).hex())
    jets = {}
    col = 3
    for entry in recipe['radial_fields_in_output_order']:
        count = len(entry['ordinary_derivatives'])
        jets[entry['name']] = [exact(x) for x in row[col:col+count]]
        col += count
    origin = all(x == 0 for x in xyz)
    def scalar(name, m):
        return (jets[name][0] if sum(m) == 0 else mp.mpf(0)) if origin else scalar_cart(jets[name], xyz, m)
    def vector(name, m, i):
        return mp.mpf(0) if origin else vector_cart(jets[name], xyz, m, i)
    def tensor(name, tangential, m, i, j):
        if origin:
            return jets[tangential][0]*int(i == j) if sum(m) == 0 else mp.mpf(0)
        return tensor_cart(jets[name], jets[tangential], xyz, m, i, j)
    for name, value in fields.items():
        if name == 'Theta':
            continue
        if isinstance(value, Jet):
            items = [((), value)]
        elif name in ('beta', 'Lambda'):
            items = [((i,), v) for i, v in enumerate(value)]
        else:
            items = [((i, j), value[i][j]) for i in range(3) for j in range(i, 3)]
        for components, jet in items:
            for m4 in jet.c:
                if m4[0]:
                    continue
                m = m4[1:]
                if name in ('beta', 'Lambda'):
                    expected = vector('beta' if name == 'beta' else 'lambda', m, components[0])
                elif name in ('metric', 'A'):
                    expected = tensor('g_radial' if name == 'metric' else 'A_radial',
                                      'chi' if name == 'metric' else 'A_tangent', m, *components)
                else:
                    expected = scalar('omega' if name == 'omega' else name, m)
                actual = jet.derivative(*[i for i, count in enumerate(m4) for _ in range(count)])
                check('native_ref_%s_%s_%s' % (name, components, m), actual-expected,
                      [actual, expected], mp.mpf('2e-10'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = json.loads((HERE/'plan.json').read_text())
    admission = json.loads((HERE/'execution-admission.json').read_text())
    for row in admission['sources']:
        assert sha(HERE/row['path']) == row['sha256']
    for row in admission['external_inputs']:
        assert sha(ROOT/row['path']) == row['sha256']
    assert sha(HERE/'plan.json') == admission['plan_sha256']
    for row in plan['inputs']:
        assert sha(ROOT/row['path']) == row['sha256']
    all_outputs = []
    for digits in plan['precision_digits']:
        rows = []
        with mp.workdps(digits):
            height = Height(plan['height_panels'])
            for point in plan['points']:
                for eps in plan['gaussian']['epsilon']:
                    (out/'active.json').write_text(json.dumps(dict(digits=digits, point=point['name'], epsilon=eps))+'\n')
                    row = case(point, mp.mpf(eps), height, digits)
                    row['point_name'] = point['name']
                    rows.append(row)
            payload = dict(digits=digits, height_constant=mp.nstr(height.constant, digits),
                           matching_height=mp.nstr(height.match, digits), cases=rows)
            (out/('oracle-%d.json' % digits)).write_text(json.dumps(payload, indent=2)+'\n')
            all_outputs.append(payload)
    agreement = []
    with mp.workdps(130):
        for left, right in zip(all_outputs[0]['cases'], all_outputs[1]['cases']):
            lf, rf = dict(flattened(left)), dict(flattened(right))
            assert lf.keys() == rf.keys()
            for name in lf:
                err = abs(lf[name]-rf[name])/max(1, abs(lf[name]), abs(rf[name]))
                agreement.append(dict(point=left['point_name'], epsilon=left['epsilon'], field=name,
                                      scaled=float(err), passed=bool(err <= mp.mpf('1e-55'))))
    (out/'precision-agreement.json').write_text(json.dumps(agreement, indent=2)+'\n')
    all_checks = [c for p in all_outputs for row in p['cases'] for c in row['checks']]
    analytic_checks = [row for row in all_checks if not row['name'].startswith('native_ref_')]
    reference_checks = [row for row in all_checks if row['name'].startswith('native_ref_')]
    receipt = dict(kind='Independent finite sampled nonlinear Minkowski wave-map oracle',
                   source_sha256=sha(__file__), plan_sha256=sha(HERE/'plan.json'), admission_sha256=sha(HERE/'execution-admission.json'),
                   python=platform.python_version(), mpmath=mp.__version__, seconds=time.monotonic()-started,
                   cases=sum(len(p['cases']) for p in all_outputs), checks=len(all_checks),
                   maximum_scaled_residual=max(float(row['scaled']) for row in all_checks),
                   analytic_scaled_max=max(float(row['scaled']) for row in analytic_checks),
                   native_reference_checks=len(reference_checks),
                   native_reference_scaled_max=max(float(row['scaled']) for row in reference_checks),
                   precision_comparisons=len(agreement), precision_scaled_max=max(row['scaled'] for row in agreement),
                   minimum_inverse_determinant=min(float(row['inverse_determinant']) for p in all_outputs for row in p['cases']),
                   minimum_physical_lapse=min(float(row['physical_lapse']) for p in all_outputs for row in p['cases']),
                   failed_checks=[row for row in all_checks if not row['passed']],
                   passed=all(row['passed'] for row in all_checks) and all(row['passed'] for row in agreement),
                   scope=plan['scope'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
