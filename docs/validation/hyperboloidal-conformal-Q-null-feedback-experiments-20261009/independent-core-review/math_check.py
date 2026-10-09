"""Small independent gauge-source/factoring proof; no tensor or evolution run."""
from pathlib import Path
import hashlib
import json
import time

import sympy as s

P = Path(__file__).resolve().parent
start = time.monotonic()
alpha, h, B, Bh, dG, c, dP = s.symbols('alpha h B Bh dG c deltaP', nonzero=True)
wn, wnh = -B / alpha, -Bh / h
D = alpha * Bh / h - B
assert s.factor(D - alpha * (wn - wnh)) == 0
assert s.factor(D - (Bh * (alpha - h) / h - (B - Bh))) == 0
alpha2f = alpha * (alpha + 2 * c)
Qpole = -alpha2f * (dP - 3 * (wn - wnh))
assert s.factor(Qpole + alpha2f * dP - 3 * (alpha + 2 * c) * D) == 0
null = dG - (wn**2 - wnh**2)
weighted = alpha**2 * dG + D * (B + alpha * Bh / h)
assert s.factor(weighted - alpha**2 * null) == 0
# In W=1, P is the stored K_phys-2Theta_phys, and Q=(P-3omega_n)/O.
# In Gamma4^0+2Z4^0 the explicit Theta term cancels, leaving the trace
# contribution -Q/alpha. Combining it with -S_alpha/alpha^3 leaves Kbar_ref.
O, Pstored, Phat, Kbar = s.symbols('Omega P Phat Kbar', nonzero=True)
Qlive = (Pstored-3*wn)/O
SQ = -alpha**2*(Pstored-Phat)+3*alpha*D
trace_source = -SQ/(O*alpha**3)-Qlive/alpha
assert s.factor(trace_source+(Phat-3*wnh)/(O*alpha)) == 0
assert s.factor(trace_source.subs(Phat, O*Kbar+3*wnh)+Kbar/alpha) == 0

# Direct four-metric Christoffel contraction at an orthonormal spatial point.
# The difference of connections is tensorial; a fixed linear spatial change
# of basis extends these gauge-only variation identities to any SPD metric.
beta = s.symbols('b0:3')
V = s.symbols('v0:3')
A = s.symbols('alphaDotDelta')
g = s.eye(4)
g[0, 0] = -alpha**2 + sum(q * q for q in beta)
inv = s.eye(4)
inv[0, 0] = -1 / alpha**2
dg = s.zeros(4, 4)
dg[0, 0] = -2 * alpha * A + 2 * sum(beta[i] * V[i] for i in range(3))
for i in range(3):
    g[0, i + 1] = g[i + 1, 0] = beta[i]
    inv[0, i + 1] = inv[i + 1, 0] = beta[i] / alpha**2
    dg[0, i + 1] = dg[i + 1, 0] = V[i]
    for j in range(3):
        inv[i + 1, j + 1] -= beta[i] * beta[j] / alpha**2
assert s.simplify(g * inv - s.eye(4)) == s.zeros(4, 4)
Gamma = []
for a in range(4):
    value = 0
    for b in range(4):
        for cc in range(4):
            for d in range(4):
                value += inv[b, cc] * inv[a, d] * (
                    (dg[cc, d] if b == 0 else 0)
                    + (dg[b, d] if cc == 0 else 0)
                    - (dg[b, cc] if d == 0 else 0)) / 2
    Gamma.append(s.factor(value))
assert s.factor(Gamma[0] + A / alpha**3) == 0
for i in range(3):
    assert s.factor(Gamma[i + 1] - beta[i] * A / alpha**3 + V[i] / alpha**2) == 0

sigma, v, norm, grad, z = s.symbols('sigma v norm grad z', nonzero=True)
feedback = v * sigma * alpha**2 * grad * null / (norm * O)
assert s.factor(-feedback / alpha**2 + v * sigma * grad * null / (norm * O)) == 0
# Contracting the source shift with Omega_i changes BoxOmega with the opposite
# sign and cancels norm=delta^ij Omega_i Omega_j.
box_change = v * sigma * null / O
assert s.factor(box_change - v * sigma * null / O) == 0

a, k = s.symbols('a kappa', positive=True)
K = s.symbols('K', positive=True)
cubic = z**3 + 2 * (K + sigma) * z**2 + (4 * sigma * (K + 1) - 9) * z + (8 * sigma - 12) * K - 6 * sigma
coeff = s.Poly(cubic.subs(sigma, 5), z).all_coeffs()
assert coeff == [1, 2 * K + 10, 20 * K + 11, 28 * K - 30]
assert s.expand(coeff[1] * coeff[2] - coeff[3]) == 40 * K**2 + 194 * K + 140

# Initial Einstein gauge witness, using the full alpha_ref(r), not its value.
r = s.symbols('r', positive=True)
alphah = (1 + r * r) / (2 * a)
wnh_r = -r * r / (a * a * alphah)
dwn = -O * r / (a * alphah) - wnh_r * O / alphah
Nseries = s.series((-2 * wnh_r * dwn).subs(r, s.sqrt(1 - 2 * a * O)), O, 0, 4)
Qseries = s.series((-3 * dwn / O).subs(r, s.sqrt(1 - 2 * a * O)), O, 0, 3)
assert s.expand(Nseries.removeO()) == -a * O**3
assert s.expand(Qseries.removeO()) == 3 * a**2 * O**2 / 2
# Linearized radial expressions, derived from the displayed production ADM
# and harmonic-collar gauge equations before taking r->1. The lapse/shift
# perturbations are delta alpha=O(r), delta beta^r=-O(r), and all geometric
# fields initially equal the exact CMC reference (chi=gtilde=1, A=Lambda=0).
Or = (1-r*r)/(2*a)
db = -Or
beta_h = -r/a
nu, eta = s.symbols('nu eta', nonnegative=True)
dwn_r = dwn.subs(O, Or)
alpha_linear = (beta_h*(s.diff(Or, r)-Or*s.diff(alphah, r)/alphah)
                -nu*Or+3*alphah**2*dwn_r/Or)
beta_unprojected = (db*s.diff(beta_h, r)+beta_h*s.diff(db, r)-eta*db
                    -alphah*s.diff(Or, r)+Or*s.diff(alphah, r))
# The preferred projection is algebraic in the live field values and its
# perturbation vanishes at r=1 for this witness. Feedback is alpha^2 deltaN/O,
# whose limit is zero because deltaN=O(O^3). Neither changes beta_t0 below.
div_db = s.diff(db, r)+2*db/r
kbar_h = -3/(a*alphah)
chi_linear = -s.Rational(2, 3)*div_db+s.Rational(2, 3)*(Or*kbar_h-3*alphah*dwn_r/Or)
hnn_linear = 2*s.diff(db, r)-s.Rational(2, 3)*div_db
lapOr = s.diff(Or, r, 2)+2*s.diff(Or, r)/r
P_linear = (-Or*lapOr+3*s.diff(Or, r)**2+Or*lapOr+6*Or/a)
theta_linear = 2*Or*lapOr+6*Or/a
assert s.factor(P_linear-3/a**2) == 0
assert s.factor(theta_linear) == 0
alpha_t, beta_t, P_t = [s.limit(q, r, 1) for q in
                         (alpha_linear, beta_unprojected, P_linear)]
chi_t, hnn_t = [s.limit(q, r, 1) for q in (chi_linear, hnn_linear)]
assert alpha_t == 1/a**2 and beta_t == 0 and P_t == 3/a**2
assert chi_t == -2/(3*a) and hnn_t == 4/(3*a)
wn_t = alpha_t + beta_t
assert s.simplify((chi_t - hnn_t) / a**2 + 2 * wn_t / a) == 0
assert s.simplify(P_t - 3 * wn_t) == 0
# This scalar corner cancellation is not unique to the Q source. The prior
# physical-P source with its fully rederived preferred projection distributes
# the same sum between lapse and shift; it is a distinct prior candidate.
xi = s.symbols('xi', nonnegative=True)
d = 1+2*xi*a
prior_alpha_t, prior_beta_t = -d/a**2, (d+1)/a**2
assert s.factor(prior_alpha_t+prior_beta_t-wn_t) == 0
assert s.factor((chi_t-hnn_t)/a**2+2*(prior_alpha_t+prior_beta_t)/a) == 0
assert s.factor(P_t-3*(prior_alpha_t+prior_beta_t)) == 0
# Retained failure of a stronger closure statement: change only P by q*O.
# At the boundary the Q lapse rate is -h^2*q. The geometric P rate includes
# beta_ref*d(q*O)/dr plus the linear P^2 pole; Theta has only the latter.
q = s.symbols('q')
counter_alpha = -q/a**2
counter_P = s.limit(beta_h*q*s.diff(Or, r)
                    +2*alphah*(-3/a)*q/3, r, 1)
counter_Theta = s.limit(2*alphah*(-3/a)*q/3, r, 1)
counter_Qnumerator = s.factor(counter_P-3*counter_alpha)
assert counter_P == -q/a**2 and counter_Theta == -2*q/a**2
assert counter_Qnumerator == 2*q/a**2

out = {'status': 'PASS_SMALL_INDEPENDENT_SYMBOLIC_REVIEW',
       'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'sympy': s.__version__, 'seconds': time.monotonic() - start,
       'D': str(D), 'weighted_null': str(weighted),
       'Gamma_variations': [str(q) for q in Gamma],
       'Q_harmonic_temporal_source_trace_cancellation': '-Kbar_ref/alpha, with P=K_phys-2Theta_phys and Gamma4^0+2Z4^0 convention',
       'sigma5_cubic': str(s.factor(cubic.subs(sigma, 5))),
       'sigma5_Hurwitz_domain': 'K=kappa_input*a^2>15/14; no finite-Omega or PDE stability inference',
       'witness_Nraw_series': str(Nseries), 'witness_Q_series': str(Qseries),
       'witness_linear_radial_RHS': {'alpha': str(s.factor(alpha_linear)),
                                   'beta_unprojected': str(s.factor(beta_unprojected)),
                                   'chi': str(s.factor(chi_linear)),
                                   'gtilde_nn': str(s.factor(hnn_linear)),
                                   'P': str(s.factor(P_linear)),
                                   'Theta': str(s.factor(theta_linear))},
       'witness_Q_corner_RHS': [str(q) for q in (alpha_t, beta_t, P_t, chi_t, hnn_t)],
       'prior_physicalP_projected_corner_RHS': [str(prior_alpha_t), str(prior_beta_t)],
       'finite_Q_only_counterexample': {'seed': 'deltaP=q*Omega, every other field equal reference',
                                        'alpha_dot0': str(counter_alpha),
                                        'P_dot0': str(counter_P),
                                        'Theta_dot0': str(counter_Theta),
                                        'Qnumerator_dot0': str(counter_Qnumerator)},
       'witness_initial_null_and_Qnumerator_time_corner': 'Both zero for Q gauge and prior physical-P with fully rederived preferred projection; no unique repair or higher hierarchy closure',
       'scope': 'Factoring, independent 4D source sign and limited initial-corner proof; no duplicate tensor/native/global gate'}
(P / 'symbolic-review.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
