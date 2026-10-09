"""Necessary nonlinear scri values for the scaled spatial-norm gauge.

Assume the harmonic outer collar, positive spatial geometry/lapse, finite
conformal trace Q, vacuum Theta_phys=0 at scri, null N_raw=0, and finite gauge
RHS. This proof does not establish preservation under evolution, constrain
finite A/Lambda/angular metric limits, or resolve off-constraint finite Q.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import sympy as s


z, d, x, y = s.symbols('z d x y', real=True)
scale, curvature, rho = s.symbols('S a rho', positive=True)
alpha_ref = scale / curvature
alpha, beta = alpha_ref*x, alpha_ref*y
omega_r = -1 / curvature
omega_n = -beta*omega_r / alpha
physical_p = 3*omega_n  # Finite Q at Omega=0.
physical_p_ref = -3 / curvature
xi = 1 / curvature
eta = rho*scale / curvature**2
feedback = alpha_ref*(1 - 1/rho)

# Direct substitution in the production physical-P lapse pole and the
# exploratory spatial-norm shift pole, rather than a fitted linearization.
alpha_pole = (-alpha**2*(physical_p - physical_p_ref)
              - xi*(alpha + alpha_ref)*(alpha - alpha_ref)
              - (alpha*(beta + alpha_ref)
                 - alpha_ref*(alpha - alpha_ref))*omega_r)
beta_pole = -eta*(beta + alpha_ref + feedback*(z*z - 1))
lapse = 2*x*y + 4*x*x - 2
shift = y + 1 + d*(z*z - 1)
assert s.simplify(-curvature*alpha_pole/alpha_ref**2 - lapse) == 0
assert s.simplify(-beta_pole/(eta*alpha_ref)
                  - shift.subs(d, 1 - 1/rho)) == 0

# The positive spatial norm is G=z^2/a^2. Nullness and the shift pole imply
# y=-x*z<0, so the lapse pole gives x^2(2-z)=1 and therefore 0<z<2.
assert s.simplify(shift.subs(y, -(d*z*z + 1 - d))) == 0
assert s.simplify(lapse.subs(y, -x*z)
                  - 2*(2*x*x - x*x*z - 1)) == 0
f = z / (s.sqrt(2 - z)*(d*z*z + 1 - d))
numerator = d*z*z*(3*z - 4) + (1 - d)*(4 - z)
assert s.simplify(s.diff(s.log(f), z)
                  - numerator/(2*z*(2 - z)*(d*z*z + 1 - d))) == 0

# For 1<=rho<=5/2, 0<=d<=3/5. On 0<z<4/3, the numerator decreases with d:
# its d-derivative is z^2(3z-4)-(4-z)<0. The minimum occurs at d=3/5.
assert s.simplify(s.diff(numerator, d)
                  - (z*z*(3*z - 4) - (4 - z))) == 0
worst = s.expand(5*numerator.subs(d, s.Rational(3, 5)))
assert worst == 9*z**3 - 12*z**2 - 2*z + 8
bernstein = [s.Rational(8), s.Rational(22, 3), s.Rational(8, 3), s.Rational(3)]
assert s.expand(sum(bernstein[j]*s.binomial(3, j)*z**j*(1 - z)**(3 - j)
                    for j in range(4))) == worst
t = s.symbols('t', real=True)
assert s.expand(worst.subs(z, 1 + t/3)) == 3 + t/3 + 5*t*t/3 + t**3/3

# Positive Bernstein coefficients on [0,1], and positive power coefficients
# after t=3(z-1) on [1,4/3], prove the minimum positive. On [4/3,2), both
# terms in the original numerator are nonnegative and the second is positive.
# Thus f is strictly increasing, f(0+)=0, f(2-)=infinity, and f(1)=1.
assert s.simplify(f.subs(z, 1)) == 1
assert shift.subs({x: 1, y: -1, z: 1}) == 0
assert lapse.subs({x: 1, y: -1, z: 1}) == 0

path = Path(__file__).resolve()
root = path.parents[2]
dependencies = [path, root/'src/z4c/hyperboloidal/layer_gauge.hpp',
                root/'build-layer-research/continuum/preferred/native-overlay/'
                'spatial-norm-family/spatial_norm_control.hpp']
out = {
    'scope': __doc__,
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                    text=True).strip(),
    'sympy': s.__version__,
    'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in dependencies},
    'assumptions': ['W=1, xi*a=1', '1<=rho=eta*a^2/S<=5/2', 'alpha>0, G>0',
                    'N_raw=0, finite Q, Theta_phys=0 at scri',
                    'both gauge pole numerators vanish'],
    'result': ('Only alpha_scri=alpha_ref, beta_rad_scri=beta_ref, G_scri=Gref; '
               'tangent beta components are fixed separately by their pole.'),
    'derivative_numerator': str(numerator),
    'minimum_d_polynomial': str(worst),
    'bernstein_coefficients': [str(v) for v in bernstein],
    'evolution_preservation': ('Not proven; initial jet gates and full tensor evolution '
                               'are independent requirements.')}
path.with_name('result.json').write_text(json.dumps(out, indent=2) + '\n')
print('PASS exact necessary nonlinear scri value branch for 1<=rho<=5/2; '
      'evolution preservation is unproved')
