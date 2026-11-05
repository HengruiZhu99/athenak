"""Frozen scalar advection/wave screens; not a coupled Z4c stability proof."""
from pathlib import Path
import json
import numpy as np

D = Path(__file__).resolve().parent
audit = json.loads((D/'cfl-audit/point-audit.json').read_text())
theta = np.linspace(0., np.pi, 32769)
# The actual fourth-order biased beta*d_x stencil in Sample, for beta>=0.
upwind = (-3*np.exp(-1j*theta)-10+18*np.exp(1j*theta)
          -6*np.exp(2j*theta)+np.exp(3j*theta))/12
# Native fourth-order diagonal second derivative (unit mesh spacing).
minus_d2 = (30-32*np.cos(theta)+2*np.cos(2*theta))/12
assert np.min(minus_d2) > -1e-14
wave = np.sqrt(np.maximum(0, minus_d2))
def rk4(z):
    return 1+z+z*z/2+z*z*z/6+z*z*z*z/24

peak = audit['maxima']['l1']
axis = int(np.argmax(peak['directional_cfl']))
a, c = peak['advection_cfl'][axis], peak['wave_cfl'][axis]
z0 = np.r_[a*upwind+1j*c*wave, a*upwind-1j*c*wave]
rows = []
for factor in [1, 2, 4, 5, 6, 8, 10]:
    z = factor*z0
    rows.append(dict(dt_multiplier=factor,
                     max_RK4_amplification=float(np.max(abs(rk4(z)))),
                     radial_characteristic_cfl=factor*(a+c),
                     full_L1_cfl=factor*peak['cfl']))
lo, hi = 0., 30.
for _ in range(60):
    mid = (lo+hi)/2
    if np.max(abs(rk4(mid*z0))) <= 1+1e-12: lo = mid
    else: hi = mid
result = dict(theta_samples=len(theta), checkpoint_sha256=audit['checkpoint_sha256'],
              radial_advection_cfl=a, radial_wave_cfl=c,
              dominant_axis=axis, RK4_scalar_limit_multiplier=lo, screens=rows,
              L1_to_L2_timestep_ratio=peak['cfl']/audit['maxima']['l2']['cfl'],
              qualification='Scalar frozen radial advection plus second-order-in-space wave screen only. Excludes coupled Z4c eigenvectors, non-normal growth, variable coefficients, KO, sources, halos, SMR and scri. A passing screen motivates bounded evolution tests; it is not a certified timestep or a stability/accuracy/performance result.')
(D/'cfl-audit/scalar-symbol-screen.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
